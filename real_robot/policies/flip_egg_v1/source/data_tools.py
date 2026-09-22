import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from real_robot.training_pipeline import public


def valid_pose(T):
    T = np.asarray(T)
    return T.shape == (4, 4) and np.isfinite(T).all() and np.max(np.abs(T[3] - [0, 0, 0, 1])) < 1e-5 and np.linalg.norm(T[:3,:3].T @ T[:3,:3] - np.eye(3)) < 0.005 and np.linalg.det(T[:3,:3]) > 0.995


def local_rate(A, B, dt):
    return np.concatenate((A[:3,:3].T @ (B[:3,3] - A[:3,3]), Rotation.from_matrix(A[:3,:3].T @ B[:3,:3]).as_rotvec())) / dt


def image_crop(rgb, camera, cfg):
    rgb = np.asarray(rgb)
    if rgb.shape != (720, 1280, 3) or rgb.dtype != np.uint8:
        raise ValueError('RGB must be original 720x1280x3 uint8')
    x0, y0, x1, y1 = cfg[camera + '_crop']
    h, w = cfg['image_height'], cfg['image_width']
    a = rgb[y0:y1, x0:x1]
    s = min(w / a.shape[1], h / a.shape[0])
    rw, rh = int(round(a.shape[1]*s)), int(round(a.shape[0]*s))
    out = np.zeros((h,w,3), np.uint8)
    out[(h-rh)//2:(h-rh)//2+rh,(w-rw)//2:(w-rw)//2+rw] = cv2.resize(a,(rw,rh),interpolation=cv2.INTER_AREA)
    return out.transpose(2,0,1).copy()


def age_valid(ages, cfg):
    lo, hi = cfg['image_age_range_s']
    a = np.asarray(ages, dtype=np.float64)
    return np.isfinite(a) & (a >= lo) & (a <= hi)


def base_features(T, tau, width, ages, vm):
    age = np.nan_to_num(np.asarray(ages), nan=0., posinf=0., neginf=0.)
    return np.concatenate(((T[:3,3] - [.45,0,.15]) / [.25,.35,.3], T[:3,:2].T.reshape(-1), np.asarray(tau)/5., [width/.14], np.clip(age,-.02,.15)/.15, np.asarray(vm,dtype=float))).astype(np.float32)


def history_features(base, poses, times, mask):
    f = np.zeros((5,28),np.float32)
    f[:,:21] = base
    for j in range(1,5):
        dt = times[j] - times[j-1]
        if mask[j] and mask[j-1] and 0.05 <= dt <= 0.2:
            f[j,21] = dt/.1
            f[j,22:] = local_rate(poses[j-1],poses[j],dt) / np.array([.1,.1,.1,.5,.5,.5])
    f *= np.asarray(mask)[:,None]
    return f


def summary(x):
    x = np.asarray(x)
    if x.size == 0:
        return {'count':0}
    return {'count':int(len(x)), 'min':np.min(x,axis=0).tolist(), 'p50':np.median(x,axis=0).tolist(), 'p95':np.quantile(x,.95,axis=0).tolist(), 'max':np.max(x,axis=0).tolist()}


def duplicate_audit(frames, seq, poses, episodes, original_split):
    groups = list(range(len(original_split)))
    probes = []
    for ep in range(len(groups)):
        ids = np.flatnonzero(episodes == ep)
        chosen = ids[np.unique(np.linspace(0,len(ids)-1,min(32,len(ids))).astype(int))] if len(ids) else ids
        probes.append(chosen)
    features = []
    for inds in probes:
        rows = []
        for k in inds:
            im = frames[seq[k,-1]]
            rows.append(np.concatenate([cv2.resize(a.transpose(1,2,0),(12,9),interpolation=cv2.INTER_AREA).mean(2).reshape(-1) for a in im])/255.)
        features.append(np.asarray(rows))
    pairs, matched = [], []
    for a in range(len(groups)):
        for b in range(a+1,len(groups)):
            if not len(probes[a]) or not len(probes[b]):
                continue
            dist = cdist(features[a],features[b],metric='sqeuclidean') / 216.
            m, n = np.unravel_index(np.argmin(dist),dist.shape)
            k, l = int(probes[a][m]), int(probes[b][n])
            rms = float(np.sqrt(dist[m,n]))
            if original_split[a] != original_split[b]:
                pairs.append({'episode_positions':[a,b],'example_positions':[k,l],'thumbnail_rms':rms})
            if rms < .012:
                pixel = float(np.sqrt(np.mean((frames[seq[k,-1]].astype(float)-frames[seq[l,-1]].astype(float))**2)) / 255.)
                dp = np.linalg.norm(poses[k,:3,3]-poses[l,:3,3])
                dr = np.linalg.norm(Rotation.from_matrix(poses[k,:3,:3].T@poses[l,:3,:3]).as_rotvec())
                if pixel < .012 and dp < .002 and dr < .02:
                    old, new = groups[b], groups[a]
                    groups = [new if g == old else g for g in groups]
                    matched.append({'episodes':[a,b], 'pixel_rms':pixel})
    split = np.array(original_split,dtype=np.int64)
    for g in set(groups):
        ids = [j for j in range(len(groups)) if groups[j] == g]
        split[ids] = max(split[ids])
    return split, {'scope':'32 evenly spaced held-tool RGB+pose probes per episode, both views; strict pixel+pose confirmation. Not exhaustive perceptual deduplication. Shared physical objects/layouts intentionally remain.', 'cross_partition_closest':sorted(pairs,key=lambda x:x['thumbnail_rms'])[:20], 'merged':matched, 'episode_groups':groups, 'resulting_episode_split':split.tolist()}


def prepare(spec):
    cfg = spec['preprocessing']
    cut = public.source_json('ORIGINAL_CUT.json')['episodes']
    public.write_report('original_source_plan.json',spec['source_plan'])
    public.write_report('original_recording_metadata.json',spec['recording_metadata'])
    public.write_report('original_cut_transcription.json',public.source_json('ORIGINAL_CUT.json'))
    public.progress({'stage':'preparing','episodes':len(cut),'representation':'measured EE local, not blade'})
    frames, sequence, feats, histmask, ys, eps, segs, anchors, poses, dts = [], [], [], [], [], [], [], [], [], []
    source, reports, per_anchor = [], [], []
    episode_positions = {t:i for i,t in enumerate(spec['trajectory_ids'])}
    original_split = np.zeros(len(spec['trajectory_ids']),np.int64)
    goals_by_ep = {}
    for ei, row in enumerate(cut):
        tid, n, s, e, split = row
        ep = episode_positions[tid]
        original_split[ep] = split
        si = next(j for j,v in enumerate(spec['data_plan']['segments']) if v['trajectory_id'] == tid)
        r = public.records(tid)
        meta = public.metadata(tid)
        T, t = np.asarray(r['T_base_ee']), np.asarray(r['t'])
        assert len(t) == n and len(T) == n
        assert np.isfinite(t).all() and np.all(np.diff(t)>0)
        source.append({'trajectory_id':tid,'start':0,'stop':n,'use':'context','reason':'Immutable original full episode, initial reference, history, acquisition and outcome/intervention audit. No actor access to future context.'})
        source.append({'trajectory_id':tid,'start':s,'stop':e,'use':'supervision','reason':'Original autonomous-target interval; strided source anchors and validity checks below.'})
        source.append({'trajectory_id':tid,'start':s+3,'stop':e+3,'use':'context','reason':'Achieved-motion future target evidence only, never causal input.'})
        goal = image_crop(public.rgb(tid,0,'third'),'third',cfg)
        goals_by_ep[ep] = goal
        candidate = sorted(set(list(range(s,e,3)) + [e-1]))
        needed = sorted(set(j for i in candidate for j in range(i-12,i+1,3)))
        fmap, bs, vmasks = {}, {}, {}
        rejected, accepted, single = {}, 0, 0
        ages = np.stack((r['img_age_third'],r['img_age_wrist']),axis=1)
        for j in needed:
            vm = age_valid(ages[j],cfg)
            images = []
            for v,cam in enumerate(['third','wrist']):
                im = image_crop(public.rgb(tid,j,cam),cam,cfg)
                if not vm[v]:
                    im[:] = 0
                images.append(im)
            fmap[j] = len(frames)
            frames.append(np.stack(images))
            vmasks[j] = vm
            bs[j] = base_features(T[j],r['tau_ext'][j],float(r['gripper_width_m'][j]),ages[j],vm)
        episode_targets = []
        for i in candidate:
            dt = float(t[i+3]-t[i])
            reason = ''
            if not valid_pose(T[i]) or not valid_pose(T[i+3]):
                reason = 'invalid_SE3'
            elif not cfg['target_dt_range_s'][0] <= dt <= cfg['target_dt_range_s'][1]:
                reason = 'invalid_target_dt'
            elif not vmasks[i].any():
                reason = 'both_views_outside_skew_age_tolerance'
            elif not .025 <= float(r['gripper_width_m'][i]) <= .045:
                reason = 'outside_observed_retained_width_envelope'
            hs = list(range(i-12,i+1,3))
            hm = np.array([bool(vmasks[j].any()) and valid_pose(T[j]) and np.isfinite(bs[j]).all() for j in hs],dtype=np.float32)
            if not reason:
                y = local_rate(T[i],T[i+3],dt)
                if not np.isfinite(y).all() or np.linalg.norm(y[:3]) > .3 or np.linalg.norm(y[3:]) > 1.2:
                    reason = 'motion_outlier_audit_bound'
            if reason:
                rejected[reason] = rejected.get(reason,0)+1
                per_anchor.append({'trajectory_id':tid,'index':i,'retained':False,'reason':reason,'target_index':i+3,'dt':dt})
                continue
            f = history_features(np.stack([bs[j] for j in hs]),T[hs],t[hs],hm)
            if not np.isfinite(f).all():
                raise ValueError('Nonfinite causal features')
            sequence.append([fmap[j] for j in hs])
            feats.append(f)
            histmask.append(hm)
            ys.append(y)
            eps.append(ep)
            segs.append(si)
            anchors.append(i)
            poses.append(T[i])
            dts.append(dt)
            accepted += 1
            single += int(vmasks[i].sum()==1)
            episode_targets.append(y)
            per_anchor.append({'trajectory_id':tid,'index':i,'retained':True,'target_index':i+3,'dt':dt,'history_indices':hs,'view_valid':vmasks[i].tolist(),'u_ee':y.tolist()})
        montage = []
        chosen = [candidate[0],candidate[len(candidate)//3],candidate[2*len(candidate)//3],candidate[-1]]
        for j in chosen:
            pair = frames[fmap[j]]
            strip = np.concatenate([goal.transpose(1,2,0),pair[0].transpose(1,2,0),pair[1].transpose(1,2,0)],axis=1).copy()
            cv2.putText(strip, str(j), (198,16), cv2.FONT_HERSHEY_SIMPLEX,.45,(255,70,40),1)
            montage.append(strip)
        public.write_image('coverage_'+tid+'.png',np.concatenate(montage,axis=0))
        entry = {'trajectory_id':tid,'original_rows':n,'original_dense_anchors':e-s,'candidates':len(candidate),'accepted':accepted,'single_view_anchors':single,'rejected':rejected,'negative_image_age_rows':np.sum(ages<0,axis=0).tolist(),'outside_tolerance_rows':np.sum(~age_valid(ages,cfg),axis=0).tolist(),'target_rates':summary(np.asarray(episode_targets)), 'tcp':meta['tcp'],'calibration_id':meta['calibration']['generated'],'gripper_interval':summary(r['gripper_width_m'][s:e])}
        reports.append(entry)
        public.progress({'stage':'episode_complete','episode':tid,'accepted':accepted,'candidate':len(candidate)})
    frames = np.stack(frames).astype(np.uint8)
    seq = np.asarray(sequence,np.int64)
    eps = np.asarray(eps,np.int64)
    poses = np.asarray(poses,np.float64)
    split, dup = duplicate_audit(frames,seq,poses,eps,original_split)
    k = len(eps)
    y = np.asarray(ys,np.float32)
    static = (np.linalg.norm(y[:,:3],axis=1) < .003) & (np.linalg.norm(y[:,3:],axis=1) < .03)
    motion_class = np.where(static,0,np.where(np.linalg.norm(y[:,3:],axis=1) > .12,2,1)).astype(np.int64)
    a = {'frames':frames,'goals':np.stack([goals_by_ep[j] for j in range(len(cut))]),'sequence':seq,'features':np.stack(feats).astype(np.float32),'history_mask':np.stack(histmask).astype(np.float32),'target':y,'target_mask':np.ones((k,6),np.float32),'target_dt':np.asarray(dts,np.float32),'T_anchor':poses,'split':split[eps],'motion_class':motion_class,'example_episode':eps,'example_segment':np.asarray(segs,np.int64),'example_source_index':np.asarray(anchors,np.int64),'example_variant':np.zeros(k,np.int64),'observation_start':np.zeros(k,np.int64),'observation_stop':np.asarray(anchors,np.int64)+1,'target_start':np.asarray(anchors,np.int64),'target_stop':np.asarray(anchors,np.int64)+4,'policy_weight':np.ones((k,1),np.float32)}
    errors = []
    for j in range(0,k,max(1,k//128)):
        tid = spec['trajectory_ids'][int(eps[j])]
        i = int(a['example_source_index'][j])
        future = public.records(tid)['T_base_ee'][i+3]
        R, p = poses[j,:3,:3], poses[j,:3,3]
        pg = p + R @ y[j,:3] * dts[j]
        Rg = R @ Rotation.from_rotvec(y[j,3:]*dts[j]).as_matrix()
        errors.append([float(np.linalg.norm(pg-future[:3,3])),float(np.linalg.norm(Rotation.from_matrix(Rg.T@future[:3,:3]).as_rotvec()))])
    counts = {name:{'examples':int(np.sum(a['split']==s)), 'episodes':int(len(np.unique(eps[a['split']==s]))),'static':int(np.sum((a['split']==s)&static)),'rotating':int(np.sum((a['split']==s)&(motion_class==2)))} for s,name in enumerate(['train','validation','test'])}
    metadata = {'num_examples':k,'coverage':{'by_episode':reports,'by_split':counts,'independent_experience':'20 separate recordings of the same tool/pan/egg setup, not K independent flips; no success annotations.', 'original_dense_anchors':sum(v[3]-v[2] for v in cut),'stride':3},'label_provenance':{'target':'[R_i^T(p_i+3-p_i), Log(R_i^T R_i+3)] / (t_i+3-t_i); recorded EE origin and axes. Future row is training-only. No action_json or measured dq used as command.','inputs':'Current/past RGB, EE poses, joint torque estimates, width, recorded age. Initial scene third row0. No future frames, annotated phases or fitted target geometry. Online uses identical crops and normalization.','geometry':'TCP translation [0,0,.24], yaw -90deg is not a blade transform; metric blade/pan labels deliberately not claimed.'},'data_audit':{'by_episode':reports,'near_duplicate_audit':dup,'target_dt':summary(dts),'round_trip_error_m_rad':summary(errors),'recovery':'No PnP annotations/metric blade geometry supplied: replace that dependency with joint visual learning and directly measured EE targets. Preserve one-view anchors with masks. Stride avoids pseudo-independent 30Hz command repeats. Invalid dual-view anchors/outlier labels are logged, not zero-labelled. Negative age retained only within -20ms skew tolerance; no shifting; runtime monotonic ID and age checks. No physical success recovered from final images.','schema_equivalence':'Prepared arrays are used in real-source act/predict parity test; rejection paths also tested.'},'variant_definition':'0 = original same-pan goal, original source i+3 achieved EE rate; no goal variants or successes fabricated.','source_accounting':source,'trajectory_ids':spec['trajectory_ids'],'original_cut':cut,'cache_bytes':int(sum(v.nbytes for v in a.values()))}
    public.write_report('anchor_audit.json',per_anchor)
    public.write_report('preparation_summary.json',metadata)
    public.write_report('target_roundtrip.json',{'errors':summary(errors),'thresholds':[1e-6,1e-5]})
    assert max(v[0] for v in errors) < 1e-6 and max(v[1] for v in errors) < 1e-5
    assert all(counts[n]['episodes'] > 0 for n in counts)
    return {'arrays':a,'metadata':metadata}
