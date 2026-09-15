"""Complete-trajectory splits and causal, masked diffusion windows."""
import numpy as np
from .io import read, digest

# Public task goals/calibrated drawer reference; no native scene implementation
# is imported into a candidate worker's memory.
DRAWER_ORIGIN = [0.19,0.0,0.035]
OUTSIDE_GOAL = [-0.18,-0.30,0.02]
INSIDE_GOAL_LOCAL = [-0.065,0.075,0.028]

FIELDS = [('qpos',9),('qvel',9),('tcp_pose',7),('red_pose',7),('blue_pose',7),
          ('drawer_position',1),('drawer_velocity',1)]
SLICES = {}; offset = 0
for name, width in FIELDS:
    SLICES[name] = [offset, offset+width]; offset += width
SLICES['red_goal'] = [41,44]; SLICES['blue_goal'] = [44,47]


def vector(state):
    base = [x for name,_ in FIELDS for x in state[name]]
    blue_goal = np.asarray(DRAWER_ORIGIN) + [-state['drawer_position'][0],0,0] + INSIDE_GOAL_LOCAL
    return np.asarray(base + OUTSIDE_GOAL + blue_goal.tolist(),np.float32)


def episodes(cfg, ids=None, segments=None):
    result = []
    for identifier in ids or cfg['data']['train_ids']:
        path = cfg['data']['directory']/(identifier+'.json')
        value = read(path)
        assert value['provenance'] == 'original_demonstration'
        obs = np.stack([vector(o['state']) for o in value['observations']])
        action = np.asarray(value['actions'],np.float32)
        assert obs.shape == (len(action)+1,47) and action.shape[1:] == (8,)
        assert np.isfinite(obs).all() and np.isfinite(action).all()
        start,stop = (0,len(action)) if segments is None else segments[identifier]
        assert 0 <= start < stop <= len(action)
        result.append(dict(id=identifier,obs=obs[start:stop+1],action=action[start:stop],
                           start=start,stop=stop,source_sha256=digest(path)))
    return result


def normalizer(data, cfg):
    obs = np.concatenate([e['obs'][:-1] for e in data]).astype(np.float64)
    action = np.concatenate([e['action'] for e in data]).astype(np.float64)
    low,high = action.min(0),action.max(0)
    mean=obs.mean(0);std=np.maximum(obs.std(0),cfg['observation_std_floor'])
    mode=cfg.get('quaternion_normalization','training_standard_deviation')
    if mode=='unit_component_bounds':
        # Unit quaternion components have a known [-1,1] bound. A near-static
        # orientation's empirical variance is not its physical control scale.
        for field in ('tcp_pose','red_pose','blue_pose'):
            start,stop=SLICES[field];mean[start+3:stop]=0.;std[start+3:stop]=1.
    elif mode!='training_standard_deviation':raise ValueError('Unknown quaternion normalization')
    return dict(mean=mean.tolist(),std=std.tolist(),quaternion_normalization=mode,
                raw_std=obs.std(0).tolist(),action_min=low.tolist(),
                action_scale=np.maximum(high-low,cfg['action_scale_floor']).tolist(),
                fit_ids=[e['id'] for e in data],fit_samples=len(obs),observation_clipping=False)


def windows(data, cfg):
    histories, actions, masks, indices = [],[],[],[]
    h,n = cfg['horizon'],cfg['observation_steps']
    for eidx,e in enumerate(data):
        for t in range(len(e['action'])):
            oi = np.maximum(np.arange(t-n+1,t+1),0)
            ai = np.arange(t-n+1,t-n+1+h)
            valid = (ai >= 0) & (ai < len(e['action']))
            histories.append(e['obs'][oi]); actions.append(e['action'][np.clip(ai,0,len(e['action'])-1)])
            masks.append(valid[:,None]); indices.append((eidx,t))
    return dict(raw_obs=np.asarray(histories,np.float32),native_action=np.asarray(actions,np.float32),
                mask=np.asarray(masks,np.float32),indices=np.asarray(indices,np.int64))


def audit(cfg):
    train = episodes(cfg); val = episodes(cfg,cfg['data']['validation_ids'])
    norm = normalizer(train,cfg['training'])
    stats = {}
    for field in ('red_pose','blue_pose'):
        values = np.stack([e['obs'][0,slice(*SLICES[field])][:3] for e in train])
        stats[field] = dict(min=values.min(0).tolist(),max=values.max(0).tolist())
    examples=[]
    for t in (0,339,699):
        e=train[0]; x=e['obs'][t]; action=e['action'][t]
        an=2*(action-np.asarray(norm['action_min']))/norm['action_scale']-1
        inverse=(an+1)*np.asarray(norm['action_scale'])/2+norm['action_min']
        assert np.allclose(action,inverse,atol=1e-7)
        examples.append(dict(id=e['id'],time_index=t,observation_before_action=x.tolist(),
            raw_action=action.tolist(),normalized_action=an.tolist(),inverse_action=inverse.tolist()))
    return dict(train_trajectories=len(train),validation_trajectories=len(val),
        train_actions=sum(len(e['action']) for e in train),validation_actions=sum(len(e['action']) for e in val),
        observation_dimension=47,fields=SLICES,training_initial_support=stats,normalizer=norm,
        examples=examples,split_unit='complete original trajectory',source_hashes={e['id']:e['source_sha256'] for e in train+val},
        legacy_obs_std_below_1e_3=[i for i,s in enumerate(norm['raw_std']) if s<.001],
        action_semantics='7 absolute arm joint targets in radians, gripper [-1,1] -> each finger [-.01,.04] metre')
