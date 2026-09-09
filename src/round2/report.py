"""Reports observed outcomes, including absent video categories and failures."""
import csv
import json
import time
import traceback
from pathlib import Path
import numpy as np
from relative_dp.utils import ROOT,read_json,atomic_json,sha256
from .learning import RUNS
from .geometry import GeometryEnv
from .collect import SPLITS

FIG=ROOT/'artifacts/round2/figures'
VIDEO=ROOT/'artifacts/round2/videos'


def load_results():
    selection=read_json(ROOT/'results/round2/selection.json')
    results={}
    for run in RUNS:
        rid=run['run_id'];step=selection['runs'][rid]['step']
        p=ROOT/f'results/round2/test/{rid}/step_{step:06d}/complete.json'
        result=read_json(p)
        assert len(result['records'])==120
        for r in result['records']:assert sha256(ROOT/r['trajectory_path'])==r['trajectory_hash']
        results[rid]=result
    return selection,results


def expert_reference():
    """Post-test diagnostic with the unchanged frozen expert on ALL states."""
    from .calibrate import logged_rollout,LEDGER
    from .learning import code_hash
    output=ROOT/'results/round2/expert_reference.json'
    identity=dict(manifest_hash=sha256(ROOT/'data/round2/manifest.json'),code_hash=code_hash(),
                  stage='frozen_test_reference_posthoc')
    if output.exists():
        old=read_json(output);assert old['identity']==identity;return old
    manifest=read_json(ROOT/'data/round2/manifest.json')
    plan={task:[r for split in SPLITS[1:] for r in manifest['tasks'][task][split]] for task in ('drawer','door')}
    atomic_json(ROOT/'artifacts/round2/expert_reference_plan.json',dict(identity=identity,records=plan))
    ledger=read_json(LEDGER);seen={r['record']['episode_id'] for r in ledger}
    for task,records in plan.items():
        assert sum(r['record']['task_name']==task for r in ledger)+sum(r['episode_id'] not in seen for r in records)<=600
    start=time.monotonic();results={}
    for task,records in plan.items():
        e=GeometryEnv(task);rows=[]
        try:
            for rec in records:rows.append(logged_rollout(e,rec,identity['stage']))
        finally:e.close()
        metrics={}
        for split in SPLITS[1:]:
            subset=[r for r in rows if r['record']['split']==split]
            metrics[split]=dict(n=len(subset),valid_successes=sum(r['success'] and not r['invalid_joint'] for r in subset),
                                invalid_joint_episodes=sum(r['invalid_joint'] for r in subset),exceptions=sum(r['exception'] is not None for r in subset))
        results[task]=dict(records=rows,metrics=metrics)
    result=dict(identity=identity,tasks=results,wall_seconds=time.monotonic()-start,
                interpretation='Post hoc execution reference, not a pretraining precheck or a guaranteed physical ceiling. No expert/environment changes or test replacement follow these results.')
    atomic_json(output,result);return result


def figures(results,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.dpi':140,'font.size':10})
    colors={'world':'#4c78a8','frame':'#e45756'}
    names=['IID','Position','Yaw near','Yaw mid','Yaw far','Combined']
    fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
    for ax,task in zip(axes,('drawer','door')):
        for rep,dx in [('world',-.18),('frame',.18)]:
            r=results[f'r2_{task}_{rep}_n20_s0'];values=[r['metrics'][s]['success_rate'] for s in SPLITS[1:]]
            ax.bar(np.arange(6)+dx,values,.35,label=rep,color=colors[rep])
        ax.set(title=task,xticks=np.arange(6),xticklabels=names,ylim=(0,1.07),ylabel='Success / 20')
        ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.tight_layout();fig.savefig(FIG/'split_success.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for ti,task in enumerate(('drawer','door')):
        for si,sign in enumerate((-1,1)):
            ax=axes[ti,si]
            for rep in ('world','frame'):
                r=results[f'r2_{task}_{rep}_n20_s0']
                values=[r['yaw_by_sign'][f'test_yaw_{level}:{sign}']['success_rate'] for level in ('near','mid','far')]
                ax.plot([20,35,50],values,'o-',label=rep,color=colors[rep])
                actual=[x for x in r['records'] if x['split'] in ('test_yaw_near','test_yaw_mid','test_yaw_far') and np.sign(x['yaw_degrees'])==sign]
                ax.scatter([abs(x['yaw_degrees']) for x in actual],[float(x['success']) for x in actual],color=colors[rep],alpha=.25,s=12)
            ax.set(title=f'{task}, yaw sign {sign:+d}',xlabel='Yaw magnitude (degrees), jitter +/-2',ylabel='Success / 10',ylim=(-.03,1.05));ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(FIG/'yaw_sign_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    cats=['no_approach','approach_no_contact','contact_no_progress','progress_not_completed','success']
    labels=[r['run_id'].replace('r2_','').replace('_n20_s0','') for r in RUNS]
    bottom=np.zeros(4)
    for cat in cats:
        values=np.array([sum(r['failure_category']==cat for r in results[run['run_id']]['records']) for run in RUNS])
        axes[0].bar(labels,values,bottom=bottom,label=cat);bottom+=values
    axes[0].set(ylabel='All final episodes (120/model)',title='Observed interaction outcomes');axes[0].tick_params(axis='x',rotation=20)
    axes[0].legend(fontsize=7)
    axes[1].boxplot([[r['max_progress'] for r in results[run['run_id']]['records']] for run in RUNS],tick_labels=labels)
    axes[1].axhline(.75,ls='--',color='gray');axes[1].set(ylabel='Maximum joint progress',title='Progress includes failed episodes');axes[1].tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(FIG/'progress_failures.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ti,task in enumerate(('drawer','door')):
        for rep in ('world','frame'):
            rid=f'r2_{task}_{rep}_n20_s0'
            lines=[json.loads(x) for x in (ROOT/f'runs/round2/{rid}/train.jsonl').read_text().splitlines()]
            axes[ti,0].plot([r['step'] for r in lines],[r['loss'] for r in lines],label=rep,color=colors[rep])
            candidates=selection['runs'][rid]['candidates']
            axes[ti,1].plot([c['step'] for c in candidates],[c['success_rate'] for c in candidates],'o-',label=rep,color=colors[rep])
        axes[ti,0].set(title=f'{task}: training',xlabel='Update',ylabel='Masked epsilon MSE',yscale='log')
        axes[ti,1].set(title=f'{task}: dev20',xlabel='EMA checkpoint update',ylabel='Success rate',ylim=(-.03,1.05))
        for ax in axes[ti]:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'training_dev.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
    for ti,task in enumerate(('drawer','door')):
        for si,(title,splits) in enumerate([('IID',['test_iid']),('Yaw primary',['test_yaw_near','test_yaw_mid','test_yaw_far']),('Combined',['test_combined'])]):
            for rep in ('world','frame'):
                rows=[r for r in results[f'r2_{task}_{rep}_n20_s0']['records'] if r['split'] in splits]
                t=np.arange(501)
                curve=np.array([sum(r['success'] and r['first_success_step']<=x for r in rows)/len(rows) for x in t])
                axes[ti,si].plot(t,curve,label=rep,color=colors[rep])
            axes[ti,si].set(title=f'{task}: {title}',xlabel='Control steps (failures unfinished at 500)',ylabel='Cumulative success',ylim=(0,1.03));axes[ti,si].legend();axes[ti,si].grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'cumulative_success.png');plt.close(fig)


def render_actions(task,rec,trajectories,path,expected=None,captions=None):
    import os
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','1')
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    envs=[GeometryEnv(task,render_mode='rgb_array') for _ in trajectories]
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),14)
    def frame(t):
        panels=[]
        for i,e in enumerate(envs):
            canvas=Image.fromarray(np.flipud(e.native.render()))
            draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,480,45),fill=(25,25,25))
            draw.text((8,5),(captions or ['recorded rollout']*len(envs))[i],font=font,fill='white')
            draw.text((8,25),f'step {min(t,len(trajectories[i]))} | progress {e.progress:.3f} | yaw {e.yaw:+.1f} deg',font=font,fill='white')
            panels.append(np.asarray(canvas))
        return np.concatenate(panels,axis=1)
    try:
        for i,env in enumerate(envs):
            obs,_=env.reset(rec)
            if expected is not None and len(expected[i]):np.testing.assert_allclose(obs,expected[i][0],atol=1e-6,rtol=0)
        with imageio.get_writer(path,fps=20,codec='libx264',quality=7,macro_block_size=16) as writer:
            writer.append_data(frame(0))
            for t in range(max(len(a) for a in trajectories)):
                for i,(e,a) in enumerate(zip(envs,trajectories)):
                    if t<len(a):
                        obs=e.step(a[t].copy())[0]
                        if expected is not None:np.testing.assert_allclose(obs,expected[i][t+1],atol=1e-6,rtol=0)
                if (t+1)%4==0 or t==max(len(a) for a in trajectories)-1:
                    writer.append_data(frame(t+1))
    finally:
        for e in envs:e.close()


def videos(results):
    output=VIDEO/'manifest.json';VIDEO.mkdir(parents=True,exist_ok=True)
    if output.exists():
        old=read_json(output)
        if old.get('complete') and all(not r.get('path') or sha256(ROOT/r['path'])==r['sha256'] for r in old['records']):return old
    manifest=read_json(ROOT/'data/round2/manifest.json')
    records=[];start=time.monotonic()
    def produce(label,task,chosen):
        if not chosen:
            records.append(dict(label=label,absent=True,reason='No episode matches this predeclared category'))
            return
        rec=next(r for r in manifest['tasks'][task][chosen[0]['split']] if r['episode_id']==chosen[0]['episode_id'])
        path=VIDEO/f'{label}.mp4';begun=time.monotonic()
        r=dict(label=label,episode_id=rec['episode_id'],split=rec['split'],initial_state_hash=rec['initial_state_hash'],
               ordering='world left, frame right' if len(chosen)==2 else 'single model')
        try:
            trajectories=[];expected=[];captions=[]
            for c in chosen:
                with np.load(ROOT/c['trajectory_path']) as f:
                    trajectories.append(f['actions'].copy());expected.append(f['obs'].copy())
                rep='frame' if '_frame_' in c['identity']['run_id'] else 'world'
                reason='SUCCESS' if c['success'] else ('FAIL: joint limit excursion' if c['invalid_joint'] else ('FAIL: simulation/inference error' if c['exception'] else 'FAIL: goal not completed'))
                captions.append(f'{rep} | {reason}')
            render_actions(task,rec,trajectories,path,expected,captions)
            r.update(path=str(path.relative_to(ROOT)),sha256=sha256(path))
        except Exception:r['render_exception']=traceback.format_exc()
        r['wall_seconds']=time.monotonic()-begun;records.append(r)
        atomic_json(output,dict(complete=False,records=records))
        print('VIDEO',label,r.get('path',r.get('render_exception')),flush=True)
    for run in RUNS:
        rid=run['run_id'];rows=results[rid]['records']
        for split in ('test_iid','test_yaw_far'):
            for success in (True,False):
                first=next((r for r in rows if r['split']==split and r['success']==success),None)
                produce(f'{rid}_{split}_{"first_success" if success else "first_failure"}',run['task'],[first] if first else [])
    for task in ('drawer','door'):
        w=results[f'r2_{task}_world_n20_s0']['records'];f={r['episode_id']:r for r in results[f'r2_{task}_frame_n20_s0']['records']}
        first=next((r for r in w if not r['success'] and f[r['episode_id']]['success']),None)
        produce(f'{task}_first_world_failure_frame_success',task,[first,f[first['episode_id']]] if first else [])
    result=dict(complete=True,records=records,wall_seconds=time.monotonic()-start)
    atomic_json(output,result);return result


def report():
    selection,results=load_results()
    reference=expert_reference()
    success_audit={}
    for rid,result in results.items():
        counts={scope:dict(n=0,threshold_reached=0,threshold_reached_with_joint_violation=0,valid_success=0) for scope in ('all','yaw_primary')}
        for r in result['records']:
            with np.load(ROOT/r['trajectory_path']) as f:
                p=f['progress'];valid=(p>=.75)&(p<=1.01)
                reached=bool(len(p)>=3 and np.any(np.convolve(valid.astype(int),np.ones(3,dtype=int),mode='valid')==3))
            scopes=['all']+(['yaw_primary'] if r['split'] in ('test_yaw_near','test_yaw_mid','test_yaw_far') else [])
            for scope in scopes:
                c=counts[scope];c['n']+=1;c['threshold_reached']+=int(reached)
                c['threshold_reached_with_joint_violation']+=int(reached and r['invalid_joint'])
                c['valid_success']+=int(r['success'])
        success_audit[rid]=counts
    atomic_json(ROOT/'results/round2/success_audit.json',dict(
        purpose='Descriptive diagnosis of the predeclared strict success rule, not an alternative primary score',runs=success_audit))
    figures(results,selection)
    video=videos(results)
    rows=[]
    for run in RUNS:
        rid=run['run_id'];r=results[rid]
        rows.append(dict(run_id=rid,selected_step=selection['runs'][rid]['step'],
                         yaw_primary=r['yaw_primary']['success_rate'],**{s:r['metrics'][s]['success_rate'] for s in SPLITS[1:]}))
    with (ROOT/'results/round2/summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    detail=[]
    for rid,result in results.items():
        for split,m in dict(result['metrics'],yaw_primary=result['yaw_primary']).items():
            detail.append(dict(run_id=rid,split=split,**{k:v for k,v in m.items() if k!='failure_categories'}))
    with (ROOT/'results/round2/metrics_by_split.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    atomic_json(ROOT/'results/round2/summary.json',dict(rows=rows,results=results,selection=selection))
    all_expert=read_json(ROOT/'artifacts/round2/calibration_rollouts.json')
    ledger=[r for r in all_expert if r['stage']!='frozen_test_reference_posthoc']
    reference_rows=[r for r in all_expert if r['stage']=='frozen_test_reference_posthoc']
    zero_files=list((ROOT/'artifacts/round2').glob('zero_actions*.json'))
    zero_seconds=sum(sum(r['wall_seconds'] for r in read_json(p)) for p in zero_files)
    manifest=read_json(ROOT/'data/round2/manifest.json')
    audit=read_json(ROOT/'artifacts/round2/data_audit.json')
    debug=[read_json(ROOT/f'artifacts/round2/debug/{r["run_id"]}/complete.json') for r in RUNS]
    formal=[read_json(ROOT/f'runs/round2/{r["run_id"]}/complete.json') for r in RUNS]
    dev_seconds=sum(sum(r['wall_seconds'] for r in read_json(ROOT/f'results/round2/dev/{run["run_id"]}/step_{step:06d}/complete.json')['records']) for run in RUNS for step in (5000,10000,20000))
    test_seconds=sum(sum(r['wall_seconds'] for r in value['records']) for value in results.values())
    cost=dict(calibration_expert_rollouts=len(ledger),calibration_by_task={t:sum(r['record']['task_name']==t for r in ledger) for t in ('drawer','door')},
              calibration_expert_seconds=sum(r['wall_seconds'] for r in ledger),zero_action_seconds=zero_seconds,
              frozen_test_expert_reference_rollouts=len(reference_rows),
              frozen_test_expert_reference_seconds=sum(r['wall_seconds'] for r in reference_rows),
              all_expert_checks_by_task={t:sum(r['record']['task_name']==t for r in all_expert) for t in ('drawer','door')},
              data_collection_seconds=manifest['collection_wall_seconds'],data_audit_seconds=audit['wall_seconds'],
              debug_update_seconds=sum(r['update_seconds'] for r in debug),debug_actual_updates=sum(r['step'] for r in debug),
              debug_rollout_seconds=sum(read_json(ROOT/f'artifacts/round2/debug/{r["run_id"]}/rollout_check.json')['wall_seconds'] for r in RUNS),
              formal_update_seconds=sum(r['update_seconds'] for r in formal),formal_actual_updates=sum(r['step'] for r in formal),
              dev_rollout_seconds=dev_seconds,test_rollout_seconds=test_seconds,video_seconds=video['wall_seconds'],
              note='Nonoverlapping measured intervals. Update times include GPU synchronization and first compilation; process startup/checkpoint serialization/figure rendering and early unlogged zero-action diagnosis are not fully timed. Allocated GPU-hours cannot be inferred from these wall intervals; idle/edit time is excluded.')
    atomic_json(ROOT/'results/round2/compute.json',cost)
    lines=['# Round 2 实验报告','',
           '本轮在真实旋转柜体的自定义 MetaWorld 扩展中，完成了 40 条新示范、四次各 20,000 步正式训练、240 次 dev 评测和 480 次冻结测试。两方法共享数据、信息、网络参数量和训练预算。', '',
           '| 模型 | EMA 步数 | IID | 位置 OOD | yaw near | yaw mid | yaw far | combined | yaw 主指标 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        vals=[f'{100*row[s]:.1f}%' for s in SPLITS[1:]]
        lines.append('| '+row['run_id']+' | '+str(row['selected_step'])+' | '+' | '.join(vals)+f' | {100*row["yaw_primary"]:.1f}% |')
    lines+=['','各单项分母为20；yaw 主指标为 near/mid/far 等权平均，合计60。所有异常保留在原分母中。训练仅一个 seed，每个测试状态各执行一次推理、两方法共用该状态的推理 seed；不能把这些结果外推为跨训练 seed 的稳定提升。','']
    differences=[]
    stage_names={'no_approach':'未接近','approach_no_contact':'接近但无直接接触','contact_no_progress':'接触但未开始打开','progress_not_completed':'已开始打开但未完成'}
    for task in ('drawer','door'):
        world=results[f'r2_{task}_world_n20_s0'];frame=results[f'r2_{task}_frame_n20_s0']
        diff=100*(frame['yaw_primary']['success_rate']-world['yaw_primary']['success_rate']);differences.append(diff)
        lines.append(f'{task}：frame 的 yaw 主成功率相对 world 变化 {diff:+.1f} 个百分点。world/frame 的 yaw 接触率分别为 {world["yaw_primary"]["contact_rate"]:.1%}/{frame["yaw_primary"]["contact_rate"]:.1%}，开始打开率为 {world["yaw_primary"]["opening_rate"]:.1%}/{frame["yaw_primary"]["opening_rate"]:.1%}。这些诊断是相关证据，不能单独证明失败机制。')
        wc=world['yaw_primary']['failure_categories'];fc=frame['yaw_primary']['failure_categories']
        lines.append('yaw 失败类别（world/frame 回合数）：'+ '；'.join(f'{label} {wc[k]}/{fc[k]}' for k,label in stage_names.items())+'。')
        largest=max(stage_names,key=lambda k:wc[k]-fc[k])
        if wc[largest]>fc[largest]:lines.append(f'其中减少最多的失败类别是“{stage_names[largest]}”，是本轮最直接的诊断线索；该分类依据控制步接触与进度记录，仍无法排除多个因素共同作用。')
        lines.append('')
    saturated=all(all(row[s]==1 for s in SPLITS[1:]) for row in rows)
    if saturated:finding='四模型在全部冻结测试上再次饱和；当前测试范围缺少区分度，无法据此判断坐标先验是否提高成功率。'
    elif all(d>0 for d in differences):finding='当前配对实验支持带方向的坐标先验改善 yaw 泛化；改善幅度与任务相关，尚待独立训练种子确认。'
    elif any(d>0 for d in differences):finding='坐标先验的收益具有任务依赖性；至少一项 yaw 主指标提高，但本轮不支持对两任务作统一的提升结论。'
    else:finding='本轮没有观察到 frame 在 yaw 主成功率上的净提升。应按已冻结结果报告这一点，不能把表示变换本身当作收益证据。'
    next_step='扩大经过专家与物理校准的几何测试范围，以获得能区分两种表示的任务难度；本轮不自动启动额外实验。' if saturated else '保持本轮环境、数据规模和评测协议不变，补充独立训练 seed，检查当前坐标先验效应的稳定性；本轮不自动启动额外训练。'
    lines+=['远档存在明显方向差异：'+ '；'.join(f'{task} frame 在约−50°为{results[f"r2_{task}_frame_n20_s0"]["yaw_by_sign"]["test_yaw_far:-1"]["successes"]}/10、约+50°为{results[f"r2_{task}_frame_n20_s0"]["yaw_by_sign"]["test_yaw_far:1"]["successes"]}/10' for task in ('drawer','door'))+'。这一参数化没有消除所有方向上的泛化失败。','']
    lines+=['',finding,'','yaw 主指标对应的操作诊断如下。首次成功均步数仅对成功回合计算；不同失败率下不能单独比较。裁剪率包含 xyz 与 gripper，逐坐标计数保留在逐回合 JSON。','',
            '| 模型 | 首次成功均步数 | 平均最大 / 最终进度 | 实际动作步裁剪率 | 越界 / 异常回合 |',
            '|---|---:|---:|---:|---:|']
    for run in RUNS:
        m=results[run['run_id']]['yaw_primary']
        first='—' if m['mean_first_success_step'] is None else f'{m["mean_first_success_step"]:.2f}'
        final='—' if m['mean_final_progress'] is None else f'{m["mean_final_progress"]:.3f}'
        lines.append(f'| {run["run_id"]} | {first} | {m["mean_max_progress"]:.3f} / {final} | {m["clipping_step_rate"]:.1%} | {m["invalid_joint_episodes"]} / {m["exceptions"]} |')
    lines+=['','三步开度阈值达成不一定等于有效成功：若回合历史出现超过预定容差的关节越界，仍判失败。因此“未完成”类别可能包含已打开但无效的轨迹，不能单凭该标签断言未打开。以下只用于解释既定规则，不替换主指标。','',
            '| 模型 | yaw 三步阈值达成 | 其中曾越界 | 有效成功 |','|---|---:|---:|---:|']
    for run in RUNS:
        c=success_audit[run['run_id']]['yaw_primary']
        lines.append(f'| {run["run_id"]} | {c["threshold_reached"]}/60 | {c["threshold_reached_with_joint_violation"]} | {c["valid_success"]}/60 |')
    lines+=['','全部拆分的次要指标见 [metrics_by_split.csv](results/round2/metrics_by_split.csv)。','','## 设置与可比性','',
        '两任务最终都采用 A 档：训练 yaw ±10°，测试 ±20°/±35°/±50°，各含 ±2° 扰动。每任务最终57项物理/专家复查全部通过；校准没有读取模型结果。训练仅使用按四个 yaw 区间直接采集的20条成功示范，dev/test使用独立初始状态。',
        '相对 Round 1：本轮真实旋转柜体，统一 XYZW 四元数、修正抽屉把手偏移、以真实关节进度和连续三步定义成功；frame 同时变换观测和动作。两方法的四元数与 yaw 不标准化，扩散采样均取消 clean-sample 裁剪，动作先解码到 world 再作原生裁剪。门的目标与旧版 world-X success 不同，因此不能直接把两轮成功率作同任务难度比较。',
        '校准曾发现后挡条穿透旋转抽屉并造成零动作开启；统一后移该配件后修复。旧探测和失败版本均保留。机器人、桌面、重力、关节范围、原生控制尺度与频率保持一致。详见 [协议](ROUND2_PROTOCOL.md) 与 [几何审计](docs/ROUND2_GEOMETRY.md)。',
        '该表示保留 world 把手位置与 yaw，未强制网络丢弃世界信息；固定末端姿态也意味着完整系统不是严格旋转等变的。','',
        '模型测试结束后，使用完全不变的冻结专家对全部240个测试状态补充了执行参考，用于解释剩余失败。该结果是后置诊断，未用于选角度、调专家、换测试样本或训练模型，也不应被称为理论物理上限。','',
        '| 专家参考 | IID | 位置 | near | mid | far | combined |','|---|---:|---:|---:|---:|---:|---:|',
        *['| '+task+' | '+' | '.join(f'{reference["tasks"][task]["metrics"][s]["valid_successes"]}/20' for s in SPLITS[1:])+' |' for task in ('drawer','door')],
        '','完整记录见 [expert_reference.json](results/round2/expert_reference.json)。所有校准与该参考检查合计仍受每任务600次专家 rollout 上限约束。',
        f'门专家参考中有{sum(r["success"] and r["invalid_joint"] for r in reference["tasks"]["door"]["records"])}次已达到三步开度阈值、但因历史关节越界而无效的回合。',
        '冻结专家在两个任务的正向 far 状态上的有效成功数分别为：'+ '；'.join(f'{task} {sum(r["success"] and not r["invalid_joint"] for r in reference["tasks"][task]["records"] if r["record"]["split"]=="test_yaw_far" and r["record"]["task_params"]["yaw_degrees"]>0)}/10' for task in ('drawer','door'))+'。这些相同状态上的成功专家轨迹说明，frame 在该方向的失败不能一概解释为物理不可行；仍有未解决的策略泛化问题。','',
        '## 验证与诊断','',
        '40/40 条示范完整重放，280/280 个 dev/test 初始快照重置通过。几何/表示/动作/归一化/时间对齐测试通过。每配置200步短调试在100步换进程恢复，正式训练各从头初始化。配对初始权重、参数量以及20,000步采样/噪声/timestep摘要一致；EMA选择只依据dev20，并在全部测试前统一冻结。',
        '接触来自真实把手碰撞 geom 与 gripper geom 的接触对；距离单独报告。按控制步采样可能漏掉物理子步内短暂接触。关节异常与仿真异常见逐回合 JSON；有异常的回合不能算成功。成功者平均步数是条件均值，失败率不同时不宜单独比较。累计成功曲线将失败回合保留到500步。','',
        '![各测试拆分成功率](artifacts/round2/figures/split_success.png)','',
        '![yaw 大小与符号](artifacts/round2/figures/yaw_sign_curves.png)','',
        '![进度和失败诊断](artifacts/round2/figures/progress_failures.png)','',
        '![训练与开发集](artifacts/round2/figures/training_dev.png)','',
        '![累计成功曲线](artifacts/round2/figures/cumulative_success.png)','',
        '## 视频','',
        '按预先固定规则选取 IID/far 的首个成功和首个失败；另取首个 world 失败而 frame 成功的配对状态并排展示（左 world、右 frame）。不存在的类别明确列出；没有挑选最佳表现。','']
    for r in video['records']:
        if r.get('path'):lines.append(f'- [{r["label"]}]({r["path"]})：{r["episode_id"]}')
        elif r.get('absent'):lines.append(f'- {r["label"]}：不存在符合条件的回合。')
        else:lines.append(f'- {r["label"]}：渲染失败，错误保留于视频 manifest；数值结果有效。')
    lines+=['','## 计算成本与复现','',
        '| 阶段 | 实际计量 |','|---|---:|',
        f'| 专家校准 | {cost["calibration_expert_rollouts"]} 次；{cost["calibration_expert_seconds"]:.1f} 秒 |',
        f'| 冻结测试的后置专家参考 | {cost["frozen_test_expert_reference_rollouts"]} 次；{cost["frozen_test_expert_reference_seconds"]:.1f} 秒 |',
        f'| 已记录零动作检查 | {cost["zero_action_seconds"]:.1f} 秒 |',
        f'| 数据采集 / 完整审计 | {cost["data_collection_seconds"]:.1f} / {cost["data_audit_seconds"]:.1f} 秒 |',
        f'| 短调试更新 / 8步闭环检查 | {cost["debug_actual_updates"]} updates；{cost["debug_update_seconds"]:.1f} / {cost["debug_rollout_seconds"]:.1f} 秒 |',
        f'| 正式训练 | {cost["formal_actual_updates"]} updates；{cost["formal_update_seconds"]:.1f} 秒 |',
        f'| dev / final test | {cost["dev_rollout_seconds"]:.1f} / {cost["test_rollout_seconds"]:.1f} 秒 |',
        f'| 视频渲染 | {cost["video_seconds"]:.1f} 秒 |','',
        '表中各项为互不重复的已计量区间；训练时间包括同步和首次编译，未完整覆盖进程启动、checkpoint写盘、绘图以及早期未计时零动作诊断。无法从这些墙钟区间推断精确分配 GPU 小时，未把编辑和等待时间算作训练。具体数据见 [compute.json](results/round2/compute.json)。',
        '运行：`pixi run round2`；阶段入口见 [协议](ROUND2_PROTOCOL.md)。主要产物：[汇总 CSV](results/round2/summary.csv)、[完整结果 JSON](results/round2/summary.json)、[checkpoint 选择](results/round2/selection.json)、[数据 manifest](data/round2/manifest.json)、`runs/round2/` 中的 checkpoint 和日志、[视频清单](artifacts/round2/videos/manifest.json)。',
        '','## 唯一下一步','',
        next_step,'']
    (ROOT/'ROUND2_REPORT.md').write_text('\n'.join(lines))
    complete=dict(completed=True,formal_updates=80000,dev_episodes=240,test_episodes=480,
                  report_hash=sha256(ROOT/'ROUND2_REPORT.md'),summary_hash=sha256(ROOT/'results/round2/summary.json'),
                  video_manifest_hash=sha256(VIDEO/'manifest.json'))
    atomic_json(ROOT/'results/round2/complete.json',complete)
    print('ROUND2 COMPLETE',complete,flush=True)
    return complete


if __name__=='__main__':report()
