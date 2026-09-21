"""Shared replay and infrastructure checks; M0 runs live in dp_baseline."""
from pathlib import Path
import time
import numpy as np
from .io import read,atomic,event,digest
from . import data


def capability_m1(cfg):
    """No-prior control with the exact API capability candidate's source cuts."""
    from .train import fit
    from .evaluate import diagnostic_rollouts
    base=cfg['output']/'api_capability';submitted=read(base/'submission.json')
    metadata=read(base/'versions'/submitted['version']/'candidate.json')
    output=cfg['output']/'diagnostics/capability_M1'
    atomic(output/'plan.json',dict(stage='framework capability matched baseline, not formal',
        api_version=submitted['version'],segments=metadata['segments'],updates=5000,seed=0,
        shared='Original training IDs, API cuts, DP backbone widths/depth, scheduler, input information and fixed skill checks'))
    training=fit(cfg,output/'training',segments=metadata['segments'],updates=5000)
    start=metadata['segments']['demo1000'][0]
    results=diagnostic_rollouts(cfg,training['checkpoint'],output/'rollouts',[1000],skill=metadata['skill'],prefix_id='demo1000',boundary=start)
    value=dict(stage='capability_M1',success=all(r['success'] for r in results),results=results,training=training,
        api_version=submitted['version'],formal_result=False)
    atomic(output/'result.json',value);return value


def api_mujoco(cfg):
    from .verification import run
    base=cfg['output']/'api_capability';submission=read(base/'submission.json')
    directory=base/'versions'/submission['version'];metadata=read(directory/'candidate.json')
    output=base/'mujoco_capability';output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():return read(output/'result.json')
    request=dict(output=str(output),scope='framework_capability',round=0,
        policies=[dict(metadata=metadata,candidate=str(directory),checkpoint=str(Path(submission['training']['output'])/'last.pt'),
                       checkpoint_sha256=submission['training']['checkpoint_sha256'])])
    atomic(output/'request.json',request)
    return run(cfg,request)


def resume_check(cfg):
    import copy
    from .train import fit
    cfg=copy.deepcopy(cfg);cfg['training'].update(updates=4,batch_size=8,checkpoint_every=2)
    output=cfg['output']/'checks/exact_resume'
    if (output/'verification.json').exists():return read(output/'verification.json')
    ids=[cfg['data']['train_ids'][0]]
    control=fit(cfg,output/'uninterrupted',ids=ids,seed=42)
    paused=fit(cfg,output/'resumed',ids=ids,seed=42,stop_after_checkpoint=2)
    if paused.get('step')!=2:raise ValueError('Resume check did not stop at the declared checkpoint')
    resumed=fit(cfg,output/'resumed',ids=ids,seed=42)
    result=dict(success=control['final_weights_sha256']==resumed['final_weights_sha256'],
        uninterrupted_steps=4,resumed_steps=[2,2],total_actual_optimizer_steps=8,
        final_weights_sha256=control['final_weights_sha256'],resumed_weights_sha256=resumed['final_weights_sha256'],
        scope='Real CUDA baseline infrastructure check; no API candidate or scientific model count')
    atomic(output/'verification.json',result)
    if not result['success']:raise ValueError('Restoring optimizer/EMA/RNG changed the resumed numerical result')
    return result


def native_journal_check(cfg):
    from .gpu import verify_cuda
    from .envs.native import make_environment
    from .envs.adapter import reset
    # Intentionally import SQLite after SAPIEN to test the actual problematic
    # dynamic-loader order, not merely move the failing import out of sight.
    from .journal import Journal
    device=verify_cuda();env=make_environment();output=cfg['output']/'checks/native_journal'
    try:
        obs,state=reset(env,8000);journal=Journal(output);journal.set('probe',state['qpos'])
        result=dict(success=journal.get('probe')==state['qpos'],device=device,physics_steps=0,
                    image_shape=list(obs['sensor_data']['front']['rgb'].shape),sqlite_after_sapien=True)
        journal.db.close();atomic(output/'result.json',result);return result
    finally:env.close()


def d0(cfg):
    from .gpu import verify_cuda
    from .envs.native import make_environment
    from .envs.adapter import reset,step
    from .envs.evaluator import measure,SuccessTracker
    from PIL import Image
    output=cfg['output']/'diagnostics/D0'
    if (output/'result.json').exists():return read(output/'result.json')
    output.mkdir(parents=True,exist_ok=True)
    device=verify_cuda(); env=make_environment()
    results=[]
    try:
        native=env.unwrapped
        contract=dict(action_low=native.single_action_space.low.tolist(),action_high=native.single_action_space.high.tolist(),
            control_freq=native.control_freq,sim_freq=native.sim_freq,controller=str(native.agent.controller.configs),
            action_semantics='pd_joint_pos; absolute arm radians; gripper normalized only',
            reset='original seed/layout reset; old dataset has no complete simulator snapshot',device=device)
        atomic(output/'controller.json',contract)
        for identifier in cfg['diagnostics']['replay_ids']:
            if (output/identifier/'result.json').exists():
                results.append(read(output/identifier/'result.json'));continue
            directory=output/identifier;directory.mkdir(exist_ok=False)
            t=read(cfg['data']['directory']/(identifier+'.json'))
            obs,state=reset(env,int(identifier.removeprefix('demo')))
            tracker=SuccessTracker(cfg['evaluation']); started=time.monotonic();errors=[]
            initial_error=max(abs(np.asarray(state[k])-t['observations'][0]['state'][k]).max() for k,_ in data.FIELDS)
            for i,action in enumerate(t['actions']):
                obs,state=step(env,action)
                metrics=tracker.update(measure(env,state,cfg['evaluation']))
                reference=t['observations'][i+1]['state']
                errors.append([np.linalg.norm(np.asarray(state[k])-reference[k]) for k in ('qpos','tcp_pose','red_pose','blue_pose')])
                event(directory/'trace.jsonl','step',step=i+1,state=state,action=action,metrics=metrics)
                if i%20==0 or i==len(t['actions'])-1:
                    Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(directory/f'frame_{i+1:04d}.png')
            result=dict(id=identifier,success=metrics['success'],steps=len(t['actions']),final=metrics,
                initial_max_error=initial_error,rmse_fields=['qpos','tcp_pose','red_pose','blue_pose'],
                rms_l2_errors=np.sqrt(np.mean(np.square(errors),axis=0)).tolist(),
                elapsed_seconds=time.monotonic()-started,source_sha256=digest(cfg['data']['directory']/(identifier+'.json')))
            atomic(directory/'result.json',result);results.append(result)
            print(result,flush=True)
        result=dict(stage='D0',success=all(r['success'] for r in results),episodes=results,device=device)
        atomic(output/'result.json',result)
        return result
    finally:env.close()
