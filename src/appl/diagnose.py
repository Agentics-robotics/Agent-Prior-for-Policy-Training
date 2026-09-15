"""The ordered DP diagnosis ladder; no formal test feedback enters these jobs."""
from pathlib import Path
import time
import numpy as np
from .io import ROOT,read,atomic,event,digest
from . import data


def d1(cfg):
    from .train import fit
    assert read(cfg['output']/'diagnostics/D0/result.json')['success']
    output=cfg['output']/'diagnostics/D1'
    result=fit(cfg,output,ids=[cfg['data']['train_ids'][0]],updates=cfg['diagnostics']['d1_updates'],fixed_batch=True)
    summary=dict(stage='D1',neural_training_verified=result['neural_training_verified'],
        sampling_joint_rmse=result['sampling_joint_rmse'],sampling_gripper_mae=result['sampling_gripper_mae'],
        success=result['sampling_joint_rmse']<.05 and result['sampling_gripper_mae']<.15,
        sampling_gate='joint RMSE <0.05 rad; gripper command MAE <0.15; fixed multi-phase training batch',
        reload_max_action_error=result['reload_max_action_error'])
    atomic(output/'diagnostic.json',summary)
    return summary


def d2(cfg):
    from .train import fit
    from .evaluate import diagnostic_rollouts
    assert read(cfg['output']/'diagnostics/D1/diagnostic.json')['success']
    output=cfg['output']/'diagnostics/D2'
    result=fit(cfg,output/'training',ids=[cfg['data']['train_ids'][0]],updates=cfg['diagnostics']['d2_updates'])
    trials=diagnostic_rollouts(cfg,result['checkpoint'],output/'rollouts',[1000])
    value=dict(stage='D2',success=all(t['success'] for t in trials),results=trials,training=result)
    atomic(output/'result.json',value);return value


def d2_sampler(cfg):
    from .gpu import verify_cuda
    from .train import LoadedPolicy
    from .evaluate import episode
    from .envs.native import make_environment
    verify_cuda();env=make_environment();results=[]
    checkpoint=cfg['output']/'diagnostics/D2/training/last.pt'
    variants=[dict(name='ddim50_chunk8',denoising_inference_steps=50,execution_steps=8),
              dict(name='ddim50_chunk4',denoising_inference_steps=50,execution_steps=4)]
    output=cfg['output']/'diagnostics/D2_sampling'
    atomic(output/'plan.json',dict(variants=variants,checkpoint_sha256=digest(checkpoint),scope='training-reset diagnostic only; inference sensitivity, not final test'))
    try:
        for change in variants:
            policy=LoadedPolicy(checkpoint,seed=1000)
            policy.spec['training'].update({k:v for k,v in change.items() if k!='name'})
            result=episode(cfg,env,policy,output/change['name'],1000)
            results.append(dict(sampling_config=change,**result))
        value=dict(stage='D2_sampler',success=any(r['success'] for r in results),results=results)
        atomic(output/'result.json',value);return value
    finally:env.close()


def d3_skill(cfg,role):
    from .train import fit
    from .evaluate import diagnostic_rollouts
    assert read(cfg['output']/'diagnostics/D1/diagnostic.json')['success']
    identifier=cfg['data']['train_ids'][0]
    recorded=read(cfg['data']['directory']/identifier/'collection_receipt.json')
    marks={r['phase']:r['step'] for r in recorded['phases']}
    total=len(read(cfg['data']['directory']/(identifier+'.json'))['actions'])
    spans={'open_drawer':(0,marks['take_red_out']),'move_red':(marks['take_red_out'],marks['put_blue_in']),
           'move_blue':(marks['put_blue_in'],total)}
    start,stop=spans[role];output=cfg['output']/'diagnostics/D3'/role
    atomic(output/'diagnostic_scope.json',dict(cuts_source='original acquisition phase marks; not the eventual API-defined M1/M2 cuts',span=[start,stop],trajectories=[identifier]))
    result=fit(cfg,output/'training',ids=[identifier],segments={identifier:[start,stop]},updates=cfg['diagnostics']['d3_updates'])
    trials=diagnostic_rollouts(cfg,result['checkpoint'],output/'rollouts',[int(identifier[4:])],skill=role,prefix_id=identifier,boundary=start)
    value=dict(stage='D3',skill=role,success=all(t['success'] for t in trials),results=trials,training=result)
    atomic(output/'result.json',value);return value


def d3_open(cfg):return d3_skill(cfg,'open_drawer')
def d3_red(cfg):return d3_skill(cfg,'move_red')
def d3_blue(cfg):return d3_skill(cfg,'move_blue')


def d2_quaternion(cfg):
    """A declared numerical-scaling diagnostic, no extra data or controller."""
    import copy
    from .train import fit
    from .evaluate import diagnostic_rollouts
    cfg=copy.deepcopy(cfg);cfg['training']['quaternion_normalization']='unit_component_bounds'
    output=cfg['output']/'diagnostics/D2_quaternion'
    atomic(output/'plan.json',dict(scope='Single training-reset diagnostic, same data/seed/backbone/update budget as D2',
        change='All 12 quaternion components use their physical [-1,1] bounds; other fields retain training-only statistics',
        training=cfg['training'],seed=0,reset=1000))
    training=fit(cfg,output/'training',ids=[cfg['data']['train_ids'][0]],updates=cfg['diagnostics']['d2_updates'])
    results=diagnostic_rollouts(cfg,training['checkpoint'],output/'rollouts',[1000])
    value=dict(stage='D2_quaternion',success=all(r['success'] for r in results),results=results,training=training)
    atomic(output/'result.json',value);return value


def d4(cfg):
    """Full original training budget and the preregistered 20 ID development resets."""
    from .train import fit
    from .evaluate import diagnostic_rollouts
    for role in ('open_drawer','move_red','move_blue'):
        if not (cfg['output']/'diagnostics/D3'/role/'result.json').exists():
            raise ValueError('Complete and inspect all D3 diagnostics before D4')
    output=cfg['output']/'diagnostics/D4'
    seeds=cfg['diagnostics']['id_development_seeds']
    plan=dict(stage='D4',seeds=seeds,required_successes=cfg['diagnostics']['id_gate_successes'],
        trials=cfg['diagnostics']['id_gate_trials'],training=cfg['training'],ids=cfg['data']['train_ids'],
        selection='One predeclared last EMA, no checkpoint choice from these results')
    if (output/'plan.json').exists() and read(output/'plan.json')!=plan:raise ValueError('D4 plan changed')
    atomic(output/'plan.json',plan)
    training=fit(cfg,output/'training')
    results=diagnostic_rollouts(cfg,training['checkpoint'],output/'rollouts',seeds)
    successes=sum(r['success'] for r in results)
    value=dict(stage='D4',success=successes>=plan['required_successes'],successes=successes,trials=len(results),
               results=results,training=training,required_successes=plan['required_successes'])
    atomic(output/'result.json',value);return value


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
