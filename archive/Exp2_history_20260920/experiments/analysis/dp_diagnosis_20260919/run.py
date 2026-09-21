"""Frozen-DP diagnostics on original training resets; no training or Runtime API."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from appl.io import ROOT, atomic, digest, read, source_manifest, object_hash

BASE = Path(__file__).resolve().parent
TASKS = ['drawer_exchange', 'two_block_sort', 'buffer_swap', 'unstack_sort', 'tray_pack']


def settings(task):
    from appl.scaleup.protocol import config, naive_paths
    cfg = config(task)
    source, checkpoint = naive_paths(task)
    if task == 'drawer_exchange':
        original = read(ROOT / 'experiments/exp2/configs/m0.json')['data']
        directory, ids = ROOT / original['directory'], original['train_ids']
    else:
        directory = ROOT / cfg['normalization']['directory']
        ids = cfg['normalization']['train_ids']
    return cfg, source, checkpoint, directory, ids


def prepare():
    if (BASE / 'plan.json').exists():
        return read(BASE / 'plan.json')
    tasks = {}
    for name in TASKS:
        cfg, source, checkpoint, directory, ids = settings(name)
        tasks[name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            source=None if source is None else str(source),
            demonstrations={i:dict(path=str(directory / (i+'.json')),
                sha256=digest(directory / (i+'.json'))) for i in ids},
            completion_contract=str(cfg['completion_contract']),
            completion_contract_sha256=digest(cfg['completion_contract']))
    plan = dict(created=time.time(), owner='developer diagnostic', tasks=tasks,
        purpose='Separate reset/environment reproduction, conditional action prediction and closed-loop competence.',
        maximum_control_steps=1500, environment_wrapper_limit=5000,
        diffusion_seed='original layout seed parsed from demonstration identifier',
        scope='12 original training resets per task, not new held-out results',
        training_updates=0, runtime_API_requests=0,
        devices=[2,3,4,5,7], max_simultaneous_physical_devices=5,
        framework_source=source_manifest(), diagnostic_source_sha256=digest(__file__),
        occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv'],text=True))
    atomic(BASE / 'plan.json', plan)
    return plan


def difference(actual, expected):
    import numpy as np
    return {k:float(np.max(np.abs(np.asarray(actual[k])-v))) for k,v in expected.items()}


def snapshot(obs):
    return obs['sensor_data']['front']['rgb'][0].cpu().numpy()


def video(frames, output):
    from PIL import Image
    directory=output/'frames';directory.mkdir()
    for i,frame in enumerate(frames):Image.fromarray(frame).save(directory/f'{i:04d}.png')
    subprocess.run(['ffmpeg','-v','error','-threads','1','-framerate','6','-i',str(directory/'%04d.png'),
        '-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(output/'replay.mp4')],check=True)


def reset_environment(task, seed):
    from appl.scaleup.evaluate import environment
    return environment(task,seed,'ID')


def replay(task, identifier, demo, seed, goal, output):
    import numpy as np
    from appl.envs.adapter import step
    from appl.scaleup.tasks import measure
    output.mkdir(parents=True)
    env=None
    try:
        env,obs,state=reset_environment(task,seed)
        initial_errors=difference(state,demo['observations'][0]['state'])
        if max(initial_errors.values()) > 1e-6:raise ValueError(('Initial reset mismatch',initial_errors))
        frames=[snapshot(obs)];max_errors=dict(initial_errors);first_success=None
        native=env.unwrapped
        controller=dict(control_mode=native.control_mode,control_timestep=float(native.control_timestep),
            action_low=native.single_action_space.low.tolist(),action_high=native.single_action_space.high.tolist(),
            controller_config=repr(native.agent.controller.configs),sim_config=repr(native.sim_config))
        atomic(output/'environment.json',controller)
        for index,action in enumerate(demo['actions'],1):
            obs,state=step(env,np.asarray(action,np.float32))
            for k,v in difference(state,demo['observations'][index]['state']).items():max_errors[k]=max(max_errors[k],v)
            if first_success is None and measure(state,goal)['success']:first_success=index
            if index%20==0:frames.append(snapshot(obs))
        frames.append(snapshot(obs));video(frames,output)
        result=dict(task=task,demonstration=identifier,steps=len(demo['actions']),success=measure(state,goal)['success'],
            first_success=first_success,initial_errors=initial_errors,max_observation_errors=max_errors,
            final=measure(state,goal),finished=time.time())
        atomic(output/'result.json',result)
        return result
    finally:
        if env is not None:env.close()


def rollout(task,identifier,demo,seed,goal,policy,output,max_steps):
    import numpy as np
    from appl.envs.adapter import step
    from appl.prior_policies.data import vector
    from appl.scaleup.tasks import measure
    output.mkdir(parents=True)
    env=None;started=time.monotonic()
    policy.history.clear();policy.queue.clear();policy.generator.manual_seed(seed);policy.timings.clear()
    try:
        env,obs,state=reset_environment(task,seed)
        errors=difference(state,demo['observations'][0]['state'])
        if max(errors.values()) > 1e-6:raise ValueError(('Initial reset mismatch',errors))
        atomic(output/'initial_state.json',state);frames=[snapshot(obs)]
        n=policy.spec['normalizer'];normal_max=0.;first_goals={};narrow=0;longest=0
        heights={k:state[k+'_pose'][2] for k in ['red','blue']};rise={k:0. for k in heights}
        with (output/'trace.jsonl').open('w') as stream:
            for index in range(1,max_steps+1):
                normalized=(vector(state)-np.asarray(n['mean']))/n['std'];normal_max=max(normal_max,float(np.abs(normalized).max()))
                action=np.asarray(policy.action(state),np.float32);space=env.unwrapped.single_action_space
                bounded=np.clip(action,space.low,space.high)
                obs,state=step(env,bounded);metrics=measure(state,goal)
                for k,v in metrics.items():
                    if v and k not in first_goals:first_goals[k]=index
                narrow=narrow+1 if sum(state['qpos'][7:9])<=.005 else 0;longest=max(longest,narrow)
                for k in rise:rise[k]=max(rise[k],state[k+'_pose'][2]-heights[k])
                stream.write(json.dumps(dict(step=index,state=state,raw_action=action.tolist(),action=bounded.tolist(),metrics=metrics))+'\n')
                if index%20==0:frames.append(snapshot(obs))
                if metrics['success']:break
                if index%100==0:atomic(output/'progress.json',dict(step=index,first_goals=first_goals,elapsed_seconds=time.monotonic()-started))
        frames.append(snapshot(obs));video(frames,output)
        result=dict(task=task,demonstration=identifier,layout_seed=seed,diffusion_seed=seed,
            max_steps=max_steps,steps=index,success=metrics['success'],final=metrics,first_goals=first_goals,
            initial_errors=errors,normalized_input_max=normal_max,maximum_object_rise=rise,
            longest_narrow_width_states=longest,inference_chunks=len(policy.timings),
            status='succeeded' if metrics['success'] else 'diagnostic_1500_step_cap',
            elapsed_seconds=time.monotonic()-started,finished=time.time())
        atomic(output/'result.json',result);return result
    finally:
        if env is not None:env.close()


def worker(args):
    import torch
    from appl.gpu import verify_cuda
    plan=read(BASE/'plan.json');frozen=plan['tasks'][args.task]
    if source_manifest()!=plan['framework_source']:raise ValueError('Scientific framework changed')
    if digest(__file__)!=plan['diagnostic_source_sha256']:raise ValueError('Diagnostic source changed')
    torch.set_num_threads(2);device=verify_cuda()
    cfg,source,checkpoint,directory,ids=settings(args.task)
    if digest(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    if args.task=='drawer_exchange':
        from appl.dp_baseline.train import LoadedPolicy
        policy=LoadedPolicy(checkpoint,seed=0)
    else:
        from appl.prior_policies.engine import LoadedPolicy
        policy=LoadedPolicy(source,checkpoint,seed=0)
    goal=read(cfg['completion_contract'])
    if digest(cfg['completion_contract'])!=frozen['completion_contract_sha256']:raise ValueError('Goal changed')
    selected=args.ids or [i for j,i in enumerate(ids) if j%args.shards==args.shard]
    for identifier in selected:
        if identifier not in ids:raise ValueError('Not an original training ID')
        path=directory/(identifier+'.json')
        if digest(path)!=frozen['demonstrations'][identifier]['sha256']:raise ValueError('Demo changed')
        demo=read(path);seed=int(identifier.removeprefix('demo'))
        root=BASE/'runs'/args.task/identifier;root.mkdir(parents=True,exist_ok=True)
        if (root/'complete.json').exists():continue
        atomic(root/'execution.json',dict(pid=os.getpid(),device=device,started=time.time(),checkpoint_sha256=digest(checkpoint)))
        try:
            replay_result=replay(args.task,identifier,demo,seed,goal,root/'expert_replay')
            outcome=rollout(args.task,identifier,demo,seed,goal,policy,root/'frozen_DP',plan['maximum_control_steps'])
            atomic(root/'complete.json',dict(expert_replay=replay_result,policy=outcome,finished=time.time()))
            print(json.dumps(dict(task=args.task,demo=identifier,replay_success=replay_result['success'],policy_success=outcome['success'],steps=outcome['steps'])),flush=True)
        except Exception as error:
            atomic(root/'failure.json',dict(error=str(error),time=time.time(),automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','worker'])
    parser.add_argument('--task',choices=TASKS);parser.add_argument('--gpu',type=int)
    parser.add_argument('--ids',nargs='+');parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--device-isolated',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':return prepare()
    if args.gpu not in read(BASE/'plan.json')['devices']:raise ValueError('Outside diagnostic allocation')
    if not args.device_isolated:
        from appl.gpu import launch
        return launch(args.gpu,sys.argv[1:],module='experiments.exp2.analysis.dp_diagnosis_20260919.run')
    worker(args)


if __name__=='__main__':main()
