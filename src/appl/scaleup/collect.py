"""Fixed motion-planning expert; its actions are training data, never a learned-policy fallback."""
from pathlib import Path
import time
import numpy as np
from PIL import Image
from ..io import read,atomic,digest,archive_source,event
from ..envs.adapter import state_from_obs
from .tasks import measure,contract
from .environment import make,reset


class Recorder:
    def __init__(self,env,root,identifier,obs):
        self.env=env;self.root=Path(root);self.identifier=identifier
        self.actions=[];self.observations=[];self.phases=[]
        (self.root/identifier).mkdir(parents=True,exist_ok=False)
        self.record(obs)

    def __getattr__(self,name):return getattr(self.env,name)

    def record(self,obs):
        index=len(self.actions);path=f'{self.identifier}/frame_{index:04d}.png'
        Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(self.root/path)
        self.observations.append(dict(timestamp=index/20,state=state_from_obs(obs),images=dict(front=path)))

    def mark(self,name):self.phases.append(dict(name=name,step=len(self.actions)))

    def step(self,action):
        if len(self.actions)>=5000:raise RuntimeError('Expert exceeded physical collection cap')
        result=self.env.step(action)
        self.actions.append(np.asarray(action,dtype=np.float32).reshape(-1).tolist());self.record(result[0])
        return result


def demonstrate(env,recorder,spec):
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    native=env.unwrapped
    planner=PandaArmMotionPlanningSolver(recorder,vis=False,debug=False,base_pose=native.agent.robot.pose,
        visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=.5,joint_acc_limits=.5)
    def current(name):return getattr(native,name).pose.p[0].cpu().numpy()
    def move(point,free=False):
        # Parallel-jaw symmetry permits the opposite closing direction, keeping
        # the wrist away from its lower joint limit across the tabletop region.
        pose=native.agent.build_grasp_pose(np.array([0,0,-1.]),np.array([0,-1.,0]),np.asarray(point))
        kwargs=dict(time_step=native.control_timestep)
        fn=planner.planner.plan_qpos_to_pose if free else planner.planner.plan_screw
        if free:kwargs['wrt_world']=True
        result=fn(np.r_[pose.p,pose.q],native.agent.robot.get_qpos()[0].cpu().numpy(),**kwargs)
        if result['status']!='Success':raise RuntimeError('Expert planning failed: '+str(result['status']))
        planner.follow_path(result,refine_steps=8)
    try:
        for index,(name,target) in enumerate(spec['program']):
            recorder.mark(f'{index}_{name}_pick');start=current(name).copy()
            move(start+[0,0,.25],free=index==0);move(start)
            planner.close_gripper(t=12);move(start+[0,0,.26])
            if current(name)[2]<start[2]+.18:raise RuntimeError('Expert failed physical pickup: '+name)
            recorder.mark(f'{index}_{name}_place')
            goal=np.asarray(spec['buffer'] if target=='buffer' else spec['goals'][target])
            move(goal+[0,0,.28]);move(goal+[0,0,.003]);planner.open_gripper(t=12);move(goal+[0,0,.28])
        recorder.mark('settle');planner.open_gripper(t=20)
        result=measure(recorder.observations[-1]['state'],contract(spec))
        if not result['success']:raise RuntimeError('Expert final goals failed: '+str(result))
        return result
    finally:planner.close()


def collect(spec,directory,seeds,run_root,condition='ID',scope='training'):
    from ..gpu import verify_cuda
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);run_root=Path(run_root)
    source=archive_source(run_root);device=verify_cuda();env=make(spec)
    results=[]
    try:
        for seed in seeds:
            identifier='demo'+str(seed);path=root/(identifier+'.json')
            if path.exists():
                old=read(path)
                if old['task_spec_sha256']!=digest(run_root/'task.json'):raise ValueError('Collected task changed')
                results.append(dict(seed=seed,steps=len(old['actions']),sha256=digest(path)));continue
            obs,state=reset(env,seed,condition)
            recorder=Recorder(env,root,identifier,obs);started=time.monotonic()
            try:
                goals=demonstrate(env,recorder,spec)
                record=dict(schema='appl.demonstration.v1',trajectory_id=identifier,provenance='original_demonstration',
                    task_id=spec['task_id'],goal=spec['description'],actions=recorder.actions,observations=recorder.observations,
                    phases=recorder.phases,seed=seed,condition=condition,scope=scope,source_version=source,
                    task_spec_sha256=digest(run_root/'task.json'),device=device,success=goals,
                    elapsed_seconds=time.monotonic()-started,collector='fixed_native_motion_planner_no_pose_overwrite')
                atomic(path,record);results.append(dict(seed=seed,steps=len(recorder.actions),sha256=digest(path)))
                print(dict(task=spec['task_id'],seed=seed,steps=len(recorder.actions),success=True),flush=True)
            except Exception as error:
                atomic(root/(identifier+'.failure.json'),dict(error=str(error),actions=recorder.actions,
                    observations=recorder.observations,phases=recorder.phases,seed=seed,source_version=source,
                    device=device,scope=scope,automatic_retry=False));raise
        atomic(root/'collection.json',dict(task_id=spec['task_id'],scope=scope,condition=condition,
            count=len(results),episodes=results,source_version=source,device=device))
        return results
    finally:env.close()
