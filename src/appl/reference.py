"""Independent fixed motion planner checks, never added to training examples."""
import time
import numpy as np
from .io import atomic,event,read
from .envs.adapter import reset,state_from_obs
from .envs.evaluator import measure,SuccessTracker
from .protocol import CONDITIONS


class Recorder:
    def __init__(self,env,directory,cfg):
        self.env=env;self.directory=directory;self.cfg=cfg
        self.steps=0;self.tracker=SuccessTracker(cfg['evaluation']);self.phases=[]

    def __getattr__(self,name):return getattr(self.env,name)

    def mark(self,name):
        self.phases.append(dict(name=name,step=self.steps))
        atomic(self.directory/'phases.json',self.phases)

    def step(self,action):
        from PIL import Image
        result=self.env.step(action);obs=result[0];self.steps+=1
        state=state_from_obs(obs);metrics=self.tracker.update(measure(self.env,state,self.cfg['evaluation']))
        event(self.directory/'trace.jsonl','step',step=self.steps,state=state,action=np.asarray(action).reshape(-1).tolist(),metrics=metrics)
        if self.steps%20==1:Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(self.directory/f'frame_{self.steps:04d}.png')
        return result


def check(cfg):
    from .gpu import verify_cuda
    from .envs.native import make_environment
    from .envs.demonstrator import collect_episode
    output=cfg['output']/'reference'
    if (output/'result.json').exists():return read(output/'result.json')
    plan=[dict(condition=condition,seed=seed,**options) for condition,options in CONDITIONS.items() for seed in (8000,8001)]
    atomic(output/'plan.json',dict(cases=plan,scope='Independent reference controller only; no training data or learned-policy scene filtering',
        position_ood='red world y +0.03 m, blue world x +0.05 m relative to unchanged base distribution',
        budget='six representative reachability episodes, not proof for every continuous initial state'))
    device=verify_cuda();env=make_environment();results=[]
    try:
        for case in plan:
            d=output/(case['condition']+'_'+str(case['seed']))
            if (d/'result.json').exists():results.append(read(d/'result.json'));continue
            if d.exists():raise RuntimeError('Interrupted reference execution needs a recorded new attempt')
            d.mkdir();_,state=reset(env,case['seed'],case['variant'],case['offsets'])
            atomic(d/'initial_state.json',state);recorder=Recorder(env,d,cfg);started=time.monotonic()
            try:
                collect_episode(env,recorder)
                result=dict(**case,success=recorder.tracker.last['success'],status='completed',final=recorder.tracker.last)
            except RuntimeError as error:
                result=dict(**case,success=False,status='reference_failed',error=str(error),final=recorder.tracker.last)
            result.update(steps=recorder.steps,elapsed_seconds=time.monotonic()-started,training_data=False)
            atomic(d/'result.json',result);results.append(result)
            print({k:result[k] for k in ('condition','seed','success','steps')},flush=True)
        result=dict(success=all(r['success'] for r in results),cases=results,device=device)
        atomic(output/'result.json',result);return result
    finally:env.close()
