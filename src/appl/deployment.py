"""Frozen high-level API; persistent local numerical workers execute each skill."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jsonschema
from .agent import AgentLoop
from .design import schema,client
from .journal import Journal,encode
from .io import atomic,event
from .envs.adapter import step
from .envs.evaluator import measure,SuccessTracker
from .worker import PolicyProcess

PROMPT='''You are the frozen deployment agent for the drawer exchange task.
Use the current public observation and call history to select from the frozen
skill library, monitor execution, and replan within the fixed invocation budget.
Goal: drawer remains open, red cube released/stable on the outside pad, blue cube
released/stable inside. A subgoal may already hold; decide which skills are needed.
No code, training, simulator edits, extra controllers or direct low-level actions.
Skill tools execute local neural policies; timeouts are failures. Only physical
steps advance time. An already_satisfied return is distinct from choosing never to
invoke a skill. Do not claim final success by assertion: the independent fixed
evaluator decides it after execution. Finish with an honest reason. Same prompt,
tools, observations, applicability descriptions and budgets for both methods.'''


class DeploymentTools:
    def __init__(self,cfg,env,state,policies,directory):
        self.cfg=cfg;self.env=env;self.state=state;self.policies={p['metadata']['policy_id']:p for p in policies}
        self.j=Journal(directory);self.root=Path(directory);self.budget=SimpleNamespace(max_api_calls=12)
        self.tracker=SuccessTracker(cfg['evaluation']);self.steps=0;self.started=time.monotonic();self.invocations=[]
        # Physical episodes have no resumable simulator snapshot. Never replay a
        # partly completed tool against a fresh initial state.
        if self.j.db.execute('SELECT COUNT(*) FROM tools').fetchone()[0]:raise ValueError('Deployment episode already started; new recorded attempt required')

    def schemas(self):
        return [schema('observe','Read current authorized state and prior skill invocation outcomes; no physics advance.',{}),
            schema('invoke_skill','Execute a frozen neural skill locally, at most 600 physical steps, bounded by remaining task time.',
                   {'policy_id':dict(type='string',enum=list(self.policies))}),
            schema('finish_task','End deployment without overriding the independent task-success predicate.',{'reason':dict(type='string')})]

    def done(self):return self.j.get('finished',False)

    def visible(self):
        return dict(state=self.state,history=self.invocations,physical_steps=self.steps,
                    remaining_steps=self.cfg['evaluation']['max_steps']-self.steps,
                    library=[dict(policy_id=k,skill=p['metadata']['skill'],applicable_conditions=p['metadata']['applicable_conditions']) for k,p in self.policies.items()])

    def execute(self,item):
        args=json.loads(item['arguments']);name=item['name'];cid=item['call_id']
        previous=self.j.db.execute('SELECT * FROM tools WHERE id=?',(cid,)).fetchone()
        if previous:
            if previous['status']=='completed':return json.loads(previous['result'])
            raise RuntimeError('Interrupted physical skill is not automatically replayable')
        definition={s['name']:s for s in self.schemas()}[name];jsonschema.validate(args,definition['parameters'])
        self.j.charge('tool_call',18,name=name)
        with self.j.db:self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)',(cid,name,encode(args),'started',None))
        if name=='observe':result=self.visible()
        elif name=='invoke_skill':result=self.invoke(args['policy_id'])
        elif name=='finish_task':
            result=dict(reason=args['reason'],physical_steps=self.steps)
            self.j.set('finished',True);self.j.set('frozen',result)
        else:raise ValueError('Unknown fixed deployment tool')
        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
        return result

    def invoke(self,identifier):
        from PIL import Image
        if len(self.invocations)>=6 or self.steps>=self.cfg['evaluation']['max_steps']:
            return dict(status='budget_exhausted',steps=0,state=self.state)
        selected=self.policies[identifier];role=selected['metadata']['skill']
        # This zero-step check only establishes current observed geometry and
        # release, not final task stability. It never increments success streaks.
        m=measure(self.env,self.state,self.cfg['evaluation'])
        geometric={'open_drawer':m['drawer_open'],'move_red':m['red_on_pad'],'move_blue':m['blue_inside']}[role]
        if geometric and m['gripper_released'] and m['red_clear'] and m['blue_clear'] and m['handoff_clear']:
            result=dict(policy_id=identifier,skill=role,status='already_satisfied',steps=0,state=self.state,
                        note='Observed geometry only; no elapsed stability window credited')
            self.invocations.append(result);return result
        number=len(self.invocations);output=self.root/('skill_'+str(number));begin=self.steps;started=time.monotonic()
        if selected.get('candidate') is None:
            from .train import LoadedPolicy
            policy=LoadedPolicy(selected['checkpoint'],seed=self.j.get('policy_seed',0))
        else:
            policy=PolicyProcess(self.cfg,selected['candidate'],selected['checkpoint'],output/'policy_worker',
                int(os.environ['APPL_PHYSICAL_GPU']),self.j.get('policy_seed',0))
        status='timeout';space=self.env.unwrapped.single_action_space
        try:
            for _ in range(min(600,self.cfg['evaluation']['max_steps']-self.steps)):
                raw=np.asarray(policy.action(self.state),np.float32)
                if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Invalid learned action')
                command=np.clip(raw,space.low,space.high);obs,self.state=step(self.env,command);self.steps+=1
                metrics=self.tracker.update(measure(self.env,self.state,self.cfg['evaluation']))
                event(self.root/'trace.jsonl','step',step=self.steps,state=self.state,action=command.tolist(),raw_action=raw.tolist(),metrics=metrics,skill=role,policy_id=identifier)
                if self.steps%20==1:Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(self.root/f'frame_{self.steps:04d}.png')
                if self.tracker.skill_succeeded(role):status='succeeded';break
            result=dict(policy_id=identifier,skill=role,status=status,steps=self.steps-begin,state=self.state,
                elapsed_seconds=time.monotonic()-started,inference_seconds=sum(policy.timings),inference_chunks=len(policy.timings))
        finally:
            if hasattr(policy,'close'):policy.close()
        self.invocations.append(result);atomic(self.root/'invocations.json',self.invocations)
        # Only the final evaluation file, never the design agent, receives the
        # independent full-task success metric.
        return result


def run(cfg,env,state,policies,directory,seed):
    tools=DeploymentTools(cfg,env,state,policies,directory);tools.j.set('policy_seed',seed)
    AgentLoop(tools.j,tools,client(cfg),PROMPT).run(tools.visible())
    result=dict(success=bool(tools.tracker.last.get('success',False)),final=tools.tracker.last,steps=tools.steps,
        status='succeeded' if tools.tracker.last.get('success',False) else 'agent_stopped_without_task_success',
        invocations=tools.invocations,elapsed_seconds=time.monotonic()-tools.started,
        called_policy_ids=[r['policy_id'] for r in tools.invocations],
        zero_step_invocations=sum(r['status']=='already_satisfied' for r in tools.invocations),completed_evaluation=True)
    atomic(Path(directory)/'result.json',result);return result
