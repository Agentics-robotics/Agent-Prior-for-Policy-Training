"""Inference API selects actual frozen policies by their API-authored priors."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time
import numpy as np
import jsonschema
from PIL import Image
from ..agent import AgentLoop
from ..journal import Journal,encode
from ..io import atomic,read,digest,event,object_hash,archive_source
from ..envs.adapter import reset,step
from .design import schema,client
from .data import catalog,evaluation_output
from .goals import measure
from .runner import PolicyProcess

PROMPT='''You are the inference-time policy-selection agent for the drawer task.
Complete the three supplied geometric task goals using the frozen learned prior
Diffusion Policies. Inspect the current state and the policy catalogue, reason
about which prior and its applicability fit the current situation, read the
selected policy's API-authored PRIOR.md, then invoke that policy for a chosen
number of physical control steps. There is no prescribed sequence of policies.
Each skill has three alternatives based on different priors. Choose among them
using their actual documented mechanisms, applicability and current state.
After each invocation inspect progress and decide whether to continue, change
policy or finish. You may revisit a skill or stop calling an already satisfied
subgoal. Policies execute local learned low-level control. You cannot write code,
train, provide actions, change the environment or redefine success.
For M1_v2, read each selected policy's HANDOFF.json and measured training handoff
support together with PRIOR.md. Assess the actual robot/finger/object state before
transfer; completing a geometric subgoal does not guarantee the successor can
take over. Expanded overlaps train shared transition behavior. Use those documented
entry/exit states and continuation cues to decide when and to whom to transfer.
Measured demonstration ranges describe support, not absolute applicability gates;
contact/grasp is inferred from observations, never guaranteed by a single distance.
There is no hard-coded skill order or framework handoff controller.
The only success conditions are drawer open, red on the marked pad and blue
inside the drawer simultaneously, exactly as specified in the completion contract.
No extra release, clearance, speed or holding-time condition is required. Skill
handoff can still require appropriate manipulation state as explained in the
policy documentation. Waiting or observe calls do not advance physical time.
Invocation duration is a control budget, not a claim that the skill succeeded.
Use smaller invocations near uncertain handoffs when supported by the observation.
Explain your policy choice in each invocation's reason. All text is in English.
The framework ends the episode when the fixed goal predicate holds, the physical
step budget is exhausted, or you explicitly finish. Report failure honestly.
'''


def library(cfg):
    values={}
    for entry in catalog(cfg):
        folder=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
        submission=read(folder/'submission.json');trained=read(folder/'training/result.json')
        for name,sha in submission['files'].items():
            if digest(folder/'source'/name)!=sha:raise ValueError('Submitted policy modified')
        if digest(folder/'training/last.pt')!=trained['checkpoint_sha256']:raise ValueError('Checkpoint modified')
        values[entry['policy_id']]=dict(folder=folder,metadata=read(folder/'source/pipeline.json'),
            document=(folder/'source/PRIOR.md').read_text(),version=submission['version'],
            checkpoint_sha256=trained['checkpoint_sha256'],source_heuristic=entry['heuristic']['statement'])
        if entry.get('experiment_version')=='M1_v2':
            context=read(folder/'training/policy_context.json')
            trained_norm=context['normalization'];assigned_norm=entry['normalization']
            if (Path(trained_norm['path']).resolve()!=Path(assigned_norm['path']).resolve()
                or {k:v for k,v in trained_norm.items() if k!='path'}!={k:v for k,v in assigned_norm.items() if k!='path'}):
                raise ValueError('Policy/shared normalizer mismatch')
            if digest(folder/'training/handoff_evidence.json')!=context['training_handoff_evidence_sha256']:
                raise ValueError('Training handoff evidence changed')
            values[entry['policy_id']].update(handoff=read(folder/'source/HANDOFF.json'),
                handoff_evidence=read(folder/'training/handoff_evidence.json'))
    return values


def freeze(cfg):
    """Freeze all trained packages before any physical evaluation."""
    from .report import authorship
    policies=library(cfg);output=evaluation_output(cfg)
    if output==cfg['output']:
        for p in policies.values():authorship(p['folder'],read(p['folder']/'submission.json'))
    record=dict(experiment_version=cfg.get('experiment_version','M1_v1'),
        configuration_sha256=digest(cfg['_path']),completion_contract_sha256=digest(cfg['completion_contract']),
        dataset_manifest_sha256=digest(cfg['dataset']/'manifest.json'),
        policies={key:dict(version=p['version'],checkpoint_sha256=p['checkpoint_sha256']) for key,p in policies.items()},
        evaluation=cfg['evaluation'],policy_count=len(policies))
    if cfg.get('experiment_version')=='M1_v2':record['normalization_sha256']=digest(cfg['output']/'normalization.json')
    if output!=cfg['output']:
        parent=cfg['output']/'library_freeze.json';original=read(parent)['library']
        for key in ('policies','completion_contract_sha256','dataset_manifest_sha256','normalization_sha256'):
            if original[key]!=record[key]:raise ValueError('Reevaluation changed the frozen policy library or inputs')
        record['parent_library_freeze_sha256']=digest(parent)
    path=output/'library_freeze.json'
    if path.exists():
        if read(path)['library']!=record:raise ValueError('Frozen library changed')
    else:
        if any((output/'evaluation').glob('*/plan.json')):raise ValueError('Cannot freeze after evaluation has started')
        atomic(path,dict(frozen=time.time(),library=record,library_hash=object_hash(record)))
    return read(path)


class DeploymentTools:
    def __init__(self,cfg,env,obs,state,policies,root,seed,goal_measure=measure,goal_names=None):
        self.cfg=cfg;self.env=env;self.obs=obs;self.state=state;self.policies=policies;self.root=Path(root)
        self.seed=seed;self.contract=read(cfg['completion_contract']);self.j=Journal(self.root/'api')
        self.goal_measure=goal_measure;self.goal_names=goal_names
        self.budget=SimpleNamespace(max_api_calls=cfg['evaluation']['max_api_calls'],max_output_tokens=4096)
        self.steps=0;self.calls=[];self.workers={};self.last_policy=None;self.read_docs=set()
        self.started=time.monotonic();self.finished=False;self.saturated=0
        self.feedback=cfg['evaluation'].get('feedback_protocol')=='api_conditions_v1'
        if self.feedback and not cfg['evaluation'].get('hide_step_budget'):
            raise ValueError('This feedback protocol requires a private episode budget')

    def schemas(self):
        identifier=dict(type='string',enum=list(self.policies))
        definitions=[schema('observe','Read current state, task goal predicates and invocation history. No physical steps.',{}),
            schema('read_policy','Read the selected policy prior, actual learned mechanism and applicability documentation.',dict(policy_id=identifier)),
            schema('invoke_policy','Execute the chosen frozen learned policy for up to the requested number of control steps; task success ends early.',
                dict(policy_id=identifier,steps=dict(type='integer',minimum=1,maximum=self.cfg['evaluation']['max_invocation_steps']),reason=dict(type='string',minLength=1))),
            schema('finish','End this trial with an honest reason; cannot override measured success.',dict(reason=dict(type='string',minLength=1)))]
        if self.feedback:
            from .feedback import stop_schema
            definitions[0]['description']='Read current state, measured metrics and latest invocation. No physical steps.'
            invocation=definitions[2]
            invocation['description']='Execute a frozen policy until the requested duration, one of your per-step stop conditions, or task success; then return control for your next decision.'
            invocation['parameters']['properties'].update(stop_when=stop_schema(self.goal_names),
                notebook=dict(type='string',minLength=1,maxLength=3000))
            invocation['parameters']['required']+=['stop_when','notebook']
            definitions.append(schema('read_invocation','Read a previous invocation from this episode, including full boundary states and your original reason.',
                dict(invocation_id=dict(type='integer',minimum=1))))
        return definitions

    def goals(self,state):return self.goal_measure(state,self.contract)

    def done(self):return self.finished or self.goals(self.state)['success'] or self.steps>=self.cfg['evaluation']['max_steps']

    def visible(self):
        if self.feedback:
            from .feedback import measurements
            value=dict(state=self.state,task_goals=self.goals(self.state),physical_steps=self.steps,
                observed_metrics=measurements(self.state,self.goals(self.state)),last_invocation=None)
            if self.calls:
                keys=('invocation_id','policy_id','requested_steps','executed_steps','stop_reason',
                      'matched_stop_rules','metric_ranges','notebook')
                value['last_invocation']={k:self.calls[-1][k] for k in keys}
            return value
        return dict(state=self.state,task_goals=self.goals(self.state),physical_steps=self.steps,
            remaining_steps=self.cfg['evaluation']['max_steps']-self.steps,invocations=self.calls)

    def context_input(self,history):
        from .feedback import project_context
        return project_context(self.j,history,self.cfg['evaluation']['context_recent_turns'])

    def execute(self,item):
        name=item['name'];cid=item['call_id'];args=json.loads(item['arguments'])
        definition={s['name']:s for s in self.schemas()}[name];jsonschema.validate(args,definition['parameters'])
        previous=self.j.db.execute('SELECT * FROM tools WHERE id=?',(cid,)).fetchone()
        if previous:
            if previous['name']!=name or previous['args']!=item['arguments']:raise ValueError('Tool ID collision')
            if previous['status']!='completed':raise RuntimeError('An interrupted physical action cannot be replayed')
            return json.loads(previous['result'])
        self.j.charge('tool_call',self.budget.max_api_calls,name=name,call_id=cid)
        with self.j.db:self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)',(cid,name,item['arguments'],'started',None))
        if name=='observe':result=self.visible()
        elif name=='read_policy':
            self.read_docs.add(args['policy_id']);p=self.policies[args['policy_id']]
            result=dict(metadata=p['metadata'],prior_document=p['document'],source_heuristic=p['source_heuristic'])
            if 'handoff' in p:
                evidence=p['handoff_evidence']
                result.update(handoff=p['handoff'],measured_support=dict(fields=evidence['fields'],
                    boundary_statistics=evidence['boundary_statistics'],interpretation=evidence['interpretation']))
        elif name=='invoke_policy':
            if args['policy_id'] not in self.read_docs:result=dict(error='Read this policy document before invoking it')
            else:result=self.invoke(**args)
        elif name=='read_invocation':
            index=args['invocation_id']-1
            result=self.calls[index] if index<len(self.calls) else dict(error='No such completed invocation')
        else:
            self.finished=True;result=dict(reason=args['reason'],**self.visible())
        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
        return result

    def invoke(self,policy_id,steps,reason,stop_when=None,notebook=None):
        p=self.policies[policy_id]
        if policy_id not in self.workers:
            self.workers[policy_id]=PolicyProcess(self.cfg,p['folder'],self.root/'workers'/policy_id,self.seed)
        worker=self.workers[policy_id];begin=self.steps;before=self.state
        limit=min(steps,self.cfg['evaluation']['max_steps']-self.steps)
        space=self.env.unwrapped.single_action_space
        if self.feedback:
            from .feedback import measurements,matches,update_ranges
            initial=measurements(before,self.goals(before));ranges={};matched=[]
            stop_reason='requested_duration'
            update_ranges(ranges,initial,begin)
        for i in range(limit):
            raw=np.asarray(worker.action(self.state,reset=i==0 and self.last_policy!=policy_id),np.float32)
            if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Invalid policy action')
            action=np.clip(raw,space.low,space.high);self.saturated+=int(np.any(raw!=action))
            self.obs,self.state=step(self.env,action);self.steps+=1
            metrics=self.goals(self.state)
            event(self.root/'trace.jsonl','step',step=self.steps,policy_id=policy_id,state=self.state,
                raw_action=raw.tolist(),action=action.tolist(),metrics=metrics)
            if self.steps%20==1:Image.fromarray(self.obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(self.root/f'frame_{self.steps:04d}.png')
            if self.feedback:
                current=measurements(self.state,metrics);update_ranges(ranges,current,self.steps)
                matched=matches(stop_when,current,initial)
                if metrics['success']:stop_reason='task_success';break
                if matched:stop_reason='api_condition';break
            elif metrics['success']:break
        self.last_policy=policy_id
        call=dict(policy_id=policy_id,reason=reason,requested_steps=steps,executed_steps=self.steps-begin,
            before=before,after=self.state,task_goals=self.goals(self.state),
            source_version=p['version'],checkpoint_sha256=p['checkpoint_sha256'])
        if 'handoff' in p:call['api_handoff_sha256']=digest(p['folder']/'source/HANDOFF.json')
        if self.feedback:
            if self.steps>=self.cfg['evaluation']['max_steps'] and stop_reason=='requested_duration':stop_reason='executor_limit'
            call.update(invocation_id=len(self.calls)+1,stop_when=stop_when,notebook=notebook,
                stop_reason=stop_reason,matched_stop_rules=matched,metric_ranges=ranges)
        self.calls.append(call);atomic(self.root/'invocations.json',self.calls)
        return self.visible()


def make_evaluation_environment(max_steps):
    """Override the registered TimeLimit through Gym, without changing M0's factory."""
    import gymnasium as gym
    from ..envs import native  # Register the existing, unchanged task.
    return gym.make('APPLDrawerExchange-v2',max_episode_steps=max_steps,num_envs=1,
        obs_mode='rgb+state_dict',control_mode='pd_joint_pos',sim_backend='physx_cpu',
        render_backend='cuda:0',reward_mode='none')


def evaluate(cfg,seed):
    from ..gpu import verify_cuda
    if seed not in cfg['evaluation']['seeds']+cfg['evaluation']['diagnostic_seeds']:raise ValueError('Unplanned evaluation seed')
    output=evaluation_output(cfg);root=output/'evaluation'/str(seed)
    if (root/'result.json').exists():return read(root/'result.json')
    if root.exists():raise ValueError('Interrupted physical trial retained; explicit new attempt is required')
    policies=library(cfg)
    if cfg.get('experiment_version')=='M1_v2':
        frozen=read(output/'library_freeze.json')['library']
        current={key:dict(version=p['version'],checkpoint_sha256=p['checkpoint_sha256']) for key,p in policies.items()}
        if frozen['policies']!=current or frozen['configuration_sha256']!=digest(cfg['_path']) or frozen['normalization_sha256']!=digest(cfg['output']/'normalization.json') or frozen['completion_contract_sha256']!=digest(cfg['completion_contract']) or frozen['dataset_manifest_sha256']!=digest(cfg['dataset']/'manifest.json'):
            raise ValueError('Evaluation must use the completely frozen library/configuration')
    root.mkdir(parents=True)
    atomic(root/'plan.json',dict(seed=seed,scope='training_reset_diagnostic' if seed in cfg['evaluation']['diagnostic_seeds'] else ('same_seed_ID_budget_retest' if output!=cfg['output'] else 'independent_ID'),
        source=archive_source(output),device=verify_cuda(),completion_contract=read(cfg['completion_contract']),
        library={k:dict(version=v['version'],checkpoint_sha256=v['checkpoint_sha256']) for k,v in policies.items()},
        max_steps=cfg['evaluation']['max_steps']))
    env=make_evaluation_environment(cfg['evaluation']['max_steps']);tools=None
    try:
        wrappers=[];wrapper=env
        while hasattr(wrapper,'env'):
            if '_max_episode_steps' in vars(wrapper):
                wrappers.append(dict(wrapper=type(wrapper).__name__,max_episode_steps=wrapper._max_episode_steps))
            wrapper=wrapper.env
        if not wrappers or any(w['max_episode_steps']!=cfg['evaluation']['max_steps'] for w in wrappers):
            raise ValueError('Simulator time limit differs from the declared evaluation budget')
        atomic(root/'environment.json',dict(time_limits=wrappers,max_steps=cfg['evaluation']['max_steps']))
        obs,state=reset(env,seed);atomic(root/'initial_state.json',state)
        tools=DeploymentTools(cfg,env,obs,state,policies,root,seed)
        message=dict(completion_contract=tools.contract,observation=tools.visible(),
            policy_catalogue=[dict(**v['metadata'],source_heuristic=v['source_heuristic']) for v in policies.values()])
        prompt=PROMPT
        if tools.feedback:
            from .feedback import PROMPT as feedback_prompt
            prompt=feedback_prompt
        AgentLoop(tools.j,tools,client(cfg),prompt,
            context_builder=tools.context_input if tools.feedback else None).run(message)
        metrics=measure(tools.state,tools.contract)
        Image.fromarray(tools.obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'final.png')
        result=dict(success=metrics['success'],seed=seed,steps=tools.steps,final=metrics,
            status='succeeded' if metrics['success'] else ('physical_budget_exhausted' if tools.steps>=cfg['evaluation']['max_steps'] else 'agent_finished'),
            invocations=tools.calls,elapsed_seconds=time.monotonic()-tools.started,
            actuator_saturated_steps=tools.saturated,completed_evaluation=True,api_selected_all_policies=True)
        atomic(root/'result.json',result);print(json.dumps(dict(seed=seed,success=result['success'],steps=tools.steps)),flush=True)
        return result
    except Exception as error:
        atomic(root/'failure.json',dict(error=str(error),physical_steps=0 if tools is None else tools.steps,automatic_retry=False))
        raise
    finally:
        os.environ.pop('APPL_PRIOR_TOKEN',None)
        if tools is not None:
            for worker in tools.workers.values():worker.close()
            tools.j.db.close()
        env.close()
