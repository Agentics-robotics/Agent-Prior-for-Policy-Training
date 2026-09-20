"""Actual Responses design tools; only candidate workspace writes are exposed."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import shutil
import time
import base64
import jsonschema

from .agent import AgentLoop,ResponsesClient
from .journal import Journal,safe,encode,sha
from .io import ROOT,read,atomic,digest,event
from . import data
from .security import audit


def client(cfg):
    options=cfg['api'];credential=Path(options['credential_path'])
    if credential.stat().st_mode & 0o077:raise ValueError('Private provider credential must have mode 0600')
    os.environ['APPL_RUNTIME_TOKEN']=read(credential)['LOCAL_OPENAI_BEARER_TOKEN']
    return ResponsesClient(base_url=options['base_url'],model=options['model'],key_env='APPL_RUNTIME_TOKEN',
        headers={'x-openai-actor-authorization':'local-image-extension'},reasoning_effort=options['reasoning_effort'],
        timeout_seconds=options['timeout_seconds'],allow_network=True)


def schema(name,description,properties):
    return dict(type='function',name=name,description=description,strict=True,
        parameters=dict(type='object',properties=properties,required=list(properties),additionalProperties=False))


INTERFACE = '''# APPL neural candidate contract
You write policy.py, candidate.json, and optionally your own helper modules.
Allowed imports: torch, numpy, math, appl.public, and your own modules. Kernel
Landlock/seccomp prohibit network, shell, arbitrary files and other sessions.
No credentials enter your worker. Your model stays loaded for local low-level
inference; the remote agent is not called on each action.

policy.py must implement:
build_model(factory, spec) -> torch.nn.Module. Call factory(condition_dimension)
exactly once and retain its shared DP backbone in the returned model. Additional
encoders may use at most 10% extra parameters; large architecture exploration is
outside this matched-backbone pilot. No replacement scripted controller or replay.
condition(raw_history, spec) -> [B,condition_dimension], finite float32.
encode_action(native_actions, spec) -> [B,H,8].
decode_action(encoded_actions, spec) -> [B,H,8] in native units; exact inverse of
encode_action. Merely changing observation coordinates does not make absolute
joint actions equivariant. Explain action-side implications of your prior.
compute_loss(model, batch, spec) -> {'loss': differentiable finite scalar}.
batch keys: raw_obs [B,2,47], native_action [B,16,8], mask [B,16,1],
encoded_action, noisy_action, noise, timesteps [B]. The framework owns fixed DDPM
epsilon noising, DDIM inference, optimizer, device, mask, data and final success.
You can change representation, input encoding, loss/auxiliary constraints and
small structure while retaining the fixed diffusion backbone. Priors are open
code choices, not a menu. Explain why a proposed prior applies or does not apply.
spec contains training, normalizer, fields, observation_dimension, skill and
candidate_config. See baseline.py and public.py for executable numerical APIs.
Training reads only segments you name. It rejects absent/zero gradients, unchanged
weights and save/reload mismatch. Optimizer/RNG/EMA progress is checkpointed.

candidate.json must have policy_id, skill (open_drawer / move_red / move_blue),
prior (explanation), applicable_conditions, action_transform_explanation,
segments ({each training trajectory ID: [start,stop]}), config (object).
The three skill names refer to task subgoals, not prescribed cuts or priors.
Choose cuts yourself from complete trajectories; do not count segments as extra
demonstrations. Include release/settling/clearance needed for the next skill.
No additional deployment requirements were given.

State layout (world metres; quaternions wxyz; arm radians): qpos[0:9], qvel[9:18],
tcp_pose[18:25], red_pose[25:32], blue_pose[32:39], drawer_position[39],
drawer_velocity[40], red_goal[41:44], blue_goal[44:47]. Two causal observations.
The gripper action [-1,1] maps to per-finger position [-.01,.04] metres. Positive
opens. Seven other actions are absolute joint radians. No external IK/planner is
available to a learned candidate. Same actuator saturation applies to M0/M1/M2.

Success is owned by the environment evaluator: drawer >.26 m; whole red cube
inside square pad halfwidth .06 m; blue cube within drawer interior; released
fingers >=.035 m, TCP >=.08 m from objects; linear speed <.02 m/s and angular
speed <.2 rad/s continuously for 10 physical steps at 20 Hz. Timeouts are not
success. Waiting for this API does not advance physics.
'''


class DesignTools:
    def __init__(self,cfg,stage='capability'):
        self.cfg=cfg;self.stage=stage
        self.root=cfg['output']/('api_'+stage);self.j=Journal(self.root)
        self.budget=SimpleNamespace(max_api_calls=50 if stage=='capability' else cfg['api']['max_requests'])
        self.public=self.root/'public';self.public.mkdir(exist_ok=True)
        if not (self.public/'INTERFACE.md').exists():
            (self.public/'INTERFACE.md').write_text(INTERFACE)
            for name in ('baseline.py','public.py'):
                shutil.copy2(ROOT/'src/appl'/name,self.public/name)
            atomic(self.public/'data.json',data.audit(cfg))
        self.max_updates=5000 if stage=='capability' else cfg['training']['updates']

    def schemas(self):
        string=dict(type='string');integer=dict(type='integer')
        return [
            schema('read_public','Read the explicit public interface, baseline, backbone or data contract.',{'name':dict(type='string',enum=['INTERFACE.md','baseline.py','public.py','data.json'])}),
            schema('read_trajectory','Read bounded indexed original training observations and aligned actions.',{'trajectory_id':string,'indices':dict(type='array',items=integer,minItems=1,maxItems=24)}),
            schema('read_training_image','Read one synchronized original training RGB frame by trajectory ID and index.',{'trajectory_id':string,'index':integer}),
            schema('write_file','Create/edit only this unsubmitted candidate. Return package hash; source version retained.',{'candidate_id':string,'path':string,'content':string}),
            schema('read_file','Read a file in your current candidate workspace.',{'candidate_id':string,'path':string}),
            schema('check_candidate','Check code/interface plus TWO actual GPU optimizer steps and save/reload; recorded separately from full candidate training.',{'candidate_id':string}),
            schema('submit_candidate','Explicitly freeze code, config, prior and segments at the expected content hash.',{'candidate_id':string,'expected_hash':string}),
            schema('train_candidate','Run the fixed GPU training entry on this submitted candidate; returns actual training evidence.',{'version':string,'updates':dict(type='integer',minimum=2,maximum=self.max_updates)}),
            schema('status','Read current submissions, training receipts and remaining budgets.',{}),
            schema('finish_capability','Submit a actually GPU-trained candidate to the framework capability evaluation; not a formal scientific result.',{'version':string,'design_note':string}),
        ]

    def done(self):return bool(self.j.get('finished'))

    def candidate(self,identifier):
        if not re.fullmatch('[a-z][a-z0-9_]{0,47}',identifier):raise ValueError('Invalid candidate identifier')
        return safe(self.root/'work',identifier)

    def files(self,identifier):
        return {p.name:digest(p) for p in self.candidate(identifier).glob('*') if p.is_file()}

    def metadata(self,directory):
        value=read(directory/'candidate.json')
        required={'policy_id','skill','prior','applicable_conditions','action_transform_explanation','segments','config'}
        if set(value)!=required:raise ValueError('candidate.json keys must exactly match '+str(sorted(required)))
        if not isinstance(value['policy_id'],str) or not re.fullmatch('[a-z][a-z0-9_]{0,47}',value['policy_id']):raise ValueError('Invalid policy_id')
        for name in ('prior','applicable_conditions','action_transform_explanation'):
            if not isinstance(value[name],str) or not value[name].strip():raise ValueError('Candidate '+name+' must be nonempty text')
        if value['skill'] not in ('open_drawer','move_red','move_blue'):raise ValueError('Unknown fixed task subgoal')
        if set(value['segments'])!=set(self.cfg['data']['train_ids']):raise ValueError('One segment per original training trajectory required')
        for identifier,(start,stop) in value['segments'].items():
            count=len(read(self.cfg['data']['directory']/(identifier+'.json'))['actions'])
            if type(start) is not int or type(stop) is not int or not 0<=start<stop<=count:raise ValueError('Invalid source segment')
        if not isinstance(value['config'],dict):raise ValueError('Candidate config must be an object')
        return value

    def snapshot(self,identifier):
        directory=self.candidate(identifier);self.metadata(directory);audit(directory)
        files=self.files(identifier)
        if 'policy.py' not in files:raise ValueError('Missing policy.py')
        version=sha(files);out=self.root/'versions'/version
        if not out.exists():shutil.copytree(directory,out)
        for name,expected in files.items():
            if digest(out/name)!=expected:raise ValueError('Immutable version changed')
        atomic(out/'manifest.json',files)
        return version,out

    def job(self,directory,updates,kind):
        from .worker import launch
        number=self.j.count(kind)
        self.j.charge(kind,self.cfg['api']['max_checks'] if kind=='interface_check' else self.cfg['api']['max_training_attempts'],updates=updates,version=directory.name)
        output=self.root/'jobs'/f'{kind}_{number:03d}'
        request=dict(operation='train',candidate=str(directory),updates=updates,seed=0)
        process=launch(self.cfg,request,output,self.cfg['devices']['candidate'])
        started=time.monotonic();code=process.wait()
        receipt=dict(kind=kind,version=directory.name,updates=updates,returncode=code,elapsed_seconds=time.monotonic()-started,output=str(output))
        self.j.event('job_cost',**receipt)
        if code:
            receipt['error']=(output/'stderr.log').read_text()[-4500:]
            atomic(output/'failure.json',receipt)
            raise RuntimeError(encode(receipt))
        result=read(output/'result.json')
        receipt.update({k:result[k] for k in ('optimizer_steps','trainable_parameters','weight_l2_change','reload_max_action_error','checkpoint_sha256','device','neural_training_verified')})
        if kind=='training_attempt':
            trained=self.j.get('trained',{});trained[directory.name]=dict(receipt)
            self.j.set('trained',trained)
        return receipt

    def execute(self,item):
        name=item['name'];args=json.loads(item['arguments']);cid=item['call_id']
        previous=self.j.db.execute('SELECT * FROM tools WHERE id=?',(cid,)).fetchone()
        if previous:
            if previous['status']=='completed':return json.loads(previous['result'])
            raise RuntimeError('Interrupted tool requires explicit job receipt reconciliation; no automatic replay')
        definitions={s['name']:s for s in self.schemas()}
        if name not in definitions:raise ValueError('Unknown tool')
        jsonschema.validate(args,definitions[name]['parameters'])
        self.j.charge('tool_call',self.cfg['api']['max_tools'],call_id=cid,name=name)
        with self.j.db:self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)',(cid,name,encode(args),'started',None))
        try:result=self.dispatch(name,args)
        except (ValueError,RuntimeError,KeyError,TypeError) as error:
            result=dict(error=str(error),repair_allowed=not self.done())
            self.j.event('tool_diagnostic',call_id=cid,name=name,error=str(error))
        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
        return result

    def dispatch(self,name,args):
        if name=='read_public':return dict(name=args['name'],content=(self.public/args['name']).read_text())
        if name=='read_training_image':
            identifier=args['trajectory_id'];index=args['index']
            if identifier not in self.cfg['data']['train_ids']:raise ValueError('Only training trajectories are readable')
            trajectory=read(self.cfg['data']['directory']/(identifier+'.json'))
            if not 0<=index<len(trajectory['observations']):raise ValueError('Frame index outside trajectory')
            path=self.cfg['data']['directory']/trajectory['observations'][index]['images']['front']
            return [dict(type='input_text',text=encode(dict(trajectory_id=identifier,index=index))),
                dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),detail='high')]
        if name=='read_trajectory':
            identifier=args['trajectory_id']
            if identifier not in self.cfg['data']['train_ids']:raise ValueError('Only original training IDs are readable')
            t=read(self.cfg['data']['directory']/(identifier+'.json'))
            rows=[]
            for index in args['indices']:
                if not 0<=index<len(t['observations']):raise ValueError('Observation index out of range')
                rows.append(dict(index=index,state=t['observations'][index]['state'],action=t['actions'][index] if index<len(t['actions']) else None))
            return dict(id=identifier,action_count=len(t['actions']),rows=rows)
        if name in ('write_file','read_file'):
            identifier=args['candidate_id'];directory=self.candidate(identifier)
            if '/' in args['path'] or args['path'] in ('manifest.json','') or Path(args['path']).suffix not in ('.py','.json','.md'):
                raise ValueError('Use explicit flat candidate source/config files')
            if name=='read_file':return dict(content=safe(directory,args['path']).read_text())
            if identifier in self.j.get('submitted',{}):raise ValueError('Submitted candidate is immutable; choose a new revision identifier')
            self.j.write_candidate(identifier+'/'+args['path'],args['content'])
            return dict(candidate_id=identifier,files=self.files(identifier),package_hash=sha(self.files(identifier)))
        if name=='check_candidate':
            version,directory=self.snapshot(args['candidate_id'])
            result=self.job(directory,2,'interface_check')
            checked=self.j.get('checked',{});checked[version]=result;self.j.set('checked',checked)
            return result
        if name=='submit_candidate':
            version,directory=self.snapshot(args['candidate_id'])
            if version!=args['expected_hash']:raise ValueError('Expected source/config hash mismatch')
            if version not in self.j.get('checked',{}):raise ValueError('This exact version has no successful GPU interface check')
            submitted=self.j.get('submitted',{});submitted[args['candidate_id']]=version;self.j.set('submitted',submitted)
            self.j.event('submitted',version=version,candidate_id=args['candidate_id'])
            return dict(version=version,immutable=True)
        if name=='train_candidate':
            if args['version'] not in self.j.get('submitted',{}).values():raise ValueError('Only submitted versions may train')
            directory=self.root/'versions'/args['version']
            return self.job(directory,args['updates'],'training_attempt')
        if name=='status':
            return dict(submitted=self.j.get('submitted',{}),trained=self.j.get('trained',{}),
                requests_remaining=self.budget.max_api_calls-self.j.db.execute('SELECT COUNT(*) FROM api').fetchone()[0],
                tools_remaining=self.cfg['api']['max_tools']-self.j.count('tool_call'))
        if name=='finish_capability':
            if self.stage!='capability':raise ValueError('Not a capability-only stage')
            trained=self.j.get('trained',{}).get(args['version'])
            if not trained or trained['optimizer_steps']<self.max_updates:raise ValueError('Complete the declared neural training budget first')
            value=dict(version=args['version'],training=trained,design_note=args['design_note'],stage='framework_capability',formal_result=False)
            self.j.set('finished',value);self.j.set('frozen',value);atomic(self.root/'submission.json',value)
            return value
        raise ValueError('Unavailable tool')


def capability(cfg):
    completed=cfg['output']/'api_capability/submission.json'
    if completed.exists():
        value=read(completed)
        if digest(Path(value['training']['output'])/'last.pt')!=value['training']['checkpoint_sha256']:raise ValueError('Completed API capability checkpoint changed')
        return value
    tools=DesignTools(cfg,'capability')
    prompt='''You are the sole APPL policy/prior designer. This is a DEVELOPMENT
capability run before the formal experiment, not a formal result. Read INTERFACE.md,
baseline.py and public.py, inspect actual training states/actions and choose ONE
useful neural subpolicy of the drawer task. Decide its cuts and a justified prior
yourself. Write actual Torch mechanism code, run the GPU interface check, repair
as needed, explicitly submit the expected hash, train for 5000 real updates, then
finish_capability with an honest design note. Do not produce a replay/nearest-
neighbor controller, do not return suggestions instead of tool calls. No candidate
performance has been measured yet. No prescribed prior, cuts, extra data or hidden
test access. Keep the common diffusion backbone and explain action-space validity.
The later main comparison will share your cuts/interfaces/backbone with a no-prior
hierarchical baseline. There are no additional deployment requirements.'''
    result=AgentLoop(tools.j,tools,client(cfg),prompt).run(dict(task='Implement and actually train a neural APPL subpolicy',public_files=['INTERFACE.md','baseline.py','public.py','data.json']))
    return result
