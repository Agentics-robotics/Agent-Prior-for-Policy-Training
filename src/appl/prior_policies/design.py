"""Runtime API writes each policy package; repository code only supplies tools."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import time
import jsonschema
from ..agent import AgentLoop,ResponsesClient
from ..journal import Journal,encode,safe
from ..io import ROOT,read,atomic,digest,object_hash
from ..security import audit

PROMPT='''You are the scientific implementation agent for one prior-based Diffusion Policy.
Read your assigned skill and heuristic, then implement that heuristic as a real
learned diffusion policy. Your source code, pipeline metadata and prior document
are your scientific deliverables. The repository supplies data, fixed DDPM
noising/sampling, optimizer, safe execution and accounting, not your prior design.
Keep the assigned segmentation and heuristic identity. Choose the representation,
learned modules, conditioning structure and differentiable losses appropriate to
that heuristic. Every policy must remain a learned action diffusion model.
Read INTERFACE.md and the numerical capabilities before coding. Read actual
training observations to resolve details. Future observations are training labels
only; deployment forward receives only causal observation history. Make auxiliary
losses depend on trainable predictions: a penalty computed only from observed
poses is constant with respect to the model and cannot train it. Explicitly
document any adaptation needed to turn the research hypothesis into an executable
diffusion model with the provided data. Do not claim unimplemented IK, image
encoders, invariance or constraints. No scripted action controller or replay.
This is M1_v2. Read the assigned heuristic's handoff interface and the measured
training boundary/overlap evidence. Implement a coherent policy over its FULL
expanded slice, including the shared transition phases. Explain the observation
information that lets it take over partway through a transition and reach a
useful exit state. All models share one normalization fit on the complete original
training demonstrations. Do not refit tiny per-skill scales in your model.
Write policy.py, pipeline.json, PRIOR.md and HANDOFF.json with write_file, then check_policy.
You may repair failed interface checks yourself. After a successful check, submit
the exact checked package hash. Submitted code is immutable. The outer framework
will train the final package at the declared budget; no performance feedback is
available during this design session. All source, documentation and tool text
must be in English. Never modify the source heuristic; document your implementation
choices and limitations separately in your own PRIOR.md.
'''

INTERFACE='''# Prior Diffusion Policy interface

You own policy.py, pipeline.json, PRIOR.md, HANDOFF.json and optional flat helper .py files.
Allowed imports: torch, numpy, math, appl.public, and your own flat modules.
No file/network/shell access; generated code runs after Landlock/seccomp.

policy.py hooks:
- build_model(spec) -> torch.nn.Module, at most 64 million trainable parameters.
- model.forward(noisy_action [B,H,8], timestep scalar or [B], raw_history [B,2,47])
  -> predicted epsilon [B,H,8]. Both training and inference use this forward.
- compute_loss(model, batch, spec) -> dict with scalar tensors loss,
  diffusion_loss, prior_loss. Total and diffusion loss must be differentiable.
  Pure architectural priors may report zero prior_loss and explain this.

Public numerical helpers in appl.public:
DiffusionBackbone(condition_dimension, spec['training']) constructs a standard
Conditional U-Net with forward(sample, timestep, condition). It is available as
a building block, not compulsory. normalize_observation(raw,spec),
normalize_action(native,spec), denormalize_action(encoded,spec),
epsilon_loss(predicted_noise, noise, mask) are available. Wrap the backbone and
your learned encoder in the model if you use it; forward receives raw history.
There is no factory argument in build_model. You may implement other Torch
diffusion architectures. Implement your prior in trainable representation,
structure or loss, rather than an external nonlearned control policy.

Fixed training recipe: seed 0, 20,000 updates, batch 128, horizon 16, history 2,
execution 8, DDPM 100 train and inference steps, epsilon prediction, clip_sample
true, AdamW 1e-4/1e-6, EMA .999, cosine learning rate with 500 warmup steps.
Default U-Net widths 128/256/512, timestep embed 128, kernel 5, groups 8.
Interface checks perform 2 updates in a fresh worker, then sample and reload EMA.
Their cost is separate from final training. No architecture parameter equality
constraint across priors; actual parameter counts are recorded.

spec: training, normalizer, fields (named [start,stop] indices),
observation_dimension=47, skill_id, candidate_config=your pipeline.json config.
Input state (world metres, quaternion wxyz): qpos 0:9, qvel 9:18,
tcp_pose 18:25, red_pose 25:32, blue_pose 32:39, drawer_position 39:40,
drawer_velocity 40:41, red_goal 41:44, blue_goal 44:47.
Two causal observations are available. No images enter the learned model in this
state-based experiment. M1_v2 observations AND actions use a single shared range
normalizer fitted once on all COMPLETE original training trajectories. Expanded
slices do not refit or reweight it. The assignment includes the exact normalizer.
Unit quaternion components have fixed unit bounds; constant features have unit
scale. In limits mode 'std' means HALF RANGE, not standard deviation. There is
no observation clipping. Use the shared scales consistently if normalizing learned
relative features; avoid dividing by narrow per-skill empirical ranges.

Actions: seven absolute Panda joint targets (radians) and one gripper command
in [-1,1], increasing toward open; per-finger target = .025*command+.015 metres.
Diffusion predicts affinely normalized native joint/gripper actions; the fixed
sampler decodes them. A representation prior does not itself make joint actions
equivariant. No external IK or forward-kinematics provider is available. A learned
predictor can use training future states if your prior requires action/state coupling.

batch: raw_obs [B,2,47], native_action [B,16,8], encoded_action [B,16,8],
noisy_action [B,16,8], noise [B,16,8], timesteps [B], mask [B,16,1],
future_obs [B,16,47], future_mask [B,16,1], alpha_bar [B].
At current observation t, action slots correspond to t-1 through t+14;
future_obs slot j is the state AFTER that action. Padding is masked and never
crosses a skill slice. Use masks for all future targets. DDPM clean estimate:
x0=(noisy_action-sqrt(1-alpha_bar)*predicted_epsilon)/sqrt(alpha_bar), with
alpha_bar reshaped [B,1,1]. Be mindful of high-noise amplification in auxiliaries.

pipeline.json exact keys: policy_id, skill_id, heuristic_index, prior_summary,
applicable_conditions, termination_guidance, limitations, config (JSON object).
The first three identify the assigned policy. Other text fields are nonempty
English strings. PRIOR.md describes the original heuristic, implemented prior,
architecture, actual loss terms and gradient paths, causal deployment inputs,
applicability and termination cues, dependencies and limitations. The inference
API will read this document to select a policy. It chooses policy and invocation
duration; your model executes local low-level actions. No preset skill schedule.

HANDOFF.json exact keys: policy_id, entry_conditions, exit_conditions, overlap_role,
successor_readiness, failure_signatures, continuation_guidance, limitations, evidence.
policy_id must match the assignment. All fields except evidence are nonempty
English strings. evidence is a nonempty list of {trajectory_id, indices}, citing
observations within the assigned slices. Explain causal observation cues (fields,
frames, units), what transition motion this model actually learns, when to continue
it versus transfer control, and what state the successor should receive. Include
finger opening, TCP/object relation and motion where supported by the data. A
single geometric subgoal may hold before manipulation is ready for transfer.
Separate observed support, inferred grasp/contact and untested recovery claims.
Use PRIOR.md to explain how the implemented prior relates to these interfaces.
The framework also saves measured training start/end/overlap states separately;
these measurements are evidence, not a controller or hard rejection threshold.
Your HANDOFF.json remains API-authored and is read by the inference agent.

Task success uses only the supplied completion contract. A task can
complete without an additional gripper/clearance/speed/stability hold condition.
Skill termination cues guide the inference API and never redefine task success.
'''


def schema(name,description,properties):
    return dict(type='function',name=name,description=description,strict=True,
        parameters=dict(type='object',properties=properties,required=list(properties),additionalProperties=False))


def client(cfg):
    p=cfg['api'];credential=Path(p['credential_path'])
    if credential.stat().st_mode&0o077:raise ValueError('Credential permissions must be private')
    os.environ['APPL_PRIOR_TOKEN']=read(credential)['LOCAL_OPENAI_BEARER_TOKEN']
    return ResponsesClient(base_url=p['base_url'],model=p['model'],key_env='APPL_PRIOR_TOKEN',
        headers={'x-openai-actor-authorization':'local-image-extension'},reasoning_effort=p['reasoning_effort'],
        timeout_seconds=p['timeout_seconds'],allow_network=True)


class DesignTools:
    def __init__(self,cfg,folder,gpu):
        self.cfg=cfg;self.folder=Path(folder);self.gpu=gpu
        self.entry=read(self.folder/'assignment.json');self.j=Journal(self.folder/'design')
        self.work=self.folder/'design/work';self.work.mkdir(parents=True,exist_ok=True)
        self.budget=SimpleNamespace(**cfg['design'])

    def schemas(self):
        text=dict(type='string',minLength=1)
        return [schema('read_assignment','Read the assigned original API heuristic, skill, segment ranges and completion contract.',{}),
            schema('read_public','Read the English implementation interface or numerical source.',dict(name=dict(type='string',enum=['INTERFACE.md','public.py','backbone.py']))),
            schema('read_steps','Read original states/actions at source indices within your assigned skill.',dict(trajectory_id=text,indices=dict(type='array',items=dict(type='integer',minimum=0),minItems=1,maxItems=24))),
            schema('write_file','Write your unsubmitted policy source, metadata or prior document; records exact bytes.',dict(path=text,content=text)),
            schema('read_file','Read your current policy file.',dict(path=text)),
            schema('check_policy','Validate package and run 2 real GPU training updates plus DDPM/EMA reload checks.',{}),
            schema('submit_policy','Freeze the exact successfully checked package hash.',dict(expected_hash=text))]

    def done(self):return bool(self.j.get('submitted'))

    def hashes(self):return {p.name:digest(p) for p in self.work.iterdir() if p.is_file()}

    def validate(self):
        required={'policy.py','pipeline.json','PRIOR.md'}
        if self.entry.get('experiment_version')=='M1_v2':required.add('HANDOFF.json')
        if not required<=set(self.hashes()):raise ValueError('Write all required files: '+str(sorted(required)))
        if not self.j.get('assignment_read'):raise ValueError('Read the source heuristic first')
        m=read(self.work/'pipeline.json')
        keys={'policy_id','skill_id','heuristic_index','prior_summary','applicable_conditions','termination_guidance','limitations','config'}
        if set(m)!=keys:raise ValueError('pipeline.json keys must exactly match '+str(sorted(keys)))
        for key in ('policy_id','skill_id','heuristic_index'):
            if m[key]!=self.entry[key]:raise ValueError('Assigned identity mismatch: '+key)
        for key in ('prior_summary','applicable_conditions','termination_guidance','limitations'):
            if not isinstance(m[key],str) or not m[key].strip():raise ValueError('Missing explanatory text: '+key)
        if not isinstance(m['config'],dict):raise ValueError('config must be a JSON object')
        if self.entry.get('experiment_version')=='M1_v2':
            h=read(self.work/'HANDOFF.json')
            keys={'policy_id','entry_conditions','exit_conditions','overlap_role','successor_readiness',
                'failure_signatures','continuation_guidance','limitations','evidence'}
            if set(h)!=keys or h['policy_id']!=self.entry['policy_id']:raise ValueError('HANDOFF.json identity/keys mismatch')
            for key in keys-{'evidence'}:
                if not isinstance(h[key],str) or not h[key].strip():raise ValueError('Missing handoff text: '+key)
            if not isinstance(h['evidence'],list) or not h['evidence']:raise ValueError('Handoff requires training evidence')
            segments=read(self.entry['dataset'])['segments']
            for evidence in h['evidence']:
                if set(evidence)!={'trajectory_id','indices'} or not evidence['indices']:raise ValueError('Invalid handoff evidence')
                if any(type(i) is not int or not any(s['trajectory_id']==evidence['trajectory_id'] and s['start']<=i<=s['stop'] for s in segments) for i in evidence['indices']):
                    raise ValueError('Handoff evidence must belong to the assigned slices')
        audit(self.work)
        return object_hash(self.hashes())

    def execute(self,item):
        name=item['name'];cid=item['call_id'];args=json.loads(item['arguments'])
        row=self.j.db.execute('SELECT * FROM tools WHERE id=?',(cid,)).fetchone()
        if row:
            if row['name']!=name or row['args']!=item['arguments']:raise ValueError('Tool ID collision')
            if row['status']!='completed':raise RuntimeError('Interrupted tool requires explicit receipt reconciliation')
            return json.loads(row['result'])
        definitions={v['name']:v for v in self.schemas()};jsonschema.validate(args,definitions[name]['parameters'])
        self.j.charge('tool_call',self.budget.max_tool_calls,name=name,call_id=cid)
        with self.j.db:self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)',(cid,name,item['arguments'],'started',None))
        try:result=self.dispatch(name,args)
        except (ValueError,RuntimeError,KeyError,TypeError,SyntaxError) as error:
            result=dict(error=str(error),repair_allowed=not self.done());self.j.event('tool_diagnostic',name=name,error=str(error))
        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
        return result

    def dispatch(self,name,args):
        if name=='read_assignment':
            self.j.set('assignment_read',True)
            result=dict(**self.entry,segments=read(self.entry['dataset'])['segments'],
                completion_contract=read(self.cfg['completion_contract']),training=self.cfg['training'])
            if self.entry.get('experiment_version')=='M1_v2':
                from .data import handoff_evidence,shared_normalizer
                evidence=handoff_evidence(self.entry)
                atomic(self.folder/'handoff_evidence.json',evidence)
                result.update(shared_normalizer=shared_normalizer(self.entry),
                    measured_handoff_support=dict(fields=evidence['fields'],
                        boundary_statistics=evidence['boundary_statistics'],
                        overlap_ranges=[{k:v for k,v in r.items() if k not in ('start_state','end_state')} for r in evidence['overlaps']],
                        interpretation=evidence['interpretation']))
            return result
        if name=='read_public':
            paths={'public.py':ROOT/'src/appl/public.py','backbone.py':ROOT/'src/appl/vendor/diffusion_policy/conditional_unet1d.py'}
            return dict(content=INTERFACE if args['name']=='INTERFACE.md' else paths[args['name']].read_text())
        if name=='read_steps':
            dataset=read(self.entry['dataset']);matches=[s for s in dataset['segments'] if s['trajectory_id']==args['trajectory_id']]
            if not matches:raise ValueError('Unknown training trajectory')
            rows=[]
            for i in args['indices']:
                containing=[s for s in matches if s['start']<=i<=s['stop']]
                if not containing:raise ValueError('Index is outside assigned skill segments')
                # A reusable skill can recur within one original trajectory.
                # Prefer a slice containing this action, then an end-only label.
                containing.sort(key=lambda s:i==s['stop'])
                s=containing[0];v=read(Path(self.entry['dataset']).parent/s['file']);k=i-s['start']
                rows.append(dict(index=i,state=v['observations'][k]['state'],action=v['actions'][k] if k<len(v['actions']) else None))
            return dict(trajectory_id=args['trajectory_id'],rows=rows)
        if name in ('write_file','read_file'):
            path=args['path']
            if '/' in path or Path(path).suffix not in ('.py','.json','.md'):raise ValueError('Use a flat .py/.json/.md file')
            if name=='read_file':return dict(content=safe(self.work,path).read_text())
            if self.done():raise ValueError('Submitted source is immutable')
            self.j.write_candidate(path,args['content'])
            # Journal root/work is the source workspace; never rewrite API content.
            return dict(package_hash=object_hash(self.hashes()),files=self.hashes())
        if name=='check_policy':
            from .runner import gpu_job
            version=self.validate();n=self.j.count('interface_check')
            self.j.charge('interface_check',self.budget.max_checks,version=version,updates=2)
            source=self.folder/'design/versions'/version
            if not source.exists():shutil.copytree(self.work,source)
            output=self.folder/'checks'/f'{n:02d}'
            result=gpu_job(self.cfg,self.folder,source,output,self.gpu,updates=2,check=True)
            checked=self.j.get('checked',{});checked[version]=result;self.j.set('checked',checked)
            return dict(package_hash=version,**result)
        if name=='submit_policy':
            version=self.validate()
            if version!=args['expected_hash'] or version not in self.j.get('checked',{}):raise ValueError('Submit a successfully checked exact hash')
            source=self.folder/'source'
            if source.exists():raise ValueError('Publication already exists')
            shutil.copytree(self.work,source)
            record=dict(version=version,files=self.hashes(),assignment=self.entry,check=self.j.get('checked')[version])
            atomic(self.folder/'submission.json',record);self.j.set('submitted',record)
            return dict(submitted=True,version=version)
        raise ValueError('Unknown tool')


def design(cfg,folder,gpu):
    import fcntl
    folder=Path(folder)
    (folder/'design').mkdir(parents=True,exist_ok=True)
    # Queue and prefetch share one package owner. Waiting for that local owner
    # does not replay an API request or resume an uncertain provider outcome.
    with (folder/'design/package.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        # Also recognize sessions launched before the package lock existed.
        with (folder/'design/agent.lock').open('a') as active:
            fcntl.flock(active,fcntl.LOCK_EX)
        if (folder/'submission.json').exists():return read(folder/'submission.json')
        return _design_locked(cfg,folder,gpu)


def _design_locked(cfg,folder,gpu):
    tools=DesignTools(cfg,folder,gpu)
    atomic(folder/'design/interface.json',dict(prompt=PROMPT,interface=INTERFACE))
    try:
        AgentLoop(tools.j,tools,client(cfg),PROMPT).run(dict(policy_id=tools.entry['policy_id'],task='Implement and submit the assigned prior-based diffusion policy.'))
        return read(folder/'submission.json')
    finally:
        tools.j.db.close();os.environ.pop('APPL_PRIOR_TOKEN',None)
