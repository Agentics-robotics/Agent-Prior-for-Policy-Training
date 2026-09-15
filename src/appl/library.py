"""Two-version API library lifecycle. No target performance in design tools."""
import time
import json
from pathlib import Path
from .design import DesignTools,client,schema
from .agent import AgentLoop
from .io import atomic,read,digest,object_hash
from .protocol import ROLES,require_design_gates,freeze

PROMPT='''You are the sole neural policy and prior designer for APPL. Use the
structured tools to inspect the original complete training trajectories and public
contracts. Determine segments yourself, organize the three task subgoals, implement
actual Torch priors/representations/losses, check, explicitly submit and train each
candidate for the entire declared budget. More than one prior per skill is allowed;
at most six retained policies. Keep the common diffusion backbone. No replay,
nearest-neighbour controller, IK, hidden data, target queries or extra demonstrations.
Build a first complete trained library, call verify_library ONCE for its fixed
MuJoCo suite, then revise the library ONCE using that feedback. You may keep,
replace or delete policies, change cuts or mechanisms, and must train new versions.
Submit freeze_library to irrevocably freeze version two (even if unchanged).
Version two may have ONE read-only verification; no subsequent edits or selection.
Then finish_design. Failed validations must remain explicit. The runner, never your
completion claim, judges task success. There are no additional deployment demands.
M1 receives your final cuts, skill interfaces, same inputs/backbone/training exposure
and the same deployment agent, with your numerical prior removed. The target tests
are inaccessible here and start only after all formal configuration is frozen.'''


class LibraryTools(DesignTools):
    def __init__(self,cfg):
        require_design_gates(cfg)
        super().__init__(cfg,'library')
        if self.j.get('phase') is None:self.j.set('phase','initial')
        (self.public/'LIFECYCLE.md').write_text('''One first-library MuJoCo feedback, one revision, immutable second library.
Every version must train for the full configured updates. Retained versions may be reused
without falsely claiming retraining. Skill handoffs additionally require TCP world z > .28 m.
Verification covers each policy at two original training segment starts and two composed
tasks, all in the independently calibrated frozen MuJoCo model. Recorded-state resets
do not restore a native contact solver snapshot. No target performance is returned.
The composed diagnostic uses the first listed policy per role in open/red/blue order;
the identical M1/M2 deployment API can instead select from all retained policies.
''')

    def schemas(self):
        items=[s for s in super().schemas() if s['name']!='finish_capability']
        for item in items:
            if item['name']=='read_public':item['parameters']['properties']['name']['enum'].append('LIFECYCLE.md')
            if item['name']=='train_candidate':
                item['parameters']['properties']['updates']=dict(type='integer',enum=[self.max_updates])
        versions=dict(type='array',items=dict(type='string'),minItems=3,maxItems=6)
        items.extend([
            schema('verify_library','One bounded frozen-model suite for first library, or the read-only frozen second library.',{'versions':versions}),
            schema('read_verification','Inspect bounded recorded state/action/metric rows from an already completed MuJoCo suite.',
                {'round':dict(type='integer',enum=[1,2]),'case_id':dict(type='string'),
                 'indices':dict(type='array',items=dict(type='integer'),minItems=1,maxItems=24)}),
            schema('freeze_library','Irrevocably freeze second library, its membership and design note before read-only verification.',
                   {'versions':versions,'design_note':dict(type='string')}),
            schema('finish_design','Finish only after first feedback, one revision/freeze and read-only second verification.',{}),
        ])
        return items

    def selected(self,versions):
        if len(set(versions))!=len(versions):raise ValueError('Duplicate version')
        rows=[];roles=set();policy_ids=set()
        for version in versions:
            trained=self.j.get('trained',{}).get(version)
            if not trained or trained['optimizer_steps']!=self.max_updates:raise ValueError('Every selected version must finish the full neural budget')
            directory=self.root/'versions'/version;metadata=self.metadata(directory)
            if metadata['policy_id'] in policy_ids:raise ValueError('Library policy IDs must be unique')
            policy_ids.add(metadata['policy_id']);roles.add(metadata['skill'])
            checkpoint=Path(trained['output'])/'last.pt'
            if digest(checkpoint)!=trained['checkpoint_sha256']:raise ValueError('Checkpoint content changed')
            files=read(directory/'manifest.json')
            if any(digest(directory/name)!=h for name,h in files.items()):raise ValueError('Candidate version changed')
            rows.append(dict(version=version,candidate=str(directory),checkpoint=str(checkpoint),
                checkpoint_sha256=trained['checkpoint_sha256'],metadata=metadata,training=trained))
        if roles!=set(ROLES):raise ValueError('The complete library must cover all three task subgoals')
        return rows

    def dispatch(self,name,args):
        phase=self.j.get('phase')
        if phase in ('frozen','finished') and name in ('write_file','check_candidate','submit_candidate','train_candidate','freeze_library'):
            raise ValueError('Second-version library is frozen; no further code, training or membership changes')
        if name=='read_verification':
            if not self.j.get('verification_'+str(args['round'])):raise ValueError('This suite has not completed')
            directory=self.root/'verification'/str(args['round'])
            names={case['name'] for case in read(directory/'suite.json')['cases']}
            if args['case_id'] not in names:raise ValueError('Unknown verified case')
            rows=[json.loads(line) for line in (directory/args['case_id']/'trace.jsonl').read_text().splitlines()]
            if any(not 0<=index<len(rows) for index in args['indices']):raise ValueError('Verification index outside recorded trace')
            return dict(rows=[rows[index] for index in args['indices']])
        if name=='verify_library':
            from .worker import launch
            if phase not in ('initial','frozen'):raise ValueError('Only one feedback validation before revision and one after freeze')
            selected=self.selected(args['versions'])
            if phase=='frozen' and args['versions']!=self.j.get('second')['versions']:raise ValueError('Read-only verification must use exact frozen membership/order')
            round_number=1 if phase=='initial' else 2
            if self.j.get('verification_'+str(round_number)):raise ValueError('This library already received its single verification')
            self.j.charge('library_verification',2,round=round_number)
            output=self.root/'verification'/str(round_number)
            request=dict(operation='verify_library',policies=selected,round=round_number)
            process=launch(self.cfg,request,output,self.cfg['devices']['secondary']);started=time.monotonic();code=process.wait()
            self.j.event('verification_cost',round=round_number,elapsed_seconds=time.monotonic()-started,returncode=code)
            if code:raise RuntimeError('Verification worker failed: '+(output/'stderr.log').read_text()[-3000:])
            result=read(output/'result.json');self.j.set('verification_'+str(round_number),result)
            if phase=='initial':
                self.j.set('first',dict(versions=args['versions'],policies=selected))
                self.j.set('phase','revision')
            return result
        if name=='freeze_library':
            if phase!='revision':raise ValueError('Read first-library verification before the single revision')
            selected=self.selected(args['versions'])
            value=dict(versions=args['versions'],policies=selected,design_note=args['design_note'],
                       first_versions=self.j.get('first')['versions'],feedback_revisions=1)
            value['library_sha256']=object_hash(value)
            self.j.set('second',value);self.j.set('phase','frozen');atomic(self.root/'second_library.json',value)
            return dict(library_sha256=value['library_sha256'],immutable=True,read_only_verification_remaining=1)
        if name=='finish_design':
            if phase!='frozen' or not self.j.get('verification_2'):raise ValueError('Freeze and read-only verify the second library first')
            value=dict(**self.j.get('second'),first_verification=self.j.get('verification_1'),
                second_verification=self.j.get('verification_2'))
            self.j.set('finished',value);self.j.set('frozen',value);self.j.set('phase','finished')
            atomic(self.root/'submission.json',value)
            return dict(frozen=True,library_sha256=value['library_sha256'],validation_passed=value['second_verification']['success'])
        return super().dispatch(name,args)


def design(cfg):
    tools=LibraryTools(cfg)
    result=AgentLoop(tools.j,tools,client(cfg),PROMPT).run(dict(task='One complete APPL neural library and exactly one MuJoCo feedback revision',
        public_files=['INTERFACE.md','LIFECYCLE.md','baseline.py','public.py','data.json'],updates_per_candidate=cfg['training']['updates']))
    freeze(cfg,result)
    return result
