"""API-owned MuJoCo reconstruction, independent of learned-policy performance."""
from pathlib import Path
from types import SimpleNamespace
import base64
import json
import shutil
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import jsonschema

from .io import ROOT,read,atomic,digest
from .journal import Journal,encode,sha,safe
from .design import client,schema
from .agent import AgentLoop
from .model_contract import parse_xml,materialize,audit_model,edited_xml
from .envs.evaluator import measure,SuccessTracker

LIMITS=dict(tcp=.01,drawer=.01,red=.02,blue=.02)
CASES=[('demo1000',0),('demo1007',0),('demo1000',340),('demo1000',700)]


class Simulator:
    def __init__(self,path):
        self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model)
        self.tcp=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'tcp')

    def reset(self,state,previous=None):
        m,d=self.model,self.data;mujoco.mj_resetData(m,d)
        d.qpos[:9]=state['qpos'];d.qvel[:9]=state['qvel'];d.qpos[9]=state['drawer_position'][0];d.qvel[9]=state['drawer_velocity'][0]
        for name,q,v in [('red',10,10),('blue',17,16)]:
            d.qpos[q:q+7]=state[name+'_pose']
            if previous is not None:d.qvel[v:v+3]=(np.asarray(state[name+'_pose'][:3])-previous[name+'_pose'][:3])/.05
        d.ctrl[:9]=state['qpos'];mujoco.mj_forward(m,d)
        return self.state()

    def state(self):
        d=self.data;quat=np.zeros(4);mujoco.mju_mat2Quat(quat,d.site_xmat[self.tcp])
        return dict(qpos=d.qpos[:9].tolist(),qvel=d.qvel[:9].tolist(),
            tcp_pose=[*d.site_xpos[self.tcp].tolist(),*quat.tolist()],drawer_position=[float(d.qpos[9])],
            drawer_velocity=[float(d.qvel[9])],red_pose=d.qpos[10:17].tolist(),blue_pose=d.qpos[17:24].tolist())

    def step(self,action):
        self.data.ctrl[:7]=action[:7];self.data.ctrl[7:]=.025*action[7]+.015
        mujoco.mj_step(self.model,self.data,nstep=25)
        if not np.isfinite(self.data.qpos).all():raise ValueError('Nonfinite reconstructed dynamics')
        return self.state()

    def speeds(self):
        d=self.data
        return dict(red_linear_speed=float(np.linalg.norm(d.qvel[10:13])),red_angular_speed=float(np.linalg.norm(d.qvel[13:16])),
                    blue_linear_speed=float(np.linalg.norm(d.qvel[16:19])),blue_angular_speed=float(np.linalg.norm(d.qvel[19:22])))

    def contacts(self):
        result=[]
        for c in self.data.contact:
            names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
            if any(n in ('red_geom','blue_geom') for n in names):result.append(dict(geoms=names,distance=float(c.dist)))
        return result


def calibrate(cfg,request):
    from .security import lockdown
    out=Path(request['output']);candidate=Path(request['candidate']);assets=Path(request['assets'])
    # Load only the explicitly allowed original calibration observations before
    # lockdown. Generated XML never executes Python or gets target access.
    trajectories={identifier:read(cfg['data']['directory']/(identifier+'.json')) for identifier,_ in CASES}
    public=read(ROOT/'data/exp2/reconstruction_public/interface.json')
    lockdown(candidate,out,[assets],gpu=False)
    materialize(candidate/'model.xml',assets,request['asset_manifest'],out/'resolved.xml')
    sim=Simulator(out/'resolved.xml');checks=audit_model(sim.model,public)
    fk=[]
    for t in trajectories.values():
        for o in t['observations'][::10]:
            sim.reset(o['state']);fk.append(float(np.linalg.norm(sim.data.site_xpos[sim.tcp]-o['state']['tcp_pose'][:3])))
    checks['kinematics']=max(fk)<.001
    results=[]
    for identifier,start in CASES:
        t=trajectories[identifier];sim.reset(t['observations'][start]['state'],t['observations'][start-1]['state'] if start else None)
        tracker=SuccessTracker(cfg['evaluation']);errors={k:[] for k in LIMITS};trace=[]
        for index,action in enumerate(t['actions'][start:],start):
            state=sim.step(action);reference=t['observations'][index+1]['state']
            metrics=tracker.update(measure(None,state,cfg['evaluation'],sim.speeds()))
            for name,field in [('tcp','tcp_pose'),('drawer','drawer_position'),('red','red_pose'),('blue','blue_pose')]:
                errors[name].append(float(np.linalg.norm(np.asarray(state[field][:3])-reference[field][:3])))
            trace.append(dict(step=index+1,state=state,reference=reference,metrics=metrics,contacts=sim.contacts()))
        rms={name:float(np.sqrt(np.mean(np.square(values)))) for name,values in errors.items()}
        name=f'{identifier}_{start}'
        atomic(out/(name+'.json'),trace)
        result=dict(id=name,trajectory=identifier,start=start,steps=len(trace),rmse_m=rms,final=metrics,
                    limits=LIMITS,passed=all(rms[k]<=LIMITS[k] for k in LIMITS) and metrics['success'])
        results.append(result)
    value=dict(success=all(checks.values()) and all(r['passed'] for r in results),checks=checks,fk_max_m=max(fk),cases=results,
        original_demonstrations_only=True,learned_policy_performance=False,physical_steps=sum(r['steps'] for r in results),
        calibration_version=request['version'],reset_limitation='No complete native contact snapshot; recorded q/qvel and estimated object linear velocity, angular velocity zero at nonzero reset')
    atomic(out/'result.json',value);return value


PROMPT='''You are the only MuJoCo reconstruction designer. This is a separate
development calibration phase before formal policy verification. Read CONTRACT.md,
model.xml, native_panda.urdf and original training state/image evidence. The initial
model comes from a previous real API session with documented poor contact accuracy;
its provenance is explicit, not a new design claim. Repair it using allowed data
and controlled XML edits/calibration. The developer will not hand-tune it for you.
You may change reconstructed scene geometry/contact parameters using independent
observations, never learned-policy or hidden-test results. Native robot/control,
known object dimensions/mass and the numerical interface remain fixed. Calibrate
grasp, transport, release and settling, not just TCP. Do not attach objects to the
robot, teleport objects or disable physical collisions. No shell or target calls.
Submit a model only if the fixed calibration succeeds. If the bounded reconstruction
cannot meet the declared gate, finish_model honestly with successful=false and an
explanation; no weaker fallback model will be called trusted.'''


class ModelTools:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=cfg['output']/'reconstruction';self.j=Journal(self.root)
        self.work=self.root/'work';self.work.mkdir(exist_ok=True)
        self.budget=SimpleNamespace(max_api_calls=80)
        if not (self.work/'model.xml').exists():
            shutil.copy2(ROOT/'data/exp2/public/api_model.xml',self.work/'model.xml')
            atomic(self.root/'lineage.json',dict(source='previous actual API reconstruction',sha256=digest(self.work/'model.xml'),
                source_path='data/exp2/public/api_model.xml',new_model_claimed=False))
        self.assets=(ROOT/'data/exp2/assets').resolve()
        self.assets_manifest={str(p.relative_to(self.assets)):digest(p) for p in self.assets.rglob('*') if p.is_file() and p.suffix in ('.obj','.stl','.png','.xml','.urdf','.md')}
        contract=dict(limits_rmse_m=LIMITS,cases=CASES,fk_max_m=.001,object_edge_m=.04,object_mass_kg=.064,
            expected_joint_names=[f'joint{i}' for i in range(1,8)]+['finger_joint1','finger_joint2','drawer_slide','red_joint','blue_joint'],
            expected_nq=24,expected_nv=22,expected_nu=9,robot_base=[-.85,0,0],
            actuator='9 ordered position actuators; kp1000, kd100, force +/-100; native gripper .025*g+.015',
            gravity=[0,0,-9.81],timestep=.002,substeps=25,drawer_axis=[-1,0,0],drawer_range=[0,.30],
            final_evaluation=cfg['evaluation'],max_calibrations=12,
            allowed_body_names=['world','link0','link1','link2','link3','link4','link5','link6','link7','hand','left_finger','right_finger','drawer','red','blue'],
            limits_rationale='20 mm object RMSE is half the 40 mm cube edge and below remaining pad clearance; require actual released/stable final goals too',
            forbidden='includes/plugins/mocap/object actuators/attachments/hidden target data; only finger mimic equality allowed')
        if (self.root/'contract.json').exists() and sha(read(self.root/'contract.json'))!=sha(contract):raise ValueError('Calibration contract changed during session')
        atomic(self.root/'contract.json',contract)

    def schemas(self):
        s=dict(type='string');i=dict(type='integer')
        return [schema('read_model_file','Read an explicitly authorized model or public robot/contract file.',{'name':dict(type='string',enum=['model.xml','native_panda.urdf','CONTRACT.md'])}),
            schema('edit_model_xml','Atomic XML changes. JSON list of set/remove/append with selector, attributes or xml. Every selector must match.',{'edits_json':s}),
            schema('read_training','Read original training observations/actions at explicit indices.',{'trajectory_id':s,'indices':dict(type='array',items=i,minItems=1,maxItems=24)}),
            schema('read_training_image','View an original synchronized training frame.',{'trajectory_id':s,'index':i}),
            schema('calibrate_model','Compile, audit and replay the fixed four calibration cases; no learned policy or target calls.',{}),
            schema('read_calibration','Read indexed calibration states, references, contact pairs and fixed predicates.',{'calibration_id':s,'case_id':s,'indices':dict(type='array',items=i,minItems=1,maxItems=30)}),
            schema('finish_model','Freeze a passing model or explicitly end with an unmet calibration gate.',{'expected_hash':s,'successful':dict(type='boolean'),'design_note':s})]

    def done(self):return self.j.get('finished',False)

    def execute(self,item):
        name=item['name'];args=json.loads(item['arguments']);cid=item['call_id']
        existing=self.j.db.execute('SELECT * FROM tools WHERE id=?',(cid,)).fetchone()
        if existing:
            if existing['status']=='completed':return json.loads(existing['result'])
            if name=='edit_model_xml':
                last=self.j.db.execute("SELECT payload FROM events WHERE kind='model_edit' ORDER BY id DESC LIMIT 1").fetchone()
                expected=json.loads(last['payload'])['sha256'] if last else read(self.root/'lineage.json')['sha256']
                if digest(self.work/'model.xml')==expected:
                    try:edited_xml((self.work/'model.xml').read_text(),json.loads(args['edits_json']))
                    except SyntaxError as error:
                        result=dict(error=str(error)+'. Selectors use relative ElementTree XPath, e.g. .//geom; the entire edit batch applied zero changes.',reconciled=True,applied_edits=0)
                        self.j.event('operator_tool_reconciled',call_id=cid,reason='Uncaught XPath syntax diagnostic; unchanged model hash verified',model_sha256=expected)
                        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
                        return result
            raise RuntimeError('Interrupted reconstruction tool requires explicit reconciliation')
        definition={s['name']:s for s in self.schemas()}[name];jsonschema.validate(args,definition['parameters'])
        self.j.charge('tool_call',120,name=name,call_id=cid)
        with self.j.db:self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)',(cid,name,encode(args),'started',None))
        try:result=self.dispatch(name,args)
        except (ValueError,KeyError,RuntimeError,ET.ParseError,SyntaxError) as error:
            result=dict(error=str(error));self.j.event('tool_diagnostic',call_id=cid,error=str(error))
        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
        return result

    def dispatch(self,name,args):
        if name=='read_model_file':
            if args['name']=='model.xml':return dict(content=(self.work/'model.xml').read_text(),sha256=digest(self.work/'model.xml'))
            if args['name']=='CONTRACT.md':return read(self.root/'contract.json')
            return dict(content=(ROOT/'data/exp2/reconstruction_public/native_panda.urdf').read_text())
        if name=='edit_model_xml':
            edits=json.loads(args['edits_json'])
            content=edited_xml((self.work/'model.xml').read_text(),edits)
            version=self.j.object(content);temporary=self.work/'model.xml.tmp';temporary.write_text(content);temporary.replace(self.work/'model.xml')
            self.j.event('model_edit',sha256=version,edits=edits)
            return dict(sha256=version)
        if name in ('read_training','read_training_image'):
            identifier=args['trajectory_id']
            if identifier not in self.cfg['data']['train_ids']:raise ValueError('Only the declared training trajectories are allowed')
            t=read(self.cfg['data']['directory']/(identifier+'.json'))
            if name=='read_training':
                rows=[]
                for index in args['indices']:
                    if not 0<=index<len(t['observations']):raise ValueError('Index outside trajectory')
                    rows.append(dict(index=index,state=t['observations'][index]['state'],action=t['actions'][index] if index<len(t['actions']) else None))
                return dict(id=identifier,steps=len(t['actions']),rows=rows)
            index=args['index']
            if not 0<=index<len(t['observations']):raise ValueError('Index outside trajectory')
            path=self.cfg['data']['directory']/t['observations'][index]['images']['front']
            return [dict(type='input_text',text=encode(dict(trajectory=identifier,index=index))),dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),detail='high')]
        if name=='calibrate_model':
            from .worker import launch
            number=self.j.count('calibration');self.j.charge('calibration',12)
            version=digest(self.work/'model.xml');directory=self.root/'versions'/version
            directory.mkdir(parents=True,exist_ok=True);shutil.copy2(self.work/'model.xml',directory/'model.xml')
            output=self.root/'calibration'/f'{number:03d}'
            request=dict(operation='calibrate',candidate=str(directory),assets=str(self.assets),asset_manifest=self.assets_manifest,version=version)
            started=time.monotonic();process=launch(self.cfg,request,output,self.cfg['devices']['secondary']);code=process.wait()
            receipt=dict(calibration_id=f'{number:03d}',returncode=code,version=version,elapsed_seconds=time.monotonic()-started)
            self.j.event('calibration_cost',**receipt)
            if code:raise RuntimeError((output/'stderr.log').read_text()[-5000:])
            result=read(output/'result.json');self.j.set('latest_calibration',dict(**receipt,result=result))
            return dict(**receipt,result=result,remaining_calibrations=12-self.j.count('calibration'))
        if name=='read_calibration':
            if not args['calibration_id'].isdigit():raise ValueError('Invalid calibration ID')
            if args['case_id'] not in [f'{i}_{s}' for i,s in CASES]:raise ValueError('Unknown calibration case')
            trace=read(safe(self.root/'calibration',args['calibration_id']+'/'+args['case_id']+'.json'))
            if any(not 0<=i<len(trace) for i in args['indices']):raise ValueError('Trace indices are zero based')
            return dict(rows=[trace[i] for i in args['indices']])
        if name=='finish_model':
            version=digest(self.work/'model.xml')
            if args['expected_hash']!=version:raise ValueError('Model hash mismatch')
            latest=self.j.get('latest_calibration')
            if args['successful'] and (not latest or latest['version']!=version or not latest['result']['success']):raise ValueError('Current model has not passed the independent calibration gate')
            value=dict(version=version,calibration_passed=args['successful'],design_note=args['design_note'],latest_calibration=latest)
            atomic(self.root/'submission.json',value);self.j.set('finished',True);self.j.set('frozen',value)
            return value
        raise ValueError('Unknown reconstruction tool')


def reconstruct(cfg):
    tools=ModelTools(cfg)
    return AgentLoop(tools.j,tools,client(cfg),PROMPT).run(dict(task='Repair and validate the development simulator using original calibration evidence only',calibration_limits=LIMITS))
