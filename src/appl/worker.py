"""Fixed worker operations; generated code never shares the environment process."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from .io import ROOT,atomic,read,event
from .gpu import command


def clean_environment():
    result=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
        LD_LIBRARY_PATH=str(Path(sys.prefix)/'lib'),
        PYTHONPATH=str(ROOT/'src'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        PIXI_CACHE_DIR='/home/storage/oscar/appl_exp2/cache/pixi',UV_CACHE_DIR='/home/storage/oscar/appl_exp2/cache/uv')
    for key in ('APPL_GPU_UUID','APPL_GPU_MINOR','APPL_PHYSICAL_GPU'):
        if key in os.environ:result[key]=os.environ[key]
    return result


def launch(cfg,request,output,gpu):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request=dict(request,output=str(output))
    atomic(output/'worker_request.json',request)
    args=['worker','--config',cfg['_path'],'--request',str(output/'worker_request.json'),'--gpu',str(gpu)]
    with (output/'stdout.log').open('a') as out,(output/'stderr.log').open('a') as err:
        process=subprocess.Popen(command(args),cwd=ROOT,env=clean_environment(),stdout=out,stderr=err,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
    atomic(output/'process.json',dict(pid=process.pid,started=time.time(),gpu=gpu,command=command(args)))
    return process


def run(cfg,request):
    from .security import lockdown
    output=Path(request['output'])
    operation=request['operation']
    if operation=='verify_library':
        from .verification import run
        return run(cfg,request)
    if operation=='calibrate':
        from .surrogate import calibrate
        return calibrate(cfg,request)
    if operation=='security_probe':
        import socket
        import torch
        enforcement=lockdown(Path(request['candidate']),output)
        denied={}
        for name,path in [('outside',request['sentinel']),('environment','/proc/self/environ')]:
            try:Path(path).read_bytes()
            except PermissionError:denied[name]=True
            else:denied[name]=False
        try:socket.socket()
        except PermissionError:denied['network']=True
        else:denied['network']=False
        try:subprocess.run(['/usr/bin/true'],check=True)
        except PermissionError:denied['process_creation']=True
        else:denied['process_creation']=False
        model=torch.nn.Linear(4,2).cuda();optimizer=torch.optim.AdamW(model.parameters())
        before=model.weight.detach().clone()
        model(torch.ones(3,4,device='cuda')).square().mean().backward();optimizer.step()
        changed=bool(torch.any(before!=model.weight))
        value=dict(success=all(denied.values()) and changed,denied=denied,post_lockdown_gpu_update=changed,enforcement=enforcement)
        atomic(output/'result.json',value)
        if not value['success']:raise RuntimeError('Kernel/GPU security probe failed')
        return value
    if operation=='train':
        from .train import fit
        candidate=Path(request['candidate'])
        metadata=read(candidate/'candidate.json')
        return fit(cfg,output,ids=request.get('ids'),segments=metadata['segments'],candidate=candidate,
            updates=request['updates'],seed=request.get('seed',0),
            lockdown=lambda:lockdown(candidate,output))
    if operation=='baseline_train':
        from .train import fit
        return fit(cfg,output,segments=request.get('segments'),updates=request['updates'],seed=request['seed'])
    if operation=='policy':
        from .train import LoadedPolicy
        candidate=Path(request['candidate'])
        checkpoint=Path(request['checkpoint'])
        lockdown(candidate,output,[checkpoint])
        policy=LoadedPolicy(checkpoint,candidate,seed=request.get('seed',0))
        print(json.dumps(dict(status='ready')),flush=True)
        for line in sys.stdin:
            value=json.loads(line)
            before=len(policy.timings);action=policy.action(value['state'])
            print(json.dumps(dict(action=action,inference_seconds=policy.timings[-1] if len(policy.timings)>before else 0)),flush=True)
        return None
    raise ValueError('Unknown fixed worker operation')


class PolicyProcess:
    def __init__(self,cfg,candidate,checkpoint,output,gpu,seed):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        request=dict(operation='policy',candidate=str(candidate),checkpoint=str(checkpoint),output=str(self.output),seed=seed)
        atomic(self.output/'worker_request.json',request)
        self.errors=(self.output/'stderr.log').open('w')
        if int(os.environ['APPL_PHYSICAL_GPU'])!=gpu:raise ValueError('Persistent policy must use its parent device namespace')
        self.process=subprocess.Popen(command(['worker','--config',cfg['_path'],'--request',str(self.output/'worker_request.json'),'--gpu',str(gpu),'--device-isolated']),
            cwd=ROOT,env=clean_environment(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,bufsize=1)
        line=self.process.stdout.readline()
        if not line or json.loads(line).get('status')!='ready':
            raise RuntimeError('Policy worker initialization failed: '+(self.output/'stderr.log').read_text()[-3000:])
        self.timings=[]

    def action(self,state):
        self.process.stdin.write(json.dumps(dict(state=state))+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Policy worker exited: '+(self.output/'stderr.log').read_text()[-3000:])
        value=json.loads(line)
        if value['inference_seconds']:self.timings.append(value['inference_seconds'])
        return value['action']

    def close(self):
        self.process.stdin.close();code=self.process.wait(timeout=30);self.errors.close()
        atomic(self.output/'closed.json',dict(returncode=code,inference_chunks=len(self.timings),inference_seconds=sum(self.timings)))
        if code:raise RuntimeError('Policy worker failed during shutdown')
