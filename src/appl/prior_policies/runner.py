"""Bounded jobs and independent policy folders, without hand-authored candidates."""
from pathlib import Path
import os
import subprocess
import sys
import time
from ..io import ROOT,PIXI,MANIFEST,atomic,read,event,digest
from .data import catalog


def command(args):return [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','python','-m','appl.prior_policies',*args]


def clean_environment():
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(ROOT/'src'),
        LD_LIBRARY_PATH=str(Path(sys.prefix)/'lib'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for key in ('APPL_GPU_UUID','APPL_GPU_MINOR','APPL_PHYSICAL_GPU'):
        if key in os.environ:env[key]=os.environ[key]
    return env


def gpu_job(cfg,folder,source,output,gpu,updates,check=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from ..gpu import identity
    device=identity(gpu)
    occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv'],text=True)
    atomic(output/'gpu_occupancy.json',dict(device=device,compute_processes=occupancy,checked=time.time()))
    request=dict(folder=str(folder),source=str(source),output=str(output),updates=updates,check=check)
    atomic(output/'worker_request.json',request)
    args=['worker','--config',cfg['_path'],'--request',str(output/'worker_request.json'),'--gpu',str(gpu)]
    started=time.monotonic()
    with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
        proc=subprocess.Popen(command(args),cwd=ROOT,env=clean_environment(),stdout=out,stderr=err)
        atomic(output/'process.json',dict(pid=proc.pid,gpu=gpu,started=time.time()))
        code=proc.wait()
    record=dict(returncode=code,wall_seconds=time.monotonic()-started,updates=updates,interface_check=check,gpu=gpu)
    atomic(output/'process_result.json',record)
    if code:
        record['error']=(output/'stderr.log').read_text()[-6000:];atomic(output/'failure.json',record)
        raise RuntimeError(str(record))
    return read(output/'result.json')


def run_queue(cfg,gpu,shard,shards):
    from .design import design
    entries=catalog(cfg)
    for index,entry in enumerate(entries):
        if index%shards!=shard:continue
        folder=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'assignment.json').exists() and read(folder/'assignment.json')!=entry:raise ValueError('Assignment changed')
        atomic(folder/'assignment.json',entry)
        if (folder/'failure.json').exists():raise RuntimeError('Recorded terminal failure; explicit reconciliation required')
        try:
            print('Designing '+entry['policy_id'],flush=True)
            submission=design(cfg,folder,gpu)
            train_submitted(cfg,entry['policy_id'],gpu)
            print('Completed '+entry['policy_id'],flush=True)
        except Exception as error:
            atomic(folder/'failure.json',dict(error=str(error),time=time.time(),automatic_retry=False))
            raise


def train_submitted(cfg,policy_id,gpu):
    """Train one already submitted package, with one owner per checkpoint."""
    import fcntl
    matches=[e for e in catalog(cfg) if e['policy_id']==policy_id]
    if len(matches)!=1:raise ValueError('Unknown policy ID')
    entry=matches[0]
    folder=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
    submission=read(folder/'submission.json')
    if read(folder/'assignment.json')!=entry:raise ValueError('Assignment changed')
    for name,sha in submission['files'].items():
        if digest(folder/'source'/name)!=sha:raise ValueError('Submitted API file changed')
    with (folder/'training.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        if (folder/'training/result.json').exists():
            result=read(folder/'training/result.json')
            if digest(folder/'training/last.pt')!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            return result
        print('Training '+policy_id,flush=True)
        return gpu_job(cfg,folder,folder/'source',folder/'training',gpu,cfg['training']['updates'])


def prepare_queue(cfg,gpu,shard,shards):
    """Prepare later API packages while the same GPU's full training continues.

    The per-policy journal lock prevents duplicate API design sessions. Only
    short interface checks share the GPU. Training ownership is per package.
    """
    from .design import design
    for index,entry in enumerate(catalog(cfg)):
        if index%shards!=shard:continue
        folder=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
        if (folder/'submission.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'assignment.json').exists() and read(folder/'assignment.json')!=entry:raise ValueError('Assignment changed')
        atomic(folder/'assignment.json',entry)
        if (folder/'failure.json').exists():raise RuntimeError('Recorded failure requires reconciliation')
        print('Preparing '+entry['policy_id'],flush=True)
        design(cfg,folder,gpu)
        print('Submitted '+entry['policy_id'],flush=True)


class PolicyProcess:
    def __init__(self,cfg,folder,output,seed):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        request=dict(folder=str(folder),output=str(output),seed=seed)
        atomic(self.output/'request.json',request)
        self.errors=(self.output/'stderr.log').open('w')
        args=['policy-worker','--config',cfg['_path'],'--request',str(self.output/'request.json'),
              '--gpu',os.environ['APPL_PHYSICAL_GPU'],'--device-isolated']
        self.proc=subprocess.Popen(command(args),cwd=ROOT,env=clean_environment(),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.errors,text=True,bufsize=1)
        line=self.proc.stdout.readline()
        if not line or read_json(line).get('status')!='ready':raise RuntimeError('Policy worker failed: '+(self.output/'stderr.log').read_text()[-5000:])
        self.timings=[]

    def action(self,state,reset=False):
        import json
        self.proc.stdin.write(json.dumps(dict(state=state,reset=reset))+'\n');self.proc.stdin.flush()
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('Policy worker exited: '+(self.output/'stderr.log').read_text()[-5000:])
        value=read_json(line)
        if value['seconds']:self.timings.append(value['seconds'])
        return value['action']

    def close(self):
        self.proc.stdin.close();code=self.proc.wait(timeout=30);self.errors.close()
        atomic(self.output/'closed.json',dict(returncode=code,inference_chunks=len(self.timings),inference_seconds=sum(self.timings)))
        if code:raise RuntimeError('Policy worker failed')


def read_json(text):
    import json
    return json.loads(text)
