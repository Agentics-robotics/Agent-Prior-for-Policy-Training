"""Finite evaluation matrix: no automatic retry of a physical or API attempt."""
from collections import deque
import fcntl
import os
import subprocess
import time
from ..io import ROOT,PIXI,MANIFEST,read,atomic,event
from ..prior_policies.data import catalog
from .protocol import BASE,NAMES,config,freeze,naive_paths


def training_status():
    missing=[];failures=[]
    for name in NAMES:
        cfg=config(name)
        if name=='drawer_exchange':continue
        checks=BASE/name/'inference_checks'
        if not (checks/'summary.json').exists():missing.append(str(checks/'summary.json'))
        failures.extend(str(p) for p in checks.glob('*/failure.json'))
        naive=BASE/name/'naive_DP/training'
        if (naive/'failure.json').exists():failures.append(str(naive))
        if not (naive/'result.json').exists():missing.append(str(naive))
        for entry in catalog(cfg):
            folder=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
            if (folder/'failure.json').exists() or (folder/'training/failure.json').exists():failures.append(str(folder))
            if not (folder/'training/result.json').exists():missing.append(str(folder))
    return missing,failures


def occupancy():
    from ..gpu import identity
    return {i:identity(i) for i in config('two_block_sort')['devices']}


def supplement(name,gpu):
    """Keep two training jobs on the same GPU after its naive DP completes.

    Only already API-submitted packages are eligible. The existing per-policy
    training lock and checkpoint validation still ensure exactly one training.
    """
    from ..prior_policies.runner import train_submitted
    cfg=config(name)
    if gpu not in cfg['devices'] or len(cfg['devices'])>4:raise ValueError('Invalid allocation')
    folder=BASE/name/'supplement';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (BASE/name/'naive_DP/training/failure.json').exists():raise RuntimeError('Baseline failed')
            if (BASE/name/'naive_DP/training/result.json').exists():break
            time.sleep(20)
        while True:
            entries=catalog(cfg);remaining=[];ready=[]
            for e in entries:
                p=cfg['output']/'policies'/e['skill_id']/f"heuristic_{e['heuristic_index']:02d}"
                if (p/'failure.json').exists() or (p/'training/failure.json').exists():raise RuntimeError('Recorded terminal policy failure')
                if (p/'training/result.json').exists():continue
                remaining.append(e)
                if (p/'submission.json').exists() and not (p/'training').exists():ready.append(e)
            if not remaining:break
            atomic(folder/'status.json',dict(remaining=len(remaining),eligible=len(ready),gpu=gpu,time=time.time()))
            if ready:
                selected=ready[-1]
                event(folder/'events.jsonl','train_submitted',policy_id=selected['policy_id'],gpu=gpu)
                train_submitted(cfg,selected['policy_id'],gpu)
            else:time.sleep(20)
        atomic(folder/'completed.json',dict(gpu=gpu,completed=time.time()))


def run():
    out=BASE/'batch';out.mkdir(parents=True,exist_ok=True)
    with (out/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'completed.json').exists():return read(out/'completed.json')
        while True:
            missing,failures=training_status()
            atomic(out/'status.json',dict(state='waiting_for_all_frozen_libraries',missing=missing,failures=failures,time=time.time()))
            if failures:raise RuntimeError('Recorded training/design failure requires explicit reconciliation')
            if not missing:break
            time.sleep(30)
        freeze()
        devices=config('two_block_sort')['devices']
        if len(devices)>4 or any(config(n)['devices']!=devices for n in NAMES):raise ValueError('Four-device allocation mismatch')
        pending={method:deque() for method in ('APPL','naive_DP')}
        for index in range(30):
            for condition in ('ID','OOD'):
                for name in NAMES:
                    seed=config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][index]
                    for method in pending:pending[method].append((name,condition,seed))
        active={};finished=[]
        while any(pending.values()) or active:
            for slot,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['stdout'].close();item['stderr'].close()
                record=dict(task=item['task'],method=slot[1],condition=item['condition'],seed=item['seed'],gpu=slot[0],returncode=code)
                atomic(item['launch_dir']/'process_result.json',record);finished.append(record);del active[slot]
                event(out/'events.jsonl','episode_process_finished',**record)
            devices_now=occupancy()
            for gpu in devices:
                for method,slot_index in (('APPL',0),('naive_DP',0),('APPL',1)):
                    slot=(gpu,method,slot_index)
                    if slot in active or not pending[method]:continue
                    # Two APPL episodes can overlap API waits. Reserve 10 GiB per
                    # APPL library and 3 GiB for DP, within each 24 GiB card.
                    required=10000 if method=='APPL' else 3000
                    if devices_now[gpu]['free_mib']<required:continue
                    name,condition,seed=pending[method].popleft()
                    episode=BASE/name/'evaluation'/method/condition/str(seed)
                    if episode.exists():
                        record=dict(task=name,method=method,condition=condition,seed=seed,
                            state='retained_completed' if (episode/'result.json').exists() else 'retained_interruption_no_retry')
                        finished.append(record);continue
                    launch_dir=out/'jobs'/name/method/condition/str(seed)
                    if launch_dir.exists():
                        if not (launch_dir/'process_result.json').exists():raise RuntimeError('Unreconciled existing evaluation process')
                        finished.append(dict(task=name,method=method,condition=condition,seed=seed,state='retained_launch_failure_no_retry'))
                        continue
                    launch_dir.mkdir(parents=True,exist_ok=False)
                    args=[PIXI,'run','--manifest-path',str(MANIFEST),'--locked','python','-m','appl.scaleup',
                        'evaluate','--task',name,'--method',method,'--condition',condition,'--seeds',str(seed),'--gpu',str(gpu)]
                    stdout=(launch_dir/'stdout.log').open('w');stderr=(launch_dir/'stderr.log').open('w')
                    proc=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
                    atomic(launch_dir/'process.json',dict(pid=proc.pid,command=args,started=time.time(),occupancy=devices_now[gpu]))
                    active[slot]=dict(process=proc,stdout=stdout,stderr=stderr,launch_dir=launch_dir,
                        task=name,condition=condition,seed=seed)
                    devices_now[gpu]['free_mib']-=required
            atomic(out/'status.json',dict(state='evaluating',finished=len(finished),pending={m:len(q) for m,q in pending.items()},
                active=[dict(gpu=g,method=m,slot=s,task=v['task'],condition=v['condition'],seed=v['seed'],pid=v['process'].pid)
                    for (g,m,s),v in active.items()],time=time.time()))
            time.sleep(5)
        from .report import generate
        result=generate();atomic(out/'completed.json',dict(episodes=finished,report=result,finished=time.time()))
        return result


if __name__=='__main__':run()
