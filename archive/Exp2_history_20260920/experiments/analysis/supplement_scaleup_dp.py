"""Resource-only dispatch of already planned DP trials, without retries.

The frozen main queue proceeds from the front; this bounded helper takes only
the declared contiguous tail indices. Both use the same exclusive episode/job directories
and the unchanged frozen evaluator. No training or API design occurs here.
"""
from collections import deque
import argparse
import fcntl
import subprocess
import time

from appl.gpu import identity
from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, read
from appl.scaleup.protocol import BASE, NAMES, config


def main(first_index=29,last_index=28,wait_for_initial=False,predecessor=None):
    if not 0<=last_index<=first_index<30:raise ValueError('Invalid fixed seed-index interval')
    name='supplement_DP' if (first_index,last_index)==(29,28) else f'supplement_DP_{first_index}_{last_index}'
    out=BASE/'batch'/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'completed.json').exists():return
        freeze=read(BASE/'study_freeze.json');devices=config('two_block_sort')['devices']
        if devices!=[1,2,3,7]:raise ValueError('Unexpected four-card allocation')
        queue=deque((name,condition,config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][index])
            for index in range(first_index,last_index-1,-1) for condition in ('ID','OOD') for name in NAMES)
        previous=None
        if predecessor:
            if predecessor==name or '/' in predecessor:raise ValueError('Invalid predecessor queue')
            previous=BASE/'batch'/predecessor
            previous_plan=read(previous/'plan.json')
            if previous_plan['gpu_pool']!=devices or previous_plan['additional_DP_slots_per_gpu']!=2:
                raise ValueError('Predecessor slot allocation differs')
            if set(queue)&{tuple(v) for v in previous_plan['tasks']}:
                raise ValueError('Predecessor assignments overlap')
        atomic(out/'plan.json',dict(study_sha256=freeze['study_sha256'],tasks=list(queue),
            additional_DP_slots_per_gpu=2,gpu_pool=devices,new_scientific_slots=0,
            source_sha256=digest(__file__),automatic_retries=False,wait_for_initial=wait_for_initial,
            predecessor=predecessor,slot_handoff='Only after predecessor pending count is zero; retain all its active slots.'))
        if wait_for_initial:
            if name=='supplement_DP':raise ValueError('Cannot wait for this same queue')
            while not (BASE/'batch/supplement_DP/completed.json').exists():
                atomic(out/'status.json',dict(state='waiting_for_initial_supplement',time=time.time(),
                    finished=0,pending=len(queue),active=[]))
                time.sleep(20)
        active={};done=[]
        while queue or active:
            for slot,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['stdout'].close();item['stderr'].close()
                result=dict(task=item['task'],condition=item['condition'],seed=item['seed'],
                    method='naive_DP',gpu=slot[0],returncode=code,resource_dispatch='supplement_DP')
                atomic(item['job']/'process_result.json',result);done.append(result);del active[slot]
            occupied=set()
            if previous is not None and not (previous/'completed.json').exists():
                status=read(previous/'status.json')
                if status['pending']:
                    occupied={(gpu,index) for gpu in devices for index in range(2)}
                else:
                    # With no pending jobs, this predecessor can only release
                    # slots; it cannot race us by filling a released slot again.
                    occupied={(v['gpu'],v['slot']) for v in status['active']}
            for gpu in devices:
                free=identity(gpu)['free_mib']
                for index in range(2):
                    if (gpu,index) in active or (gpu,index) in occupied or not queue or free<6000:continue
                    name,condition,seed=queue.popleft()
                    episode=BASE/name/'evaluation/naive_DP'/condition/str(seed)
                    job=BASE/'batch/jobs'/name/'naive_DP'/condition/str(seed)
                    if episode.exists() or job.exists():
                        done.append(dict(task=name,condition=condition,seed=seed,state='existing_attempt_retained'));continue
                    job.mkdir(parents=True,exist_ok=False)
                    args=[PIXI,'run','--manifest-path',str(MANIFEST),'--locked','python','-m','appl.scaleup',
                        'evaluate','--task',name,'--method','naive_DP','--condition',condition,'--seeds',str(seed),'--gpu',str(gpu)]
                    stdout=(job/'stdout.log').open('w');stderr=(job/'stderr.log').open('w')
                    proc=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
                    atomic(job/'process.json',dict(pid=proc.pid,command=args,started=time.time(),
                        occupancy=identity(gpu),resource_dispatch='supplement_DP'))
                    active[gpu,index]=dict(process=proc,stdout=stdout,stderr=stderr,job=job,
                        task=name,condition=condition,seed=seed)
                    free-=3000
            atomic(out/'status.json',dict(finished=len(done),pending=len(queue),time=time.time(),
                active=[dict(gpu=g,slot=s,task=v['task'],condition=v['condition'],seed=v['seed'],pid=v['process'].pid)
                    for (g,s),v in active.items()]))
            time.sleep(5)
        atomic(out/'completed.json',dict(episodes=done,finished=time.time(),new_scientific_slots=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-index',type=int,default=29)
    parser.add_argument('--last-index',type=int,default=28)
    parser.add_argument('--wait-for-initial',action='store_true')
    parser.add_argument('--predecessor',help='Reuse this queue\'s released slots once it has no pending jobs.')
    args=parser.parse_args()
    main(args.first_index,args.last_index,args.wait_for_initial,args.predecessor)
