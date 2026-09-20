"""Five-task scientific comparison, raw accounting and replay index."""
from collections import Counter
import csv
import html
import json
import math
import shutil
import subprocess
from ..io import read,atomic,digest
from ..prior_policies.report import api_accounting
from .protocol import BASE,NAMES,config
from .tasks import measure


def wilson(success,total):
    if not total:return [0.,1.]
    z=1.959963984540054;p=success/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-half,center+half]


def audit_episode(root,contract,result):
    steps=0
    last=measure(read(root/'initial_state.json'),contract) if (root/'initial_state.json').exists() else None
    first=0 if last and last['success'] else None
    goals={key:0 for key,met in (last or {}).items() if met}
    if (root/'trace.jsonl').exists():
        with (root/'trace.jsonl').open() as f:
            for line in f:
                row=json.loads(line);steps+=1
                if row['step']!=steps:raise ValueError('Non-contiguous physical trace')
                metrics=measure(row['state'],contract)
                if metrics!=row['metrics']:raise ValueError('Recorded geometric predicate mismatch')
                if len(row['action'])!=8 or not all(math.isfinite(v) for v in row['action']):raise ValueError('Invalid physical action')
                for key,met in metrics.items():
                    if met and key not in goals:goals[key]=steps
                if metrics['success'] and first is None:first=steps
                last=metrics
    if result:
        if result['steps']!=steps or result['success']!=(first is not None) or result['final']!=last:
            raise ValueError('Trial result differs from its physical trace')
    return dict(steps=steps,first_success=first,first_goals=goals)


def video(root):
    path=root/'replay.mp4'
    if path.exists():return path
    frames=([root/'initial.png'] if (root/'initial.png').exists() else [])+sorted(root.glob('frame_*.png'))
    if (root/'final.png').exists():frames.append(root/'final.png')
    if not frames:return None
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:raise RuntimeError('ffmpeg is required for browser-compatible replay export')
    args=[ffmpeg,'-hide_banner','-loglevel','error','-f','image2pipe','-vcodec','png','-framerate','6',
        '-i','pipe:0','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    process=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for frame in frames:process.stdin.write(frame.read_bytes())
    process.stdin.close();error=process.stderr.read().decode();code=process.wait()
    if code:raise RuntimeError('Replay export failed: '+error)
    atomic(root/'replay_provenance.json',dict(display_fps=6,source_stride_steps=20,control_hz=20,
        note='Approximately 6x real time; initial/final snapshots are also included.',
        frames={p.name:digest(p) for p in frames},video_sha256=digest(path)))
    return path


def generate(export_video=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];groups={};accounts=[];replays=[]
    for name in NAMES:
        cfg=config(name);contract=read(cfg['completion_contract'])
        for condition in ('ID','OOD'):
            for method in ('naive_DP','APPL'):
                group=[]
                for seed in cfg['evaluation']['seeds' if condition=='ID' else 'ood_seeds']:
                    root=BASE/name/'evaluation'/method/condition/str(seed)
                    result=read(root/'result.json') if (root/'result.json').exists() else None
                    audit=audit_episode(root,contract,result)
                    state=result['status'] if result else ('interrupted' if root.exists() else 'not_started')
                    row=dict(task=name,condition=condition,method=method,seed=seed,status=state,
                        completed=bool(result),success=result['success'] if result else None,**audit)
                    for cap in (1500,3000,5000):row['success_by_'+str(cap)]=audit['first_success'] is not None and audit['first_success']<=cap
                    group.append(row);rows.append(row)
                    if method=='APPL':accounts.append(dict(task=name,condition=condition,seed=seed,**api_accounting(root/'api/journal.sqlite')))
                    if export_video and root.exists():
                        path=video(root)
                        if path:replays.append(dict(**row,path=str(path.relative_to(BASE))))
                completed=sum(r['completed'] for r in group);success=sum(r['success'] is True for r in group)
                groups[(name,condition,method)]=dict(completed=completed,success=success,planned=30,
                    unknown=30-completed,rate=success/30 if completed==30 else None,
                    wilson95=wilson(success,30) if completed==30 else None,
                    success_by={str(cap):sum(r['success_by_'+str(cap)] for r in group) for cap in (1500,3000,5000)})
    paired=[]
    for name in NAMES:
        for condition in ('ID','OOD'):
            dp={r['seed']:r for r in rows if r['task']==name and r['condition']==condition and r['method']=='naive_DP'}
            ap={r['seed']:r for r in rows if r['task']==name and r['condition']==condition and r['method']=='APPL'}
            counts=Counter()
            for seed in dp:
                left=BASE/name/'evaluation/naive_DP'/condition/str(seed)/'initial_state.json'
                right=BASE/name/'evaluation/APPL'/condition/str(seed)/'initial_state.json'
                if left.exists() and right.exists() and read(left)!=read(right):raise ValueError('Paired reset mismatch')
                if dp[seed]['completed'] and ap[seed]['completed']:counts[(dp[seed]['success'],ap[seed]['success'])]+=1
            win=counts[(False,True)];loss=counts[(True,False)];n=win+loss
            p=min(1.,2*sum(math.comb(n,i) for i in range(min(win,loss)+1))/2**n) if n else 1.
            paired.append(dict(task=name,condition=condition,complete_pairs=sum(counts.values()),
                both_success=counts[(True,True)],both_failure=counts[(False,False)],APPL_only=win,DP_only=loss,
                paired_difference=(win-loss)/30 if sum(counts.values())==30 else None,exact_mcnemar_p=p,
                p_value_scope='Descriptive, unadjusted, available complete pairs; no policy selection.'))
    with (BASE/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='first_goals']);writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k!='first_goals'} for r in rows)
    summaries=[dict(task=k[0],condition=k[1],method=k[2],**v) for k,v in groups.items()]
    costs=[]
    for name in NAMES:
        if name=='drawer_exchange':continue
        cfg=config(name)
        costs.append(dict(task=name,stage='segmentation',**api_accounting(cfg['dataset']/'_session/journal.sqlite')))
        for path in (cfg['output']/'policies').glob('*/*/design/journal.sqlite'):
            costs.append(dict(task=name,stage='policy_design',policy=str(path.parent.parent.relative_to(cfg['output'])),**api_accounting(path)))
    usage=Counter();calls=Counter()
    for a in accounts+costs:usage.update(a['usage']);calls.update(a['states'])
    training=[]
    for path in BASE.glob('*/naive_DP/training/result.json'):
        training.append(dict(path=str(path),owner='developer_naive',**read(path)))
    for path in BASE.glob('*/policies/*/*/training/result.json'):
        training.append(dict(path=str(path),owner='Runtime_API_prior',**read(path)))
    checks=[]
    for p in BASE.glob('*/policies/*/*/checks/*/worker_request.json'):
        folder=p.parent;request=read(p)
        process=read(folder/'process_result.json') if (folder/'process_result.json').exists() else None
        checked=read(folder/'result.json') if (folder/'result.json').exists() else None
        progress=read(folder/'progress.json') if (folder/'progress.json').exists() else {}
        confirmed=checked['optimizer_steps'] if checked else progress.get('step',0)
        checks.append(dict(path=str(folder),requested_updates=request['updates'],confirmed_updates=confirmed,
            passed=checked is not None,process=process,result=checked,
            unconfirmed_update_upper_bound=0 if checked else request['updates']-confirmed,
            accounting='Failed checks retain their own receipts; requested updates are not assumed completed.'))
    deployment_checks={p.parent.parent.name:read(p) for p in BASE.glob('*/inference_checks/summary.json')}
    result=dict(groups=summaries,paired=paired,episodes=rows,completed=sum(r['completed'] for r in rows),planned=600,
        all_completed=all(r['completed'] for r in rows),api=dict(states=dict(calls),usage=dict(usage),evaluation=accounts,design=costs,
        currency_cost='Not asserted: provider billing/rates unavailable; exact reported tokens retained. Interrupted calls may have unknown usage.'),
        training=training,interface_checks=checks,deployment_checks=deployment_checks)
    atomic(BASE/'results.json',result)
    fig,axes=plt.subplots(2,5,figsize=(16,7),sharey=True)
    for row,condition in enumerate(('ID','OOD')):
        for col,name in enumerate(NAMES):
            ax=axes[row,col]
            for x,method,color in [(0,'naive_DP','#8d99ae'),(1,'APPL','#247ba0')]:
                g=groups[(name,condition,method)]
                if g['rate'] is not None:
                    rate=g['rate'];lo,hi=g['wilson95'];ax.bar(x,rate,color=color,width=.65)
                    ax.errorbar(x,rate,yerr=[[max(0,rate-lo)],[max(0,hi-rate)]],fmt='none',color='black',capsize=3)
                    ax.text(x,min(.99,hi+.04),f"{g['success']}/30",ha='center',fontsize=10)
                else:ax.text(x,.35,f"{g['completed']}/30\ncompleted",ha='center',fontsize=9)
            ax.set_xticks([0,1],['DP','APPL']);ax.set_ylim(0,1.12);ax.set_title(name.replace('_',' ')+' / '+condition,fontsize=10)
            ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylabel('Success rate (95% Wilson interval)')
    fig.tight_layout();fig.savefig(BASE/'comparison.png',dpi=180);plt.close(fig)
    lines=['# Five-task DP versus APPL','',f"Completed outcomes: {result['completed']}/600. Unknown/interrupted outcomes remain separate.",'',
        'Both methods use paired initial observations, DDPM100 / execution 8, and the same geometric goals with a 5000-step cap.',
        'API sees no total/remaining episode budget. Each trained library was frozen before test. Training is not compute-matched: APPL trains multiple policies.','',
        '| Task | Condition | DP successes / 30 | APPL successes / 30 | Unknown DP / APPL |',
        '| --- | --- | ---: | ---: | ---: |']
    for name in NAMES:
        for condition in ('ID','OOD'):
            d=groups[(name,condition,'naive_DP')];a=groups[(name,condition,'APPL')]
            lines.append(f"| {name} | {condition} | {d['success']} | {a['success']} | {d['unknown']} / {a['unknown']} |")
    lines+=['','Success counts with unknown outcomes are confirmed successes, not complete success-rate estimates.',
        'Each task has one training seed and twelve demonstrations; these intervals concern test layouts, not training-seed variability.',
        'Environment and diffusion seeds are fixed; API generation is not seeded. One API decision trajectory is tested per layout.',
        'Simulation pauses during API calls. API latency and elapsed time are recorded separately; physical-step success does not establish real-time deployment.',
        'Geometric success can precede release or stable rest. Position OOD keeps the task and dynamics fixed; it does not test new objects or robots.',
        '','![Comparison](comparison.png)','','[All outcomes](episodes.csv) · [Raw summary and costs](results.json) · [Replay browser](replays.html) · [Data validation](data_validation.json)','']
    (BASE/'REPORT.md').write_text('\n'.join(lines))
    cards=[]
    for r in replays:
        label=f"{r['task']} / {r['method']} / {r['condition']} / seed {r['seed']} / {r['status']} / {r['steps']} steps"
        cards.append(f'<article data-task="{r["task"]}"><h3>{html.escape(label)}</h3><video controls preload="none" src="{html.escape(r["path"])}"></video></article>')
    options=''.join(f'<option>{n}</option>' for n in NAMES)
    (BASE/'replays.html').write_text('<!doctype html><meta charset="utf-8"><title>Five-task policy replays</title>'
        '<style>body{font-family:system-ui;margin:24px;background:#f2f4f7}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{padding:12px;background:white}video{width:100%;image-rendering:pixelated}h3{font-size:14px}</style>'
        '<h1>DP / APPL replays</h1><p>Original 128×128 camera snapshots, approximately 6× real time. No action interpolation.</p>'
        '<select onchange="document.querySelectorAll(\'article\').forEach(e=>e.hidden=this.value!==\'All\'&&e.dataset.task!==this.value)"><option>All</option>'+options+'</select><main>'+''.join(cards)+'</main>')
    return dict(completed=result['completed'],planned=600,all_completed=result['all_completed'],report=str(BASE/'REPORT.md'))


if __name__=='__main__':generate()
