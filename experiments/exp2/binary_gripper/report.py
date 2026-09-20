"""Trace validation and paired reporting for the fixed inference intervention."""
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from appl.io import atomic,read,digest
from .common import BASE,TASKS,METHODS,verify,model_record,episode_root,execute_action


def audit_episode(cell):
    from appl.scaleup.tasks import measure
    record=model_record(cell);root=episode_root(cell);result=read(root/'result.json')
    goal=read(record['goal_contract']);env=read(root/'environment.json');first={};count=0;changed=0;saturated=0
    reference=Path(cell['reference_root'])
    for name,key in [('result.json','reference_result_sha256'),('initial_state.json','reference_initial_sha256'),('trace.jsonl','reference_trace_sha256')]:
        if digest(reference/name)!=cell[key]:raise ValueError('Original evidence changed: '+str(reference/name))
    if read(root/'initial_state.json')!=read(reference/'initial_state.json'):raise ValueError('Initial state differs')
    low=np.asarray(env['action_low'],np.float32);high=np.asarray(env['action_high'],np.float32)
    with (root/'trace.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);count+=1
            if row['step']!=count:raise ValueError('Non-contiguous physical trace')
            raw=np.asarray(row['raw_action'],np.float32);actual=np.asarray(row['action'],np.float32)
            if not np.array_equal(execute_action(raw,low,high),actual):raise ValueError('Executed action differs from declared intervention')
            changed+=int(actual[7]!=raw[7]);saturated+=int(np.any(np.clip(raw,low,high)!=raw))
            metrics=measure(row['state'],goal)
            if metrics!=row['metrics']:raise ValueError('Geometric predicate differs')
            for key,met in metrics.items():
                if met and key not in first:first[key]=count
            if metrics['success'] and count!=result['steps']:raise ValueError('Episode continued after success')
    if count!=result['steps'] or count<1 or count>5000:raise ValueError('Invalid physical step count')
    if not metrics['success'] and count!=5000:raise ValueError('Incomplete failure')
    if result['success']!=metrics['success'] or result['final']!=metrics or result['first_goals']!=first:raise ValueError('Outcome differs from trace')
    if result['changed_gripper_commands']!=changed or result['actuator_saturated_steps']!=saturated:raise ValueError('Action counters differ')
    if result['API_calls']!=0 or result['learned_optimizer_updates']!=0:raise ValueError('Not inference only')
    if cell['method']=='SinglePrior_6_xhigh':
        if read(root/'worker/enforcement.json')['network'] is not False:raise ValueError('API policy network is not disabled')
        if read(root/'worker/closed.json')['returncode']!=0:raise ValueError('Policy worker did not exit cleanly')
    value=dict(passed=True,steps=count,first_success=first.get('success'),first_goals=first,
        checkpoint_sha256=record['checkpoint_sha256'],reference_initial_state_equal=True,
        trace_sha256=digest(root/'trace.jsonl'),result_sha256=digest(root/'result.json'),
        original_evidence_unchanged=True,all_executed_actions_match_rule=True,API_calls=0,learned_optimizer_updates=0)
    atomic(root/'audit.json',value);return value


def video(root):
    path=root/'replay.mp4';frames=[root/'initial.png']+sorted(root.glob('frame_*.png'))+[root/'final.png']
    if not path.exists():
        command=[shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-threads','1',
            '-f','image2pipe','-vcodec','png','-framerate','6','-i','pipe:0',
            '-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for frame in frames:process.stdin.write(frame.read_bytes())
        process.stdin.close();error=process.stderr.read().decode();code=process.wait()
        if code:raise RuntimeError('Video export failed: '+error)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames,width,height,codec_name','-of','json',str(path)],text=True))['streams'][0]
    if int(probe['nb_read_frames'])!=len(frames) or probe['codec_name']!='h264':raise ValueError('Video audit failed')
    atomic(root/'replay_provenance.json',dict(display_fps=6,source_stride_steps=20,control_hz=20,
        note='Approximately 6x simulation speed, including initial and final snapshots.',
        frames={p.name:digest(p) for p in frames},video_sha256=digest(path),ffprobe=probe))
    return path


def write_csv(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def generate(require_complete=False):
    plan=verify();rows=[];cards=[];pairs=[]
    for cell in plan['cells']:
        root=episode_root(cell);complete=(root/'audit.json').exists() and (root/'replay_provenance.json').exists()
        result=read(root/'result.json') if complete else None
        if complete:
            audit=read(root/'audit.json');movie=read(root/'replay_provenance.json')
            if not audit['passed'] or digest(root/'result.json')!=audit['result_sha256'] or digest(root/'trace.jsonl')!=audit['trace_sha256']:raise ValueError('Audited evidence changed')
            if digest(root/'replay.mp4')!=movie['video_sha256']:raise ValueError('Audited replay changed')
        row={k:cell[k] for k in ['index','task','method','condition','seed']}
        row.update(completed=complete,original_success=cell['original_success'],original_steps=cell['original_steps'],
            success=result['success'] if result else None,steps=result['steps'] if result else None,
            first_success=result['first_goals'].get('success') if result else None,
            episode_root=str(root),reference_root=cell['reference_root'])
        for cap in [1500,3000,5000]:row['success_by_'+str(cap)]=bool(row['first_success'] and row['first_success']<=cap)
        rows.append(row)
        if complete:
            title=f"{cell['task']} / {cell['method']} / {cell['condition']} / {cell['seed']} — {'success' if result['success'] else 'failure'} at {result['steps']} steps"
            clip=str((root/'replay.mp4').relative_to(BASE))
            cards.append(f'<article data-task="{cell["task"]}" data-method="{cell["method"]}" data-condition="{cell["condition"]}"><h3>{html.escape(title)}</h3><video controls preload="none" src="{clip}"></video><a href="{root.relative_to(BASE)}/result.json">Result</a></article>')
            if cell['seed']==next(c['seed'] for c in plan['cells'] if c['task']==cell['task'] and c['condition']==cell['condition']):
                old=Path(cell['reference_root'])/'replay.mp4'
                pairs.append(f'<article><h3>{html.escape(title)}</h3><p>Original: {cell["original_success"]}; binary: {result["success"]}</p><div class="pair"><video controls preload="none" src="{old.as_uri()}"></video><video controls preload="none" src="{clip}"></video></div></article>')
    completed=sum(r['completed'] for r in rows)
    if require_complete and completed!=600:raise ValueError(f'Only {completed}/600 audited outcomes')
    groups=[]
    for task in TASKS+['all']:
        for condition in ['ID','OOD']:
            for method in METHODS:
                group=[r for r in rows if (r['task']==task or task=='all') and r['condition']==condition and r['method']==method]
                observed=[r for r in group if r['completed']];n=len(group);done=len(observed)
                g=dict(task=task,condition=condition,method=method,planned=n,completed=done,unknown=n-done,
                    original_success=sum(r['original_success'] for r in group),new_success=sum(r['success'] for r in observed),
                    paired_gains=sum(r['success'] and not r['original_success'] for r in observed),
                    paired_losses=sum(not r['success'] and r['original_success'] for r in observed))
                g['new_rate']=g['new_success']/n if done==n else None
                for cap in [1500,3000,5000]:g['success_by_'+str(cap)]=sum(r['success_by_'+str(cap)] for r in observed)
                groups.append(g)
    write_csv(BASE/'episodes.csv',rows);write_csv(BASE/'summary.csv',groups)
    summary=dict(study=plan['study'],completed=completed,planned=600,unknown=600-completed,groups=groups,
        API_calls=0,learned_optimizer_updates=0,physical_steps=sum(r['steps'] or 0 for r in rows),
        interpretation='Paired post-study inference ablation on already examined layouts; not a fresh untouched test and not a matched corrected comparison against either APPL.',
        plan_sha256=digest(BASE/'plan.json'))
    atomic(BASE/'results.json',summary)
    lines=['# Exp2: inference-only binary-gripper comparison','',f'Validated outcomes: **{completed}/600**. Runtime API requests: **0**. Learned-model updates: **0**.','',
        '## Fixed comparison','',
        'Reuse the ten frozen naive-DP and GPT-6 Astra/xhigh single-prior checkpoints. Each task retains its original thirty ID and thirty position-OOD layouts, diffusion seeds, DDPM100, two-frame observation history, prediction horizon 16, eight executed actions per chunk, physical controller, geometric success predicates and 5,000-step cap.',
        'The only scientific change is final gripper execution: command +1 when the raw predicted gripper is nonnegative, otherwise −1. Seven arm targets keep their original native bound clipping. Raw and executed actions are recorded separately. There is no runtime agent or policy redesign.',
        'This is a developer-authored decoder intervention. It does not modify API-generated policy source. The earlier diagnostics and these layouts were already examined, so this is a paired post-study ablation, not an untouched confirmatory test.','',
        '## Paired results','',
        '| Task | Split | Method | Original successes | Binary successes | Completed | Gains / losses |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | {g['original_success']}/{g['planned']} | {g['new_success']} | {g['completed']}/{g['planned']} | {g['paired_gains']} / {g['paired_losses']} |")
    lines+=['','Gains and losses count paired layouts that change outcome. Incomplete groups report confirmed successes, not a completed rate.',
        '','## Interpretation and limits','',
        'The native controller interprets intermediate gripper values as intermediate finger position targets. Successful demonstrations command only full opening or closure. Binarization removes gradual commanded partial opening, but does not enforce correct grasp timing or guarantee recovery from off-demonstration placements.',
        'The frozen single-prior models share that same execution interface. Their API-designed inductive biases remain intact; deployment and this rerun require no API calls. The prior API design cost is historical and is not charged again here.',
        'Both APPL versions are paused. Their historical scores use continuous gripper execution, so do not present those scores against these corrected baselines as a comparison with identical execution settings. A future matched APPL correction is outstanding and was not run.',
        'One checkpoint/training seed per task is reused. Thirty layouts do not measure variability across training runs. Success is the agreed geometric predicate and does not require final release, clearance, settling, or hold duration.',
        '','## Evidence','',
        '- [Frozen plan, inputs and hashes](plan.json)',
        '- [Per-layout outcomes and original reference paths](episodes.csv)',
        '- [Task/split summary and gains/losses](summary.csv)',
        '- [Machine-readable result](results.json)',
        '- [All new videos](replays.html)',
        '- [First-seed paired videos](paired_examples.html)',
        '- `checks/<task>/<method>`: original-action reproduction before physical testing.',
        '- `<task>/evaluation/<method>/<split>/<seed>`: plan, exact initial state, full raw/executed action and state trace, geometric result, audit, worker security/exit receipts and video provenance.',
        '- `supervisor`: actual GPU admission snapshots, process IDs, stdout/stderr, process exits and completion receipt.',
        '','The storage tree is append-only for physical attempts. Original checkpoints, policy sources and historical evaluation traces are hash-checked. No automatic physical retry is performed.','']
    (BASE/'REPORT.md').write_text('\n'.join(lines))
    head='<!doctype html><meta charset="utf-8"><title>Exp2 binary-gripper evaluation</title><style>body{font:16px system-ui;margin:2rem;background:#f5f7fa}article{padding:1rem;margin:1rem 0;background:white;border-radius:8px}video{width:320px;max-width:46vw;image-rendering:auto}.pair{display:flex;gap:1rem}select{margin:1rem;padding:.5rem}</style><h1>Exp2 binary-gripper evaluation</h1><p>Original versus corrected inference; frozen checkpoints, 0 Runtime API requests. Clips play at approximately 6× simulation speed.</p>'
    controls=''.join('<label>'+key+' <select id="'+key+'"><option value="">All</option>'+''.join('<option>'+v+'</option>' for v in values)+'</select></label>' for key,values in [('task',TASKS),('method',METHODS),('condition',['ID','OOD'])])
    script='<script>for(const s of document.querySelectorAll("select"))s.onchange=()=>{for(const a of document.querySelectorAll("article"))a.hidden=["task","method","condition"].some(k=>document.getElementById(k).value&&a.dataset[k]!==document.getElementById(k).value)}</script>'
    (BASE/'replays.html').write_text(head+controls+''.join(cards)+script)
    (BASE/'paired_examples.html').write_text(head+'<p>Each pair: original left, binary gripper right. The first planned seed per task/split/method is shown without outcome selection.</p>'+''.join(pairs))
    if completed==600:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,5,figsize=(16,7),sharey=True)
        for row,condition in enumerate(['ID','OOD']):
            for col,task in enumerate(TASKS):
                ax=axes[row,col]
                for i,method in enumerate(METHODS):
                    g=next(g for g in groups if g['task']==task and g['condition']==condition and g['method']==method)
                    for offset,key,color in [(-.18,'original_success','#b5bec9'),(.18,'new_success','#167b98')]:
                        ax.bar(i+offset,g[key]/30,width=.32,color=color,label=('Original' if offset<0 else 'Binary') if i==0 else None)
                        ax.text(i+offset,g[key]/30+.025,str(g[key]),ha='center',fontsize=9)
                ax.set_title(task.replace('_',' ')+' / '+condition,fontsize=10);ax.set_xticks([0,1],['Naive DP','Single prior']);ax.set_ylim(0,1.12);ax.grid(axis='y',alpha=.2)
                if col==0:ax.set_ylabel('Success rate (count / 30)')
        axes[0,0].legend();fig.tight_layout();fig.savefig(BASE/'comparison.png',dpi=180);plt.close(fig)
        with (BASE/'REPORT.md').open('a') as stream:stream.write('\n![Paired comparison](comparison.png)\n')
        atomic(BASE/'completion.json',dict(completed=time.time(),episodes=600,validated_videos=600,unknown=0,
            API_calls=0,learned_optimizer_updates=0,physical_steps=summary['physical_steps'],
            framework_and_checkpoints_unchanged=True,plan_sha256=digest(BASE/'plan.json'),
            results_sha256=digest(BASE/'results.json'),note='Worker exits are separately recorded in supervisor/completed.json.'))
    print(dict(completed=completed,planned=600,output=str(BASE)),flush=True)
    return summary
