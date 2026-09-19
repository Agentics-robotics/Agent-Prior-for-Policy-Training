"""Measured comparison of the additional APPL round and retained original runs."""
from collections import Counter
import csv
import html
import json
import math
import os
from pathlib import Path
import subprocess

from appl.io import ROOT, read, atomic, digest, object_hash
from appl.prior_policies.report import api_accounting
from appl.scaleup.report import audit_episode, video, wilson
from experiments.exp2.analysis.audit_scaleup import audit_api
from .runner import BASE, OLD, NAMES, MODEL, EFFORT, config, original_config, api_identity


def verify_video(root):
    provenance=read(root / 'replay_provenance.json')
    frames=([root / 'initial.png'] if (root / 'initial.png').exists() else [])+sorted(root.glob('frame_*.png'))
    if (root / 'final.png').exists():frames.append(root / 'final.png')
    if provenance['frames'] != {p.name:digest(p) for p in frames}:
        raise ValueError('Video source frames changed')
    path=root / 'replay.mp4'
    if provenance['video_sha256'] != digest(path):
        raise ValueError('Encoded video differs from its provenance')
    response=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,codec_name','-of','json',str(path)],
        capture_output=True,text=True,check=True)
    stream=json.loads(response.stdout)['streams'][0]
    if (stream['width'],stream['height'],stream['codec_name'],int(stream['nb_frames'])) != (128,128,'h264',len(frames)):
        raise ValueError('Video dimensions, codec or frame count mismatch')
    return dict(frames=len(frames),codec='h264',resolution=[128,128],passed=True)


def generate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    freeze_record=read(BASE / 'study_freeze.json')
    freeze=freeze_record['study']
    if object_hash(freeze) != freeze_record['study_sha256']:
        raise ValueError('Study freeze digest mismatch')
    if digest(OLD / 'results.json') != freeze['comparator_results_sha256']:
        raise ValueError('Original comparator report changed')
    original=read(OLD / 'results.json')
    methods=['naive_DP','APPL_5_5_high','APPL_6_xhigh']
    rows=[];audits=[];accounts=[];clips=[];sources=[];policy_rows=[];checks=[];processes=[];reference_training=[]
    for name in NAMES:
        cfg=config(name);contract=read(cfg['completion_contract'])
        reference_root=ROOT / original_config(name)['output']
        for record in sorted((reference_root / 'policies').glob('*/*/training/result.json')):
            trained=read(record)
            reference_training.append(dict(task=name,path=str(record),record_sha256=digest(record),
                optimizer_updates=trained['optimizer_steps'],parameters=trained['trainable_parameters'],
                checkpoint_sha256=trained['checkpoint_sha256'],reused_original_model=True))
        for condition in ('ID','OOD'):
            for method in methods:
                for seed in cfg['evaluation']['seeds' if condition=='ID' else 'ood_seeds']:
                    fresh=method=='APPL_6_xhigh'
                    root=(BASE if fresh else OLD) / name / 'evaluation' / ('naive_DP' if method=='naive_DP' else 'APPL') / condition / str(seed)
                    if fresh:
                        job=BASE / 'jobs/evaluation' / name / condition / str(seed)
                        process=read(job / 'process_result.json')
                        processes.append(dict(task=name,condition=condition,seed=seed,**process))
                        result=read(root / 'result.json') if (root / 'result.json').exists() else None
                        measured=audit_episode(root,contract,result)
                        completed=result is not None
                        if completed and process['returncode'] != 0:
                            raise ValueError('Result exists but its process did not exit cleanly')
                        if not completed and process['returncode'] == 0:
                            raise ValueError('Successful process has no evaluation result')
                        row=dict(task=name,condition=condition,method=method,seed=seed,
                            status=result['status'] if completed else ('interrupted' if root.exists() else 'not_started'),
                            completed=completed,success=result['success'] if completed else None,**measured)
                        for cap in (1500,3000,5000):
                            row['success_by_'+str(cap)]=measured['first_success'] is not None and measured['first_success']<=cap
                        if completed:
                            if result['steps'] > 5000 or (result['success'] and measured['first_success'] != result['steps']):
                                raise ValueError('Evaluation exceeded its physical or success stop')
                            initial=read(root / 'initial_state.json')
                            comparator=OLD / name / 'evaluation/naive_DP' / condition / str(seed) / 'initial_state.json'
                            if initial != read(comparator):
                                raise ValueError('Original and new paired resets differ')
                            trace=[json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
                            receipt=audit_api(root,contract,trace,initial,result,freeze['tasks'][name])
                            receipt['model_identity']=api_identity(root / 'api/journal.sqlite')
                            audits.append(dict(task=name,condition=condition,seed=seed,**receipt))
                        elif (root / 'api/journal.sqlite').exists():
                            api_identity(root / 'api/journal.sqlite',require_consumed=False)
                        account=api_accounting(root / 'api/journal.sqlite')
                        accounts.append(dict(task=name,stage='deployment',condition=condition,seed=seed,**account))
                        if root.exists():
                            path=video(root)
                            if path:
                                clips.append(dict(task=name,condition=condition,seed=seed,
                                    status=row['status'],steps=row['steps'],path=str(path.relative_to(BASE)),
                                    video_sha256=digest(path),validation=verify_video(root)))
                        if completed:
                            sources.append(dict(path=str(root / 'result.json'),sha256=digest(root / 'result.json'),
                                trace_sha256=digest(root / 'trace.jsonl')))
                    else:
                        old_method='naive_DP' if method=='naive_DP' else 'APPL'
                        matches=[r for r in original['episodes'] if (r['task'],r['condition'],r['method'],r['seed']) ==
                                 (name,condition,old_method,seed)]
                        prior,=matches
                        row=dict(prior,method=method)
                    row['reused_original_result']=not fresh
                    row['episode_root']=str(root)
                    rows.append(row)
        accounts.append(dict(task=name,stage='segmentation',**api_accounting(cfg['dataset'] / '_session/journal.sqlite')))
        for p in sorted((cfg['output'] / 'policies').glob('*/*/submission.json')):
            folder=p.parent
            accounts.append(dict(task=name,stage='policy_design',policy=str(folder.relative_to(BASE.resolve())),
                **api_accounting(folder / 'design/journal.sqlite')))
            trained=read(folder / 'training/result.json')
            policy_rows.append(dict(task=name,folder=str(folder.relative_to(BASE.resolve())),
                policy_id=read(folder / 'assignment.json')['policy_id'],
                optimizer_updates=trained['optimizer_steps'],parameters=trained['trainable_parameters'],
                checkpoint_sha256=trained['checkpoint_sha256']))
            for request_path in sorted((folder / 'checks').glob('*/worker_request.json')):
                checked_root=request_path.parent
                request=read(request_path)
                checked=read(checked_root / 'result.json') if (checked_root / 'result.json').exists() else None
                progress=read(checked_root / 'progress.json') if (checked_root / 'progress.json').exists() else {}
                confirmed=checked['optimizer_steps'] if checked else progress.get('step',0)
                checks.append(dict(task=name,policy_id=policy_rows[-1]['policy_id'],
                    path=str(checked_root.relative_to(BASE.resolve())),requested_updates=request['updates'],
                    confirmed_updates=confirmed,passed=checked is not None,
                    unconfirmed_update_upper_bound=0 if checked else request['updates']-confirmed,
                    process=read(checked_root / 'process_result.json') if (checked_root / 'process_result.json').exists() else None))
    groups=[]
    for name in NAMES:
        for condition in ('ID','OOD'):
            for method in methods:
                es=[r for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,method)]
                if len(es)!=30:
                    raise ValueError('Incomplete declared matrix')
                success=sum(e['success'] is True for e in es);completed=sum(e['completed'] for e in es)
                groups.append(dict(task=name,condition=condition,method=method,success=success,
                    completed=completed,planned=30,unknown=30-completed,
                    rate=success/30 if completed==30 else None,
                    wilson95=wilson(success,30) if completed==30 else None,
                    success_by={str(cap):sum(e['success_by_'+str(cap)] for e in es) for cap in (1500,3000,5000)}))
    for journal in sorted((BASE / 'retained_design_attempts').glob('*/*/attempt_*/design/journal.sqlite')):
        task=journal.parents[3].name
        policy=journal.parents[2].name
        api_identity(journal,require_consumed=False)
        accounts.append(dict(task=task,policy=policy,stage='retained_interrupted_policy_design',
            path=str(journal.relative_to(BASE)),**api_accounting(journal)))
    usage=Counter();states=Counter()
    for a in accounts:
        usage.update(a['usage']);states.update(a['states'])
    fresh=[r for r in rows if not r['reused_original_result']]
    paired=[]
    for name in NAMES:
        for condition in ('ID','OOD'):
            new={r['seed']:r for r in fresh if (r['task'],r['condition'])==(name,condition)}
            for reference in methods[:2]:
                old={r['seed']:r for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,reference)}
                outcomes=Counter((old[s]['success'],new[s]['success']) for s in new
                                 if old[s]['completed'] and new[s]['completed'])
                win=outcomes[(False,True)];loss=outcomes[(True,False)];discordant=win+loss
                probability=min(1.,2*sum(math.comb(discordant,i) for i in range(min(win,loss)+1))/2**discordant) if discordant else 1.
                count=sum(outcomes.values())
                paired.append(dict(task=name,condition=condition,reference=reference,complete_pairs=count,
                    both_success=outcomes[(True,True)],both_failure=outcomes[(False,False)],
                    new_only=win,reference_only=loss,paired_difference=(win-loss)/30 if count==30 else None,
                    exact_mcnemar_p=probability,
                    scope='Descriptive, unadjusted complete pairs; previously analyzed layouts, one training seed.'))
    diagnostics=[]
    for path in sorted((BASE / 'incidents').glob('*/*/result.json')):
        value=read(path)
        if 'diagnostic_sha256' in value:
            diagnostics.append(dict(path=str(path.relative_to(BASE)),**value))
    recovery_paths = [BASE / 'incidents/design_recovery_authorization.json',
        BASE / 'incidents/buffer_red_h02_reload/recovery_authorization.json',
        BASE / 'incidents/buffer_red_h02_reload/continuation.json',
        BASE / 'incidents/buffer_red_h02_reload/framework_revision.json',
        BASE / 'incidents/tray_recovery_recording_error/incident.json']
    recovery_paths += sorted((BASE / 'incidents/design_recovery').glob('*.json'))
    recoveries = [dict(path=str(p.relative_to(BASE)), sha256=digest(p), record=read(p))
                  for p in recovery_paths]
    summary=dict(model=MODEL,effort=EFFORT,new_planned=300,new_completed=sum(r['completed'] for r in fresh),
        new_successes=sum(r['success'] is True for r in fresh),new_unknown=sum(not r['completed'] for r in fresh),
        comparisons_are_reused=True,API_states=dict(states),reported_API_usage=dict(usage),
        new_policies=len(policy_rows),formal_training_updates=sum(p['optimizer_updates'] for p in policy_rows),
        reference_policies=len(reference_training),reference_formal_training_updates=sum(p['optimizer_updates'] for p in reference_training),
        bounded_interface_checks=len(checks),interface_confirmed_updates=sum(c['confirmed_updates'] for c in checks),
        interface_unconfirmed_update_upper_bound=sum(c['unconfirmed_update_upper_bound'] for c in checks),
        framework_diagnostic_policy_updates=sum(d['policy_updates'] for d in diagnostics),
        framework_diagnostic_DDPM100_chunks=sum(d['DDPM100_chunks'] for d in diagnostics),
        currency_cost='Provider-reported tokens only; no currency bill asserted. Interrupted calls may have unknown usage.')
    atomic(BASE / 'results.json',dict(summary=summary,groups=groups,episodes=rows,API_accounting=accounts,
        policies=policy_rows,reference_training=reference_training,interface_checks=checks,paired=paired,study_sha256=freeze_record['study_sha256'],
        framework_diagnostics=diagnostics,authorized_recoveries=recoveries,
        report_source_sha256=digest(__file__)))
    atomic(BASE / 'evaluation_audit.json',dict(completed_original_API_decisions_verified=len(audits),
        records=audits,results_and_traces=sources,processes=processes,
        original_comparator_sha256=digest(OLD / 'results.json')))
    atomic(BASE / 'videos.json',dict(count=len(clips),videos=clips))
    with (BASE / 'episodes.csv').open('w') as f:
        keys=['task','condition','method','seed','status','completed','success','steps','first_success',
              'success_by_1500','success_by_3000','success_by_5000','reused_original_result','episode_root']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in rows)
    labels=['DP (reused)','APPL 5.5/high (reused)','APPL 6/xhigh (new)']
    colors=['#8894a5','#d99d38','#167d9a']
    fig,axes=plt.subplots(2,5,figsize=(18,7),sharey=True)
    for ri,condition in enumerate(('ID','OOD')):
        for ci,name in enumerate(NAMES):
            ax=axes[ri,ci]
            for x,method in enumerate(methods):
                g,=[v for v in groups if (v['task'],v['condition'],v['method'])==(name,condition,method)]
                ax.bar(x,g['success']/30,color=colors[x],label=labels[x])
                if g['unknown']:
                    ax.bar(x,g['unknown']/30,bottom=g['success']/30,color='none',edgecolor=colors[x],hatch='///')
                else:
                    lo,hi=g['wilson95'];rate=g['rate']
                    ax.errorbar(x,rate,yerr=[[max(0,rate-lo)],[max(0,hi-rate)]],fmt='none',color='black',capsize=3)
                ax.text(x,1.03,f"{g['success']}/30"+(' + ?' if g['unknown'] else ''),ha='center',fontsize=9)
            ax.set_xticks([0,1,2],['DP','5.5/high','6/xhigh']);ax.set_ylim(0,1.13)
            ax.set_title(name.replace('_',' ')+' / '+condition,fontsize=10)
            ax.grid(axis='y',alpha=.2)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles[:3],legend[:3],loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(BASE / 'comparison.png',dpi=180);plt.close(fig)
    lines=['# Additional APPL: GPT-6 Astra / xhigh','',
        f"New completed outcomes: {summary['new_completed']}/300; unknown: {summary['new_unknown']}.",
        'Original DP and GPT-5.5/high APPL results are reused, not rerun.','',
        '| Task | Split | DP | APPL 5.5/high | APPL 6/xhigh |','| --- | --- | --- | --- | --- |']
    for name in NAMES:
        for condition in ('ID','OOD'):
            cells=[]
            for method in methods:
                g,=[v for v in groups if (v['task'],v['condition'],v['method'])==(name,condition,method)]
                cells.append(f"{g['success']}/30"+(f"; {g['unknown']} unknown" if g['unknown'] else ''))
            lines.append('| '+name+' | '+condition+' | '+' | '.join(cells)+' |')
    lines+=['','![Comparison](comparison.png)','',
        'These are paired follow-up layouts already analyzed in the original study. The new design API received only original training evidence. Both API model and reasoning effort changed; this is not a single-factor reasoning-effort ablation.',
        'Policy counts, architectures and aggregate training compute can differ. Task success uses the unchanged geometric goals with no terminal extras.',
        f"The new library contains {summary['new_policies']} policies with {summary['formal_training_updates']:,} formal updates; the retained reference contains {summary['reference_policies']} policies with {summary['reference_formal_training_updates']:,} historical formal updates. Historical models were not retrained in this study.",
        '', '## Recorded design interruptions and framework repair', '',
        'Three design sessions stopped on HTTP errors before generating policy files. The user explicitly authorized one new attempt per case; previous calls remain charged against the original budgets, and the old sessions remain under retained_design_attempts. These are included in API accounting.',
        'After the tray API successfully submitted, an outer status-recording error stopped its wrapper. The original error and source are retained. The existing API submission and successful check were verified before correcting the outer status; this reconciliation added no API request, candidate edit or training update. The two remaining authorized recoveries then ran separately.',
        'The buffer_red__h02 design exhausted its initial budget after four numerical EMA reload mismatches. The verification compared identical weights with different requires_grad flags. A recorded one-line framework correction matches those flags in the reload check; sampling, training updates, checkpoint selection and deployment behavior are unchanged. All 32 unaffected models had already finished before this correction.',
        'The user authorized up to eight additional API calls and two checks in that same buffer session. It used two calls and one successful check to submit its existing package. Original API responses, tool results and the conversation prefix were verified unchanged; the executor did not edit policy source. The final four policies use the corrected verification. Full receipts and the exact patch are retained under incidents/buffer_red_h02_reload.',
        '', '[Paired video examples](paired_examples.html) · [All new videos](replays.html) · [Policy documents](POLICIES.md) · [Machine-readable results](results.json) · [Execution audit](evaluation_audit.json)', '']
    (BASE / 'REPORT.md').write_text('\n'.join(lines))
    index=['# API-authored policies','', '| Task | Policy | Prior | Handoff | Source | Checkpoint |',
           '| --- | --- | --- | --- | --- | --- |']
    for p in policy_rows:
        f=p['folder']
        index.append(f"| {p['task']} | {p['policy_id']} | [PRIOR]({f}/source/PRIOR.md) | [HANDOFF]({f}/source/HANDOFF.json) | [source]({f}/source/policy.py) | [checkpoint]({f}/training/last.pt) |")
    (BASE / 'POLICIES.md').write_text('\n'.join(index)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>GPT-6 Astra/xhigh APPL replays</title>',
          '<style>body{font:16px system-ui;max-width:1100px;margin:24px auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}article{padding:12px;background:#f3f5f7}</style>',
          '<h1>GPT-6 Astra / xhigh APPL replays</h1><p>Original snapshots, approximately 6x physical speed. Interrupted clips contain only the recorded prefix.</p>',
          '<select id="task" onchange="document.querySelectorAll(\'article\').forEach(x=>x.hidden=this.value!==\'all\'&&x.dataset.task!==this.value)"><option value="all">All tasks</option>']
    page += [f'<option value="{name}">{name}</option>' for name in NAMES]
    page += ['</select><div class="grid">']
    for v in clips:
        page.append(f'<article data-task="{v["task"]}"><h3>{v["task"]} / {v["condition"]} / {v["seed"]}</h3><p>{html.escape(v["status"])} / {v["steps"]} steps</p><video controls preload="none" src="{html.escape(v["path"])}"></video></article>')
    page+=['</div>'];(BASE / 'replays.html').write_text('\n'.join(page))
    paired_page=['<!doctype html><meta charset="utf-8"><title>Paired three-method examples</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}article{padding:12px;background:#f3f5f7}</style>',
        '<h1>Five tasks: paired three-method replays</h1>',
        '<p>Fixed first declared seed for each task and split, regardless of outcome. Original recorded frames only; approximately 6x physical speed. DP and GPT-5.5/high videos are reused.</p>']
    for name in NAMES:
        cfg=config(name)
        for condition in ('ID','OOD'):
            seed=cfg['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][0]
            paired_page.append(f'<h2>{name} / {condition} / {seed}</h2><div class="grid">')
            for method,label in zip(methods,labels):
                row,=[r for r in rows if (r['task'],r['condition'],r['seed'],r['method']) == (name,condition,seed,method)]
                path=Path(row['episode_root']) / 'replay.mp4'
                media=(f'<video controls preload="none" src="{html.escape(os.path.relpath(path,BASE))}"></video>'
                       if path.exists() else '<p>No recorded video available.</p>')
                outcome='unknown' if not row['completed'] else 'success' if row['success'] else 'failure'
                paired_page.append(f'<article><h3>{html.escape(label)}</h3><p>{outcome} / {row["steps"]} steps</p>{media}</article>')
            paired_page.append('</div>')
    (BASE / 'paired_examples.html').write_text('\n'.join(paired_page))
    return summary
