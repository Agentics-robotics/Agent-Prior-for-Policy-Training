"""Audit new outcomes and compare with the three frozen methods."""
from collections import Counter
import csv
import html
import os
from pathlib import Path

from appl.io import atomic, digest, read
from appl.prior_policies.report import api_accounting
from appl.scaleup.report import audit_episode, video, wilson
from experiments.exp2.astra.report import verify_video
from .common import BASE, COMPARISON, METHOD, NAMES, config, folder, verify_preparation

METHODS=['naive_DP','APPL_5_5_high',METHOD,'APPL_6_xhigh']
LABELS=['Naive DP','APPL 5.5','Single prior 6/xhigh','APPL 6/xhigh']


def relative(path): return os.path.relpath(path,BASE)


def generate():
    verify_preparation();frozen=read(BASE/'study_freeze.json');old=read(COMPARISON)
    rows=list(old['episodes']);audits=[];accounts=[];models=[]
    for name in NAMES:
        cfg=config(name);f=folder(name);trained=read(f/'training/result.json')
        if digest(f/'training/last.pt')!=frozen['study']['tasks'][name]['checkpoint_sha256']:
            raise ValueError('Trained policy changed')
        for filename,sha in frozen['study']['tasks'][name]['source_hashes'].items():
            if digest(f/'source'/filename)!=sha: raise ValueError('API submission changed')
        accounts.append(dict(task=name,**api_accounting(f/'design/journal.sqlite')))
        metadata=read(f/'source/pipeline.json')
        models.append(dict(task=name,parameters=trained['trainable_parameters'],
            optimizer_updates=trained['optimizer_steps'],training_seconds=trained['training_elapsed_seconds'],
            prior_summary_API_authored=metadata['prior_summary'],source=str(f/'source'),
            checkpoint_sha256=trained['checkpoint_sha256']))
        for condition in ('ID','OOD'):
            for seed in cfg['evaluation']['seeds' if condition=='ID' else 'ood_seeds']:
                root=BASE/name/'evaluation'/METHOD/condition/str(seed)
                result=read(root/'result.json')
                process=read(BASE/'jobs/evaluation'/name/condition/str(seed)/'process_result.json')
                if process['returncode']!=0: raise ValueError('Evaluation process did not exit cleanly')
                audit=audit_episode(root,read(cfg['completion_contract']),result)
                saved=read(root/'audit.json')
                if not saved['passed'] or digest(root/'trace.jsonl')!=saved['trace_sha256']:
                    raise ValueError('Trace changed after episode audit')
                if (root/'api').exists() or result['deployment_API_calls']!=0:
                    raise ValueError('Single-policy deployment must not call an API')
                video(root);clip=verify_video(root)
                if result['steps']>5000 or (result['success'] and audit['first_success']!=result['steps']):
                    raise ValueError('Episode stopping contract mismatch')
                if not result['success'] and result['steps']!=5000: raise ValueError('Unexplained early failure termination')
                row=dict(task=name,condition=condition,method=METHOD,seed=seed,status=result['status'],
                    completed=True,success=result['success'],**audit,episode_root=str(root),reused_original_result=False)
                for cap in (1500,3000,5000): row['success_by_'+str(cap)]=audit['first_success'] is not None and audit['first_success']<=cap
                rows.append(row);audits.append(dict(task=name,condition=condition,seed=seed,**saved,
                    result_sha256=digest(root/'result.json'),video=clip,video_sha256=digest(root/'replay.mp4')))
    groups=[];paired=[]
    for name in NAMES:
        for condition in ('ID','OOD'):
            for method in METHODS:
                es=[r for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,method)]
                if len(es)!=30: raise ValueError('Comparison matrix incomplete')
                n=sum(e['completed'] for e in es);s=sum(e['success'] is True for e in es)
                groups.append(dict(task=name,condition=condition,method=method,success=s,completed=n,
                    planned=30,unknown=30-n,rate=s/30 if n==30 else None,
                    wilson95=wilson(s,30) if n==30 else None,
                    success_by={str(cap):sum(e['success_by_'+str(cap)] for e in es) for cap in (1500,3000,5000)}))
            fresh={r['seed']:r for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,METHOD)}
            for reference in METHODS:
                if reference==METHOD: continue
                previous={r['seed']:r for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,reference)}
                cells=Counter((previous[s]['success'],fresh[s]['success']) for s in fresh if previous[s]['completed'])
                paired.append(dict(task=name,condition=condition,reference=reference,complete_pairs=sum(cells.values()),
                    single_only=cells[(False,True)],reference_only=cells[(True,False)],
                    both_success=cells[(True,True)],both_failure=cells[(False,False)]))
    fresh=[r for r in rows if r['method']==METHOD];usage=Counter();states=Counter()
    for account in accounts: usage.update(account['usage']);states.update(account['states'])
    summary=dict(new_outcomes=len(fresh),successes=sum(r['success'] for r in fresh),unknown=0,
        success_by_condition={c:sum(r['success'] for r in fresh if r['condition']==c) for c in ('ID','OOD')},
        trained_policies=5,formal_training_updates=sum(m['optimizer_updates'] for m in models),
        design_API_states=dict(states),reported_design_API_usage=dict(usage),deployment_API_requests=0,
        success_by={str(cap):sum(r['success_by_'+str(cap)] for r in fresh) for cap in (1500,3000,5000)},
        comparison_scope='Paired follow-up on previously examined layouts, one training seed. APPL total training compute differs.',
        failure_usage='Unreported transport-error usage remains unknown; no currency bill inferred.')
    atomic(BASE/'results.json',dict(summary=summary,groups=groups,episodes=rows,models=models,
        API_accounting=accounts,paired=paired,comparator_results_sha256=digest(COMPARISON),
        study_sha256=frozen['study_sha256']))
    atomic(BASE/'evaluation_audit.json',dict(passed=True,episodes=audits,original_comparators_unchanged=True))
    with (BASE/'episodes.csv').open('w') as stream:
        keys=['task','condition','method','seed','completed','success','steps','status','episode_root']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in rows)
    render(rows,groups,models,summary)
    atomic(BASE/'completion.json',dict(status='complete',completed_new_trials=300,validated_videos=300,
        **summary,artifacts={n:digest(BASE/n) for n in ('results.json','evaluation_audit.json','episodes.csv',
            'REPORT.md','POLICIES.md','comparison.png','replays.html','paired_examples.html')}))
    return summary


def render(rows,groups,models,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#929aa5','#32936f','#bf6f20','#356bb3']
    fig,axes=plt.subplots(2,5,figsize=(15,6),sharey=True,layout='constrained')
    for j,name in enumerate(NAMES):
        for i,condition in enumerate(('ID','OOD')):
            ax=axes[i,j]
            for k,method in enumerate(METHODS):
                g=next(g for g in groups if (g['task'],g['condition'],g['method'])==(name,condition,method))
                height=g['success']/30
                ax.bar(k,height,color=colors[k],alpha=.55 if g['unknown'] else 1)
                label=f"{g['success']}/30"+(' + ?' if g['unknown'] else '')
                ax.text(k,height+.025,label,ha='center',fontsize=8)
            ax.set_title(name.replace('_',' ')+' / '+condition,fontsize=10)
            ax.set_xticks(range(4),['DP','5.5','Single','6'],fontsize=8);ax.set_ylim(0,1.15);ax.grid(axis='y',alpha=.15)
            if j==0: ax.set_ylabel('Success / planned trials')
    fig.suptitle('Full-task API prior without deployment agent vs frozen comparators\n30 paired layouts per condition; old buffer ID retains one unknown',fontsize=13)
    fig.savefig(BASE/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Single-policy GPT-6 Astra/xhigh baseline','',
        f"Completed: 300/300 new outcomes; {summary['successes']} successes. ID {summary['success_by_condition']['ID']}/150; position OOD {summary['success_by_condition']['OOD']}/150.",'',
        'One API-authored full-task policy per task, twelve original demonstrations, 60,000 updates, final EMA, seed 0, shared full-demo normalization, DDPM100. No segmentation or runtime agent. Same geometric goals and 5,000-step cap.',
        '', '| Task | Condition | DP | APPL 5.5 | Single prior 6/xhigh | APPL 6/xhigh |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        for condition in ('ID','OOD'):
            cells=[]
            for method in METHODS:
                g=next(g for g in groups if (g['task'],g['condition'],g['method'])==(name,condition,method))
                cells.append(f"{g['success']}/30"+(f" ({g['unknown']} unknown)" if g['unknown'] else ''))
            lines.append('| '+name+' | '+condition+' | '+' | '.join(cells)+' |')
    lines+=['','![Comparison](comparison.png)','',
        '## Interpretation and scope','',
        'Versus naive DP, this measures an offline API-designed inductive bias under the same data and optimizer-update budget. Architectures and FLOPs can differ. Versus APPL, decomposition, policy count, total training compute and runtime decisions change together; this is not an isolated runtime-agent ablation. Layouts were previously analyzed and are paired follow-up tests; only one training seed is used. No test evidence was supplied to the design API and no test-driven revision was performed.',
        '',f"Successes by physical step cap: `{summary['success_by']}`. Deployment API requests: **0**. Formal optimizer updates: {summary['formal_training_updates']:,}.",
        '',f"Reported design API usage: `{summary['reported_design_API_usage']}`; request states: `{summary['design_API_states']}`. Interface checks are additional bounded updates recorded separately in each package.",
        '', '[API-authored policy documents](POLICIES.md) · [All new replays](replays.html) · [Four-method first-seed examples](paired_examples.html) · [Audit](evaluation_audit.json) · [Machine-readable results](results.json)']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    lines=['# API-authored full-task policies','','All scientific summaries below are copied from API-authored pipeline metadata.','']
    for model in models:
        source=Path(model['source'])
        lines += [f"## {model['task']}",'',model['prior_summary_API_authored'],'',
            f"Trainable parameters: {model['parameters']:,}; updates: {model['optimizer_updates']:,}.",
            '',f"[PRIOR.md]({relative(source/'PRIOR.md')}) · [policy.py]({relative(source/'policy.py')}) · [Checkpoint]({relative(source.parent/'training/last.pt')})",'']
    (BASE/'POLICIES.md').write_text('\n'.join(lines))
    header='<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px;max-width:1400px}section{margin:24px 0}video{width:300px;image-rendering:auto}.row{display:flex;gap:16px;flex-wrap:wrap}article{width:310px}</style>'
    def clip(row):
        path=html.escape(relative(Path(row['episode_root'])/'replay.mp4'))
        label=html.escape(f"{row['method']} / {row['seed']} / success={row['success']}")
        return f'<article><h4>{label}</h4><video controls preload="none" src="{path}"></video></article>'
    replay=[header,'<h1>Single-policy replays</h1><p>Original camera frames; approximately 6x real time.</p>']
    for name in NAMES:
        for condition in ('ID','OOD'):
            replay += [f'<section><h2>{name} / {condition}</h2><div class="row">']
            replay += [clip(r) for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,METHOD)]
            replay += ['</div></section>']
    (BASE/'replays.html').write_text('\n'.join(replay))
    paired=[header,'<h1>Four-method paired examples</h1><p>First predeclared seed in each task/condition; examples are not selected by outcome.</p>']
    for name in NAMES:
        for condition in ('ID','OOD'):
            seed=config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][0]
            paired += [f'<section><h2>{name} / {condition} / {seed}</h2><div class="row">']
            for method in METHODS:
                paired.append(clip(next(r for r in rows if (r['task'],r['condition'],r['seed'],r['method'])==(name,condition,seed,method))))
            paired+=['</div></section>']
    (BASE/'paired_examples.html').write_text('\n'.join(paired))
