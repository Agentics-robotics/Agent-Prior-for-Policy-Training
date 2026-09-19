"""Build paper evidence from validated results; no API, training or simulator calls."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np

from appl.io import ROOT, atomic, digest, object_hash, read
from appl.prior_policies.data import configuration, catalog
from appl.scaleup.protocol import naive_paths
from appl.scaleup.report import wilson
from experiments.exp2.single_policy.common import BASE, COMPARISON, METHOD, NAMES, folder

OUT=Path(__file__).parent
METHODS=['naive_DP','SinglePrior_6_xhigh','APPL_5_5_high','APPL_6_xhigh']
LABELS={'naive_DP':'Naive DP','SinglePrior_6_xhigh':'Single prior 6/xhigh',
        'APPL_5_5_high':'APPL 5.5/high','APPL_6_xhigh':'APPL 6/xhigh'}
TASK_LABELS={'drawer_exchange':'Drawer exchange','two_block_sort':'Two-block sorting',
    'buffer_swap':'Buffer exchange','unstack_sort':'Unstack and sort','tray_pack':'Tray packing'}


def repo_path(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    try:return str(p.relative_to(ROOT))
    except ValueError:pass
    for directory in ('runs/exp2','data/exp2'):
        try:return str(Path(directory)/p.resolve().relative_to((ROOT/directory).resolve()))
        except ValueError:pass
    raise ValueError('Artifact lies outside the repository data/runtime mappings: '+str(p))


def link(path,label):
    original=repo_path(path)
    # Prefer an unchanged bundled copy when available, so the main narrative is
    # usable after extracting the archive away from the repository.
    for section in ('supporting_reports','protocols'):
        for row in read(OUT/section/'index.json')['files']:
            if row['source']==original:
                target=row['bundle_file'] if section=='protocols' else section+'/'+row['bundle_file']
                return '['+label+']('+target+')'
    return '['+label+']('+os.path.relpath(ROOT/original,OUT)+')'


def table(path,rows):
    if not rows:raise ValueError('Empty table: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def prepare():
    """Export training-only metadata and fixed layouts without requiring results."""
    tasks=[];layouts=[];sources=[]
    for name in NAMES:
        cfg=read(ROOT/f'experiments/exp2/configs/scaleup/{name}.json')
        spec=read(ROOT/f'runs/exp2/M1_scaleup/{name}/task.json')
        for source,destination in [(ROOT/cfg['completion_contract'],OUT/'contracts'/(name+'.json')),
            (ROOT/f'runs/exp2/M1_scaleup/{name}/task.json',OUT/'task_definitions'/(name+'.json'))]:
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
            assert digest(source)==digest(destination)
        norm=cfg['normalization'];lengths=[]
        for identifier in norm['train_ids']:
            p=ROOT/norm['directory']/(identifier+'.json');episode=read(p)
            assert len(episode['observations'])==len(episode['actions'])+1
            lengths.append(len(episode['actions']))
            sources.append(dict(task=name,trajectory_id=identifier,path=repo_path(p),sha256=digest(p),
                actions=len(episode['actions']),observations=len(episode['observations'])))
        assert len(lengths)==12
        tasks.append(dict(task=name,label=TASK_LABELS[name],demonstrations=12,
            total_actions=sum(lengths),minimum_actions=min(lengths),maximum_actions=max(lengths),
            mean_actions=float(np.mean(lengths)),mean_duration_sim_seconds=float(np.mean(lengths))/20,
            train_ids=norm['train_ids'],ID_seeds=cfg['evaluation']['seeds'],OOD_seeds=cfg['evaluation']['ood_seeds'],
            ID_red_half_width_m=.012,ID_blue_half_width_m=.015 if name=='drawer_exchange' else .012,
            OOD_forced_axis_min_m=.022,OOD_max_m=.04,
            shared_block_XY_offset=name=='unstack_sort',
            goal_contract=cfg['completion_contract'],goal_contract_sha256=digest(ROOT/cfg['completion_contract']),
            task_definition=f'runs/exp2/M1_scaleup/{name}/task.json',
            description=spec.get('description','Open the drawer, put red on its pad, and put blue inside the drawer.')))
        frozen=read(ROOT/'runs/exp2/M1_scaleup/study_freeze.json')['study']['tasks'][name]
        for condition in ('ID','OOD'):
            for row in frozen['layouts'][condition]:
                layouts.append(dict(task=name,condition=condition,seed=row['seed'],**row['layout']))
    table(OUT/'tables/tasks.csv',tasks);table(OUT/'tables/training_trajectories.csv',sources)
    table(OUT/'tables/layouts.csv',layouts)
    atomic(OUT/'tasks.json',dict(tasks=tasks,training_sources=sources,paired_layouts=layouts))
    return tasks


def models():
    rows=[];segmentations=[]
    def add(name,method,f,source,policy_id,owner):
        trained=read(f/'training/result.json')
        row=dict(task=name,method=method,policy_id=policy_id,owner=owner,folder=repo_path(f),
            checkpoint=repo_path(f/'training/last.pt'),checkpoint_sha256=trained['checkpoint_sha256'],
            optimizer_updates=trained['optimizer_steps'],parameters=trained['trainable_parameters'],
            training_seconds=trained['training_elapsed_seconds'],training_receipt=repo_path(f/'training/result.json'))
        if source:
            row['source']=repo_path(source)
            if owner=='Runtime API':
                metadata=read(source/'pipeline.json')
                row.update(prior_document=repo_path(source/'PRIOR.md'),
                    prior_summary_API_authored=metadata['prior_summary'],
                    source_version=read(f/'submission.json')['version'],
                    authorship_audit=repo_path(f/'authorship_audit.json'))
                assert read(f/'authorship_audit.json')['passed']
                if (source/'HANDOFF.json').exists():row['handoff_document']=repo_path(source/'HANDOFF.json')
                card=OUT/'policy_cards'/method/name/policy_id;card.mkdir(parents=True,exist_ok=True)
                for filename,sha in read(f/'submission.json')['files'].items():
                    assert digest(source/filename)==sha
                    shutil.copyfile(source/filename,card/filename)
                    assert digest(card/filename)==sha
                row['bundled_API_source']=str(card.relative_to(OUT))
        rows.append(row)
    for name in NAMES:
        source,checkpoint=naive_paths(name)
        add(name,'naive_DP',checkpoint.parent.parent,source,'naive_DP','Developer baseline')
        f=folder(name);add(name,METHOD,f,f/'source',METHOD,'Runtime API')
        for method,dirname in [('APPL_5_5_high','scaleup'),('APPL_6_xhigh','astra_xhigh')]:
            cfg=configuration(ROOT/f'experiments/exp2/configs/{dirname}/{name}.json')
            entries=catalog(cfg);manifest=read(cfg['dataset']/'manifest.json')
            for entry in entries:
                f=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
                add(name,method,f,f/'source',entry['policy_id'],'Runtime API')
            segmentations.append(dict(task=name,method=method,skills=len(manifest['datasets']),
                policies=len(entries),**manifest['checks'],manifest=repo_path(cfg['dataset']/'manifest.json')))
    expected={'naive_DP':5,METHOD:5,'APPL_5_5_high':51,'APPL_6_xhigh':36}
    assert Counter(r['method'] for r in rows)==expected
    table(OUT/'POLICY_INDEX.csv',rows);table(OUT/'tables/segmentation.csv',segmentations)
    atomic(OUT/'models.json',dict(models=rows,segmentations=segmentations))
    lines=['# API-authored prior catalog','','Each summary below is copied unchanged from the API-authored pipeline metadata. It describes the intended inductive bias, not a proven mechanism or observed success. Source and prior links point to byte-identical bundled submissions.','']
    for method in METHODS:
        if method=='naive_DP':continue
        lines += ['## '+LABELS[method],'']
        for name in NAMES:
            lines += ['### '+TASK_LABELS[name],'']
            for row in rows:
                if (row['method'],row['task'])!=(method,name):continue
                source=row['bundled_API_source']
                lines += ['#### '+row['policy_id'],'',row['prior_summary_API_authored'],'',
                    f"Parameters: {row['parameters']:,}; formal updates: {row['optimizer_updates']:,}.",'',
                    f'[API prior document]({source}/PRIOR.md) · [Exact source]({source}/policy.py) · [Pipeline]({source}/pipeline.json)','']
    (OUT/'PRIOR_CATALOG.md').write_text('\n'.join(lines)+'\n')
    return rows,segmentations


def plots(groups,episodes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    colors=['#777f8e','#bd7225','#32956e','#3c65aa']
    fig,axes=plt.subplots(2,5,figsize=(15.5,6.5),sharey=True)
    for i,condition in enumerate(('ID','OOD')):
        for j,name in enumerate(NAMES):
            ax=axes[i,j]
            for k,method in enumerate(METHODS):
                g=next(g for g in groups if (g['task'],g['condition'],g['method'])==(name,condition,method))
                y=g['success']/30;top=y
                ax.bar(k,y,width=.72,color=colors[k])
                if g['unknown']:
                    ax.bar(k,g['unknown']/30,bottom=y,color='white',edgecolor='black',hatch='///',width=.72)
                    top+=g['unknown']/30
                else:
                    lo,hi=wilson(g['success'],30);top=hi
                    ax.errorbar(k,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='none',color='black',capsize=2,lw=.8)
                ax.text(k,top+.025,f"{g['success']}/30"+(' + ?' if g['unknown'] else ''),ha='center',fontsize=8)
            ax.set_xticks(range(4),['DP','Single','5.5','6']);ax.set_ylim(0,1.18)
            ax.set_title(TASK_LABELS[name]+' / '+condition,fontsize=10)
            ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);ax.grid(axis='y',alpha=.15)
            if j==0:ax.set_ylabel('Success among planned layouts')
    fig.legend([Patch(facecolor=c) for c in colors],[LABELS[m] for m in METHODS],loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'main_results.{ext}',dpi=240)
    plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(12.5,3))
    for ax,name in zip(axes,NAMES):
        row=next(r for r in episodes if r['task']==name and r['condition']=='ID' and r['method']=='naive_DP')
        p=Path(row['episode_root'])/'initial.png'
        ax.imshow(Image.open(p));ax.set_title(TASK_LABELS[name],fontsize=10);ax.axis('off')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(dest/f'tasks.{ext}',dpi=240)
    plt.close(fig)
    pipeline_figure()


def pipeline_figure():
    """Protocol diagram only; boxes describe executed roles, not outcome claims."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,7.5));ax.set_xlim(0,14);ax.set_ylim(0,7.5);ax.axis('off')
    colors={'framework':'#e6eef6','API':'#f8e8d4','learned':'#e1efe5'}
    columns=[(0.05,2.2),(2.8,2.4),(5.75,2.65),(9.0,2.65),(12.2,1.75)]
    headings=['Training input','Offline design','Trained policies','Deployment','Measurement']
    for (x,w),title in zip(columns,headings):
        ax.text(x+w/2,6.85,title,ha='center',va='center',fontsize=12,fontweight='bold')
    rows=[
        ('Naive DP',[
            ('12 full trajectories','framework'),('Fixed conditional\ndiffusion U-Net','framework'),
            ('1 policy per task\n60,000 updates','learned'),('Same policy repeatedly\nNo runtime API','learned')]),
        ('Single prior 6/xhigh',[
            ('Same 12 full\ntrajectories','framework'),('API designs 1 candidate\nFeatures / structure / loss','API'),
            ('1 policy per task\n60,000 updates','learned'),('Same policy repeatedly\nNo runtime API','learned')]),
        ('APPL 5.5/high',[
            ('Same 12 trajectories\nAPI-defined overlap','framework'),('API cuts, priors, code\nand handoff documents','API'),
            ('Independent skill policies\n20,000 updates each\n51 across five tasks','learned'),
            ('API chooses policy,\nduration and stop rules\nLearned policies act','API')]),
        ('APPL 6/xhigh',[
            ('Same 12 trajectories\nNew API-defined overlap','framework'),('New API cuts, priors, code\nand handoff documents','API'),
            ('Independent skill policies\n20,000 updates each\n36 across five tasks','learned'),
            ('API chooses policy,\nduration and stop rules\nLearned policies act','API')]),
    ]
    for row_index,(label,cells) in enumerate(rows):
        y=5.5-row_index*1.42
        ax.text(.05,y+1.1,label,fontsize=11,fontweight='bold')
        for i,((x,w),(text_value,role)) in enumerate(zip(columns,cells)):
            ax.add_patch(FancyBboxPatch((x,y),w,.92,boxstyle='round,pad=0.025',
                facecolor=colors[role],edgecolor='#88939d',linewidth=.8))
            ax.text(x+w/2,y+.46,text_value,ha='center',va='center',fontsize=9.5)
            right=columns[i+1][0]
            ax.add_patch(FancyArrowPatch((x+w+.06,y+.46),(right-.07,y+.46),
                arrowstyle='-|>',mutation_scale=11,color='#63717e',linewidth=.9))
    x,w=columns[-1]
    ax.add_patch(FancyBboxPatch((x,1.15),w,5.26,boxstyle='round,pad=0.025',
        facecolor=colors['framework'],edgecolor='#88939d',linewidth=.8))
    ax.text(x+w/2,3.78,'Same paired\nreset states\n\n30 ID + 30 OOD\nper task\n\nGeometric\nsuccess\n\nPrivate cap:\n5,000 steps',ha='center',va='center',fontsize=10)
    ax.legend([Patch(facecolor=colors[k],edgecolor='#88939d') for k in colors],
        ['Framework / data','API-authored decisions','Learned policy'],loc='lower center',
        ncol=3,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=10)
    ax.text(7,.55,'All methods: two causal observations, DDPM100, 16-action prediction / 8-action execution.\nOne training replicate; model count and total compute differ across methods.',
        ha='center',va='center',fontsize=10)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'protocol.{ext}',dpi=220)
    plt.close(fig)


def design_evidence(inventory,episodes):
    prompts=[]
    def prompt(path,stage,task,method,policy_id=''):
        request=read(path)
        # Exact executed system instructions/tool schema; omit the large native
        # conversation, whose unchanged original remains at its indexed source.
        selected={k:request[k] for k in ('model','instructions','tools','reasoning','max_output_tokens',
            'parallel_tool_calls','tool_choice','store','include') if k in request}
        sha=object_hash(selected);dest=OUT/'prompts'/f'{sha}.json';atomic(dest,selected)
        prompts.append(dict(stage=stage,task=task,method=method,policy_id=policy_id,
            model=request['model'],effort=request.get('reasoning',{}).get('effort'),
            source_request=repo_path(path),source_request_sha256=digest(path),
            bundled_request_contract=str(dest.relative_to(OUT)),selected_contract_sha256=sha))
    for row in inventory:
        if row['owner']=='Runtime API':
            prompt(ROOT/row['folder']/'design/api/0001.request.json','policy_design',row['task'],row['method'],row['policy_id'])
    for name in NAMES:
        for method,directory in [('APPL_5_5_high','scaleup'),('APPL_6_xhigh','astra_xhigh')]:
            cfg=configuration(ROOT/f'experiments/exp2/configs/{directory}/{name}.json')
            manifest=read(cfg['dataset']/'manifest.json')
            root=OUT/'segmentations'/method/name;root.mkdir(parents=True,exist_ok=True)
            plan=cfg['dataset']/'_session/plans'/(manifest['plan_hash']+'.json')
            shutil.copyfile(plan,root/'plan.json');shutil.copyfile(cfg['dataset']/'manifest.json',root/'manifest.json')
            for item in manifest['datasets']:
                shutil.copyfile(cfg['dataset']/item['heuristic'],root/(item['skill_id']+'.md'))
            prompt(cfg['dataset']/'_session/api/0001.request.json','segmentation',name,method)
            episode=next(e for e in episodes if (e['task'],e['method'],e['condition'])==(name,method,'ID'))
            prompt(Path(episode['episode_root'])/'api/api/0001.request.json','deployment_first_ID_example',name,method)
    table(OUT/'PROMPT_INDEX.csv',prompts)


def interface_accounting(inventory):
    rows=[]
    for model in inventory:
        if model['owner']!='Runtime API':continue
        root=ROOT/model['folder']/'checks'
        for check in sorted(root.iterdir()):
            if not check.is_dir():continue
            request=read(check/'worker_request.json');limit=request['updates']
            result=read(check/'result.json') if (check/'result.json').exists() else None
            progress=read(check/'progress.json') if (check/'progress.json').exists() else None
            process=read(check/'process_result.json') if (check/'process_result.json').exists() else None
            confirmed=result['optimizer_steps'] if result else (progress['step'] if progress else 0)
            assert 0<=confirmed<=limit
            rows.append(dict(task=model['task'],method=model['method'],policy_id=model['policy_id'],
                check=repo_path(check),passed=bool(result and process and process['returncode']==0),
                confirmed_optimizer_updates=confirmed,optimizer_update_upper_bound=limit,
                uncertainty_upper_bound=limit-confirmed))
    table(OUT/'tables/API_interface_checks.csv',rows)
    return rows


def diagnostics(episodes):
    from experiments.exp2.analysis.analyze_outcomes import analyze
    old=read(ROOT/'experiments/exp2/analysis/outcome_diagnostics.json')['episodes']
    cache={(r['task'],r['condition'],r['seed'],'APPL_5_5_high' if r['method']=='APPL' else r['method']):r for r in old}
    norms={n:read(BASE/n/'normalization.json')['normalizer'] for n in NAMES}
    assert norms['drawer_exchange']==read(ROOT/'runs/exp2/m0/training/result.json')['spec']['normalizer']
    rows=[]
    for episode in episodes:
        if not episode['completed']:continue
        key=(episode['task'],episode['condition'],episode['seed'],episode['method'])
        root=Path(episode['episode_root'])
        if key in cache:
            row=dict(cache[key]);assert row['result_sha256']==digest(root/'result.json')
            assert row['trace_sha256']==digest(root/'trace.jsonl')
        else:row=analyze(root,norms[episode['task']])
        row['method']=episode['method'];row['episode_root']=repo_path(root)
        rows.append(row)
        if len(rows)%100==0:print(dict(paper_diagnostics_complete=len(rows),planned=sum(r['completed'] for r in episodes)),flush=True)
    assert len(rows)==sum(r['completed'] for r in episodes)
    groups=[]
    for name in NAMES:
        for condition in ('ID','OOD'):
            for method in METHODS:
                selected=[r for r in rows if (r['task'],r['condition'],r['method'])==(name,condition,method)]
                failed=[r for r in selected if not r['success']]
                maxima=[r['maximum_normalized_action_input']['abs_value'] for r in selected]
                groups.append(dict(task=name,condition=condition,method=method,completed=len(selected),
                    success=sum(r['success'] for r in selected),failures=len(failed),
                    failed_without_red_3cm_rise=sum(r['object_heights']['red']['first_3cm_rise_step'] is None for r in failed),
                    failed_without_blue_3cm_rise=sum(r['object_heights']['blue']['first_3cm_rise_step'] is None for r in failed),
                    failed_with_100_narrow_width_states=sum(r['longest_5mm_or_less_width_run_states']>=100 for r in failed),
                    normalized_state_max_median=float(np.median(maxima)),
                    normalized_state_max_p90=float(np.quantile(maxima,.9)),normalized_state_max_largest=max(maxima)))
    atomic(OUT/'outcome_diagnostics.json',dict(episodes=rows,groups=groups,
        interpretation='Descriptive conventions, not success predicates or causal contact tests. Common normalization of states preceding actions excludes the terminal state; it does not represent all custom API features.',
        analysis_source_sha256=digest(ROOT/'experiments/exp2/analysis/analyze_outcomes.py'),
        retained_original_diagnostics_sha256=digest(ROOT/'experiments/exp2/analysis/outcome_diagnostics.json'),
        API_calls=0,optimizer_updates=0,simulator_steps=0))
    table(OUT/'tables/failure_diagnostics.csv',groups)


def physical_execution_index(episodes):
    """Locate actual physical executions, excluding later result-reuse jobs."""
    supplement=read(BASE/'evaluation_supplement/allocation.json')['supplemental_cells']
    tail=read(BASE/'evaluation_tail/allocation.json')['selected_cells']
    key=lambda r:(r['task'],r['condition'],r['seed'])
    extra_keys={key(r) for r in supplement};tail_keys={key(r) for r in tail}
    assert len(extra_keys)==150 and len(tail_keys)==30 and not extra_keys&tail_keys
    rows=[]
    for row in episodes:
        if row['method']!=METHOD:continue
        root=Path(row['episode_root']);plan=read(root/'plan.json');gpu=plan['device']['physical_gpu']
        if key(row) in extra_keys:
            owner='supplement';job=BASE/'evaluation_supplement/jobs/evaluate'/str(gpu)
        elif key(row) in tail_keys:
            owner='tail';job=BASE/'evaluation_tail/jobs'
        else:
            owner='original';job=BASE/'jobs/evaluation'
        job=job/row['task']/row['condition']/str(row['seed'])
        process=read(job/'process.json');exited=read(job/'process_result.json')
        closed=read(root/'worker/closed.json');worker=read(root/'worker/request.json')
        assert process['gpu']==exited['gpu']==gpu and exited['returncode']==closed['returncode']==0
        assert closed['inference_chunks']==(row['steps']+7)//8
        assert worker['seed']==row['seed'] and Path(worker['folder']).resolve()==folder(row['task']).resolve()
        assert plan['method']==METHOD and not (root/'failure.json').exists()
        rows.append(dict(task=row['task'],condition=row['condition'],seed=row['seed'],
            owner=owner,physical_gpu=gpu,steps=row['steps'],inference_chunks=closed['inference_chunks'],
            actual_process=repo_path(job/'process.json'),process_sha256=digest(job/'process.json'),
            actual_exit=repo_path(job/'process_result.json'),exit_sha256=digest(job/'process_result.json'),
            physical_plan=repo_path(root/'plan.json'),plan_sha256=digest(root/'plan.json')))
    assert len(rows)==300 and Counter(r['owner'] for r in rows)=={'original':120,'supplement':150,'tail':30}
    table(OUT/'tables/single_policy_physical_executions.csv',rows)
    atomic(OUT/'physical_execution_audit.json',dict(passed=True,physical_trials=300,
        owners=dict(Counter(r['owner'] for r in rows)),episodes=rows,
        interpretation='One actual physical owner per new study cell. Later original-coordinator jobs for supplement/tail cells use the frozen existing-result branch before environment creation.'))


def deployment_timing(episodes):
    """Document recorded timings without treating shared execution as GPU-hours."""
    rows=[]
    for episode in episodes:
        if not episode['completed']:continue
        source=Path(episode['episode_root'])/'result.json'
        recorded=read(source)
        elapsed=recorded['elapsed_seconds']
        assert np.isfinite(elapsed) and elapsed>=0
        rows.append(dict(task=episode['task'],condition=episode['condition'],
            method=episode['method'],seed=episode['seed'],success=episode['success'],
            physical_steps=episode['steps'],elapsed_seconds=elapsed,
            inference_chunks=recorded.get('inference_chunks'),
            inference_seconds=recorded.get('inference_seconds'),
            policy_invocations=recorded.get('invocations'),source=repo_path(source)))
    table(OUT/'tables/deployment_episode_timing.csv',rows)
    summary=[]
    for method in METHODS:
        for condition in ('ID','OOD'):
            selected=[r for r in rows if r['method']==method and r['condition']==condition]
            elapsed=[r['elapsed_seconds'] for r in selected]
            summary.append(dict(method=method,condition=condition,complete_episodes=len(selected),
                unknown_excluded=150-len(selected),physical_steps=sum(r['physical_steps'] for r in selected),
                summed_episode_elapsed_seconds=sum(elapsed),mean_episode_seconds=float(np.mean(elapsed)),
                median_episode_seconds=float(np.median(elapsed)),p90_episode_seconds=float(np.percentile(elapsed,90)),
                scope='Selected complete outcomes only; excludes interrupted attempts and preparation. Different resource sharing, early termination and success lengths; not exclusive GPU-hours or a matched latency comparison.'))
    table(OUT/'tables/deployment_timing.csv',summary)


def evaluation_attempts(episodes):
    """Index selected cells and every retained APPL 6 outage/reset attempt."""
    selected={Path(r['episode_root']).resolve() for r in episodes}
    roots=[]
    for row in episodes:
        if row['method']=='APPL_6_xhigh':continue
        roots.append((row['method'],row.get('selected_attempt','original'),row['task'],Path(row['episode_root'])))
        if row.get('selected_attempt')=='authorized_reset_retest_20260919':
            roots.append((row['method'],'original',row['task'],Path(row['original_episode_root'])))
    base=ROOT/'runs/exp2/M1_astra_xhigh'
    for scope in ('original','evaluation_recovery','evaluation_recovery_20260918','evaluation_followup_20260918'):
        directory=base if scope=='original' else base/scope
        for name in NAMES:
            for plan in sorted((directory/name/'evaluation/APPL').glob('*/*/plan.json')):
                roots.append(('APPL_6_xhigh',scope,name,plan.parent))
    assert len({root.resolve() for _,_,_,root in roots})==1292
    rows=[]
    for method,scope,name,root in roots:
        complete=(root/'result.json').exists()
        outcome=root/('result.json' if complete else 'failure.json')
        assert outcome.exists() and (root/'plan.json').exists()
        result=read(outcome)
        if complete:assert result['completed_evaluation']
        rows.append(dict(method=method,attempt_scope=scope,task=name,condition=root.parent.name,
            seed=int(root.name),selected_for_main_matrix=root.resolve() in selected,
            completed=complete,success=result['success'] if complete else None,
            status=result['status'] if complete else 'interrupted',episode_root=repo_path(root),
            plan_sha256=digest(root/'plan.json'),outcome_file=repo_path(outcome),outcome_sha256=digest(outcome)))
    assert sum(r['completed'] for r in rows)==1200
    assert sum(r['selected_for_main_matrix'] for r in rows)==1200
    summary=[]
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,planned_cells=300,recorded_reset_attempts=len(subset),
            complete_outcomes=sum(r['completed'] for r in subset),
            interrupted_attempts=sum(not r['completed'] for r in subset),
            additional_attempts_beyond_original_matrix=len(subset)-300))
    table(OUT/'tables/evaluation_attempts.csv',rows)
    table(OUT/'tables/evaluation_attempt_summary.csv',summary)
    atomic(OUT/'evaluation_attempts.json',dict(planned_cells=1200,recorded_reset_attempts=1292,
        complete_outcomes=1200,interrupted_attempts=92,selected_unknown_outcomes=0,
        scope='Main-comparison evaluations only; includes zero-step interrupted attempts. Historical preliminary studies are separate. Authorized recoveries reset initial states; no completed task failure was rerun.',
        summary=summary,attempts=rows))


def build():
    complete=read(BASE/'completion.json')
    assert complete['status']=='complete' and complete['completed_new_trials']==300
    recovered=read(BASE/'incidents/tray_design_http502/recovery_completed.json')
    assert recovered['original_attempt_preserved']
    for rel,sha in complete['artifacts'].items():assert digest(BASE/rel)==sha,rel
    supplement=BASE/'evaluation_supplement'
    if supplement.exists():
        extra=read(supplement/'completed.json');assert extra['physical_trials']==150
        assert extra['no_repeated_physical_trials'] and len(extra['episodes'])==150
        assert len(read(supplement/'checks_completed.json')['checks'])==10
        for r in extra['episodes']:
            root=BASE/r['task']/'evaluation/SinglePrior_6_xhigh'/r['condition']/str(r['seed'])
            assert digest(root/'result.json')==r['result_sha256'] and digest(root/'replay.mp4')==r['video_sha256']
            process=read(supplement/'jobs/evaluate'/str(r['gpu'])/r['task']/r['condition']/str(r['seed'])/'process_result.json')
            assert process['returncode']==r['returncode']==0
    tail=BASE/'evaluation_tail'
    if tail.exists():
        extra=read(tail/'completed.json');assert extra['physical_trials']==30
        assert extra['no_repeated_physical_trials'] and len(extra['episodes'])==30
        allocation=read(tail/'allocation.json')
        assert digest(ROOT/'experiments/exp2/analysis/single_policy_evaluation_tail.py')==allocation['source_sha256']
        assert digest(tail/'allocation.json')==extra['allocation_sha256']
        for r in extra['episodes']:
            root=BASE/r['task']/'evaluation'/METHOD/r['condition']/str(r['seed'])
            assert digest(root/'result.json')==r['result_sha256'] and digest(root/'replay.mp4')==r['video_sha256']
            process=read(tail/'jobs'/r['task']/r['condition']/str(r['seed'])/'process_result.json')
            assert process['returncode']==r['returncode']==0
    result=read(BASE/'results.json');previous=read(COMPARISON)
    assert result['summary']==recovered['summary'], 'Retained-failure accounting or final outcome summary changed'
    assert result['episodes'][:900]==previous['episodes']
    assert len(result['episodes'])==1200 and len(result['groups'])==40
    assert sum(r['completed'] for r in result['episodes'])==1199
    from .recovery import apply
    result=apply(result,OUT)
    assert sum(r['completed'] for r in result['episodes'])==1200
    copied=[]
    framework_index=read(OUT/'framework/index.json')
    for rel,sha in (framework_index['files']|framework_index['additional_attribution_files']).items():
        assert digest(ROOT/rel)==sha==digest(OUT/'framework'/rel),rel
        copied.append(dict(source=rel,copy='framework/'+rel,sha256=sha))
    for section in ('protocols','supporting_reports'):
        for row in read(OUT/section/'index.json')['files']:
            relative=row['bundle_file'] if section=='protocols' else section+'/'+row['bundle_file']
            assert digest(ROOT/row['source'])==row['sha256']==digest(OUT/relative),relative
            copied.append(dict(source=row['source'],copy=relative,sha256=row['sha256']))
    atomic(OUT/'bundle_source_audit.json',dict(passed=True,unchanged_copies=copied))
    initial={}
    for row in result['episodes']:
        key=(row['task'],row['condition'],row['seed'])
        state=read(Path(row['episode_root'])/'initial_state.json')
        if key in initial:assert initial[key]==state,('Paired initial state differs',key,row['method'])
        else:initial[key]=state
    assert len(initial)==300
    atomic(OUT/'initial_states.json',dict(all_four_methods_identical=True,
        states=[dict(task=t,condition=c,seed=s,state=v) for (t,c,s),v in initial.items()]))
    tasks=prepare();inventory,segments=models()
    episodes=result['episodes'];groups=result['groups'];aggregates=[];prefixes=[];stops=[]
    design_evidence(inventory,episodes)
    checks=interface_accounting(inventory)
    for method in METHODS:
        for condition in ('ID','OOD'):
            selected=[r for r in episodes if r['method']==method and r['condition']==condition]
            s=sum(r['success'] is True for r in selected);n=sum(r['completed'] for r in selected);u=150-n
            aggregates.append(dict(method=method,condition=condition,planned=150,completed=n,success=s,unknown=u,
                success_count_lower=s,success_count_upper=s+u,rate=s/150 if not u else None))
        selected=[r for r in episodes if r['method']==method]
        prefixes.append(dict(method=method,planned=300,unknown=sum(not r['completed'] for r in selected),
            **{str(cap):sum(r['success_by_'+str(cap)] for r in selected) for cap in (1500,3000,5000)}))
        for name in NAMES:
            for condition in ('ID','OOD'):
                selected=[r for r in episodes if (r['method'],r['task'],r['condition'])==(method,name,condition)]
                stops.append(dict(method=method,task=name,condition=condition,
                    **dict(Counter(r['status'] for r in selected))))
    table(OUT/'tables/results.csv',groups);table(OUT/'tables/aggregate.csv',aggregates)
    table(OUT/'tables/episodes.csv',episodes);table(OUT/'tables/success_prefixes.csv',prefixes)
    table(OUT/'tables/termination.csv',stops);table(OUT/'tables/paired_single_prior.csv',result['paired'])
    compute=[]
    for method in METHODS:
        selected=[r for r in inventory if r['method']==method]
        method_checks=[r for r in checks if r['method']==method]
        compute.append(dict(method=method,models=len(selected),formal_updates=sum(r['optimizer_updates'] for r in selected),
            minimum_model_parameters=min(r['parameters'] for r in selected),maximum_model_parameters=max(r['parameters'] for r in selected),
            total_portfolio_parameters=sum(r['parameters'] for r in selected),
            sum_model_training_wall_seconds=sum(r['training_seconds'] for r in selected),
            API_interface_checks=len(method_checks),
            confirmed_API_check_updates=sum(r['confirmed_optimizer_updates'] for r in method_checks),
            API_check_update_upper_bound=sum(r['optimizer_update_upper_bound'] for r in method_checks),
            timing_scope='Includes resource sharing; not exclusive GPU-hours or FLOPs. Includes reused models.'))
    table(OUT/'tables/compute.csv',compute)
    diagnostics(episodes)
    physical_execution_index(episodes)
    deployment_timing(episodes)
    evaluation_attempts(episodes)
    plots(groups,episodes)
    from experiments.exp2.paper.examples import export
    export(episodes)
    # Rendering and provenance are kept separate from the immutable experiment.
    from experiments.exp2.paper.render import render
    render(result,tasks,inventory,segments,aggregates,prefixes,compute)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','build'])
    args=parser.parse_args()
    print(prepare() if args.command=='prepare' else build())
