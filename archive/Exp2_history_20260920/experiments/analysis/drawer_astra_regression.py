"""Read-only post-hoc comparison of retained drawer trajectories and API libraries.

No API calls, policy changes, training, or simulator execution. New outputs are
analysis artifacts only; diagnostic thresholds never change task predicates.
"""
from collections import Counter
from pathlib import Path
import json
import statistics

from appl.io import ROOT, atomic, digest, read

OUT = ROOT / 'experiments/exp2/analysis/drawer_astra_regression'
FINAL = ROOT / 'runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/results.json'
GOALS = ['drawer_open', 'red_on_pad', 'blue_inside']


def episode(row):
    root = Path(row['episode_root'])
    trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
    invocations = read(root / 'invocations.json')
    result = read(root / 'result.json')
    first = {goal: next((t['step'] for t in trace if t['metrics'][goal]), None) for goal in GOALS}
    d = [float(t['state']['drawer_position'][0]) for t in trace]
    opened = first['drawer_open']
    closed = next((t for t in trace if opened is not None and t['step'] > opened
                   and t['state']['drawer_position'][0] < .24), None)
    records, offset = [], 0
    for call in invocations:
        n = call['executed_steps']
        chunk = trace[offset:offset+n]
        records.append(dict(policy=call['policy_id'], start_step=offset, stop_step=offset+n,
            steps=n, reason=call['reason'], stop_reason=call['stop_reason'],
            stop_when=call['stop_when'], matched_stop_rules=call['matched_stop_rules'],
            before_drawer=call['before']['drawer_position'][0],
            after_drawer=call['after']['drawer_position'][0],
            before_red_z=call['before']['red_pose'][2], after_red_z=call['after']['red_pose'][2],
            before_blue=call['before']['blue_pose'][:3], before_red=call['before']['red_pose'][:3],
            before_finger_width=sum(call['before']['qpos'][7:9]),
            after_red=call['after']['red_pose'][:3], after_blue=call['after']['blue_pose'][:3],
            goals=call['task_goals'], notebook=call['notebook'],
            first_goals={g:next((t['step'] for t in chunk if t['metrics'][g]),None) for g in GOALS}))
        offset += n
    assert offset == result['steps']
    return dict(method=row['method'], condition=row['condition'], seed=row['seed'],
        success=row['success'], status=result['status'], steps=result['steps'], final=result['final'], first_goals=first,
        first_policy=invocations[0]['policy_id'], max_drawer=max(d),
        red_above_015=any(t['state']['red_pose'][2] > .15 for t in trace),
        blue_above_010=any(t['state']['blue_pose'][2] > .10 for t in trace),
        closure_below_024_after_open=dict(step=closed['step'], policy=closed['policy_id'],
            drawer=closed['state']['drawer_position'][0], state=closed['state'],metrics=closed['metrics']) if closed else None,
        end_drawer=d[-1], calls=records, root=str(root),
        inputs={name:digest(root/name) for name in ('trace.jsonl','invocations.json','result.json')})


def plot_example(episodes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True, layout='constrained')
    pair = [e for e in episodes if e['condition']=='ID' and e['seed']==6302]
    for e, color in zip(pair, ('#2466a2', '#d04a32')):
        trace = [json.loads(line) for line in (Path(e['root'])/'trace.jsonl').read_text().splitlines()]
        label = 'Old APPL: success at 972' if e['method']=='APPL_5_5_high' else 'Astra APPL: failure at 1069'
        steps = [t['step'] for t in trace]
        for ax, (key, index) in zip(axes, (('drawer_position',0), ('red_pose',2), ('blue_pose',2))):
            ax.plot(steps, [t['state'][key][index] for t in trace], color=color, lw=1.7, label=label)
    for ax, label in zip(axes, ('Drawer opening (m)', 'Red center height (m)', 'Blue center height (m)')):
        ax.set_ylabel(label)
        ax.axvspan(473, 617, color='#d04a32', alpha=.10)
        ax.axvspan(968, 1069, color='#916fbd', alpha=.10)
        ax.grid(alpha=.18)
        ax.set_xlim(0, 1090)
    axes[0].axhline(.26, color='#444444', ls='--', lw=1, label='Task threshold: opening > 0.26 m')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[0].set_ylim(-.012,.34)
    axes[1].text(480,.34,'Astra first red-transfer call',fontsize=8,color='#9b3121')
    axes[1].text(955,.28,'Astra reopening\nattempts',ha='right',fontsize=8,color='#685081')
    axes[-1].set_xlabel('Physical control step (each method starts from reset)')
    fig.suptitle('Drawer exchange, paired ID seed 6302\nExisting trace replay; shading marks Astra calls only', fontsize=13)
    fig.savefig(OUT/'seed6302_timeline.png', dpi=160)
    plt.close(fig)


def main():
    source=read(FINAL)
    rows=[r for r in source['episodes'] if r['task']=='drawer_exchange' and r['method']!='naive_DP']
    assert len(rows)==120 and all(r['completed'] for r in rows)
    episodes=[episode(row) for row in rows]
    groups=[]
    for method in ('APPL_5_5_high','APPL_6_xhigh'):
        for condition in ('ID','OOD'):
            es=[e for e in episodes if (e['method'],e['condition'])==(method,condition)]
            failed=[e for e in es if not e['success']]
            closures=[e for e in es if e['closure_below_024_after_open']]
            stats=dict(method=method,condition=condition,episodes=len(es),successes=sum(e['success'] for e in es),
                ever_goals={g:sum(e['first_goals'][g] is not None for e in es) for g in GOALS},
                final_goals={g:sum(e['final'][g] for e in es) for g in GOALS},
                red_above_015=sum(e['red_above_015'] for e in es),
                blue_above_010=sum(e['blue_above_010'] for e in es),
                substantial_reclosures=len(closures),
                reclosures_by_policy=dict(Counter(e['closure_below_024_after_open']['policy'] for e in closures)),
                reclosures_before_red_goal=sum(e['first_goals']['red_on_pad'] is None or
                    e['closure_below_024_after_open']['step'] < e['first_goals']['red_on_pad'] for e in closures),
                failures_with_reclosure=sum(bool(e['closure_below_024_after_open']) for e in failed),
                failure_final_goal_patterns=dict(Counter(','.join(g for g in GOALS if e['final'][g]) or 'none' for e in failed)),
                initial_policy_counts=dict(Counter(e['first_policy'] for e in es)),
                used_policy_counts=dict(Counter(c['policy'] for e in es for c in e['calls'])),
                median_invocations=statistics.median(len(e['calls']) for e in es),
                median_steps=statistics.median(e['steps'] for e in es),
                failure_step_range=[min(e['steps'] for e in failed),max(e['steps'] for e in failed)] if failed else None)
            stats['failure_status_counts'] = dict(Counter(e['status'] for e in failed))
            stats['blue_lift_and_inside_same_episode_set'] = all(
                e['blue_above_010'] == (e['first_goals']['blue_inside'] is not None) for e in es)
            if method == 'APPL_6_xhigh':
                evac = [(e, next((c for c in e['calls'] if c['policy'].startswith('evacuate_red')), None)) for e in es]
                closings = [(e,c) for e,c in evac if c and c['before_drawer']>.26 and c['after_drawer']<=.26]
                insert = [(e,next((c for c in e['calls'] if c['policy'].startswith('insert_blue')),None)) for e in es]
                stats['first_evacuation'] = dict(policy_counts=dict(Counter(c['policy'] for _,c in evac if c)),
                    open_to_not_open=len(closings),successes_after_this_transition=sum(e['success'] for e,_ in closings),
                    transition_policy_counts=dict(Counter(c['policy'] for _,c in closings)),
                    transition_stop_reasons=dict(Counter(c['stop_reason'] for _,c in closings)),
                    transition_matched_conditions=dict(Counter(json.dumps(c['stop_when'][i]['conditions'],sort_keys=True)
                        for _,c in closings for i in c['matched_stop_rules'])),
                    transition_seeds=[e['seed'] for e,_ in closings])
                stats['first_insertion'] = {label:dict(count=sum(c is not None and predicate(c) for _,c in insert),
                    successes=sum(c is not None and predicate(c) and e['success'] for e,c in insert))
                    for label,predicate in [('blue_above_010',lambda c:c['before_blue'][2]>.10),
                                            ('blue_at_or_below_010',lambda c:c['before_blue'][2]<=.10)]}
            groups.append(stats)
            print(json.dumps(stats),flush=True)
    portfolios=[]
    for label,cfg_path in [('old','experiments/exp2/configs/scaleup/drawer_exchange.json'),
                           ('new','experiments/exp2/configs/astra_xhigh/drawer_exchange.json')]:
        cfg=read(ROOT/cfg_path); data=ROOT/cfg['dataset']; root=ROOT/cfg['output']
        manifest=read(data/'manifest.json'); norm=read(root/'normalization.json')
        skills=[]; policies=[]
        for d in manifest['datasets']:
            ds=read(data/d['dataset']); lengths=[s['stop']-s['start'] for s in ds['segments']]
            skills.append(dict(skill=d['skill_id'],segments=ds['segments'],
                mean_length=statistics.mean(lengths),total_actions=sum(lengths),heuristics=ds['heuristics']))
        for p in sorted(root.glob('policies/*/*/source/pipeline.json')):
            pipeline=read(p); training=read(p.parents[1]/'training/result.json')
            policies.append(dict(policy=pipeline['policy_id'],parameters=training['trainable_parameters'],
                updates=training['optimizer_steps'],sample_exposures=training['sample_exposures'],
                pipeline=pipeline,source=str(p.parent),source_hashes=training['source_hashes']))
        portfolios.append(dict(label=label,normalizer_sha256=norm['normalizer_sha256'],
            checks=manifest['checks'],skills=skills,policies=policies,
            total_updates=sum(p['updates'] for p in policies),
            parameters_range=[min(p['parameters'] for p in policies),max(p['parameters'] for p in policies)]))
    assert portfolios[0]['normalizer_sha256']==portfolios[1]['normalizer_sha256']
    OUT.mkdir(exist_ok=True)
    atomic(OUT/'evidence.json',dict(source_results_sha256=digest(FINAL),groups=groups,episodes=episodes,
        portfolios=portfolios,diagnostic_definitions=dict(substantial_reclosure='After first drawer_open (>0.26 m), later d<0.24 m; descriptive 2 cm hysteresis only.',
        height_checks='Red z>0.15 m and blue z>0.10 m are descriptive lifting proxies, not measured attachment or new completion conditions.'),
        new_API_requests=0,new_training_updates=0,new_simulator_steps=0,script_sha256=digest(__file__)))
    plot_example(episodes)


if __name__=='__main__':
    main()
