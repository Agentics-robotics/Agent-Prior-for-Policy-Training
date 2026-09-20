"""Summarize all planned diagnostics without changing frozen experiment results."""
import csv
import json
from pathlib import Path
import numpy as np
from appl.io import ROOT,read,atomic,digest,source_manifest
from .run import BASE,TASKS


def main():
    plan=read(BASE/'plan.json');results=[];teacher=[];support=read(BASE/'initial_support.json')['tasks']
    for task in TASKS:
        original=[read(p) for p in sorted((BASE/'runs'/task).glob('*/complete.json'))]
        binary=[read(p) for p in sorted((BASE/'binary_gripper'/task).glob('*/result.json'))]
        takeover=[read(p) for root in ['takeover','takeover_physical_replay'] for p in sorted((BASE/root/task).glob('*/result.json'))]
        assert len(original)==12 and len(binary)==len(takeover)==3,(task,len(original),len(binary),len(takeover))
        assert len({x['demonstration'] for x in takeover})==3
        demos=list(plan['tasks'][task]['demonstrations']);selected={demos[i] for i in [0,5,11]}
        assert {x['demonstration'] for x in binary}==selected
        assert {x['demonstration'] for x in takeover}==selected
        baseline=[x['policy'] for x in original if x['policy']['demonstration'] in selected]
        learned=[x['policy'] for x in original]
        replay=[x['expert_replay'] for x in original]
        row=dict(task=task,original_reset_success=sum(x['success'] for x in learned),original_reset_count=12,
            original_on_paired_three=sum(x['success'] for x in baseline),binary_gripper_success=sum(x['success'] for x in binary),paired_count=3,
            oracle_prefix_success=sum(x['success'] for x in takeover),oracle_prefix_count=3,
            expert_replay_success=sum(x['success'] for x in replay),expert_replay_count=12,
            exact_reset_count=sum(max(x['initial_errors'].values())==0 for x in replay),
            bitwise_exact_full_replays=sum(max(x['max_observation_errors'].values())==0 for x in replay),
            max_normalized_input=max(x['normalized_input_max'] for x in learned),
            closed_loop_no_red_rise_3cm=sum(x['maximum_object_rise']['red']<=.03 for x in learned),
            closed_loop_no_blue_rise_3cm=sum(x['maximum_object_rise']['blue']<=.03 for x in learned),
            ID_inside_empirical_initial_hull=support[task]['conditions']['ID']['inside_empirical_convex_hull'],
            ID_nearest_initial_median_m=support[task]['conditions']['ID']['nearest_training_layout_L2_m']['median'])
        results.append(row)
        for weights,groups in read(BASE/'teacher'/task/'result.json')['results'].items():
            for group,values in groups.items():teacher.append(dict(task=task,weights=weights,group=group,**values))
        assert digest(plan['tasks'][task]['checkpoint'])==plan['tasks'][task]['checkpoint_sha256']
        for d in plan['tasks'][task]['demonstrations'].values():assert digest(d['path'])==d['sha256']
    assert source_manifest()==plan['framework_source']
    for name,rows in [('summary',results),('teacher_metrics',teacher)]:
        with (BASE/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic(BASE/'summary.json',dict(tasks=results,teacher=teacher,API_requests=0,optimizer_updates=0,
        framework_checkpoints_and_demonstrations_unchanged=True,
        caveat='Training-reset diagnostic, 1500 policy steps. Adaptive three-reset interventions and oracle prefixes are not held-out performance results. Contact replay is not bitwise exact for every demonstration.'))
    plots();print(results)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    output=BASE/'figures';output.mkdir(exist_ok=True)
    task='tray_pack';identifier='demo10300';demo=read(read(BASE/'plan.json')['tasks'][task]['demonstrations'][identifier]['path'])
    paths=dict(Original=BASE/'runs'/task/identifier/'frozen_DP',Binary=BASE/'binary_gripper'/task/identifier)
    fig,axes=plt.subplots(3,1,figsize=(8.5,7),sharex=True)
    for label,p in paths.items():
        rows=[json.loads(l) for l in (p/'trace.jsonl').read_text().splitlines()][:160];x=np.arange(1,len(rows)+1)
        axes[0].plot(x,[r['action'][7] for r in rows],label=label,lw=1.5)
        axes[1].plot(x,[sum(r['state']['qpos'][7:])*100 for r in rows],label=label,lw=1.5)
        axes[2].plot(x,[r['state']['tcp_pose'][2]*100 for r in rows],label=label,lw=1.5)
    x=np.arange(1,161);states=[v['state'] for v in demo['observations'][1:161]]
    axes[0].plot(x,[v[7] for v in demo['actions'][:160]],'--',label='Expert',color='black',lw=1)
    axes[1].plot(x,[sum(s['qpos'][7:])*100 for s in states],'--',label='Expert',color='black',lw=1)
    axes[2].plot(x,[s['tcp_pose'][2]*100 for s in states],'--',label='Expert',color='black',lw=1)
    for ax,label in zip(axes,['Executed gripper command','Actual finger opening (cm)','TCP height (cm)']):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.axvline(49,color='grey',alpha=.5,lw=1);ax.legend(loc='best',ncol=3)
    axes[-1].set_xlabel('Control step (20 Hz)');fig.suptitle('Tray / original training reset 10300: small gripper errors amplify')
    fig.tight_layout();fig.savefig(output/'tray_gripper_feedback.png');fig.savefig(output/'tray_gripper_feedback.pdf');plt.close(fig)
    probe=read(BASE/'conditioning/result.json');variants=['observed','replace_finger_qpos','replace_arm_qpos_qvel','replace_TCP_pose','expert_same_time']
    fig,ax=plt.subplots(figsize=(8.5,4.3))
    for v in variants:
        rows=[r for r in probe if r['variant']==v];ax.plot([r['step'] for r in rows],[r['first_grip_mean'] for r in rows],marker='o',label=v)
    ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Original rollout control step',ylabel='Next gripper command, mean of 16 samples',title='Paired-noise input probe: finger position drives the premature closure')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'tray_conditioning.png');fig.savefig(output/'tray_conditioning.pdf');plt.close(fig)


if __name__=='__main__':main()
