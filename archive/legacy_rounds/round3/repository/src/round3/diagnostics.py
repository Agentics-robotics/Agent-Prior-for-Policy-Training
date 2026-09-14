"""Descriptive offline diagnostics; never change policy inputs or scored success."""
from collections import Counter
from pathlib import Path
import numpy as np
from relative_dp.utils import atomic_json, read_json, sha256
from .common import R3, ROOT, now

RULE = dict(version='round3-offline-diagnostics-v1',
    near_interaction_distance_m=.05, object_lift_threshold_m=.03, low_recorded_cabinet_progress=.1,
    approach_point='Current shared observation object/interaction point [4:7]; proximity does not establish contact or grasp',
    goal_points={'pick-place-wall':'object [4:7]', 'assembly':'observed ring body center [39:42]',
                 'peg-insert-side':'observed peg head [39:42]', 'stick-push':'observed pushed-object point [11:14]',
                 'drawer':'observed handle [4:7]', 'door':'observed handle [4:7]'},
    lift_points={'assembly':'ring body center [39:42]', 'other_manipulation':'object/tool observation [4:7]'},
    geometry='Euclidean distances in world metres; assembly also reports xy distance and signed ring_z-goal_z',
    cabinet_progress='Use recorded trajectory progress array when present, never re-query simulator; unavailable otherwise',
    scope='Descriptive thresholds, not contact detection, expert phases, success rules, model inputs, or causal failure diagnoses',
    precedence=['scored_success','exception','recorded_joint_violation','never_near_interaction_point',
                'near_handle_low_recorded_progress','near_handle_goal_not_completed',
                'near_object_without_lift','lift_observed_goal_not_completed'])


def freeze_rule():
    path=R3/'design_records'/'offline_diagnostic_rule.json'
    if path.exists():
        assert read_json(path)['rule']==RULE, 'Offline diagnostic rule changed after freezing'
    else:
        assert not any((R3/'design_records').glob('*/feedback/bundle.json')), 'Freeze descriptive rules before first feedback'
        atomic_json(path,dict(rule=RULE,created_at=now(),implementation_sha256=sha256(__file__)))
    return path


def trajectory_diagnostics(task,row,arrays):
    obs=np.asarray(arrays['obs'],dtype=float)
    assert obs.ndim==2 and obs.shape[0]>=1 and obs.shape[1]>=39 and np.isfinite(obs).all()
    hand,interaction,goal=obs[:,:3],obs[:,4:7],obs[:,36:39]
    goal_point=obs[:,39:42] if task in ('assembly','peg-insert-side') else obs[:,11:14] if task=='stick-push' else interaction
    lift_point=obs[:,39:42] if task=='assembly' else interaction
    distances=np.linalg.norm(hand-interaction,axis=1)
    target_dist=np.linalg.norm(goal_point-goal,axis=1)
    near=np.flatnonzero(distances<=RULE['near_interaction_distance_m'])
    peak_lift=float(np.max(lift_point[:,2]-lift_point[0,2]))
    progress=np.asarray(arrays.get('progress',[]),float)
    progress=progress[np.isfinite(progress)]
    result=dict(episode_id=row['episode_id'],scored_success=bool(row['success']),
        original_failure_category=row.get('failure_category'),termination_reason=row.get('termination_reason'),
        min_hand_to_interaction_m=float(distances.min()),final_hand_to_interaction_m=float(distances[-1]),
        first_within_5cm_step=int(near[0]) if len(near) else None,
        fraction_observations_within_5cm=float(np.mean(distances<=.05)),
        peak_observed_object_or_tool_lift_m=peak_lift,
        min_observed_goal_point_distance_m=float(target_dist.min()),final_observed_goal_point_distance_m=float(target_dist[-1]),
        maximum_recorded_cabinet_progress=float(progress.max()) if len(progress) else None,
        final_recorded_cabinet_progress=float(progress[-1]) if len(progress) else None)
    if task=='assembly':
        result.update(min_ring_goal_xy_distance_m=float(np.linalg.norm(goal_point[:,:2]-goal[:,:2],axis=1).min()),
                      final_ring_minus_goal_z_m=float(goal_point[-1,2]-goal[-1,2]))
    if row['success']: label='scored_success'
    elif row.get('exception'): label='exception'
    elif row.get('invalid_joint'): label='recorded_joint_violation'
    elif not len(near): label='never_near_interaction_point'
    elif task in ('drawer','door'):
        label='near_handle_low_recorded_progress' if len(progress) and progress.max()<.1 else 'near_handle_goal_not_completed'
    elif peak_lift<.03: label='near_object_without_lift'
    else: label='lift_observed_goal_not_completed'
    result['descriptive_label']=label
    return result


def summarize_result(task,result):
    rule=freeze_rule()
    rows=[]
    for record in result['records']:
        path=ROOT/record['trajectory_path']
        assert sha256(path)==record['trajectory_hash']
        with np.load(path,allow_pickle=False) as arrays:
            row=trajectory_diagnostics(task,record,arrays)
        row['split']=record['split']
        rows.append(row)
    metrics=['min_hand_to_interaction_m','peak_observed_object_or_tool_lift_m',
             'min_observed_goal_point_distance_m','final_observed_goal_point_distance_m','maximum_recorded_cabinet_progress']
    groups={}
    for split in ('IID','C','E'):
        selected=[r for r in rows if r['split']==split]
        means={key:float(np.mean(values)) if (values:=[r[key] for r in selected if r[key] is not None]) else None for key in metrics}
        groups[split]=dict(n=len(selected),descriptive_labels=dict(Counter(r['descriptive_label'] for r in selected)),mean=means)
    return dict(rule_path=str(rule.relative_to(ROOT)),rule_sha256=sha256(rule),
                implementation_sha256=sha256(__file__),changes_scored_outcomes=False,additional_rollouts=0,
                interpretation=RULE['scope'],splits=groups,records=rows)
