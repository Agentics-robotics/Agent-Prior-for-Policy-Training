import numpy as np
from round3.diagnostics import trajectory_diagnostics


def record(**kwargs):
    return dict(episode_id='synthetic',success=False,failure_category='native_truncated',termination_reason='native_truncated',**kwargs)


def test_timeout_diagnostic_preserves_score_and_separates_proximity_from_lift():
    obs=np.zeros((4,45));obs[:,:3]=1
    result=trajectory_diagnostics('pick-place-wall',record(),dict(obs=obs))
    assert result['descriptive_label']=='never_near_interaction_point'
    assert result['original_failure_category']=='native_truncated' and not result['scored_success']
    obs[:,:3]=0
    result=trajectory_diagnostics('pick-place-wall',record(),dict(obs=obs))
    assert result['descriptive_label']=='near_object_without_lift'
    obs[2:,6]=.06
    result=trajectory_diagnostics('pick-place-wall',record(),dict(obs=obs))
    assert result['descriptive_label']=='lift_observed_goal_not_completed'


def test_correct_observed_goal_point_for_assembly_peg_and_stick():
    obs=np.zeros((2,45));obs[:,36:39]=[1,2,3]
    obs[:,39:42]=[1,2,3];obs[:,11:14]=[1,2,3]
    for task in ('assembly','peg-insert-side','stick-push'):
        result=trajectory_diagnostics(task,record(),dict(obs=obs))
        assert result['final_observed_goal_point_distance_m']==0
    result=trajectory_diagnostics('pick-place-wall',record(),dict(obs=obs))
    assert result['final_observed_goal_point_distance_m']>0


def test_cabinet_uses_only_recorded_progress_and_retains_invalidity():
    obs=np.zeros((3,41))
    result=trajectory_diagnostics('drawer',record(),dict(obs=obs,progress=np.array([0,.02,.03])))
    assert result['descriptive_label']=='near_handle_low_recorded_progress'
    assert result['maximum_recorded_cabinet_progress']==.03
    invalid=trajectory_diagnostics('door',record(invalid_joint=True),dict(obs=obs))
    assert invalid['descriptive_label']=='recorded_joint_violation'
    assert invalid['maximum_recorded_cabinet_progress'] is None
