"""Controller-contract gate. Serializes a candidate only; never performs robot IO."""
import numpy as np
from perception import finite


def decode(action,contract,spec):
    def refuse(reason): return {'enabled':False,'command':None,'reason':reason}
    if not finite(action,(7,)): return refuse('Candidate must be a finite native dq[7]')
    if not isinstance(contract,dict): return refuse('Verified recording-controller contract is absent')
    for key in ('recording_verified','units_verified','joint_order_verified','timing_verified','limits_verified',
                'watchdog_verified','stop_verified','tool_verified','workspace_verified','current_state_verified',
                'singularity_check_passed','workspace_check_passed'):
        if contract.get(key) is not True: return refuse('Required verification absent: '+key)
    if not contract.get('verification_id') or not contract.get('safety_check_id'):
        return refuse('Commissioning and current safety verification identifiers are required')
    if contract.get('source_field')!='action_json.dq' or contract.get('mode')!='joint_velocity':
        return refuse('Must verify the recorded action_json.dq direct joint_velocity semantics')
    units=contract.get('native_units')
    if units not in ('rad/s','deg/s') or contract.get('controller_units')!=units:
        return refuse('Explicit matching native/controller angular velocity units required; no inferred conversion')
    order=contract.get('joint_order'); recorded=contract.get('recorded_joint_order')
    if not isinstance(order,list) or len(order)!=7 or len(set(order))!=7 or order!=recorded:
        return refuse('Seven unique joints in verified recorded order required')
    for key in ('command_rate_hz','hold_s','watchdog_s','max_latency_s','dt_s'):
        if not finite(contract.get(key)) or float(contract[key])<=0:
            return refuse('Invalid verified timing: '+key)
    hold=float(contract['hold_s']); watchdog=float(contract['watchdog_s']); dt=float(contract['dt_s'])
    if hold>watchdog: return refuse('Command hold exceeds watchdog')
    age=contract.get('command_age_s')
    if not finite(age) or not 0<=float(age)<=float(contract['max_latency_s']):
        return refuse('Current command latency verification failed')
    for key in ('velocity_limits','acceleration_limits','q','q_min','q_max','previous_command','checked_action'):
        if not finite(contract.get(key),(7,)): return refuse('Missing finite seven-vector: '+key)
    a=np.asarray(action,np.float64); lim=np.asarray(contract['velocity_limits'],np.float64)
    acc=np.asarray(contract['acceleration_limits'],np.float64)
    if (lim<=0).any() or (acc<=0).any(): return refuse('Nonpositive commissioned limits')
    if not np.array_equal(a,np.asarray(contract['checked_action'],np.float64)):
        return refuse('Current workspace/singularity check is not bound to this exact candidate')
    q=np.asarray(contract['q']); lo=np.asarray(contract['q_min']); hi=np.asarray(contract['q_max'])
    if (lo>=hi).any() or (q<lo).any() or (q>hi).any(): return refuse('Current joint-position limits failed')
    if (np.abs(a)>lim).any(): return refuse('Velocity limit violation; candidate is not silently clipped')
    prev=np.asarray(contract['previous_command'])
    if (np.abs(a-prev)>acc*dt).any(): return refuse('Acceleration limit violation; request verified stop/replan')
    # A conservative constant-hold position envelope is a check, NOT an integrated position command.
    position_unit='rad' if units=='rad/s' else 'deg'
    if contract.get('joint_position_units')!=position_unit: return refuse('Joint-position units not verified')
    endpoint=q+a*hold
    if (endpoint<lo).any() or (endpoint>hi).any(): return refuse('Hold-horizon position envelope violation')
    return {'enabled':True,'command':{'mode':'joint_velocity','dq':a.tolist(),'units':units,
            'joint_order':order,'rate_hz':float(contract['command_rate_hz']),'hold_s':hold,
            'watchdog_s':watchdog,'verification_id':contract['verification_id']},
            'reason':'Dry-run serialization only. All verification values are caller attestations; no hardware IO or robot authorization is implemented.'}
