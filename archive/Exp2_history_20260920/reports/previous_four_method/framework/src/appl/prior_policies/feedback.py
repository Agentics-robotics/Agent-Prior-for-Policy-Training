"""Generic measurements and API-owned stopping rules; no task-specific controller."""
import json
import math
import operator


PROMPT = '''You are the inference-time policy-selection agent for the drawer task.
Complete the supplied geometric goals using the frozen learned Diffusion Policies.
Choose policies by their actual priors and applicability, reading PRIOR.md,
HANDOFF.json and measured support before using a policy. You choose the sequence,
invocation duration, observation-based stopping conditions, and when to finish.

invoke_policy executes until your requested duration ends, any of your stop_when
groups becomes true, or the whole task succeeds. Conditions are checked after
EVERY physical step. All conditions within a group must hold; any matching group
returns control to you immediately. You then inspect the returned state and
choose whether to continue that policy, transfer to another, or finish. The
framework does not infer skill completion or choose successor policies for you.
Use documented state cues to express useful handoff and failure-observation
conditions. Choose the metrics, comparisons and thresholds yourself. An empty
stop_when list requests the full duration. Conditions are absolute or measured
relative to the state at the beginning of THIS invocation. They can already hold
at entry and then stop after the first step; revise a stale condition before
continuing. A stopped invocation is an opportunity to assess, not proof of success.

Measurements use world metres and seconds: tcp_x/y/z, red_x/y/z, blue_x/y/z,
drawer_position (m), drawer_velocity (m/s), finger_width (sum of both finger
joint positions, m), tcp_red_distance and tcp_blue_distance (3D Euclidean m).
drawer_open, red_on_pad, blue_inside and success are the supplied predicates as
0 or 1. Returned metric_ranges contain observed minima/maxima and their steps.
The full state also includes qpos/qvel and quaternion poses. Only observations
are measurements; grasp/contact must be inferred from their relationships and
motion. A low object at the END of a call can mean it never lifted OR that it
lifted and lowered again; inspect ranges and use stop conditions to observe the
transition before continuing past a useful handoff.

Assess successor readiness from actual object height, finger configuration,
TCP-object relation/motion and drawer state together with the selected documents.
Temporal training overlap is evidence about demonstrated transitions; reason
explicitly about whether it supports the current state after a missed grasp,
object displacement or disturbed drawer. The policies remain learned local
controllers: a conditioning feature or auxiliary objective is not a guarantee
of recovery or collision avoidance. You may attempt recovery and revisit skills;
explain why that policy can plausibly help and what observation would change your
decision. Continue while you have a credible route to progress and finish honestly
if you judge the available policies cannot complete the task.

The context contains the initial task/catalogue, the latest complete read of each
policy document, a recent window of complete exchanges, and your latest invocation
with its notebook. Full raw history stays in the journal. Each invoke_policy must
include an updated concise notebook IN YOUR OWN WORDS: important attempts and
outcomes, unresolved facts and current plan worth retaining after older exchanges
leave the window. Your previous notebook remains in the latest invocation's
arguments/result. Use read_invocation to retrieve an older numbered invocation's
full before/after states and reason when needed. Use the latest observation,
not the initial snapshot, for current control decisions. observe does not advance
physical time. Consecutive calls to the same policy continue its action stream;
switching policy resets the incoming observation/action queue, not its RNG seed.

Task success requires only simultaneous drawer-open, red-on-pad and blue-inside,
exactly as supplied. No terminal release, clearance, speed or hold condition is
added. Skill handoff cues may still require a suitable manipulation state.
You cannot modify policies, supply low-level actions or redefine the task.
Write all tool arguments and explanations in English.
'''

METRICS = (
    'tcp_x','tcp_y','tcp_z','red_x','red_y','red_z','blue_x','blue_y','blue_z',
    'drawer_position','drawer_velocity','finger_width','tcp_red_distance',
    'tcp_blue_distance','drawer_open','red_on_pad','blue_inside','success',
)
COMPARISONS = dict(lt=operator.lt,le=operator.le,gt=operator.gt,ge=operator.ge,
                   eq=operator.eq)


def measurements(state, goals):
    values={f'{name}_{axis}':float(state[f'{name}_pose'][i])
            for name in ('tcp','red','blue') for i,axis in enumerate('xyz')}
    values.update(drawer_position=float(state['drawer_position'][0]),
        drawer_velocity=float(state['drawer_velocity'][0]),
        finger_width=float(sum(state['qpos'][7:9])),
        tcp_red_distance=math.dist(state['tcp_pose'][:3],state['red_pose'][:3]),
        tcp_blue_distance=math.dist(state['tcp_pose'][:3],state['blue_pose'][:3]))
    values.update({key:float(value) for key,value in goals.items()})
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite feedback metric')
    return values


def stop_schema(goal_names=None):
    metrics=METRICS if goal_names is None else METRICS[:-4]+tuple(goal_names)
    condition=dict(type='object',properties=dict(
        metric=dict(type='string',enum=list(metrics)),
        comparison=dict(type='string',enum=list(COMPARISONS)),
        value=dict(type='number'),reference=dict(type='string',enum=['absolute','invocation_start'])),
        required=['metric','comparison','value','reference'],additionalProperties=False)
    group=dict(type='object',properties=dict(label=dict(type='string',minLength=1,maxLength=160),
        conditions=dict(type='array',items=condition,minItems=1,maxItems=8)),
        required=['label','conditions'],additionalProperties=False)
    return dict(type='array',items=group,minItems=0,maxItems=6)


def matches(groups, current, initial):
    def satisfied(c):
        value=float(c['value'])
        if not math.isfinite(value):raise ValueError('Nonfinite API stopping value')
        actual=current[c['metric']]
        if c['reference']=='invocation_start':actual-=initial[c['metric']]
        return COMPARISONS[c['comparison']](actual,value)
    return [i for i,group in enumerate(groups) if all(satisfied(c) for c in group['conditions'])]


def update_ranges(ranges, current, step):
    for key,value in current.items():
        if key not in ranges:
            ranges[key]=dict(min=value,max=value,min_step=step,max_step=step)
        else:
            r=ranges[key]
            if value<r['min']:r.update(min=value,min_step=step)
            if value>r['max']:r.update(max=value,max_step=step)


def project_context(journal, history, recent_turns):
    """Select whole original exchanges. Never edit API items or the durable history."""
    if recent_turns<1:raise ValueError('At least one recent exchange is required')
    turns=[];offset=1;documents={};last_invocation=None
    for seq,raw in journal.db.execute("SELECT seq,response FROM api WHERE status='consumed' ORDER BY seq"):
        output=json.loads(raw)['output']
        calls=[item for item in output if item.get('type')=='function_call']
        size=len(output)+len(calls);items=history[offset:offset+size]
        if items[:len(output)]!=output or len(items)!=size:raise ValueError('Journal/history exchange mismatch')
        for call,result in zip(calls,items[len(output):]):
            if result.get('type')!='function_call_output' or result['call_id']!=call['call_id']:
                raise ValueError('Incomplete tool exchange')
            args=json.loads(call['arguments'])
            if call['name']=='read_policy':documents[args['policy_id']]=len(turns)
            if call['name']=='invoke_policy':last_invocation=len(turns)
        turns.append((seq,items));offset+=size
    if offset!=len(history):raise ValueError('Uncommitted history in context projection')
    selected=set(range(max(0,len(turns)-recent_turns),len(turns)))|set(documents.values())
    if last_invocation is not None:selected.add(last_invocation)
    projected=[history[0]]+[item for i in sorted(selected) for item in turns[i][1]]
    journal.event('context_projection',source_items=len(history),request_items=len(projected),
        retained_api_sequences=[turns[i][0] for i in sorted(selected)],
        retained_policy_documents=sorted(documents),api_items_modified=0)
    return projected
