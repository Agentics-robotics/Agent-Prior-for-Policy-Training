"""Physical feasibility only; no learned-policy scores are available here."""
import time
import numpy as np
from relative_dp.utils import ROOT, atomic_json, read_json
from .geometry import GeometryEnv, CanonicalExpert

LEDGER = ROOT / 'artifacts/round2/calibration_rollouts.json'
TIERS = {'A': [10, 20, 35, 50], 'B': [7, 15, 25, 35], 'C': [5, 10, 17, 25]}


def record(task, seed, yaw, region='train', episode_id=None):
    rng = np.random.default_rng(seed)
    limits = {'drawer': {'train': [-.04, .04], 'left': [-.1, -.06], 'right': [.06, .1]},
              'door': {'train': [.03, .07], 'left': [0, .02], 'right': [.08, .1]}}
    x = rng.uniform(*limits[task][region])
    pos = [x, .9, 0.] if task == 'drawer' else [x, rng.uniform(.88, .92), .15]
    return dict(task_name=task, seed=int(seed), episode_id=episode_id or f'cal_{task}_{seed}_{yaw}',
                region=region, task_params=dict(base_position=pos, yaw_degrees=float(yaw)))


def rollout(env, rec, store=False):
    started = time.monotonic()
    obs, info = env.reset(rec)
    snap = env.snapshot(obs)
    expert = CanonicalExpert(env)
    observations, actions, rewards, infos = [obs], [], [], [info]
    exception = None
    for t in range(500):
        try:
            action = expert.action(obs)
            obs, reward, _, _, info = env.step(action)
            actions.append(action)
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)
            if info['success']:
                break
        except Exception as exc:
            exception = repr(exc)
            break
    result = dict(record=rec, success=bool(info['success']) and exception is None,
                  steps=len(actions), max_progress=max(i['progress'] for i in infos),
                  final_progress=info['progress'], min_distance=min(i['distance'] for i in infos),
                  min_progress=min(i['progress'] for i in infos),
                  contact=any(i['direct_contact'] for i in infos), exception=exception,
                  invalid_joint=any(not i['valid_joint'] for i in infos),
                  wall_seconds=time.monotonic() - started)
    if store:
        arrays = dict(obs=np.asarray(observations, np.float32), actions=np.asarray(actions, np.float32),
                      rewards=np.asarray(rewards, np.float32),
                      success=np.asarray([i['success'] for i in infos[1:]], bool),
                      terminated=np.zeros(len(actions), bool),
                      truncated=np.asarray([i == 499 for i in range(len(actions))], bool),
                      q=np.asarray([i['q'] for i in infos]), progress=np.asarray([i['progress'] for i in infos]),
                      direct_contact=np.asarray([i['direct_contact'] for i in infos]),
                      distance=np.asarray([i['distance'] for i in infos]), **snap)
        return result, arrays
    return result


def logged_rollout(env, rec, stage):
    ledger = read_json(LEDGER) if LEDGER.exists() else []
    existing = [r for r in ledger if r['record']['episode_id'] == rec['episode_id']]
    if existing:
        return existing[0]
    assert sum(r['record']['task_name'] == env.task for r in ledger) < 600
    result = rollout(env, rec)
    result['stage'] = stage
    ledger.append(result)
    atomic_json(LEDGER, ledger)
    print(env.task, stage, rec['region'], rec['task_params']['yaw_degrees'], result['success'],
          result['steps'], round(result['max_progress'], 3), 'contact', result['contact'], flush=True)
    return result


def probe():
    for task in ('door',):
        env = GeometryEnv(task)
        try:
            for i, yaw in enumerate([0, -10, 10, -20, 20, -35, 35, -50, 50]):
                rec = record(task, 22000000 + (task == 'door') * 1000 + i, yaw, episode_id=f'probe_v5_{task}_{i}')
                logged_rollout(env, rec, 'probe_v5')
        finally:
            env.close()


def calibrate():
    output = ROOT / 'configs/round2/calibration.json'
    if output.exists():
        return read_json(output)
    started = time.monotonic()
    # Drawer controller is unchanged from v5; retain its complete A-tier check.
    selections = {'drawer': read_json(ROOT/'artifacts/round2/calibration_v5_superseded.json')['tasks']['drawer']}
    for ti, task in ((1, 'door'),):
        env = GeometryEnv(task)
        summaries = []
        try:
            for tier_index, (tier, angles) in enumerate(TIERS.items()):
                plan = []
                rng = np.random.default_rng(23000000 + ti * 10000 + tier_index * 1000)
                for level, magnitude in zip(('near','mid','far'), angles[1:]):
                    for sign in (-1, 1):
                        for region in ('train','left','right'):
                            for repeat in range(2):
                                plan.append((level, sign * magnitude + rng.uniform(-2, 2), region))
                for yaw in (0, -angles[0], angles[0]):
                    for region in ('train', 'left', 'right'):
                        plan.append(('train_boundary', yaw, region))
                for level, magnitude in zip(('near','mid','far'), angles[1:]):
                    for sign in (-1, 1):
                        for edge in (-2, 2):
                            plan.append((level, sign * magnitude + edge, 'train'))
                records = []
                for i, (level, yaw, region) in enumerate(plan):
                    rec = record(task, 23000000 + ti * 10000 + tier_index * 1000 + i, yaw, region,
                                 f'cal_v6_{task}_{tier}_{i:03d}')
                    rec['level'] = level
                    records.append(rec)
                atomic_json(ROOT / f'artifacts/round2/calibration_plan_v6_{task}_{tier}.json', records)
                results, zero = [], []
                for rec in records:
                    results.append(logged_rollout(env, rec, f'tier_{tier}_v6'))
                    zstart = time.monotonic()
                    _, info = env.reset(rec)
                    initial_contacts = [(int(c.geom1), int(c.geom2), float(c.dist)) for c in env.data.contact[:env.data.ncon]
                                        if c.dist < -.002 and (c.geom1 in env.handle_geoms or c.geom2 in env.handle_geoms or
                                        env.model.geom_bodyid[c.geom1] == env.model.body('drawer_link' if task == 'drawer' else 'door_link').id or
                                        env.model.geom_bodyid[c.geom2] == env.model.body('drawer_link' if task == 'drawer' else 'door_link').id)]
                    progress = [info['progress']]
                    for _ in range(100):
                        _, _, _, _, info = env.step(np.zeros(4))
                        progress.append(info['progress'])
                    zero.append(dict(episode_id=rec['episode_id'], max_abs_progress=float(np.max(np.abs(progress))),
                                     initial_penetrations=initial_contacts, wall_seconds=time.monotonic()-zstart))
                atomic_json(ROOT / f'artifacts/round2/zero_actions_v6_{task}_{tier}.json', zero)
                rates = {level: float(np.mean([r['success'] for r in results if r['record']['level'] == level]))
                         for level in ('near','mid','far')}
                valid = all(z['max_abs_progress'] < .02 and not z['initial_penetrations'] for z in zero)
                # MuJoCo soft constraints admit <=1% endpoint compliance, but
                # gross violations are never counted as successful episodes.
                valid = valid and not any(r['invalid_joint'] or r['exception'] for r in results)
                summary = dict(tier=tier, angles=angles, success_rate=float(np.mean([r['success'] for r in results])),
                               yaw_rates=rates, physical_valid=valid, count=len(results),
                               signs={str(s): float(np.mean([r['success'] for r in results if np.sign(r['record']['task_params']['yaw_degrees']) == s])) for s in (-1,1)})
                summary['feasible'] = valid and summary['success_rate'] >= .9 and min(rates.values()) >= .8
                summaries.append(summary)
                atomic_json(ROOT / f'artifacts/round2/calibration_summary_v6_{task}.json', summaries)
                print('TIER SUMMARY', task, summary, flush=True)
                if summary['feasible']:
                    selections[task] = summary
                    break
            else:
                selections[task] = dict(blocked=True, attempts=summaries)
        finally:
            env.close()
    result = dict(protocol_version='round2-v1', tasks=selections, wall_seconds=time.monotonic()-started,
                  expert_version='v6_door_slow_approach_tangent_feedback_drawer_v5_slow_finish',
                  rear_rail_world_y=1.15, success_progress=.75, consecutive_steps=3, target_progress=.8,
                  joint_compliance_tolerance=.01)
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    import sys
    calibrate() if '--full' in sys.argv else probe()
