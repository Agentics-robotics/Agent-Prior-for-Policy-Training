"""Auditable progress/final reports and real-simulator trajectory replays.

Development feedback never opens sealed states. Test charts and videos require
the global freeze. Replays execute stored actions and do not call policies or
experts, select checkpoints, change outcomes, or add scored evaluation episodes.
"""
from __future__ import annotations

import csv
import contextlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from relative_dp.utils import atomic_json, object_hash, read_json, sha256
from .common import CANDIDATES, NS, R3, ROOT, TASKS, now, planned_runs, run_id
from .evaluate import SPLITS, _checked_result, _verify_snapshot, freeze_selection, validate_test_gate

COLORS = {"B0": "#333333", "P1": "#2878b5", "P2": "#e07a27", "P3": "#33975b",
          "P4": "#b04fa0", "initial-selected": "#8463a7"}
FRAME_RULE = dict(
    version="round3-feedback-frames-v1", stage="dev", train_n=5,
    candidates=["B0", "P1", "P2", "P3"], split="C", physical_steps=[50, 200], maximum_frames=8,
    state_selection="First C state in manifest order whose valid N5 models disagree on success; otherwise first C state",
    unavailable_candidates="Retain invalid candidates with their reason; select using all valid completed N5 models",
    terminated_policy="Hold final real simulator frame when the requested physical step exceeds episode length",
    image_source="Replay cached actions in the real simulator; no policy/expert calls; verify every replay observation",
)


def freeze_feedback_rule():
    path = R3 / "design_records" / "feedback_frame_rule.json"
    if path.exists():
        assert read_json(path) == FRAME_RULE, "Feedback frame rule changed"
    else:
        assert not any((R3 / "design_records").glob("*/feedback/bundle.json")), "Cannot define frame rule after feedback"
        atomic_json(path, FRAME_RULE)
    return path


def _train_rows(identifier):
    path = R3 / "checkpoints" / identifier / "train.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _diagnostic_at(row, step):
    calls = [x for x in row.get("inference", []) if x["at_step"] <= step]
    diag = calls[-1].get("diagnostics", {}) if calls else {}
    return {k: v for k, v in diag.items() if any(s in k.lower() for s in ("route", "phase", "stage", "mode", "skill"))}


def _annotated_frame(env, row, step, label, requested=None):
    from PIL import Image, ImageDraw
    image = Image.fromarray(np.flipud(env.native.render()).copy())
    output = Image.new("RGB", (image.width, image.height + 78), "black")
    output.paste(image, (0, 78))
    draw = ImageDraw.Draw(output)
    obs_goal = np.asarray(env.native._target_pos)
    clock = f"step {step} | {step * env.native.dt:.3f}s"
    if requested is not None and requested > step:
        clock += f" | final held at requested step {requested}"
    lines = [label, clock, f"goal {np.round(obs_goal, 3).tolist()} | episode_success={int(row['success'])}",
             "route/phase " + str(_diagnostic_at(row, step) or "flat / not emitted")]
    for i, line in enumerate(lines):
        draw.text((6, 4 + i * 18), line[:95], fill="white")
    return np.asarray(output)


def _replay_frames(task, reset, row, requested_steps, label):
    """Collect a few real frames, checking the complete replay state sequence."""
    from .environment import TaskEnv
    path = ROOT / row["trajectory_path"]
    assert sha256(path) == row["trajectory_hash"]
    with np.load(path, allow_pickle=False) as arrays:
        obs, actions = arrays["obs"].copy(), arrays["actions"].copy()
    env = TaskEnv(task, render_mode="rgb_array")
    selected, wanted = {}, {min(int(s), len(actions)) for s in requested_steps}
    try:
        current, _ = env.reset(reset)
        _verify_snapshot(env, current, reset)
        for step in range(len(actions) + 1):
            np.testing.assert_allclose(current, obs[step], atol=1e-6, rtol=0,
                                       err_msg=f"Visual replay changed {row['episode_id']} step {step}")
            if step in wanted:
                selected[step] = _annotated_frame(env, row, step, label)
            if step < len(actions):
                current, *_ = env.step(actions[step].copy())
    finally:
        env.close()
    return {s: selected[min(s, len(actions))] for s in requested_steps}


def feedback_bundle(task):
    """Create the one authorized N5 feedback packet from development artifacts."""
    from PIL import Image
    from .diagnostics import freeze_rule, summarize_result
    rule = freeze_feedback_rule()
    diagnostic_rule=freeze_rule()
    directory = R3 / "design_records" / task / "feedback"
    output = directory / "bundle.json"
    if output.exists():
        old = read_json(output)
        for path, digest in old["bound_files"].items():
            assert sha256(ROOT / path) == digest, f"Frozen feedback changed: {path}"
        return old
    runs = {r["candidate_id"]: r for r in planned_runs() if r["task"] == task and r["train_n"] == 5}
    results, invalid = {}, []
    for candidate in CANDIDATES:
        run = runs[candidate]
        if run["status"] == "invalid":
            invalid.append(dict(candidate_id=candidate, reason=run.get("invalid_reason", run.get("reason"))))
            assert invalid[-1]["reason"]
        else:
            path = R3 / "dev_results" / run["run_id"] / "complete.json"
            assert path.exists(), f"Feedback waits for all valid initial N5 models: {path}"
            results[candidate] = _checked_result(path)
    assert results, "No valid models available for feedback"
    manifest_path = R3 / "data" / task / "manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["complete"] is True
    records = manifest["dev"]["C"]
    by_id = {c: {r["episode_id"]: r for r in result["records"]} for c, result in results.items()}
    selected = next((rec for rec in records if len({rows[rec["episode_id"]]["success"] for rows in by_id.values()}) > 1), records[0])
    directory.mkdir(parents=True, exist_ok=True)
    frames, summaries, bound = [], {}, [rule, diagnostic_rule, manifest_path, R3 / "design_records" / task / "proposal.json"]
    for candidate, result in results.items():
        row = by_id[candidate][selected["episode_id"]]
        imgs = _replay_frames(task, selected, row, FRAME_RULE["physical_steps"], f"{task} | {candidate} N5 | development C")
        for step, frame in imgs.items():
            image_path = directory / f"{candidate}_step_{step:03d}.png"
            Image.fromarray(frame).save(image_path)
            frames.append(dict(candidate_id=candidate, episode_id=selected["episode_id"], requested_step=step,
                               actual_step=min(step, row["steps"]), success=row["success"],
                               path=str(image_path.relative_to(ROOT)), sha256=sha256(image_path)))
            bound.append(image_path)
        rid = result["identity"]["run_id"]
        train = _train_rows(rid)
        training_file = directory / f"{candidate}_training_curve.json"
        atomic_json(training_file, train)
        diagnostic_file=directory/f'{candidate}_offline_diagnostics.json'
        diagnostic=summarize_result(task,result)
        atomic_json(diagnostic_file,diagnostic)
        bound.append(diagnostic_file)
        bound.extend([training_file, ROOT / row["trajectory_path"], R3 / "dev_results" / rid / "complete.json"])
        summaries[candidate] = dict(run_id=rid, metrics=result["metrics"], ood_success_rate=result["ood_success_rate"],
                                    selected_episode={k: row[k] for k in ("episode_id", "success", "steps", "failure_category", "termination_reason", "replans")},
                                    parameter_count=result["parameter_count"], training_curve=str(training_file.relative_to(ROOT)),
                                    offline_diagnostics=diagnostic['splits'],diagnostic_file=str(diagnostic_file.relative_to(ROOT)))
    assert len(frames) <= 8
    _plot_training_curves({c: _train_rows(r["identity"]["run_id"]) for c, r in results.items()}, directory / "training_curves.png", task + " N5 feedback")
    bound.append(directory / "training_curves.png")
    packet = dict(task=task, created_at=now(), stage="dev", train_n=5, frame_rule_sha256=sha256(rule),
                  original_proposal=str((R3 / "design_records" / task / "proposal.json").relative_to(ROOT)),
                  results=summaries, invalid=invalid, frames=frames, frame_count=len(frames),
                  scored_development_rollouts=sum(len(r["records"]) for r in results.values()),
                  additional_policy_calls_for_visuals=0, visualization_replay_steps=sum(by_id[c][selected["episode_id"]]["steps"] for c in results),
                  selected_episode_id=selected["episode_id"], contains_locked_test=False,
                  instructions="First state what the evidence supports and cannot support; then save one concrete P4 or no_revision. P4 has one main revision intent and all actual changes. No more feedback cycles.",
                  bound_files={str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(bound))})
    atomic_json(output, packet)
    return packet


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    return plt


def _plot_training_curves(curves, path, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for candidate, rows in curves.items():
        if rows:
            ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], label=candidate, color=COLORS.get(candidate), alpha=.8)
    ax.set(title=title, xlabel="Optimizer updates", ylabel="Reported training loss")
    ax.grid(alpha=.2)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _load_results(stage):
    if stage == "test":
        validate_test_gate()
    location = R3 / ("dev_results" if stage == "dev" else "locked_test_results")
    results = {}
    for run in planned_runs():
        path = location / run["run_id"] / "complete.json"
        if path.exists():
            result = _checked_result(path)
            assert result["identity"]["stage"] == stage
            results[run["run_id"]] = result
    return results


def paired_ood_delta(candidate, baseline, draws=5000):
    """Stratified paired snapshot bootstrap, C/E equally weighted; percentage points."""
    rng = np.random.default_rng(301709)
    means, bootstrap = [], np.zeros(draws)
    for split in ("C", "E"):
        left = {r["episode_id"]: r for r in candidate["records"] if r["split"] == split}
        right = {r["episode_id"]: r for r in baseline["records"] if r["split"] == split}
        assert left.keys() == right.keys()
        for key in left:
            assert left[key]["initial_state_hash"] == right[key]["initial_state_hash"]
            for field in ("reset_record_hash", "inference_seed"):
                assert left[key].get(field) == right[key].get(field), f"Paired rollout {field} differs: {key}"
        values = np.asarray([int(left[key]["success"]) - int(right[key]["success"]) for key in sorted(left)], float)
        means.append(float(values.mean()))
        bootstrap += values[rng.integers(0, len(values), (draws, len(values)))].mean(1) / 2
    return dict(delta_pp=100 * float(np.mean(means)), paired_bootstrap_95ci_pp=(100 * np.quantile(bootstrap, [.025, .975])).tolist(),
                draws=draws, seed=301709, scope="Evaluation snapshot sampling for these fixed models; excludes training seed/data-subset randomness")


def _selected(selection, task, n, kind):
    item = selection.get("groups", {}).get(f"{task}:n{n}", {}).get(kind)
    return item["run_id"] if item else None


def result_figures(results, selection, stage):
    plt = _plt()
    directory = R3 / "figures" / stage
    directory.mkdir(parents=True, exist_ok=True)
    generated = []
    for task in TASKS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for ax, split in zip(axes, SPLITS):
            for candidate in [*CANDIDATES, "initial-selected"]:
                values = []
                for n in NS:
                    rid = _selected(selection, task, n, "initial_selected") if candidate == "initial-selected" else run_id(task, candidate, n)
                    result = results.get(rid)
                    values.append(result["metrics"][split]["success_rate"] if result else np.nan)
                label = candidate + (" (provisional)" if candidate == "initial-selected" and not selection.get("frozen") else "")
                ax.plot(NS, values, "D--" if candidate == "initial-selected" else "o-", color=COLORS[candidate],
                        label=label, linewidth=2.1 if candidate == "initial-selected" else 1.2)
            ax.set(title=split, xlabel="Complete demonstrations N", xticks=NS, ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Success rate")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{task} | {stage} | initial-selected uses development only")
        fig.tight_layout()
        path = directory / f"{task}_data_curve.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
    labels = [(task, candidate) for task in TASKS for candidate in ("P1", "P2", "P3", "P4")]
    heat = np.full((len(labels), len(NS)), np.nan)
    for i, (task, candidate) in enumerate(labels):
        for j, n in enumerate(NS):
            value, baseline = results.get(run_id(task, candidate, n)), results.get(run_id(task, "B0", n))
            if value and baseline:
                heat[i, j] = 100 * (value["ood_success_rate"] - baseline["ood_success_rate"])
    fig, ax = plt.subplots(figsize=(7, 11))
    maxval = max(1., float(np.nanmax(np.abs(heat)))) if np.isfinite(heat).any() else 1.
    plot = ax.imshow(np.ma.masked_invalid(heat), cmap="RdBu", vmin=-maxval, vmax=maxval, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(NS)):
            value = heat[i, j]
            rgba = plot.cmap(plot.norm(value)) if np.isfinite(value) else (1., 1., 1., 1.)
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(j, i, f"{value:+.1f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8,
                    color="white" if luminance < .5 else "black")
    ax.set(xticks=range(len(NS)), xticklabels=NS, yticks=range(len(labels)),
           yticklabels=[f"{t} / {c}" for t, c in labels], xlabel="N", title=f"{stage}: mean C/E success difference from B0 (pp)")
    fig.colorbar(plot, ax=ax, label="Percentage points")
    fig.tight_layout()
    path = directory / "ood_delta_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    generated.append(path)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        for candidate, values in [
            ("initial-selected", [results.get(_selected(selection, task, n, "initial_selected")) for n in (5, 20)]),
            ("P4", [results.get(run_id(task, "P4", n)) for n in (5, 20)]),
        ]:
            ax.plot([5, 20], [r["ood_success_rate"] if r else np.nan for r in values], "o-", label=candidate, color=COLORS[candidate])
        revision_path = R3 / "design_records" / task / "revision.json"
        revision = read_json(revision_path) if revision_path.exists() else {}
        decision = revision.get("decision", "feedback pending")
        ax.set(title=f"{task} | {decision}", xticks=[5, 20], xlabel="N", ylabel="Mean C/E success", ylim=(-.03, 1.03))
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle(f"{stage}: optional feedback revision; P4 uses N5 dev feedback and up to 2 extra trainings/task"
                 + ("\nInitial selection is provisional until the global freeze" if not selection.get("frozen") else ""))
    fig.tight_layout()
    path = directory / "feedback_revision_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    generated.append(path)
    return generated


def coverage_figures(stage):
    if stage == "test":
        validate_test_gate()
    plt = _plt()
    from matplotlib.patches import Rectangle
    generated = []
    directory = R3 / "figures" / stage
    directory.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        path = R3 / "data" / task / "manifest.json"
        if not path.exists() or not read_json(path).get("complete"):
            continue
        data = read_json(path)
        protocol = read_json(R3 / "protocol_and_task_manifests" / f"{task}.json")
        factors = protocol["distribution"]["factors"]
        states = data["dev"] if stage == "dev" else read_json(R3 / "locked_test_states" / task / "manifest.json")["test"]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)
        for ax, n in zip(axes, NS):
            for region, pairs, color in [
                ("train/IID", [("low", "low"), ("high", "high")], "#929292"),
                ("C", [("low", "high"), ("high", "low")], "#2878b5"),
                ("E", [("extrap_" + a, "extrap_" + b) for a in ("low", "high") for b in ("low", "high")], "#e07a27"),
            ]:
                for index, (a, b) in enumerate(pairs):
                    x, y = factors["a"][a], factors["b"][b]
                    ax.add_patch(Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], facecolor=color, alpha=.12,
                                           edgecolor=color, label=region + " region" if index == 0 else None))
            for split, marker, color in [("IID", ".", "gray"), ("C", "+", "#2878b5"), ("E", "x", "#e07a27")]:
                recs = states[split]
                ax.scatter([r["factors"]["a"] for r in recs], [r["factors"]["b"] for r in recs], s=16, marker=marker, color=color, label=stage + " " + split)
            recs = data["train20"][:n]
            ax.scatter([r["factors"]["a"] for r in recs], [r["factors"]["b"] for r in recs], s=28, facecolors="none", edgecolors="black", label=f"D{n}")
            ax.set(title=f"N={n}", xlabel=factors["a"]["name"] + f" ({factors['a']['units']})")
            ax.grid(alpha=.15)
        axes[0].set_ylabel(factors["b"]["name"] + f" ({factors['b']['units']})")
        axes[-1].legend(fontsize=6)
        fig.suptitle(task + " | exact sampled factors; nuisance parameters omitted from 2D projection")
        fig.tight_layout()
        path = directory / f"{task}_layout_coverage.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(path)
    return generated


def _paired_video(task, reset, left, right, output):
    import imageio.v2 as imageio
    from .environment import TaskEnv
    identity = dict(task=task, episode_id=reset["episode_id"], initial_state_hash=reset["initial_state_hash"],
                    trajectories=[left["trajectory_hash"], right["trajectory_hash"]], renderer_sha256=sha256(__file__),
                    selection="Fixed C manifest order plus first available failure; no outcome maximization")
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists():
        old = read_json(metadata_path)
        assert old["identity"] == identity and old["video_sha256"] == sha256(output)
        return old
    cleanup_errors = []

    def close_resource(resource):
        try:
            resource.close()
        except BaseException as error:
            # Close every resource; preserve an active replay/creation error.
            cleanup_errors.append(error)

    with contextlib.ExitStack() as resources:
        envs = []
        for _ in range(2):
            env = TaskEnv(task, render_mode="rgb_array")
            envs.append(env)
            resources.callback(close_resource, env)
        rows, trajectories, current = [left, right], [], []
        for row in rows:
            path = ROOT / row["trajectory_path"]
            assert sha256(path) == row["trajectory_hash"]
            with np.load(path, allow_pickle=False) as arrays:
                trajectories.append(dict(obs=arrays["obs"].copy(), actions=arrays["actions"].copy()))
        max_steps = max(len(t["actions"]) for t in trajectories)
        dt, stride = float(envs[0].native.dt), 4
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.stem + ".partial.mp4")
        writer = imageio.get_writer(temporary, fps=1 / (dt * stride), codec="libx264", quality=7, macro_block_size=2,
                                   ffmpeg_params=["-threads", "1"])
        resources.callback(close_resource, writer)
        for env in envs:
            obs, _ = env.reset(reset)
            _verify_snapshot(env, obs, reset)
            current.append(obs)
        for step in range(max_steps + 1):
            frames = []
            for i, (env, row, trajectory) in enumerate(zip(envs, rows, trajectories)):
                actual_step = min(step, len(trajectory["actions"]))
                np.testing.assert_allclose(current[i], trajectory["obs"][actual_step], atol=1e-6, rtol=0,
                                           err_msg=f"Video replay differs {row['episode_id']} {step}")
                if step % stride == 0 or step == max_steps:
                    frames.append(_annotated_frame(env, row, actual_step,
                        f"{task} | {row['identity']['candidate_id']} N{row['identity']['train_n']} | {row['split']}", requested=step))
            if frames:
                writer.append_data(np.concatenate(frames, axis=1))
            for i, (env, trajectory) in enumerate(zip(envs, trajectories)):
                if step < len(trajectory["actions"]):
                    current[i], *_ = env.step(trajectory["actions"][step].copy())
    if cleanup_errors:
        raise cleanup_errors[0]
    temporary.replace(output)
    result = dict(identity=identity, path=str(output.relative_to(ROOT)), video_sha256=sha256(output),
                  fps=1 / (dt * stride), rendering_stride=stride, physical_dt=dt,
                  policy_calls=0, scored_evaluation_episodes_added=0,
                  visualization_replay_steps=sum(len(t["actions"]) for t in trajectories),
                  success=[row["success"] for row in rows], termination=[row["termination_reason"] for row in rows])
    atomic_json(metadata_path, result)
    return result


def test_videos(results, selection, *, tasks=None, write_manifest=True):
    """Replay the fixed pairs; task workers leave the shared manifest to report."""
    tasks = list(TASKS if tasks is None else tasks)
    assert tasks and len(tasks) == len(set(tasks)) and set(tasks) <= set(TASKS)
    assert not write_manifest or tasks == list(TASKS), "Partial task workers must not replace the full video manifest"
    validate_test_gate()
    directory = R3 / "videos"
    records = []
    for task in tasks:
        baseline = results.get(run_id(task, "B0", 20))
        chosen = _selected(selection, task, 20, "initial_selected")
        candidate = results.get(chosen)
        if baseline is None or candidate is None:
            records.append(dict(task=task, status="pending", reason="N20 B0 and initial-selected test results required"))
            continue
        states = read_json(R3 / "locked_test_states" / task / "manifest.json")["test"]
        left = {r["episode_id"]: r for r in baseline["records"]}
        right = {r["episode_id"]: r for r in candidate["records"]}
        fixed = states["C"][0]
        records.append(_paired_video(task, fixed, left[fixed["episode_id"]], right[fixed["episode_id"]], directory / f"{task}_fixed_C_N20.mp4"))
        ordered = [r for s in ("C", "E", "IID") for r in states[s]]
        failure = next((r for r in ordered if r["episode_id"] != fixed["episode_id"] and
                        not (left[r["episode_id"]]["success"] and right[r["episode_id"]]["success"])), None)
        if failure is not None:
            records.append(_paired_video(task, failure, left[failure["episode_id"]], right[failure["episode_id"]], directory / f"{task}_first_failure_N20.mp4"))
        else:
            # If N20 has no other failure, retain one at a smaller N
            # using a declared order, without changing any policy or score.
            fallback = None
            for n in (5, 2, 10):
                low_baseline = results.get(run_id(task, 'B0', n))
                low_candidate = results.get(_selected(selection, task, n, 'initial_selected'))
                if low_baseline is None or low_candidate is None:
                    continue
                low_left = {r['episode_id']: r for r in low_baseline['records']}
                low_right = {r['episode_id']: r for r in low_candidate['records']}
                failed = next((r for r in ordered if not (low_left[r['episode_id']]['success'] and low_right[r['episode_id']]['success'])), None)
                if failed is not None:
                    fallback = _paired_video(task, failed, low_left[failed['episode_id']], low_right[failed['episode_id']], directory / f'{task}_first_failure_N{n}.mp4')
                    break
            if fallback:
                records.append(fallback)
            else:
                records.append(dict(task=task, status="no_additional_failure", reason="No failure in the other N20 states or in paired N5/N2/N10 B0/initial-selected results", fixed_video_includes_failure=not(left[fixed['episode_id']]['success'] and right[fixed['episode_id']]['success'])))
    if write_manifest:
        atomic_json(directory / "manifest.json", dict(created_at=now(), records=records,
                    comparison="B0 versus P1/P2/P3 initial-selected by dev: first C state at N20 plus first other N20 failure in C/E/IID order; if absent, first failure at N5 then N2 then N10"))
    return records


def _format_rate(value, interval=False):
    text = f"{100 * value['success_rate']:.1f}%"
    if interval:
        text += " [" + ", ".join(f"{100*x:.1f}" for x in value["binomial_95ci"]) + "]"
    return text


def data_quality_and_debug():
    quality=[]
    for task in TASKS:
        base=R3/'data'/task
        path=base/'manifest.json'
        if not path.exists():
            continue
        data=read_json(path)
        calibration=R3/'calibration'/task/'results.json'
        checks=read_json(calibration) if calibration.exists() else []
        if data.get('calibration_hash'):
            assert sha256(calibration)==data['calibration_hash']
        replay=read_json(base/'replay_audit.json') if (base/'replay_audit.json').exists() else {}
        attempts=read_json(base/'collection_attempts.json') if (base/'collection_attempts.json').exists() else []
        quality.append(dict(task=task,complete=data.get('complete',False),
            calibration={s:dict(successes=sum(r['success'] for r in checks if r['record']['split']==s),
                                episodes=sum(r['record']['split']==s for r in checks)) for s in SPLITS},
            calibration_failures=[dict(episode_id=r['episode_id'],reason=r['termination_reason']) for r in checks if not r['success']],
            collection_attempts=len(attempts),failed_collection_attempts=sum(not r['success'] for r in attempts),
            collection_failure_reasons=dict(Counter(r.get('termination_reason', 'unavailable') for r in attempts if not r['success'])),
            accepted_demonstrations=len(data.get('train20',[])),replay_passed=replay.get('passed',False),
            full_replays=replay.get('full_demos',0),maximum_replay_observation_error=max([r['max_error'] for r in replay.get('records',[])] or [None])))
    sessions=[]
    for path in sorted((R3/'debug').glob('*/sessions.jsonl')):
        for row in [json.loads(line) for line in path.read_text().splitlines() if line.strip()]:
            sessions.append(dict(path=str(path.relative_to(ROOT)),updates=row['end_step']-row['start_step'],wall_seconds=row['wall_seconds']))
    reference=R3/'audits'/'debug_uninterrupted_reference.json'
    extra=read_json(reference) if reference.exists() else {}
    debug=dict(completed_session_updates=sum(r['updates'] for r in sessions),
        independent_reference_updates=extra.get('steps',0),
        accounted_optimizer_updates=sum(r['updates'] for r in sessions)+extra.get('steps',0),
        completed_session_wall_seconds=sum(r['wall_seconds'] for r in sessions),
        independent_reference_active_seconds=extra.get('elapsed_seconds'),sessions=sessions,
        scope='Debug session ledger plus separate uninterrupted recovery reference; CPU one-update mathematical checks and test-suite optimizer work are additional unaggregated diagnostics. No debug work counts as formal training.')
    smoke=R3/'audits'/'debug_closed_loop.json'
    if smoke.exists():
        debug['infrastructure_closed_loop_steps']=sum(r['steps'] for r in read_json(smoke)['records'])
    return quality,debug


def feedback_accounting(runs, trained, dev):
    """Separate the one feedback opportunity and any additional P4 work by task."""
    rows = []
    for task in TASKS:
        directory = R3 / "design_records" / task
        revision_path, bundle_path = directory / "revision.json", directory / "feedback" / "bundle.json"
        revision = read_json(revision_path) if revision_path.exists() else {}
        bundle = read_json(bundle_path) if bundle_path.exists() else {}
        candidate = revision.get("candidate") or {}
        identifiers = [r["run_id"] for r in runs if r["task"] == task and r["candidate_id"] == "P4"]
        completed = [trained[rid] for rid in identifiers if rid in trained]
        debug_sessions = []
        for rid in identifiers:
            path = R3 / "debug" / rid / "sessions.jsonl"
            if path.exists():
                debug_sessions.extend(dict(path=str(path.relative_to(ROOT)), **json.loads(line))
                                      for line in path.read_text().splitlines() if line.strip())
        rows.append(dict(task=task, decision=revision.get("decision", revision.get("choice", "pending")),
                         p4_base_candidate_id=candidate.get("base_candidate_id"),
                         p4_principal_revision_intent=candidate.get("principal_revision_intent"),
                         feedback_cycles_used=revision.get("feedback_cycles_used", int(bool(revision))),
                         feedback_development_episodes=bundle.get("scored_development_rollouts", 0),
                         feedback_packet_frames=bundle.get("frame_count", 0),
                         p4_planned_trainings=len(identifiers), p4_completed_trainings=len(completed),
                         p4_development_episodes=sum(len(dev[rid]["records"]) for rid in identifiers if rid in dev),
                         p4_train_wall_seconds=sum(r["total_session_wall_seconds"] for r in completed),
                         p4_gpu_active_work_seconds=sum(r["gpu_active_work_seconds"] for r in completed),
                         p4_debug_session_updates=sum(r["end_step"] - r["start_step"] for r in debug_sessions),
                         p4_debug_session_wall_seconds=sum(r["wall_seconds"] for r in debug_sessions),
                         p4_debug_sessions=debug_sessions,
                         p4_debug_cost_scope="Completed P4 debug session ledger only; included in global debug totals, not formal training. CPU backward/feature audits with zero optimizer updates and other unaggregated checks remain separate; their wall time is unavailable here."))
    return rows


def feedback_effects(feedback, results, selection):
    """Keep the frozen revision base distinct from this N's development selection."""
    comparisons = []
    for item in feedback:
        if item["decision"] != "P4":
            continue
        for n in (5, 20):
            revised_id = run_id(item["task"], "P4", n)
            revised = results.get(revised_id)
            base = item.get("p4_base_candidate_id")
            references = [("base-candidate", run_id(item["task"], base, n) if base else None),
                          ("initial-selected", _selected(selection, item["task"], n, "initial_selected"))]
            for kind, identifier in references:
                reference = results.get(identifier)
                comparisons.append(dict(task=item["task"], train_n=n, p4_run_id=revised_id,
                    reference_kind=kind, reference_run_id=identifier,
                    reference_candidate_id=reference["identity"]["candidate_id"] if reference else base if kind == "base-candidate" else None,
                    reference_ood_success_rate=reference["ood_success_rate"] if reference else None,
                    p4_ood_success_rate=revised["ood_success_rate"] if revised else None,
                    paired_ood_delta=paired_ood_delta(revised, reference) if revised and reference else None))
    return comparisons


def shared_extremes(results, quality):
    """Descriptive exact floors/ceilings after all four initial scores exist."""
    rows = []
    by_task = {r['task']: r for r in quality}
    for task in TASKS:
        for n in NS:
            group = [results.get(run_id(task, c, n)) for c in CANDIDATES]
            if not all(group):
                continue
            for split in ('C', 'E'):
                scores = [r['metrics'][split] for r in group]
                floor = all(m['successes'] == 0 for m in scores)
                ceiling = all(m['successes'] == m['n'] for m in scores)
                if not (floor or ceiling):
                    continue
                rows.append(dict(task=task, train_n=n, split=split,
                    observation='all zero' if floor else 'all perfect',
                    iid_successes={c: r['metrics']['IID']['successes'] for c, r in zip(CANDIDATES, group)},
                    iid_episodes=group[0]['metrics']['IID']['n'],
                    independent_calibration=by_task.get(task, {}).get('calibration', {}).get(split),
                    demonstration_replay_passed=by_task.get(task, {}).get('replay_passed')))
    return rows


def generate(stage=None, make_videos=True):
    """Write a truthful partial report, or a validated final report when ready."""
    gate_exists = (R3 / "global_freeze.json").exists()
    stage = stage or ("test" if gate_exists else "dev")
    assert stage in ("dev", "test")
    if stage == "test":
        validate_test_gate()
    runs = planned_runs()
    results = _load_results(stage)
    selection = read_json(R3 / "selection.json") if gate_exists else freeze_selection(require_all=False)
    trained, invalid, pending = {}, [], []
    for run in runs:
        path = R3 / "checkpoints" / run["run_id"] / "complete.json"
        if run["status"] == "invalid":
            invalid.append(run)
        elif path.exists():
            complete = read_json(path)
            assert complete["step"] == 20000 and complete["debug"] is False
            for checkpoint in complete["checkpoints"].values():
                assert sha256(ROOT / checkpoint["path"]) == checkpoint["sha256"]
            trained[run["run_id"]] = complete
        else:
            pending.append(run["run_id"])
    evaluations_complete = stage == "test" and not pending and all(r["run_id"] in results for r in runs if r["status"] != "invalid")
    figures = result_figures(results, selection, stage) + coverage_figures(stage) if results else []
    videos = test_videos(results, selection) if stage == "test" and make_videos and results else []
    final = evaluations_complete and make_videos and not any(v.get('status')=='pending' for v in videos) and bool(videos)
    rows, deltas = [], {}
    for run in runs:
        rid = run["run_id"]
        result = results.get(rid)
        row = dict(run_id=rid, task=run["task"], candidate_id=run["candidate_id"], train_n=run["train_n"],
                   status="invalid" if run["status"] == "invalid" else "evaluated" if result else "pending",
                   reason=run.get("invalid_reason", run.get("reason", "")))
        if result:
            for split in SPLITS:
                metric = result["metrics"][split]
                row.update({split + "_success_rate": metric["success_rate"], split + "_successes": metric["successes"],
                            split + "_episodes": metric["n"], split + "_ci_low": metric["binomial_95ci"][0], split + "_ci_high": metric["binomial_95ci"][1]})
            row.update(ood_success_rate=result["ood_success_rate"], iid_minus_ood=result["metrics"]["IID"]["success_rate"] - result["ood_success_rate"],
                       parameter_count=result["parameter_count"], mean_latency_ms=sum(r["inference_seconds"] for r in result["records"]) * 1000 / max(1, sum(r["replans"] for r in result["records"])),
                       mean_replans=float(np.mean([r["replans"] for r in result["records"]])))
            baseline = results.get(run_id(run["task"], "B0", run["train_n"]))
            if baseline:
                deltas[rid] = paired_ood_delta(result, baseline)
                row["ood_delta_B0_pp"] = deltas[rid]["delta_pp"]
        if rid in trained:
            row.update(train_wall_seconds=trained[rid]["total_session_wall_seconds"],
                       gpu_active_work_seconds=trained[rid]["gpu_active_work_seconds"],
                       training_chunk_draws=trained[rid].get("samples_drawn"))
            if "parameter_count" not in row:
                row["parameter_count"] = trained[rid].get("parameter_count", trained[rid].get("parameters"))
        rows.append(row)
    directory = R3 / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    from .diagnostics import summarize_result
    offline={rid:summarize_result(result['identity']['task'],result) for rid,result in results.items()}
    atomic_json(directory/f'{stage}_offline_diagnostics.json',offline)
    quality,debug=data_quality_and_debug()
    extremes = shared_extremes(results, quality)
    dev = results if stage == "dev" else _load_results("dev")
    feedback = feedback_accounting(runs, trained, dev)
    revision_effects = feedback_effects(feedback, results, selection)
    atomic_json(directory/'data_quality_and_debug.json',dict(data_quality=quality,debug=debug))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    (directory / f"{stage}_all_results.csv").write_text(stream.getvalue())
    summary = dict(created_at=now(), stage=stage, final=final, completed_formal_runs=len(trained),
                   completed_main_runs=sum(not r["feedback_revision"] and r["run_id"] in trained for r in runs),
                   completed_p4_runs=sum(r["feedback_revision"] and r["run_id"] in trained for r in runs),
                   planned_runs=len(runs), evaluated_runs=len(results), scored_episodes=sum(len(r["records"]) for r in results.values()),
                   pending_training=pending, invalid=invalid, results=rows, paired_ood_deltas=deltas,
                   figures=[str(p.relative_to(ROOT)) for p in figures], videos=videos)
    summary.update(data_quality=quality,debug=debug,feedback_accounting=feedback,feedback_effects=revision_effects,shared_floor_ceiling=extremes,
                   selection_frozen=bool(selection.get("frozen")),
                   offline_diagnostics_path=str((directory/f'{stage}_offline_diagnostics.json').relative_to(ROOT)))
    atomic_json(directory / f"{stage}_summary.json", summary)
    lines = ["# Round 3 " + ("final report" if final else "progress report"), "",
             f"Status: **{'validated final locked-test results' if final else 'incomplete; results below are '+stage+' only'}**. "
             f"Validated formal training: **{len(trained)}/{len(runs)}** systems, each 20,000 optimizer updates; "
             f"{len(results)} complete {stage} evaluations ({summary['scored_episodes']} scored episodes). "
             f"Invalid configurations: {len(invalid)}. Training seed: 0 only.", "",
             "Design backend is the current Codex session and actual Codex subagents, with no separate API dependency. "
             "The coordinating session has historical/source exposure; isolated initial designers receive D2 only. "
             "This is a Codex-assisted exploratory experiment, not a claim of a clean unseen-task evaluation of the coordinator. "
             "Exact model/session identifiers, token usage and API price are unavailable unless recorded in the design artifacts.", "",
             "The deployed policies are state-input policies using the same declared numerical observations. "
             "Real simulator images inform design and development feedback; these experiments do not measure learned visual generalization. "
             "P1/P2/P3 are task-local candidate IDs: the same ID across tasks can denote different mechanisms and architectures.", ""]
    effects = [r for r in rows if r["candidate_id"] != "B0" and "ood_delta_B0_pp" in r]
    main_effects = [r for r in effects if r["candidate_id"] != "P4"]
    if main_effects:
        counts = Counter("helped" if r["ood_delta_B0_pp"] > 0 else "hurt" if r["ood_delta_B0_pp"] < 0 else "tied" for r in main_effects)
        largest = max(main_effects, key=lambda r: r["ood_delta_B0_pp"])
        smallest = min(main_effects, key=lambda r: r["ood_delta_B0_pp"])
        lines += [f"Observed initial-candidate/N comparisons against B0 on mean C/E: {dict(counts)}. "
                  f"Largest delta: {largest['task']} N{largest['train_n']} {largest['candidate_id']} {largest['ood_delta_B0_pp']:+.1f} pp; "
                  f"smallest delta: {smallest['task']} N{smallest['train_n']} {smallest['candidate_id']} {smallest['ood_delta_B0_pp']:+.1f} pp. "
                  "All per-distribution scores and negative effects appear below. Differences describe these trained models and do not identify causal mechanisms.", ""]
    else:
        lines += ["No paired candidate/B0 result is available yet; improved generalization has not been established.", ""]
    decisions = Counter(r["decision"] for r in feedback)
    lines += [f"Feedback decisions: {dict(decisions)}; additional P4 trainings completed: {summary['completed_p4_runs']}. "
              "No-revision decisions add no P4 treatment; any P4 results and their additional costs are separated below.", ""]
    for task in TASKS:
        proposal = R3 / "design_records" / task / "proposal.json"
        revision = proposal.with_name("revision.json")
        if proposal.exists():
            doc = read_json(proposal)
            designer=doc.get('designer')
            if isinstance(designer,dict):
                designer=designer.get('name',str(designer))
            exposure=doc.get('context_exposure')
            if isinstance(exposure,dict):
                exposure=exposure.get('scope',exposure.get('inherited_instruction_scope','Task-isolated D2 evidence; see saved exposure record'))
            lines += [f"**{task} design.** Designer: {designer}. Initial demonstration exposure: D2 (two complete demonstrations, 16 real frames). "
                      f"{exposure} Full evidence and provenance: [saved proposal](design_records/{task}/proposal.json).", ""]
            for candidate in doc["candidates"]:
                hypotheses=candidate['knowledge_hypotheses']
                if isinstance(hypotheses,list):
                    hypotheses=' '.join(f"[{h.get('source', 'unspecified source')}] {h.get('claim',str(h))}" if isinstance(h,dict) else str(h) for h in hypotheses)
                implementation_path = R3 / "configs" / task / (candidate["candidate_id"] + ".json")
                links = []
                if implementation_path.exists():
                    implementation = read_json(implementation_path)
                    links.append(f"[implementation config]({implementation_path.relative_to(R3)})")
                    audit = implementation.get("implementation_audit")
                    if isinstance(audit, dict):
                        audit = audit.get("path")
                    if audit and (ROOT / audit).exists():
                        links.append(f"[implementation checks and limitations]({(ROOT / audit).relative_to(R3)})")
                lines += [f"- {candidate['candidate_id']}: {hypotheses} Predicted gain: {candidate['predicted_generalization_gain']} "
                          + "; ".join(links)]
            lines.append("")
        else:
            lines += [f"**{task}:** initial proposal pending.", ""]
        amendment=proposal.with_name('initial_feasibility_revision.json')
        if amendment.exists():
            change=read_json(amendment)
            lines += [f"Initial feasibility revision: 1 saved amendment; original proposal preserved. "
                      f"Details: [amendment](design_records/{task}/initial_feasibility_revision.json). "
                      f"Reason: {change.get('reason',change.get('rationale','See saved audit'))}.", ""]
        if revision.exists():
            decision = read_json(revision)
            lines += [f"Feedback decision: {decision.get('decision', decision.get('choice'))}. "
                      f"Rationale: {decision.get('rationale', decision.get('evidence_assessment', 'See saved revision.json'))}.", ""]
    lines += ["## Complete candidate results", "", "All valid candidates and negative effects are retained. NA means pending, invalid, or (for P4 N2/N10) outside the authorized matrix. "
              "Brackets are 95% binomial Wilson intervals for fixed trained models. C/E are equally weighted.", "",
              "| Task | N | Candidate | IID [95% CI] | C [95% CI] | E [95% CI] | OOD | IID−OOD pp | Δ B0 pp | Status/reason |",
              "|---|---:|---|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        result = results.get(row["run_id"])
        rates = [_format_rate(result["metrics"][s], True) for s in SPLITS] if result else ["NA"] * 3
        ood = f"{100*result['ood_success_rate']:.1f}%" if result else "NA"
        gap = f"{100*row['iid_minus_ood']:+.1f}" if result else "NA"
        delta = f"{row['ood_delta_B0_pp']:+.1f}" if "ood_delta_B0_pp" in row else "NA"
        reason = str(row["reason"]).replace("|", "/") if row["reason"] else row["status"]
        lines.append(f"| {row['task']} | {row['train_n']} | {row['candidate_id']} | {' | '.join(rates)} | {ood} | {gap} | {delta} | {reason} |")
    lines += ["", "## Development selection and observed effects", "",
              "Selection uses this N's dev mean C/E, then IID, fewer parameters, and candidate ID. "
              "initial-selected is chosen programmatically among P1/P2/P3; final-system includes B0 and eligible P4. P4 never participates at N2/N10. "
              + ("These choices are frozen before locked testing." if selection.get("frozen") else
                 "Every choice and B0 fallback below is provisional: missing evaluations and outstanding feedback decisions can change eligibility and rankings. "
                 "A provisional B0 choice does not establish that it beats unfinished candidates.") + " "
              "Development-selected results reuse the states used for selection and therefore have selection bias; their bootstrap intervals do not adjust for selection.", "",
              "| Task | N | initial-selected | final-system | B0 fallback |", "|---|---:|---|---|---|"]
    for task in TASKS:
        for n in NS:
            group = selection.get("groups", {}).get(f"{task}:n{n}", {})
            initial, system = group.get("initial_selected"), group.get("final_system")
            qualifier = " (provisional)" if not selection.get("frozen") else ""
            fallback = str(group["b0_fallback"]) + qualifier if system else "pending"
            lines.append(f"| {task} | {n} | {initial['candidate_id'] + qualifier if initial else 'pending'} | "
                         f"{system['candidate_id'] + qualifier if system else 'pending'} | {fallback} |")
    if effects:
        counts = Counter("helped" if r["ood_delta_B0_pp"] > 0 else "hurt" if r["ood_delta_B0_pp"] < 0 else "tied" for r in effects)
        lines += ["", f"Observed candidate/N comparisons: {dict(counts)}. These are descriptive effects for one trained seed."]
        for row in effects:
            result, baseline = results[row["run_id"]], results[run_id(row["task"], "B0", row["train_n"])]
            iid_delta = 100 * (result["metrics"]["IID"]["success_rate"] - baseline["metrics"]["IID"]["success_rate"])
            lines += [f"- {row['task']} N{row['train_n']} {row['candidate_id']}: OOD {row['ood_delta_B0_pp']:+.1f} pp; IID {iid_delta:+.1f} pp; "
                      f"paired OOD 95% bootstrap {deltas[row['run_id']]['paired_bootstrap_95ci_pp']}. "
                      + ("Both IID and OOD improved, so the gain may include improved fitting." if row["ood_delta_B0_pp"] > 0 and iid_delta > 0 else "")]
    else:
        lines += ["", "No completed paired candidate/B0 evaluation is available; no claim of improved generalization can yet be made."]
    if extremes:
        lines += ["", "## Shared floors and ceilings", "",
                  "These exact-zero or exact-perfect C/E groups include all four initial models B0/P1/P2/P3. "
                  "All reported models completed 20k updates. IID scores and the pretraining expert calibration provide context; "
                  "calibration uses different states and does not prove reachability of every evaluation snapshot. "
                  "Data replays and semantic audits are retained; these scores alone do not justify changing splits or success criteria.", "",
                  "| Task | N | Split | Initial-model scores | IID successes B0/P1/P2/P3 | Expert calibration | Demo replay |",
                  "|---|---:|---|---|---|---|---|"]
        for item in extremes:
            calibration = item['independent_calibration']
            expert = f"{calibration['successes']}/{calibration['episodes']}" if calibration else 'unavailable'
            iid = '/'.join(str(item['iid_successes'][c]) for c in CANDIDATES)
            lines.append(f"| {item['task']} | {item['train_n']} | {item['split']} | {item['observation']} | "
                         f"{iid} (each /{item['iid_episodes']}) | {expert} | {item['demonstration_replay_passed']} |")
    lines += ["", "## Feedback revision and additional costs", "",
              "Feedback episodes below are the existing N5 B0/P1/P2/P3 development episodes made available to the feedback designer; "
              "they are a subset of main-model development totals, not extra rollouts. Packet frames count prepared evidence. "
              "Each implemented P4 adds two independent from-scratch formal trainings, N5 and N20, under the unchanged per-model budget. "
              "P4 training time includes completed P4 systems only; pending or interrupted work is excluded.", "",
              "| Task | Decision | Feedback cycles | Feedback episodes | Packet frames | P4 trained/planned | P4 dev episodes | P4 train wall h | P4 worker active h |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in feedback:
        lines.append(f"| {item['task']} | {item['decision']} | {item['feedback_cycles_used']} | "
                     f"{item['feedback_development_episodes']} | {item['feedback_packet_frames']} | "
                     f"{item['p4_completed_trainings']}/{item['p4_planned_trainings']} | {item['p4_development_episodes']} | "
                     f"{item['p4_train_wall_seconds']/3600:.3f} | {item['p4_gpu_active_work_seconds']/3600:.3f} |")
    lines += ["", "P4 implementation debug is additional work, already included in the global debug ledger below. "
              "These are completed debug-session updates and wall seconds only. CPU feature/backward audits with zero optimizer updates, "
              "other unaggregated checks, and unfinished work have no invented timing estimate.", "",
              "| Task | Frozen base candidate | Main revision intent | P4 debug updates | P4 debug wall s |",
              "|---|---|---|---:|---:|"]
    for item in feedback:
        if item["decision"] != "P4":
            continue
        intent = str(item['p4_principal_revision_intent'] or 'unavailable').replace('|', '/').replace('\n', ' ')
        lines.append(f"| {item['task']} | {item['p4_base_candidate_id'] or 'unavailable'} | {intent} | "
                     f"{item['p4_debug_session_updates']} | {item['p4_debug_session_wall_seconds']:.3f} |")
    lines += ["", f"P4 comparisons on {stage} mean C/E; initial selection is "
              + ("frozen." if selection.get("frozen") else "provisional.") + " The base-candidate is the original design actually revised; "
              "initial-selected is the programmatic P1/P2/P3 winner for this N and can differ from that base. "
              "Both are descriptive fixed-model comparisons; neither alone identifies a causal mechanism. "
              "A no_revision decision has no P4 effect estimate. NA means a required validated result is unavailable.", "",
              "| Task | N | Reference kind | Reference candidate | Reference OOD | P4 OOD | P4 Δ reference pp [paired 95% CI] |",
              "|---|---:|---|---|---:|---:|---|"]
    for item in revision_effects:
        delta = item['paired_ood_delta']
        comparison = f"{delta['delta_pp']:+.1f} [{delta['paired_bootstrap_95ci_pp'][0]:+.1f}, {delta['paired_bootstrap_95ci_pp'][1]:+.1f}]" if delta else "NA"
        reference_rate = f"{100*item['reference_ood_success_rate']:.1f}%" if item['reference_ood_success_rate'] is not None else "NA"
        revised_rate = f"{100*item['p4_ood_success_rate']:.1f}%" if item['p4_ood_success_rate'] is not None else "NA"
        lines.append(f"| {item['task']} | {item['train_n']} | {item['reference_kind']} | {item['reference_candidate_id'] or 'pending'} | "
                     f"{reference_rate} | {revised_rate} | {comparison} |")
    if not any(item["decision"] == "P4" for item in feedback):
        lines += ["", "No P4 treatment has been proposed; feedback improvement has not been tested."]
    lines += ["", "## Equal task averages", "", "Averages below require all six tasks for the same N/method. Missing tasks are never silently dropped. "
              "P1/P2/P3 average the corresponding task-local proposal slots; they do not evaluate one shared architecture across six tasks. "
              + ("Selection is frozen." if selection.get("frozen") else "Selected-system averages remain provisional until selection is frozen."), "",
              "| N | Method | Mean task IID | Mean task C | Mean task E | Mean task OOD |", "|---:|---|---:|---:|---:|---:|"]
    for n in NS:
        for candidate in [*CANDIDATES, "initial-selected", "final-system"]:
            ids = [(_selected(selection, t, n, "initial_selected" if candidate == "initial-selected" else "final_system")
                    if candidate in ("initial-selected", "final-system") else run_id(t, candidate, n)) for t in TASKS]
            group = [results.get(rid) for rid in ids]
            if all(group):
                values = [100 * float(np.mean([r["metrics"][s]["success_rate"] for r in group])) for s in SPLITS]
                ood = 100 * float(np.mean([r["ood_success_rate"] for r in group]))
                lines.append(f"| {n} | {candidate} | {' | '.join(f'{v:.1f}%' for v in values)} | {ood:.1f}% |")
            else:
                lines.append(f"| {n} | {candidate} | NA | NA | NA | NA |")
    lines += ["", "## Data and computation", "", "| Task | D2 transitions | D5 | D10 | D20 | Training cells |", "|---|---:|---:|---:|---:|---|"]
    for task in TASKS:
        path = R3 / "data" / task / "manifest.json"
        data = read_json(path) if path.exists() else {}
        if data.get("complete"):
            counts = [sum(r["transitions"] for r in data["train20"][:n]) for n in NS]
            lines.append(f"| {task} | {' | '.join(map(str, counts))} | LL/HH nested alternating |")
        else:
            lines.append(f"| {task} | NA | NA | NA | NA | data preparation incomplete |")
    lines += ['', '### Data quality', '', '| Task | Expert IID | Expert C | Expert E | Collection attempts / failures | Accepted / fully replayed | Max replay error |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for item in quality:
        rates=[f"{item['calibration'][s]['successes']}/{item['calibration'][s]['episodes']}" for s in SPLITS]
        lines.append(f"| {item['task']} | {' | '.join(rates)} | {item['collection_attempts']} / {item['failed_collection_attempts']} | "
                     f"{item['accepted_demonstrations']} / {item['full_replays']} (passed={item['replay_passed']}) | {item['maximum_replay_observation_error']} |")
    lines += ['', 'Failed demonstration collection reasons by task: ' + '; '.join(
        f"{item['task']}: {item['collection_failure_reasons'] or 'none'}" for item in quality) + '.']
    lines += ['', 'Expert calibration, accepted demonstrations and replay checks are infrastructure/data evidence, not learned-policy evaluations. '
              'Expert failures remain in the calibration record and do not by themselves prove a layout impossible.', '',
              f"Accounted debug optimizer updates: {debug['accounted_optimizer_updates']} "
              f"({debug['completed_session_updates']} completed session-ledger updates + {debug['independent_reference_updates']} independent recovery-reference updates). "
              f"Debug session wall time: {debug['completed_session_wall_seconds']:.2f}s; separate reference active time: {debug['independent_reference_active_seconds']}s. "
              f"Infrastructure closed-loop steps recorded: {debug.get('infrastructure_closed_loop_steps',0)}. {debug['scope']}", '',
              '### Offline descriptive failure diagnostics', '',
              'These summaries use saved rollout observations only, plus cabinet progress already present in the saved trajectory. '
              'The frozen descriptive thresholds are 5 cm hand/interaction proximity, 3 cm observed object lift, and 0.1 recorded cabinet progress. '
              'Proximity does not establish contact or grasp. Distances to declared observed goal points do not replace native success rules. '
              'No policy inputs, termination categories or scored outcomes change. See the per-episode '
              f'[diagnostic JSON](reports/{stage}_offline_diagnostics.json) for distances, lift and original termination reasons.', '',
              '| Run | Split | Descriptive labels | Mean minimum hand distance (m) | Mean peak lift (m) | Mean final goal-point distance (m) |',
              '|---|---|---|---:|---:|---:|']
    for rid,diagnostic in offline.items():
        for split,group in diagnostic['splits'].items():
            m=group['mean']
            lines.append(f"| {rid} | {split} | {group['descriptive_labels']} | {m['min_hand_to_interaction_m']} | "
                         f"{m['peak_observed_object_or_tool_lift_m']} | {m['final_observed_goal_point_distance_m']} |")
    initial_dev_episodes = sum(len(result["records"]) for result in dev.values() if result["identity"]["candidate_id"] != "P4")
    revision_dev_episodes = sum(len(result["records"]) for result in dev.values() if result["identity"]["candidate_id"] == "P4")
    lines += ["", f"Formal training wall time summed across systems: {sum(r['total_session_wall_seconds'] for r in trained.values()) / 3600:.3f} hours. "
              f"Sum of per-worker synchronized training time: {sum(r['gpu_active_work_seconds'] for r in trained.values()) / 3600:.3f} worker-hours. "
              "Concurrent system times overlap, including when two workers share a GPU; they are not physical GPU occupancy, experiment elapsed time or exclusive GPU kernel time. "
              "These totals cover validated completed systems only; running/interrupted formal sessions are not included.", "",
              f"Main-model development episodes: {initial_dev_episodes}; P4 development episodes: {revision_dev_episodes}. "
              f"Prepared feedback packet frames: {sum(p['feedback_packet_frames'] for p in feedback)} "
              f"across {sum(p['feedback_packet_frames'] > 0 for p in feedback)} tasks. "
              f"Additional P4 trainings completed: {summary['completed_p4_runs']}/12 maximum. "
              "Replayed visualization actions add zero policy inference calls and zero scored episodes.", "",
              "Per-model parameters, measured train wall/synchronized-worker time, mean inference latency and replanning counts are in "
              f"[the full CSV](reports/{stage}_all_results.csv). Equal updates do not establish equal FLOPs. "
              "Every N uses its own complete trajectories and normalization statistics; deployment inputs are shared. "
              "Training chunk draws are also recorded in the CSV; each transition defines one training window and repeated draws do not create new demonstrations. "
              "Augmentation and derived-label definitions are in each saved proposal and implementation config. "
              "Auxiliary label counts are reconstructed separately from the same frozen D_N artifacts; they are not additional demonstrations.", ""]
    label_audit = R3 / "reports" / "dataset_accounting.json"
    if label_audit.exists():
        lines += [f"Exact auxiliary tensor shapes, label counts and phase frequencies: [current-D_N tensor inventory]({label_audit.relative_to(R3)}). This reconstruction adds no optimizer updates or rollouts and does not imply that pending runs have trained.", ""]
    else:
        lines += ["Exact auxiliary label-count aggregation is pending the current-D_N tensor inventory.", ""]
    if figures:
        lines += ["## Figures and real simulator videos", ""]
        for path in figures:
            lines += [f"![{path.stem}]({path.relative_to(R3)})", ""]
        for video in videos:
            if video.get("path"):
                lines += [f"- [{Path(video['path']).name}]({Path(video['path']).relative_to('round3')})"]
    lines += ["", "## Limitations and integrity", "",
              "One training seed and one nested demonstration family do not establish stability across seeds or data selection. "
              "Wilson/bootstrap intervals reflect the sampled deployment states for fixed trained models only. "
              "P4 uses N5 development feedback and additional search/training, reported separately. Drawer/door have historical development exposure. "
              "These comparisons cannot establish superiority to human design or random search, universal cross-task priors, or that each task needs a distinct prior. "
              "Zero/ceiling outcomes and invalid candidates remain visible; no test redistribution or checkpoint selection follows outcomes.", "",
              "Frozen earlier-round data, configurations and available reports are retained. Some historical checkpoint/run directories were absent from the received migration; "
              "their old report claims cannot be independently revalidated from those missing artifacts.", ""]
    if not final:
        lines += [f"Outstanding: {len(pending)} formal trainings and {sum(r['status'] != 'invalid' and r['run_id'] not in results for r in runs)} {stage} evaluations; "
                  "initial proposals, one feedback decision per task, global selection freeze, locked test evaluation and final visuals must all be completed before final conclusions.", ""]
    report = R3 / "ROUND3_REPORT.md"
    report.write_text("\n".join(lines))
    atomic_json(directory / "integrity.json", dict(report_sha256=sha256(report), final=final, stage=stage,
                all_completed_results_validated=True, scored_test_access_requires_global_freeze=True,
                report_path=str(report.relative_to(ROOT)), summary_path=str((directory / f"{stage}_summary.json").relative_to(ROOT))))
    return dict(report=str(report), final=final, trained=len(trained), evaluated=len(results), stage=stage)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-task", choices=TASKS)
    parser.add_argument("--stage", choices=("dev", "test"))
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    print(feedback_bundle(args.feedback_task) if args.feedback_task else generate(args.stage, not args.no_videos))
