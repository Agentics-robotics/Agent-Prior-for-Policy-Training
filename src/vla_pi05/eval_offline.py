"""Open-loop evaluation of a checkpoint on a held-out LeRobotDataset split.

For every ``stride``-th frame the policy predicts a full action chunk from the stored
observation; we compare it with the recorded future actions (unnormalized, dataset units)
and report masked MSE/MAE per action dimension and per horizon step. Outputs:

    <out>/eval.json   metrics + provenance
    <out>/eval.png    error-vs-horizon curve, per-dim bars, sample trajectories
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

from .convert_dataset import load_sidecar

log = logging.getLogger(__name__)


def evaluate(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    out_dir: str | Path,
    stride: int = 10,
    batch_size: int = 16,
    max_samples: int | None = None,
    device: str = "cuda:0",
    num_inference_steps: int | None = None,
    seed: int = 0,
) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader, Subset, default_collate

    from .inference import Pi05Runner

    checkpoint_dir, dataset_dir, out_dir = Path(checkpoint_dir), Path(dataset_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = load_sidecar(dataset_dir)
    runner = Pi05Runner(checkpoint_dir, device=device, num_inference_steps=num_inference_steps)
    policy, pre, post = runner.policy, runner.preprocessor, runner.postprocessor
    chunk = runner.chunk_size
    fps = int(sidecar["convert_config"]["action_space"]["target_fps"])
    names = sidecar["convert_config"]["action_space"]["action_names"]

    ds = LeRobotDataset(sidecar["repo_id"], root=dataset_dir, delta_timestamps={"action": [i / fps for i in range(chunk)]})
    idx = list(range(0, len(ds), max(1, stride)))
    if max_samples:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(idx, size=min(max_samples, len(idx)), replace=False).tolist())
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=default_collate)

    sq_sum = np.zeros((chunk, len(names)))
    abs_sum = np.zeros((chunk, len(names)))
    cnt = np.zeros((chunk, 1))
    examples = []
    n_done = 0
    t_infer = 0.0
    for batch in loader:
        gt = batch["action"].numpy()  # (B, chunk, D) unnormalized
        pad = batch["action_is_pad"].numpy()  # (B, chunk)
        batch_in = {k: v for k, v in batch.items() if k != "action" and k != "action_is_pad"}
        for cam in ds.meta.camera_keys:
            if batch_in[cam].dtype == torch.uint8:
                batch_in[cam] = batch_in[cam].float() / 255.0
        import time

        t0 = time.perf_counter()
        proc = pre(batch_in)
        with torch.inference_mode():
            pred = policy.predict_action_chunk(proc)
        pred = post(pred).float().cpu().numpy()
        t_infer += time.perf_counter() - t0
        valid = (~pad)[:, :, None].astype(np.float64)
        err = pred - gt
        sq_sum += (err**2 * valid).sum(0)
        abs_sum += (np.abs(err) * valid).sum(0)
        cnt += valid[:, :, :1].sum(0)
        if len(examples) < 4:
            for b in range(min(gt.shape[0], 4 - len(examples))):
                examples.append({"gt": gt[b].tolist(), "pred": pred[b].tolist(), "pad": pad[b].tolist(), "task": batch["task"][b]})
        n_done += gt.shape[0]
        log.info("evaluated %d/%d samples", n_done, len(idx))

    cnt = np.maximum(cnt, 1)
    mse_step_dim = sq_sum / cnt
    mae_step_dim = abs_sum / cnt
    metrics = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_dir),
        "dataset_dir": str(dataset_dir),
        "n_samples": int(n_done),
        "stride": stride,
        "chunk_size": chunk,
        "fps": fps,
        "action_names": names,
        "infer_s_per_sample": t_infer / max(n_done, 1),
        "mse_overall": float(mse_step_dim.mean()),
        "mae_overall": float(mae_step_dim.mean()),
        "mse_per_dim": dict(zip(names, mse_step_dim.mean(0).tolist())),
        "mae_per_dim": dict(zip(names, mae_step_dim.mean(0).tolist())),
        "mse_per_step": mse_step_dim.mean(1).tolist(),
        "mae_per_step": mae_step_dim.mean(1).tolist(),
        "mae_first_step_per_dim": dict(zip(names, mae_step_dim[0].tolist())),
        "examples": examples,
    }
    (out_dir / "eval.json").write_text(json.dumps(metrics, indent=1))
    try:
        _plot(metrics, out_dir / "eval.png")
    except Exception as e:  # plotting must never fail the eval
        log.warning("plot failed: %s", e)
    log.info("MAE overall %.4f  MSE overall %.5f  (%d samples) -> %s", metrics["mae_overall"], metrics["mse_overall"], n_done, out_dir)
    return metrics


def _plot(m: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = m["action_names"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(np.arange(m["chunk_size"]) / m["fps"], m["mae_per_step"])
    axes[0].set_xlabel("horizon (s)")
    axes[0].set_ylabel("MAE (dataset units)")
    axes[0].set_title("error vs horizon")
    axes[1].bar(range(len(names)), [m["mae_per_dim"][n] for n in names])
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_title("MAE per dim")
    if m["examples"]:
        ex = m["examples"][0]
        gt, pred = np.array(ex["gt"]), np.array(ex["pred"])
        t = np.arange(gt.shape[0]) / m["fps"]
        for d in range(gt.shape[1]):
            axes[2].plot(t, gt[:, d], color=f"C{d}", lw=1.2)
            axes[2].plot(t, pred[:, d], color=f"C{d}", lw=1.2, ls="--")
        axes[2].set_title(f"sample chunk (solid=gt, dashed=pred)\n{ex['task'][:50]}")
        axes[2].set_xlabel("horizon (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
