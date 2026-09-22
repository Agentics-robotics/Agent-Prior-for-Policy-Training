# vla_pi05 — fine-tuning pi0.5 on the real Franka data

Everything runs in the isolated Pixi environment `vla` (Python 3.12, torch 2.7.1+cu126,
LeRobot 0.6.1 with the PyTorch port of pi0.5). The default environment (Rounds 1–3) is untouched.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi install -e vla                 # once
pixi run -e vla vla-doctor          # GPUs, packages, raw data, assets
pixi run -e vla vla-test            # unit tests (no GPU needed)
```

## Pipeline

```
raw episodes ──manifest──▶ segments (instruction, [start,end)) ──convert──▶ LeRobotDataset ──train──▶ checkpoint
   (30/10 Hz)                                                            (10 Hz, 256x144)     (4 GPUs)      │
                                                                                                          eval / serve
```

| step | command | notes |
|---|---|---|
| 1. manifest | `pixi run -e vla vla-manifest --name all_tasks` | one segment per raw episode, placeholder instruction = teleop `notes`; writes `configs/vla/manifests/all_tasks.json` |
| 2. convert | `pixi run -e vla vla-convert --manifest all_tasks --name franka_pi05` | → `/home/storage/tianrunhu/vla_pi05/datasets/franka_pi05_{train,val}` |
| 3. pretrained | `pixi run -e vla python -m vla_pi05.cli prepare-pretrained` | downloads `lerobot/pi05_base` (14.5 GB) and resolves the tokenizer (see below) |
| 4. train | `pixi run -e vla vla-train --dataset franka_pi05 --run-name franka_v1` | GPUs 4–7, preset `expert_only`; add `--smoke` for a 1-GPU 10-step check, `--dry-run` to print the command |
| 5. eval | `pixi run -e vla vla-eval --run franka_v1 --dataset franka_pi05` | open-loop MAE/MSE on the held-out split → `runs/franka_v1/eval/` |
| 6. serve | `pixi run -e vla vla-serve --run franka_v1 --port 8000` | websocket server; robot PC uses `vla_pi05/client.py` or `openpi_client.WebsocketClientPolicy` |

Runs live in `/home/storage/tianrunhu/vla_pi05/runs/<run>/` with `vla_run.json` (command, git sha,
dataset digest), `vla_conversion.json` (what state/action mean), `train.log` and
`train/checkpoints/<step>/pretrained_model/` (LeRobot checkpoint incl. processors). Resume with
`vla-train --run-name <run> --resume`.

## Push task from the cut data (`/home/storage/oscar/cut_v3`)

```bash
pixi run -e vla python -m vla_pi05.cli manifest-cut --name push_cut_v3      # 22 segments -> prompts, 2 held out
pixi run -e vla vla-convert --manifest push_cut_v3 --name push_cut_v3
pixi run -e vla vla-train --dataset push_cut_v3 --preset lora --run-name push_lora_v1
pixi run -e vla vla-eval  --run push_lora_v1 --dataset push_cut_v3
```

Prompt per segment (see `cut_import.PROMPTS`): `<verb> the <piece> <target>, goal at x<gx> y<gy>` —
e.g. `push the O-shaped ring piece to the right and set it above the W, goal at x46 y59`. The piece is
named by shape (+ a location cue for the two crossbar pieces), the target relative to the arrangement,
`stage`/`align` mark temporary vs final placements, and the goal is the cut's goal-box centre in percent
of the 1280×720 third image (the deployment-time condition). `--no-goal-coords` drops the numbers.
The 1 s exclusion windows at every cut boundary are trimmed so no action loss lands inside them.

## What the model sees / outputs (defaults)

* **Rate**: 10 Hz (30 Hz recordings are subsampled by nearest timestamp). Inference at 10 Hz is the target.
* **Images**: `observation.images.base_0_rgb` ← `third/`, `observation.images.left_wrist_0_rgb` ← `wrist/`,
  stored 256×144 (aspect-preserving 1/5 of 1280×720); pi0.5 letterboxes to 224×224 itself. The names match
  the pretrained pi0.5 features, the third pretrained slot (`right_wrist_0_rgb`) is padded as an empty camera.
* **State** (8): 7 joint positions `q` + gripper opening `gripper_width_m / 0.14` ∈ [0,1].
* **Action** (8, chunk of 50 = 5 s): `joint_abs` — absolute joint positions + gripper at the *next* 10 Hz frame.
  Alternatives: `--action-space joint_delta` or `ee_abs` (xyz + 6D rotation + gripper). The runner's
  `chunk_to_targets` returns absolute targets in every case.
* **Language**: the `task` string per segment (currently the placeholder from the manifest).

## GPUs

The `vla` environment sets `CUDA_VISIBLE_DEVICES=4,5,6,7` (physical GPUs 4–7 are reserved for this work;
0–3 stay with Round 3). Inside the env `cuda:0` therefore means physical GPU 4. Override per command with
`--gpus 4,5` (train) or `--device cuda:1` (eval/serve), or `VLA_GPUS` for the default.

**Inference always runs on a single visible GPU.** `vla-eval`, `vla-serve` and `infer-raw` translate
`--device cuda:N` into `CUDA_VISIBLE_DEVICES=<physical N>` + `cuda:0` before torch loads. With several
GPUs visible, pi0.5 inference on `cuda:1..3` returns wrong or NaN actions (verified 2026-09-21; the
same checkpoint is correct on `cuda:0` or on any card exposed alone). DDP training is not affected: LeRobot logs the loss averaged over all ranks, and it decreased normally (0.49 -> 0.03) on `push_lora_v1`.

## Presets for 4× A5000 (24 GB)

| preset | trainable | parallelism | batch/GPU | status |
|---|---|---|---|---|
| `expert_only` (default) | action expert + projections, VLM frozen, bf16 | DDP | 8 | smoke-tested |
| `lora` | LoRA r=32 on expert attention + projections | DDP | 8 | untested |
| `full` | all params, bf16 | FSDP v2 (`configs/vla/accelerate_fsdp.yaml`) | 4 | untested, experimental |

Override with `--steps`, `--batch-size`, `--lr`, `--gpus 4,5`, or forward raw LeRobot flags after `--`.
Non-smoke runs log to wandb project `vla_pi05` (needs `wandb login` or `WANDB_API_KEY`; without a key the
run logs offline and can be pushed later with `wandb sync <run>/train/wandb/...`). Photometric image
augmentation is on by default (`--no-image-aug` to disable).

## Tokenizer note

pi0.5 tokenizes prompts with the PaliGemma tokenizer from `google/paligemma-3b-pt-224`, a **gated**
HF repo. Without an approved `HF_TOKEN`, `prepare-pretrained` converts Google's public SentencePiece
model (`gs://big_vision/paligemma_tokenizer.model`) to a HF fast tokenizer (token ids verified identical)
and patches the policy preprocessor to point at it. Set `VLA_TOKENIZER_PATH` to force a specific one.

## When the cut / labelled data arrives

1. If the recordings keep today's folder layout, only the **manifest** changes: produce
   `configs/vla/manifests/<name>.json` (or a CSV with `episode,start,end,instruction[,split]`, loaded by
   `Manifest.load`) with one row per (instruction, `[start,end)` frame range at the raw rate).
   Then `vla-convert --manifest <name> --name <dataset>` and train.
2. If the layout changes, edit only `raw_episode.py` (`RawEpisode.load`, `image_path`, `discover_episodes`).
   `tests/test_vla_pi05.py` builds a synthetic episode you can mirror to the new format.
3. `manifest.validate()` refuses empty instructions, so nothing trains on unlabeled segments by accident.

## Robot-side protocol

`vla-serve` speaks openpi's websocket/msgpack protocol. Request (either form):

```python
{"images": {"base_0_rgb": u8[H,W,3], "left_wrist_0_rgb": u8[H,W,3]}, "state": f32[8], "instruction": "..."}
{"observation/base_0_rgb": ..., "observation/left_wrist_0_rgb": ..., "observation/state": ..., "prompt": "..."}
```

Response: `{"actions": f32[50,8], "targets": f32[50,8], "server_timing": {...}}` where `targets` are
absolute joint positions + gripper opening to track at 10 Hz. Send `{"type": "reset"}` between episodes.
Images can be any RGB size; the server resizes with padding. Run `python -m vla_pi05.cli infer-raw --run <run>
--episode flip_egg/episode_... --index 100` to sanity-check a checkpoint on a recorded frame.

## Env vars

`VLA_RAW_DATA_ROOT` (default `/home/storage/tianrunhu/real_robot_data`), `VLA_WORK_ROOT`
(`/home/storage/tianrunhu/vla_pi05`), `VLA_PRETRAINED_REPO` (`lerobot/pi05_base`), `VLA_TOKENIZER_PATH`, `HF_TOKEN`.
