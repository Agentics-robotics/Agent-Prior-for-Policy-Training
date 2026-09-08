# 第一轮：Diffusion Policy 相对坐标实验

实施规格保存在 [ROUND1_SPEC.txt](ROUND1_SPEC.txt)。本项目比较 20 条相同专家示范下，原始状态与可逆相对坐标表示；每个任务只训练一个 seed。

本轮已实际完成：4×20,000 updates、240 个 dev episode、160 个正式测试 episode。四个模型的 IID/OOD 均为 20/20；成功率没有观察到差异。Drawer OOD 的成功平均完成步数为 raw 135.70、relative 93.55，这是需要多训练 seed 复验的次要观察。完整结果见 [ROUND1_REPORT.md](ROUND1_REPORT.md)，交付审计见 [artifacts/delivery_audit.json](artifacts/delivery_audit.json)。

## 环境与执行

本机 Pixi 位于 `/home/users/oscar/.pixi/bin/pixi`。若 PATH 尚未配置，可执行 `export PATH="/home/users/oscar/.pixi/bin:$PATH"`。所有项目运行命令必须通过 Pixi，不使用独立 venv、Conda 或裸 pip。项目目录为 `/home/users/oscar/Agent_Training/agent_training`。

```bash
pixi install --locked
pixi run doctor
pixi run test
pixi run collect-round1
pixi run train-round1
pixi run evaluate-round1
pixi run report-round1
```

完整执行及恢复入口：

```bash
pixi run round1
```

单个 run 的恢复（会核对已有状态，并从最新完整 checkpoint 继续）：

```bash
pixi run train-round1 --run-id drawer_raw_n20_s0
```

主入口持有进程锁，避免同时启动两个正式优化器。训练每 1000 updates 以及 5k/10k/20k 保存原子 checkpoint；恢复 optimizer、EMA、loader/diffusion RNG 与采样位置。中断用正常的 SIGINT/SIGTERM，让程序在当前 update 后保存。`PROGRESS.md` 保存 PID，`logs/round1.log` 保存完整流水线输出；先检查原进程，避免重复启动。

每个正式 run 依次使用独立 Pixi 子进程，避免编译运行时状态从 debug inference 传入下一次训练。首次运行出现过速度异常，原始日志及 65 步逐位一致性恢复验证均保留，详见 [RUNTIME_RECOVERY.md](docs/RUNTIME_RECOVERY.md)。该调整未改变冻结模型、配置、数据或训练预算。

全部产物齐备后的独立审计命令：

```bash
pixi run python scripts/verify_delivery.py
```

Python 3.11、MetaWorld 3.1.1（commit `6e01ad7e2ffb2302e4dca04f796fcd8837df8540`）、MuJoCo 3.3.0、PyTorch 2.7.1、diffusers 0.35.1。全部传递依赖见 `pixi.lock`，运行时设备与版本见 `artifacts/environment.json`。本机默认使用物理 GPU 1、EGL，无自动购买算力或修改驱动行为。

## 四个固定实验

| run_id | 任务 | 表示 | 示范 | seed | updates |
|---|---|---|---:|---:|---:|
| drawer_raw_n20_s0 | drawer-open-v3 | raw | 20 | 0 | 20000 |
| drawer_relative_n20_s0 | drawer-open-v3 | relative | 20 | 0 | 20000 |
| door_raw_n20_s0 | door-open-v3 | raw | 20 | 0 | 20000 |
| door_relative_n20_s0 | door-open-v3 | relative | 20 | 0 | 20000 |

## 数据与模型语义

原生 observation 保留 39 维：当前 18 维、原生历史 18 维、当前目标 3 维。每块的手位置和交互点进行 `(hand, handle) -> (hand-handle, handle)`；当前目标替换为 `goal-current_handle`。其余字段不变。完整字段及任务特定四元数顺序见 `artifacts/observation_schema.json`。原生 reset 用当前状态填充历史；DP 在此基础上再使用相邻两步 observation。

在时刻 t，输入 `(o[t-1], o[t])`，预测 `a[t:t+16]`，执行前四个动作。起点重复 observation，尾部缺失动作以 mask 排除。动作标签是裁剪到原生边界后实际执行的四维输入，不再缩放或做示范 min/max normalization。每种表示的 mean/std 只使用固定 train20 的合法输入状态，std 下限 0.001；OOD 输入不裁剪。

每任务 100 条成功专家数据组成池，预先固定 RNG 选 20 条优化；其余 80 条不进入优化器或统计。dev、IID、OOD 各 20 个独立初始条件，OOD 左右各 10 个。采样通过原生 Task 设置和确定性 reset 适配层完成；真实位置、snapshot、恢复验证、失败专家尝试及文件 hash 均记录。

条件一维 U-Net 源于官方 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)，许可证与修改说明随必要模块保存。训练及推理配置见 `src/relative_dp/config.py` 和冻结 artifact。

四个模型各 4,696,644 参数。资源执行设置统一采用 `torch.compile(mode="reduce-overhead")`，训练和推理都为 float32、禁用 TF32、使用确定性算法；有效 batch=128，无 gradient accumulation。设置在正式训练前根据吞吐实测固定，测量仅用于运行效率，不据开发/测试成功率调参。编译与 eager 的性能记录见 `artifacts/runtime_benchmark.json`，完整编译恢复测试随正确性测试执行。

## 选择、评测与产物

每个模型的 EMA 5k/10k/20k checkpoint 均用同一 dev20 评测，先比较成功率，再比较成功 episode 平均步数，最后选更早 checkpoint。四个选择全部冻结后再运行 IID/OOD 共 160 个测试 episode。每个 episode 固定独立 diffusion RNG，同任务表示配对。

- `data/dataset_manifest.json`：split、train20、实际位置、版本与 hash。
- `artifacts/`：硬件、状态 schema、正确性检查、冻结实验配置。
- `runs/<run_id>/`：正式配置、normalizer、训练日志、EMA 与完整恢复状态。
- `results/`：开发/测试逐 episode 数据、checkpoint 选择、summary、曲线与视频。
- `ROUND1_REPORT.md`：真实结果、计算成本、局限与下一步。
- `PROGRESS.md`：当前完成状态和精确恢复信息。

大文件已加入 `.gitignore`；源码、配置、lockfile 和小型结果可纳入版本控制。当前目录的原始文件将保留，不自动推送远端。


## Round 2 cabinet yaw experiment

Custom physically rotated MetaWorld tasks with paired world/frame diffusion policies. See [protocol](ROUND2_PROTOCOL.md), [geometry audit](docs/ROUND2_GEOMETRY.md), and the [completed report](ROUND2_REPORT.md).

Run the complete validated pipeline with `pixi run round2`; individual tasks are `round2-preflight`, `round2-calibrate`, `round2-collect`, `round2-train`, `round2-evaluate`, and `round2-report`. Round 2 artifacts are isolated in directories named `round2`. Four formal runs each use 20 demonstrations and 20,000 updates, followed by 240 dev and 480 final test episodes.

Round 1 frozen source, data, model and result files are preserved. Its original Pixi task file is archived at `artifacts/round2/round1_archive/pixi.toml`, because adding the new tasks necessarily changes the historical whole-workspace fingerprint. All other Round 1 frozen input hashes still match.
