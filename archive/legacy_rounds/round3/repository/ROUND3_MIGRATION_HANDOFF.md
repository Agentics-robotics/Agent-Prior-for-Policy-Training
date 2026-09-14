# Round 3：迁移交接（用户要求暂停）

记录日期：2026-09-08。本机项目目录 `/home/users/oscar/Agent_Training/agent_training`。

**用户主动要求停止并迁往其他服务器。当前没有继续执行实验的授权；迁移后的 Codex 收到继续指令后再恢复。正式训练 0 次，optimizer updates 0，模型开发评测 0 回合，锁定测试评测 0 回合。不得把准备完成写成实验完成。**

## 1. 已停止的位置

数据准备子进程 Python PID 3610230 收到 SIGINT 后退出 130（其 Pixi PID 3610073；工具 session 60091）。只停止了本项目所属进程，没有停止其他人的 GPU 作业。后续进程检查未发现运行中的 Round 3 Python/Pixi 进程。设计和评测实现子代理也已停止。

| 任务 | 已保存专家校准 | 成功训练示范 | 完整示范重放 | 已冻结 dev 状态 | 已封存 test 状态 | 初始设计证据 |
|---|---:|---:|---:|---:|---:|---|
| pick-place-wall | 60/60 | 20 | 20 条通过 | 50 | 100 | D2 数值轨迹、摘要、16 张真实帧 |
| assembly | 60/60 | 20 | 20 条通过 | 50 | 100 | 同上 |
| drawer | 60/60 | 20 | 20 条通过 | 50 | 100 | 同上 |
| door | 37/60（IID20、C17） | 0 | — | 0 | 0 | 尚无 |
| peg-insert-side | 尚未开始正式准备 | 0 | — | 0 | 0 | 尚无 |
| stick-push | 尚未开始正式准备 | 0 | — | 0 | 0 | 尚无 |

上述完整任务的数据 manifest 均为 `complete=true`，60 条成功示范采集均无失败尝试。累计 217 条已保存校准记录（完整任务各 IID/C/E20；另有 door37），以及独立基础设施审计的 48 次 reset/零动作检查和 18 次官方专家探测。不要把这些校准、专家探测或已生成评测状态误称为模型评测。

| 任务 | D2 transitions | D5 | D10 | D20 |
|---|---:|---:|---:|---:|
| pick-place-wall | 163 | 413 | 810 | 1590 |
| assembly | 179 | 446 | 885 | 1781 |
| drawer | 178 | 443 | 889 | 1774 |

D2/D5/D10/D20 按预先声明 LL、HH 交替顺序嵌套；训练 LL/HH，组合留出 LH/HL，外推 E 使用训练区间之外的值。具体范围、额外 nuisance 参数和来源见 `round3/protocol_and_task_manifests/`，不应重新挑选成功轨迹或测试状态。

## 2. 实施规格和真实设计状态

- 完整自包含规格已复制到 `round3/ROUND3_SPEC.txt`。
- 96 次主训练：6任务 × B0/P1/P2/P3 × N2/5/10/20 × seed0；每系统20k updates、batch128。
- 每任务最多一次 N5 开发反馈修订 P4，仅追加 N5/N20，从头训练；正式训练总上限108次。
- **优先完成 pick-place-wall、assembly 的 N5/N20 主矩阵16次**，不要等六任务所有候选模块完工才启动首批。
- 先测两个完整新任务训练及开发评测的实际耗时，再估计剩余时间。当前尚无本轮实测训练成本。
- 主指标固定20k EMA；所有设计、反馈决策和按dev选出的模型冻结后，才允许产生锁定测试结果。
- 用户本机最终指定 GPU **0、2**，同时最多两张，每张最多一个正式训练进程。新机器要重新核对设备；历史 `AGENTS.md` 的 GPU1 默认不适用于本轮用户覆盖。

一个真实隔离设计子代理（`fork_turns="none"`）已读取 pick-place-wall 授权 D2 证据并看完16张图片，但用户停止时尚未完成设计。仅保存：

`round3/design_records/pick-place-wall/proposal.draft.json`

该文件明确标记 **DRAFT_NOT_FROZEN_NOT_FINAL**，不是有效正式提案。没有 `proposal.json`，没有 `initial_freeze.json`，没有 P1/P2/P3 实现，更没有候选成绩。恢复时完成这一首次设计，不可把草稿说成已冻结候选。其余任务还没有启动设计。隔离设计约束见 `round3/DESIGN_CONTRACT.md`。

主协调 Codex 曾读过 Round1/2 的结果、D20摘要及专家/环境源码。基础设施子代理也读过新任务源代码；设计子代理仅读授权 bundle。总体应称 **Codex 辅助设计的探索实验**。详见 `round3/context_exposure.md`。无 API 调用，无 API key 依赖，精确模型版本/session/token/费用不可得，不能编造。

## 3. 已写代码与验证边界

全部新代码位于 `src/round3/`，旧 `src/relative_dp/` 和 `src/round2/` 保留。

- `tasks.py`：六任务布局参数化、真实原生 ID、LL/HH/C/E 和独立 seed。
- `environment.py`：新任务 canonical native Task+seed reset、共同观测、官方专家、snapshot；drawer/door 直接复用 Round2 物理旋转、后挡条修复和严格 joint-validity 成功规则。
- `prepare.py`：校准、示范采集、失败尝试记录、每条完整重放、D2证据和 dev/封存test 状态；逐条原子保存，可断点继续。
- `common.py`：96配置 manifest、原子更新、run/GPU锁、至多108次的P4登记、事件日志。
- `plugins.py`：小型候选接口及 B0 默认实现；未预填假候选。
- `learning.py`：按当前 D_N 拟合统计、历史2/预测16/尾mask、复用Round2不裁剪DDIM的DP、系统总预算、checkpoint/optimizer/EMA/RNG恢复、参数预算和源码归档框架；**尚未实际进行任何 optimizer update，完整训练/恢复未验证**。
- `proposals.py`：提案必填字段、证据hash、初始冻结和B0配置保存。
- `evaluate.py`：固定20kEMA、配对闭环、逐步终止和joint-validity、Wilson区间、逐回合缓存、dev选择和全局测试门禁。子代理完成了CPU合成环境语义检查（chunk内成功/终止、500步、历史越界、裁剪、reset完整性、Wilson边界）；**尚未在实际模型上运行**。
- `cli.py`：prepare/validate-proposals/train/dev-eval/test/report/resume/all 入口及0/2双GPU子进程调度初稿。**总流程未验收**。`report.py` 尚未交付，反馈图片生成/最终图表视频报告尚未完成，不能称所有入口已经可用。

已验证真实数据能进入B0窗口管线：pickwall N5 observations `(413,2,45)`、actions `(413,16,4)`；来源正好前5条示范。未执行训练。

共同 raw observation 维度：pickwall45（原生39+墙中心和半尺寸），assembly42（+螺母环中心），peg42（+peg head），stick39，drawer/door41。所有额外字段 B0 和候选共同可用。assembly 原生 nut x 退化为0，已明确记录 native reset-vector 范围扩展，并独立校准，未改动力学。assembly 原生四元数 WXYZ，其余当前声明 XYZW。原生 action 是世界 xyz 增量+gripper，不能假装支持自由旋转/力控制。

源码语义审计：`docs/round3/native_source_audit.md`，探测结果 `artifacts/round3/source_audit_expert_probes.json`。这些审计不是 Agent 的默认设计证据包，禁止把专家动作规则递给隔离设计者。

## 4. 下一位执行者必须先核对的未完成集成

这不是已跑通的 Round3成品。先阅读代码再补齐，不要直接宣称 `pixi run round3` 可无人接续完整设计。

1. `learning.data_for_run()` 和 CLI `_eligible()` 目前仅依赖已有 manifest/记录，**还应显式要求数据 `complete=true`**，避免采集未冻结时启动训练。
2. 单 run CLI 及 report 的 GPU 环境仍可能继承 Pixi activation 的 `CUDA_VISIBLE_DEVICES=1`。总调度子进程已显式指定0/2，但所有独立入口需统一遵守新机器设备约束。必须在 `pixi run` **内部**覆盖环境；外层设置会被activation覆盖。EGL设备也单独设置。
3. `cli.py` 硬编码本机 Pixi路径 `/home/users/oscar/.pixi/bin/pixi`。新服务器恢复前改为已安装的 Pixi路径或可靠发现方式；冻结训练尚未开始，因此现在修正不会影响正式run身份。
4. 总调度中断后的 stale running 状态恢复/进程存活核对尚待完整测试；当前96项均未运行，没有遗留running条目。
5. 已写 `report` 分支引用尚未交付的 `round3.report.generate`。反馈bundle渲染及全部报告图表/真实配对视频仍待实现。
6. 评测与common/proposal/修订schema的真实端到端集成、baseline与R2行为对齐、完整RNG恢复、候选坐标/监督的数学和信息公平性验证尚待做。
7. 训练源码hash是严格身份的一部分；在第一轮正式训练前完成共享核心必要修正，后续候选以单独插件模块加入，避免无记录改变已训练逻辑。
8. 提案只能由真实Codex写，不能以固定占位配置填充。脚本遇到缺提案写 pending-design，由正在运行的Codex完成后继续；脱离会话不能自主产生新设计。

开发反馈图片规则已由评测子代理预先提出：在C集合按manifest顺序取首个N5四模型有成功分歧的状态，无则首个C；四模型各取物理步50和200，已结束则保持末帧，共最多8帧。恢复实现时检查是否已有冻结文件，否则在首次模型反馈前明确保存规则；不得从锁定测试取反馈。

## 5. 迁移范围与环境

迁移整个项目目录，保留 `src/`、`round3/`、`pixi.toml`、`pixi.lock`、`pyproject.toml`、旧轮次源码/配置/数据/报告/模型以及本交接文件。建议排除本机 `.pixi/` 和 `__pycache__/`，在新机器按 lockfile 重建环境，避免绝对环境前缀失效。当前整个项目约9.0GB，其中 `.pixi`约5.8GB，Round3数据和证据约14MB；不带环境的完整项目约3.2GB。

没有需要迁移的 Round3 checkpoint。已有 Round1/2 `runs/` 约2.3GB 属于历史产物，保留则历史审计和复核更完整。不要删除旧数据来腾空间。

锁定环境：Python3.11、MetaWorld3.1.1 commit `6e01ad7e2ffb2302e4dca04f796fcd8837df8540`、MuJoCo3.3.0、Torch2.7.1、diffusers0.35.1。所有项目命令通过Pixi；无需API key，不新增云服务/购买计算资源。

Pixi新增入口已写入 `pixi.toml`：`round3-prepare`、`round3-validate-proposals`、`round3-train`、`round3-dev-eval`、`round3-test`、`round3-report`、`round3-resume`、`round3`。当前它们只是阶段代码入口，受上一节未完成项限制。旧Pixi配置备份在 `round3/audits/pre_round3_pixi.toml`，lockfile未改变。

迁移后可先做环境安装，并按新GPU编号恢复未完成数据准备（下面只是恢复指令，本机未执行）：

```bash
pixi install --locked
pixi run env CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  python -m round3.prepare --task door --task peg-insert-side --task stick-push
```

door已保存37条校准会复用；其余完整任务不应重采。迁移后对已冻结数据做hash和少量replay验证，平台浮点差异不能通过静默改标签/状态解决。evidence内历史绝对源文件路径作为溯源保留，不要无理由改写冻结证据的hash。

给新服务器 Codex 的接续指令：

> 请先读 ROUND3_MIGRATION_HANDOFF.md、round3/ROUND3_SPEC.txt 和 round3/context_exposure.md。用户现在授权继续 Round3。保留既有数据和未冻结设计草稿，修完交接中列明的集成问题，完成实际Codex设计后优先运行pick-place-wall与assembly的N5/N20共16次训练，再完成其余矩阵、一次反馈及冻结测试报告。正式主矩阵96次、总上限108次，每系统20k updates；同时最多用两张本机可用GPU。所有正式结果必须来自真实训练与闭环评测。

## 6. 旧轮次状态

Round1、Round2此前已完整完成，报告分别为 `ROUND1_REPORT.md`、`ROUND2_REPORT.md`，未被本轮覆盖。Round2实际四次20k训练、240次dev和480次测试；yaw主指标 drawer world/frame 0%/81.7%，door45%/80%，均单seed。它们是历史证据，不能冒充Round3的96次训练结果，也不应交给严格隔离的初始设计子代理。
