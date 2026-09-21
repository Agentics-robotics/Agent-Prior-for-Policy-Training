# Experiment 1 — Repository Agent Execution Handoff

版本：2026-09-12。交给能够访问实际训练仓库和 GPU 服务器的 Codex/开发 agent 直接执行。

### 最新执行修订：工具调用设计 agent（优先于下文一次性源码返回示例）

- 仅使用物理 GPU 4、5、6、7；运行 worker 按 UUID 核验实际设备。0–3 属于其他用户，不干预其进程。历史记录保留实际原始卡号。
- RuntimePriorAPI 通过持久化工具循环读取冻结公共接口、允许的公共源码和当前 task×N 证据，创建、读取、修改自己的未提交候选代码，调用固定零更新接口检查并依据诊断修复，最终显式 `submit_candidate` 提交源码、配置、设计说明和 SHA256 manifest。不能要求一次性返回完整源码代替该循环。
- 公共训练器、评估器、划分、成功判定均只读；工具无 shell 或任意执行入口，不允许访问隐藏测试、其他设计会话或 API 密钥。生成代码仅在验证过的 Landlock/seccomp 隔离 worker 内执行。
- A1/A2/A3 全部提交后，外层 runner 才调度训练及开发评估并返回性能反馈；随后仅可设计、提交 A4。已提交代码不可修改，禁止将性能搜索隐藏为 debug。
- 工具调用、检查及修复预算与正式训练预算分别记录。同一未提交候选的多次编辑不增加模型槽位；保留全部调用、代码版本、检查成本、训练 attempts 和最终提交。
- 恢复时重用已落盘响应、工具结果及提交；保留中断调用费用的不确定性，不做自动传输重试。候选归纳偏置与源码必须由 RuntimePriorAPI 实际产生，开发 Codex 只实现工具、公共框架及固定 B0/B1。

这是一份实现与运行指令，不是已完成实验的报告，也不声称下述命令已经存在于仓库。接收本文件的 agent 必须先适配真实代码、实现入口并验证，再运行。不要仅生成计划、空配置或报告模板后结束。

## 0. 用户已经决定的事项

- 本轮名称固定为 **Experiment 1**，文件目录使用 `experiment1`。不是 Round 4、Round 5 或 Experiment 2。
- 删除未完成 Round 4 的专属代码、配置、结果和实验入口；归档 Round 1、2、3。保留可复用公共能力，并整理到正式代码中。
- 只做六个 MetaWorld 单技能任务，数据量为每任务 **N = 2, 5, 10, 20 条完整示范**。
- 每个 task × N 训练六个系统：`B0_vanilla_dp`、`B1_rule_prior`、`A1`、`A2`、`A3`、`A4`。
- `A1/A2/A3` 来自设计 API 在性能反馈前提出的三个可执行 prior 设计；训练并评估三者后，由同一个设计 API 根据反馈提出 `A4`，再训练、评估。
- 预算节点使用 **q = 1, 3, 4**。这些节点复用四个候选的结果，不另外训练 1+3+4 个模型。
- 本轮默认 **一个完整重复，训练 seed = 0**。总计划为 **6 × 4 × 6 = 144 个正式系统训练槽位**。不要擅自乘三个 seed、添加搜索对照或扩展到柜体、长任务、真机。
- 必须区分开发 agent 与实际设计 API；每个候选必须有真实 API 输出的可执行实现和来源记录。
- 完成清理、实现、实际训练、独立测试、统计、图表及完整报告。真实阻塞应记录并明确告知，不能伪造完成。

最新用户决定优先于旧 Round 指令与论文中的可选 `no_revision`、q=1/2/4、跨三次完整重复等规划。适用的系统/安全/仓库权限要求仍需遵守，不得绕过访问控制。

## 1. 先正确理解两个角色

| 角色 | 本轮职责 | 不能混淆成什么 |
|---|---|---|
| **RepositoryAgent / 开发 agent** | 阅读、清理仓库；实现任务适配、数据、统一 DP、固定规则 prior baseline、API 客户端、插件接口、检查器、训练器、评估器、调度、报告；执行实验 | 不能自己设计 A1–A4，再声称这些候选来自 API |
| **RuntimePriorAPI / 设计 API** | 根据当前 task×N 的证据，实际选择/构造归纳偏置，返回可执行架构和训练设计；解释开发失败；生成 A4；依据预定开发规则选择候选 | 不是仅润色理由、给开发 agent 预制方案打标签，或报告结果的 API |
| **Trainer / Evaluator** | 在公共框架中实例化 API 返回代码，训练与执行；按固定规则记录开发与测试指标 | 不是第三个自由决定 prior 的设计者 |

这里的 API 是实验时调用的模型服务；不是机器人部署时的在线策略。部署策略仍是冻结的 learned policy 与确定性代码，不进行 LLM/API 调用。

开发 agent 编写的公共 prior 实现接口和 B1 属于本项目代码。实际方法的 A1–A4 设计行为必须由指定模型 API 产生。即使开发 agent 和 API 使用相同模型家族，也要区分调用、上下文、代码作者和信息权限。

历史 Round 3 使用 Codex 会话/子 agent 作为设计后端，不能将其记录重新命名为本轮真实 API 调用。历史结果只作实现与开发参考，不算 Experiment 1 数据。

## 2. 矩阵、q 节点与预算：不得改变含义

### 2.1 六个任务

使用项目任务别名；由实际安装版本解析准确环境 ID，不能猜测 v2/v3 等版本。

| task alias | 因素 A | 因素 B |
|---|---|---|
| `pick-place-wall` | initial object x | goal x |
| `assembly` | initial nut x | post/goal x |
| `drawer` | cabinet x | cabinet yaw |
| `door` | cabinet/base x | cabinet yaw |
| `peg-insert-side` | peg y | target box y |
| `stick-push` | stick x | target y |

不得为了完成计数把不工作的任务静默换成 reach 等任务。适配失败先修复；无法修复则保留原矩阵并报告阻塞。

### 2.2 六个实际训练系统

| system_id | 设计来源 | 当前 task×N 的训练次数 | 用途 |
|---|---|---:|---|
| `B0_vanilla_dp` | 开发 agent 实现统一普通 DP | 1 | 无专门结构设计的基础对照 |
| `B1_rule_prior` | 开发 agent 预先实现并冻结的通用规则 prior | 1 | 简单固定结构对照 |
| `A1` | API 初始返回的第一顺位设计 | 1 | q=1，也参与 q=3/q=4 |
| `A2` | API 初始返回的第二个设计 | 1 | 参与 q=3/q=4 |
| `A3` | API 初始返回的第三个设计 | 1 | 参与 q=3/q=4 |
| `A4` | API 查看 A1–A3 开发反馈后提出的修订设计 | 1 | 参与 q=4 |

24 个设计实例：6 tasks × 4 data tiers。

144 个系统槽位：24 B0 + 24 B1 + 72 initial API candidates + 24 revised API candidates。

**q=4 必须包括新的第四个设计及训练。仅让 API 从前三个里挑一个，不构成第四个模型。**

### 2.3 q=1/3/4 的精确定义

- `API_q1`：初始 A1 的系统；A1 的顺位在任何性能反馈前提交。
- `API_q3`：API 根据固定开发评分规则，从 A1/A2/A3 中选择的系统。
- `API_q4`：API 根据同一规则，从 A1/A2/A3/A4 中选择的系统。
- A4 不必胜过前三个。q4 可以继续选 A1/A2/A3；不得为了产生“修订收益”强行选择 A4。
- q1/q3/q4 不重新训练，也不按测试得分重排候选。
- 为保持用户的 144 次训练，q1 是同一批初始三方案中的预先第一顺位候选，属于预算前缀分析；不是额外运行一次“只允许提出一个方案”的独立 prompt 实验。报告必须说清楚这一点。
- 主表的 API_q1/q3/q4 只在 API 候选中选择，B0/B1 单独列出，以免 baseline fallback 掩盖 API 候选失败。这是本轮相对旧稿统一 fallback 展示方式的明确调整。
- 若希望保留部署时 B0 fallback，同时报告独立标记的 `API_q*_with_B0_fallback`；它不能替代上述主结果，不新增训练。

### 2.4 API 选择与确定性验证

开发评分为 `0.5 * dev_C_success + 0.5 * dev_E_success`，在每个分布内部对预定子条件等权。IID 单独报告。

同分时依次选择：固定测量方法下推理延迟更低、全系统参数更少、候选 ID 更小的方案。API 必须返回符合这条规则的选择及理由，runner 重新计算并核验。API 不能因为“更有潜力”而选择分数更低的方案。

若 API 返回错误选择，最多一次只包含选择规则与原开发数字的纠正请求；仍错误则由固定 selector 输出、记录 `api_selection_noncompliant`，不能声称该次正确选择来自 API。不得重训候选或改变规则。

q4 的开发最优值理论上不会低于 q3，但独立测试效果可能更差。保留负结果。

### 2.5 预算不等于所有费用

- 144 是正式系统槽位，不包括 API 调用数、开发/测试回合数、预检与失败修复费用。
- 默认所有模块在一次联合训练 job 中优化，预测头/编码器参数计入候选。不要额外先训练 task-specific labeler/selector；若某设计无法在预算内实现，应给 API 明确接口诊断。
- 正式重复为一次，seed 0。完整重跑三次会变成 432 个系统槽位，不在本轮默认范围内。
- 科学性检查或短训练检查要记账，但不能伪装成额外免费搜索；不把它们写成正式完成的系统。
- 相同进度断点的精确恢复不是新候选；从头重启产生的实际费用单独累计。
- 计划 144 个不保证能得到 144 个有效完成系统。有效性失败应保留为失败槽位，不能用伪造或复制 checkpoint 补足。

## 3. 仓库清理：已获用户授权，执行精确目标

先读当前根目录与适用子目录的 `AGENTS.md`、README、Pixi 配置和真实代码。用 `git status` 与 `rg` 检查，不根据本文件推测真实路径。

### 3.1 清理前清点

生成 `experiments/experiment1/migration_inventory.json`，列出：

- 真实 repo root、当前 revision、已有未提交修改的文件名，不打印秘密文件内容。
- Round 1/2/3/4 的实际目录、散落报告、配置、日志、模型、入口、任务依赖。
- 哪些是共享源码、共享示范/资产、环境锁文件、实验专属文件。
- 每个待归档、迁移、删除目标的精确路径、类型、大小与可用 hash。
- 是否存在正在使用 Round 4 文件的本项目进程。只停止已确认属于本项目 Round 4 的 job；不杀其他用户进程。

此清点是自动执行记录，不是要求用户再次逐项确认正常清理。只有确实无法判断归属、无法访问或会伤及无关工作时，先完成其余工作并提出具体阻塞。

### 3.2 Archive Round 1/2/3

- 归档根目录为 `archive/legacy_rounds/`，保持 `round1/`、`round2/`、`round3/` 可辨认。
- 用移动避免大模型重复占磁盘；旧目录中的相对结构尽量保留。
- 保存旧配置、报告、日志、proposals、评估明细和存在的 checkpoints，不把旧测量改写成新结果。
- 若旧实验只剩报告而没有 checkpoint，在 archive index 中注明 missing，不能声称完整可复现。
- 共享数据可以保持在原公共位置，用 manifest 记录引用与 hash；不要复制大批数据，也不要因为归档断开 Experiment 1 所需引用。
- 建立 `archive/legacy_rounds/README.md` 和机器可读路径映射。现有旧归档不要覆盖，按来源路径消除冲突。
- 历史受控结果、候选代码和文本理由不得进入本轮 RuntimePriorAPI 的证据目录。开发 agent 的历史接触在报告中披露。

### 3.3 Delete unfinished Round 4

- 先将本轮实际需要的通用代码迁移到正常 `src/` 模块，验证入口使用新路径。
- 删除已确认仅属未完成 Round 4 的目录、输出、配置、测试草稿、专属入口和过时文档入口。
- 这不是把 Round 4 改名继续运行；本轮不是 ManiSkill 柜体实验。
- 不删除共享示范、资产、通用控制器、API 客户端、Pixi 本身或其他项目资源。
- 不执行 `rm -rf *round*`、全仓 `git clean -fdx`、`git reset --hard` 等广泛操作。删除前解析精确路径、符号链接和所属 repo 范围；对外部共享挂载默认不删除。
- 不改写 git 历史，也不为“删除”而 force push。保留删除日志即可，不另造一个未被要求的 Round 4 结果归档。
- 在 `migration_log.jsonl` 记录实际归档/迁移/删除的结果；复跑清理命令应幂等，不能重新删除新工作。

### 3.4 整理正式代码

优先复用并修正真实的 Round 3 DP、窗口数据集、动作 frame、stage/future 通道代码；不要平行复制一个几乎相同的训练框架。

目标是一个统一训练入口、一套明确接口、一个实际运行的 Experiment 1 目录。目录示例可以按现有包名调整，但必须记录映射：

```text
src/<existing_package>/
  data/
  environments/
  policies/
  priors/
  experiment1/
    api_client.py
    evidence.py
    contracts.py
    plugin_validation.py
    runner.py
    selection.py
    reporting.py
experiments/experiment1/
  protocol.lock.json
  matrix.jsonl
  manifests/
  prompts/
  evidence/
  api_records/
  generated/
  runs/
  selections/
  reports/
  figures/
  state.sqlite
archive/legacy_rounds/
```

保持项目现有 git 同步；不要将大型 checkpoint、秘密或大量临时产物加进 git。正文代码/配置/报告应可追踪；不要未经请求发布仓库或推送大文件。

## 4. 固定基础设施与两种 baseline

### 4.1 使用 Pixi

- 使用当前仓库 Pixi；检查实际版本和可用 task。已有锁文件正常时用 `pixi install --locked`。
- 如果仓库没有有效 Pixi manifest，不反复运行必然失败的命令；根据现有依赖建立并验证 `pixi.toml` 或 `[tool.pixi]`，解析锁文件，之后冻结。
- 不创建 Conda 环境，不在 base 环境安装项目依赖。所有正式命令通过 `pixi run`。
- 保持 GPU/CUDA、MuJoCo 与当前项目兼容，记录精确依赖版本。不是为了“最新”升级已经有效的栈。

### 4.2 共同训练配方

下列为从现有方案继承的起始配置；先检查代码和开发期拟合，之后共同冻结，不按正式候选得分个别加预算：

| 项目 | 默认 |
|---|---|
| policy input | declared numerical state + common permitted metadata |
| learner | conditional 1D U-Net diffusion policy |
| observation history | 2 |
| prediction horizon | 16 actions |
| executed prefix | first 4 actions |
| optimizer updates | 20,000 |
| batch size | 128 |
| optimizer | AdamW |
| learning rate / weight decay | 1e-4 / 1e-6 |
| gradient clipping | norm 1 |
| EMA | decay 0.995 |
| inference | 16 DDIM steps; exact scheduler pinned |
| checkpoint for reporting | final fixed-update EMA |
| training seed | 0 |
| complete-system capacity limit | 3 × current B0 parameter count |

记录完整 denoiser widths、diffusion train steps、noise schedule、prediction target、padding/masks、normalization floors、action scaling、rollout horizon/frequency、采样种子。不能只写“使用 DP 默认值”。

短拟合预检应确认 loss 与动作语义正常、能够拟合支持示范。不能为让 prior 赢而故意欠训练 B0。若共同配方必须调整，在正式矩阵开始前冻结；启动后的共享实现严重错误需明确失效哪些 run，而不是悄悄混合版本。

代码允许 API 改动 representation、action interface、encoder/共享方式、stage conditioning、auxiliary targets/losses及兼容的有限架构细节。优化预算、基础 denoiser、示范和控制语义共同固定。API 不能靠多数据、多训练轮数或更换超大主干获得额外预算。

### 4.3 B0：Vanilla DP

- 使用同一允许的原始状态、目标、机器人/几何事实及 common observation schema；普通拼接 conditioning。
- 原生动作表示；无额外的 task-relative encoding、stage auxiliary 或 action–future auxiliary。
- 不给它隐藏更差的传感器或错误动作语义。若 API 使用某个真实 raw field，B0/B1 也必须能访问该 raw field。
- 训练与推理配方同上，独立新训，不复用旧 Round checkpoint。

### 4.4 B1：开发 agent 编写的固定规则 prior DP

这是本轮唯一的规则/naive prior baseline，不是另加一个 scripted expert robot controller。

默认实现一个足够明确且跨任务共享的配方：保留 B0 所有 raw/world/robot 字段，附加角色之间的世界轴相对位置向量，例如 hand→interaction、manipulated part→goal、tool→target；使用共同的小型关系特征编码并与 raw conditioning 融合，动作仍保持 native。

- 开发 agent 根据公开任务语义冻结角色字段表、可用关系、编码结构和缺失字段处理。每个 task 的角色映射可不同，但选择规则本身不随 N 或开发分数改变。
- 只使用 common schema 中已经授权的字段。若某关系不存在，按预定 mask/规则禁用；不凭 asset ID 猜几何。
- 默认不叠加 stage、future head、joint diffusion、动态 frame 动作等全部结构，以免把“naive prior”写成隐含调优组合。
- 若已有仓库中更成熟、明确且符合此定位的固定关系 baseline，可以复用其经过审查的实现；必须在正式 API 设计和评估前冻结并记录理由。
- 不让 API 设计 B1；不把看过 API 胜出候选之后抄写的结构标成预先固定规则 baseline。
- 不为每个 task×N 根据结果手动选择最佳规则。按同一规则配方独立训练 24 个 B1 权重。
- 报告命名为 `Fixed rule-prior DP`，不夸称最强人工方法或 SOTA。

## 5. 数据与物理环境：先通过实际验证

### 5.1 场景与控制

复用历史实现前检查真实安装版本下：reset、零动作、原生控制缩放、夹爪通道、goal/success 语义、专家演示可回放、终止与 timeout。

drawer/door yaw 要实际旋转几何、关节轴、把手、目标等相关量；不能仅修改 observation。确认新初态在当前控制自由度下可执行。专家失败不是物理不可达的充分证据。

为六个任务冻结实际 continuous low/high/extrapolation intervals、单位、采样方法、reset 实现和环境 hash。旧表只规定因素名称，不提供最终区间；必须读取现有实现并验证。若范围需要修订，写入开发期决策，不依据正式测试得分调难度。

### 5.2 支持示范

- 每任务建立一个有来源记录的 20-episode support pool，嵌套得到 D2⊂D5⊂D10⊂D20。
- 保持 LL/HH 支持模式，尽可能平衡；N=5 等奇数按固定顺序交替，记录实际覆盖。
- N 始终计完整原始 episode；切成训练窗口或阶段片段不增加 N。
- 六个系统在同一个 task×N 使用完全相同的原始示范、拆分、控制与允许信息。
- 可复用合格的历史 support 原始示范，明确来源；不得复用旧 checkpoint 或把旧 test 当新 dev/support。
- 新 dev/test 使用独立且未被实验设计使用过的回合 manifest。做不到“历史上全新”时，诚实披露历史曝光，不宣称严格全新 benchmark。
- 每个候选的 normalization、标签、统计、代码生成证据只能由当前 D_N 得到。N=2 不能用 D20 拟合 normalizer、stage labeler 或 target scale。
- 每任务历史已参与开发，不主张 unseen-task transfer。主张限于所测的布局和数据条件。

### 5.3 三种评估分布

- IID：LL/HH 支持区间内独立 episode。
- C：LH/HL 交叉因素。它是空间组合，也可能改变相对位移与路径长度；不是长任务技能组合。
- E：超出训练边际支持但在冻结可执行范围内。
- 保存因素、相对距离/方向和可用路径长度描述；把 C/E 的实际改变记录清楚。

### 5.4 开发与测试回合默认预算

- 每个系统 dev：10 IID + 20 C + 20 E = 50 回合；子条件按预定分配等权。
- 每个系统 hidden test：20 IID + 40 C + 40 E = 100 回合。
- 同一 task 的六种系统与不同 N 使用配对初态/rollout randomness，明确随机数来源。不能从大量种子中事后挑选。
- 六个系统全部开发评估、全部独立测试，用来报告候选负结果与选择行为；测试需在全局冻结之后。
- 计划总数：144 × 50 = 7,200 dev episodes，144 × 100 = 14,400 hidden-test episodes。预检/失败重试额外记账。
- q1/q3/q4 引用已有测试结果，不重复跑相同 checkpoint 的评估来增加样本。
- 在正式运行前可以基于纯开发期成本校准统一回合预算；一旦修改，更新 protocol.lock 的所有计数并说明。不能只缩减较差候选的评估。

## 6. RuntimePriorAPI：实际连接、可执行输出与权限

### 6.1 API 配置与真实性

检查仓库已有客户端，复用可工作的真实服务。provider、base URL、model ID、API 类型、超时和输出上限由本地环境配置提供；不要编造 endpoint/model，也不要假定所有兼容接口支持同一种结构化响应协议。

建议在配置层支持这些项目专用键（名称可适配现有变量）：

```text
EXPERIMENT1_API_PROVIDER
EXPERIMENT1_API_BASE_URL
EXPERIMENT1_API_MODEL
EXPERIMENT1_API_KEY_ENV_NAME
EXPERIMENT1_API_MODE
EXPERIMENT1_API_TIMEOUT_SECONDS
EXPERIMENT1_API_MAX_OUTPUT_TOKENS
```

API key 通过环境/现有安全配置使用；日志不得打印 key、Authorization header、登录 token、带密钥 URL。不要让用户把秘密贴进报告。生成策略代码的进程不能继承 API 凭据。

- 先执行一次真实 schema/小插件 smoke call，检查连通性、实际模型返回、内容提取和代码导入。该调用是基础设施成本，不提供额外候选设计搜索。
- 如果没有凭据或网络不可用，继续完成清理、接口、baseline、数据、验证、可恢复队列和阻塞报告。API 分支标为 blocked，不用本地 agent、mock、旧代码或静默换模型冒充 API。
- Mock 仅用于明确标记的测试，正式 runner 必须拒绝 mock provenance。
- 记录可得的精确模型标识、provider返回的request ID、调用时间、用量；未知字段为 null/unknown，不能从产品名猜。

### 6.2 每个 task×N 独立证据

24 个初始设计会话彼此隔离。固定公共 prompt、接口文档、B0 公共实现语义和字段说明可以共用；不能携带其他 task、其他 N、旧 Round 的候选、得分或对话记忆。

证据包包含：

1. 当前 task、动作目标、部署变化类型 C/E。
2. 当前 D_N episode IDs/hashes、原始状态/动作轨迹或可访问的受控摘要、轨迹长度与覆盖统计。
3. 观察字段名、shape、单位、来源、运行时可用性、角色/控制事实。
4. 序列长度、padding与mask、训练/推理 contract、允许改动、参数上限。
5. 从当前 D_N 按固定规则采样的最多 16 帧真实图像；模型不支持图像时固定为数值证据条件并披露，不能发明视觉观察。
6. 公共接口示例和允许的模块，不包含历史胜出 prior、B1 源码或正式分数。

初始 API 看不到任何候选 dev/test 分数。隐藏测试的具体初态、坐标、视频、结果不能进 API 目录。基础设施验证人员的曝光单独记录。

### 6.3 不是只返回自然语言，也不是开发 agent 代写

API 必须返回 **足以实例化并训练的设计包**。允许复用公共模块，但必须实际提交与证据对应的结构选择、组合、参数和可执行源码；不能只输出一句“用相对坐标”，由开发 agent 再决定全部实现。

推荐输出为经过 JSON/schema 验证的文件映射，避免自由 shell patch：

```json
{
  "schema_version": "experiment1.proposal.v1",
  "instance_id": "<task>__N<N>__rep0",
  "phase": "initial",
  "candidates": [
    {
      "candidate_id": "A1",
      "rank_before_feedback": 1,
      "title": "<specific design>",
      "coverage_gap": "<current-support evidence>",
      "reusable_regularity": "<hypothesis>",
      "dependencies_preserved": ["<robot/world/contact dependency>"],
      "evidence_refs": ["<support id or declared metadata fact>"],
      "expected_failure_signature": "<falsifiable development observation>",
      "required_runtime_fields": ["<allowed field>"],
      "training_only_targets": [],
      "implementation": {
        "entrypoint": "candidate:build_design",
        "files": {"candidate.py": "<actual complete Python source>"},
        "config": {},
        "estimated_parameters": null
      }
    }
  ]
}
```

这个 JSON 是 schema 形状示例，含占位符，不能冒充实际 proposal。initial 必须恰好返回 A1/A2/A3，revision 恰好 A4。公共 contract 在首次调用前实现、文档化、冻结；不要在获得候选后临时改变其语义。

API 可以设计角色关系表示、合适的坐标/action frame、共享编码器、learned soft stage、独立 future head、joint action–future diffusion及预算内的组合。三个候选要有实质结构差异，不只改随机 seed、名字或理由。不要硬编码 A1永远relative/A2永远stage/A3永远future 的模板答案。

本轮仍是单技能、共同 DP 框架。不要额外引入长任务 planner、多个额外预训练模型、奖励优化、测试时语言模型控制或新的数据采集训练。

### 6.4 验证与 repair 的作者边界

开发 agent 负责结构化解析、编译、接口与训练有效性检查；A1–A4 的科研设计修改应回到 API。

- API 输出语法/shape/禁止字段/参数超限错误：返回明确接口诊断，请 API 在同一槽位修复。
- 每个槽位默认最多 2 次纯接口修复调用，首次调用加最多两次修复；所有版本保留。此次数在 protocol.lock 固定。
- 正式 score-driven 改动只允许 A4。初始候选得分差不能反复“debug”到变好。
- 开发 agent 可以修公共 loader 的 bug 或无语义格式问题，但必须留 diff/hash；不得私下更换 frame、标签、loss、架构或超参数。
- API 无法返回有效候选时记录 invalid；不能由开发 agent 手写替代，也不能将 A1 checkpoint 复制为 A2。
- 设计去重在规范化后执行。重复可作为一次无分数的合规修复请求；仍相同标记重复/invalid，不能假装四个不同设计。
- A4 允许有证据支持的简化、组合、改变一个结构或移除有害结构；不得因“不确定会不会涨”默认 no_revision。本轮要求真实尝试第四个设计。若确实无法产生合法不同方案，诚实报告失败槽位。

### 6.5 生成代码的执行边界

生成代码只写入该 candidate 的目录；文件名拒绝绝对路径、`..`、symlink escape。代码不得改 evaluator、support/test manifests、公共成功判定或其他候选。

在可用的受限 worker 中验证和执行，只有当前允许数据与输出路径，无密钥、无网络、无 shell/subprocess、无任意实验目录读取。AST/import检查可作为预检，但不是完整隔离的证明；记录实际可用的隔离方式，别声称不存在的 sandbox 保证。

任何受限执行能力不足先采用只暴露公共模块/可审计插件的更窄接口，不规避本地访问控制。检验 wrapper 是否把全量数据或 test path 透传给插件。

## 7. 推荐插件 contract：实现之后给 API

以下是接口设计要求，接收本文件的开发 agent 需要与现有代码整合，不是要求按不存在的函数直接启动。

```python
class CandidateDesign:
    # All task facts are supplied through audited schemas, never env internals.
    def fit_support(self, support_view, common_spec): ...
    def build_modules(self, common_dp_factory): ...
    def condition(self, causal_history, causal_context): ...
    def encode_actions(self, native_actions, chunk_start_context): ...
    def decode_actions(self, encoded_actions, chunk_start_context): ...
    def training_loss(self, batch, common_diffusion_state): ...
    def deployment_state_dict(self): ...

def build_design(common_spec, config) -> CandidateDesign: ...
```

`...` 表示开发 agent 需要实现的接口；禁止将这些 stub 作为完成产物。可替换为更符合真实框架的 dataclass/torch modules，但语义必须清楚。

必须区分 support-fit、train-only labels 和 causal inference 三条数据路径。

关键验证：

- 因果性：在完全相同的历史下改变未来标签，推理输入和动作不得随未来真值改变。
- 无未来标签/有效性 mask 泄露到推理 conditioning。未来 targets 只用于训练，推理时由模型预测/采样。
- frame 对历史、标签和输出一致；一个 chunk 默认固定使用起点 frame。缩放与物理位移变换次序正确，夹爪通道不当三维向量旋转。
- encode/decode round trip；旋转通道如存在使用正确表示，不能把 quaternion 当平移向量。
- tail padding loss mask 正确；action loss 与 auxiliary loss 分开按有效元素归一化。
- joint action–future diffusion 的 auxiliary block 不能因为增加通道数任意缩放动作 loss。
- stage 推理使用 causal predicted stage，训练不能只喂 oracle stage 然后部署换预测。
- 几何框架退化时有确定性因果 fallback，保留必要 robot/world/obstacle context。
- 参数、训练模块和推理时间记账覆盖整个候选，包括训练期 auxiliary。
- 输出 native action shape、范围、dtype、有限值；共同 clipping/timeout 语义。

这些是与本实验解释直接相关的有效性测试；不要求穷举所有低风险函数或为了测试而重写整个仓库。

## 8. 三次 API 交互与完整状态机

每个 task×N 默认是 3 次主要 API 请求：initial、revision（同时选出 q3）、final selection。final selection 不训练新模型。24 个实例共有 72 次主要请求，另有真实 smoke、传输重试和合规修复调用。

### 8.1 Initial request

发送当前证据与冻结 contract，要求返回有预先顺序的 A1/A2/A3。三个槽位全部完成验证并以 valid/invalid 状态提交代码或失败记录 hash 后，才能向 API 反馈任一候选的表现；存在 invalid 不意味着免费补一个新方案。

API 生成三个方案可以在一个响应中完成；必要时因输出限制分成固定子请求，但所有初始请求必须发生在任何性能反馈前，记录完整调用链，不能借机多试候选。

### 8.2 Train/evaluate initial candidates

用同一 D_N 和 seed 训练 B0、B1、A1、A2、A3，固定配方。训练完成后做预定 dev rollout。

向 API 的反馈包包含 A1–A3 的：原设计与代码 hash、训练完成/无效状态、训练曲线摘要、IID/C/E 开发率和分母、统一失败类别/连续诊断、固定测量的推理时间和容量、固定采样的最多 16 帧/候选。可包含零成功候选，不能挑漂亮案例。

不要把 B1 源码、测试结果或另一个 N 的运行记录塞入这次反馈。baseline 结果在报告中正常比较，不用于隐藏的额外设计引导。

### 8.3 Revision request

要求 API：

1. 按固定规则返回 `best_initial_candidate`，保存为 q3。
2. 说明已观察事实与未确认推断。
3. 返回一个 A4 的完整可执行包，明确 parent IDs、改变哪些设计、依据哪些反馈、预期修复什么、可能有什么副作用。

A4 不能只写“选择 A2”。它必须是第四个训练设计。A4 训练从新初始化开始，使用同一 D_N、seed、更新数；默认不 fine-tune A1–A3，以免引入额外累计训练差异。

### 8.4 Final selection request

完成 A4 的预定 dev evaluation 后，仅发送四个候选的开发指标、成本与已有设计记录。要求 API 按固定规则返回 `best_final_candidate` 及简短理由，runner 核验并保存为 q4。

不允许返回 A5、增加训练、用 test score 重新选模型或更改 A1 的原顺位。

### 8.5 全局冻结与测试

先完成所有 24 个实例的设计/开发/选择，关闭候选设计任务并生成全局冻结 manifest，包含源代码、common protocol、每个候选 checkpoint、normalizer、API records、q selectors 的 hashes。

**然后**才执行整个隐藏测试。这样后运行的 task×N 不会受到早先 test 结果的人工/agent 调优影响。

只有冻结成功的有效系统进入 test；失败槽位保留为 missing/invalid，不伪造 0 成功率。测试成功但零成功的回合才计 measured zero。

### 8.6 控制流参考

下面是必须实现的逻辑伪代码，真实类名由开发 agent 对接已有仓库；不是当前可以直接运行的脚本。

```python
TASKS = (
    "pick-place-wall", "assembly", "drawer", "door",
    "peg-insert-side", "stick-push",
)
TIERS = (2, 5, 10, 20)
SYSTEMS = ("B0_vanilla_dp", "B1_rule_prior", "A1", "A2", "A3", "A4")
REPLICATE = 0
assert len(TASKS) * len(TIERS) * len(SYSTEMS) == 144

cleanup_legacy_rounds_from_exact_inventory()
audit_and_freeze_common_infrastructure()
freeze_rule_baseline()
freeze_data_and_evaluation_manifests()

for task in TASKS:
    for n in TIERS:
        instance = load_or_create_instance(task, n, REPLICATE)
        evidence = build_current_support_only_evidence(instance)
        initial = runtime_api.propose_three(evidence, frozen_contract)
        commit_initial_order_and_code(initial)  # Before any performance feedback.

        # The real implementation persists/reuses completed jobs idempotently.
        for system in ("B0_vanilla_dp", "B1_rule_prior", "A1", "A2", "A3"):
            validate_train_and_dev_or_record_invalid(instance, system)

        feedback = make_dev_feedback(instance, ("A1", "A2", "A3"))
        revision = runtime_api.revise_and_select_initial(evidence, initial, feedback)
        save_q1_pointer(instance, "A1")
        verify_and_save_q3_pointer(instance, revision.best_initial_candidate)
        commit_fourth_code_or_record_invalid(instance, revision.candidate_a4)
        validate_train_and_dev_or_record_invalid(instance, "A4")

        final = runtime_api.select_final(make_dev_feedback(instance, ("A1", "A2", "A3", "A4")))
        verify_and_save_q4_pointer(instance, final.best_final_candidate)
        freeze_instance(instance)

freeze_global_design_manifest()
assert_no_design_jobs_can_receive_test_feedback()
evaluate_all_valid_frozen_systems_on_hidden_tests()
generate_report_tables_figures_and_provenance()
validate_completion_or_report_actual_blockers()
```

真实实现还必须处理所有候选 invalid 时 q selection unavailable、API outage、超时、失败记录和恢复，不能因为伪代码 happy path 简化而遗漏。

## 9. 给 RuntimePriorAPI 的正式 prompt 模板

开发 agent 可为真实 SDK/消息格式做无语义适配；首次正式运行前冻结文本 hash。

### 9.1 Initial prompt

```text
You are RuntimePriorAPI, the actual prior-design backend in Experiment 1.
The repository agent built the common training system; you must make the
candidate-specific scientific design decisions and return executable code.

Use only the supplied evidence for this task and current demonstration tier.
Identify (1) a coverage gap, (2) a reusable regularity, (3) dependencies that
must remain represented, and (4) an executable intervention and a check that
could falsify it. Distinguish observations, supplied facts and hypotheses.

Return exactly three materially distinct valid designs, A1/A2/A3, in the
provided schema. A1 is your recommended first candidate before execution
feedback; this ordering will not be changed. Return complete source files,
entrypoints, configuration, required input fields and any training targets.
Do not return prose-only suggestions, extra candidates, or placeholder code.
You may reuse audited public modules through the supplied plugin contract.
Use the fixed DP backbone/training/data/control budgets. No new data, online
LLM control, privileged state, evaluator edits, or access outside this packet.

Your code must train and deploy causally. Future ground truth is training-only.
The three candidates need not improve over the baseline; give plausible failure
conditions, not invented expected scores. No execution scores are available yet.
```

### 9.2 Revision prompt

```text
You are RuntimePriorAPI continuing the same task/data-tier design session.
You now receive the committed A1/A2/A3 designs and their allowed development
feedback. Test outcomes are unavailable and must not be requested.

First return best_initial_candidate using the fixed development score and
tie rule. Then propose exactly one new executable candidate A4. Explain the
specific feedback, what changed relative to its parent design(s), what failure
it addresses, and what remains uncertain. Return complete code and config.

A4 is a new design to train from scratch on the same current support set with
the same training budget. It can simplify, combine or revise an earlier design.
Merely selecting an existing candidate is not A4. Do not propose A5, increase
the dataset or training schedule, copy a checkpoint, or fabricate diagnostics.
Report a real inability to provide a valid distinct design rather than inventing
an implementation. Negative results are legitimate; improvement is not required.
```

### 9.3 Final-selection prompt

```text
All valid A1/A2/A3/A4 candidates for this instance have completed their allowed
development evaluations. Return best_final_candidate from the valid candidates
by the fixed development score and tie rule, plus a brief evidence-based reason.
The runner will recompute the choice. Do not use any test result, generate code,
propose another design, or ask for another training/evaluation round. If no valid
candidate exists, return selection_unavailable with the recorded failure reason.
```

### 9.4 Implementation-repair prompt

```text
This is an interface repair for the same candidate slot, not a new search slot.
Here are the frozen contract and deterministic validation errors. No new
performance feedback is provided. Correct the implementation within the same
scientific design and budget. Return the full corrected package and an exact
change description. If a structural redesign is required, state that explicitly
instead of hiding it as a syntax fix. Do not request additional demonstrations.
```

## 10. 可恢复执行与命令入口

实现本仓库真实可运行的 Pixi tasks；下面的命令是**需要交付的入口契约**，不能在它们尚未实现时声称已经运行：

```bash
pixi run experiment1-preflight
pixi run experiment1-run --resume
pixi run experiment1-status
pixi run experiment1-report
```

- `preflight`：核对已授权清理、依赖/环境/数据/协议、B0/B1、真实 API 与插件；生成明确检查状态和缺失项。
- `run --resume`：执行完整 24 实例设计/开发、全局冻结、测试、报告；重复启动只继续未完成阶段。
- `status`：当前真实 counts、失败/阻塞、当前 job、GPU/磁盘和估计剩余时间（有测量依据才估计）。
- `report`：从已有不可变记录生成完整或明确标记 partial 的报告；不能触发新设计与新训练。

接收本文件后，开发 agent 必须在实现这些入口之后实际执行 `preflight` 和 `run --resume`，不是只告诉用户可以运行。

使用现有 Slurm/tmux/进程管理方式使训练能独立于聊天连接继续，并保存确切 job/session ID 和日志路径。启动前查询实际资源，不抢占其他用户 GPU、不终止无关作业。训练并行度基于显存检查；默认保守逐 GPU 调度，不要求多 agent。

不得保证 LLM 会话中断后没有独立进程的工作仍会继续。runner 必须本身能调用 API 和推进状态机；不能每一阶段都依赖聊天 agent 人工点下一步。

状态至少覆盖：

```text
planned -> evidence_ready -> initial_committed -> validated
-> training -> trained -> dev_evaluated -> revision_committed
-> selections_frozen -> globally_frozen -> test_evaluated -> reported
```

同时支持 `invalid`、`failed`、`blocked`、`interrupted`，保留原因与可恢复位置。实例和单系统状态分开存；多个阶段不要硬塞进一条不可区分的成功 flag。

API response 在消费前原子保存；request ID 与 evidence/code hash 匹配。恢复时重用已提交 response，不重新抽取一个更有利的 proposal。传输重试日志保留且有有限次数。无法确定 provider 是否已经处理请求时注明可能重复 API 费用。

checkpoint 包含 model、EMA、optimizer、scheduler、RNG、sampler/window 进度与 step，支持同一 job 精确恢复。若只能重启，明确记为新训练 attempt、费用累计，不称精确续训。

每系统最多保留三个常用 checkpoint 文件：最终固定步 EMA、最新恢复状态、一个按固定时间/步数策略保留的恢复备份。不得保留多个 dev-best 供隐藏选择。API代码、配置、曲线和评估记录长期保留。

有可恢复局部失败继续其他独立实例；共同数据/控制错误会影响所有结果时停止受影响队列并修复，记录失效范围。不要在错误公共接口上盲目跑满144次。

## 11. 必须留下的记录

### 11.1 Protocol 与模型索引

`protocol.lock.json` 至少含任务/版本、factor ranges、N列表、seed、六种系统、训练/评估配方、API模型/模式与调用限制、baseline与prompt/code hashes、候选预算、selection rule、容量约束、数据权限、开始时间。

`matrix.jsonl` 恰好 144 个唯一正式槽位，主键 `(task, n_demos, replicate_id, system_id)`。训练 attempts 另表，不能用一次重试制造第145个科学候选。

### 11.2 每个 API 设计

保存 evidence manifest、实际请求与响应（去秘密）、模型元信息、candidate源码/配置/hash、可执行入口、prior解释、证据来源、验证日志、repair版本链、训练job与checkpoint hash、参数/时间、dev记录、revision父子关系和q选择来源。

即使 API 仅组合公共原语，也要保存其真实提交组合与调用证据。报告区分 `API-authored code`、`API-selected public module`、`repository-agent-authored shared code`、`repository-agent-authored B1`。

### 11.3 每个 rollout

至少包含 task、N、system、replicate、split、subcondition、initial_state_id、episode_seed、policy_sampling_seed、success、timeout、termination_reason、steps、候选诊断、checkpoint hash、protocol hash、开始/结束时间。

避免只存均值；每种成功率要有分子分母。窗口、演示和独立回合不是同一种样本单位。

### 11.4 成本

分别记录唯一训练 jobs、system slots、失败/重启、updates/训练窗口数、GPU秒/型号、总参数/可训练参数、推理延迟、env steps/时间、API calls/tokens/可得费用、开发 agent/API 作者边界及人工改动。

q预算曲线用实际累计候选成本；生成初始三方案的API调用成本不能假装在q1时只支付三分之一。q1主要是复用批量方案中的候选预算前缀，不是独立单方案API总成本实验。

## 12. 最终完整报告：Experiment 1

交付 `experiments/experiment1/EXPERIMENT1_REPORT.md`，并提供可读中文结论；术语/表格列名可用英文。报告从真实 records 生成，同时输出可复用 CSV/JSON 和独立 PNG/PDF 图。

不得以一个“运行完成”段落替代报告。至少包括：

1. **完成状态**：计划144、有效完成训练数、dev/test完成数、invalid/failed/blocked数，实际seed与支持数据重复数；缺失项逐一列出。
2. **仓库迁移**：Round1–3归档位置、Round4已删除的精确范围、公共代码变化、保留/缺失历史材料。
3. **角色与执行真实性**：开发agent做什么，API实际做什么，实际模型/调用数、代码来源、人工/agent repair，有无选择不合规。
4. **实验设置**：六任务、factor intervals、D_N、state input、controller、版本、固定训练预算、dev/test预算与信息边界。
5. **144行模型结果表**：每系统IID/C/E成功率与分母、OOD宏平均、B0/B1差值、参数、训练时间、延迟、有效性状态。
6. **24行过程结果表**：每task×N的B0、B1、API_q1、API_q3、API_q4结果，q3/q4选择了谁、是否接受A4、开发/测试增量。
7. **少数据曲线**：六任务分别画N=2/5/10/20的IID、C、E或明确OOD图；主线B0/B1/q1/q3/q4。另提供六任务等权宏平均；不要用不同数量rollout混成样本加权“总体”。
8. **API具体设计**：每task×N的A1–A4做了什么、在哪个模块实现、有什么信息依赖、与B1的区别；相同A1编号跨任务不是同一种prior。
9. **修订分析**：API看到什么失败、A4改了什么、训练是否有效、是否被选、dev/test是否提高；列出退化和未被选案例。
10. **规则prior与API价值**：q1/q3/q4分别相对B0、B1的配对差值；区分单候选质量、多个候选选择和额外修订费用。
11. **成本/稳定性限制**：总GPU时间、API成本、env交互、144槽位外的检查/重试成本；明确一个完整重复与一个训练seed不能证明跨seed稳定。
12. **问题与下一步**：真实失败类别、协议偏差、未解决工程问题、哪些结论现在不能支持；不要为了正面结论修改实验。

建议机器可读输出：

```text
reports/model_results.csv
reports/procedure_results.csv
reports/episode_results.parquet   # CSV acceptable if stack lacks parquet.
reports/prior_designs.jsonl
reports/revision_effects.csv
reports/costs.csv
reports/validation_summary.json
reports/completion_manifest.json
figures/learning_curves_*.png
figures/learning_curves_*.pdf
figures/revision_effects.png
figures/task_tier_effects.png
```

绘制科研图使用真实matplotlib等绘图库，不用生成式图像。图表的数值源、聚合规则与缺失数据处理要可追溯。

### 12.1 统计边界

- OOD = 等权C/E；任务宏平均 = 六任务等权。N每档单独报告；不要把四个嵌套数据集当四次独立随机重复。
- 固定N的成功率差值用 percentage points。可报告配对episode bootstrap，但保持相同初态配对和子条件分层。
- 单seed的区间最多反映这些冻结模型在抽样测试状态上的不确定性，不能解释成完整设计过程的跨seed置信区间。
- 六个已开发任务不是来自所有机器人任务的随机样本，不能外推为普适agent能力。
- q4−q3同时包含一次额外候选训练与反馈修订；本轮没有一次性四proposal的batch对照，所以不能单独归因于反馈。记录为“增加反馈修订候选后的实际增量”。不要擅自加该对照扩大矩阵。
- 本轮没有generic coding agent/随机搜索/最强专家portfolio，不能声称已经胜过这些方法。
- 设计先验多因素共同改变时，不把效果解释成某一个loss的独立因果贡献。
- q1不等同独立单proposal prompt；相同数据量下每个条件独立设计，不把API给D2设计的固定prior复制到所有N后声称按N自适应。
- 一条图线中每个N可以来自不同设计；报告这是完整设计流程随数据量变化的性能，不是同一固定架构单纯加数据的曲线。

## 13. 验收条件与结束方式

所有完成声明须由记录支持：

- [ ] 实际Round1/2/3已归档，Round4专属目标已删除，公共代码无断链。
- [ ] Pixi依赖锁定、六任务控制/数据审查通过或明确标记阻塞。
- [ ] B0/B1定义、API提示、插件接口、数据与开发/测试规则先冻结。
- [ ] 24个实例、144个唯一正式槽位；默认只有replicate0、seed0。
- [ ] A1/A2/A3顺位在反馈前提交；每task×N独立API证据。
- [ ] A4确实由API看前三个dev后设计并新训，或留下真实invalid/blocked原因。
- [ ] q1/q3/q4是指向实际checkpoint的选择，不能用test最大值。
- [ ] 所有设计和选择先全局冻结，再开始隐藏测试。
- [ ] 正式模型和API调用确实运行；mock/旧checkpoint没有混入。
- [ ] 每系统/回合/成本/失败/代码来源记录齐全，可恢复、可复核。
- [ ] 完整报告、CSV/JSON与图表已生成并检查数值一致。
- [ ] 没有把pending写成结果；没有凭空声称144/144完成。

执行者最终回复必须给出真实完成计数、实际命令、报告路径、选定结果要点与剩余阻塞。如果工作仍在独立进程运行，提供真实job/session、当前进度和报告路径，明确尚未完成；不能写“实验已完成”。在当前授权与资源可用范围内持续推进，不在仅有计划或可执行脚本时结束任务。

## 14. 用户可直接粘贴给仓库 Codex 的启动语

```text
请读取 EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md 并在当前实际训练仓库执行。
先精确删除未完成Round4的专属内容，archive Round1/2/3，整理公共代码。
本轮叫Experiment1：六个MetaWorld任务，N=2/5/10/20，一个完整重复seed0，
每个task×N训练B0普通DP、B1你预先实现冻结的rule-prior DP，以及API生成的
A1/A2/A3和根据前三者开发反馈生成的A4，共144个正式系统训练槽位。
开发agent负责实现与运行基础设施/B1，真正候选prior必须通过实际模型API
返回可执行代码；不能由你代写候选后冒充API设计。q=1/3/4复用这四个候选。
按文档实现可恢复的Pixi入口，实际运行、全局冻结后做独立测试，生成完整报告。
遵守已有授权持续完成，不只给计划；若API/数据/资源缺失，完成能做的部分，
准确记录blocked状态并告诉我具体缺少什么，不用mock或旧结果替代。
```
