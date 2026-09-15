# Exp2 执行报告

正式实验门槛：未通过。未满足项：ID_development。
正式 M0/M1/M2 ID/OOD 矩阵尚未完成时，下面所有诊断都不能冒充正式主表。

| 诊断 | 完成状态 | 闭环成功 |
|---|---|---|
| D0 | 通过 | 3/3 |
| D1 | 通过 | 采样检查通过 |
| D2 | 未通过，结果保留 | 0/1 |
| D2_sampling | 未通过，结果保留 | 0/2 |
| D2_quaternion | 未通过，结果保留 | 0/1 |
| D3/open_drawer | 未通过，结果保留 | 0/1 |
| D3/move_red | 未通过，结果保留 | 0/1 |
| D3/move_blue | 未通过，结果保留 | 0/1 |
| D4 | 未通过，结果保留 | 0/20 |
| capability_M1 | 未通过，结果保留 | 0/1 |

当前记录 15 次神经训练、接口或恢复检查作业，共 195,020 次梯度更新、24,960,640 个窗口曝光，训练作业计时 3975.3 GPU 秒（不是 GPU kernel profiler 时间）。这不是正式候选模型数。接口/基础设施检查单列于 CSV。

## API 实际贡献
真实 API 候选 `drawer_geom_prior_v1`，版本 `c4bc9a08b86b830f8e8515cde161e71b57ee30fff2c42d43c9deebd6557e3c7e`。

Neural diffusion subpolicy for the opening phase.  The prior is a drawer-frame state representation: in addition to the common normalized two-step observation history, the condition vector exposes TCP and red-block positions relative to the estimated drawer handle point (x=-0.14-drawer_position, y=0, z=0.128), drawer travel progress/remaining travel/velocity, and gripper aperture/symmetry.  This encodes the useful geometry of grasping the handle, pulling along the drawer axis, releasing, and clearing upward, while leaving action generation to the learned diffusion backbone.  A mild loss weight emphasizes the gripper channel because the close-pull-open sequence is discrete and important for the drawer subgoal.

API 自选所有训练轨迹前 310 步作为开抽屉片段，加入抽屉/把手相对几何条件与轻度夹爪加权 epsilon loss。
统一 DP 的条件从 94 维增至 122 维；主干宽度/深度保持一致，参数量变化单独记录。
动作仍解码到原生绝对关节角，不能将观测相对化解释成关节动作平移等变。
完成 5000 次真实更新，权重 L2 变化 20.2107，重载最大误差 0.0。
原生执行 未成功。同切分、同 5000 步 M1 能力对照也未成功；单一诊断不能估计 prior 的泛化收益。

## MuJoCo 与场景有效性
API 重建版本 `b8d81679df15efc9640ebed2cd7c809326a5ab04e25b4d5b8a7e319ac79fd7be`，固定校准通过：True。
API 从原始 URDF 和授权示范修订指尖接触几何、摩擦与接触求解参数；开发者仅修复受控 XML 工具。
固定校准实际结果见 calibration.csv。完整 demo1000 红块位置 RMSE 从 0.177240 m 降到 0.009316 m。
这证明已校准案例的重放一致性，不证明所有状态的仿真可信度；中途复位缺原生接触快照，物体角速度设零的限制明确保留。
独立参考控制器在 ID / 位置 OOD / 状态 OOD 各两例全部通过，未增加训练数据。位置 OOD 与状态 OOD 范围见 PROTOCOL.md。
该 API 神经候选在校准后的 MuJoCo 中也实际执行了 2 次，成功 0 次。这是外层能力验证，不冒充 API 完整库的一次正式反馈修订。

## 尚未得到的研究结论
1. DP 当前失败的剩余证据支持接触误差、闭环分布偏移和阶段行为错误；训练不足/覆盖仍不能排除。
2. 尚无完整 M0/M1 配对结果，不能量化分段/层级收益。
3. 尚无正式 M1/M2 ID/OOD 配对结果，不能宣称 prior 提升泛化或不损害 ID。
4. 真实 API 已贡献可训练的几何条件/loss 候选和通过校准的 MuJoCo 模型；候选成功率未因此自动获得保证。

## 交付与边界
唯一有效源码 src/appl，配置/文档 experiments/exp2，独立 Pixi 环境 environments/exp2；大产物在 runs/exp2 的存储盘映射。
Exp1 的 85,566 个持久文件 hash 一致；原本为空的 SQLite WAL 和 SHM 是连接临时文件，主数据库 hash 不变。原入口核验 144 槽位完成、artifact issues=0。
正式设计/训练/评估入口有门槛保护；第二版冻结后拒绝写代码、重训和更换成员。整个正式三技能周期尚未通过真实运行验证，不写成完成。
真机驱动、硬件控制语义、标定与感知仍待配置；没有伪装成真实机器人验证。
完整 API 工具调用/响应、版本、模型、预算保留；代理服务未提供美元账单，因此仅报告 token、调用、时间和训练成本。
正式结果未形成时没有置信区间或效果量；配对多种子统计须等实际矩阵存在后生成。
视频为固定随机种子从保存回合抽样、1 Hz 的原始 RGB 帧序列：[reference_success](../../runs/exp2/rebuild_20260915/report/reference_success.mp4), [learned_failure](../../runs/exp2/rebuild_20260915/report/learned_failure.mp4), [API_capability](../../runs/exp2/rebuild_20260915/report/API_capability.mp4).

数字来源：[summary.json](../../runs/exp2/rebuild_20260915/report/summary.json)、[训练表](../../runs/exp2/rebuild_20260915/report/training.csv)、[逐回合表](../../runs/exp2/rebuild_20260915/report/episodes.csv)、[API 成本](../../runs/exp2/rebuild_20260915/report/api_costs.csv)。
