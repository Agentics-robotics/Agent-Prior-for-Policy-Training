# training_v4 的 prompt、接口与样本流失审查

审查日期：2026-09-22。开发者对已保存请求、API 源码、标签审计和代表图像的复核。
本次未调用 Runtime API、生成新训练标签或训练模型；原训练源码和权重哈希保持不变。
数值证据见 [审查 JSON](PUSH_TRAINING_V4_PROMPT_AUDIT.json)。

## 结论与约束来源

用户要求 API 根据数据编写模型、完成必要的数据处理并训练，并未要求固定 47 个训练位置。
47 个稀疏位置最初由 cut_v6 的 Runtime API 选择；之后由 RepositoryAgent 编写的
training_v4 prompt 和校验器将它们固定为唯一合法监督位置。这是开发者额外施加的约束，
不是用户要求，也不是数据本身的限制。

开发者将“继承上一轮设计并保留来源”过度收紧成了“训练阶段不可调整采样方案”。
保留原始录制和已完成版本，与允许 API 在新版本中扩展监督采样是可以同时做到的。
因此，当前结果既暴露了感知/标签生成的不足，也暴露了外层任务与验收设计的问题。
此前“训练完成”的确对应真实优化和可加载权重，但不能代表已合理利用数据或产出可用策略。

| 决定 | 谁作出 | 实际后果 |
|---|---|---|
| cut 可选择 dense 或 sparse | 开发者提供两种模式，API 作选择 | cut 阶段没有固定 47 的配额 |
| 12 个学习片段、47 个稀疏点 | cut_v6 Runtime API | 是设计阶段的采样决定，尚未经过数值标签验证 |
| 47 个位置必须保持不变 | RepositoryAgent 的实现 prompt / 接口 | 实现 API 不能在别的帧补充监督 |
| 颜色分割、几何门槛、完整联合标签 | 实现 Runtime API | 多个条件必须同时满足才保留一行 |
| 少量样本也可以通过训练入口 | 开发者的验证器只要求 K >= 1 等结构条件 | 数据覆盖不足没有触发继续修复数据流程的要求 |
| 最后选 500 步 | 实现 Runtime API | 是面对四条样本的下游预算决定，不是样本减少的起因 |

直接来源：

- [通用 cut prompt](../prompts/general_cut_and_prior_v5.md) 第 33、79–82 行允许 dense/sparse。
- [实现 prompt](../prompts/implement_policies_v4.md) 第 13 行要求保持 47 个位置不变。
- [接口](../policy_training_v4/INTERFACE.md) 第 114–119 行禁止从其他行新增训练锚点。
- [校验器](../policy_training_v4/engine.py) 第 139–147 行实际拒绝未列出的索引；第 90–92 行仅以 K >= 1 检查非空。
- 已核对全部 77 个保存请求，实际 instructions 与上述实现 prompt 一致，均为 Astra/xhigh。

## 数据如何缩减

16,745 行原始录制 → 47 个允许的监督锚点 → 几何处理保留 7 个 → 图像/运动复核保留 4 个。

16,698 个未列出的行没有作为新的决策样本参与筛选，其中既有连接运动，也有推动期间的
其他观测。它们可被读取作为上下文或目标证据，但不能生成独立训练行。不能把这些行说成
“被证明没用”。同样，邻近帧高度相关，扩展样本也不等于增加同等数量的独立接触事件。

以下是每个排除点记录的首个失败原因，不是相互独立的所有问题计数：

| 排除原因 | 数量 | 能支持的结论 |
|---|---:|---|
| 目标轮廓拓扑不符或配准误差 >10 mm | 19 | 当前轮廓/目标转换未通过；不是物理动作不可学习的证明 |
| 接触搜索未通过关联、距离或高度条件 | 14 | 当前接触估计未成功；原因还包含感知和工具几何估计 |
| 当前或未来物体轮廓提取失败 | 4 | 当前分割路径失败 |
| 短推动结束处未通过接触验证 | 2 | 完整短推动标签尚不充分 |
| 短前缀位移不足 4 mm | 1 | 不适合当前短行程标签规则 |
| 物体运动难以与配准噪声区分 | 1 | 不能直接把估计运动当可靠推动监督 |
| API 图像复核发现阴影/遮挡导致局部轮廓 | 2 | 当前标签需要修复分割或另取观测 |
| **合计** | **43** | **尚未测量其中多少可以恢复** |

## 为什么处理流程容易丢样本

1. 实现使用固定 HSV/RGB 颜色阈值和单帧轮廓，目标姿态依赖轮廓刚性配准。
   工具遮挡、阴影、孔洞未检出会改变轮廓，继而触发排除。
   SAM/Qwen 依赖在 cut 中被提出，环境和下载工具已提供，但本次 request_model 与
   request_asset 调用均为 0。API 明确选择了颜色方案，未验证另一种感知方案的恢复能力。
   不能据此保证换成 SAM 就能解决全部问题。
2. 当前锚点与指定未来帧的完整目标几何先通过检查，才继续提取接触。
   即使局部推动可能清楚，只要指定未来帧的轮廓有问题，整行也会在更早阶段被排除。
3. loss 是完整接触点/方向/长度的联合候选 NLL。虽然 cut 文档提到无效分量可 mask，
   实现实际把排除行的 target_validity 全部置 false，并丢弃该训练行。
   没有实现“已知分量训练、未知分量屏蔽”的部分监督损失。
4. review.py 还写有四个已审阅位置的 approved 集合。最终四个有实际图像审阅记录，
   但这个位置清单并不是可扩展的数据处理规则。此次从 7 到 4 的实际原因是
   A5600 运动不确定、B6500 阴影轮廓问题、B7500 工具遮挡；不能将所有 43 个排除
   都归因于 approved 集合。
5. prompt 强调有效性、审计、提交可执行包，但缺少“覆盖损失很大时，继续恢复监督数据”
   的成功标准。校验器证明来源、有限数值、梯度和接口正确，没有证明数据充分。
   这是本地代码所显示的任务激励；不是对模型内部思维的断言。

## 代表证据

- B3600：配准 RMS 6.215 mm，低于 10 mm；根据代码的两项判据，排除触发项是
  当前/未来轮廓的环数不一致。A5100 同理，RMS 9.328 mm。真实物体的孔洞不会因
  一次平面运动凭空改变，需处理可见性、分割或关联问题；这两点仍不能直接升格为有效标签。
  [B3600 审计](../runs/push_letters/training_v4/package_00/prepared/evidence/final_B_3600.json)
- B5500：完整 L 形块仍在图像中，而绿色分割只覆盖其一小部分。已有叠加图支持
  “当前感知提取不足”的判断，没有完成修复后标定/接触验证。
  [B5500 图像](../runs/push_letters/training_v4/package_00/prepared/evidence/B_5500.png)
- A1600 的有效短推动内还有连续录制帧，但当前规则只允许 A1600 作为监督锚点。
  可以研究从同一稳定交互中采样更多不同当前状态及后续短动作；实际可用数量待转换和验证。
- 原始两段中的 T_base_flange、T_base_ee、q 共 16,745 行全部为有限数值，
  不是因为机器人位姿字段整体缺失而只剩四条。物体接触几何与可靠目标标签仍需估计。

## 建议的新任务合同（尚未执行）

新一轮应保留原始数据、已有 cut/policy 版本和来源追溯，同时将训练样本构造交还 API。
以下是建议的目标表述，不是对已完成 training_v4 的改写：

> Use the previous decomposition and prior documents as the starting design.
> Build a trainable dataset from the available demonstrations, choosing the
> supervision density, decision times, observation windows and target derivation
> appropriate for the learned output. Trace every example to its source records.
> Existing inspected decision points are evidence references for this revision;
> select additional valid examples throughout the assigned interaction intervals.
>
> Make useful supervision coverage and event diversity explicit implementation
> objectives. Diagnose losses by interaction type and processing stage. Recover
> supported supervision through improved perception, temporal association,
> geometric estimation and appropriate sampling. Use per-target validity and
> uncertainty where partial supervision is scientifically meaningful. Select the
> model and fixed training budget after reviewing the resulting dataset coverage.
> If required information is absent, identify the precise annotation or
> calibration needed with representative evidence and a measured coverage report.

对应接口也要支持新样本索引和必要的标签 mask；只修改文字、保留原校验器不会生效。
有效性检查依然必要，调整方向是改善标签与使用可支持的信息，而不是无差别放宽阈值。
推理时调用频率和训练时取样密度应分别决定；输出为短接触决策并不要求整段推动只有一条监督。
泛化判断需要独立接触事件/片段与行为覆盖，不能仅靠重复抽样次数。

OpenAI 官方提示建议强调清晰的最终目标与成功标准；这里的具体失效原因由上述本地证据确定。
参考：[Reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)。
