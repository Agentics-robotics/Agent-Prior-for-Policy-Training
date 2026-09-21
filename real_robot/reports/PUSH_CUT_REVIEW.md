# Push cut_v3 审阅与片段清单

核对日期：2026-09-20。开发者审阅记录；切点、条件、分组和 heuristic 全部来自 Runtime API，未人工改写已提交方案。

本轮完成 **22 个片段（两条轨迹分别 12 / 10）、1 个共享数据组、2 个独立 prior / policy 设计**。12 项接口测试、71 个发布文件核验及 25 次实际请求审计通过。模型为 GPT-6 Astra/xhigh；本轮费用估算 **$4.4980**，不是账单。

旧 cut_v2 数据与会话已删除，仅留删除清单和费用；原始记录与 Exp1/Exp2 未改动。

## 切分与 condition

每段指定一个物理实例和一个固定的局部空间目标，以明确的目标锚点帧与原始像素框定义 hindsight goal。临时挪动与后续再调整被分开；同一目标下的接近、换接触点和纠正可以保留在同一段。下表文字为 API 原文；字母名称是视觉助记，不是已验证字符标签。

所有范围均为原始配对 sample_index 的 [start,stop)。A = episode_2026091922002701；B = episode_2026091922062101。

| 片段 | 配对索引范围 | API condition |
| --- | --- | --- |
| a_w | [0, 1720) | Selected episode-local zigzag/W-like instance; fixed achieved placement G(1719). No word/class token is a policy input. |
| a_i_stage1 | [1720, 1920) | Lower crossbar/I-like instance, local staging placement G(1919). |
| a_o | [1920, 3220) | Annular/O-like instance; achieved placement G(3219). |
| a_r_stage | [3220, 3600) | R-like instance, temporary placement G(3599), not the final R row slot. |
| a_i_stage2 | [3600, 4230) | Lower crossbar/I-like instance; second local staging placement G(4229). |
| a_h_stage | [4230, 4620) | Upper crossbar/H-like instance; temporary lower-left placement G(4619). |
| a_d_stage | [4620, 4950) | D-like instance; staging placement G(4949). |
| a_r | [4950, 5770) | R-like instance; later achieved placement G(5769), distinct from G(3599). |
| a_d_align | [5770, 6140) | D-like instance; alignment placement G(6139). |
| a_l | [6140, 8050) | L-shaped instance; achieved placement G(8049). |
| a_i_align | [8050, 8500) | Lower crossbar/I-like instance; later placement G(8499). |
| a_h_align | [8500, 9015) | Upper crossbar/H-like instance; final recorded local placement G(9014), without success label. |
| b_w | [0, 1220) | B episode zigzag/W-like instance; achieved placement G(1219). |
| b_i_stage | [1220, 1440) | Central crossbar/I-like instance; local placement G(1439). |
| b_d_stage | [1440, 1900) | D-like instance; staging placement G(1899). |
| b_o | [1900, 2820) | Annular/O-like instance; achieved placement G(2819). |
| b_r | [2820, 4560) | R-like instance; achieved placement G(4559), constant through its correction/re-contact. |
| b_d_align | [4560, 4840) | D-like instance; alignment placement G(4839). |
| b_l | [4840, 6080) | L-shaped instance; achieved placement G(6079). |
| b_h_stage | [6080, 6900) | Upper-left crossbar/H-like instance; temporary placement G(6899). |
| b_i_align | [6900, 7300) | Central crossbar/I-like instance; later placement G(7299). |
| b_h_align | [7300, 7730) | Upper-left-origin crossbar/H-like instance; last recorded local placement G(7729), without success label. |

## 数据与监督范围

全部 **16,745 条原始记录**均落入切割文件，没有整段丢弃或拼接为虚假的连续轨迹。API 在 20 次实例交接两侧各屏蔽 30 行动作 loss，共 **1,200 行**，仍保留为上下文或目标锚点证据。剩余 **15,545 行**只是显式边界屏蔽后的监督候选上限，还要排除无效命令及失败的输入转换。

| 原始轨迹 | 边界屏蔽后候选行 | 其中有限 dq 命令行 | 其中缺失/无效 dq 行 |
| --- | ---: | ---: | ---: |
| episode_2026091922002701 | 8355 | 8095 | 260 |
| episode_2026091922062101 | 7190 | 7122 | 68 |

目标 mask、跟踪与其他衍生输入尚未计算，因此不能把上表有限命令行数当成最终可训练样本数。跨设备时钟没有重新匹配；配对索引核验也不等于曝光级同步测量。

## 两个独立设计，共享同一数据组

- **contour_push / boundary_graph**：利用对象轮廓、孔洞、局部接触几何及因果历史；形状和目标由当前观测及显式 condition 给出。
- **visual_push / dual_view_memory**：利用第三视角与腕部 RGB、目标标记和因果时序，保留轮廓抽取可能丢失的视觉线索。

两者计划分别训练，供 High-level Agent 选择。Agent 指定物理实例、空间目标、容差和调用时限，监控结果并决定切换；policy 内部决定接近、推动和重接触动作。现阶段没有实现或训练这两个 policy。

## 审阅结论与后续依赖

22 个文本契约均已给出实例、局部目标、像素框/锚点及 merge/split 解释，已将上一轮留待内部再判定的目标切换显式化。抽查了 A/1920、A/4230、A/4949、B/4559、B/6899 原图；这是有限抽查，不是逐帧真值标注。切点附近的不确定性保留在 API 的监督排除项中；统一的 30 行缓冲宽度尚未经实测验证。

未知字母身份、未知物理形状、全新词汇与布局的泛化要求已写入设计和独立评估计划，但尚未验证成功。分割/跟踪与字符方向识别模型权重、goal 转换、必要平面/工具标定、完整 policy/preprocessing/decoder 代码及真实控制器语义核验均是后续依赖。没有用人工逐次选择推哪个物体来替代 High-level Agent。

## 证据

- [完整 API 结果报告](PUSH_CUT.md)
- [API 原始 plan](../data/push_letters/cut_v3/plan.json)
- [完整 heuristic 与逐段监督契约](../data/push_letters/cut_v3/datasets/instance_relocation/heuristic.md)
- [拟议 policy 调用目录](../data/push_letters/cut_v3/POLICY_CATALOG.md)
- [结构与来源核验](../runs/push_letters/cut_v3/validation.json)
- [本审阅的机器可读记录](../runs/push_letters/cut_v3/requirements_audit.json)
- [费用记录](../runs/push_letters/cut_v3/cost_ledger.json)
