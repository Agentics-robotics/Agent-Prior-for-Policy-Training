# Push v5：训练与开发数据划分审查

核对日期：2026-09-22。用户质疑极少数据下开发集94例大于训练集74例。
本次检查已保存的prompt、API设计、采样源码及缓存；没有新增模型训练。

| 来源 | 有效例 | 原交互组 | 实际用途 |
| --- | ---: | --- | --- |
| 录像A / episode_2026091922002701 | 74 | 0–5，共6组 | 唯一梯度训练来源 |
| 录像B / episode_2026091922062101 | 94 | 6–11，共6组 | 开发指标、早停、checkpoint选择 |

全部168例的policy_weight均为正，B不是因标签不合格而无法参与训练。
最终代码 `sample_indices` 只允许example_episode==0；`after_update` 用B的
等权组平均误差选择权重。训练实际更新800次，部署选第600步。没有B→A第二折，
也没有A+B最终拟合。准备数据覆盖12/12组；真正参与参数更新的是A的6/12组。

划分来源是Runtime API，通用prompt/框架未指定A/B或固定开发比例。
INTERFACE.md允许API自主选择采样与划分。旧cut_v6的
`sharing_and_diversity_audit` 已提出整录像分组，初步实验先A训练/B评估再交换，
避免随机相邻帧或重叠窗口泄漏。实现阶段采用A训练/B开发，并承认B已参与数据
转换设计，不能算未接触的测试集。B有效例更多是筛选后的结果，不是预先规定
开发集占56%。此前两个缓存的A/B计数分别65/77、99/106，最终为74/94。

整录像隔离用于迁移诊断有合理目的；在当前“尽量利用两条示范，交付真机可试
模型”的目标下，把仅A训练的开发模型直接作为最终交付，使B的全部有效动作
经验没有用于参数学习。原prompt允许API选择训练与checkpoint，但没有明确
区分开发诊断与最终交付拟合，框架也没有单独的最终拟合阶段。这是流程目标
表达与实现选择的不足，不能用“防止过拟合”一笔带过。

建议后续通用流程明确两个阶段：API决定有意义的开发诊断与训练方案；在冻结
方案后，为部署目的利用全部适用有效示范拟合最终模型。小数据时应解释留出
造成的行为覆盖损失，而不是机械套固定比例。新独立录制/真机测试才用于后续
泛化测量；全量拟合后的B误差不能再作为留出证据。增加到168例不保证解决标签、
表示或模型相对基准的劣势，需要新实验确认。

证据：

- [最终采样与checkpoint选择](../runs/push_letters/training_v5/package_01/source/policy.py)
- [首次实现的划分说明](../runs/push_letters/training_v5/package_00/source/PRIOR.md)
- [原cut的sharing_and_diversity_audit](../data/push_letters/cut_v6/plan.json)
- [通用接口](../training_pipeline/INTERFACE.md)
- [已执行的通用prompt](../prompts/implement_and_train.md)
- [实际数值摘要](../runs/push_letters/training_v5/training_summary.json)

方法参考：[scikit-learn交叉验证说明](https://scikit-learn.org/stable/modules/cross_validation.html)
讨论固定留出消耗训练样本的问题；[模型选择后的refit接口](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
将选择方案后在全部拟合数据上重新训练列为独立步骤。这里引用的是阶段区分，
不建议给当前策略机械套网格搜索或随机帧划分。
