# Push v6：从原 v5 流程重新执行

日期：2026-09-22。按用户最新指令，新运行叫 Push v6；不是利用 v5 训练结果
再做一轮修正。此前已取消的额外修正轮次及其专用 prompt、源码、journal、
预处理产物已删除，仅保留删除和费用记录。

`implement_and_train_final_fit.md` 是原 `implement_and_train.md` 的完整内容，
再增加一段跨任务要求：尽可能使用适用的有效示范训练并交付最终模型，API
自行选择诊断、采样、模型及训练设置，说明实际梯度覆盖、排除项与评价性质。
没有增加具体样本数、拆分比例、模型结构或训练步数。

新旧配置差异仅为运行路径、配置路径、prompt 路径、授权和范围说明。cut_v6、
任务事实、原始数据、可用工具、接口、资源上限、模型档位和原有 v4 流程审查
输入均与原 v5 一致。API 不接收已完成 v5 的源码、训练指标、划分审查或取消
轮次的任何科学方案。设计、预处理及训练实现全部重新由 API 产生。

使用现有通用 `train_policies` 入口，执行 API 实现、数据检查、提交、训练和
调用核验；已有的有界实现修复/数据恢复机制保留，无自动传输重试。此前 frozen
训练框架、v5 模型、Flip v1 模型和旧部署包的哈希已复核不变。不重做 cut。

- [本轮配置](../configs/push_training_v6.json)
- [通用 prompt](../prompts/implement_and_train_final_fit.md)
- [新旧差异与旧产物核验](../runs/push_letters/training_v6/setup_receipt.json)
- [实时流程状态](../runs/push_letters/training_v6/workflow.json)
- [取消轮删除记录](../deletions/push_v5_feedback_continuation_20260922/deletion_receipt.json)
- [取消轮费用](../deletions/push_v5_feedback_continuation_20260922/API_USAGE.json)：25次，估算$4.699118，usage无未知；不计入新v6。

本文说明运行输入与流程，不代表训练完成或模型性能。
