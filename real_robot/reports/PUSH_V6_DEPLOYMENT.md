# Push v6：实际下载包与调用核验

核查日期：2026-09-22。开发者仅导出/验证部署框架；API科学源码与选中张量完全保留。
旧Push v5、Flip egg v1、deployment_pipeline和pipeline_v1归档不变。

## 下载内容

| 归档 | 字节数 | 内容 |
| --- | ---: | --- |
| [完整包](../exports/pipeline_v2/pipeline_v2_bundle.tar.gz) | 342,920,786 | 代码、API契约、示例、环境锁、权重、SAM，约327MiB |
| [策略权重](../exports/pipeline_v2/pipeline_v2_weights.tar.gz) | 171,448,983 | 选中state dict，包括API模型中冻结的SAM张量 |
| [SAM资产](../exports/pipeline_v2/pipeline_v2_sam_assets.tar.gz) | 171,378,608 | API模型初始化所需的独立冻结资产 |

先Git同步代码时，需要后两个归档；直接用完整包则不依赖Git是否推送。
本轮SAM资产为必需，不能照旧v5可选资产的说明省略。推理不需要训练数据或API key。
安装/Agent接入见[README](../deployment_pipeline_v2/README.md)，
哈希见[SHA256SUMS](../exports/pipeline_v2/SHA256SUMS)及[manifest](../exports/pipeline_v2/bundle_manifest.json)。
尚未执行Git提交/推送或Drive上传。

## 实际核验

- 16个API源码/契约文件逐字节导出；18,497个可训练参数，冻结SAM另有46,060,610个state元素。
- 推理checkpoint的选中张量与原训练checkpoint逐一完全相等；没有改架构、量化或重新训练。
- 在不含runs/原始示范的独立目录解压实际完整归档，核对47个文件。
- 在独立GPU worker实际加载，接触点、方向、行程相对原worker差异均为0；执行器目标也一致。
- 过期场景、中断、不完整场景按预期拒绝；无标定返回calibration_required。
- 完整包里的CLI example实际运行并与原worker输出一致，使用的是真实选中模型。
- 单次测得加载9.09秒、已给几何场景的act调用86.37ms；包含IPC，不含RGB感知/物理执行，非稳定延迟分布。

例子采用记录中的物体轮廓、事后目标、合成单物体场景与合成标定标志，验证调用链。
它不是接收端实时标定或闭环成功试验。实际输出之一为边界接触[0.504166,-0.232668]m、
方向[-0.999494,0.031821]、行程0.010m；不能直接将该例坐标当成真机指令。

接收端仍需提供实时实例关联/几何完整性、目标布局、相机/工具标定、碰撞模型和
planner/controller执行反馈。调用契约已提供这些字段，但没有自动完成外部系统集成。
正式训练接触标签差异16.403mm，不能从接口通过推断物理推移效果。

证据：[实际归档迁移回执](PUSH_V6_DEPLOYMENT_CHECK.json)、
[独立worker](../runs/push_letters/training_v6/host_worker_check/result.json)、
[训练报告](PUSH_TRAINING_V6.md)、
[旧产物保留核验](../runs/push_letters/training_v6/previous_artifacts_preservation.json)。
