# 两任务跨电脑推理包

核对日期：2026-09-22。用户希望Git同步后通过Drive下载权重并在另一台电脑测试。
已提供独立 [部署入口与安装说明](../deployment_pipeline/README.md)，未执行Git
commit/push、Drive上传、新API设计、训练或机器人动作。

旧训练调用器依赖本机Pixi/GPU配置，且API模型源码位于Git忽略的runs目录。
新增 `deployment_pipeline` 使用接收电脑的当前Python和显式GPU编号；两份API
源码、调用说明、handoff、manifest、训练回执与usage完整复制到Git可管理的
`policies/push_v5`、`policies/flip_egg_v1`。科学源码哈希与原提交一致。

推理导出逐张量等于原checkpoint的 `selected`，无量化、无参数删减或重训。
Push为600步选中权重，Flip为2000步选中权重；原始checkpoint保持原SHA-256。
小文件来自参数量和剥离优化器/训练元数据，不表示缺少模型层。

| 导出模型 | 参数量 | 文件字节 | 原模型输出 |
| --- | ---: | ---: | --- |
| push_v5/shape_push_v1.pt | 9,641 | 47,909 | 接触后二维EE位移；初始接触使用API几何规则 |
| flip_egg_v1/egg_interaction_v1.pt | 278,903 | 1,130,869 | 末端局部六维速度与预测分布信息 |

下载文件已在本机生成，可由用户通过Drive或SCP传输：

| 文件 | 压缩后字节 | 内容 |
| --- | ---: | --- |
| [两模型权重](../exports/pipeline_v1/pipeline_v1_weights.tar.gz) | 1,076,362 | 两个学习模型的推理checkpoint，含目标解压路径 |
| [可选SAM资产](../exports/pipeline_v1/pipeline_v1_sam_assets.tar.gz) | 171,378,608 | Push的inventory使用；若上游提供几何则不需要 |
| [完整独立包](../exports/pipeline_v1/pipeline_v1_bundle.tar.gz) | 172,573,695 | 代码、锁定环境清单、两模型、SAM及调用说明 |

归档SHA-256见 [SHA256SUMS](../exports/pipeline_v1/SHA256SUMS)，完整allowlist
与每个文件哈希见 [bundle_manifest.json](../exports/pipeline_v1/bundle_manifest.json)。
包中没有原始录制、训练缓存、API凭据或完整runs目录。环境软件依赖由接收端
`pixi install --locked`下载；在线观测、标定和真实执行器绑定由接收端提供。

实际完整包在 `/tmp` 新目录解压，62个文件全部核验。两模型使用原录制输入及
明确标注的合成接口确认字段，预测与原独立调用器最大绝对误差均为0；中断与
未绑定执行器的拒绝返回保持一致。可选SAM用真实录制RGB实际运行，返回9个
未验证分割提案，未把其当成完整物体几何或接触证明。见
[迁移接口检查](PIPELINE_DEPLOYMENT_CHECK.json)。另检查缺少SAM资产时仅用
几何输入调用Push，见 [仅权重路径检查](PIPELINE_WEIGHTS_ONLY_CHECK.json)。

这些检查验证迁移、加载和接口一致性，不新增任务性能证据。Push开发误差
5.095mm仍高于目标平移基准4.764mm，Flip仍为隐式视觉末端模仿；原训练报告
的泛化与感知/执行依赖限制继续适用。
