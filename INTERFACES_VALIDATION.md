# 本次接口实施的验证记录

实现分支：`dev`。所有 Python／项目验证均通过现有 Pixi 环境执行。没有运行正式训练、重新生成历史权重、开发集选模或锁定测试。

## 已通过

| 验证 | 实际结果 |
| --- | --- |
| 默认 Pixi 完整回归：`python -m pytest -q` | 248 passed，4 skipped |
| Round 4 Pixi：`python -m pytest -q tests/interfaces` | 153 passed，3 skipped |
| GPT-6 Astra／max 真实 API | HTTP 200；返回模型和推理档位与请求一致，状态 `completed`，输出“接口检查通过。” |
| MetaWorld 实际 reset/step 对照 | 1 passed；使用 drawer 的 `train20[0]`，不读取开发／测试状态 |
| 物理 GPU 0：Round 4 数值策略／适配器对照 | 1 passed；沙箱外实际执行，动作与随机流一致 |
| CUDA 编译、恢复与 EMA 对照 | 1 passed；沙箱外单独执行既有测试 |
| Round 3 调度器回归 | 已通过；测试明确检查 GPU 0–3 上每卡两个 worker，共八个并发 slot |
| `describe-env` 实际 CLI | 成功导出 Round 3 drawer 的环境规格 |
| 原文件摘要核对 | 本次开始时记录的 69 个既有源码、实验规范和 Pixi 文件全部未变化 |
| 新代码静态检查 | 无 `except` 异常捕获处理器；`git diff --check` 通过 |

Round 4 接口测试中的 3 项默认跳过分别为显式启用的 CUDA、MetaWorld 和 ManiSkill 实际测试；默认完整回归另有一项平台 CUDA 编译测试跳过。MetaWorld、GPU 0 数值对照和 CUDA 编译测试均已另外单独执行并通过。常规 API 测试使用模拟 HTTP 响应；真实调用状态单独列在下方。Round 4 数值一致性测试使用临时目录中明确标记为 debug、**未训练、零更新**的真实数值模型 checkpoint；验证动作与随机流的一致性，不代表任务成功或正式模型训练完成。

早期调度器回归失败源于测试替身仍使用 `queue(gpu, stages)`；现已按既有 `gpu_queue(gpu, stages, slot)` 更新测试，并加强八个授权 slot 的并发检查，历史调度器源码未变。早期沙箱内 CUDA 不可用的问题已通过获准的沙箱外实际测试完成验证；未向物理 GPU 4、5 分配工作。

随后完整 GPU／MetaWorld 整合套件的沙箱外请求被中断，未取得执行结果，不计为通过；上表 GPU 结论仅来自已完成的两个独立实测。

## 真实 API 验证

最新一次按用户要求重试已成功：使用最新提供的凭证向 `http://127.0.0.1:8080/responses` 发送真实 POST，保留指定 actor 请求头，请求模型 `gpt-6-astra`、推理档位 `max`、`store=false`。服务返回 HTTP 200，实际模型为 `gpt-6-astra`、返回 `reasoning.effort="max"`、状态为 `completed`，文本为“接口检查通过。”。输入 31 token、输出 82 token（其中推理 72 token），合计 113 token。CLI 退出码为 0，见[原始回执](outputs/interfaces/local-api-8080-006/receipt.json)和[已验证结果](outputs/interfaces/local-api-8080-006/result.json)。凭证仅传入该请求进程，未写入文件。本次只验证 API 调用，没有运行先验设计或机器人控制。

此前的余额不足响应均保留为历史记录：[首次实际请求](outputs/interfaces/local-api-8080-002/receipt.json)、[重试 003](outputs/interfaces/local-api-8080-003/receipt.json)、[重试 004](outputs/interfaces/local-api-8080-004/receipt.json)以及[更换凭证后的请求 005](outputs/interfaces/local-api-8080-005/receipt.json)均返回 HTTP 403 `INSUFFICIENT_BALANCE`。每次重试均由用户明确要求，没有自动重试或切换模型；余额错误不再阻塞最新一次验证。

## 已有问题与未验证项目

- **Round 3 checkpoint 来源**：保留 drawer B0 checkpoint 可在 CPU 反序列化，但该 run 记录的历史 `pixi.lock` 与本次任务开始时的工作区锁文件不同。新适配器保留来源检查并拒绝此不匹配；没有把反序列化成功当作当前来源校验通过。
- **ManiSkill 柜体**：`round4/assets` 当前无资产，未执行真实柜体适配器测试。
- **真实机器人**：硬件接入信息未知，`RobotEnv` 保持抽象，未进行物理部署或运动。

## 复现命令

```bash
pixi run --locked python -m pytest -q
pixi run --locked -e round4 python -m pytest -q tests/interfaces
pixi run --locked env INTERFACES_TEST_SIMULATOR=round3 \
  python -m pytest -q tests/interfaces/test_simulators.py -k round3
```

已通过的物理 GPU 0 CUDA 对照测试可用以下命令复现，执行环境须能访问该设备：

```bash
pixi run --locked -e round4 env CUDA_VISIBLE_DEVICES=0 INTERFACES_TEST_CUDA_DEVICE=cuda:0 \
  python -m pytest -q tests/interfaces/test_policy_integration.py -k cuda
```

接口使用方式、明确的本地 provider 配置与模型绑定契约见 [INTERFACES.md](INTERFACES.md)。
