# 公共 API、环境与策略接口

当前实验是 Experiment 1。`experiment_interfaces` 保留 GPT-6 Responses 客户端、`Env`／`RobotEnv`、`Policy`／`ActionChunk` 以及通用 Python rollout；实验专属训练、工具设计 agent、评估和选模由 `experiment1` 实现。开发规则见 [agent.md](agent.md)，执行规范见 [EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md](EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md)。所有项目命令通过 Pixi，开发分支为 `dev`。

旧 Round 3 checkpoint 绑定器和专属接口源码片段、示例、测试已移入 `archive/legacy_rounds/round3/interface_source/`，用于保留来源记录。未完成 Round 4 的适配器、测试和示例已删除。公共 `MetaWorldEnv` 使用 `experiment1.metaworld.environment.TaskEnv`；不再导入已归档的旧包，也不将旧 checkpoint 声称为 Experiment 1 模型。

## Responses API

本地代理配置为 `examples/interfaces/local_openai_provider.json`，目标 `http://127.0.0.1:8080/responses`，默认模型 `gpt-6-astra`、推理档位 `max`、`store=false`。凭证仅从配置指定的环境变量读取；请求 JSON、代码与回执均不保存令牌。

```bash
read -rs -p '代理令牌: ' LOCAL_OPENAI_BEARER_TOKEN
export LOCAL_OPENAI_BEARER_TOKEN
/home/users/oscar/.pixi/bin/pixi run --locked python -m experiment_interfaces gpt6 \
  --provider examples/interfaces/local_openai_provider.json \
  --request examples/interfaces/gpt6_request.json \
  --output outputs/interfaces/api-check-007
```

输出目录必须尚不存在。HTTP JSON 回应首先写入 `receipt.json`；仅在返回模型、档位、完成状态和内容验证通过后生成 `result.json`。没有自动重试、备用模型或凭证文件读取。此前单次真实 API 验证的[成功回执](outputs/interfaces/local-api-8080-006/receipt.json)是连通性证据，不代表候选设计或正式训练完成。

Experiment 1 的 `RuntimePriorAPI` 使用受控工具循环。API 决定何时读取冻结公共接口与当前 task×N 证据，编辑候选代码、调用固定检查、读取诊断并显式提交；公共代码、训练器、评估器、数据划分与成功判定保持只读。A1–A3 全部提交后外层 runner 才提供开发反馈，随后 API 设计 A4。正式训练与开发评估不属于工具修复预算。完整调用、版本、提交与成本存放在 `experiments/experiment1/`。

## 环境与机器人

`Env` 的公共方法为 `spec`、`reset(seed=..., options=...)`、`step(action)`、`close()`。`EnvSpec` 明确观测 schema、动作上下界与语义、控制频率。渲染是独立 `RenderableEnv.render()` 能力。

`MetaWorldEnv(task)` 转发公共 `TaskEnv`，保留原生观测、4 维世界轴平移／夹爪动作和控制周期。reset 要求 `options={"record": record}`；另传 seed 时必须与 record 相同。适配器不增加坐标变换、动作裁切或控制逻辑。

```bash
/home/users/oscar/.pixi/bin/pixi run --locked python -m experiment_interfaces describe-env \
  --config examples/interfaces/metaworld_env.json \
  --output outputs/interfaces/metaworld-env.json
```

`RobotEnv` 保持抽象。真实接入需要明确控制器、传感器、标定与终止契约，再实现所有抽象方法；当前不宣称已部署或接通真机。

## 策略与执行

通用策略实现 `Policy.spec`、`Policy.reset(inference_seed=..., auxiliary_seed=...)` 和 `Policy.predict(raw_history) -> ActionChunk`。`ActionChunk` 包含原生动作数组、执行步数和诊断。

`experiment_interfaces.rollout.run_episode(env, policy, ...)` 要求双方完整 `EnvSpec` 相同，初始历史为 `[o0, o0]`，每个动作更新历史；依照策略声明的执行步数、原生终止／截断、明确步数上限及显式停止谓词执行。动作必须已经满足原生范围，通用 runner 不另作裁切或猜测成功。

Experiment 1 使用其专属公共训练器与可信评估外层，固定 history 2、horizon 16、execute 4；候选策略在受限进程中仅接收因果观测和独立采样随机流。checkpoint 来源、固定预算、最终 EMA 和提交 hash 的验证属于该实验 runner。旧 `bind-policy`／`rollout` CLI 已撤下；历史模型不能通过新命令绕过归档来源验证。

## 验证命令

```bash
/home/users/oscar/.pixi/bin/pixi run --locked python -m pytest -q tests/interfaces tests/experiment1
/home/users/oscar/.pixi/bin/pixi run --locked env INTERFACES_TEST_SIMULATOR=metaworld \
  python -m pytest -q tests/interfaces/test_simulators.py
```

默认测试使用 CPU 与明确标记的合成 fixture，不请求真实设计 API。按用户最新分配，GPU 仅允许物理卡 4–7。迁移前的 [INTERFACES_VALIDATION.md](INTERFACES_VALIDATION.md) 保留原始验证历史；其旧 Round 4 测试数字不能作为当前 Experiment 1 完成证明。
