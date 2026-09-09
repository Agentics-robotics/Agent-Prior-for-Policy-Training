# 第一轮实验：全部完成

截至 2026-09-07，代码、Pixi 环境、数据、正式训练、开发集选择、正式测试、视频与报告均已完成。当前没有运行中的项目训练、评测或资源监测进程。

| Run | 正式 updates | 选定 checkpoint | IID | 位置 OOD |
|---|---:|---:|---:|---:|
| drawer_raw_n20_s0 | 20000 | 10000 | 20/20 | 20/20 |
| drawer_relative_n20_s0 | 20000 | 10000 | 20/20 | 20/20 |
| door_raw_n20_s0 | 20000 | 5000 | 20/20 | 20/20 |
| door_relative_n20_s0 | 20000 | 20000 | 20/20 | 20/20 |

- 17 项 preflight 检查通过；每任务 100 条成功示范、同一份 train20 及独立 split 已冻结。
- 完整重放 320 个初始状态、200 条示范，共 16,444 步；数据与动作标签校验通过。
- 4 次正式训练各完成 20,000 updates；12 个规定 checkpoint、optimizer、EMA 与完整 RNG 恢复状态已保存。
- 240 个 dev episode 已完成，四个 checkpoint 选择冻结后才运行 160 个最终测试。
- 160 个正式测试全部成功；8 个成功视频重放及轨迹 hash 验证通过。没有失败 episode，故 8 个失败视频类型明确记为 absent。
- 配对公平性与交付审计见 `results/paired_integrity.json`、`artifacts/delivery_audit.json`。
- 再次运行 `pixi run round1` 已验证退出码 0：全部训练跳过，开发/测试/视频缓存按完整 hash 验证复用，无新增 optimizer update 或正式测试 episode。日志见 `logs/idempotence_check.log`。

完整结果：[ROUND1_REPORT.md](ROUND1_REPORT.md)；四行结果：[results/summary.csv](results/summary.csv)；阶段计时：[results/compute_cost.json](results/compute_cost.json)。成功率没有观察到表示间差异；Drawer OOD 平均完成步数的次要差异需多训练 seed 复验。

## 历史进程与运行恢复

- PID 804594 / tool session 1364：首个 raw 训练完成后，relative 的速度异常；在第 65 步安全保存后暂停。原始日志、耗时和逐位一致性诊断完整保留。
- PID 822128 / tool session 26467：采用独立 Pixi 子进程顺序恢复，完成剩余训练、全部评测及视频，退出码 0。
- PID 864337 / tool session 2304：完成后的缓存复用检查，退出码 0。
- 资源监测 tool sessions 68535、25291 已结束。
- 冻结模型、数据、配置及训练预算未改。细节见 [docs/RUNTIME_RECOVERY.md](docs/RUNTIME_RECOVERY.md)。

## 可复现与恢复命令

项目目录：`/home/users/oscar/Agent_Training/agent_training`。本机 Pixi：`/home/users/oscar/.pixi/bin/pixi`。

```bash
pixi run round1
pixi run train-round1 --run-id drawer_relative_n20_s0
pixi run evaluate-round1
pixi run report-round1
pixi run python scripts/verify_delivery.py
```

已完成的训练会校验后跳过；未完成训练会恢复 optimizer、EMA、独立 RNG 与采样位置。所有正式数值结果均已冻结。未完成项：无；失败视频缺席是实际全成功结果，阶段机制归因与跨训练 seed 稳定性属于研究限制。
