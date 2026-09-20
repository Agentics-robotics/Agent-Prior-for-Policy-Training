"""Update navigation and report links after the recorded artifact migration."""
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
from experiments.exp2.maintenance.cleanup_20260920 import ROOT, ARCHIVE, RECEIPTS, EXP, CURRENT, read, write


def rel(path, base=EXP/'reports'):
    return os.path.relpath(path,base)


def main():
    assert read(RECEIPTS/'validation.json')['passed']
    manifest=read(CURRENT/'manifest.json')
    reports=EXP/'reports'
    prior=ARCHIVE/'reports/current_before_navigation'
    shutil.copytree(reports,prior)
    replacements={
        'episodes':rel(CURRENT/'episodes'),
        'policies':rel(CURRENT/'models'),
        'execution_history':rel(ARCHIVE/'runs/Exp2_new'),
        'segmentations':rel(CURRENT/'inputs/appl'),
    }
    def replace_link(m):
        target=m.group(1)
        for old,new in replacements.items():
            if target==old or target.startswith(old+'/'):
                return ']('+new+target[len(old):]+')'
        return m.group(0)
    for p in reports.glob('*.md'):
        text=re.sub(r'\]\(([^)]+)\)',replace_link,p.read_text())
        text=text.replace('The package includes all sixty numerical trajectories',
                          'The repository retains all sixty numerical trajectories')
        p.write_text(text)
    write(reports/'results.json',manifest['episodes'])
    write(reports/'models.json',manifest['models'])
    # Main table is now exactly the retained 225 OOD outcomes. Preserve the
    # earlier 255-row table (including partial ID) in the historical report copy.
    for filename in ['episodes.csv','models.csv','checkpoint_inventory.csv','training_trajectories.csv']:
        p=reports/'tables'/filename
        rows=list(csv.DictReader(p.open()));fields=list(rows[0])
        if filename=='episodes.csv':
            rows=[r for r in rows if r['condition']=='OOD']
            for r in rows:
                selected=next(x for x in manifest['episodes'] if (x['task'],x['seed'],x['method'])==(r['task'],int(r['seed']),r['method']))
                for key in ['source_root','bundle_root','root']:
                    if key in r: r[key]=selected['root']
                if 'video' in r:r['video']=selected['video']
        else:
            for row in rows:
                for key,value in row.items():
                    for model in manifest['models']:
                        old=model['original_folder'];new=model['folder']
                        if value==old or value.startswith(old+'/'):
                            row[key]=new+value[len(old):];break
        with p.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    methods=['naive_DP','SinglePrior_6_xhigh','APPL_6_xhigh']
    html=['<!doctype html><meta charset="utf-8"><title>Exp2_new: 225 OOD replays</title>',
          '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #ddd}video{width:240px}</style>',
          '<h1>Exp2_new — 75 matched OOD layouts, three methods</h1>',
          '<p>Corrected binary gripper. Replays are accelerated; physical trace steps are authoritative. Historical and partial ID outcomes are archived.</p>']
    for task in manifest['tasks']:
        html += ['<h2>'+task['label']+'</h2>','<table><tr><th>Seed</th>'+''.join('<th>'+m+'</th>' for m in methods)+'</tr>']
        for seed in task['selected_OOD_seeds']:
            html.append('<tr><td>'+str(seed)+'</td>')
            for method in methods:
                row=next(r for r in manifest['episodes'] if (r['task'],r['seed'],r['method'])==(task['task'],seed,method))
                folder=rel(ROOT/row['root'])
                html.append(f'<td>{"Success" if row["success"] else "Failure"}, {row["steps"]} steps<br><video controls preload="none" src="{folder}/replay.mp4"></video><br><a href="{folder}/result.json">result</a></td>')
            html.append('</tr>')
        html.append('</table>')
    (reports/'VIDEOS.html').write_text('\n'.join(html)+'\n')
    (reports/'README.md').write_text('''# Exp2_new experiment reports

Reviewed 2026-09-20. The current comparison is **225 complete position-OOD outcomes**: five tasks, fifteen matched layouts per task, three methods. These reports are retained scientific records, not a writing/export package.

- [Full report](REPORT.md)
- [Failure analysis](FAILURE_ANALYSIS.md)
- [Observed API-selected recoveries](RECOVERY_CASES.md)
- [API-authored prior catalog](PRIOR_CATALOG.md)
- [225 replay videos](VIDEOS.html)
- [225-row current result table](tables/episodes.csv), [task summary](tables/ood_summary.csv)
- [Canonical artifact manifest](../../../runs/exp2/current/manifest.json)
- [Historical reports and partial-ID evidence](../../../archive/Exp2_history_20260920/README.md)

The full report discusses the ten previously completed APPL ID episodes only as supplementary history. They are excluded from the current OOD manifest and primary table. Original narrative text and earlier 255-row tables are preserved in the archive; scientific measurements are unchanged. Report links and current artifact indexes were updated after migration.
''')
    (EXP/'README.md').write_text('''# Exp2：当前唯一实验入口

**核对日期：2026-09-20。已完成并停止，无待执行实验。**

当前只展示 **Exp2_new：五任务 × 每任务 15 个位置 OOD 初态 × 三种方法，共 225 个完整结果**。三种方法使用相同初态和修正后的二值夹爪执行器，最多 5,000 个物理步。历史错误夹爪结果、其他轮次、额外 ID/OOD 结果和诊断均归档；原始记录保留。

| 方法 | 模型数量 | 部署时调用 API | 当前 OOD 成功 |
| --- | ---: | --- | ---: |
| Naive DP | 5，每任务一个 | 否 | 5/75 |
| Agent Prior DP / SinglePrior | 5，每任务一个；GPT-6 Astra/xhigh 编写 prior 与策略 | 否 | 10/75 |
| APPL / GPT-6 Astra/xhigh | 36 个子策略 | 是，依据 prior、状态和交接信息选择 | 30/75 |

## 先看这些

- [完整实验报告](reports/REPORT.md)、[失败分析](reports/FAILURE_ANALYSIS.md)、[恢复案例](reports/RECOVERY_CASES.md)
- [每个 API policy 的 prior 与源码](reports/PRIOR_CATALOG.md)
- [225 个视频](reports/VIDEOS.html)、[结果表](reports/tables/episodes.csv)
- [当前资产清单](../../runs/exp2/current/manifest.json)：唯一确定这 225 个回合和 46 个模型的清单
- [历史归档入口](../../archive/Exp2_history_20260920/README.md)、[迁移核验](../../archive/Exp2_history_20260920/migration/validation.json)

## 当前目录与依赖

```text
data/exp2/                          原始示范、环境资产、任务输入
  demonstrations_v2/              抽屉示范；冻结清单还保留原验证数据
  scaleup/<task>/demonstrations/   其余四任务的示范
  scaleup_astra_xhigh/             GPT-6 的切分数据

src/appl/                          冻结的数值/环境/工具框架
  scaleup/                        五任务定义、示范获取与任务协议
  envs/                           仿真环境、状态与动作接口、几何判定
  dp_baseline/                    完整任务 DP 的训练/推理
  demonstrations/                 提供给 API 的示范阅读与切分工具
  prior_policies/                 独立 prior policy 训练与部署工具
  vendor/                         Diffusion Policy 网络实现

experiments/exp2/                   三条方法的编排与实验说明
  current.py                      只读状态/完整性检查入口
  single_policy/                  API 设计完整任务 prior DP、训练接口
  astra/                          GPT-6 技能切分、prior 设计与训练编排
  binary_gripper/                 两个 baseline 的修正夹爪评估器
  exp2_new/                       APPL 修正夹爪部署、预算和审计
  analysis/                       当前结果核验及实际执行过的续跑代码
  configs/                        冻结配置；历史配置链接不代表新运行计划
  reports/                        完整报告、统计表、图、视频索引
  maintenance/                    本次整理的迁移与校验记录工具

runs/exp2/current/                 当前科学资产（实际位于实验存储盘）
  manifest.json                   225 回合、46 模型及原路径/哈希
  episodes/<method>/<task>/OOD/    15 回合：结果、轨迹、API 日志、视频
  models/<method>/<task>/<policy>/ 源码、prior、提交、训练和 checkpoint
  inputs/                         任务定义、API 切分、共享尺度及检查证据

archive/Exp2_history_20260920/      旧运行、旧诊断代码、旧数据和全部旧报告
```

训练流程：环境与示范 → 固定 DP / API 单策略设计 / API 技能切分和多 prior 设计 → 数值训练 → 保存 checkpoint。评估流程：同一组 75 个 OOD 初态 → 两个直接部署 baseline / API 选择 APPL 子策略 → 同一几何目标判定 → 结果与视频。

环境、任务、示范采集、工具、训练器和判定器由开发者实现；切分、heuristic/prior、API policy 源码与交接文档、APPL 部署决策由 Runtime API 产生。此次整理没有改动这些内容。

## 只读命令

在仓库根目录执行；不会训练、仿真或发送 API 请求。

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m experiments.exp2.current
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m experiments.exp2.current --verify
```

`--verify` 读取 46 个 checkpoint 并检查哈希，因此需要较多磁盘读取。接口测试：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q tests/exp2
```

## 冻结实现与兼容路径

旧 `M0`、`M1_*`、`Exp2_new`、`binary_gripper_v1_20260919` 等运行入口现在是指向归档或当前资产的兼容链接，不是并列的活跃实验。冻结配置、原始 API 日志仍能按当时路径解析。目录名中的历史版本不改变原实验的真实设置。

`src/appl`、三个当前评估/设计模块及环境锁被原实验的完整性校验覆盖，保持原字节；其中少量历史框架接口因此仍在。一次性诊断/恢复脚本已移入归档；当前代码不导入归档代码。请从本页和 `current.py` 进入，勿因历史脚本存在而重启旧实验。

[部署协议](EXP2_NEW.md)、[二值夹爪修正](BINARY_GRIPPER.md)、[单策略设计协议](SINGLE_POLICY.md)、[GPT-6 切分与训练来源](ASTRA_XHIGH.md)保留为执行时的原始规范；其中的历史任务数量和预算不替代当前 manifest。所有未来 Exp2/新实验使用 `xhigh`，Exp1 相关实验使用 `max`。Exp1 的代码、数据、模型与报告完全保留。
''')
    original=(ARCHIVE/'navigation_before/README.md').read_text()
    start=original.index('## 实验导航');end=original.index('## 环境与常用命令')
    exp1row=next(line for line in original.splitlines() if line.startswith('| Exp1 |'))
    nav='''## 实验导航

**Exp2 当前只保留一条主线：Exp2_new，五任务 × 15 个位置 OOD × 三种方法，共 225 个完整结果。** 修正夹爪后，DP 5/75、Agent Prior DP 10/75、APPL GPT-6 Astra/xhigh 30/75。实验已停止；旧错误夹爪结果及其他历史轮次归档，实验报告保留，写作 ZIP 和重复打包导出已删除。

| 工作线 | 源码 | 配置与说明 | 输入数据 | 运行产物与结果 |
| --- | --- | --- | --- | --- |
'''+exp1row+'''
| Exp2_new | [冻结框架](src/appl/)、[方法编排](experiments/exp2/) | [唯一入口与目录图](experiments/exp2/README.md) | [原始示范与 API 切分](data/exp2/) | [225 回合、46 个模型](runs/exp2/current/)、[完整报告](experiments/exp2/reports/REPORT.md)、[视频](experiments/exp2/reports/VIDEOS.html) |

历史 Exp2 见 [归档索引](archive/Exp2_history_20260920/README.md)。旧路径保留兼容链接以核查冻结配置和日志；不会混入当前结果。Exp1 未改动。

'''
    text=original[:start]+nav+original[end:]
    text=text.replace('experiments/exp2/README.md#命令','experiments/exp2/README.md#只读命令')
    (ROOT/'README.md').write_text(text)
    original_progress=(ARCHIVE/'navigation_before/PROGRESS.md').read_text()
    exp1_history=original_progress[original_progress.index('## 历史记录：Exp1 执行过程'):]
    (ROOT/'PROGRESS.md').write_text('''# 项目进度

## 当前摘要

**核对日期：2026-09-20。Exp2 目录整理完成，实验仍已停止。** 当前唯一主线为 **Exp2_new：五任务 × 每任务 15 个位置 OOD × DP / Agent Prior DP / APPL GPT-6 Astra/xhigh，共 225 个完整结果**。成功分别为 **5/75、10/75、30/75**；无新增 API、训练或仿真。

- [当前入口与目录结构](experiments/exp2/README.md)、[完整报告](experiments/exp2/reports/REPORT.md)、[失败分析](experiments/exp2/reports/FAILURE_ANALYSIS.md)、[恢复案例](experiments/exp2/reports/RECOVERY_CASES.md)、[225 个视频](experiments/exp2/reports/VIDEOS.html)。
- [唯一资产清单](runs/exp2/current/manifest.json)：225 个 OOD 回合、46 个冻结模型及哈希。三个方法使用同初态与修正后的二值夹爪执行器；没有重新训练、选择模型或筛除失败。
- 旧错误夹爪实验、其余轮次、部分 ID 和诊断移入 [归档](archive/Exp2_history_20260920/README.md)。历史报告/API 输出/轨迹/模型保留；旧路径通过兼容链接可解析。Exp1 原有资产与冻结科学框架保持不变，见[迁移核验](archive/Exp2_history_20260920/migration/validation.json)。
- 已删除 Exp2 写作 ZIP、重复打包目录和打包脚本；其中独有报告、表格、图与分析已保留。详见[删除清单](archive/Exp2_history_20260920/migration/deletion_inventory.json)。
- GPT-6 Astra/xhigh 的示范切分、prior、源码、部署决策仍为 API 产物；开发者提供环境、示范、工具、训练与判定。此整理没有修改实验逻辑。未来 Exp2/其他新实验使用 `xhigh`；Exp1 相关使用 `max`。

完整 Exp2 执行沿革见[整理前进度记录](archive/Exp2_history_20260920/navigation_before/PROGRESS.md)；各阶段原报告均保留，不在当前摘要重复历史状态。

'''+exp1_history)
    oldagents=(ARCHIVE/'navigation_before/experiments/exp2/AGENTS.md').read_text()
    (EXP/'AGENTS.md').write_text('''# Exp2 current scope (reviewed 2026-09-20)

The user's latest instruction is a repository cleanup, now recorded in
`archive/Exp2_history_20260920/migration`. Exp1 must remain untouched.

The ONLY active comparison is `runs/exp2/current/manifest.json`: five tasks,
15 matched position-OOD initial states each, and three methods (naive DP,
SinglePrior_6_xhigh, APPL_6_xhigh), 225 complete outcomes. Scores are 5/75,
10/75, and 30/75. The 46 frozen models (5 + 5 + 36) are reused without retraining.
Read README.md and reports/REPORT.md. `python -m experiments.exp2.current`
is the read-only entry; `--verify` validates artifact identities and paired resets.

All evaluation is completed and stopped. Do not restart historical supervisors,
run remaining ID trials, repeat failures, or send new API requests merely because
old execution instructions remain in a protocol, log, or archived script.

History (wrong-gripper evaluations, GPT-5.5, initial M1, other seeds, partial ID,
interrupted prefixes, costs, diagnostics and reports) is archived. Historical
paths are compatibility links; do not count them as separate active studies.
Current episodes/models are physically under runs/exp2/current. Original requests,
API-authored code, priors, submissions, source snapshots, data, checkpoints and
reports retain their scientific contents. Do not rewrite frozen files to rename paths.
Current code must not import archived runners. The frozen src/appl framework and
environment locks are intentionally unchanged, including legacy interfaces that
are covered by source_manifest().

Writing ZIPs, duplicated export trees and packaging scripts were explicitly
authorized for deletion. Their unique scientific reports/tables/figures were
retained first. Do not recreate writing exports unless requested again.

Always use /home/users/oscar/.pixi/bin/pixi run --manifest-path
environments/exp2/pixi.toml --locked for Python/project commands. Work on dev.
For every NEW Exp2/other experiment use reasoning effort xhigh; Exp1-related
work uses max. Check actual requests, never relabel historical records. Latest
resource authorization permits GPUs 0-7, at most five simultaneously; inspect
occupancy and preserve unrelated processes before any future authorized run.

Task/environment design, demonstration acquisition, fixed baseline, prompts,
interfaces, training, evaluation, scheduling and reporting are developer-owned.
Segmentation, heuristic/prior content, API policy implementation and semantic
handoff documents, and APPL online policy/duration/stop/finish choices are
Runtime API-owned. No ad hoc API candidate edits, silent fallback, automatic
transport retries or concealed performance tuning. Preserve uncertainty, costs,
all interrupted-call evidence and the agreed geometric success definitions.

The executed protocols are EXP2_NEW.md, BINARY_GRIPPER.md, SINGLE_POLICY.md and
ASTRA_XHIGH.md; their historical counts/launch language do not authorize new runs.
The complete previous instructions and authorizations are preserved at
[historical AGENTS.md](../../archive/Exp2_history_20260920/navigation_before/experiments/exp2/AGENTS.md).
Follow root AGENTS.md and agent.md; the latest user scope takes precedence.
''')
    (ARCHIVE/'README.md').write_text('''# Exp2 history archive — 2026-09-20

The active study is [Exp2_new](../../experiments/exp2/README.md): 225 complete OOD outcomes and 46 frozen models. This archive is historical evidence, not an execution queue. It includes both obsolete wrong-gripper experiments and other valid historical/supplementary results; archival does not imply that all archived data had the gripper bug.

| Directory | Contents |
| --- | --- |
| [runs](runs/) | Original M0/M1, GPT-5.5/APPL, GPT-6 old deployment, SinglePrior, extra corrected baseline seeds, partial ID, interrupted prefixes, budgets, original reports and raw journals |
| [experiments](experiments/) | Historical protocols and one-off diagnostic/recovery scripts, including failure reviews |
| [data](data/) | Earlier demonstrations/segmentation and GPT-5.5 processed skill data; preserved unchanged |
| [reports/previous_four_method](reports/previous_four_method/) | Earlier full four-method scientific report and its tables/figures/evidence |
| [reports/before_recovery](reports/before_recovery/) | Report revision recovered from the old ZIP before deleting it |
| [reports/current_before_navigation](reports/current_before_navigation/) | Exact current scientific report assets before link/index cleanup; includes the 255-row table with supplementary ID |
| [navigation_before](navigation_before/) | Previous README, PROGRESS and Exp2 agent instructions |
| [migration](migration/) | Exact move/delete inventory, original hashes, per-move journal and validation receipts |

The runtime archive physically lives at `/home/storage/oscar/appl_exp2/legacy/Exp2_history_20260920/runs`; the link above exposes it here. Selected episode/model leaves were moved to `runs/exp2/current`, and their old locations link back to them. Original top-level runtime names link into this archive so frozen absolute paths still resolve. Shared/Exp1 archives are untouched.

Scientific reports, original API output, checkpoint contents, trajectories and data are preserved. Only explicitly inventoried writing ZIPs, duplicate export directories and packaging utilities are deleted. Current reports are at [experiments/exp2/reports](../../experiments/exp2/reports/README.md).

Some older prose references the now-deleted ZIPs or original packaging layout. Those historical texts remain byte-preserved; use the active report and current manifest for working navigation. Historical scripts are retained as executed and are not supported run entry points.
''')
    (EXP/'analysis/README.md').write_text('''# Current Exp2 execution audits

These unchanged modules support the completed corrected-gripper study and its
recorded budget/resource continuations. They are provenance, not a queue to run.
Use `python -m experiments.exp2.current` to inspect current results without API
calls or simulator execution. Other diagnostics/recovery helpers are archived at
`archive/Exp2_history_20260920/experiments/analysis`.
''')
    write(RECEIPTS/'navigation.json',dict(updated=True,primary_rows=225,
        original_report_assets=str(prior),scientific_policy_content_changed=False))
    print('Current navigation, reports, tables and 225-video index updated.')


if __name__=='__main__':main()
