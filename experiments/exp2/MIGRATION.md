# Exp2 migration

## M1 version naming and checkpoint retirement (2026-09-16)

`runs/exp2/M1_v1` is the previous `prior_policies_20260916` run. The original
directory was atomically renamed on the storage disk; the historical path is a
relative symlink to M1_v1. Immutable source, submissions, API journals, training
metrics, traces and reports remain. The user explicitly authorized deleting its
31 training/interface checkpoints (11,075,462,809 bytes). Exact pre-deletion hashes
and the receipt are in `runs/exp2/M1_v2/setup`. Old reports describe historical
training verification; they do not imply the deleted checkpoints are available.

M1_v2 uses `runs/exp2/M1_v2`, `data/exp2/processed/M1_v2` and
`experiments/exp2/configs/m1_v2.json`. M1_v1's published segmentation remains for
provenance. Exp1, M0, raw data and environment locks are unchanged by this migration.

## M0 baseline cleanup (2026-09-16)

The active M0 path is `src/appl/dp_baseline` → `configs/m0.json` →
[`M0_REPORT.md`](M0_REPORT.md): 12 original demonstrations, DDPM100, execute 8.
Historical M0 diagnostic reports, research configurations and a complete pre-cleanup
Exp2 source snapshot are in [`archive/Exp2_M0DP`](../../archive/Exp2_M0DP/README.md).
Diagnostic runtime files were moved within the storage disk to
`/home/storage/oscar/appl_exp2/legacy/Exp2_M0DP`; old paths forward to the same bytes.
The retained model and its development/confirmation episodes remain at their
original physical paths, with a convenient `runs/exp2/m0` entry.

D0's shared controller contract and non-M0 workflows remain in place. The original
rebuild specification's `DP_DIAGNOSIS.md` deliverable is preserved in the archive;
`exp2 report` no longer recreates that retired document in the active directory.
Frozen configs, checkpoints, receipts and reports were not rewritten. Per-file
relocation hashes and validation are linked from [M0_LAYOUT.json](M0_LAYOUT.json).

## Original rebuild migration (2026-09-15)

The active source is `src/appl/`; experiment configuration and reports are here.
The independent locked environment is `environments/exp2/`. The original Exp1
entry remains `pixi run --locked experiment1-{status,run,report}` in the root
environment. Exp1 was not moved or retrained. Its source, lock, data and result
hashes are recorded in `runs/exp2/exp1_preservation_before.json` (85,568 files).

| Old location | New responsibility |
| --- | --- |
| `frameworks/appl` and its nested `round2` | Archived at `archive/frameworks/appl`; root `frameworks/` was removed by the user-requested layout cleanup |
| Original drawer demonstrations and model assets | Real directory `data/exp2`, extracted from the archive with content hashes verified |
| Old round2 runtime/environment | Retained at existing storage locations as archived evidence |
| New code | `src/appl`, with provenance for reused scene, demonstrator, DP backbone and kernel isolation |
| New runtime | `runs/exp2` → `/home/storage/oscar/appl_exp2/runs` |
| New dependencies | `environments/exp2/.pixi` → `/home/storage/oscar/appl_exp2/environment` |

The initial shared home filesystem was full. Exactly 114,058 old Exp2 files and
symlinks were copied, hashed twice against the source, then the redundant source
was removed without following symlinks. The full receipt is
`/home/storage/oscar/appl_exp2/legacy/migration_20260915.json`.
No archived experiment is on the new Python import path. `archive/frameworks/appl`
and `archive/legacy_exp2/source` reference the same historical source tree on the
storage disk; old runners are not active entry points.

## Data extraction and framework archiving (2026-09-15)

At the user's request, `frameworks/` moved to `archive/frameworks/`, and `/archive/`
was added to `.gitignore`. Already tracked `archive/legacy_rounds` files retain
their existing index entries and contents.

`data/exp2` is now a real repository directory, rather than a symlink into the
archive. It contains both original demonstration collections (`demonstrations`
and `demonstrations_v2`), robot assets, `reconstruction_public`, `public`, and the
associated acquisition, collection-freeze and input manifests. Archived design
sessions, old checkpoints and old rollout diagnostics remain in the archive.
Each extracted file was copied and independently SHA-256 checked before its old
copy was replaced with a forwarding link to `data/exp2`. These links preserve
saved storage paths without keeping the active data inside the archived tree.
Historical records retain their original strings; a provenance `current_source`
field maps old source locations to `archive/frameworks/appl`.

The per-file relocation receipt is `runs/exp2/layout_relocation_20260915.json`.
The active configuration continues to use `data/exp2/demonstrations_v2`; splits,
source data, API-authored models/candidates and experimental results are unchanged.
The model loader resolves assets through `data/exp2/assets` and verifies the
original receipt's asset hashes; it no longer uses that receipt's old directory.

Validation: all 32,206 extracted files retain their hashes; the 12 training and
3 validation trajectories load with their original hashes, all 16,021 referenced
frames exist, and 30 sampled PNGs decode. The frozen MuJoCo model compiles and
passes its structural contract even when the receipt's old asset path is made
unavailable in the check. The existing 10 smoke tests pass. Exp1's checked source,
environment locks, protocol, main database and report are unchanged. No API calls,
training updates or rollout steps were run. Details are recorded in
`runs/exp2/layout_validation_20260915.json`.

Unified command: `/home/users/oscar/.pixi/bin/pixi run --manifest-path
environments/exp2/pixi.toml --locked exp2 <command>`. Verified command outcomes
will be recorded in the final report; implementation in progress is not a pass.

Final preservation check: 85,566 persistent Exp1 files retain exact hashes. The original empty SQLite WAL and its SHM sidecar disappeared on normal connection closure; the main database hash is unchanged. See runs/exp2/exp1_preservation_after.json. The original locked Exp1 status command independently reports all 144 slots complete and zero artifact issues.

## 存储与版本管理索引（2026-09-15）

本节补充当前路径和依赖说明，未执行新的迁移。上文的迁移与验证数字属于原操作记录。流程导航见 [Exp2 入口](README.md)，当前进度见 [项目进度](../../PROGRESS.md)。

### 实际目录与符号链接

| 仓库路径 | 实际位置与含义 |
| --- | --- |
| [data/exp2](../../data/exp2/) | 仓库内真实目录，含原始示范、资产、校准公开输入和 `processed/` 派生数据 |
| [runs/exp2](../../runs/exp2/) | 符号链接到 `/home/storage/oscar/appl_exp2/runs`，保存当前 Exp2 运行产物 |
| [environments/exp2/.pixi](../../environments/exp2/.pixi) | 符号链接到 `/home/storage/oscar/appl_exp2/environment`，保存实际依赖安装 |
| [archive/frameworks/appl](../../archive/frameworks/appl) | 符号链接到 `/home/storage/oscar/appl_exp2/legacy/source_20260915` |
| [archive/legacy_exp2/source](../../archive/legacy_exp2/source) | 指向同一个历史源码树，是另一入口，不是第二份副本 |

旧 Exp2 中被迁出的数据路径通过转发链接指向 `data/exp2`；历史 receipt 保留原始路径字符串。当前代码以新目录加载资产并核验原 hash。归档树不在当前 Exp2 的 Python import 路径上。

### 归档中的实际依赖

Exp1 的 [support manifests](../experiment1/manifests/)仍引用 `archive/legacy_rounds/round3/repository/round3/data/<task>/train20/` 中的原始示范；例如 [drawer/support.json](../experiment1/manifests/drawer/support.json)。这些是数据依赖，旧候选、分数和会话不属于当前 API 证据。

因此 `archive/` 不是整体可移除的目录。历史 Round 1/2/3 与旧 Exp2 是不同的归档系列；旧 Exp2 自身的 `round2` 也不等于 `legacy_rounds/round2`。原始路径映射见 [旧轮次归档索引](../../archive/legacy_rounds/README.md)。

### Git 跟踪与本机内容

核对日期：2026-09-15，文档整理前的工作树状态。Git 跟踪情况会随提交变化；实际以 `git status --short`、`git ls-files` 和 [.gitignore](../../.gitignore) 为准。

| 内容 | 本次核对时的状态 |
| --- | --- |
| 根 README、PROGRESS、AGENTS 及 Exp1 主源码等 | 已跟踪文件；工作树中的修改不代表已经提交 |
| `src/appl`、`experiments/exp2`、`environments/exp2`、`tests/exp2` 与 Exp2 重建规范 | 当时尚未跟踪，允许纳入版本管理；本次文档整理不自动暂存或提交这些已有工作 |
| `data/`、`runs/`、`.pixi/`、`outputs/` | 被忽略的本机数据/产物/环境目录；忽略规则不取消其中已有文件的跟踪 |
| `archive/` | 新内容被忽略；此前已经跟踪的 `legacy_rounds` 历史文件继续保持跟踪 |
| Exp1 的数据库、checkpoint、API 原始记录等 | 大量产物被单独忽略；完整性依据实验 manifest、hash 与存储记录 |

Git 提交记录、工作树文件与外部存储共同构成当前工作现场。仅克隆仓库不会获得所有本机数据、环境和运行产物；恢复时需要对应存储及其校验记录。符号链接提供路径入口，不复制或备份目标内容。
# 2026-09-16: current policy workflow and authorized deletions

The active workflow is [PRIOR_POLICIES.md](PRIOR_POLICIES.md). Only
`data/exp2/processed/drawer_exchange_20260916_overlap` remains as a processed
demonstration dataset. The two superseded processed versions and old M1/M2-only
candidate/training/deployment directories were removed at the user's explicit
request. Exact paths, file hashes and small historical receipts are retained in
[the deletion inventory](../../runs/exp2/prior_policies_20260916/cleanup/deletion_inventory.json)
and [completion receipt](../../runs/exp2/prior_policies_20260916/cleanup/deletion_receipt.json).
The old library/formal/deployment modules were removed after dependency checks;
shared numerical, M0, environment, reconstruction and viewer code was retained.
Earlier path descriptions in this file are historical where the deletion inventory
supersedes them. Original data, current segmentation provenance, Exp1 and M0 are
preserved.
# M1 initial tests and scale-up (2026-09-16)

`runs/exp2/M1_initial_test` is the canonical retained initial study; its shared
policy library and `inference_5000` artifacts are unchanged. Earlier 1500/3000
results and original configurations moved to `archive/Exp2_M1_initial_tests`.
Historical paths remain compatibility links so frozen receipts and model requests
remain valid. [M1_INITIAL_LAYOUT.json](M1_INITIAL_LAYOUT.json) records 4,615
byte-preserved runtime files and exact moves. New tasks and their data live under
`runs/exp2/M1_scaleup` and `data/exp2/scaleup`, respectively; no new results are
written into the frozen initial study or M0.
