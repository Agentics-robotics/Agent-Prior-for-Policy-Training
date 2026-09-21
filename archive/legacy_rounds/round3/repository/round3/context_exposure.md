# Round 3 design context and execution provenance

Backend: `codex_session`. Exact model build, session identifier, token counts and API cost are unavailable. No API call or API response identifier is claimed.

The coordinating Codex session previously implemented and read Round 1/2, including their full D20 data summaries, expert source, environment/reward source and results. Drawer and door are previously used development tasks. Infrastructure subagents may inspect pinned simulator and expert source for collection and semantic audits. The overall project is therefore a **Codex-assisted exploratory experiment**, not a claim that the entire design process saw only two demonstrations or an unseen task.

Initial proposal designers will, where available, be separate real Codex subagents with `fork_turns="none"`. Their explicit reading scope is the task's D2 evidence bundle and design contract. They must not inspect expert/reward source, other demonstration trajectories, historical results, concrete dev/test states, or test results. Each proposal records the actual evidence hashes and model/session availability. Implementation by the exposed coordinator remains a limitation even when the initial designer is isolated.

The training set size N always counts the complete trajectories actually used for that policy and its normalization. Design exposure is a separate accounting item. A feedback designer may additionally see the fixed N5 development results, training curves and at most eight predeclared dev frames, and may make one revision. Feedback exposure and extra trainings will be reported separately.

No locked test state or outcome may enter a proposal or implementation decision. Test outcomes will be generated only after all task designs, revision decisions and per-N dev selections are frozen.

User steering overrides the historical AGENTS.md GPU default: Round 3 uses physical GPUs **0 and 2**, at most one training process per GPU and at most two GPUs simultaneously. Other users' processes are preserved.

## 2026-09-08 resumed server session

The new coordinating Codex read the migration handoff, Round3 specification, Round1/2 README/progress/report, shared infrastructure and collection code. Actual initial design is assigned to fresh fork_turns=none subagents with the task D2 bundles. This remains a Codex-assisted exploratory experiment. User now authorized continuation on GPUs **0,1,2,3**, maximum four concurrent formal training processes, one per GPU, superseding the old 0/2 setting above. All project contents moved to the repository root; historical absolute evidence paths remain provenance. Historical runs/ checkpoints were absent in the received repo; no Round3 checkpoint had existed.

Latest user steering expanded Round3 to physical GPUs **0,1,2,3,4,5** (six cards), one formal training process perGPU. This supersedes prior4GPU entries; no model, data, budget or selection change. See audits/gpu_expansion_0_to_5.json.

At 2026-09-08 07:03 UTC, the user withdrew GPUs4/5 and explicitly requested using all of GPUs **0,1,2,3 only**. The supplemental scheduler was terminated and its two workers saved complete recovery states before exiting; no further work may use4/5. This is the latest device authorization, superseding the temporary six-card interval. Paused runs resume without resetting training on0–3; see audits/gpu_reduction_to_0_3.json.

The user then clarified that0–3 were underutilized and explicitly authorized additional concurrent trainers on those cards. Execution now uses two independent worker processes per physical GPU, eight total, with no additional formal configurations, seeds, steps or demonstrations. Per-worker synchronized time overlaps on shared GPUs and is not physical GPU occupancy. See audits/gpu_concurrency_2_per_device.json. This overrides the historical one-worker-per-GPU restriction.
