# Round 3 design context and execution provenance

Backend: `codex_session`. Exact model build, session identifier, token counts and API cost are unavailable. No API call or API response identifier is claimed.

The coordinating Codex session previously implemented and read Round 1/2, including their full D20 data summaries, expert source, environment/reward source and results. Drawer and door are previously used development tasks. Infrastructure subagents may inspect pinned simulator and expert source for collection and semantic audits. The overall project is therefore a **Codex-assisted exploratory experiment**, not a claim that the entire design process saw only two demonstrations or an unseen task.

Initial proposal designers will, where available, be separate real Codex subagents with `fork_turns="none"`. Their explicit reading scope is the task's D2 evidence bundle and design contract. They must not inspect expert/reward source, other demonstration trajectories, historical results, concrete dev/test states, or test results. Each proposal records the actual evidence hashes and model/session availability. Implementation by the exposed coordinator remains a limitation even when the initial designer is isolated.

The training set size N always counts the complete trajectories actually used for that policy and its normalization. Design exposure is a separate accounting item. A feedback designer may additionally see the fixed N5 development results, training curves and at most eight predeclared dev frames, and may make one revision. Feedback exposure and extra trainings will be reported separately.

No locked test state or outcome may enter a proposal or implementation decision. Test outcomes will be generated only after all task designs, revision decisions and per-N dev selections are frozen.

User steering overrides the historical AGENTS.md GPU default: Round 3 uses physical GPUs **0 and 2**, at most one training process per GPU and at most two GPUs simultaneously. Other users' processes are preserved.
