# Runtime recovery, 2026-09-07

The first formal `drawer_raw_n20_s0` completed20,000 updates at50.30 updates/s.
The next `drawer_relative_n20_s0` slowed dramatically in the shared process,
reaching65 updates in227.93 seconds. The process was alive and computing,
not an absent checkpoint mistaken for completion. SIGTERM requested a graceful
save; the complete optimizer/EMA/RNG checkpoint at65 was preserved.

Inspection of pinned PyTorch2.7.1 `torch/_inductor/cudagraph_trees.py:2338`
shows that once `MarkStepBox.mark_step_counter` is nonzero, it supersedes the
automatic generation counter. The debug inference uses explicit step marks;
training does not. This is the likely explanation for retained CUDA graph state
between Trainers. External stack attachment was unavailable under host ptrace
permissions; no system permission or driver changes were made, so the precise
stack was not observed.

A fresh-process diagnostic replay of exactly65 updates reproduced the saved
relative model, EMA, complete AdamW optimizer, loader RNG, diffusion RNG,
initial-weight hash, sample count and pairing digest bit-for-bit. It took4.392
seconds including compilation. See `artifacts/runtime_recovery_check.json` and
`logs/runtime_recovery_check.log`. The diagnostic is not a formal initialization
or additional optimizer budget; formal training resumes from the original65
update checkpoint. The successfully completed raw model remains valid.

Only the CLI orchestration was changed: each formal training is launched in a
fresh Pixi child process, sequentially, with the parent holding the pipeline
lock. The frozen training/model/data/config sources and all formal hashes are
unchanged. Every run uses the same precision/compiler/configuration; the
interrupted relative run receives the remaining19,935 updates, for20,000 total.
No formal result was discarded, and no extra valid formal run was added.

Original invocation: PID804594, retained tool session1364, exited1 on deliberate
pause. Full original log is `logs/round1_invocation_804594.log`; timings are
`artifacts/pipeline_timing_804594.json`. The ongoing log appends to
`logs/round1.log`. Resume remains `pixi run round1` or
`pixi run train-round1 --run-id drawer_relative_n20_s0`.

Source: https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/cudagraph_trees.py
