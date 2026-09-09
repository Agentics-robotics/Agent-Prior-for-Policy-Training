# Peg insertion N5 feedback revision

Choose P4 from P1 to test one remaining representation hypothesis: retain the successful explicit relations while replacing direct absolute object/goal position blocks with hand-relative positions. P3 provides no additional observed E successes over P1 and P2 underperforms in this packet, so P4 retains action-only diffusion. The evidence motivates a bounded test, not a guaranteed improvement; no controller, auxiliary training objective or hyperparameter search is added.

All 804 logged curve rows, 200 development diagnostic records, eight real frames and the curve plot were consumed. All 29 bound hashes match. P1 and P3 both reach C 20/20 and E 10/20, with identical paired descriptive labels. This supports preserving the relational action-only base while leaving extrapolation unresolved. Descriptive failure categories are not causal/contact labels, and no E failure frames are supplied.

P4 changes only how four position blocks are represented in P1 conditioning: current grasp, native previous grasp, goal and observed peg head become hand-relative displacements divided by 0.1 m. Current/previous absolute hands and P1 appended features remain. The same 62D size, identity world actions, U-Net, loss, masks and optimizer settings apply. Absolute object locations remain reconstructible, so neither invariance nor guaranteed shortcut removal is claimed. Formal P4 runs are fresh N5/N20 seed0 at 20,000 total updates, batch128.

Decision was saved before implementation source inspection. Actual designer /root/feedback_peg, codex_session; exact model/session/tokens/cost unavailable. No N2/N10/N20 scores, other tasks, old reports, larger-N demos, expert/reward source, run_manifest or locked-test content was read.
