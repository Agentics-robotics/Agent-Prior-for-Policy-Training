# Peg insertion initial design (version 1)

Actual isolated designer: /root/design_peg, codex_session. Exact model/session/token/cost unavailable. Only the contract and peg evidence bundle were opened before this record. Both complete bundled trajectories were summarized with Pixi NumPy, and all 16 real frames were viewed individually. No learned model outcomes were inspected.

P1 adds explicit world-axis relations: grasp versus hand and head versus goal. The supplied geometry fixes insertion to world x while gravity remains z. The supplied source attribution is used as geometry evidence; its implementation was not opened. Fixed metric scales avoid fitting feature magnitudes to two correlated LL/HH layouts. Original fields remain available because robot placement breaks full translation symmetry.

P2 jointly diffuses physical actions with six training-only future geometry channels: head displacement from the current chunk anchor and future grasp-minus-hand offset. Both are computed from the same demonstration, at each next observation, without success/contact labels. Its auxiliary epsilon loss weighs 0.2, while physical action epsilon retains weight 1. Runtime samples all ten channels once and executes only decoded first four. These extra channels neither control nor rank actions.

P3 combines P1 and P2 with one shared U-Net and exactly the same system update budget. This makes the combination a specific independently auditable candidate. It may fail by adding capacity burden or compounding bias, and gains are only predictions.

D2 episodes have 77 and 91 actions. Gripper command first changes positive at steps 32 and 34; observed grasp height first exceeds 0.05 m at 58 and 62. Head-minus-grasp is approximately [-0.13,0,-0.01] m in both, while carried grasp-minus-hand differs slightly. The final head-minus-goal x residual is approximately 0.0644 and 0.0641 m. Therefore no zero-distance success rule, phase threshold, analytic controller, symmetry augmentation, or terminal label is inferred. This is a state-policy proposal, not a visual-generalization claim.

Shared protocol is two observations, predict 16/execute 4, fixed leading DDIM16 eta0 without clean clipping, masked zero-padded uniformly sampled training windows, seed0 fresh per N, and 20,000 total updates/batch128. All action coordinates remain physical world native units, with the world cube clipped only after decode. Proposal saved before any implementation interface inspection; implementation tests may use only bundled D2 and synthetic arrays, never learned outcomes.
