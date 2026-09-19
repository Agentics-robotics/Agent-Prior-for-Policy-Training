# Evacuate red: object-role relational attention

## Identity and assigned hypothesis

Policy `evacuate_red__h01`, skill `evacuate_red`, heuristic index 1, experiment M1_v2.

The assigned heuristic statement is:

> Object-role relational attention: represent pickup/placement with object-to-tool and object-to-target relations while retaining drawer and completed-goal context. Hypothesis: explicit roles reduce spurious absolute-coordinate dependence and improve transition reuse.

The source heuristic and segmentation are unchanged. The design below is an implementation of that hypothesis, not a revision of its evidence or a claim of demonstrated policy performance. The intended comparison is against absolute-state concatenation and target-only cropping. Better sample efficiency under the demonstrated source-position variation is a hypothesis; held-out relocation and erroneous role switching could falsify it.

## Full expanded slice and causal takeover

Every assigned action is retained, with no phase-specific filtering or reweighting. All starts are 195 inclusive. Stops are exclusive: demo1000 845, demo1001 859, demo1002 847, demo1003 845, demo1004 846, demo1005 852, demo1006 846, demo1007 847, demo1008 853, demo1009 846, demo1010 849, demo1011 846. The learned policy covers remaining drawer pull, handle release, empty retreat and red approach, red closure/lift/transport/lowering, red release and empty retreat, then blue approach/closure/initial lift. It does not merely learn the red-in-air core.

Measured incoming overlaps with open-access begin at 195 and extend to stops 474-476, depending on trajectory. Outgoing overlaps with insert-blue begin at 575, 581 or 582 and extend to each assigned stop. These are temporal manifolds of demonstrated transitions, not rectangular applicability boxes. Source indices are documentation only and never model inputs.

Actual observations inspected include:

- demo1000 195: drawer d=0.186297 m, vd=0.107098 m/s, TCP approximately [-0.326315, -0.000053, 0.127929] m, each finger about 0.00706 m. Red is still on the moving drawer at z=0.063 m and blue is on the table at z=0.020 m. The recorded action gripper command is -1. At 240 d is near 0.300 m and each finger is near 0.03939 m, with the command +1. Thus takeover cannot assume red is already grasped or the opening finished.
- demo1000 390: open fingers and TCP z=0.313 m above red. At 450 the fingers are near 0.01824 m and TCP/red z are both near 0.063 m. At 475 red is rising at z=0.14317 m, TCP z=0.14405 m, and TCP is about 9 mm to red's negative world-X side. This offset is represented, not replaced by a rule equating TCP and object goals.
- Source variation is visible in demo1000 red at approximately [-0.197, -0.063] m during acquisition versus demo1001 approximately [-0.195, -0.077] and demo1008 approximately [-0.206, -0.058]. Blue begins near [-0.401, 0.291], [-0.409, 0.311] and [-0.408, 0.315] m respectively. Both source objects remain in the representation throughout.
- demo1000 575, demo1001 581 and demo1008 575: red is still held above the pad at z approximately 0.211 m with closed fingers. These are supported early successor entries, not evidence of completed red placement. At demo1000 595 red z=0.031365 m is still just ABOVE the allowed open z interval; XY alignment alone is insufficient. By 615 red is near z=0.020 m, d is near 0.297 m and fingers are opening/open. Demo1001 615 has fingers approximately 0.03518 m each, whereas demo1000/1008 have approximately 0.03976 m each, illustrating different points in release at the same source index.
- demo1000 680 and 750 show empty retreat and travel toward blue while red remains on the pad. At 805 blue is near the open TCP at table height; at 815 the fingers are near 0.01824 m and the blue/TCP relation has changed during closure. At 844 blue z is 0.12950 m. Adjacent late states demo1001 857/858 show blue rising from 0.11955 to 0.12990 m, and demo1008 851/852 from 0.11985 to 0.13022 m. The supplied post-last-action boundary statistics place blue near z=0.139-0.146 m with total finger width about 0.03648 m, not at a stationary resting endpoint.

Deployment has two causal raw observations of 47 scalars. It does not receive gripper command history, future poses, demonstration ID, source index or elapsed skill time. Recorded commands above are evidence only. Finger qpos/qvel, joint motion, TCP and block pose differences, relative rotations, target geometry and d/vd let the network distinguish ongoing pull, open empty travel, possible block closure, coupled transport, release and blue lift. Two-sample differences are not labeled contact and are not converted into velocities using an assumed sample period. Stationary dwell ambiguity remains.

## Implemented representation and learned architecture

All geometry is computed from supplied state, in metres in the world frame. Robot joint coordinates retain the shared original normalization. The drawer center is [0.19-d, 0, 0.035] m. The red pad is the supplied red_goal, fixed at [-0.18, -0.30, 0.02] m in these data, not translated with the drawer. The supplied blue_goal is retained separately from the drawer cavity center.

Each of the two observations yields five typed feature vectors:

1. **Robot, 32 features:** shared-normalized nine qpos and nine qvel; world TCP position; TCP rotation matrix; shared-normalized d/vd. The two finger positions are retained individually.
2. **Drawer, 21 features:** shared-normalized d/vd; absolute drawer center; center-to-TCP displacement; absolute supplied blue goal; blue-goal-to-TCP displacement; both objects relative to the drawer center; signed opening margin d-0.26.
3. **Red and blue, 43 features each:** absolute object position and rotation matrix; object-minus-TCP displacement in world and TCP axes; relative rotation R_tcp^T R_object; goal-minus-object displacement; object-minus-drawer-center displacement; absolute supplied target; rotated cube axis-aligned extents; XY containment slack; signed lower/upper center-z margins. Red and blue share an object projection but have different learned role embeddings. For blue, the goal displacement uses blue_goal while containment slack uses the cavity center. Red containment uses the fixed pad.
4. **Pad, 15 features:** absolute supplied pad position and red, TCP, blue and drawer displacements from it. This fixed anchor remains present during blue acquisition.

Quaternion wxyz values are normalized and converted to full 3x3 rotation matrices. No raw quaternion branch bypasses this conversion. Therefore equivalent q and -q give the same encoded features and forward output, up to numerical roundoff. This is quaternion-sign invariance only; neither general SE(3) invariance nor equivariant absolute joint commands is claimed.

For every typed feature vector, [previous, current, current-minus-previous] is concatenated and projected by a 192-hidden-unit SiLU MLP to a 128-dimensional token with LayerNorm. Learned role embeddings are added. Two pre-LayerNorm self-attention blocks each have four heads and a 256-wide SiLU feedforward network. A contextual robot-token query attends softly over all five contextual tokens using learned query/key/value maps. Finger configuration and motion, TCP/object relations and completed-goal context thus condition the focus through learned attention, rather than a hard grasp label or an elapsed-time phase selector.

The flattened ordered five contextual tokens AND the attention-weighted focus are projected through a 384-dimensional MLP and LayerNorm. Retaining all roles outside the pooling bottleneck prevents a forced discard of the already placed red or the open drawer when attention turns to blue. Attention weights are neither supervised phase labels nor calibrated grasp probabilities.

This 384-dimensional condition drives the supplied conditional 1-D U-Net, with widths [128,256,512], kernel size 5, groups 8 and diffusion timestep embedding 128. Forward predicts epsilon for an entire 16-by-8 noisy action chunk. The eight actions are the original shared-normalized seven absolute Panda joint targets and one gripper command. All actions result from learned DDPM denoising; no action residual controller, waypoint list, replay, IK solver or geometric action correction is used.

## Shared scales, not per-skill fitting

Buffers are initialized from spec.normalizer, the one normalizer fit on complete original demonstrations (12 fit IDs, 12,809 samples). Its normalizer SHA-256 is `e75d5e1c8d9862f1611bf577ea5fa6e6d3fd5a25a69627c1f55602c602a7193f`. No statistics are fit by this model. In limits mode std is the shared half range, not an empirical standard deviation.

For derived world positions, extents, displacements, containment slack and object displacement targets, the axis scales are the elementwise maximum of the shared TCP, red and blue position half ranges: approximately [0.159006, 0.308779, 0.212680] m. Absolute world anchors are divided by those scales without recentering, deliberately preserving base-relative geometry and physical height. Tool-axis displacement uses the maximum of these three scales (approximately 0.308779 m), avoiding an inconsistent rotation of axis-specific scales. Raw qpos/qvel and d/vd use their original shared means and half ranges. The opening margin uses the original d half range. Rotations are dimensionless. None of these operations refits narrow slice scales or clips observations. Native action normalization and DDPM decoding/bounds remain the framework's unchanged shared mapping.

## Losses and gradient paths

Let epsilon_hat be the forward prediction. The diffusion loss is the supplied action-mask-weighted epsilon MSE over valid slots and eight channels. This updates the U-Net, condition projection, both attention mechanisms, role embeddings and feature projections.

The auxiliary predictor takes the contextual red or blue token, the shared attention focus and a learned 32-dimensional **relative prediction-slot** embedding. Its MLP is (288 -> 192 -> 128 -> 3), with SiLU hidden activations. It predicts world XYZ displacement for each object and each of 16 slots, in the shared workspace scales. It receives no clean future action and no future state as input. Its slot embedding specifies prediction horizon, not demonstration phase; the epsilon forward does not use it.

At current state t, action slot j represents action t-1+j and future_obs[j] is the state after that action, at t+j. The target is therefore `(p_future[j] - p_current) / space_scale`, including a nominal zero displacement at slot 0. Predictions for both objects use the product of future_mask and action mask. If m is this mask, the unweighted auxiliary loss is `sum(m * squared_prediction_error) / max(6 * sum(m), 1)`, with six XYZ channels across the two blocks. No padded or cross-slice future labels are used. The reported prior_loss is 0.5 times this quantity, and total loss is diffusion_loss + prior_loss.

The auxiliary loss has genuine gradients through the displacement head, slot embeddings, contextual object tokens, focus query/key/value maps, self-attention blocks and input projections. It regularizes a representation shared with the diffusion model; it does not directly impose physics on the denoiser's decoded action. It is not a penalty on observed geometry alone. No high-noise x0 reconstruction is used, so this term has no inverse-alpha amplification. Future observations occur solely on the supervised target side of compute_loss and are absent from deployment forward.

Adaptations needed for this executable state-only setting are: use supplied poses rather than images; express the relational prior through trainable token attention rather than a symbolic controller; retain absolute anchors for Panda reachability; supervise multi-slot future object displacement as a causal representation task rather than claim analytical action-to-pose dynamics. Observed containment/opening margins are extra completion-context features, not constraints or a task-success loss. The auxiliary head predicts average possible displacement from history and can be ambiguous near stationary closure/release; it is not an exact dynamics model.

## Training and inference recipe

Use the unchanged fixed recipe: seed 0, 20,000 updates, batch 128, history 2, horizon 16, execution prefix 8, 100 DDPM train/inference steps, epsilon prediction, clip_sample true, AdamW learning rate 1e-4 and weight decay 1e-6, cosine schedule with 500 warmup steps, gradient norm bound 1, EMA decay 0.999, and last EMA at the declared budget. The framework owns noising, optimization, action normalization/decoding, execution and accounting. Causal features and attention are recomputed on each updated observation history while replanning. The encoder is pure and stateless; there is no inferred phase cache to reset at handoff. The auxiliary head is trained and checkpointed but is not used to generate actions at deployment.

## Invocation, continuation and useful exit

See HANDOFF.json for the machine-readable interface. Supported entry can be late pull, post-handle release/empty retreat, red closure or red rising; do not require a held-red precondition. If red is being transported or lowered, continue to finish lowering unless the successor is explicitly taking over that supported overlap. Red XY alignment while z is high is not placement. Once red is contained, retain its context and learn release/retreat before blue acquisition rather than immediately treating a geometric subgoal as an instruction to switch attention. If invoked longer, this model's suffix establishes a blue grasp and starts lifting while red remains on the pad and d stays around 0.297 m. Prefer transfer before extrapolating beyond the demonstrated early lift. The successor must receive the entire causal state, including red and drawer, not merely blue pose.

The outgoing overlap also permits earlier transfer during held-red descent at z about 0.211 m. Insert-blue's own interface must then finish red placement/release. Late transfer should preserve the inferred blue grasp; closed fingers alone are not proof, but an approximately stable TCP-to-blue offset and jointly rising heights provide stronger observational evidence. No index-driven or threshold-driven transition is built into the network. The external inference API selects policy and invocation duration.

## Task completion versus handoff guidance

The task's completion contract is simultaneous drawer-open, red-on-pad and blue-inside. Drawer d must be strictly greater than 0.26 m. Full XY containment uses rotated cube extents (cube half-size 0.02 m): red within the pad half-width [0.06,0.06] m, with red center z strictly in (0.014,0.031) m; blue within the drawer cavity half-width [0.172,0.182] m about [0.19-d,0,0.035], with blue center z strictly in (0.053,0.074) m. A single observation suffices. Release, object speed, tool clearance and sustained stability are NOT extra success requirements. They may inform safer handoff selection but do not redefine completion. This slice retains release and blue acquisition as useful overlap; final blue insertion is delegated because it is outside this policy's assigned retained suffix.

## Dependencies and limits

Implementation imports only torch, math and appl.public. No visual encoder, external kinematics, collision checker, contact sensor or simulator is assumed. Only the same Panda/gravity/workspace and known object/goal poses are supported by the supplied data. Modest reachable target shifts are plausible but untested; targets are fixed in the demonstrations, and absolute anchors mean global translation invariance is explicitly not assumed. Attention can still learn spurious joint or world correlations; the prior encourages relationships, it does not guarantee their use. Fixed color roles, two blocks and a single drawer do not establish arbitrary object counts, role exchange, broad obstacle avoidance or dropped-object recovery. The learned loss cannot guarantee grasp preservation, no collisions, drawer openness or containment. Measured boundary support and two-update interface checks are not rollout success evidence.
