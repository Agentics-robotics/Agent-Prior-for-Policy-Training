# API-authored prior catalog

Each summary below is copied unchanged from the API-authored pipeline metadata. It describes the intended inductive bias, not a proven mechanism or observed success. Source and prior links point to byte-identical bundled submissions.

## Single prior 6/xhigh

### Drawer exchange

#### SinglePrior_6_xhigh

Object/drawer-relative anticipation: shared object encoders and a causally predicted short-horizon relational future condition one learned action diffusion U-Net. A masked future-relation regression loss supervises the same anticipation features used by the denoiser.

Parameters: 19,038,112; formal updates: 60,000.

[API prior document](policy_cards/SinglePrior_6_xhigh/drawer_exchange/SinglePrior_6_xhigh/PRIOR.md) · [Exact source](policy_cards/SinglePrior_6_xhigh/drawer_exchange/SinglePrior_6_xhigh/policy.py) · [Pipeline](policy_cards/SinglePrior_6_xhigh/drawer_exchange/SinglePrior_6_xhigh/pipeline.json)

### Two-block sorting

#### SinglePrior_6_xhigh

A goal-relative predictive object-factorization prior: a shared encoder represents each block's two-frame relationship to the TCP and its own pad, and supervised causal relational forecasts condition one absolute-joint epsilon diffusion U-Net alongside the complete normalized observations.

Parameters: 19,379,044; formal updates: 60,000.

[API prior document](policy_cards/SinglePrior_6_xhigh/two_block_sort/SinglePrior_6_xhigh/PRIOR.md) · [Exact source](policy_cards/SinglePrior_6_xhigh/two_block_sort/SinglePrior_6_xhigh/policy.py) · [Pipeline](policy_cards/SinglePrior_6_xhigh/two_block_sort/SinglePrior_6_xhigh/pipeline.json)

### Buffer exchange

#### SinglePrior_6_xhigh

Occupancy-aware object relational conditioning with supervised causal interaction forecasts. A shared two-object encoder represents gripper contact geometry, each object's assigned and opposite goals, and the other object's occupancy of its goal. Learned short-horizon TCP, object and finger forecasts condition one epsilon-prediction diffusion U-Net alongside the full normalized observation history.

Parameters: 19,504,329; formal updates: 60,000.

[API prior document](policy_cards/SinglePrior_6_xhigh/buffer_swap/SinglePrior_6_xhigh/PRIOR.md) · [Exact source](policy_cards/SinglePrior_6_xhigh/buffer_swap/SinglePrior_6_xhigh/policy.py) · [Pipeline](policy_cards/SinglePrior_6_xhigh/buffer_swap/SinglePrior_6_xhigh/pipeline.json)

### Unstack and sort

#### SinglePrior_6_xhigh

Object-centric predictive relation conditioning: a shared two-object encoder and shared short-horizon relation forecaster expose object-to-TCP, object-to-goal, and inter-object structure to one learned action diffusion U-Net. Forecasts are supervised by demonstration futures during training and computed only from the two causal observations at deployment.

Parameters: 19,487,544; formal updates: 60,000.

[API prior document](policy_cards/SinglePrior_6_xhigh/unstack_sort/SinglePrior_6_xhigh/PRIOR.md) · [Exact source](policy_cards/SinglePrior_6_xhigh/unstack_sort/SinglePrior_6_xhigh/policy.py) · [Pipeline](policy_cards/SinglePrior_6_xhigh/unstack_sort/SinglePrior_6_xhigh/pipeline.json)

### Tray packing

#### SinglePrior_6_xhigh

A shared temporal tool-object-goal relation encoder conditions one action diffusion U-Net. A training-only, shared short-horizon relation-displacement forecast regularizes this representation without imposing an object order or a controller.

Parameters: 20,020,712; formal updates: 60,000.

[API prior document](policy_cards/SinglePrior_6_xhigh/tray_pack/SinglePrior_6_xhigh/PRIOR.md) · [Exact source](policy_cards/SinglePrior_6_xhigh/tray_pack/SinglePrior_6_xhigh/policy.py) · [Pipeline](policy_cards/SinglePrior_6_xhigh/tray_pack/SinglePrior_6_xhigh/pipeline.json)

## APPL 5.5/high

### Drawer exchange

#### side_grasp_red_pull_drawer_open__h01

A learned DDPM action policy conditioned on red-relative side-grasp geometry, drawer progress, finger aperture, and TCP/object motion; auxiliary trainable heads predict manipulation phase and short-horizon TCP/red/drawer consequences to bias the shared encoder toward the assigned side-grasp pull heuristic.

Parameters: 18,596,317; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h01/pipeline.json)

#### side_grasp_red_pull_drawer_open__h02

Learned action diffusion policy conditioned by a soft contact-phase latent prior for approach, close, pull, release, and retreat. The phase belief is inferred from the two causal state observations and trained with heuristic phase labels, drawer-progress regression, and gripper-intention prediction while the U-Net still learns the full normalized joint/gripper action diffusion distribution.

Parameters: 17,908,628; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h02/pipeline.json)

#### side_grasp_red_pull_drawer_open__h03

Drawer-progress coordinate prior: a learned action DDPM is conditioned on causal drawer_position, drawer_velocity, pull-line object/TCP relations, and shared-normalized robot state. Trainable auxiliary effect heads predict future drawer opening, open-threshold crossing, drawer velocity, and red-block lateral drift from the denoised action trajectory, biasing the learned sampler toward monotonic world-x pulling until the drawer is safely open and then toward release/retreat in the overlap phase.

Parameters: 18,346,736; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/side_grasp_red_pull_drawer_open__h03/pipeline.json)

#### red_top_regrasp_lift__h01

Learn a DDPM action policy conditioned on a red-centered top-grasp affordance frame: TCP-minus-red position, block-relative motion, gripper aperture, relative quaternions, drawer opening and red-goal geometry are encoded causally, with an auxiliary learned future relative-state loss for lift/attachment consistency.

Parameters: 18,392,976; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h01/pipeline.json)

#### red_top_regrasp_lift__h02

Learned state-conditioned DDPM action policy for the full expanded regrasp-and-lift slice, with an auxiliary learned future-geometry head that biases denoised actions toward the demonstrated open-drawer clearance envelope, top regrasp, finger closure, and red lift.

Parameters: 18,832,422; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h02/pipeline.json)

#### red_top_regrasp_lift__h03

Attachment-switch prior: a learned causal encoder infers free, contact, and carry probabilities from joint, TCP, red block, and drawer state history, then conditions a DDPM action denoiser for the expanded red top-regrasp and lift slice.

Parameters: 19,081,520; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/red_top_regrasp_lift__h03/pipeline.json)

#### red_transport_place_pad__h01

Learned action diffusion policy conditioned by pad-frame red placement features: red-to-pad error, TCP-to-red offset, drawer-open state, gripper state, and blue-transition cues. An auxiliary learned head predicts future red placement and release labels from the denoised action plan to bias the diffusion model toward placing red on the fixed outside pad.

Parameters: 18,519,128; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h01/pipeline.json)

#### red_transport_place_pad__h02

A learned DDPM action policy whose conditioning is bottlenecked by a trainable four-stage high-carry, descend, release, and retreat representation for transporting the red block to the outside pad.

Parameters: 18,687,468; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h02/pipeline.json)

#### red_transport_place_pad__h03

Carry-release object dynamics prior: the learned diffusion policy conditions on a causal estimate of red-block attachment and TCP-to-red offset, and is trained with an auxiliary action-conditioned rollout that treats the red block as moving with the arm while the gripper is closed and becoming stationary after opening on the pad.

Parameters: 18,744,087; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/red_transport_place_pad__h03/pipeline.json)

#### blue_approach_grasp_lift__h01

A learned action DDPM for the full expanded blue approach, grasp and lift slice. The prior is implemented as blue-relative conditioning around the observed block pose, TCP pose, gripper opening, red-on-pad cue and drawer-open cue, plus a differentiable auxiliary future-readiness loss for contact and lift predictions from the denoised action estimate.

Parameters: 18,465,295; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h01/pipeline.json)

#### blue_approach_grasp_lift__h02

Pad-to-blue transit waypoint prior implemented as a learned recurrent causal encoder that predicts soft retreat, high-transit, blue-descent, and grasp-lift region latents plus short-horizon TCP waypoint and z-clearance summaries. These trainable predictions condition an epsilon-predicting DDPM action U-Net.

Parameters: 18,582,929; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h02/pipeline.json)

#### blue_approach_grasp_lift__h03

Learn a causal blue-attachment latent from qpos/qvel, TCP pose, blue pose and relative motion, inject it into the diffusion U-Net condition, and supervise it from future lift/contact evidence during training.

Parameters: 18,263,817; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_approach_grasp_lift__h03/pipeline.json)

#### blue_transport_to_drawer__h01

Drawer-frame blue insertion prior: the diffusion model conditions on the causal observation history augmented with learned features for blue pose, TCP pose, gripper state, and goal errors expressed relative to the open drawer cavity center [0.19 - drawer_position, 0, 0.035]. A trainable auxiliary predictor uses the denoised action estimate to predict future drawer-frame blue placement and inside-cavity status.

Parameters: 18,593,356; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h01/pipeline.json)

#### blue_transport_to_drawer__h02

Rigid blue-carry prior implemented as a learned action diffusion policy: a conditional U-Net predicts normalized joint/gripper action noise, while an auxiliary differentiable dynamics head learns to predict future TCP and blue-block positions from the denoised action estimate. The dynamics head contains a gated rigid branch that keeps the current TCP-to-blue offset during closed-finger carry and a free branch for pre-grasp and release portions of the expanded slice.

Parameters: 18,541,967; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h02/pipeline.json)

#### blue_transport_to_drawer__h03

A learned action diffusion policy conditioned on a causal latent stage representation for high-arc blue-block transport: lift to clearance, translate laterally while high, then descend and begin release in the open drawer cavity.

Parameters: 18,637,227; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_transport_to_drawer__h03/pipeline.json)

#### blue_release_retreat_complete__h01

Completion-predicate local insertion prior: a learned state-conditioned DDPM action policy uses drawer-frame blue error, red-pad/drawer-open/blue-inside predicate features, and auxiliary future-completion supervision to center/release the blue block in the open drawer and retreat.

Parameters: 18,403,985; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h01/pipeline.json)

#### blue_release_retreat_complete__h02

Support-before-retreat prior implemented as a learned epsilon-predicting action diffusion model. A causal observation encoder predicts trainable support, opening, and retreat-readiness latents; these latents condition the diffusion U-Net, and auxiliary future-prediction losses train them from the demonstrated release and withdrawal outcomes.

Parameters: 18,456,300; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h02/pipeline.json)

#### blue_release_retreat_complete__h03

Near-goal learned residual stabilizer for centering the blue block in the open drawer, releasing it, and retreating. A learned nominal action trajectory, conditioned on causal drawer-frame state features, is used as the local stabilizing prior for a DDPM action diffusion model.

Parameters: 18,808,872; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/drawer_exchange/blue_release_retreat_complete__h03/pipeline.json)

### Two-block sorting

#### acquire_active_block__h01

A learned DDPM action policy conditions its denoiser on active-object-relative features: TCP-to-selected-block, selected-block-to-goal, gripper state, qpos/qvel, and inactive-block context. The same encoder is used for red-first acquisition and blue-after-red acquisition; the active block is selected causally from the observed red-at-goal state rather than by a separate color-specific policy.

Parameters: 17,840,944; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h01/pipeline.json)

#### acquire_active_block__h02

A learned phase-conditioned DDPM action policy for acquisition. A causal two-observation encoder predicts an ordered posterior over approach, descent, closure, lift and early-carry modes; the diffusion denoiser is conditioned on that posterior and trained with masked phase and monotonic-progress auxiliary losses.

Parameters: 18,490,949; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h02/pipeline.json)

#### acquire_active_block__h03

Attachment-confirmation prior implemented as a learned action diffusion policy: the denoiser is conditioned on causal TCP/object/finger relative-motion features, a learned active-block selector, and learned attachment probabilities; auxiliary losses supervise attachment from future demonstration evidence and train an action-conditioned future relative-offset predictor.

Parameters: 18,726,380; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/two_block_sort/acquire_active_block__h03/pipeline.json)

#### deliver_active_block__h01

Goal-relative active-block delivery implemented as a learned DDPM action policy. Red and blue candidate blocks are encoded with a shared object-goal-relative module, a learned active-object gate aggregates the selected delivery candidate, and a Conditional U-Net predicts normalized action noise.

Parameters: 18,659,022; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h01/pipeline.json)

#### deliver_active_block__h02

A learned action diffusion policy conditioned by a trainable latent vertical stage classifier for lift/clearance, high transport, goal descent, low release, and open retreat. The model uses world-frame block/goal/TCP geometry from the two causal observations and learns DDPM epsilon for normalized joint and gripper action sequences.

Parameters: 19,407,288; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h02/pipeline.json)

#### deliver_active_block__h03

Predicate-and-readiness prior implemented as a learned action diffusion policy. The denoiser is conditioned on causal forecasts of red/blue goal predicates and release/readiness predicates, and training adds differentiable auxiliary losses for those forecasts plus a terminal-phase clean-action consistency term.

Parameters: 18,432,660; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/two_block_sort/deliver_active_block__h03/pipeline.json)

### Buffer exchange

#### red_to_buffer_handoff__h01

A learned action diffusion policy with an object-relative contact-funnel conditioning path: two causal state observations are encoded in the red-block and blue-block frames, a learned soft phase embedding represents approach, pinch/lift, carry, lower/release, and blue-handoff phases, and auxiliary prediction of short-horizon red/blue displacements trains the same representation.

Parameters: 18,500,965; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h01/pipeline.json)

#### red_to_buffer_handoff__h02

Swap-topology buffer prior implemented as a learned DDPM action policy: causal state history is encoded with explicit red-in-blue-goal, blue-in-red-goal and red-buffered geometric features; trainable heads predict the temporary red buffer placement and red-buffered predicate; those predictions condition the action diffusion backbone.

Parameters: 17,910,572; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h02/pipeline.json)

#### red_to_buffer_handoff__h03

A learned action diffusion policy for the full red-to-buffer slice that adds an auxiliary overlap-readiness head. The readiness head is trained from causal geometric state cues so the red-release to blue-approach overlap can be treated as a continuum instead of a fixed switch index.

Parameters: 18,511,945; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/red_to_buffer_handoff__h03/pipeline.json)

#### blue_to_goal_handoff__h01

Active-object reflection prior for the blue transfer: a learned action diffusion model conditions on causal state history through shared relative encoders for the blue-active view and a y-reflected red/blue-swapped view, with a small task-space auxiliary loss for predictive and reflection-consistent latent features.

Parameters: 18,471,032; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h01/pipeline.json)

#### blue_to_goal_handoff__h02

Goal-anchored learned diffusion policy for placing the blue block: the causal condition explicitly encodes blue_pose-blue_goal and tcp_pose-blue_goal in world metres, and auxiliary trainable heads predict future blue-goal error, future minimum distance, and the blue_at_goal predicate from the DDPM clean-action estimate.

Parameters: 18,434,253; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h02/pipeline.json)

#### blue_to_goal_handoff__h03

A learned recurrent contact-mode diffusion policy for the full expanded blue-to-goal handoff slice. A GRU encodes the two causal state observations plus physical relative features, predicts a six-way latent mode distribution, and conditions a DDPM action denoiser with small mode-gated residual experts.

Parameters: 18,699,390; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/blue_to_goal_handoff__h03/pipeline.json)

#### red_buffer_to_goal_finish__h01

Active-red final-transfer diffusion policy. The learned denoiser predicts normalized absolute Panda joint/gripper actions while conditioning on causal red-relative, red-goal-relative, blue-goal-error and finger/TCP phase features. Trainable auxiliary heads predict future red lift and red-goal error from the shared condition and from denoised action estimates.

Parameters: 18,740,107; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h01/pipeline.json)

#### red_buffer_to_goal_finish__h02

A learned epsilon-predicting diffusion policy for the full expanded final red placement slice, augmented with explicit trainable red_at_goal and blue_at_goal predicate heads and weak post-success open/settle action losses.

Parameters: 18,416,716; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h02/pipeline.json)

#### red_buffer_to_goal_finish__h03

Waypoint-chain transport prior implemented as a learned action diffusion model: a causal state encoder predicts progress through lift, high carry, descend, release and retreat plus relative task-space waypoints, and a Conditional U-Net denoises joint/gripper action chunks conditioned on that bottleneck.

Parameters: 18,467,101; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/buffer_swap/red_buffer_to_goal_finish__h03/pipeline.json)

### Unstack and sort

#### red_unstack_lift__h01

A learned DDPM action policy conditioned on a causal red-object-relative representation: TCP, blue support cube, and red/blue goals are encoded relative to the current red cube so the unstack lift is not learned only as an absolute joint path.

Parameters: 18,642,269; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h01/pipeline.json)

#### red_unstack_lift__h02

A learned action DDPM conditioned on a causal recurrent encoder with an ordered three-mode latent event state: approach_open, close_contact, and lift_carry. The latent mode gates the denoiser through trainable embeddings and is supervised by demonstration-derived phase and transition losses.

Parameters: 18,464,854; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h02/pipeline.json)

#### red_unstack_lift__h03

Support-preservation learned diffusion policy: condition the action denoiser on causal red, blue, TCP, gripper and goal relations, and train an auxiliary action-to-future decoder that penalizes predicted blue xy disturbance and lateral red motion before sufficient red-blue clearance.

Parameters: 18,494,098; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/red_unstack_lift__h03/pipeline.json)

#### red_place_to_blue_entry__h01

A learned DDPM action policy whose causal condition is organized around red_pose-to-red_goal error and tcp_pose-to-red_pose grasp offset, with auxiliary learned future heads for red placement phase and release/blue-entry readiness.

Parameters: 18,470,898; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h01/pipeline.json)

#### red_place_to_blue_entry__h02

Release-reset prior implemented as a learned epsilon-predicting diffusion policy with a causal four-state phase latent for carry, descent/contact, open-release, and retreat/switch-to-blue. The phase latent and auxiliary gripper/red-future heads bias the model to release the red block at the red pad, leave it stable, retreat with open fingers, and then enter the blue acquisition/lift transition.

Parameters: 18,663,372; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h02/pipeline.json)

#### red_place_to_blue_entry__h03

Sequential object-attention diffusion policy. The denoiser is conditioned by learned red and blue object-centric embeddings and a learned two-way attention switch that is trained to move from red to blue after red is observed at its goal with an open gripper, and during blue approach/lift.

Parameters: 18,471,309; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/red_place_to_blue_entry__h03/pipeline.json)

#### blue_acquire_transport__h01

Identity-conditioned top-grasp reuse for the blue target: a learned DDPM action policy encodes blue as the active object, red as an already placed context object, and uses object-relative TCP, goal, lift, and finger cues to learn approach, closure, lift, and high carry toward the blue goal over the full expanded slice.

Parameters: 18,362,200; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h01/pipeline.json)

#### blue_acquire_transport__h02

A height-staged learned diffusion policy for the full blue acquire-and-transport slice. It conditions the denoiser on learned phase, waypoint, gripper, and short-horizon blue-motion predictions so the action diffusion model represents red-to-blue transition, pregrasp descent, grasp/lift, high lateral carry, and goal-near handoff as connected phases rather than a single unstructured motion.

Parameters: 18,497,499; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h02/pipeline.json)

#### blue_acquire_transport__h03

Completed-object invariance prior for the blue acquisition and transport slice: after red has been placed, the diffusion denoiser is conditioned on red-goal, red-blue and red-TCP relations and is trained with an action-conditioned auxiliary predictor that discourages red displacement and low paths near the red pad while still learning the demonstrated blue pickup and carry actions.

Parameters: 19,204,776; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/blue_acquire_transport__h03/pipeline.json)

#### blue_place_release_retreat__h01

Blue goal-basin servo diffusion prior: a learned conditional DDPM uses shared-normalized state history plus world-frame blue-goal, TCP-blue, red-goal and gripper features, with auxiliary learned success and terminal-error heads trained from demonstration futures to bias precise blue placement, release and retreat.

Parameters: 18,554,893; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h01/pipeline.json)

#### blue_place_release_retreat__h02

Supported-release diffusion policy: a learned causal support/event classifier conditions the action denoiser and auxiliary losses discourage open-gripper denoised actions until the blue block is aligned with its goal and near table height.

Parameters: 18,365,069; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h02/pipeline.json)

#### blue_place_release_retreat__h03

Post-release verification and retreat prior implemented as a learned state-conditioned action diffusion policy with auxiliary terminal-success, object-disturbance-risk, release-phase, and open-gripper losses.

Parameters: 18,607,566; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/unstack_sort/blue_place_release_retreat__h03/pipeline.json)

### Tray packing

#### acquire_block__h01

Learned diffusion policy with an active-object-relative representation: red-active and blue-active candidate features are encoded by a shared network, a causal role module selects/mixes the active block, and a conditional U-Net predicts normalized joint/gripper action noise.

Parameters: 18,499,505; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h01/pipeline.json)

#### acquire_block__h02

Learned contact-phase bottleneck diffusion policy for block acquisition. A causal observation encoder predicts approach, align, close, lift and carry phase posteriors, uses them to condition a DDPM action denoiser, and is trained with phase and grasp-state auxiliary losses in addition to epsilon prediction.

Parameters: 19,605,446; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h02/pipeline.json)

#### acquire_block__h03

Clearance via-point acquisition implemented as a learned state-conditioned DDPM: a trainable waypoint head predicts hover, grasp, lift-clearance and goal-approach TCP subgoals from causal world-frame object, goal, gripper and TCP state, and the diffusion U-Net conditions on those predictions while denoising joint/gripper action sequences.

Parameters: 18,328,528; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/tray_pack/acquire_block__h03/pipeline.json)

#### place_block__h01

Active-goal placement funnel: the learned diffusion policy conditions on world-frame block-to-goal residuals, TCP-to-object offsets, finger state, and a causal soft red/blue role estimate so one controller can learn the carry-in, descent, release, and retreat pattern for both tray targets.

Parameters: 18,561,935; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/tray_pack/place_block__h01/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/tray_pack/place_block__h01/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/tray_pack/place_block__h01/pipeline.json)

#### place_block__h02

Support-gated release and retreat implemented as a learned epsilon-predicting diffusion policy. A causal state encoder predicts a five-phase posterior (carry_to_funnel, descend, supported_settle, open_release, retreat), support probability, and active-object future height, and the denoising U-Net is conditioned on these trainable predictions. Auxiliary losses train the phase/support/height heads and a differentiable gripper gate on the predicted clean action so that opening is learned after support rather than scripted.

Parameters: 18,406,430; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/tray_pack/place_block__h02/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/tray_pack/place_block__h02/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/tray_pack/place_block__h02/pipeline.json)

#### place_block__h03

Tray packing noninterference prior: a learned action diffusion policy conditions on both block poses, both goal boxes and analytic tray geometry, while auxiliary learned object-future heads penalize predicted inactive-object motion, active-object goal/containment error and low-path keep-out violations.

Parameters: 18,626,283; formal updates: 20,000.

[API prior document](policy_cards/APPL_5_5_high/tray_pack/place_block__h03/PRIOR.md) · [Exact source](policy_cards/APPL_5_5_high/tray_pack/place_block__h03/policy.py) · [Pipeline](policy_cards/APPL_5_5_high/tray_pack/place_block__h03/pipeline.json)

## APPL 6/xhigh

### Drawer exchange

#### open_access__h01

Prismatic relational conditioning for a learned action DDPM: typed robot, drawer, red and blue nodes exchange messages using moving-drawer coordinates, with separate world-X axial and YZ transverse channels on drawer edges. Robot joints and world positions preserve reachability and gravity information. A masked drawer-displacement prediction auxiliary trains the shared causal encoder.

Parameters: 19,489,208; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h01/pipeline.json)

#### open_access__h02

Contact-event memory rather than an episode clock. A two-observation GRU feeds a learned eight-phase, duration-augmented ordered filter. Its soft posterior and censored dwell statistic gate phase-specific conditioning adapters of a shared learned action diffusion denoiser. Soft contact-motion phase supervision, ordered future-phase prediction, anti-chatter and transition-consistency losses train the latent representation over the entire expanded slice.

Parameters: 19,131,216; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h02/pipeline.json)

#### open_access__h03

Successor-aware short-horizon diffusion selection. A full-state diffusion proposal has three learned paired epsilon/clean-action refinements. A separately supervised three-member recurrent dynamics ensemble predicts eight-action consequences and opening, release/retreat and red-lift readiness. Within-denoising soft selection prefers demonstration-like predicted consequences with less disagreement, opening loss and stage-inappropriate coupling. A differentiable training consequence loss also teaches the epsilon proposal through frozen dynamics. This is an explicit adaptation of completed-chunk reranking to the fixed epsilon-only deployment interface, not an external planner.

Parameters: 18,041,340; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/open_access__h03/pipeline.json)

#### evacuate_red__h01

Object-role relational attention conditions a learned joint-action DDPM on causal robot, drawer, red, blue and fixed-pad tokens. Explicit tool/object, object/target and object/drawer relations coexist with world anchors and sign-invariant rotation matrices. Learned robot-query attention changes focus without discarding background goals. A masked future-object displacement predictor trains the same attended representation.

Parameters: 20,082,059; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h01/pipeline.json)

#### evacuate_red__h02

Clearance waypoint hierarchy: a causal full-state encoder predicts event and manipulated-entity probabilities and conditions a learned spatial waypoint diffusion denoiser. A separate conditional action DDPM consumes the resulting robot/object intent and current state. Local future waypoints, event changes, capped duration and ongoing motion are supervised; teacher and predicted intents are mixed during action training.

Parameters: 19,812,790; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h02/pipeline.json)

#### evacuate_red__h03

Attachment-consistent action diffusion: a causal learned none/handle/red/blue posterior and per-block continuous tool-relative transform refinement condition an absolute-joint DDPM. A learned action-conditioned rollout predicts TCP, block, drawer and finger states. Masked predicted block-relative translation and rotation changes are penalized only on confident co-motion windows; the handle receives an X-only prismatic coupling penalty. Release observations and commands deactivate the training constraint, not the denoiser.

Parameters: 18,602,102; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/evacuate_red__h03/pipeline.json)

#### insert_blue__h01

Articulated containment margins condition a learned action diffusion transformer. World-frame wxyz poses yield rotated block extents, full XY containment margins, strict Z-interval margins and the drawer-opening margin. Both objects remain represented throughout red placement, blue acquisition, insertion and optional terminal retreat. A differentiable denoiser-latent head predicts masked demonstration future margins at four horizons; it does not impose premature goal satisfaction.

Parameters: 4,505,681; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h01/pipeline.json)

#### insert_blue__h02

Contact-belief adaptive diffusion: a causal two-frame GRU infers an eight-mode soft contact belief and a continuous relative-position latent. One learned epsilon diffusion decoder is conditioned on that belief and predicted contact changes. A learned temporal residual blends current and forecast contact embeddings with entropy/event-dependent retention. Future co-motion supplies weak training labels, and an action-conditioned learned motion predictor couples an auxiliary loss to denoised actions. The fixed interface does not permit changing the executed prefix; adaptive commitment is implemented inside the denoiser, not as an adaptive execution scheduler.

Parameters: 18,107,916; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h02/pipeline.json)

#### insert_blue__h03

Concurrent-goal preserving lookahead, implemented as an amortized selection objective for a learned action diffusion policy. Three bootstrapped action-conditioned transition models score five local alternatives to training-time DDPM clean estimates using exact world-pose containment margins, learned short-horizon remaining-work forecasts, retention of observed red/drawer/blue goals, ensemble disagreement and a learned behavior-support penalty. Detached selections supervise the denoiser. Deployment uses the fixed DDPM sampler, not an unsupported external controller or runtime K-chunk reranker.

Parameters: 20,186,659; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/drawer_exchange/insert_blue__h03/pipeline.json)

### Two-block sorting

#### acquire_transport__h01

Role-relative object-goal binding with a shared two-object graph encoder, learned requested/predecessor role probabilities, and a FiLM-conditioned learned action-chunk epsilon diffusion model. Both object nodes remain available during predecessor disengagement. Because the fixed forward interface has no caller role-ID fields, a causal learned role resolver is supervised for the demonstrated red-then-blue order; there is no scripted action controller.

Parameters: 20,116,875; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h01/pipeline.json)

#### acquire_transport__h02

Contact belief rather than command state. A learned two-observation recurrent contact filter, soft mode transitions, and a persistent probabilistic gripper decoder condition a joint eight-channel action diffusion model. Masked finger-response and TCP-relative object-transform forecasting provide additional trainable contact/co-motion supervision. The complete expanded acquisition slices include predecessor red descent, opening, retreat, blue approach, closure, lift and initial transport.

Parameters: 18,742,319; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h02/pipeline.json)

#### acquire_transport__h03

Height-separated waypoint hierarchy: an eight-step auxiliary pose/duration diffusion with learned phase-conditioned progress produces four local role-relative TCP/requested-object knots. Smooth bounded interpolation conditions the standard learned actuator-action DDPM. Soft prediction-dependent geometry losses favor separated approach, empty retreat/travel, and loaded lift regimes while exempting predecessor lowering.

Parameters: 19,629,388; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/two_block_sort/acquire_transport__h03/pipeline.json)

#### deliver_disengage__h01

Grasp-frame transport with explicit detachment, implemented as an action-indexed geometric reference denoiser conditioning a learned joint-action DDPM. Separate red and blue future object poses, a free TCP reference, causal attachment probabilities and a learned two-observation grasp-transform filter provide offset-compensated TCP references. Masked prediction losses teach rigidity only across inferred held intervals, not during empty retreat.

Parameters: 19,859,748; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h01/pipeline.json)

#### deliver_disengage__h02

Contract-aware learned action diffusion with a causal placement-window predictor, separate disengagement-readiness predictions, and a recurrent action-conditioned pose/aperture forecast. Analytic rotated-corner containment and centre-height losses train the forecast and, through a detached-parameter forecast, regularize denoised action chunks. Already released contained objects receive a preservation loss. Outcome guidance is amortized during training; this implementation does not change the fixed DDPM sampler or perform online rollout-energy optimization.

Parameters: 18,752,545; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h02/pipeline.json)

#### deliver_disengage__h03

Event progress with elastic duration: a two-observation recurrent belief predicts ordered manipulation events, within-event progress, positive nominal and remaining durations, and uncertainty. A learned progress-knot action reference is time-decoded, and a progress-resampled temporal diffusion denoiser is pulled back to physical action timestamps. A separate persistent gripper head conditions a direct-time gripper diffusion score.

Parameters: 20,075,093; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/two_block_sort/deliver_disengage__h03/pipeline.json)

### Buffer exchange

#### buffer_red__h01

Occupancy-conditioned staging graph with learned role probabilities and a probabilistic staging-site prediction conditioning an absolute-joint action diffusion model. A full five-node causal scene graph distinguishes red, blue, their different goal regions, and TCP. A blue-plus-goals layout subgraph proposes a free red buffer; the robot/world branch retains absolute Panda configuration. Training combines masked epsilon MSE, locally available supported-buffer likelihood, five-role cross-entropy, and soft predicted-site footprint/table penalties. All deployment conditioning uses predictions, never future labels or a scripted controller.

Parameters: 19,402,517; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h01/pipeline.json)

#### buffer_red__h02

Learned two-frame causal GRU contact belief conditions an action DDPM. Finger aperture, TCP-object relative rotations/translations and backward co-motion support six soft manipulation modes plus independent red/blue support predictions. Learned transitions encourage persistence without a scripted controller. Event-weighted gripper denoising and trainable short-horizon action-conditioned attachment/co-motion forecasts supervise the full red-buffer-to-blue-acquisition slice.

Parameters: 18,201,960; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h02/pipeline.json)

#### buffer_red__h03

Hierarchical learned diffusion: an internally denoised, ordered four-waypoint TCP route with orientation, gripper and within-window timing conditions the fixed joint-action epsilon diffuser. A supervised action-conditioned TCP trajectory predictor couples task geometry to decoded motor trajectories, with weak low-height lateral-sweep regularization.

Parameters: 19,691,731; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/buffer_red__h03/pipeline.json)

#### transfer_blue__h01

Learned attachment-conditioned action diffusion with a causal two-object TCP-relative transform estimator, none/red/blue attachment probabilities, and an action-conditioned recurrent future-state predictor. Masked future-state supervision and attachment-gated relative-pose consistency train both the representation and, through reconstructed clean actions, the denoiser. The full expanded slice includes initial red descent/release, blue acquisition/carry/setdown/release, and buffered-red regrasp/lift.

Parameters: 19,473,719; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h01/pipeline.json)

#### transfer_blue__h02

Role-conditioned noninterference in a learned action diffusion model: a TCP/red/blue interaction graph estimates passive versus active roles, and an action-conditioned ensemble provides masked preservation and conservative point-to-cube clearance training losses. The fixed sampler has no candidate-ranking hook; ranking is not implemented and preservation is amortized into the epsilon predictor during training.

Parameters: 19,219,835; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h02/pipeline.json)

#### transfer_blue__h03

A learned finite-context event posterior and state-dependent residual-duration hazards condition an epsilon-predicting joint-action diffusion model. Soft stay/advance belief propagation encourages event persistence without a trajectory clock. Training-only command/geometry annotations, right-censored future event targets and a soft ordered-transition loss align the full red-release to blue-transfer to red-reacquisition sequence.

Parameters: 18,468,920; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/transfer_blue__h03/pipeline.json)

#### finish_red__h01

Rotated-footprint contract-aware action diffusion. Exact observed margins condition a learned epsilon U-Net. An action-conditioned future-pose/support surrogate supplies masked supervision and low-noise differentiable containment and blue-preservation guidance to denoised action estimates during training. Online actions remain learned DDPM samples, not a controller.

Parameters: 18,608,842; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h01/pipeline.json)

#### finish_red__h02

Soft, observation-correctable progress-memory diffusion policy. A learned state initializer and two-observation GRU infer six contact/progress modes; their belief, predicted goal geometry, and observation disagreement condition a learned action DDPM. Action-conditioned recurrent progress forecasts, supervised only by training future labels, teach ordering and persistence. This is a bounded-memory adaptation, not a persistent hidden-state controller.

Parameters: 18,336,918; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h02/pipeline.json)

#### finish_red__h03

Grasp-frame-conditioned joint-action diffusion. A learned two-observation blue/empty/red attachment encoder gates measured object-to-TCP transforms and grasp-compensated goal TCP errors. A state-conditioned learned joint-noise decoder and masked action-conditioned future-pose losses learn local action/motion coupling without IK. Every action is sampled by the supplied joint-action DDPM.

Parameters: 18,620,517; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/buffer_swap/finish_red__h03/pipeline.json)

### Unstack and sort

#### acquire_lift__h01

Role-relative geometry with support context, implemented as a learned two-object message-passing encoder conditioning an action DDPM. Shared object and edge networks combine TCP/object/goal differences, table and inter-object height context, sign-invariant rotation matrices, and causal motion with a global robot branch. Three learned latent attention slots replace unavailable invocation tags; both objects also remain in an ungated pooled path.

Parameters: 18,883,662; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h01/pipeline.json)

#### acquire_lift__h02

A causal recurrent contact-belief prior conditions a learned absolute-joint action DDPM. Six soft object-indexed modes distinguish empty approach/retreat, red closing, red held, red releasing, blue closing and blue held. Future finger positions, coupled object/TCP displacement and relative orientation changes provide training-only outcome supervision; no measured contact is asserted.

Parameters: 18,697,570; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h02/pipeline.json)

#### acquire_lift__h03

Clear support before lateral loading. A history-conditioned action diffusion U-Net is trained with an action-conditioned response ensemble, frozen-response matching, and a soft world-z clearance energy. Small late-denoising energy-gradient steps are accepted only against an unguided candidate with bounded action changes and low ensemble disagreement. Explicit causal selected/previous-object role features separate red pickup, prior-red completion, and blue pickup.

Parameters: 18,853,648; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/unstack_sort/acquire_lift__h03/pipeline.json)

#### carry_place_clear__h01

Transport the object rather than the TCP: a learned attachment-confidence gate exposes causal TCP-to-object rigid transforms to a shared object-token encoder, while both object-to-pad errors remain visible after placement. A learned action-conditioned future-pose predictor supplies masked held-object transform losses to the epsilon diffusion model.

Parameters: 18,976,841; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h01/pipeline.json)

#### carry_place_clear__h02

History-informed, phase-conditioned action diffusion with separate arm and gripper epsilon heads. A binary clean-gripper auxiliary, weak ordered event supervision, masked persistence losses and an action-conditioned measured-response predictor distinguish persistent grip events from smooth absolute joint setpoints. The fixed Gaussian DDPM is retained; categorical gripper diffusion is adapted to a Bernoulli auxiliary and binary-support consistency, not represented as an implemented categorical sampler.

Parameters: 19,939,226; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/pipeline.json)

#### carry_place_clear__h03

Contract-aware learned action diffusion with a three-member action-conditioned outcome ensemble, causal horizon-progress conditioning, phase-gated rotated-corner containment shaping and preservation of already achieved goals. Goal energy has a zero-cost feasible region rather than a pad-center target. Opening, retreat and the red-to-blue transition retain ordinary demonstration supervision.

Parameters: 18,812,877; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/unstack_sort/carry_place_clear__h03/pipeline.json)

### Tray packing

#### acquire_block__h01

Role-factored relative geometry: a shared encoder processes both pose-and-goal bundles using world and TCP-relative translations, relative rotations, goal errors and causal differences. Learned latent-role attention pools object interactions; a separate fixed-base robot stream conditions a learned action DDPM. This is a representation-sharing prior, not a phase controller.

Parameters: 19,874,888; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h01/pipeline.json)

#### acquire_block__h02

Contact-progress-conditioned hidden semi-Markov latent diffusion: a two-frame recurrent belief encoder, six soft manipulation modes, uncertain run-age initialization and learned progress-dependent stay/exit hazards condition an absolute-joint/gripper DDPM. Training-only action, aperture, height and masked future co-motion cues supervise phase, pending-role, duration likelihood and readiness. Finite transition preferences favor demonstrated dependencies without an irreversible automaton.

Parameters: 18,598,171; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h02/pipeline.json)

#### acquire_block__h03

Attachment as an action-conditioned invariant: an original joint/gripper action diffusion model is conditioned on learned exclusive none/red/blue attachment beliefs and structured TCP/object forecasts. Attached forecasts compose a learned slowly varying TCP-to-object transform; independent forecasts have separate support-motion branches. Future pose, regime, rigidity, stationary-object and low-noise denoising-consequence losses train this distinction over the complete expanded slice.

Parameters: 20,021,364; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/tray_pack/acquire_block__h03/pipeline.json)

#### deliver_block__h01

Move the payload rather than only the TCP. A learned goal-relative TCP path latent with continuous rotations and gripper commands is conditioned on causal TCP-to-object transforms and attachment confidence. A local learned inverse-dynamics decoder supplies bounded normalized joint proposals to the joint-action DDPM denoiser. Future-path supervision, attached-payload consistency and a differentiable denoised-action/path cycle train the geometry. The mandatory sampler remains in joint-action space; this is an explicitly documented geometric-latent adaptation, not a separate Cartesian DDPM or exact IK.

Parameters: 19,583,897; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h01/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h01/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h01/pipeline.json)

#### deliver_block__h02

Rim-aware transport topology implemented as an absolute-joint diffusion policy with fixture-relative causal conditioning, learned contact/motion-mode forecasts, and a supervised action-chunk-conditioned pose ensemble. A differentiable, confidence-weighted geometric preference compares a denoised chunk with the demonstrated chunk through frozen rollout weights. This amortizes the assigned predictive-ranking hypothesis into diffusion training; the fixed inference interface does not perform completed-sample ranking.

Parameters: 19,661,368; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h02/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h02/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h02/pipeline.json)

#### deliver_block__h03

Containment is not disengagement. A learned two-frame recurrent contact belief with probabilistic phase persistence conditions an original-action diffusion policy. Separate containment, attachment and readiness predictions, future motion supervision, release-event survival likelihood and a detached-branch anti-reclosing loss distinguish closed-held goal entry from reusable empty-hand transitions.

Parameters: 18,730,377; formal updates: 20,000.

[API prior document](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h03/PRIOR.md) · [Exact source](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h03/policy.py) · [Pipeline](policy_cards/APPL_6_xhigh/tray_pack/deliver_block__h03/pipeline.json)

