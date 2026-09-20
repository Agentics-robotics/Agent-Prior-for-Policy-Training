# PRIOR.md: acquire_block__h03

## Assigned research prior

The assigned heuristic is **clearance via-point acquisition** for the `acquire_block` skill in `tray_pack`.  The intended behavior is not an arbitrary end-to-end curve: the robot approaches the active table block from above, reaches a grasp pose, closes the gripper, lifts to a safe clearance height, then carries the object along a high corridor toward the corresponding marked tray region.  The expanded M1_v2 slices include both the red acquisition segment (`0:250`) and the blue acquisition segment (`330:625`) for every demonstration, including overlap with adjacent `place_block` transitions.

The original heuristic suggested reusable waypoints:

1. pre-grasp hover above the active block,
2. grasp at the block,
3. lift-clearance above the table/tray rim,
4. goal-approach or handoff waypoint above the active block goal.

## Executable adaptation

The dataset does not provide annotated waypoint labels, robot forward kinematics, collision checking, or an IK controller.  Therefore I implemented the prior as a **learned action diffusion policy with an auxiliary waypoint representation**, not as a scripted Cartesian controller.

At each denoising call the model receives only the causal two-step observation history.  It computes world-frame geometric features from:

- TCP position,
- red and blue object positions,
- red and blue goal positions,
- Panda joint/gripper state,
- the previous TCP position from the two-frame history.

A deterministic role cue is used only to form features and labels: red is treated as active until it is low and close to its red goal; after that the active role switches to blue.  The soft red-done score is also provided as a feature so the network can smooth the shared transition.  This matches the assigned full slices: red acquisition first, then blue acquisition after red has been placed.

Because waypoint annotations are absent, analytic waypoint targets are generated from the current active object and goal poses using the same world coordinate frame as the observations.  A trainable waypoint head predicts a bounded residual on the current next waypoint, plus phase logits for hover/grasp/lift/goal-approach and a progress scalar.  The diffusion model conditions on these trainable waypoint predictions.

## Architecture

`policy.py` defines `WaypointConditionedDiffusion`.

- Observation normalization uses `appl.public.normalize_observation` and the single shared normalizer supplied by the assignment.  No per-skill rescaling is fitted.
- Geometric features include active/inactive object-to-TCP vectors, object-to-goal vectors, TCP-to-goal vectors, gripper opening, role indicators, four candidate waypoint offsets, phase weights and a two-frame TCP delta.
- A trainable MLP waypoint head maps the 166-D feature vector to:
  - next TCP waypoint residual in metres,
  - four phase logits: hover, grasp, lift, goal approach,
  - progress scalar.
- A condition MLP maps geometric features plus predicted waypoint/phase/progress to a 256-D global condition.
- The denoising network is the provided `DiffusionBackbone` U-Net and predicts epsilon for the normalized 16-step, 8-D joint/gripper action sequence.

The policy remains a learned DDPM action model.  It does not output Cartesian waypoints directly to the robot and does not replay demonstrations.

## Losses and gradient paths

`compute_loss` returns the required `loss`, `diffusion_loss`, and `prior_loss`.

- `diffusion_loss`: standard masked epsilon prediction loss against the sampled DDPM noise.
- waypoint MSE: trainable predicted waypoint versus the current analytic via-point target.
- phase cross-entropy: trainable phase logits versus the analytic hover/grasp/lift/goal mixture.
- progress MSE: trainable progress scalar versus phase-derived progress.
- clearance barrier: penalizes the trainable predicted waypoint if it falls below a conservative clearance z during lift/carry conditions.  This is not a state-only penalty; gradients flow into the waypoint head and condition encoder.
- gripper clean-action consistency: reconstructs the denoised action estimate from the predicted epsilon and applies a small masked consistency term on the normalized gripper command.  Alpha-bar weighting avoids making high-noise samples dominate this auxiliary term.

The prior losses affect trainable predictions used by the diffusion condition.  The final action distribution is still learned through DDPM denoising.

## Causal deployment information

At deployment, `forward(noisy_action, timestep, raw_history)` uses only `raw_history [B,2,47]`.  Future observations are used only during training as labels by the outer framework and are not read in inference.

The model can take over partway through a transition because the observation contains the causal state needed to infer phase:

- gripper opening distinguishes open approach/retreat from closed carrying,
- active object height distinguishes table approach from lift/carry,
- active object-to-goal vector distinguishes transport from goal funnel entry,
- TCP-to-object vector distinguishes hover/approach from near-grasp states,
- the red low-at-goal cue distinguishes the first red acquisition from the later blue acquisition.

Thus the same policy can produce the shared motions in the expanded slice: red carry into the place handoff, retreat/approach transition before blue acquisition, and blue carry into the place handoff.

## Applicability and limitations

This prior is appropriate for the demonstrated tray/table geometry and top-down Panda grasping setup with reliable state observations.  It is designed for generalization over block positions within the same workspace and goal layout.

Limitations:

- waypoint heights are fixed constants with learned residuals, not collision-checked constraints;
- no external IK, FK, image encoder, force/contact detector or tray geometry parser is implemented;
- grasp success is inferred only from state and demonstrations, not explicitly sensed;
- behavior outside the demonstrated top-down acquisition and high-carry corridor is untested;
- the overlap documentation guides the inference API but does not redefine task success.
