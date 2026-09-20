# PRIOR: blue_place_release_retreat__h02

## Assigned heuristic

The assigned heuristic is a supported-release prior for the `blue_place_release_retreat` skill in `unstack_sort`.  The intended behavior is to keep the gripper closed while the blue block is carried and descended toward the blue goal, open only after the block is aligned with the blue goal and near table support, and then retreat with the fingers open while leaving both blocks on their pads.

The expanded M1_v2 slice for this policy is the full assigned range `520:stop` in each demonstration, not only the visible release instant.  Therefore the policy covers the shared transition from the predecessor's blue transport phase, the final descent, the release event, and the retreat/stabilization tail.

## Executable adaptation of the research hypothesis

The original hypothesis mentions table contact and support, but the provided state vector has no force/contact channel.  I therefore implement support as a learned causal state estimate trained from demonstration-derived labels: blue block xy alignment to `blue_goal`, blue block z near the table-height goal, finger aperture, and demonstrated gripper command.  At deployment those labels are not available; the model receives only the two causal observations supplied by the interface.

No IK, image encoder, contact solver, hard constraint, scripted release controller, or action replay is implemented.  The final action generator is still an epsilon-predicting DDPM over the eight native action dimensions: seven absolute Panda joint targets and one gripper command.

## Model architecture

`policy.py` builds `ReleaseGatedDiffusionPolicy`:

1. The two-step raw observation history is normalized with the shared M1_v2 normalizer supplied by the assignment.
2. A small set of broad-scale causal geometric features is appended: blue-goal displacement, red-goal displacement, TCP-blue displacement, TCP-blue-goal displacement, one-step TCP/blue motion, finger width, finger width change, blue z error, blue xy error norm, and finger velocities.  These use fixed metre-scale divisors, not per-skill refitted scales.
3. An MLP encoder maps the normalized history and geometric features to a latent state.
4. Two learned heads predict:
   - a binary `supported/aligned` logit;
   - a four-way event distribution over `carry_closed`, `descend_closed`, `release_open`, and `retreat_open`.
5. The latent state, support probability, event probabilities, and geometric features condition the public Conditional U-Net diffusion backbone.

The event/support heads are part of the differentiable model and their outputs are part of the denoiser condition.  Thus the prior is represented as a learned conditioning structure rather than as an external controller.

## Losses and gradient paths

`compute_loss` returns the required `loss`, `diffusion_loss`, and `prior_loss`.

- `diffusion_loss`: standard masked epsilon MSE between the denoiser prediction and the sampled DDPM noise.
- `support_bce`: binary cross-entropy on the learned support/alignment head.  Labels are derived from the current training state by checking blue block proximity to the blue goal in xy and z.
- `event_ce`: cross-entropy on the four event states.  Labels use the training-only demonstrated gripper command/finger aperture plus blue z and TCP retreat geometry.
- `unsupported_open_loss`: forms a differentiable low-noise estimate of the clean action `x0` from the denoiser output and penalizes positive/open gripper commands when the learned causal support probability is low.  The support probability is detached in this term so the denoiser cannot satisfy the penalty by declaring all states supported; the support head is trained by its BCE term.
- `gripper_bce`: a small low-noise auxiliary loss on the denoised gripper command against the demonstrated open/closed action label, to sharpen the release boundary.

The auxiliary terms that affect actions depend on the trainable denoiser prediction through `x0`; they are not constants computed only from observed poses.

## Causal deployment inputs

At inference the model receives only `raw_history [B,2,47]`, the noisy action sample, and the DDPM timestep.  Useful causal cues are:

- `blue_pose` and `blue_goal` in world metres for alignment and table-height inference;
- `tcp_pose` relative to the blue block for grasp/descent/retreat phase;
- `qpos[7:9]` and `qvel[16:18]` for finger aperture and finger motion;
- `red_pose` and `red_goal` to preserve the already placed red block context.

The model can take over partway through the shared transition because the two-step history distinguishes high closed transport (blue at goal xy but z high, fingers closed), low closed descent (blue close to table but fingers still closed), opening at support, and open-finger retreat.

## Applicability and termination cues

This policy is intended for the final blue placement and release after red is already at its goal.  It is appropriate when the blue block is grasped or immediately under the TCP and the desired blue pad is the fixed goal `[-0.24, 0.25, 0.02]` m.  The policy should usually continue through release and retreat until the gripper is open and the blue block is stationary/remaining near the goal table height.  The task contract does not require release or clearance for success, but the demonstrations strongly represent open-finger final states.

## Limitations

The support estimate is supervised by state/action correlations, not physical contact force.  The policy may release too early or too late under dynamics that differ from the demonstrations, and it has no special recovery behavior for a bounced block, a wedged block, or an object no longer near the gripper.  The architecture does not enforce geometric constraints on the generated joint targets; all robot motions remain learned through diffusion training.
