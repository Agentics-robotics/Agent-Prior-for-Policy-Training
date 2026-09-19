# Prior for `blue_release_retreat_complete__h02`

## Original heuristic identity

Assigned heuristic 2 is the **support-before-retreat prior**: open the gripper and withdraw only after the blue block is likely supported inside the open drawer.  The handoff evidence shows the expanded slice starts at index 930 while the blue block is still held high over the drawer, overlaps the transport policy through the release initiation region, then finishes after the fingers are open and the TCP has retreated upward while the blue block remains fixed in the drawer.

This implementation keeps that identity and trains over the full assigned expanded slice, including the shared transition from approximately 930 to 1030 and the final retreat/settling phase.

## Executable learned diffusion model

The model is still an action diffusion policy: `forward(noisy_action, timestep, raw_history)` returns the predicted DDPM noise for a horizon of normalized Panda joint/gripper actions.  It does not output scripted Cartesian targets and does not contain inverse kinematics.

The architecture is:

1. Normalize the two causal observations with the single shared M1_v2 normalizer supplied by the framework.
2. Build additional causal state features from the last two observations:
   - TCP, blue, red, drawer, and goal relative positions in world metres.
   - Finger width and two-frame finger motion from qpos.
   - Blue-to-goal, blue-to-drawer-cavity, TCP-to-blue, and TCP-to-blue-goal offsets.
   - Drawer open margin and two-frame TCP/blue motion.
   These relative features are scaled by broad task-scale constants such as 0.15 m, the drawer cavity half-widths, or nominal finger width; no tiny per-skill normalization is fitted.
3. A trainable MLP produces a 224-dimensional causal context.
4. A trainable four-logit support/readiness head predicts whether the current history looks like: placed/supported, opening imminent or present, retreat-ready, and currently in the blue goal height/xy band.
5. The logits and their sigmoid values are embedded and concatenated with the context to form a 256-dimensional global condition for the standard conditional U-Net diffusion backbone.

Thus the decoder is conditioned on a learned support latent, rather than a hard threshold or external controller.

## Loss terms and gradient paths

`compute_loss` returns the usual diffusion epsilon loss plus a differentiable prior loss:

- **Diffusion loss:** standard masked epsilon MSE against the DDPM noise.
- **Support latent BCE:** future observations in the training batch label whether the blue block is in the demonstrated drawer goal band, whether the fingers become open, and whether the open gripper/TCP-retreat phase occurs while blue remains placed.  The loss is applied to trainable support logits from the causal encoder.
- **Future blue/TCP/finger prediction:** from the causal condition and the model's low-noise clean action estimate, an auxiliary head predicts future blue displacement, TCP z displacement, finger width, and a support-sequence logit.  This makes the support/stability objective depend on trainable predictions rather than a constant geometric penalty.
- **Gripper clean-estimate timing loss:** a small low-noise auxiliary term encourages the denoised action estimate to represent the demonstrated close-to-open gripper transition.

The clean action estimate is clamped for numerical stability in the auxiliary heads and weighted toward lower-noise DDPM steps.  The main diffusion loss remains the primary training signal.

## Causal deployment inputs

At deployment, the model receives only the two most recent state observations: qpos, qvel, TCP pose, red pose, blue pose, drawer position/velocity, red goal, and blue goal.  Future observations are used only as training labels inside `compute_loss`.  There is no image encoder, force sensor, contact detector, analytic IK, or recovery planner.

The observation information that lets this policy take over partway through the transition is the relation among TCP pose, blue pose, finger width, and drawer/goal geometry.  For example, a mid-transition state shows blue already near z 0.063 m with the fingers opening and the TCP beginning to rise; the learned support latent can condition the diffusion model toward opening/retreat actions.  Earlier in the overlap, the same inputs show the TCP and blue still high and fingers closed, so the diffusion model is trained to continue the descent/centering portion before full retreat.

## Applicability and termination cues

The policy is intended for states where the drawer is open, the red block is already on the pad, and the blue block is grasped over or just inside the drawer.  It should be continued while the blue block is being lowered/released and the gripper is withdrawing.  A useful exit state has fingers open, blue still in the drawer/goal band, and TCP high enough above the drawer that it is unlikely to disturb the block.  These are handoff cues for the inference API and do not alter the task's formal completion rule.

## Limitations

The demonstrations do not include failed early releases, force/contact labels, or recovery after disturbing the block.  The learned support latent is therefore a supervised representation of successful release timing and future stability, not a proof of physical support.  The model cannot guarantee that blue will not follow the gripper upward if the state distribution is far from the demonstrated transition.
