# PRIOR: red_to_buffer_handoff__h03

## Assigned heuristic

This policy implements heuristic 3 for `red_to_buffer_handoff`: the red-to-blue transition should be represented as a learned overlap-readiness continuum rather than as a single fixed boundary. The assigned slice is the full expanded segment `[0, 460)` for each demonstration, including the shared overlap `[320, 460)` with the blue-acquisition successor. The intended behavior is to move the red block from its initial lower region to the demonstrated central temporary buffer, release it, retreat upward with an open gripper, translate toward the blue block, and continue until the successor can take over from either a high open approach or the beginning of low blue contact.

## Executable adaptation of the hypothesis

The repository provides state histories, action labels, DDPM noising/sampling, optimizer, and a shared normalizer. It does not provide an IK solver, contact sensor, image encoder, or a policy-selection API argument inside `forward`. Therefore I implemented the heuristic as a learned state-conditioned action diffusion model with an auxiliary readiness head:

- `forward(noisy_action, timestep, raw_history)` predicts DDPM epsilon for normalized native joint/gripper action sequences.
- `handoff_readiness(raw_history)` is an additional causal method returning the readiness probability for the latest observation. This is not a scripted controller; it is a trainable head sharing the state encoder used by the diffusion model.
- The temporary buffer position is represented only as a geometric feature and soft-label reference, using the demonstrated release cluster around `[-0.181, -0.001, 0.020]` m. It is not used to overwrite actions or to command a hand-coded subgoal.

All observation normalization uses the assignment's shared M1_v2 normalizer fitted on the complete original demonstrations. No per-skill normalizer or tiny overlap scale is fitted.

## Architecture

The model is `OverlapReadinessDiffusionPolicy` in `policy.py`.

1. Each of the two causal observations is transformed into an 85-dimensional feature vector:
   - the 47 shared-normalized observation channels;
   - relative world-frame vectors among TCP, red block, blue block, goals, and the demonstrated buffer;
   - broad-scale distances and heights using metre-level constants rather than per-skill empirical ranges;
   - finger width and finger asymmetry from `qpos[7:9]`.
2. A trainable MLP encodes each observation into a 128-dimensional step embedding.
3. A trainable readiness head predicts one logit per history step.
4. The two embeddings, geometric features, temporal feature difference, and readiness logits/probabilities are fused into a 256-dimensional global condition.
5. The condition is passed to the supplied `DiffusionBackbone` Conditional U-Net, which predicts epsilon for the 16-step action horizon.

The policy remains a learned action diffusion model: deployment actions come from DDPM sampling through the learned epsilon predictor, not from a geometric controller or replay rule.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: the standard masked epsilon-prediction loss between the U-Net output and the sampled DDPM noise. Gradients train the backbone, condition encoder, state encoder, and readiness-conditioned representation.
- `prior_loss`: `0.05 * BCE(readiness_logits, soft_readiness_targets) + 0.01 * smoothness_loss`.

The soft readiness targets are inferred from causal observed state fields: red table height and proximity to the demonstrated buffer, TCP/blue XY relation, TCP height, and finger opening or low blue contact. The smoothness term matches the change in predicted readiness across the two causal observations to the change in the soft target. These terms are differentiable with respect to the trainable readiness predictions. They are not constant penalties on observed poses.

The readiness label is intentionally soft. Before red release it is near zero because red is not table-height near the buffer. It rises after release when the TCP is high/open along the buffer-to-blue translation, and it remains high for the low/closing blue-contact state even though the gripper is no longer open.

## Causal deployment inputs

At deployment the model receives only `raw_history [B, 2, 47]` with the fields defined by the interface: qpos, qvel, TCP pose, red pose, blue pose, drawer compatibility channels, and goals. Future observations are used only as training labels by the framework; this implementation does not read future observations inside `forward` or readiness inference.

The observation information that allows takeover partway through the transition is:

- red pose in world metres, indicating whether red has been released on the central buffer instead of still being carried;
- finger width from `qpos[7:9]`, indicating open release/retreat or closing at blue contact;
- TCP pose relative to the blue block and buffer, indicating whether the robot is in the high approach tube or at the low contact pose;
- blue pose, indicating whether blue is still essentially in its initial region or only just being contacted.

## Applicability and termination cues

The policy is appropriate from the beginning of the assigned red-to-buffer skill and through the demonstrated transition into blue acquisition. The selector should continue this policy if red is not yet buffered, the gripper is closed away from blue, or the TCP is not in the observed transition corridor. A successor can be considered when the learned readiness is high and the successor also reports compatibility: red is buffered, blue is not significantly displaced except by initial contact, and the TCP/gripper state lies in the observed high-open to low-closing approach continuum.

## Limitations

Readiness is inferred from demonstration geometry, not from explicit contact labels or intervention data. The model cannot prove grasp success, recover from a missed blue grasp, recover after red is displaced from the buffer, or enforce exact containment. There is no unimplemented IK, no external collision checking, no image processing, and no hard constraint projection in this policy.
