# acquire_block__h02: contact-progress latent diffusion

## Assigned hypothesis (unchanged)

The assigned heuristic is **Contact-dependent phase, not elapsed time**: infer variable-duration manipulation modes and make closing, lifting and incoming release conditional on observed progress. Its research proposal is a semi-Markov latent-conditioned diffusion policy to reduce averaging of incompatible actions at similar poses. This implementation keeps that identity and the complete assigned expanded segmentation. The following implementation choices are adaptations, not edits to the source heuristic.

## Evidence and full-slice responsibility

Actual observations and native actions were read from demo10300, demo10307 and demo10301, in addition to the supplied measured boundaries and overlap ranges.

* In demo10307 at 100 versus 110, TCP z is 0.019974 versus 0.019912 m, while each finger changes from approximately 0.040 m to 0.0183 m and the issued command changes from +1 to -1. Blue repeats the ambiguity at 465 versus 477; early lifting at 487 has blue center z approximately 0.02759 m. Purely static proximity cannot distinguish the required progression.
* Demo10300 265 is **not an empty-gripper blue start**. Red is at z 0.11212 m, next to the descending TCP at 0.11202 m over the red goal, with command -1. At 280 it is still held around z 0.0457 m; at 290 it rests at z 0.0360 m while the fingers are approximately 0.03707 m each and command is +1. At 300 the fingers are approximately 0.03998 m, before the open retreat visible at 320. Red must retain its identity through this transition even though blue is the pending table object.
* Demo10301 at 293 has only just commanded release: the fingers are still approximately 0.01828/0.01824 m and red is around z 0.04559 m. Demo10300 at 290 was already partly open. A source index or a fixed release timer is therefore not a contact label.
* Initial and final transition support includes approach at demo10300 75 and 445, closed table hold at 120 and 480, lifts at 160 and 520, and moving exits at 181 and 561. At 561, blue and TCP center heights are approximately 0.28898 and 0.28878 m, respectively, with substantial joint velocity. Across the supplied boundary measurements, exit finger widths are approximately 0.03652-0.03654 m and TCP heights approximately 0.280-0.293 m. These are observed support, not rejection thresholds or proof of contact.

Training uses every assigned sample in all 24 expanded segments. No slice is shortened to a tidy grasp episode. The initial segments teach red approach, dwell, close, lift and initial goalward motion. The later segments begin during red lowering and include its release, empty retreat, blue approach, dwell, close, lift and initial transfer. The shared overlaps with deliver_block are deliberately retained, including demo10300 [75,182), [265,415), [445,562) and the corresponding other demonstrations. There is no source-index input, memorized action sequence or external controller.

## Necessary interface adaptation

The original proposal asks for eight causal observations, prior issued actions, an intended-role input, persistent recursive belief, and 1-4-action execution near transitions. The executable interface fixes **two observations**, a 16-action chunk and **eight-action execution**; it supplies no previous-command channel, explicit role, history mask or persistent latent argument. Those facts are not changed in pipeline.json.

Accordingly, this model is a **two-frame, uncertain-age, hidden semi-Markov adaptation**, not a claim to implement the unavailable eight-frame controller. It performs one observed filtering transition, followed by a causal learned phase forecast over the rest of the action horizon. It recomputes belief on every call and does not store hidden state across DDPM iterations or policy invocations. Run age is a latent uncertainty initialized from observed progress, not a source timestamp. No global time, elapsed skill counter or trajectory ID enters the network.

A missing-history reset is represented by repeating the current observation in both slots. Fifteen percent of training batches' individual examples receive this augmentation. An exact duplicate is detected, its finite difference is zeroed, the history recurrent state is reset before the current token, and a fresh mode/age belief is inferred. Old-frame supervision is omitted for the augmented slot. The same deterministic reset behavior is available at deployment, including a truly unchanged repeated observation. This does not create the lost temporal information. With only two frames, release-versus-grasp intent and exact dwell age can remain ambiguous.

The model learns a soft **pending role** (red or blue) from both objects, their goals and robot state. Training weak role labels move toward blue when red is horizontally near its own goal, including while red is still being lowered. Thus pending role is not an assertion that the hand is empty or that blue is already attached. Incoming mode plus the unmodified red/TCP features preserves the red payload. There is no externally selectable object order; arbitrary blue-first or reversal invocations are unsupported by this dataset/interface.

## Learned representation and diffusion architecture

Deployment input is raw_history [B,2,47] only. Positions/goals are world metres, pose quaternions are wxyz, qpos[0:7] are arm radians, qpos[7:9] are measured per-finger openings, and qvel[9:18] are the corresponding velocity channels. Drawer channels are constant compatibility inputs; no drawer is inferred.

Each observation token concatenates:

1. 47 state channels normalized with the supplied **single shared full-demonstration range normalizer**;
2. 47 normalized causal differences multiplied by 20 Hz, zeroed when unavailable;
3. 12 relative coordinates: TCP-minus-red, TCP-minus-blue, red-minus-red-goal and blue-minus-blue-goal;
4. a valid-difference bit.

Relative TCP/object coordinates use root-sum-square shared positional half ranges; object/constant-goal offsets use that object's shared positional half ranges. No empirical per-skill scale is fit. Fixed metre-valued numbers in label construction are annotation hyperparameters, not normalization estimates or runtime action gates. The full normalizer is the assigned one with normalizer_sha256 874039c9e091b43eacce196d1283f9944073808a6049a059b2e07938248a7ce8, fit once to 8,897 samples from all complete original demonstrations. Quaternion channels retain their fixed unit-component scale. Observations are not clipped.

A learned MLP and 128-unit GRUCell process the causal tokens. Learned heads produce emissions for six modes, per-mode run-age initialization, a two-way pending-role distribution and a scalar readiness probability. The six modes are:

0. incoming lower/release;
1. empty retreat/approach;
2. aligned dwell;
3. closing, including closed on-table hold;
4. attached vertical lift, including elevated hold before translation;
5. early translation.

The joint mode/age belief has six modes and eight ages: local steps 0 through 6, and a right-censored 7+ bin. Every step, learned hazards depending on the recurrent progress features plus trainable mode/age offsets split probability into stay versus exit. Stays increment age; the last bin retains arbitrary further dwell. Exits reset destination age to zero. Destinations are learned softmax probabilities with a finite -2.5 logit preference against skipping the cycle 0->1->2->3->4->5->0. Nonadjacent moves remain possible. Self transitions are represented by the stay branch, not a competing exit-to-self. This is a probabilistic temporal prior, not an irreversible finite-state machine. The 5->0 link encodes the conceptual manipulation cycle; full transfer between the initial and incoming slices is absent and is not fabricated as an adjacent training transition.

For the second observed frame, a predicted joint belief is updated by learned observation emissions. A separate recurrent forecast rolls 14 additional mode/age transitions from current hidden state, posterior and pending role. It receives **no future observation and no native action**, even in training. Hazards are state/progress conditioned; the finite latent age can modulate dwell, but there is no fixed-duration rule. Ages beyond seven steps have a censored tail, so this is an approximation to a richer semi-Markov duration distribution.

The denoiser condition combines both state tokens, recurrent state, current mode belief, expected learned mode embedding, forecast mode marginals at four horizon locations, pending role and readiness, projected to 256 dimensions. A direct state pathway avoids a rigid six-mode bottleneck. These learned conditions FiLM-condition the public temporal U-Net (128/256/512 widths, kernel 5, groups 8). The output is epsilon [B,16,8] for the fixed DDPM. Actions remain the original shared-affine-normalized seven absolute Panda joint targets and continuous gripper command. Positive command opens; -1 requests closure. A commanded per-finger target of -0.01 m is not a measured attachment condition: the demonstrated block arrests the fingers around +0.0183 m.

There is no IK, forward kinematics, image encoder, geometric equivariance, contact sensor, hard collision projection or scripted action composition. All eight output channels are learned diffusion outputs. Relative state features do not make absolute joint actions equivariant.

## Training labels, alignment and losses

Labels are made under no_grad from each training batch, not queried by forward. Action slots index t-1 through t+14. The corresponding pre-action observation sequence is raw_obs[0], raw_obs[1], future_obs[1:15]; future_obs[j] is the state after action slot j. Only valid pre-states/actions contribute; padded future co-motion is disabled using future_mask and never crosses a slice.

Weak soft phase labels use native gripper commands, actual finger width, pending-object/TCP alignment, object elevation, red's relation to its own goal, and up to four post-action states of TCP/object displacement. Co-motion compares displacements with an 8 mm soft error scale and upward progress. Elevated closed nearby objects provide an additional attachment cue, so an elevated stationary hold is not labeled empty. Translation also requires high object elevation and horizontal movement. Lower/release explicitly uses red/TCP proximity near the red goal, supporting the incoming closed payload and the partially opening fingers. Open on-table alignment is distinct from commanded closing. These are heuristic soft annotations, not measured contact ground truth. Near a transition, lookahead can label imminent lifting before a large elevation is visible; they should not be interpreted as instantaneous verified attachment.

All annotation thresholds are proposed hyperparameters, not demonstrated robustness bounds. Targets receive one-percent uniform smoothing. A confidence weight derived from the largest unsmoothed phase probability downweights ambiguous modes; highly ambiguous labels are masked. Future_mask removes unavailable lookahead and uses available backward motion instead. Neither targets nor future inputs are passed into the denoiser condition.

Total loss is:

`L = L_epsilon + 0.15 L_phase + 0.08 L_duration + 0.03 L_role + 0.03 L_ready + 0.01 L_skip`.

* **L_epsilon:** ordinary mask-normalized epsilon MSE against repository DDPM noise. This trains the U-Net and all connected conditioning modules; no clean-action replacement or action replay occurs.
* **L_phase:** confidence-weighted soft cross-entropy on the first and filtered current mode marginals. The reset-augmented first slot is excluded. Gradients train emission, age/transition filtering and the recurrent encoder.
* **L_duration:** masked marginal likelihood of future phase evidence through the learned semi-Markov forecast. The initial current joint is reweighted by current phase evidence only inside this loss. Each predicted transition is followed by a training-only evidence update, marginalizing uncertain ages and modes. The evidence is `(1-confidence) + confidence*soft_label`; zero confidence is unit evidence and contributes zero likelihood loss. The negative log normalizers are divided by summed confidence. This is a noisy-label sequence likelihood, not hard run-length truth. It trains hazard heads, run-age initialization/offsets, transition destinations, forecast recurrence and shared encoder. It provides a holding-time/survival learning signal without a trajectory clock or requiring complete runs at slice boundaries.
* **L_role:** masked soft cross-entropy for pending role, training its head and encoder.
* **L_ready:** confidence-weighted binary cross-entropy predicting the soft probability of modes 4 or 5. It trains a readiness head and encoder; it is not a completion reward or contact oracle.
* **L_skip:** expected exit probability on nonadjacent dependencies, computed from **predicted joint beliefs, predicted hazards and predicted destination probabilities**. It trains those modules; it is not a pose-only constant penalty. Its small weight and finite destination bias allow uncertainty and exceptions.

Every nonzero auxiliary depends on a trainable prediction. Future observations only construct masked supervision. The duration loss's label-conditioned filter is separate from the entirely causal forecast used in the forward pass. No high-noise clean-action amplification is used by auxiliaries.

The optional model.belief method exposes phase, pending role and learned readiness for inspection. The fixed deployment forward returns epsilon only and the framework does not promise to export these diagnostics or execute a posterior-based switch. Handoff decisions can always use the external causal state evidence below; no claimed adaptive invocation mechanism is hidden in the model.

## Handoff and useful exit

HANDOFF.json specifies the API-facing evidence and responsibility. With normal operation provide the latest two causal observations, keeping real temporal differences at overlap entry. If unavailable, repeat the latest observation; uncertainty may remain higher. The inference agent can retain a longer buffer or its own issued-command history as external handoff context, but this network only receives the last two states and cannot use a supplied role directly.

Entry may be initial approach, aligned open dwell, fingers closing, closed on-table hold, early attached lift, or red descending/partly releasing with blue pending. Aperture **and its change**, joint velocity, world TCP-to-both-object relationships, two-frame co-motion, object elevation and goal-relative geometry supply the takeover evidence. Stable apparent aperture alone cannot prove grasp: finger velocity channels are nonzero even in some demonstrated arrested holds. Incoming red placement is learned as part of this policy, not discarded as another skill's work.

Continue this policy if red still needs lowering/release/empty retreat or if the selected table block is aligned/closing without corroborated elevation and co-motion. A useful ordinary exit favors attached lift or early goalward translation with the block near the TCP, measured width consistent with holding, closed intent and sustained local co-motion when available. Do not require one exact arm pose, full future transfer or zero qvel. An earlier switch is supported by overlaps only if the successor knows to finish closing/lifting or release/retreat; geometric proximity alone is not attachment. Pass both identities/goals and causal history, not just a subgoal bit.

The tray_pack completion contract remains red_at_goal AND blue_at_goal with full rotated XY containment and its stated z tolerance. It requires neither an additional gripper-release condition nor clearance, velocity or sustained-hold predicates. Handoff recommendations must not redefine that task success.

## Applicability, limitations and falsifiability

The state-based trained prior is applicable to the demonstrated red-first tray task with comparable sensing and actuation delay. Learned uncertain transitions are preferable to trapping errors in a hard automaton, but successful reversals, interrupted grasps and spatial/contact recovery are not in the evidence. No recovery robustness is claimed. The two-frame interface cannot fully resolve the original eight-observation hypothesis, exact stationary dwell age or command onset. Fixed eight-action execution can bridge a contact change before the next observation; the proposed shorter execution near contact is **not implemented**.

Possible failures are grasp/release phase toggling, alternating gripper commands at essentially fixed pose, high readiness while a block stays on the table, a stationary arrested grasp labeled empty, release of the wrong block, or a reset causing a jump to late transport. Censored ages and weak supervision may alias subtle transitions. Pending-role inference inherits the demonstrated ordering and does not support arbitrary requested roles. Neither phase confidence nor any metre threshold is calibrated by rollout success.

The frozen recipe remains 20,000 updates, batch 128, seed 0, AdamW 1e-4/1e-6, cosine with 500 warmup, gradient norm 1, EMA 0.999, H=16, two observations, eight executed actions, 100-step DDPM training/sampling, clipped samples and last EMA selection. Interface validation tests two updates and reload/sampling only; it is not performance feedback. Suitable subsequent tests are randomized overlap/reset entry, modest delay/timing perturbations, and comparisons to recurrent diffusion without phase labels, feedforward relative diffusion and a recurrence-only/duration-ablation model. None of these experiments or successful rollouts are claimed here.
