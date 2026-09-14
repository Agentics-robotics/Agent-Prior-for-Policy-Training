# Experiment 1 frozen CandidateDesign interface

You author the A-candidates; the repository agent authors the shared framework and B1.
Read contracts.py and learning.py for the actual callable contract, including method shapes.
Candidate imports allowed: torch, numpy, math, experiment1.contracts and its own local modules.
Read language_policy.json for the exact fixed import and attribute restrictions.
Use explicit absolute imports; __future__, relative and wildcard imports are not allowed.
Private attributes are forbidden except the declared base-class __init__ call.
The worker enforces a filesystem whitelist, no network/process launching, and no credentials.
The API may only read this public capability and the current task×N evidence capability.
It may edit its unsubmitted candidate.py/config.json/design.json and optional local helper files.
No IO, loading weights, reflection, importlib/sys, shell/subprocess or public-module patching.

B0 public semantics: a CandidateDesign with condition(history, context)=history,
build_modules(factory) sets self.dp=factory(common_spec['observation_schema']['raw_dim']),
identity action encode/decode and no additional losses. B0 sees all the same raw fields.
B1 is a separately frozen repository-authored rule baseline; its source is not designer evidence.
You can choose more expressive causal representations/encoders/auxiliary heads without changing
the factory's U-Net or optimizer recipe. Joint diffusion keeps encoded native action4 first.
Learned modules must be registered on self during build_modules, not constructor or fit_support.
fit_support sees complete episodes {'obs':[T+1,D], 'actions':[T,4]} from this exact D_N.
Common normalization uses only obs[:T] in D_N, population standard deviation floor .001.
condition sees normalized history [B,2,D]; context['raw_history'] and ['raw_current'] are raw.
Build condition_dim is per observation, not flattened history. Use all parameters in the
joint loss. condition must return [B,2,condition_dim]. The common DP factory is called once.
Public diffusion epsilon action loss has fixed weight 1 and denominator mask.sum()*4.
Auxiliary loss is nonnegative, separately mask-normalized, with no future truth at inference.
Use only causal chunk-start context for encode/decode; gripper is not a translation vector.
The public executor alone clips native actions to [-1,1] and executes four of sixteen actions.

Three run_checks calls per candidate are allowed: initial check and at most two repair checks.
Checks perform zero optimizer updates and return interface diagnostics, never success rates,
training curves or task rollouts. Editing does not consume a new scientific model slot.
run_checks returns the code/config/design hash to pass to submit_candidate(expected_hash).
All required files must be written before checking. Submission freezes them permanently.
Design documentation fields: title, coverage_gap, reusable_regularity, dependencies_preserved,
evidence_refs, expected_failure_signature, required_runtime_fields, training_only_targets,
parents (empty for initial; A1–A3 IDs for A4), change_summary.
Initial A1/A2/A3 must all submit before performance feedback. A4 alone uses that feedback.
Formal training/development selection/testing are exclusively the outer runner's operations.
