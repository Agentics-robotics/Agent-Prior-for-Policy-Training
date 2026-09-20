# Mathematical description of the executed interfaces

This is developer-written documentation of the frozen framework. Candidate
architectures and auxiliary objectives are API-authored and are specified in
[the prior catalog](PRIOR_CATALOG.md) and its unchanged source packages.

## Data, normalization and alignment

For each task, the training set contains twelve trajectories

\[
\mathcal D=\{(s^i_0,a^i_0,s^i_1,\ldots,a^i_{T_i-1},s^i_{T_i})\}_{i=1}^{12}.
\]

The observed vector \(s\in\mathbb R^{47}\) contains robot, object and target
state. The native action \(a\in\mathbb R^8\) contains absolute joint targets and
the gripper command. Normalizer parameters are fitted once on the complete
training set, before skill segmentation. In the current limits configuration,

\[
\widetilde s_d=(s_d-c_d)/r_d,\qquad
\widetilde a_d=2(a_d-\ell_d)/v_d-1.
\]

Here \(c,r\) are training observation midpoints and half-ranges, \(\ell\) is the
training action minimum, and \(v\) is its floored range. Quaternion dimensions
instead have \(c=0,r=1\); nearly constant observation dimensions use the fixed
handling documented in source. States are not clipped. API-designed features
may also use raw physical coordinates and declared fixed scales.

At each eligible action index \(t\), the causal history contains
\(H_t=(s_{t-1},s_t)\), with the first observation repeated at a boundary. The
target action chunk is

\[
A_t=(\widetilde a_{t-1},\widetilde a_t,\ldots,\widetilde a_{t+14})
\in\mathbb R^{16\times8}.
\]

A mask \(m_{tj}\) indicates whether chunk slot \(j\) refers to a valid action in
that trajectory or segment. Out-of-range values are padded but excluded from
the loss. APPL constructs these windows within each assigned skill segment;
overlap can therefore place the same original transition in several independent
skill datasets. The full-task methods construct windows over entire trajectories.

Training-only future label slot \(j\) is the observation after the action in
that slot, namely \(s_{t+j}\). Slot zero is ordinarily the current observation,
not the next observation. The future mask excludes invalid action slots. Future
states are never passed into deployment inference.

## Diffusion objective and execution

For a sampled diffusion index \(k\in\{0,\ldots,99\}\) and Gaussian noise
\(\epsilon\), the noised action sequence is

\[
X_k=\sqrt{\bar\alpha_k}A_t+
\sqrt{1-\bar\alpha_k}\epsilon.
\]

The public epsilon-prediction objective, including batch index \(b\), is

\[
\mathcal L_{\mathrm{diff}}=
\frac{\sum_{b,j,d}m_{bj}
  (\epsilon_{\theta}(X_k,k,H_t)_{bjd}-\epsilon_{bjd})^2}
 {8\sum_{b,j}m_{bj}}.
\]

The implementation guards an empty denominator. API packages can construct a
learned condition \(\phi_\theta(H_t)\), change trainable model structure, and
add declared auxiliary losses. The selected auxiliary terms and coefficients
are package-specific; there is no universal APPL contact or physics loss.
An auxiliary objective affects actions only through its actual gradient path
into parameters used by the denoiser. Its name is not a proof of physical
consistency.

All current methods use DDPM with the retained squared-cosine schedule and one
hundred denoising steps. The model predicts a sixteen-slot chunk; the executor
skips historical slot zero and applies the eight aligned current/future actions,
then samples again. Actions are decoded to native units and constrained by the
native controller bounds. Early geometric success or an APPL invocation stop can
interrupt the prefix. Final EMA weights are used; no test-time optimization runs.

## APPL policy decisions

Let \(\mathcal P=\{\pi_1,\ldots,\pi_K\}\) be the frozen policy library for a
task. Each policy has API-authored prior and semantic handoff documents and
framework-measured training-support information. At decision index \(r\), the
API consumes its observation/document context and may choose

\[
(p_r,d_r,C_r,n_r),
\]

where \(p_r\) identifies a learned policy, \(1\le d_r\le300\) is a requested
invocation length, \(C_r\) contains literal numeric observation conditions,
and \(n_r\) is its updated notebook. It can also read evidence, observe without
stepping, or finish. A stop group is a conjunction; any satisfied group returns
control. The framework measures those conditions after each native action,
without interpreting their labels as additional rules.

The first whole-task success, an API finish, or the executor's private physical
cap can end a completed episode. An API finish is not itself evidence of success.
All main-matrix methods share the same task-dependent geometric predicate
\(G(s)\); completion requires \(G(s_t)=1\) at one observation. The single-policy
baseline repeatedly executes its one learned model without these API decisions.

## Sources and limits

- [Public normalization/loss/backbone helpers](framework/src/appl/public.py).
- [Full-task windows and scales](framework/src/appl/dp_baseline/data.py).
- [Segment windows and future labels](framework/src/appl/prior_policies/data.py).
- [API policy process](framework/src/appl/prior_policies/runner.py).
- [Runtime invocation executor](framework/src/appl/prior_policies/deploy.py).

These equations describe the executed learning/control interface. They do not
establish that relative features make absolute joint actions equivariant, that
future regression predicts contact correctly, or that segmentation removes
closed-loop distribution shift. Refer to measured results for task performance.
