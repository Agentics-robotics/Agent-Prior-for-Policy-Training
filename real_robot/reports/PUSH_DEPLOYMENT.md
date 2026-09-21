# Push portable deployment

Reviewed 2026-09-21. The user requested a downloadable package for another GPU
computer, selected SCP/rsync from this server, and assigned robot-side adaptation
to the receiving side. No API calls, retraining, robot execution, remote publication
or deletion of original training artifacts occurred.

The [deployment guide](../deployment/README.md) provides download commands,
checkpoint placement, independent environment installation and Python calls.
The [model manifest](../policies/push_v1/manifest.json) records original training
checkpoint hashes, exact submitted API source hashes and new inference hashes.

| Artifact | Bytes | Purpose |
| --- | ---: | --- |
| `contour_push.pt` | 14,706,789 | Final EMA and required inference normalization |
| `visual_push.pt` | 36,319,187 | Final EMA and required inference normalization |
| [Complete bundle](../exports/push_v1/push_v1_bundle.tar.gz) | 47,263,977 | Standalone code, lockfile, docs and both weights |
| [Weights-only archive](../exports/push_v1/push_v1_weights.tar.gz) | 47,195,464 | Extract at the root of a checkout containing the matching deployment code |

Download checksums and per-file inventory are in
[SHA256SUMS](../exports/push_v1/SHA256SUMS) and
[bundle_manifest.json](../exports/push_v1/bundle_manifest.json).
The full bundle works independently of whether the Git remote has received the
new workspace changes. It does not contain Python/CUDA dependency installations;
those are recreated using the included Pixi lockfile.

`real_robot/policies/push_v1/source/` is an exact-byte copy of all ten API-submitted
files. No scientific code was reformatted or edited. The exporter compares all
50 contour and 126 visual state tensors, including dtype and shape, to the
original final EMA; all are exactly equal. Constructor normalization and state
dimension are copied from the original metadata. Optimizer state, non-EMA model
weights, random states and training-only metadata are omitted. No precision
conversion or quantization occurs.

The developer-owned [PushPolicy](../deployment/policy.py) uses the current Python
interpreter, dynamically resolves the package location and keeps policy/inventory
memory in an isolated worker. The CUDA-only launcher does not require simulator
render devices or the old absolute Pixi executable. It retains source/weight
hash checks, GPU UUID/minor verification, Landlock and seccomp isolation.
The minimal inference environment pins the same Python and numerical-library
versions as training: Python 3.11.16, torch 2.14.0, numpy 1.26.4, scipy 1.17.1,
opencv-python 4.11.0.86 and pillow 11.3.0. Original Exp2 environment files and
scientific framework files were not edited.

Verification used the new independently installed inference environment and a
new directory extracted from the actual downloadable archive, with only that
directory on the project import path. It contained neither `real_robot/runs/`
nor the original-data reader. Both workers loaded on the available physical GPU
2 and successfully handled invalid-observation and unverified-controller input
without producing an executable command. These are packaging/interface checks,
not physical performance or generalization evaluations. No valid demonstration
or live robot observation was used in this check. See
[PUSH_DEPLOYMENT_CHECK.json](PUSH_DEPLOYMENT_CHECK.json) and the reproducible
[verification script](verify_push_deployment.py).

This establishes relocation on this host and the declared Linux/NVIDIA stack;
it is not a test on the user's destination computer. The destination supplies
its NVIDIA driver, supported kernel, bubblewrap/libseccomp, camera calibration,
paired observations and robot/HLA adapter.

The original `runs/` retains roughly 5 GB of reusable training inputs plus model
originals, journals and receipts. They are excluded from Git and the bundle.
No claim of increased motion coverage or unseen-letter performance accompanies
deployment: the original 8,606-example, 18/22-segment training scope still applies.
