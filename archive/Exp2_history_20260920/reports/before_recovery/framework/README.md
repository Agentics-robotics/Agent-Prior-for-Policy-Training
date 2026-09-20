# Frozen framework source

This directory copies the exact framework/environment files retained in the new
single-policy preparation receipt. Paths beneath this directory match their
repository locations; hashes are in index.json. No source is rewritten.

These files help inspect normalization, diffusion sampling, API tools, simulator
interfaces and geometric success. This subset is evidence for code reading, not
a standalone installation: original data, assets and checkpoints remain in the
repository. Historical methods retain their own executed source versions and
actual request prompts, indexed separately by the paper bundle.

The original Diffusion Policy vendor LICENSE and PROVENANCE.md are also included
unchanged and hashed separately in the index. The provenance document retains an
older `relative_dp` import name; the actual bundled Exp2 source uses the `appl`
namespace, as shown in its import statements.
