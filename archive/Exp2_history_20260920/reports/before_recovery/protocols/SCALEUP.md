# Five-task DP versus APPL study

Preparation review: 2026-09-16. This is a new study, not a revision of frozen M0
results or the initial M1 tests. The user authorizes developer-owned task design
and expert demonstration collection. Segmentation, heuristics, prior source,
semantic handoff documents and inference decisions remain Runtime API-owned.

Resource amendment: the user subsequently authorizes any physical GPUs 0–7,
multiple jobs per device, with at most four physical GPUs active for this study
at once. The current pool is **1,2,3,7**, chosen after occupancy inspection. Existing
processes are preserved; checks may share a device with this study's own training.
The preceding 4–7 receipts remain unchanged. Resource-only configuration changes
and their original versions are retained in `M1_scaleup/allocation`.

## Tasks and data

Original drawer exchange plus four new native Panda tasks: two-block sorting,
block exchange using a temporary region, unstacking followed by sorting, and
packing both blocks into an open tray. New tasks use the same native joint/gripper
action interface, two causal 47-dimensional state observations, and 20 Hz control.
Absent drawer channels are zero; target channels carry the actual task targets.
Definitions and predeclared layouts are in `configs/scaleup` and
`runs/exp2/M1_scaleup/preparation.json`. Original training inputs are preserved.

Each new task has twelve predeclared training seeds. Expert planning is a data
acquisition tool only; no object poses are overwritten after reset, and no expert
is available to either learned method. Failed acquisition attempts and framework
repairs are retained with source versions. A fixed symmetric grasp orientation
was corrected during preparation after wrist-limit planning failures; seeds were
not filtered. Preliminary failed/superseded data is under `preparation_archive`.

## Fixed learning and evaluation

- Naive DP: vanilla state-conditioned U-Net, widths 128/256/512, 60,000 updates,
  final EMA, twelve complete demonstrations. No API prior or auxiliary loss.
- APPL: the existing M1_v2 procedure, large API-selected transition overlap,
  distinct API-authored prior pipelines, 20,000 updates per policy, final EMA.
- Both use DDPM 100, execution 8, horizon 16, history 2, batch 128 and seed 0.
  One observation/action normalizer per task fits only the complete twelve
  training demonstrations. API slices never refit it.
- The original drawer uses the existing frozen M0 checkpoint and 18 M1_v2 policies.
- Each task: thirty new ID and thirty new position-OOD layouts, paired across
  methods. New tasks: ID XY offsets within 1.2 cm; OOD outside that support with
  at least one axis offset 2.2–4 cm. Stacked blocks move together at reset.
  No object, robot, controller or goal changes between ID and OOD.
- Each method gets 5,000 physical steps. APPL retains the latest numeric stop
  conditions, projected context and notebook protocol; the API cannot see the
  total or remaining episode budget. It chooses whether to continue or finish.
- Identical geometric success for both methods. No additional terminal release,
  clearance, velocity or hold requirement. Report success by 1,500/3,000/5,000
  steps as well as failure/interruption status; original M0 metrics stay frozen.
- Freeze all task libraries before formal learned-policy testing. No test-driven
  prior design or selection. Preserve every provider/physical attempt; no
  automatic transport retries or replacement of unknown outcomes with failures.
- Repeat the initial study's real B=1 deployment-worker checks using nine original
  training observations per model. These generate two denoising chunks, zero
  simulator steps and zero optimizer updates; they test dispatch/shape/queue
  compatibility, not task performance. Preserve each check receipt before freezing.
- Environment and diffusion seeds are explicit and fixed; the inference API has
  no fixed generation seed. One API decision trajectory is observed per layout.
  This study does not estimate between-training-seed or repeated-API variability.
- Simulation advances only through physical control steps; it pauses during API
  calls. Record API and wall-clock time separately. These results do not establish
  real-time robot deployment performance.

Total intended matrix: five tasks × two methods × two conditions × thirty =
600 episodes. Runtime under `runs/exp2/M1_scaleup`; data under `data/exp2/scaleup`;
orchestration under `src/appl/scaleup`, reusing `src/appl/prior_policies`.
Actual costs, trained parameter counts and API journals will accompany results;
APPL's aggregate training budget is larger and is not compute-matched to DP.

## Initial-test organization

The active initial study is `runs/exp2/M1_initial_test/inference_5000`, with its
unchanged policy library in `M1_initial_test/policies`. The 1,500- and 3,000-step
studies and original configurations are in `archive/Exp2_M1_initial_tests`.
Historical paths are compatibility links. `M1_INITIAL_LAYOUT.json` records the
byte-preserving migration of 4,615 runtime files and original configurations.
