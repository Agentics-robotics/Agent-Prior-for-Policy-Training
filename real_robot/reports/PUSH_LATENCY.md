# Push inference latency diagnosis

Reviewed 2026-09-21. The user reported a destination computer measuring
0.6–0.7 seconds per inference. Its hardware, measurement boundaries and calling
loop are unavailable. The measurements below describe this server and do not
establish the destination's exact bottleneck.

Two bounded, sequential latency runs used the unchanged deployed code/checkpoints
on otherwise idle physical GPU 2 (RTX A5000), AMD EPYC 7543 CPU, two torch/OpenCV
threads, and the locked deployment environment. Each policy received 18 consecutive
original paired observations, first trajectory indices 300–317. The first three
calls were excluded from steady summaries. Image file decoding and model startup
were outside steady `act` timing. A local +20-pixel goal derived from the first
inventory instance was used solely to exercise the interface; it is not a task
label or a performance evaluation. No robot commands, API calls or training
updates occurred. No policy implementation or deployment archive was changed.

| Measured component | contour_push | visual_push |
| --- | ---: | ---: |
| Worker/model startup, once | 6.125 s | 6.321 s |
| Initial public `inventory`, 8 instances | 537.4 ms | 512.0 ms |
| First public `act` | 483.3 ms | 506.1 ms |
| Public `act`, steady median / p95 | 135.4 / 145.1 ms | 143.4 / 147.1 ms |
| Direct API `act`, instrumented median | 70.0 ms | 74.3 ms |
| Observation conversion, median | 52.0 ms | 50.2 ms |
| Detection alone, median (included in conversion) | 39.4 ms | 39.0 ms |
| Selected features, median | 11.2 ms | 11.0 ms |
| Graph construction alone, median (included in selected features) | 10.6 ms | 10.4 ms |
| Angle registration, median | 1.9 ms | 1.9 ms |
| History packing / GPU transfer, median | 0.46 ms | 4.14 ms |
| Neural forward only, synchronized median | 2.71 ms | 5.50 ms |

Stage timings include CUDA synchronization around forward and history transfer.
Nested timings must not be added twice. Instrumented direct calls and public
IPC calls are separate runs; their roughly 65–67 ms mean difference is an
approximate indication of serialization/IPC/wrapper overhead, not a same-call
exclusive breakdown. One of the 15 steady observations returned `uncertain`
before neural inference in both policies; forward/history timing therefore has
14 samples, while complete-call summaries include all 15. All other calls
returned `running`; this is not evidence of successful manipulation.

The current public `act` request was **12,294,405 bytes** for contour (visual is
one byte shorter). It transmits both full 1280x720 RGB observations and the
unchanged full-image target template through JSON/Base64 on every step. Host
encoding alone took 26–31 ms in the sampled encoding. Inventory returns a full
RGB template and full mask for each instance: eight such templates comprise
about 29.5 MB of raw arrays, or 39.3 MB after Base64 before small metadata fields.
Inventory timing includes perception, serialization and return transfer; its
individual contributions were not profiled here.

Three source-level contributors are confirmed:

1. [The host adapter](../deployment/policy.py) serializes full observation and call
   dictionaries for each request. [The wire format](../deployment/common.py) uses
   Base64 numeric arrays, not shared memory or compressed image transport.
2. [selected_features](../policies/push_v1/source/perception.py) always constructs
   both visual and graph representations. Even `visual_push` computes the graph
   it does not feed to its network. Goal distance transforms/contours are also
   recomputed as part of that graph path.
3. [The invocation implementation](../policies/push_v1/source/invocation.py) resets
   motor memory when the call/template/goal changes. Passing a newly generated
   template or `reset=True` every step changes normal steady-call behavior and
   can repeatedly incur initialization costs. Within one invocation the fixed
   call/template is reused, while current observations update the causal tracker.

One measured inventory call plus one steady act is approximately **0.65–0.67 s**.
This explains how the reported magnitude can arise if inventory is included per
step, but does not prove that the destination does this. First-call cost and
different CPU/hardware or a modified receiving-side adapter are other unresolved
possibilities.

The recording metadata describes 20-Hz commands (50-ms spacing) and 30-Hz paired
records. The present public steady path is only about 7 Hz on this scene/server,
so even its warm result does not demonstrate 20-Hz end-to-end readiness. Earlier
deployment checks established byte-preserving export, model loading and interface
operation only. They did not certify real-time operation or actual controller
timing. Reducing model file size did not establish a latency target.

The immediate usage rule is to keep a loaded `PushPolicy` and reuse its fixed
call through that invocation; obtain fresh inventory/templates for a new
instance/goal or recovery when required by CALLING.md. Optimization should first
be evaluated in transport/template reuse and redundant preprocessing, with
preserved observation and policy semantics. No such optimization was applied
in this diagnostic turn, and no faster-runtime claim is made.

Reproducible script: [profile_push_latency.py](profile_push_latency.py).
Raw receipts: [public IPC timing](../runs/push_letters/latency_v1/ipc/result.json),
[stage timing](../runs/push_letters/latency_v1/stages/result.json).
