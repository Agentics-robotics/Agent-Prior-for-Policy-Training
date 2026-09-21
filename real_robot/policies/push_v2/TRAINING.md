# Actual training coverage — training_v2

Developer-authored summary of execution receipts. Policy designs, priors and calling contracts remain exact API documents in source/.

Each model completed 20,000 independent optimizer updates (seed 0). The exported tensors are exactly the final EMA, without quantization or performance-based selection.

| Policy | Parameters | Eligible examples | Updates |
| --- | ---: | ---: | ---: |
| contour_push | 3846158 | 6314 | 20000 |
| visual_push | 4331285 | 6314 | 20000 |

The shared prepared dataset contains 6314 examples from 12/22 source cuts. The following counts are actual included supervision, not merely proposed segment support.

| Original cut | Eligible examples |
| --- | ---: |
| a_d_align | 0 |
| a_d_stage | 65 |
| a_h_align | 0 |
| a_h_stage | 246 |
| a_i_align | 0 |
| a_i_stage1 | 0 |
| a_i_stage2 | 500 |
| a_l | 0 |
| a_o | 919 |
| a_r | 465 |
| a_r_stage | 0 |
| a_w | 1336 |
| b_d_align | 0 |
| b_d_stage | 370 |
| b_h_align | 400 |
| b_h_stage | 759 |
| b_i_align | 0 |
| b_i_stage | 160 |
| b_l | 0 |
| b_o | 860 |
| b_r | 234 |
| b_w | 0 |

Zero-count cuts provide no motor training examples in this run. Missing/ambiguous foreground and goal association limit coverage. The two models share these data limitations.

Neither completed training nor interface checks establish robot success, real-time control, recovery/handoff competence or generalization to unseen letters and words. Read each policy's prior/usage and HANDOFF.json together with these coverage limits.

Frozen source package: `7b109fef60fc83744aaa2b73302e87b7fc7995df3a8da327aac275794c605b5a`.
