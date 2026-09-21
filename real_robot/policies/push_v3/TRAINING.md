# Actual training coverage — training_v3

Developer-authored summary of execution receipts. Policy designs, priors and calling contracts remain exact API documents in source/.

Each model completed 20,000 independent optimizer updates (seed 0). The exported tensors are exactly the final EMA, without quantization or performance-based selection.

| Policy | Parameters | Eligible examples | Updates |
| --- | ---: | ---: | ---: |
| tool_waypoint_v1 | 2005168 | 1362 | 20000 |
| piece_contact_graph_v1 | 2147601 | 15765 | 20000 |
| piece_goal_field_v1 | 2005168 | 15765 | 20000 |

The prepared cache contains 17127 anchors from 26 frozen segments. Tool sampling uses its 4 assigned segments; graph and field independently use the same 22 piece segments. These are correlated rows from two demonstrations, with 382 reused source rows.

| Policy | Original cut | Eligible examples |
| --- | --- | ---: |
| piece_contact_graph_v1 | a_bar_align | 360 |
| piece_contact_graph_v1 | a_bar_clear | 250 |
| piece_contact_graph_v1 | a_bar_stage | 600 |
| piece_contact_graph_v1 | a_d_align | 450 |
| piece_contact_graph_v1 | a_d_stage | 560 |
| piece_contact_graph_v1 | a_l_move | 2030 |
| piece_contact_graph_v1 | a_r_layout | 470 |
| piece_contact_graph_v1 | a_r_stage | 480 |
| piece_contact_graph_v1 | a_ring_move | 1230 |
| piece_contact_graph_v1 | a_upper_bar_align | 505 |
| piece_contact_graph_v1 | a_upper_bar_move | 340 |
| piece_contact_graph_v1 | a_wedge_move | 1140 |
| piece_contact_graph_v1 | b_bar_align | 460 |
| piece_contact_graph_v1 | b_bar_clear | 260 |
| piece_contact_graph_v1 | b_d_align | 250 |
| piece_contact_graph_v1 | b_d_stage | 450 |
| piece_contact_graph_v1 | b_l_move | 1200 |
| piece_contact_graph_v1 | b_r_move | 1850 |
| piece_contact_graph_v1 | b_ring_move | 900 |
| piece_contact_graph_v1 | b_upper_bar_align | 360 |
| piece_contact_graph_v1 | b_upper_bar_move | 760 |
| piece_contact_graph_v1 | b_wedge_move | 860 |
| piece_goal_field_v1 | a_bar_align | 360 |
| piece_goal_field_v1 | a_bar_clear | 250 |
| piece_goal_field_v1 | a_bar_stage | 600 |
| piece_goal_field_v1 | a_d_align | 450 |
| piece_goal_field_v1 | a_d_stage | 560 |
| piece_goal_field_v1 | a_l_move | 2030 |
| piece_goal_field_v1 | a_r_layout | 470 |
| piece_goal_field_v1 | a_r_stage | 480 |
| piece_goal_field_v1 | a_ring_move | 1230 |
| piece_goal_field_v1 | a_upper_bar_align | 505 |
| piece_goal_field_v1 | a_upper_bar_move | 340 |
| piece_goal_field_v1 | a_wedge_move | 1140 |
| piece_goal_field_v1 | b_bar_align | 460 |
| piece_goal_field_v1 | b_bar_clear | 260 |
| piece_goal_field_v1 | b_d_align | 250 |
| piece_goal_field_v1 | b_d_stage | 450 |
| piece_goal_field_v1 | b_l_move | 1200 |
| piece_goal_field_v1 | b_r_move | 1850 |
| piece_goal_field_v1 | b_ring_move | 900 |
| piece_goal_field_v1 | b_upper_bar_align | 360 |
| piece_goal_field_v1 | b_upper_bar_move | 760 |
| piece_goal_field_v1 | b_wedge_move | 860 |
| tool_waypoint_v1 | a_initial_tool | 600 |
| tool_waypoint_v1 | a_l_approach_tool | 131 |
| tool_waypoint_v1 | b_initial_tool | 380 |
| tool_waypoint_v1 | b_r_reset_tool | 251 |

Both piece models have 15,165 automatically accepted object-goal rows; a_bar_stage's 600 rows retain native action supervision with a missing-goal mask and endpoint TCP condition. All perception/endpoint/contact labels are unreviewed weak labels. Tool goals use TCP poses, so object-goal masks do not apply to that model.

All three produce six-dimensional native v/w candidates. They are not hard-constrained 2D push policies. The exact API decoder remains disabled; integrating the recording controller is separate from model inference.

Neither completed training nor interface checks establish robot success, real-time control, recovery/handoff competence or generalization to unseen letters and words. Read each policy's prior/usage and HANDOFF.json together with these coverage limits.

Frozen source package: `9a4aa8a8bbd45aa5b95d1acdd3cdc70f5da9cd64165b331515da2e4d165895a6`.
