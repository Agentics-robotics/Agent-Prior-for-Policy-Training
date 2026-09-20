# API-authored prior Diffusion Policies

Source: the retained 5-skill / 15-heuristic API segmentation. Each policy has an independent source package, prior document and checkpoint.

Training completed: 15/15 policies. Independent ID success: 0/5 completed trials (5 planned).

## Policies

| Policy | State | Updates | Parameters | API prior document |
| --- | --- | ---: | ---: | --- |
| open_drawer_by_prismatic_pull__h01 | trained | 20000 | 17942858 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/open_drawer_by_prismatic_pull/heuristic_01/source/PRIOR.md) |
| open_drawer_by_prismatic_pull__h02 | trained | 20000 | 18471580 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/open_drawer_by_prismatic_pull/heuristic_02/source/PRIOR.md) |
| open_drawer_by_prismatic_pull__h03 | trained | 20000 | 18416591 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/open_drawer_by_prismatic_pull/heuristic_03/source/PRIOR.md) |
| grasp_red_from_open_drawer__h01 | trained | 20000 | 18423375 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/grasp_red_from_open_drawer/heuristic_01/source/PRIOR.md) |
| grasp_red_from_open_drawer__h02 | trained | 20000 | 18024536 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/grasp_red_from_open_drawer/heuristic_02/source/PRIOR.md) |
| grasp_red_from_open_drawer__h03 | trained | 20000 | 18683682 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/grasp_red_from_open_drawer/heuristic_03/source/PRIOR.md) |
| place_red_on_outside_pad__h01 | trained | 20000 | 18815288 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/place_red_on_outside_pad/heuristic_01/source/PRIOR.md) |
| place_red_on_outside_pad__h02 | trained | 20000 | 18611414 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/place_red_on_outside_pad/heuristic_02/source/PRIOR.md) |
| place_red_on_outside_pad__h03 | trained | 20000 | 18629560 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/place_red_on_outside_pad/heuristic_03/source/PRIOR.md) |
| grasp_blue_from_table__h01 | trained | 20000 | 18625320 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/grasp_blue_from_table/heuristic_01/source/PRIOR.md) |
| grasp_blue_from_table__h02 | trained | 20000 | 18953216 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/grasp_blue_from_table/heuristic_02/source/PRIOR.md) |
| grasp_blue_from_table__h03 | trained | 20000 | 17907960 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/grasp_blue_from_table/heuristic_03/source/PRIOR.md) |
| place_blue_inside_open_drawer__h01 | trained | 20000 | 18509384 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/place_blue_inside_open_drawer/heuristic_01/source/PRIOR.md) |
| place_blue_inside_open_drawer__h02 | trained | 20000 | 17970707 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/place_blue_inside_open_drawer/heuristic_02/source/PRIOR.md) |
| place_blue_inside_open_drawer__h03 | trained | 20000 | 18668380 | [PRIOR.md](/home/storage/oscar/appl_exp2/runs/prior_policies_20260916/policies/place_blue_inside_open_drawer/heuristic_03/source/PRIOR.md) |

## Inference-time API trials

| Seed | Scope | Success | Drawer | Red | Blue | Steps | Invocations |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| [1000](evaluation/1000/REPORT.md) | training-reset diagnostic | False | True | False | False | 1400 | 8 |
| [6100](evaluation/6100/REPORT.md) | independent ID | False | True | True | False | 1500 | 9 |
| [6101](evaluation/6101/REPORT.md) | independent ID | False | False | True | False | 1500 | 7 |
| [6102](evaluation/6102/REPORT.md) | independent ID | False | True | False | False | 1300 | 6 |
| [6103](evaluation/6103/REPORT.md) | independent ID | False | True | False | False | 1220 | 6 |
| [6104](evaluation/6104/REPORT.md) | independent ID | False | True | False | False | 1340 | 7 |

Independent ID: 0 successes in 5 completed trials; 5 planned.

Success uses only the agreed simultaneous drawer-open, red-on-pad and blue-inside predicates. This does not retroactively change frozen M0 results.

## Accounting

API calls: 287; input tokens: 5,380,969 (cached 4,240,128); output tokens: 241,255; API request seconds: 4952.1.
Completed interface-check updates: 30; synthetic framework fixture updates: 2; no automatic transport retries. Partial failures remain separately recorded. Dollar invoice unavailable.

Authorship audit performed in this report: True. The audit matches policy code, pipeline metadata and PRIOR.md against raw API write_file responses and explicit submission hashes.

[Execution analysis](ANALYSIS.md) · [Machine-readable summary](summary.json) · [Policy table](policies.csv) · [Deletion receipts](cleanup/deletion_receipt.json) · [Orchestration records](orchestration/README.md)
