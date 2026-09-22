# Preserved source design: Push cut_v6

The **exact supplied source_plan**, including all original prose, 47 anchors, 12 cuts, goal-witness mappings and the two executor traces, is copied unchanged to the preparation artifact `original_source_plan.json`. The framework also retains the original assignment. This file is an index, not a rewritten substitute for that artifact. Every version of implementation files is journaled. No flip-egg cut/data was provided in this assignment, so none is synthesized.

Original model inventory: one new learned `shape_push_v1` candidate scorer; proposed frozen SAM 2.1 Hiera-small and Qwen2.5-VL-7B-Instruct; deterministic geometry plus supplied planner/controller. Original 47 sparse anchors:

A: 500,800,1100,1600; 1800,2100,2400,2800; 3500,4300,4800; 5100,5600,6000; 6200,6700,6900,7200,7600,7800; 8250,8500,8800.

B: 300,650,1000,1300; 1450,1700,1950,2300,2600; 2850,3250,3600,3900,4200; 4550,4800; 5100,5500,5850; 6200,6500,6850,7200,7500.

Original future witnesses are retained in `annotations.json`. Original learned intervals are unchanged in package.data_plan. Whole A[0,9015) and B[0,7730) remain executor/context evidence, zero motor loss. The raw images and arrays are never overwritten or re-paired. Read_steps inspection included first/last included row of all 12 cuts and both executor traces.

Original conceptual prior: full shape including holes; goal-relative geometry without letter identity; one short contact/direction/length decision; planner-generated approach and guarded stroke; fresh observation and error check after each stroke. Calibration, masks, identity, contact labels, control binding and arbitrary-shape success were explicitly not established in the source design.

Implementation changes and their rationale are in package.data_plan and PRIOR.md, not retroactive edits to this source inventory.
