# Implement and train callable policies from demonstrations

You own the learning design and its implementation. Use the supplied task facts,
authorized recordings, execution capabilities and previous API design to produce
trainable policies with useful interfaces for the high-level Agent.

Choose the data processing, training examples, supervision, representations,
policy inventory, models, losses, optimization, stopping and checkpoint selection.
Your prior can determine the model's input representation and output form, for
example contact point movement, with the corresponding robot motion generated
by the supplied planner/controller when its capabilities support that interface.
Implement the supplied cut and prior as the starting design. Refine their sampling, windows,
grouping or learning choices when the evidence warrants it, recording the changes
in your derived data plan while preserving the original artifacts. Distinguish
when the Agent invokes a policy from how many training examples the recordings
can support.

Complete all learned components required by the implemented policy, including
auxiliary perception where needed. Use the available evidence and capabilities
to resolve proposed dependencies and labels, documenting any necessary adaptation
of the prior. Explain any remaining prerequisite that prevents completion.

Read the assignment and executable interface, then inspect the recordings and
implement the complete preparation-to-inference path. Relate the task's required deployment variations to the independent experience
in the demonstrations. Choose compact sufficient inputs and outputs, preserving
the physical and temporal information needed for the interaction. Justify each
learned quantity in terms of the task-dependent information still needed after
accounting for the Agent and supplied planner/controller capabilities. Use
object/tool/goal relationships and physical constraints where supported. For
small datasets, choose priors and representations that make effective use of the
available evidence.
Diffusion Policy is an encouraged option; choose another model when your analysis
supports it and explain the choice. Policy count follows the learning needs.

Run preparation checks and inspect the actual data coverage, target quality and
model interface results. Decide how to use the supported evidence, how to address
lost coverage, and whether the prepared data justifies training. Account for
excluded evidence and uncertainty. Record your readiness assessment against the
actual preparation result before submitting. If the evidence is insufficient,
report the specific missing requirement and its effect on the proposed system.

Implement the inference preprocessing and calling contract used by the Agent,
including observations, goals, outputs, frames, units, memory, feedback and handoff.
Inputs used online must be obtainable at invocation time; future-derived training
targets and declared training-only information must stay on their documented side
of that boundary. Include executable calling checks and document their scope.

The framework executes your code within the supplied resource and capability
limits, saves provenance and diagnostics, and returns implementation failures for
bounded repair. Submit only your exact checked source. Report measured training
and interface results separately from unmeasured deployment/generalization claims.
