# Small-demonstration cut prompt revision

Review date: 2026-09-21. The user approved one new numbered experiment, cut_v5,
using the reviewed small-demonstration design objective.

The run completed with two dataset groups, three policy/prior proposals and
26 segments. It made low-dimensional motion priors more explicit while merging
alignment into relocation; it did not create a separate planar-push group.
See the [Chinese result and comparison](PUSH_CUT_V5_REVIEW.md).

The [new prompt](../prompts/general_cut_and_prior_v3.md) adds 128 words before
subtask selection, for a total of 1,905 whitespace-separated words. It asks the
API to identify reusable object/interaction-relative structure, constrained
quantities and minimal learned action variables, then jointly refine cuts,
representations and priors. Compatible buffers and complete behavior coverage
remain required. The rest of v2 is unchanged.

The addition does not prescribe a particular motion class, action dimension,
skill list or policy count. Existing tools, input data, task specification,
model/effort and submission schema remain the same. Previous API designs and
training diagnoses are not included in the new conversation. This is one
prompt revision trial, not evidence of improved learned-policy performance.

- [Exact prompt diff](../runs/push_letters/cut_v5/setup/prompt_derivation.diff)
- [Configuration](../configs/push_cut_v5.json)
- [Execution record](PUSH_CUT_V5_STATUS.md)

The OpenAI Docs skill was used to check prompting guidance. The official
[reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)
recommend direct instructions with explicit objectives; the scientific content
of this revision comes from the user's reviewed requirements. Existing
gpt-6-astra/xhigh request and transport behavior are preserved and independently
checked before execution; no model migration is performed.
