"""Exact public B0 identity interface for the synthetic smoke fixture."""

from experiment1.contracts import CandidateDesign


class IdentityDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(
            self.common_spec["observation_schema"]["raw_dim"]
        )

    def condition(self, causal_history, causal_context):
        return causal_history

    # Inherit the public identity action encoding/decoding, direct denoising,
    # empty training targets, no auxiliary loss, and empty deployment state.


def build_design(common_spec, config):
    return IdentityDesign(common_spec, config)
