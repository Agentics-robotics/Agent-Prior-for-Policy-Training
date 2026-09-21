# Door initial design v1

Designed by actual isolated subagent `/root/design_door`, before implementation, from the contract and door evidence bundle only. Both NPZ demonstrations were summarized through Pixi and all16real frames were individually viewed. No source/expert/reward/larger-D/development/test data or trained model scores were read.

The demonstrations first rise above the handle, descend, then follow a planar opening curve. Handle motion above1mm starts at step41LL and49HH; minimum hand-handle gaps are0.0253m and0.0271m. Both maintain handle z~0.150036m. The known vertical hinge and supplied cabinet yaw justify planar coordinate features, while the fixed robot/gripper/table rule out assuming whole-scene rotational symmetry.

P1 uses cabinet-relative relation features and cabinet-axis diffusion actions, retaining all raw world observations. P2 keeps world actions and jointly denoises8future geometry channels with the4actions: handle displacement, hand-handle gap and projected door direction. P3 combines these mechanisms, expressing future targets in the same cabinet axes as the actions. The future channels are generated jointly and discarded during action execution; they never control routing, correction or termination.

The combination is adopted as its own candidate because coordinate nuisance and learned interaction consistency address different potential failures. It may hurt under two demonstrations through redundant features or auxiliary denoising interference. Hard phases and analytic tangent controllers are rejected; neither hidden phases nor sufficient failure evidence are available.

All feature scales are fixed as specified in proposal.json with passthrough normalization. There is no data augmentation or extra training schedule. Every candidate uses one default-width U-Net,20,000total updates, batch128total, masked zero-padded tails, and exact shared16-step DDIM. Prediction16/execution4 freezes the replan cabinet axes and world clipping follows inverse action decode. This is prospective design reasoning, not a claim of trained performance.
