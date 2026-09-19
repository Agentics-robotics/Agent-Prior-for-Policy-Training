"""Developer-authored vanilla DP baseline, with no API prior or auxiliary loss."""
import torch
from appl.public import DiffusionBackbone,normalize_observation,epsilon_loss


class Policy(torch.nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec
        self.backbone=DiffusionBackbone(spec['observation_dimension']*spec['training']['observation_steps'],spec['training'])

    def forward(self,noisy_action,timestep,raw_history):
        condition=normalize_observation(raw_history,self.spec).flatten(start_dim=1)
        return self.backbone(noisy_action,timestep,condition)


def build_model(spec):return Policy(spec)


def compute_loss(model,batch,spec):
    predicted=model(batch['noisy_action'],batch['timesteps'],batch['raw_obs'])
    loss=epsilon_loss(predicted,batch['noise'],batch['mask'])
    return dict(loss=loss,diffusion_loss=loss,prior_loss=loss.new_zeros(()))
