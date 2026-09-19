"""Synthetic GPU interface fixture; never used as an experimental policy."""
import torch
from appl.public import epsilon_loss


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(8,8)
        self.condition=torch.nn.Linear(47,8)

    def forward(self,sample,timestep,raw):
        return self.linear(sample)+self.condition(raw[:,-1]).unsqueeze(1)


def build_model(spec):return Fixture()


def compute_loss(model,batch,spec):
    loss=epsilon_loss(model(batch['noisy_action'],batch['timesteps'],batch['raw_obs']),batch['noise'],batch['mask'])
    return dict(loss=loss,diffusion_loss=loss,prior_loss=loss*0)
