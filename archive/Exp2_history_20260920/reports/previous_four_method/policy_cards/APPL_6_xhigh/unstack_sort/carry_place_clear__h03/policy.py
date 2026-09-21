import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(din, hidden, dout):
    return torch.nn.Sequential(torch.nn.Linear(din, hidden), torch.nn.SiLU(),
                               torch.nn.Linear(hidden, dout))


def unit_quaternion(q):
    return q / q.square().sum(-1, keepdim=True).clamp_min(1.0e-12).sqrt()


def rotation_matrix(q):
    w, x, y, z = unit_quaternion(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def linear(layer, x, frozen):
    weight = layer.weight.detach() if frozen else layer.weight
    bias = layer.bias.detach() if frozen else layer.bias
    return torch.nn.functional.linear(x, weight, bias)


class OutcomeMember(torch.nn.Module):
    """Direct, causal prefix-to-future prediction, anchored at the current state."""
    def __init__(self, feature_dim, horizon):
        super().__init__()
        self.c1 = torch.nn.Linear(feature_dim, 128)
        self.c2 = torch.nn.Linear(128, 128)
        self.a1 = torch.nn.Linear(128 + horizon*8 + 3, 256)
        self.a2 = torch.nn.Linear(256, 256)
        self.out = torch.nn.Linear(256, 23)
        torch.nn.init.normal_(self.out.weight, std=0.001)
        torch.nn.init.zeros_(self.out.bias)
        h = torch.arange(horizon)
        prefix = ((h[:, None] >= h[None, :]) & (h[None, :] > 0)).float()
        self.register_buffer('prefix', prefix[:, :, None])
        time = h.float() / max(horizon-1, 1)
        self.register_buffer('times', torch.stack((time, torch.sin(math.pi*time), torch.cos(math.pi*time)), -1))
        self.register_buffer('residual_time', time[None, :, None])

    def forward(self, features, actions, base, frozen=False):
        c = torch.nn.functional.silu(linear(self.c1, features, frozen))
        c = torch.nn.functional.silu(linear(self.c2, c, frozen))
        b, h, _ = actions.shape
        # Slot zero is t-1, whose resulting state is already the current observation.
        # Prediction j uses actions 1..j, never a later action or a future state.
        prefix_actions = (actions[:, None, :, :] * self.prefix[None]).reshape(b, h, -1)
        x = torch.cat((c[:, None, :].expand(-1, h, -1), prefix_actions,
                       self.times[None].expand(b, -1, -1)), -1)
        x = torch.nn.functional.silu(linear(self.a1, x, frozen))
        x = torch.nn.functional.silu(linear(self.a2, x, frozen))
        return base[:, None, :] + self.residual_time * linear(self.out, x, frozen)


class ContractDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.horizon = spec['training']['horizon']
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('obs_index', torch.tensor(list(range(39))+list(range(41, 47)), dtype=torch.long))
        self.register_buffer('out_index', torch.tensor(list(range(18, 39))+[7, 8], dtype=torch.long))
        self.register_buffer('metric_index', torch.tensor([0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22], dtype=torch.long))
        self.register_buffer('corners', torch.tensor([[x, y, z] for x in (-0.020, 0.020)
                                                     for y in (-0.020, 0.020)
                                                     for z in (-0.020, 0.020)], dtype=torch.float32))
        self.register_buffer('tolerances', torch.tensor([0.060, 0.060, 0.011]))
        self.register_buffer('support_scale', torch.tensor([0.020, 0.020, 0.006]))
        self.register_buffer('future_slots', (torch.arange(self.horizon) > 0).float()[None, :])
        # 2 x 45 observed features, 45 differences, 24 relative coordinates, 6 slacks.
        self.encoder = torch.nn.Sequential(torch.nn.Linear(165, 256), torch.nn.SiLU(),
                                            torch.nn.Linear(256, 192), torch.nn.LayerNorm(192), torch.nn.SiLU())
        self.progress = mlp(192, 128, self.horizon*4)
        self.denoiser = DiffusionBackbone(192+self.horizon*4, spec['training'])
        self.outcomes = torch.nn.ModuleList([OutcomeMember(165, self.horizon) for _ in range(3)])
        config = spec.get('candidate_config', {})
        weights = config.get('loss_weights', {})
        self.loss_weights = {
            'dynamics': float(weights.get('dynamics', 0.25)),
            'phase': float(weights.get('phase', 0.05)),
            'outcome_consistency': float(weights.get('outcome_consistency', 0.05)),
            'arrival_energy': float(weights.get('arrival_energy', 0.04)),
            'preservation_energy': float(weights.get('preservation_energy', 0.04)),
            'continuity': float(weights.get('continuity', 0.02))}

    def poses_goals(self, raw):
        poses = torch.stack((raw[..., 25:32], raw[..., 32:39]), -2)
        goals = torch.stack((raw[..., 41:44], raw[..., 44:47]), -2)
        return poses, goals

    def slack(self, poses, goals):
        # Full eight-corner rotated XY projection; z is a CENTER condition.
        rotated = torch.einsum('...ij,kj->...ki', rotation_matrix(poses[..., 3:7]), self.corners)
        offsets = poses[..., None, :3] + rotated - goals[..., None, :]
        worst_xy = offsets[..., :2].abs().amax(dim=-2)
        center_z = (poses[..., 2] - goals[..., 2]).abs().unsqueeze(-1)
        return self.tolerances - torch.cat((worst_xy, center_z), -1)

    def goal_labels(self, raw):
        poses, goals = self.poses_goals(raw)
        return (self.slack(poses, goals) >= 0).all(-1).to(raw.dtype)

    def features(self, raw):
        norm = (raw-self.obs_mean)/self.obs_scale
        kept = norm.index_select(-1, self.obs_index)
        items = [kept.flatten(1), kept[:, -1]-kept[:, -2]]
        for p, g in ((25, 41), (32, 44)):
            items.append(((raw[..., p:p+3]-raw[..., g:g+3])/self.obs_scale[p:p+3]).flatten(1))
            relative_scale = (self.obs_scale[p:p+3].square()+self.obs_scale[18:21].square()).sqrt()
            items.append(((raw[..., p:p+3]-raw[..., 18:21])/relative_scale).flatten(1))
        poses, goals = self.poses_goals(raw[:, -1])
        # Saturate ONLY extra dimensionless contract features, not normalized observations.
        items.append((self.slack(poses, goals)/self.tolerances).clamp(-5, 5).flatten(1))
        return torch.cat(items, -1)

    def predict(self, noisy_action, timestep, raw_history):
        features = self.features(raw_history)
        latent = self.encoder(features)
        logits = self.progress(latent).reshape(-1, self.horizon, 4)
        cond = torch.cat((latent, logits.sigmoid().flatten(1)), -1)
        epsilon = self.denoiser(noisy_action, timestep, cond)
        return epsilon, logits, features

    def forward(self, noisy_action, timestep, raw_history):
        # Deployment consumes only these two causal raw observations.
        return self.predict(noisy_action, timestep, raw_history)[0]

    def base_state(self, raw_history):
        raw = raw_history[:, -1].index_select(-1, self.out_index)
        return (raw-self.obs_mean[self.out_index])/self.obs_scale[self.out_index]

    def rollout(self, features, actions, base, frozen=False):
        return torch.stack([member(features, actions, base, frozen) for member in self.outcomes], 0)

    def decode_outcome(self, encoded):
        return encoded*self.obs_scale[self.out_index]+self.obs_mean[self.out_index]

    def state_error(self, prediction, target):
        # Outputs: TCP pose, red pose, blue pose, two individual finger positions.
        p = prediction.index_select(-1, self.metric_index)
        t = target.index_select(-1, self.metric_index).expand_as(p)
        metric = torch.nn.functional.smooth_l1_loss(p, t, reduction='none').mean(-1)
        ori = torch.zeros_like(metric)
        for start in (3, 10, 17):
            qp = unit_quaternion(prediction[..., start:start+4])
            qt = unit_quaternion(target[..., start:start+4])
            # Sign-invariant supervision of normalized quaternion predictions.
            ori = ori + (1-(qp*qt).sum(-1).square()).clamp_min(0)
        return metric + 0.25*ori/3


def build_model(spec):
    return ContractDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    future = batch['future_obs']
    true_actions = batch['encoded_action']
    eps, phase_logits, features = model.predict(batch['noisy_action'], batch['timesteps'], raw)
    diffusion_loss = epsilon_loss(eps, batch['noise'], batch['mask'])
    valid = batch['future_mask'].squeeze(-1)*batch['mask'].squeeze(-1)*model.future_slots
    base = model.base_state(raw)
    target_raw = future.index_select(-1, model.out_index)
    target = (target_raw-model.obs_mean[model.out_index])/model.obs_scale[model.out_index]
    teacher = model.rollout(features.detach(), true_actions, base)
    # Independent per-window bootstrap masks; no validation/test observations.
    bootstrap = (torch.rand((3, raw.shape[0], 1), device=raw.device) < 0.8).to(raw.dtype)
    dynamics_mask = valid[None]*bootstrap
    dynamics_loss = (model.state_error(teacher, target[None])*dynamics_mask).sum()/dynamics_mask.sum().clamp_min(1)

    future_goal = model.goal_labels(future)
    opened = (future[..., 7:9].amin(-1) >= 0.038).to(raw.dtype)
    open_high = opened*(future[..., 20] >= 0.250).to(raw.dtype)
    phase_target = torch.cat((future_goal, opened[..., None], open_high[..., None]), -1)
    phase_error = torch.nn.functional.binary_cross_entropy_with_logits(phase_logits, phase_target, reduction='none').mean(-1)
    phase_loss = (phase_error*valid).sum()/valid.sum().clamp_min(1)

    a = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1.0e-5)
    x0 = (batch['noisy_action']-(1-a).sqrt()*eps)/a.sqrt()
    clipped_x0 = x0.clamp(-1, 1)
    # Frozen model WEIGHTS, not no_grad: retain d outcome / d denoised action.
    candidate = model.rollout(features.detach(), clipped_x0, base.detach(), frozen=True)
    teacher_world = model.decode_outcome(teacher.detach())
    candidate_world = model.decode_outcome(candidate)
    object_predictions = torch.stack((candidate_world[..., 7:14], candidate_world[..., 14:21]), -2)
    teacher_positions = torch.stack((teacher_world[..., 7:10], teacher_world[..., 14:17]), -2)
    truth_positions = torch.stack((future[..., 25:28], future[..., 32:35]), -2)
    mean_positions = object_predictions[..., :3].mean(0)
    disagreement = ((object_predictions[..., :3].detach()-mean_positions.detach()[None])/model.support_scale).square().mean((0, -1))
    reference_error = ((teacher_positions-truth_positions[None])/model.support_scale).square().mean((0, -1))
    # A detached support proxy, not a calibrated certificate of rollout accuracy.
    confidence = torch.exp(-(reference_error+disagreement).clamp(max=30)).detach()
    # Discourage exploiting the fitted model with very remote denoised actions.
    action_distance = ((clipped_x0.detach()-true_actions).square().mean(-1)*batch['mask'].squeeze(-1)).sum(-1)/batch['mask'].squeeze(-1).sum(-1).clamp_min(1)
    action_support = torch.exp(-action_distance/0.25)
    low_noise = (a[:, 0, 0] >= 0.5).to(raw.dtype)*a[:, 0, 0].square()*action_support
    cell_weight = valid*low_noise[:, None]
    consistency_error = model.state_error(candidate, target[None]).mean(0)
    outcome_consistency = (consistency_error*cell_weight*confidence.mean(-1)).sum()/valid.sum().clamp_min(1)

    _, goals = model.poses_goals(raw[:, -1])
    predicted_slack = model.slack(object_predictions, goals[None, :, None, :, :])
    energy = torch.relu(-predicted_slack/model.tolerances).square().mean(-1)
    # Bounded transform of squared violations; exactly zero everywhere feasible.
    energy = (energy/(1+energy)).mean(0)
    observed_goal = model.goal_labels(raw[:, -1])
    # Labels authorize placement shaping only at demonstrated attainment horizons.
    # A resting blue away from its pad never receives a goal-sliding objective.
    arrival_gate = future_goal*(1-observed_goal[:, None, :])*(0.5+0.5*phase_logits[..., :2].sigmoid().detach())
    preserve_gate = observed_goal[:, None, :].expand_as(future_goal)
    ew = cell_weight[..., None]*confidence
    energy_denom = (valid.sum()*2).clamp_min(1)
    arrival_energy = (energy*ew*arrival_gate).sum()/energy_denom
    preservation_energy = (energy*ew*preserve_gate).sum()/energy_denom

    # Match demonstrated action DIFFERENCES, including release edges, rather than
    # forcing a smooth constant gripper command or an exact Cartesian path.
    pair_mask = batch['mask'][:, 1:]*batch['mask'][:, :-1]
    delta_error = torch.nn.functional.smooth_l1_loss(x0[:, 1:]-x0[:, :-1], true_actions[:, 1:]-true_actions[:, :-1], reduction='none')
    continuity = (delta_error*pair_mask*a.square()).sum()/(pair_mask.sum()*8).clamp_min(1)
    w = model.loss_weights
    prior_loss = (w['dynamics']*dynamics_loss+w['phase']*phase_loss+
                  w['outcome_consistency']*outcome_consistency+w['arrival_energy']*arrival_energy+
                  w['preservation_energy']*preservation_energy+w['continuity']*continuity)
    return {'loss': diffusion_loss+prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'dynamics_loss': dynamics_loss,
            'phase_loss': phase_loss, 'outcome_consistency_loss': outcome_consistency,
            'arrival_energy_loss': arrival_energy, 'preservation_energy_loss': preservation_energy,
            'continuity_loss': continuity}
