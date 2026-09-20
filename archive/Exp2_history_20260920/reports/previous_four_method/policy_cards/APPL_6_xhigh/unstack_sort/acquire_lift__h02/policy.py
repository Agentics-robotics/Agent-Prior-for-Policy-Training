import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def quaternion_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def relative_quaternion(tcp, obj):
    tcp = nn.functional.normalize(tcp, dim=-1, eps=1e-6)
    obj = nn.functional.normalize(obj, dim=-1, eps=1e-6)
    conjugate = torch.cat((tcp[..., :1], -tcp[..., 1:]), dim=-1)
    q = quaternion_product(conjugate, obj)
    # Choose a consistent representative using the largest component, not w,
    # which is near zero for the demonstrated TCP-to-block rotations.
    largest = q.gather(-1, q.abs().argmax(dim=-1, keepdim=True))
    return q * torch.where(largest < 0, -torch.ones_like(largest), torch.ones_like(largest))


def masked_mean_square(error, mask):
    return (error.square() * mask).sum() / (mask.sum() * error.shape[-1]).clamp_min(1.0)


class ContactBeliefDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scales = torch.tensor(n['std'], dtype=torch.float32)
        # A conservative common world-position scale, from the full-demo normalizer.
        self.register_buffer('position_scale', torch.stack((scales[18:21], scales[25:28], scales[32:35])).amax(0))
        self.horizon = int(spec['training']['horizon'])
        self.frame_encoder = nn.Sequential(nn.Linear(135, 160), nn.SiLU(),
                                           nn.Linear(160, 160), nn.LayerNorm(160), nn.SiLU())
        self.recurrent = nn.GRUCell(160, 160)
        self.emission = nn.Sequential(nn.Linear(160, 96), nn.SiLU(), nn.Linear(96, 6))
        # Physical object-indexed modes: empty/approach, red closing, red held,
        # red releasing, blue closing, blue held. All transitions remain possible.
        allowed = torch.tensor([[1,1,0,1,1,0],
                                [1,1,1,0,0,0],
                                [0,1,1,1,0,0],
                                [1,0,1,1,0,0],
                                [1,0,0,0,1,1],
                                [1,0,0,0,1,1]], dtype=torch.float32)
        self.register_buffer('allowed_transition', allowed)
        initial = -3.0 * torch.ones(6, 6) + 2.5 * allowed + 3.5 * torch.eye(6)
        self.transition_logits = nn.Parameter(initial)
        self.mode_embedding = nn.Parameter(torch.randn(6, 32) * 0.03)
        self.condition_encoder = nn.Sequential(nn.Linear(425, 256), nn.SiLU(),
                                                nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU())
        self.backbone = DiffusionBackbone(256, spec['training'])
        # Prediction from causal belief only: no action or future-state teacher input.
        self.forecaster = nn.Sequential(nn.Linear(256, 256), nn.SiLU(),
                                         nn.Linear(256, self.horizon * 19))

    def frame_features(self, raw):
        norm = (raw - self.obs_mean) / self.obs_scale
        tcp = raw[..., 18:21]
        red = raw[..., 25:28]
        blue = raw[..., 32:35]
        relative = torch.cat(((red-tcp)/self.position_scale,
                              (blue-tcp)/self.position_scale), dim=-1)
        goals = torch.cat(((raw[..., 41:44]-red)/self.position_scale,
                           (raw[..., 44:47]-blue)/self.position_scale), dim=-1)
        rotation = torch.cat((relative_quaternion(raw[..., 21:25], raw[..., 28:32]),
                              relative_quaternion(raw[..., 21:25], raw[..., 35:39])), dim=-1)
        return torch.cat((norm, relative, goals, rotation), dim=-1)

    def encode(self, history):
        features = self.frame_features(history)
        delta = features[:, 1] - features[:, 0]
        # No history-validity flag is supplied. Exact duplicate padding is only
        # marked as uninformative motion; stationary real observations are valid.
        informative = ((history[:, 1]-history[:, 0]).abs().sum(-1, keepdim=True) > 1e-8).to(history.dtype)
        first = torch.cat((features[:, 0], torch.zeros_like(delta), torch.zeros_like(informative)), -1)
        second = torch.cat((features[:, 1], delta, informative), -1)
        h0 = self.recurrent(self.frame_encoder(first), torch.zeros(history.shape[0], 160, device=history.device, dtype=history.dtype))
        h1 = self.recurrent(self.frame_encoder(second), h0)
        p0 = self.emission(h0).softmax(-1)
        transition = self.transition_logits.softmax(-1)
        predicted_prior = p0 @ transition
        p1 = (self.emission(h1) + 0.35 * predicted_prior.clamp_min(1e-6).log()).softmax(-1)
        condition = self.condition_encoder(torch.cat((h1, h1-h0, features[:, 1], p1, p1 @ self.mode_embedding), -1))
        return condition, torch.stack((p0, p1), dim=1), transition

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _ = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def build_model(spec):
    return ContactBeliefDiffusion(spec)


def soft_phase_target(raw, future, future_mask, command, command_mask):
    # This function is called only under no_grad in compute_loss. Metre-valued
    # bandwidths express weak-label uncertainty, not deployment contact tests.
    w = future_mask[..., 0]
    count = w.sum(1).clamp_min(1.0)
    tcp = raw[:, 18:21]
    objects = torch.stack((raw[:, 25:28], raw[:, 32:35]), dim=1)
    future_objects = torch.stack((future[..., 25:28], future[..., 32:35]), dim=2)
    difference = objects - tcp[:, None, :]
    near = torch.exp(-difference.square().sum(-1) / (2.0 * 0.022**2))
    width = raw[:, 7:9].sum(-1)
    pinched = torch.sigmoid((0.046-width)/0.004)
    close_command = ((1.0-command)/2.0).clamp(0, 1)
    close_command = command_mask*close_command + (1-command_mask)*pinched
    future_width_change = ((future[..., 7:9].sum(-1)-width[:, None])*w).sum(1)/count
    opening = 1-(1-close_command.neg().add(1))*(1-torch.sigmoid((future_width_change-0.003)/0.002))
    # Equivalently: probability of current open command OR future finger opening.
    displacement = future_objects - objects[:, None, :, :]
    average_dz = (displacement[..., 2]*w[:, :, None]).sum(1)/count[:, None]
    current_relative = objects-tcp[:, None, :]
    future_relative = future_objects-future[..., 18:21].unsqueeze(2)
    relative_error = future_relative-current_relative[:, None, :, :]
    rms = ((relative_error.square().sum(-1)*w[:, :, None]).sum(1)/count[:, None]+1e-12).sqrt()
    persistent = torch.exp(-rms/0.010)
    upward = torch.sigmoid((average_dz-0.003)/0.003)
    elevation_threshold = raw.new_tensor([0.080, 0.040])
    elevated = torch.sigmoid((objects[..., 2]-elevation_threshold)/0.008)
    lift_evidence = 1-(1-upward)*(1-elevated)
    attached_fraction = pinched[:, None]*persistent*(0.15+0.85*lift_evidence)
    attached = near*attached_fraction
    red_goal = raw[:, 41:44]
    red_low_at_goal = torch.exp(-(objects[:, 0, :2]-red_goal[:, :2]).square().sum(-1)/(2*0.05**2))
    red_low_at_goal = red_low_at_goal*torch.sigmoid((red_goal[:, 2]+0.025-objects[:, 0, 2])/0.008)
    release_fraction = red_low_at_goal*(0.08+0.92*opening)
    release = 2.0*near[:, 0]*release_fraction
    closing = near*(0.06+0.94*close_command[:, None])*(1-0.95*attached_fraction)
    red_closing = closing[:, 0]*(1-0.95*red_low_at_goal)
    red_held = attached[:, 0]*(1-0.90*release_fraction)
    maximum_near = near.amax(-1)
    empty = (0.10+(1-pinched)*(0.90-0.75*close_command*maximum_near)+0.60*(1-maximum_near))
    empty = empty*(1-0.7*near[:, 0]*release_fraction)
    scores = torch.stack((empty, red_closing, red_held, release, closing[:, 1], attached[:, 1]), dim=-1)
    scores = scores+0.015
    return scores/scores.sum(-1, keepdim=True)


def future_targets(model, history, future):
    current = history[:, -1]
    fingers = (future[..., 7:9]-model.obs_mean[7:9])/model.obs_scale[7:9]
    tcp = (future[..., 18:21]-current[:, None, 18:21])/model.position_scale
    red = (future[..., 25:28]-current[:, None, 25:28])/model.position_scale
    blue = (future[..., 32:35]-current[:, None, 32:35])/model.position_scale
    rotations = []
    for start in (28, 35):
        q0 = relative_quaternion(current[:, 21:25], current[:, start:start+4])
        q1 = relative_quaternion(future[..., 21:25], future[..., start:start+4])
        dot = (q1*q0[:, None]).sum(-1, keepdim=True)
        q1 = q1*torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
        rotations.append(q1-q0[:, None])
    return torch.cat((fingers, tcp, red, blue, rotations[0], rotations[1]), dim=-1)


def compute_loss(model, batch, spec):
    condition, belief, transition = model.encode(batch['raw_obs'])
    predicted_noise = model.backbone(batch['noisy_action'], batch['timesteps'], condition)
    diffusion = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    with torch.no_grad():
        y0 = soft_phase_target(batch['raw_obs'][:, 0], batch['future_obs'][:, :8],
                               batch['future_mask'][:, :8], batch['native_action'][:, 0, 7], batch['mask'][:, 0, 0])
        y1 = soft_phase_target(batch['raw_obs'][:, 1], batch['future_obs'][:, 1:9],
                               batch['future_mask'][:, 1:9], batch['native_action'][:, 1, 7], batch['mask'][:, 1, 0])
        labels = torch.stack((y0, y1), 1)
        outcome_target = future_targets(model, batch['raw_obs'], batch['future_obs'])
        # Slot zero is the current state after the t-1 action, not a future outcome.
        future_mask = batch['future_mask'].clone()
        future_mask[:, 0] = 0
        label_weights = batch['mask'][:, :2, 0]
        phase_stability = (1-0.5*(y1-y0).abs().sum(-1)).clamp(0, 1)
        pair_mask = label_weights[:, 0]*label_weights[:, 1]
    phase_ce = -(labels*belief.clamp_min(1e-6).log()).sum(-1)
    phase_loss = (phase_ce*label_weights).sum()/label_weights.sum().clamp_min(1)
    temporal_loss = ((belief[:, 1]-belief[:, 0]).square().sum(-1)*phase_stability*pair_mask).sum()/pair_mask.sum().clamp_min(1)
    transition_loss = (transition*(1-model.allowed_transition)).sum()/6.0
    forecast = model.forecaster(condition).reshape(-1, model.horizon, 19)
    finger_loss = masked_mean_square(forecast[..., :2]-outcome_target[..., :2], future_mask)
    motion_loss = masked_mean_square(forecast[..., 2:11]-outcome_target[..., 2:11], future_mask)
    relative_pred = torch.cat((forecast[..., 5:8]-forecast[..., 2:5], forecast[..., 8:11]-forecast[..., 2:5]), -1)
    relative_target = torch.cat((outcome_target[..., 5:8]-outcome_target[..., 2:5], outcome_target[..., 8:11]-outcome_target[..., 2:5]), -1)
    relative_loss = masked_mean_square(relative_pred-relative_target, future_mask)
    rotation_loss = masked_mean_square(forecast[..., 11:19]-outcome_target[..., 11:19], future_mask)
    prior = (0.12*phase_loss + 0.01*temporal_loss + 0.002*transition_loss
             + 0.10*finger_loss + 0.10*motion_loss + 0.08*relative_loss + 0.02*rotation_loss)
    return {'loss': diffusion+prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'phase_loss': phase_loss, 'temporal_loss': temporal_loss,
            'transition_loss': transition_loss, 'finger_loss': finger_loss,
            'motion_loss': motion_loss, 'relative_loss': relative_loss, 'rotation_loss': rotation_loss}
