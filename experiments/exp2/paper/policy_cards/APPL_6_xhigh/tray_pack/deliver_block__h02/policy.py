import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


DEFAULT_FIXTURES = [
    [-0.12, 0.0, 0.008, 0.13, 0.17, 0.008],
    [-0.12, -0.178, 0.045, 0.138, 0.008, 0.045],
    [-0.12, 0.178, 0.045, 0.138, 0.008, 0.045],
    [-0.258, 0.0, 0.045, 0.008, 0.17, 0.045],
    [0.018, 0.0, 0.045, 0.008, 0.17, 0.045]]


def unit(q):
    return F.normalize(q, dim=-1, eps=1e-6)


def rotation(q):
    w, x, y, z = unit(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def poses(raw):
    return torch.stack((raw[..., 18:25], raw[..., 25:32], raw[..., 32:39]), -2)


def masked_mean(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1)


class Rollout(nn.Module):
    """A proposed action chunk predicts all three world poses, not joint IK."""
    def __init__(self, feature_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.layers = nn.ModuleList([
            nn.Linear(feature_dim + horizon * 8, 384),
            nn.Linear(384, 384), nn.Linear(384, horizon * 21)])
        nn.init.normal_(self.layers[-1].weight, std=0.001)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, actions, anchor, position_scale, frozen=False):
        x = torch.cat((features, actions.flatten(1)), -1)
        for i, layer in enumerate(self.layers):
            weight = layer.weight.detach() if frozen else layer.weight
            bias = layer.bias.detach() if frozen else layer.bias
            x = F.linear(x, weight, bias)
            if i < len(self.layers)-1:
                x = F.silu(x)
        delta = x.reshape(x.shape[0], self.horizon, 3, 7)
        position = anchor[:, None, :, :3] + delta[..., :3] * position_scale
        quaternion = unit(anchor[:, None, :, 3:] + delta[..., 3:])
        return torch.cat((position, quaternion), -1)


class RimPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        config = spec.get('candidate_config', {})
        self.horizon = spec['training']['horizon']
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('position_scale', torch.tensor(
            [n['std'][18:21], n['std'][25:28], n['std'][32:35]], dtype=torch.float32))
        fixtures = config.get('fixtures', DEFAULT_FIXTURES)
        self.register_buffer('fixtures', torch.tensor(fixtures, dtype=torch.float32))
        corners = [[x, y, z] for x in [-0.02, 0.02] for y in [-0.02, 0.02] for z in [-0.02, 0.02]]
        self.register_buffer('local_corners', torch.tensor(corners, dtype=torch.float32))
        self.margin = float(config.get('wall_margin_m', 0.003))
        # 94 observations + 47 temporal difference + 30 relations + 60 box relations.
        feature_dim = 231
        self.encoder = nn.Sequential(nn.Linear(feature_dim, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU())
        # Per object: attached, free lateral carry, low insertion, settled; plus empty retreat.
        self.mode_head = nn.Sequential(nn.Linear(256, 192), nn.SiLU(),
                                       nn.Linear(192, self.horizon * 9))
        self.mode_encoder = nn.Sequential(nn.Linear(self.horizon * 9, 64), nn.SiLU())
        self.denoiser = DiffusionBackbone(320, spec['training'])
        self.rollouts = nn.ModuleList([Rollout(feature_dim, self.horizon) for _ in range(2)])

    def features(self, raw):
        norm = (raw - self.obs_mean) / self.obs_scale
        scale = self.obs_scale[18:21]
        tcp = raw[..., 18:21]
        red = raw[..., 25:28]
        blue = raw[..., 32:35]
        relatives = torch.cat(((red-tcp)/scale, (blue-tcp)/scale,
                               (red-raw[..., 41:44])/scale,
                               (blue-raw[..., 44:47])/scale,
                               (red-blue)/scale), -1).flatten(1)
        centers = self.fixtures[:, :3]
        half = self.fixtures[:, 3:]
        b = raw.shape[0]
        fixture_features = torch.cat((
            ((centers[None]-tcp[:, -1, None])/scale).flatten(1),
            (half/scale).flatten()[None].expand(b, -1),
            ((centers[None]-red[:, -1, None])/scale).flatten(1),
            ((centers[None]-blue[:, -1, None])/scale).flatten(1)), -1)
        return torch.cat((norm.flatten(1), norm[:, 1]-norm[:, 0], relatives, fixture_features), -1)

    def predict_epsilon(self, noisy_action, timestep, raw):
        f = self.features(raw)
        c = self.encoder(f)
        logits = self.mode_head(c).reshape(-1, self.horizon, 9)
        cond = torch.cat((c, self.mode_encoder(logits.sigmoid().flatten(1))), -1)
        return self.denoiser(noisy_action, timestep, cond), logits, f

    def forward(self, noisy_action, timestep, raw_history):
        return self.predict_epsilon(noisy_action, timestep, raw_history)[0]

    def corners(self, p):
        r = rotation(p[..., 3:])
        return torch.einsum('...ij,kj->...ki', r, self.local_corners) + p[..., None, :3]

    def mode_targets(self, raw, future):
        # Weak geometric/contact proxies are labels only, never observed contacts.
        p = poses(future)
        obj = p[..., 1:, :3]
        tcp = p[..., 0, :3]
        goals = torch.stack((future[..., 41:44], future[..., 44:47]), -2)
        width = future[..., 7] + future[..., 8]
        distance = (obj-tcp.unsqueeze(-2)).square().sum(-1).sqrt()
        closed = torch.sigmoid((0.055-width)/0.004)
        attached = closed.unsqueeze(-1) * torch.sigmoid((0.035-distance)/0.004)
        previous = torch.cat((poses(raw[:, -1])[:, None, 1:, :3], obj[:, :-1]), 1)
        delta = obj-previous
        lateral = delta[..., :2].square().sum(-1).add(1e-12).sqrt()
        goal_xy = (obj[..., :2]-goals[..., :2]).square().sum(-1).add(1e-12).sqrt()
        aligned = torch.sigmoid((0.035-goal_xy)/0.006)
        free = attached * (1-aligned) * torch.sigmoid((lateral-0.0004)/0.00015)
        insert = attached * aligned * torch.sigmoid((0.080-obj[..., 2])/0.012)
        goal_z = torch.sigmoid((0.011-(obj[..., 2]-goals[..., 2]).abs())/0.002)
        open_hand = torch.sigmoid((width-0.060)/0.004)
        separate = torch.sigmoid((distance-0.05)/0.008)
        settled = aligned * goal_z * (1-(1-open_hand.unsqueeze(-1))*(1-separate))
        retreat = open_hand * settled.amax(-1) * torch.sigmoid((0.13-tcp[..., 2])/0.02)
        return torch.cat((torch.stack((attached, free, insert, settled), -1).flatten(-2), retreat.unsqueeze(-1)), -1).detach()

    def energy(self, predicted, raw, modes, mask):
        """Differentiable conservative cube-envelope risk, not arm collision checking."""
        obj = predicted[:, :, 1:]
        cor = self.corners(obj)
        center = obj[..., :3]
        extent = (cor-center.unsqueeze(-2)).abs().amax(-2)
        boxes = self.fixtures
        # AABB envelope of a rotated cube against each supplied fixture.
        sep = (center.unsqueeze(-2)-boxes[None, None, None, :, :3]).abs() - (
            extent.unsqueeze(-2)+boxes[None, None, None, :, 3:])
        separation = sep.amax(-1)
        margins = torch.cat((boxes.new_zeros(1), boxes.new_full((4,), self.margin)))
        collision = (F.relu(margins-separation)/0.02).square().mean(-1)
        pair_sep = ((center[:, :, 0]-center[:, :, 1]).abs() - extent.sum(-2)).amax(-1)
        pair_cost = (F.relu(self.margin-pair_sep)/0.02).square()
        m = modes[..., :8].reshape(modes.shape[0], self.horizon, 2, 4)
        attached, free, insert, settled = m.unbind(-1)
        # Floor contact has zero buffer; intentionally supported cubes are permitted.
        object_collision = collision * (0.2+0.8*attached)
        wall_top = (boxes[1:, 2]+boxes[1:, 5]).amax()
        bottom = cor[..., 2].amin(-1)
        carry_clearance = free * (1-insert) * (F.relu(wall_top+self.margin-bottom)/0.05).square()
        prior_center = torch.cat((poses(raw[:, -1])[:, None, 1:, :3], center[:, :-1]), 1)
        delta = center-prior_center
        goal = torch.stack((raw[:, -1, 41:44], raw[:, -1, 44:47]), -2)[:, None]
        xy_excess = F.relu((cor[..., :2]-goal[..., None, :2]).abs()-0.06)
        planar_error = (xy_excess/0.04).square().mean((-1, -2))
        wrong_descent = free * (F.relu(-delta[..., 2])/0.01).square() * planar_error
        # Do not turn every 16-step horizon into an immediate placement deadline.
        z_error = (F.relu((center[..., 2]-goal[..., 2]).abs()-0.011)/0.05).square()
        terminal = (insert+settled).clamp(max=1) * (planar_error+z_error)
        slots = torch.arange(self.horizon, device=mask.device)[None]
        last = torch.where(mask[..., 0]>0, slots, torch.zeros_like(slots)).amax(-1)
        has_future = (mask[..., 0].sum(-1)>0).to(predicted.dtype)
        terminal_end = terminal.mean(-1).gather(1, last[:, None]).squeeze(1)*has_future
        # Preserve objects that are already settled in the measured causal state.
        now_obj = poses(raw[:, -1])[:, 1:, :3]
        now_goal = goal[:, 0]
        initially_settled = ((now_obj[..., :2]-now_goal[..., :2]).abs().amax(-1)<0.035) & ((now_obj[..., 2]-now_goal[..., 2]).abs()<0.008)
        preserve = (((center-now_obj[:, None])/0.02).square().mean(-1) * initially_settled[:, None]).mean(-1)
        tcp = predicted[:, :, 0, :3]
        tcp_prev = torch.cat((raw[:, -1:, 18:21], tcp[:, :-1]), 1)
        lateral_tcp = ((tcp-tcp_prev)[..., :2]/0.01).square().sum(-1)
        retreat = modes[..., 8] * lateral_tcp * (F.relu(wall_top+0.025-tcp[..., 2])/0.05).square()
        step_energy = object_collision.mean(-1) + pair_cost + carry_clearance.mean(-1) + wrong_descent.mean(-1) + preserve + 0.25*retreat
        weighted = (step_energy*mask[..., 0]).sum(-1)/mask[..., 0].sum(-1).clamp_min(1)
        return weighted + 0.25*terminal_end


def build_model(spec):
    return RimPolicy(spec)


def pose_error(pred, target, scale):
    position = ((pred[..., :3]-target[..., :3])/scale).square().mean((-1, -2))
    # q and -q represent the same orientation; use a chordal distance on the sphere.
    dot = (unit(pred[..., 3:])*unit(target[..., 3:])).sum(-1)
    orientation = (1-dot.square().clamp(max=1)).mean(-1)
    return position + 0.2*orientation


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    eps, logits, features = model.predict_epsilon(batch['noisy_action'], batch['timesteps'], raw)
    diffusion = epsilon_loss(eps, batch['noise'], batch['mask'])
    valid = batch['future_mask'] * batch['mask']
    target = poses(batch['future_obs'])
    anchor = poses(raw[:, -1])
    scale = model.position_scale
    mode_labels = model.mode_targets(raw, batch['future_obs'])
    mode_loss = masked_mean(F.binary_cross_entropy_with_logits(logits, mode_labels, reduction='none'), valid)
    supervised = []
    true_predictions = []
    fit_errors = []
    for rollout in model.rollouts:
        p = rollout(features, batch['encoded_action'], anchor, scale)
        true_predictions.append(p)
        supervised.append(masked_mean(pose_error(p, target, scale), valid[..., 0]))
        delta_error = (((p[:, 1:, :, :3]-p[:, :-1, :, :3])-(target[:, 1:, :, :3]-target[:, :-1, :, :3]))/scale).square()
        supervised.append(0.5*masked_mean(delta_error, valid[:, 1:]*valid[:, :-1]))
        fit_errors.append(masked_mean_per_sample((p[..., :3]-target[..., :3]).square().mean((-1, -2)), valid[..., 0]))
    rollout_loss = torch.stack(supervised).sum()/len(model.rollouts)
    alpha = batch['alpha_bar'].reshape(-1, 1, 1)
    clean = (batch['noisy_action']-(1-alpha).sqrt()*eps)/alpha.sqrt().clamp_min(1e-4)
    # This is the same bounded normalized action space used by the fixed sampler.
    clean = clean.clamp(-1, 1)
    mode_weights = logits.sigmoid().detach()
    candidates = [r(features.detach(), clean, anchor, scale, frozen=True) for r in model.rollouts]
    disagreement = (candidates[0][..., :3]-candidates[1][..., :3]).square().mean((-1, -2))
    disagreement = masked_mean_per_sample(disagreement, valid[..., 0])
    fit = torch.stack(fit_errors).mean(0)
    confidence = torch.exp(-fit.detach()/(0.020**2)-disagreement.detach()/(0.015**2))
    noise_gate = ((alpha[:, 0, 0]>0.2).to(alpha.dtype)*alpha[:, 0, 0]).detach()
    weight = confidence*noise_gate
    consistency = torch.stack([masked_mean_per_sample(pose_error(p, target, scale), valid[..., 0]) for p in candidates]).mean(0)
    candidate_energy = torch.stack([model.energy(p, raw, mode_weights, valid) for p in candidates]).mean(0)
    # The reference absorbs conservative proxy cost on demonstrated contact paths.
    reference_energy = torch.stack([model.energy(p.detach(), raw, mode_weights, valid) for p in true_predictions]).mean(0).detach()
    preference = F.relu(candidate_energy-reference_energy-0.01)
    geometry_loss = (weight*preference).mean()
    clean_pose_loss = (weight*consistency).mean()
    prior = 0.25*rollout_loss + 0.05*mode_loss + 0.05*clean_pose_loss + 0.03*geometry_loss
    return {'loss': diffusion+prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'rollout_loss': rollout_loss, 'mode_loss': mode_loss,
            'geometry_preference_loss': geometry_loss, 'clean_pose_loss': clean_pose_loss}


def masked_mean_per_sample(value, mask):
    return (value*mask).sum(-1)/mask.sum(-1).clamp_min(1)
