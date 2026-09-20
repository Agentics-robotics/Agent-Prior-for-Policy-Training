import torch
from appl.public import DiffusionBackbone, epsilon_loss

nn = torch.nn
F = torch.nn.functional


def mlp(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, dout))


def unit(q):
    return F.normalize(q, dim=-1, eps=1e-6)


def object_poses(obs):
    return torch.stack((obs[..., 25:32], obs[..., 32:39]), dim=-2)


def orientation_error(q, r):
    # Quaternion sign invariant, with zero error for either representative.
    return (1.0 - (unit(q) * unit(r)).sum(-1).square()).clamp_min(0.0)


def role_labels(history):
    # Weak physical labels, not contact measurements. No time/phase index.
    prev, cur = history[:, 0], history[:, 1]
    p = object_poses(cur)[..., :3]
    old = object_poses(prev)[..., :3]
    distance = (p - cur[:, None, 18:21]).norm(dim=-1)
    associated = (distance < 0.045) & (cur[:, 7:9].sum(-1, keepdim=True) < 0.060)
    supported = (p[..., 2] - 0.020).abs() < 0.008
    stable = (p - old).norm(dim=-1) < 0.002
    passive = supported & stable & (~associated)
    labels = torch.full_like(distance, 2, dtype=torch.long)
    labels = torch.where(passive, torch.zeros_like(labels), labels)
    return torch.where(associated, torch.ones_like(labels), labels)


def box_distance(tcp, pose):
    # Signed point-to-oriented-cube distance. This is NOT arm collision checking.
    q = unit(pose[..., 3:7])
    v = tcp.unsqueeze(-2) - pose[..., :3]
    xyz = -q[..., 1:4]
    cross = 2.0 * torch.cross(xyz, v, dim=-1)
    local = v + q[..., :1] * cross + torch.cross(xyz, cross, dim=-1)
    d = local.abs() - 0.020
    return (F.relu(d).square().sum(-1) + 1e-12).sqrt() + d.amax(-1).clamp_max(0.0)


class GraphRound(nn.Module):
    def __init__(self):
        super().__init__()
        self.message = mlp(138, 128, 128)
        self.update = mlp(128, 128, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, nodes, positions, roles, scale):
        # Edges indexed by receiving node i and sending node j.
        target = nodes[:, :, None, :].expand(-1, -1, 3, -1)
        source = nodes[:, None, :, :].expand(-1, 3, -1, -1)
        rt = roles[:, :, None, :].expand(-1, -1, 3, -1)
        rs = roles[:, None, :, :].expand(-1, 3, -1, -1)
        delta = (positions[:, None, :, :] - positions[:, :, None, :]) / scale
        edge = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
        msg = self.message(torch.cat((target, source, edge, rt, rs), -1))
        active_message, passive_message = msg.chunk(2, -1)
        msg = (1.0 - rs[..., :1]) * active_message + rs[..., :1] * passive_message
        off_diagonal = 1.0 - torch.eye(3, device=nodes.device, dtype=nodes.dtype)
        aggregated = (msg * off_diagonal[None, :, :, None]).sum(2) / 2.0
        return self.norm(nodes + self.update(torch.cat((nodes, aggregated), -1)))


class ForwardHead(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.layers = nn.ModuleList((nn.Linear(256 + horizon * 8, 256),
                                     nn.Linear(256, 256), nn.Linear(256, horizon * 17)))
        nn.init.normal_(self.layers[-1].weight, std=0.001)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, condition, action, frozen=False):
        x = torch.cat((condition, action.flatten(1)), -1)
        for k, layer in enumerate(self.layers):
            w = layer.weight.detach() if frozen else layer.weight
            b = layer.bias.detach() if frozen else layer.bias
            x = F.linear(x, w, b)
            if k < 2:
                x = F.silu(x)
        return x.reshape(x.shape[0], self.horizon, 17)


class Policy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        # All relational position features use the complete-demo TCP half ranges.
        # These are shared across colors, not narrow blue-x or per-skill ranges.
        self.register_buffer('geometry_scale', torch.tensor(n['std'][18:21], dtype=torch.float32))
        self.horizon = spec['training']['horizon']
        self.global_encoder = mlp(94, 192, 128)
        self.tcp_encoder = mlp(50, 128, 64)
        self.object_encoder = mlp(36, 128, 64)
        self.role_encoder = mlp(192, 96, 3)
        self.graph = nn.ModuleList((GraphRound(), GraphRound()))
        self.condition_encoder = mlp(326, 256, 256)
        self.denoiser = DiffusionBackbone(256, spec['training'])
        self.dynamics = nn.ModuleList([ForwardHead(self.horizon) for _ in range(3)])

    def encode(self, history):
        normalized = (history - self.obs_mean) / self.obs_scale
        b = history.shape[0]
        global_feature = self.global_encoder(normalized.flatten(1))
        cur, prev = history[:, 1], history[:, 0]
        poses = object_poses(cur)
        previous = object_poses(prev)
        norm_poses = object_poses(normalized).permute(0, 2, 1, 3).reshape(b, 2, 14)
        relative_tcp = (object_poses(history)[..., :3] - history[:, :, None, 18:21]) / self.geometry_scale
        relative_tcp = relative_tcp.permute(0, 2, 1, 3).reshape(b, 2, 6)
        goals = torch.stack((cur[:, 41:44], cur[:, 44:47]), 1)
        goal_relative = (goals - poses[..., :3]) / self.geometry_scale
        other_relative = (poses.flip(1)[..., :3] - poses[..., :3]) / self.geometry_scale
        backward_displacement = (poses[..., :3] - previous[..., :3]) / self.geometry_scale
        fingers = normalized[:, :, 7:9].reshape(b, 1, 4).expand(-1, 2, -1)
        identity = torch.eye(2, device=history.device, dtype=history.dtype)[None].expand(b, -1, -1)
        height = (poses[..., 2:3] - 0.020) / self.geometry_scale[2]
        features = torch.cat((norm_poses, relative_tcp, goal_relative, other_relative,
                              backward_displacement, fingers, identity, height), -1)
        objects = self.object_encoder(features)
        role_logits = self.role_encoder(torch.cat((objects, global_feature[:, None, :].expand(-1, 2, -1)), -1))
        object_roles = role_logits.softmax(-1)
        tcp = self.tcp_encoder(torch.cat((normalized[:, :, :18], normalized[:, :, 18:25]), -1).flatten(1))
        nodes = torch.cat((tcp[:, None, :], objects), 1)
        tcp_role = history.new_tensor([0.0, 1.0, 0.0])[None, None, :].expand(b, 1, -1)
        roles = torch.cat((tcp_role, object_roles), 1)
        positions = torch.cat((cur[:, None, 18:21], poses[..., :3]), 1)
        for layer in self.graph:
            nodes = layer(nodes, positions, roles, self.geometry_scale)
        condition = self.condition_encoder(torch.cat((global_feature, nodes.flatten(1), object_roles.flatten(1)), -1))
        return condition, role_logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, _ = self.encode(raw_history)
        return self.denoiser(noisy_action, timestep, condition)

    def predict_world(self, head, condition, action, history, frozen=False):
        output = head(condition, action, frozen=frozen)
        delta = output[..., :14].reshape(output.shape[0], self.horizon, 2, 7)
        base = object_poses(history[:, 1])[:, None]
        position = base[..., :3] + delta[..., :3] * self.geometry_scale
        quaternion = unit(base[..., 3:7] + 0.1 * delta[..., 3:7])
        tcp = history[:, 1, None, 18:21] + output[..., 14:17] * self.geometry_scale
        return torch.cat((position, quaternion), -1), tcp


def build_model(spec):
    return Policy(spec)


def masked_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    condition, roles = model.encode(history)
    prediction = model.denoiser(batch['noisy_action'], batch['timesteps'], condition)
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    current_roles = role_labels(history)
    role_loss = F.cross_entropy(roles.reshape(-1, 3), current_roles.reshape(-1))
    valid = batch['future_mask'][..., 0] * batch['mask'][..., 0]
    target = object_poses(batch['future_obs'])
    target_tcp = batch['future_obs'][..., 18:21]
    base = object_poses(history[:, 1])[:, None]
    # Train-only interval labels, vetoed by future association, displacement,
    # loss of support or earlier role switch. Labels never enter forward().
    with torch.no_grad():
        future = batch['future_obs']
        width = future[..., 7:9].sum(-1, keepdim=True)
        near = (target[..., :3] - target_tcp.unsqueeze(-2)).norm(dim=-1) < 0.045
        associated = near & (width < 0.060)
        supported = (target[..., 2] - 0.020).abs() < 0.008
        unmoved = (target[..., :3] - base[..., :3]).norm(dim=-1) < 0.003
        unrotated = orientation_error(target[..., 3:7], base[..., 3:7]) < 0.001
        interval = (supported & (~associated) & unmoved & unrotated).to(history.dtype)
        interval = interval.cumprod(dim=1)
        passive = interval * (current_roles == 0)[:, None, :].to(history.dtype)
        passive = passive * valid[..., None]
        baseline_far = (base[:, 0, :, :3] - history[:, 1, None, 18:21]).norm(dim=-1) > 0.050
        clear_label = (box_distance(target_tcp, target) > 0.010) & baseline_far[:, None, :]

    expert_action = batch['encoded_action'] * batch['mask']
    fit_losses, expert_predictions = [], []
    for head in model.dynamics:
        pose, tcp = model.predict_world(head, condition, expert_action, history)
        expert_predictions.append(pose.detach())
        pos_error = ((pose[..., :3] - target[..., :3]) / model.geometry_scale).square().mean(-1)
        rot_error = orientation_error(pose[..., 3:7], target[..., 3:7])
        tcp_error = ((tcp - target_tcp) / model.geometry_scale).square().mean(-1)
        per_slot = (pos_error + 0.25 * rot_error).mean(-1) + tcp_error
        # Independent bootstrap subset, retaining distinct ensemble initializations.
        keep = (torch.rand(history.shape[0], 1, device=history.device) < 0.8).to(history.dtype)
        fit_losses.append(masked_mean(per_slot, valid * keep))
    dynamics_loss = torch.stack(fit_losses).mean()

    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-5)
    clean = (batch['noisy_action'] - (1.0 - alpha).sqrt() * prediction) / alpha.sqrt()
    clean = clean.clamp(-1.0, 1.0) * batch['mask']
    with torch.no_grad():
        ep = torch.stack(expert_predictions)[..., :3]
        spread = ((ep - ep.mean(0, keepdim=True)) / 0.020).square().mean((0, 4))
        calibration = ((ep.mean(0) - target[..., :3]) / 0.020).square().mean(-1)
        trust = torch.exp(-spread - calibration).clamp(0.0, 1.0)
        snr_weight = alpha[:, 0, 0] * (alpha[:, 0, 0] > 0.25).to(alpha.dtype)
        passive_weight = passive * trust

    preserve_losses, clearance_losses, consistency_losses = [], [], []
    for head in model.dynamics:
        # Frozen weights and condition stop the regularizer from teaching a
        # fictitiously static world. Gradients still flow from pose -> clean -> epsilon.
        pose, tcp = model.predict_world(head, condition.detach(), clean, history, frozen=True)
        preserve = ((pose[..., :3] - base[..., :3]) / 0.020).square().mean(-1)
        preserve = preserve + orientation_error(pose[..., 3:7], base[..., 3:7])
        preserve_losses.append(masked_mean(preserve * snr_weight[:, None, None], passive_weight))
        clearance = (F.relu(0.010 - box_distance(tcp, pose)) / 0.020).square()
        clearance_losses.append(masked_mean(clearance * snr_weight[:, None, None],
                                            passive_weight * clear_label.to(history.dtype)))
        consistency = ((pose[..., :3] - target[..., :3]) / model.geometry_scale).square().mean((-1, -2))
        consistency = consistency + 0.25 * orientation_error(pose[..., 3:7], target[..., 3:7]).mean(-1)
        consistency = consistency + ((tcp - target_tcp) / model.geometry_scale).square().mean(-1)
        consistency_losses.append(masked_mean(consistency * snr_weight[:, None], valid * trust.mean(-1)))
    preservation_loss = torch.stack(preserve_losses).mean()
    clearance_loss = torch.stack(clearance_losses).mean()
    consistency_loss = torch.stack(consistency_losses).mean()
    prior_loss = (0.05 * role_loss + 0.5 * dynamics_loss + 0.03 * preservation_loss
                  + 0.005 * clearance_loss + 0.05 * consistency_loss)
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'role_loss': role_loss, 'dynamics_loss': dynamics_loss,
            'preservation_loss': preservation_loss, 'clearance_loss': clearance_loss,
            'consistency_loss': consistency_loss}
