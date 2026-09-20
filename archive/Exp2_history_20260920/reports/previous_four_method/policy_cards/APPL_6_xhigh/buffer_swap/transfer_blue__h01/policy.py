import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


# Quaternion convention throughout is wxyz. These operations are geometry,
# not robot kinematics or an action controller.
def unit(q):
    return q / q.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def rotate(q, v):
    qv = q[..., 1:]
    uv = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + torch.cross(qv, uv, dim=-1))


def canonical(q):
    q = unit(q)
    largest = q.abs().argmax(-1, keepdim=True)
    sign = torch.where(q.gather(-1, largest) < 0, -torch.ones_like(q[..., :1]),
                       torch.ones_like(q[..., :1]))
    return q * sign


def exp_rotation(v):
    theta = v.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return unit(torch.cat((torch.cos(theta / 2),
                           v * (0.5 * torch.sinc(theta / (2 * torch.pi)))), -1))


def rotation_error(a, b):
    # Shortest SO(3) logarithm, including a safe small-angle limit.
    d = unit(multiply(conjugate(unit(a)), unit(b)))
    d = torch.where(d[..., :1] < 0, -d, d)
    s = d[..., 1:].square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    angle = 2 * torch.atan2(s, d[..., :1].clamp_min(0))
    return d[..., 1:] * (angle / s)


def poses(raw):
    return raw[..., 18:39].reshape(*raw.shape[:-1], 3, 7)


def relative_from_poses(p):
    tq = unit(p[..., 0, 3:7])
    op = p[..., 1:3, :3]
    oq = unit(p[..., 1:3, 3:7])
    inv = conjugate(tq).unsqueeze(-2).expand_as(oq)
    dp = rotate(inv, op - p[..., 0:1, :3])
    dq = canonical(multiply(inv, oq))
    return dp, dq


def weighted_mean(error, weight):
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def robust(x):
    return F.smooth_l1_loss(x, torch.zeros_like(x), reduction='none')


def pose_distance(dp, dq, tp, tq, metres=0.02, radians=0.15):
    # Product translation/SO(3) metric, rather than a full SE(3) logarithm.
    return (robust((dp - tp) / metres).mean(-1) +
            robust(rotation_error(dq, tq) / radians).mean(-1))


class AttachmentDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('action_min', torch.tensor(n['action_min'], dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(n['action_scale'], dtype=torch.float32))
        # One shared full-demo TCP scale for relative translations in rotated axes.
        # No empirical scale is fitted to this skill or to attachment intervals.
        self.register_buffer('position_scale', torch.tensor(max(n['std'][18:21]), dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(175, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU())
        self.attachment = nn.Linear(256, 3)  # none, red, blue
        self.transform = nn.Linear(256, 14)  # two blends and bounded SE(3) corrections
        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        # 256 learned features + 94 normalized causal observations + 3 mode
        # probabilities + 14 confidence-weighted transform coordinates.
        self.condition_dimension = 367
        self.diffusion = DiffusionBackbone(self.condition_dimension, spec['training'])
        self.initial_dynamics = nn.Linear(self.condition_dimension, 256)
        # action(8), target/realized lag(9), predicted state(30), horizon fraction(1)
        self.dynamics_cell = nn.GRUCell(48, 256)
        # qpos delta(9), pose position delta(9), rotation-vector delta(9), mode(3)
        self.dynamics_output = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 30))
        nn.init.normal_(self.dynamics_output[-1].weight, std=0.001)
        nn.init.zeros_(self.dynamics_output[-1].bias)

    def normalize_obs(self, x):
        return (x - self.obs_mean) / self.obs_scale

    def decode_action(self, x):
        return (x + 1) * self.action_scale / 2 + self.action_min

    def encode(self, history):
        normalized = self.normalize_obs(history)
        dpos, dquat = relative_from_poses(poses(history))
        # Both times are causal; quaternion features have fixed unit bounds.
        relative_feature = torch.cat((dpos / self.position_scale, dquat), -1).flatten(1)
        goals = history[:, -1, 41:47].reshape(-1, 2, 3)
        objects = poses(history[:, -1])[:, 1:3, :3]
        goal_error = ((goals - objects) / self.position_scale).flatten(1)
        features = torch.cat((normalized.flatten(1), relative_feature,
                              normalized[:, -1] - normalized[:, -2], goal_error), -1)
        latent = self.encoder(features)
        logits = self.attachment(latent)
        prob = logits.softmax(-1)
        param = self.transform(latent).reshape(-1, 2, 7)
        blend = torch.sigmoid(param[..., :1])
        oldq = dquat[:, 0]
        newq = dquat[:, 1]
        oldq = torch.where((oldq * newq).sum(-1, keepdim=True) < 0, -oldq, oldq)
        estimated_p = (blend * dpos[:, 1] + (1 - blend) * dpos[:, 0] +
                       0.005 * torch.tanh(param[..., 1:4]))
        estimated_q = canonical(multiply(unit(blend * newq + (1 - blend) * oldq),
                                         exp_rotation(0.05 * torch.tanh(param[..., 4:7]))))
        transform_feature = torch.cat((estimated_p / self.position_scale, estimated_q), -1)
        transform_feature = transform_feature * prob[:, 1:3, None]
        condition = torch.cat((latent, normalized.flatten(1), prob, transform_feature.flatten(1)), -1)
        return condition, logits, estimated_p, estimated_q

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _, _ = self.encode(raw_history)
        return self.diffusion(noisy_action, timestep, condition)

    def rollout(self, raw_history, encoded_action, condition, initial_logits):
        # Slot zero is the measured state at t, after action t-1. It must not
        # be advanced again. Slots 1..H-1 predict states after t..t+H-2.
        current = raw_history[:, -1]
        q0 = current[:, :9]
        p0 = poses(current)
        native = self.decode_action(encoded_action)
        h = torch.tanh(self.initial_dynamics(condition))
        qs, ps, gates = [q0], [p0], [initial_logits]
        prevq, prevp = q0, p0
        for j in range(1, encoded_action.shape[1]):
            arm_target = native[:, j, :7]
            finger_target = (0.025 * native[:, j, 7:8] + 0.015).expand(-1, 2)
            target = torch.cat((arm_target, finger_target), -1)
            lag = (target - prevq) / self.obs_scale[:9]
            qfeature = (prevq - self.obs_mean[:9]) / self.obs_scale[:9]
            posefeature = ((prevp.flatten(1) - self.obs_mean[18:39]) /
                           self.obs_scale[18:39])
            phase = torch.ones_like(encoded_action[:, j, :1]) * (float(j) / (encoded_action.shape[1] - 1))
            inputs = torch.cat((encoded_action[:, j], lag, qfeature, posefeature, phase), -1)
            h = self.dynamics_cell(inputs, h)
            out = self.dynamics_output(h)
            predq = q0 + out[:, :9] * self.obs_scale[:9]
            predpos = p0[..., :3] + out[:, 9:18].reshape(-1, 3, 3) * self.position_scale
            predrot = canonical(multiply(unit(p0[..., 3:7]),
                                         exp_rotation(0.3 * out[:, 18:27].reshape(-1, 3, 3))))
            predp = torch.cat((predpos, predrot), -1)
            qs.append(predq)
            ps.append(predp)
            gates.append(out[:, 27:30])
            prevq, prevp = predq, predp
        return torch.stack(qs, 1), torch.stack(ps, 1), torch.stack(gates, 1)


def build_model(spec):
    return AttachmentDiffusion(spec)


@torch.no_grad()
def attachment_labels(history, future, native_action):
    # Weak contact labels, not measured forces. Closed fingers and proximity
    # alone are ambiguous when both bodies are stationary on the table.
    fp, fq = relative_from_poses(poses(future))
    oldp, oldq = relative_from_poses(poses(history[:, 0]))
    previous_p = torch.cat((oldp[:, None], fp[:, :-1]), 1)
    previous_q = torch.cat((oldq[:, None], fq[:, :-1]), 1)
    dist = fp.square().sum(-1).sqrt()
    aperture = future[..., 7:9].sum(-1, keepdim=True)
    closed = torch.sigmoid((0.055 - aperture) / 0.004)
    nearby = torch.sigmoid((0.035 - dist) / 0.006)
    trans_change = (fp - previous_p).square().sum(-1) / (0.006 ** 2)
    angle_change = rotation_error(previous_q, fq).square().sum(-1) / (0.12 ** 2)
    comotion = torch.exp(-0.5 * (trans_change + angle_change))
    close_command = ((1 - native_action[..., 7:8]) / 2).clamp(0, 1)
    score = closed * nearby * comotion * close_command
    score = score / score.sum(-1, keepdim=True).clamp_min(1)
    distribution = torch.cat((1 - score.sum(-1, keepdim=True), score), -1)
    return distribution, fp, fq


def state_supervision(model, prediction, target, weight):
    pq, pp, _ = prediction
    targetp = poses(target)
    qerr = robust((pq - target[..., :9]) / model.obs_scale[:9]).mean(-1)
    # Global shared metric scale avoids amplifying blue's tiny world-x range.
    perr = robust((pp[..., :3] - targetp[..., :3]) / model.position_scale).mean((-1, -2))
    rerr = robust(rotation_error(pp[..., 3:7], targetp[..., 3:7]) / 0.15).mean((-1, -2))
    return weighted_mean(qerr + perr + rerr, weight)


def gate_supervision(logits, target, weight):
    ce = -(target * logits.log_softmax(-1)).sum(-1)
    return weighted_mean(ce, weight)


def rigidity_supervision(predicted_poses, refp, refq, labels, valid, sample_weight):
    dp, dq = relative_from_poses(predicted_poses)
    scores = labels[..., 1:3]
    # Pairwise rigidity also covers newly acquired objects within a chunk.
    pair_weight = (scores[:, 1:] * scores[:, :-1] * valid[:, 1:, None] *
                   valid[:, :-1, None] * sample_weight[:, None, None])
    pairerr = pose_distance(dp[:, 1:], dq[:, 1:], dp[:, :-1], dq[:, :-1])
    pairloss = weighted_mean(pairerr, pair_weight)
    # Anchor only until the first supported release / lost attachment label.
    # No stale transform from an empty approach or an earlier object is used.
    continuity = ((scores > 0.6).to(scores.dtype) * valid[..., None]).cumprod(1)
    anchor_weight = continuity * scores * sample_weight[:, None, None]
    anchor_weight = torch.cat((torch.zeros_like(anchor_weight[:, :1]), anchor_weight[:, 1:]), 1)
    anchorerr = pose_distance(dp, dq, refp[:, None].expand_as(dp), refq[:, None].expand_as(dq))
    return pairloss + weighted_mean(anchorerr, anchor_weight)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    condition, current_logits, refp, refq = model.encode(history)
    epsilon = model.diffusion(batch['noisy_action'], batch['timesteps'], condition)
    diffusion_loss = epsilon_loss(epsilon, batch['noise'], batch['mask'])
    a = batch['alpha_bar'].reshape(-1, 1, 1).clamp(1e-5, 1)
    x0 = (batch['noisy_action'] - (1 - a).sqrt() * epsilon) / a.sqrt()
    # Training-only bound prevents a very noisy x0 from destabilizing dynamics.
    # Sampling and decoding are performed by the framework's fixed DDPM.
    x0 = x0.clamp(-1.25, 1.25)
    certainty = a.reshape(-1).square()
    future = batch['future_obs']
    valid = (batch['mask'] * batch['future_mask']).squeeze(-1)
    labels, target_dp, target_dq = attachment_labels(history, future, batch['native_action'])
    step_weight = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, 1:]), 1)

    # Jointly train F with demonstrated and reconstructed action chunks. The
    # second path explicitly links future-state and rigidity losses to epsilon.
    teacher = model.rollout(history, batch['encoded_action'], condition, current_logits)
    reconstructed = model.rollout(history, x0, condition, current_logits)
    dyn_teacher = state_supervision(model, teacher, future, step_weight)
    dyn_reconstructed = state_supervision(model, reconstructed, future,
                                         step_weight * certainty[:, None])
    causal_gate = gate_supervision(current_logits, labels[:, 0], valid[:, 0])
    rollout_gate = gate_supervision(teacher[2], labels, step_weight)
    reconstruction_gate = gate_supervision(reconstructed[2], labels, step_weight * certainty[:, None])

    continuity = ((labels[..., 1:3] > 0.6).to(valid.dtype) * valid[..., None]).cumprod(1)
    offset_weight = continuity * labels[..., 1:3]
    offset_error = pose_distance(refp[:, None].expand_as(target_dp), refq[:, None].expand_as(target_dq),
                                 target_dp, target_dq)
    offset_loss = weighted_mean(offset_error, offset_weight)
    rigid_teacher = rigidity_supervision(teacher[1], refp, refq, labels, valid,
                                          torch.ones_like(certainty))
    rigid_reconstructed = rigidity_supervision(reconstructed[1], refp, refq, labels, valid, certainty)
    prior_loss = (0.08 * dyn_teacher + 0.04 * dyn_reconstructed +
                  0.03 * causal_gate + 0.015 * rollout_gate + 0.005 * reconstruction_gate +
                  0.015 * offset_loss + 0.01 * rigid_teacher + 0.01 * rigid_reconstructed)
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'dynamics_teacher': dyn_teacher,
            'dynamics_reconstructed': dyn_reconstructed, 'attachment_loss': causal_gate,
            'offset_loss': offset_loss, 'rigid_loss': rigid_teacher + rigid_reconstructed}
