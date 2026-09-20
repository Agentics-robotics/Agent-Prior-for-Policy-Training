import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


def unit_quat(q):
    return q / q.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def rotation(q):
    w, x, y, z = unit_quat(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)),
                       dim=-1).reshape(q.shape[:-1] + (3, 3))


def rotate(r, v):
    return (r @ v.unsqueeze(-1)).squeeze(-1)


def rotation_log(q):
    q = unit_quat(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n = v.square().sum(-1, keepdim=True).clamp_min(1e-16).sqrt()
    angle = 2 * torch.atan2(n, q[..., :1].clamp_min(0))
    factor = torch.where(n > 1e-6, angle / n, 2 * torch.ones_like(n))
    return factor * v


def se3_translation_log(p, phi):
    theta2 = phi.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-6).sqrt()
    general = (1 - 0.5*theta / torch.tan(0.5*theta).clamp_min(1e-6)) / theta.square()
    coefficient = torch.where(theta2 < 1e-4, 1.0/12.0 + theta2/720.0, general)
    cross = torch.cross(phi, p, dim=-1)
    return p - 0.5*cross + coefficient*torch.cross(phi, cross, dim=-1)


def upright_yaw(q):
    w, x, y, z = unit_quat(q).unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw/2), zeros, zeros, torch.sin(yaw/2)), dim=-1)


def mlp(dim_in, dim_hidden, dim_out):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU(),
                         nn.Linear(dim_hidden, dim_out))


class CausalPosePredictor(nn.Module):
    """Training-only action-conditioned dynamics, not analytic FK or an IK solver."""
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(8 + 18 + 64 + 1, 128, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(128, 128, 3, dilation=d)
                                     for d in (1, 2, 4, 8)])
        self.output = nn.Conv1d(128, 18, 1)

    def forward(self, actions, joint_state, context):
        b, h, _ = actions.shape
        slot = torch.arange(h, device=actions.device, dtype=actions.dtype)
        slot = (slot / max(h-1, 1)).reshape(1, h, 1).expand(b, -1, -1)
        x = torch.cat((actions, joint_state[:, None].expand(-1, h, -1),
                       context[:, None].expand(-1, h, -1), slot), dim=-1)
        x = F.silu(self.input(x.transpose(1, 2)))
        for layer, dilation in zip(self.blocks, (1, 2, 4, 8)):
            x = x + 0.5*F.silu(layer(F.pad(x, (2*dilation, 0))))
        return self.output(x).transpose(1, 2)


class GraspFramePolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        # A rotation-compatible scalar derived ONLY from the shared full-demo scales.
        indices = [18, 19, 20, 25, 26, 27, 32, 33, 34]
        length = max(normalizer['std'][i] for i in indices)
        self.register_buffer('length_scale', torch.tensor(length, dtype=torch.float32))
        self.role_encoder = nn.Sequential(nn.Linear(238, 128), nn.SiLU(),
                                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))
        self.condition_encoder = nn.Sequential(nn.Linear(276, 256), nn.SiLU(),
                                               nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU())
        self.backbone = DiffusionBackbone(256, spec['training'])
        self.decoder_context = mlp(256, 128, 64)
        self.joint_noise_decoder = nn.Sequential(nn.Linear(114, 128), nn.SiLU(),
                                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 8))
        nn.init.zeros_(self.joint_noise_decoder[-1].weight)
        nn.init.zeros_(self.joint_noise_decoder[-1].bias)
        self.pose_predictor = CausalPosePredictor()
        self.register_buffer('time_frequencies', torch.exp(torch.linspace(0, -math.log(10000), 8)))

    def geometry(self, raw):
        tcp = raw[..., 18:21]
        qt = unit_quat(raw[..., 21:25])
        rt = rotation(qt)
        rt_inv = rt.transpose(-1, -2)
        features = []
        # The role order is blue, empty, red; object geometry order is blue, red.
        for pose_index, goal_index in ((32, 44), (25, 41)):
            obj = raw[..., pose_index:pose_index+3]
            qo = unit_quat(raw[..., pose_index+3:pose_index+7])
            goal = raw[..., goal_index:goal_index+3]
            d = rotate(rt_inv, obj-tcp)
            qd = unit_quat(multiply(conjugate(qt), qo))
            qstar = unit_quat(multiply(upright_yaw(qo), conjugate(qd)))
            pstar = goal - rotate(rotation(qstar), d)
            world_error = pstar-tcp
            local_error = rotate(rt_inv, world_error)
            phi = rotation_log(multiply(conjugate(qt), qstar))
            rho = se3_translation_log(local_error, phi)
            rd6 = rotation(qd)[..., :, :2].reshape(raw.shape[:2] + (6,))
            feat = torch.cat((d/self.length_scale, rd6,
                              (goal-obj)/self.length_scale,
                              (pstar-self.obs_mean[18:21])/self.obs_scale[18:21],
                              world_error/self.length_scale,
                              rho/self.length_scale, phi/math.pi), dim=-1)
            features.append(feat)
        geom = torch.stack(features, dim=2)  # [B, 2, 2, 24]
        red_delta = raw[..., 25:28]-tcp
        empty = torch.cat((red_delta/self.length_scale,
                           rotate(rt_inv, red_delta)/self.length_scale,
                           (raw[..., 41:44]-raw[..., 25:28])/self.length_scale,
                           (tcp-self.obs_mean[18:21])/self.obs_scale[18:21],
                           rt[..., :, :2].reshape(raw.shape[:2] + (6,))), dim=-1)
        return geom, empty

    def encode(self, raw):
        obs = (raw-self.obs_mean)/self.obs_scale
        geom, empty = self.geometry(raw)
        role_input = torch.cat((obs.flatten(1), geom.flatten(1),
                                (geom[:, 1]-geom[:, 0]).flatten(1)), dim=-1)
        logits = self.role_encoder(role_input)
        probabilities = logits.softmax(-1)
        attached = torch.stack((probabilities[:, 0], probabilities[:, 2]), dim=-1)
        gated = geom * attached[:, None, :, None]
        empty = empty * probabilities[:, None, 1:2]
        condition_input = torch.cat((obs.flatten(1), gated.flatten(1), empty.flatten(1),
                                     probabilities, obs[:, 1]-obs[:, 0]), dim=-1)
        condition = self.condition_encoder(condition_input)
        return condition, self.decoder_context(condition), obs[:, -1, :18], logits

    def predict_bundle(self, noisy_action, timestep, raw_history):
        condition, context, joints, logits = self.encode(raw_history)
        latent = self.backbone(noisy_action, timestep, condition)
        b, h, _ = noisy_action.shape
        time = torch.as_tensor(timestep, dtype=noisy_action.dtype, device=noisy_action.device)
        time = time.reshape(-1).expand(b)
        phase = time[:, None] * self.time_frequencies[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        inp = torch.cat((latent, noisy_action, joints[:, None].expand(-1, h, -1),
                         context[:, None].expand(-1, h, -1),
                         embedding[:, None].expand(-1, h, -1)), dim=-1)
        epsilon = latent + self.joint_noise_decoder(inp)
        return epsilon, context, joints, logits

    def forward(self, noisy_action, timestep, raw_history):
        return self.predict_bundle(noisy_action, timestep, raw_history)[0]

    def future_targets(self, raw, future):
        current = raw[:, -1]
        positions = []
        angles = []
        for index in (18, 25, 32):
            positions.append((future[..., index:index+3]-current[:, None, index:index+3]) / self.length_scale)
            q0 = unit_quat(current[:, None, index+3:index+7])
            q1 = unit_quat(future[..., index+3:index+7])
            angles.append(rotation_log(multiply(conjugate(q0), q1))/math.pi)
        return torch.cat(positions + angles, dim=-1)


def attachment_targets(raw):
    """Weak causal labels, not measured contact and never used as action commands."""
    finger = raw[:, -1, 7:9].mean(-1)
    closed = ((0.04-finger)/(0.04-0.01825)).clamp(0, 1)
    tcp = raw[..., 18:21]
    tcp_motion = tcp[:, 1]-tcp[:, 0]
    scores = []
    for index in (32, 25):
        obj = raw[..., index:index+3]
        distance2 = (obj[:, -1]-tcp[:, -1]).square().sum(-1)
        motion = obj[:, 1]-obj[:, 0]
        disagreement = (motion-tcp_motion).square().sum(-1)
        motion_scale = 0.002**2 + motion.square().sum(-1) + tcp_motion.square().sum(-1)
        score = closed * torch.exp(-distance2/(2*0.02**2)) * torch.exp(-disagreement/(2*motion_scale))
        scores.append(score)
    scores = torch.stack(scores, dim=-1)
    scores = scores / scores.sum(-1, keepdim=True).clamp_min(1)
    empty = (1-scores.sum(-1)).clamp_min(0)
    target = torch.stack((scores[:, 0], empty, scores[:, 1]), dim=-1)
    return 0.99*target + 0.01/3


def masked_pose_loss(prediction, target, mask, weight=None):
    error = F.smooth_l1_loss(prediction, target, reduction='none', beta=0.1)
    if weight is not None:
        error = error * weight
    return (error*mask).sum()/(mask.sum()*prediction.shape[-1]).clamp_min(1)


def build_model(spec):
    return GraspFramePolicy(spec)


def compute_loss(model, batch, spec):
    eps, context, joints, logits = model.predict_bundle(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(eps, batch['noise'], batch['mask'])
    role_target = attachment_targets(batch['raw_obs']).detach()
    role_loss = -(role_target * logits.log_softmax(-1)).sum(-1).mean()
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp(1e-6, 1)
    x0 = (batch['noisy_action']-(1-alpha).sqrt()*eps)/alpha.sqrt()
    # Auxiliary-only bounding and alpha weighting prevent high-noise amplification.
    # The deployed epsilon and the fixed DDPM sampler are unchanged.
    predicted_future = model.pose_predictor(x0.clamp(-2, 2), joints, context)
    teacher_future = model.pose_predictor(batch['encoded_action'], joints, context)
    target = model.future_targets(batch['raw_obs'], batch['future_obs']).detach()
    future_mask = batch['future_mask'] * batch['mask']
    teacher_loss = masked_pose_loss(teacher_future, target, future_mask)
    action_pose_loss = masked_pose_loss(predicted_future, target, future_mask, alpha.detach())
    prior_loss = 0.02*role_loss + 0.10*teacher_loss + 0.05*action_pose_loss
    return {'loss': diffusion_loss+prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'attachment_loss': role_loss,
            'teacher_pose_loss': teacher_loss, 'denoised_pose_loss': action_pose_loss}
