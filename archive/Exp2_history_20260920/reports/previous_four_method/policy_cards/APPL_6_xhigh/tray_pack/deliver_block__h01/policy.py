import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def quaternion_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def rotation_six(r):
    return torch.cat((r[..., :, 0], r[..., :, 1]), -1)


def six_matrix(s):
    a = torch.nn.functional.normalize(s[..., :3], dim=-1, eps=1e-6)
    b = s[..., 3:6] - (a*s[..., 3:6]).sum(-1, keepdim=True)*a
    b = torch.nn.functional.normalize(b, dim=-1, eps=1e-6)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), -1)


def rotate_vector(r, v):
    # Explicit small products avoid a sandbox-incompatible lazy GPU compiler path.
    return (r*v.unsqueeze(-2)).sum(-1)


def multiply_rotations(a, b):
    return (a.unsqueeze(-1)*b.unsqueeze(-3)).sum(-2)


def masked_mean(error, mask):
    return (error*mask).sum() / mask.expand_as(error).sum().clamp_min(1.0)


class TemporalBlock(nn.Module):
    def __init__(self, width, condition_dim, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2*width)

    def forward(self, x, condition):
        scale, shift = self.film(condition).unsqueeze(-1).chunk(2, dim=1)
        y = self.norm1(x)*(1+0.1*scale) + 0.1*shift
        y = self.conv1(torch.nn.functional.silu(y))
        y = self.conv2(torch.nn.functional.silu(self.norm2(y)))
        return x + y


class PayloadPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('action_min', torch.tensor(n['action_min'], dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(n['action_scale'], dtype=torch.float32))
        self.register_buffer('time_frequencies', torch.exp(-math.log(10000.0)*torch.arange(32, dtype=torch.float32)/31))
        self.time_count = spec['training']['denoising_train_steps']
        self.horizon = spec['training']['horizon']
        self.context_encoder = nn.Sequential(nn.Linear(159, 256), nn.SiLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU())
        self.time_encoder = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.path_input = nn.Conv1d(8, 128, 1)
        self.path_position = nn.Parameter(torch.randn(1, 128, self.horizon)*0.01)
        self.path_blocks = nn.ModuleList([TemporalBlock(128, 256, d) for d in (1, 2, 4, 1)])
        self.path_head = nn.Conv1d(128, 10, 1)
        nn.init.normal_(self.path_head.weight, std=0.001)
        nn.init.zeros_(self.path_head.bias)
        self.decoder_input = nn.Conv1d(22, 128, 1)
        self.decoder_position = nn.Parameter(torch.randn(1, 128, self.horizon)*0.01)
        self.decoder_blocks = nn.ModuleList([TemporalBlock(128, 128, d) for d in (1, 2, 1)])
        self.decoder_head = nn.Conv1d(128, 7, 1)
        nn.init.normal_(self.decoder_head.weight, std=0.001)
        nn.init.zeros_(self.decoder_head.bias)
        self.score_condition = nn.Sequential(nn.Linear(128+self.horizon*18, 384), nn.SiLU(), nn.Linear(384, 256), nn.LayerNorm(256), nn.SiLU())
        self.score = DiffusionBackbone(256, spec['training'])

    def encode_action(self, a):
        return 2*(a-self.action_min)/self.action_scale-1

    def geometry(self, history):
        # All quantities in this function use only the supplied causal history.
        normal = (history-self.obs_mean)/self.obs_scale
        tcp = history[..., 18:21]
        rt = quaternion_matrix(history[..., 21:25])
        obj = torch.stack((history[..., 25:28], history[..., 32:35]), -2)
        ro = torch.stack((quaternion_matrix(history[..., 28:32]), quaternion_matrix(history[..., 35:39])), -3)
        goals = torch.stack((history[:, -1, 41:44], history[:, -1, 44:47]), 1)
        scale = self.obs_scale[18:21]
        delta = obj-tcp.unsqueeze(-2)
        ct = rotate_vector(rt.transpose(-1, -2).unsqueeze(-3), delta)
        cr = multiply_rotations(rt.transpose(-1, -2).unsqueeze(-3), ro)
        dc = ct[:, 1]-ct[:, 0]
        dr = (cr[:, 1]-cr[:, 0]).square().sum((-1, -2)).sqrt()
        dobj = obj[:, 1]-obj[:, 0]
        dtcp = tcp[:, 1]-tcp[:, 0]
        width = history[:, -1, 7:9].amax(-1, keepdim=True)
        prior_width = history[:, 0, 7:9].amax(-1, keepdim=True)
        motion = dobj.square().sum(-1).sqrt()
        comotion_error = (dobj-dtcp.unsqueeze(1)).square().sum(-1).sqrt()
        near = delta[:, -1].square().sum(-1).sqrt() < 0.030
        previous_near = delta[:, 0].square().sum(-1).sqrt() < 0.030
        closed = (width < 0.027) & (prior_width < 0.027)
        # A supported lifted, closed, close and stable relation is evidence, not contact sensing.
        lifted = (obj[:, -1, :, 2] > 0.026) & (obj[:, 0, :, 2] > 0.026)
        moving_together = (motion > 0.0004) & (dtcp.norm(dim=-1, keepdim=True) > 0.0004) & (comotion_error < 0.002)
        history_available = ((history[:, 1]-history[:, 0]).abs().sum(-1, keepdim=True) > 1e-8)
        valid_gate = closed & near & previous_near & (lifted | moving_together) & (dc.norm(dim=-1) < 0.003) & (dr < 0.12) & history_available
        validity = valid_gate.to(history.dtype)*torch.exp(-(dc/0.002).square().sum(-1) - (dr/0.12).square())
        # Soft causal identity: no trajectory id, phase clock or prescribed color schedule.
        dist = delta[:, -1]/scale
        role_logits = -16*(dist[..., :2].square().sum(-1)+0.25*dist[..., 2].square())
        roles = torch.softmax(role_logits, dim=-1)
        anchor = (roles.unsqueeze(-1)*goals).sum(1)
        # Rotated full-cube XY extent is a state cue; this does not terminate the policy.
        extent = 0.02*ro[:, -1, :, :2, :].abs().sum(-1)
        goal_error = obj[:, -1]-goals
        xy_margin = (0.06-extent-goal_error[..., :2].abs()).amin(-1)
        z_margin = 0.011-goal_error[..., 2].abs()
        completion = torch.sigmoid(xy_margin/0.006)*torch.sigmoid(z_margin/0.003)
        remaining = (1-completion)*(completion < 0.5).to(history.dtype)
        pending = remaining/remaining.sum(-1, keepdim=True).clamp_min(1e-6)
        base = torch.cat((normal[..., :21], rotation_six(rt), normal[..., 25:28],
                          rotation_six(ro[..., 0, :, :]), normal[..., 32:35],
                          rotation_six(ro[..., 1, :, :]), normal[..., 39:47]), -1)
        features = torch.cat((base.flatten(1), ((tcp[:, -1, None]-goals)/scale).flatten(1),
                              (goal_error/scale).flatten(1), (ct[:, -1]/scale).flatten(1),
                              rotation_six(cr[:, -1]).flatten(1), (dc/scale).flatten(1),
                              validity, roles, (dtcp/scale), (dobj/scale).flatten(1), completion, pending), -1)
        context = self.context_encoder(features)
        current_path = torch.cat(((tcp[:, -1]-anchor)/scale, rotation_six(rt[:, -1])), -1)
        return {'context': context, 'anchor': anchor, 'goals': goals, 'tcp': tcp[:, -1],
                'rotation': rt[:, -1], 'ct': ct[:, -1], 'cr': cr[:, -1],
                'validity': validity, 'roles': roles, 'pending': pending,
                'base_path': current_path, 'q': history[:, -1, :7]}

    def time_features(self, timestep, batch, device, dtype):
        t = torch.as_tensor(timestep, device=device).reshape(-1).expand(batch).to(dtype)
        a = t.unsqueeze(-1)*self.time_frequencies.to(dtype)
        return self.time_encoder(torch.cat((a.sin(), a.cos()), -1))

    def predict_path(self, sample, timestep, geo):
        h = sample.shape[1]
        time = self.time_features(timestep, sample.shape[0], sample.device, sample.dtype)
        cond = torch.cat((geo['context'], time), -1)
        z = self.path_input(sample.transpose(1, 2)) + self.path_position[..., :h]
        for block in self.path_blocks:
            z = block(z, cond)
        delta = self.path_head(torch.nn.functional.silu(z)).transpose(1, 2)
        position = geo['base_path'][:, None, :3]+delta[..., :3]
        rotation = six_matrix(geo['base_path'][:, None, 3:9]+0.1*delta[..., 3:9])
        gripper = delta[..., 9:10].tanh()
        return torch.cat((position, rotation_six(rotation), gripper), -1)

    def path_world(self, path, geo):
        return path[..., :3]*self.obs_scale[18:21]+geo['anchor'][:, None], six_matrix(path[..., 3:9])

    def payload_prediction(self, path, geo):
        world, rot = self.path_world(path, geo)
        translated = rotate_vector(rot.unsqueeze(2), geo['ct'][:, None])
        payload = world.unsqueeze(2)+translated
        payload_rot = multiply_rotations(rot.unsqueeze(2), geo['cr'][:, None])
        return payload, payload_rot

    def decode(self, path, geo):
        world, rot = self.path_world(path, geo)
        payload, _ = self.payload_prediction(path, geo)
        payload_error = ((payload-geo['goals'][:, None])/self.obs_scale[18:21])*geo['validity'][:, None, :, None]
        # Current configuration and world frame remain available to choose the local redundancy branch.
        inputs = torch.cat((path, (world-self.obs_mean[18:21])/self.obs_scale[18:21],
                            (world-geo['tcp'][:, None])/self.obs_scale[18:21], payload_error.flatten(2)), -1)
        z = self.decoder_input(inputs.transpose(1, 2)) + self.decoder_position[..., :path.shape[1]]
        for block in self.decoder_blocks:
            z = block(z, geo['context'])
        delta = self.decoder_head(torch.nn.functional.silu(z)).transpose(1, 2)
        q_encoded = 2*(geo['q']-self.action_min[:7])/self.action_scale[:7]-1
        q_anchor = torch.atanh(q_encoded.clamp(-0.98, 0.98))[:, None]
        joints = torch.tanh(q_anchor+delta)
        return torch.cat((joints, path[..., 9:10]), -1)

    def denoise(self, noisy_action, timestep, geo):
        path = self.predict_path(noisy_action, timestep, geo)
        proposal = self.decode(path, geo)
        cond = self.score_condition(torch.cat((geo['context'], path.flatten(1), proposal.flatten(1)), -1))
        epsilon = self.score(noisy_action, timestep, cond)
        return epsilon, path, proposal

    def forward(self, noisy_action, timestep, raw_history):
        geo = self.geometry(raw_history)
        return self.denoise(noisy_action, timestep, geo)[0]


def build_model(spec):
    return PayloadPolicy(spec)


def path_error(predicted, target):
    position = (predicted[..., :3]-target[..., :3]).square().mean(-1, keepdim=True)
    rotation = (predicted[..., 3:9]-target[..., 3:9]).square().mean(-1, keepdim=True)
    gripper = (predicted[..., 9:10]-target[..., 9:10]).square()
    return position+0.1*rotation+0.25*gripper


def compute_loss(model, batch, spec):
    geo = model.geometry(batch['raw_obs'])
    eps, path, proposal = model.denoise(batch['noisy_action'], batch['timesteps'], geo)
    diffusion = epsilon_loss(eps, batch['noise'], batch['mask'])
    future = batch['future_obs']
    fm = batch['future_mask']*batch['mask']
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp(1e-6, 1-1e-6)
    reliability = 0.25+0.75*alpha
    target_path = torch.cat(((future[..., 18:21]-geo['anchor'][:, None])/model.obs_scale[18:21],
                             rotation_six(quaternion_matrix(future[..., 21:25])), batch['encoded_action'][..., 7:8]), -1)
    # Padded paths never supply training information or fictitious zero poses to the inverse decoder.
    fallback = torch.cat((geo['base_path'][:, None].expand(-1, future.shape[1], -1),
                          torch.zeros_like(target_path[..., 9:10])), -1)
    target_path = torch.where(fm.bool().expand_as(target_path), target_path, fallback)
    noisy_path_loss = masked_mean(path_error(path, target_path)*reliability, fm)
    clean_path = model.predict_path(batch['encoded_action'], torch.zeros_like(batch['timesteps']), geo)
    clean_path_loss = masked_mean(path_error(clean_path, target_path), fm)
    teacher_action = model.decode(target_path, geo)
    teacher_loss = masked_mean((teacher_action[..., :7]-batch['encoded_action'][..., :7]).square(), fm)
    decoded_loss = masked_mean((proposal-batch['encoded_action']).square()*reliability, fm)
    pair_mask = fm[:, 1:]*fm[:, :-1]
    target_delta = batch['encoded_action'][:, 1:, :7]-batch['encoded_action'][:, :-1, :7]
    proposal_delta = proposal[:, 1:, :7]-proposal[:, :-1, :7]
    teacher_delta = teacher_action[:, 1:, :7]-teacher_action[:, :-1, :7]
    increment_loss = masked_mean((proposal_delta-target_delta).square()*reliability, pair_mask)
    increment_loss = increment_loss + masked_mean((teacher_delta-target_delta).square(), pair_mask)

    future_object = torch.stack((future[..., 25:28], future[..., 32:35]), 2)
    future_ro = torch.stack((quaternion_matrix(future[..., 28:32]), quaternion_matrix(future[..., 35:39])), 2)
    future_rt = quaternion_matrix(future[..., 21:25])
    future_delta = future_object-future[..., None, 18:21]
    future_ct = rotate_vector(future_rt.transpose(-1, -2).unsqueeze(2), future_delta)
    future_cr = multiply_rotations(future_rt.transpose(-1, -2).unsqueeze(2), future_ro)
    rigid_position = (future_ct-geo['ct'][:, None]).norm(dim=-1) < 0.004
    rigid_rotation = (future_cr-geo['cr'][:, None]).square().sum((-1, -2)) < 0.15**2
    future_closed = future[..., 7:9].amax(-1, keepdim=True) < 0.027
    future_near = future_delta.norm(dim=-1) < 0.030
    rigid_mask = fm*geo['validity'][:, None]*(rigid_position & rigid_rotation & future_closed & future_near).to(future.dtype)

    def payload_loss(p, reliability_weight):
        position, rotation = model.payload_prediction(p, geo)
        # Both sides are in each object's own goal frame; the transform comes only from current history.
        predicted_error = (position-geo['goals'][:, None])/model.obs_scale[18:21]
        observed_error = (future_object-geo['goals'][:, None])/model.obs_scale[18:21]
        e = (predicted_error-observed_error).square().mean(-1)
        e = e+0.1*(rotation-future_ro).square().mean((-1, -2))
        return masked_mean(e*reliability_weight, rigid_mask)

    rigid_loss = payload_loss(path, reliability)
    # Link geometry to the actual diffusion clean-action estimate, not just an observed-pose penalty.
    # Clean-action training above calibrates the shared local action-to-path predictor.
    clean_estimate = (batch['noisy_action']-(1-alpha).sqrt()*eps)/alpha.sqrt()
    cycle_path = model.predict_path(clean_estimate.clamp(-1.25, 1.25), torch.zeros_like(batch['timesteps']), geo)
    cycle_weight = alpha.square()
    cycle_path_loss = masked_mean(path_error(cycle_path, target_path)*cycle_weight, fm)
    cycle_rigid_loss = payload_loss(cycle_path, cycle_weight)
    prior = (noisy_path_loss+0.5*clean_path_loss+0.5*teacher_loss+0.5*decoded_loss+
             0.1*increment_loss+2.0*rigid_loss+0.2*cycle_path_loss+cycle_rigid_loss)
    total = diffusion+prior
    return {'loss': total, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'path_loss': noisy_path_loss, 'clean_path_loss': clean_path_loss,
            'inverse_loss': teacher_loss, 'decoded_loss': decoded_loss,
            'increment_loss': increment_loss, 'payload_loss': rigid_loss,
            'cycle_path_loss': cycle_path_loss, 'cycle_payload_loss': cycle_rigid_loss}
