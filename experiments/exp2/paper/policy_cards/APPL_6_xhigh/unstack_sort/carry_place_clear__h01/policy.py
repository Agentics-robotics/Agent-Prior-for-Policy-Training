import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def matvec(a, v):
    # Tiny fixed geometry products avoid lazy compiled batched outer-product kernels.
    return (a*v.unsqueeze(-2)).sum(-1)


def matmat(a, b):
    return (a.unsqueeze(-1)*b.unsqueeze(-3)).sum(-2)


def quaternion_matrix(q):
    q = q / q.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def rotation_exp(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1]+(3, 3))
    theta = (v.square().sum(-1)+1e-8).sqrt()
    a = torch.sinc(theta/torch.pi)
    b = 0.5*torch.sinc(theta/(2*torch.pi)).square()
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    return eye + a[..., None, None]*k + b[..., None, None]*matmat(k, k)


def mlp(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dout))


def geometry(raw):
    tcp = raw[..., 18:21]
    rt = quaternion_matrix(raw[..., 21:25])
    p = torch.stack((raw[..., 25:28], raw[..., 32:35]), -2)
    ro = quaternion_matrix(torch.stack((raw[..., 28:32], raw[..., 35:39]), -2))
    goals = torch.stack((raw[..., 41:44], raw[..., 44:47]), -2)
    rel = p-tcp.unsqueeze(-2)
    local = matvec(rt.transpose(-1, -2).unsqueeze(-3), rel)
    rg = matmat(rt.transpose(-1, -2).unsqueeze(-3), ro)
    return tcp, rt, p, ro, goals, rel, local, rg


def evidence(raw):
    # Conservative two-observation evidence of moving attachment, not contact truth.
    tcp, rt, p, ro, goals, rel, local, rg = geometry(raw)
    dt = tcp[:, 1]-tcp[:, 0]
    dp = p[:, 1]-p[:, 0]
    dl = local[:, 1]-local[:, 0]
    st = dt.square().sum(-1).sqrt()
    so = dp.square().sum(-1).sqrt()
    widths = raw[..., 7:9].sum(-1)
    near = (rel.square().sum(-1).sqrt() < 0.045).all(dim=1)
    closed = (widths[:, 0] < 0.058) & (widths[:, 1] < 0.055)
    moving = (st[:, None] > 0.00002) & (so > 0.00002)
    eligible = (near & closed[:, None] & moving).to(raw.dtype)
    dr = (rg[:, 1]-rg[:, 0]).square().sum(dim=(-1, -2))
    label = eligible * torch.exp(-dl.square().sum(-1)/(0.0015**2)-dr/(0.03**2))
    return eligible, label.detach()


class GraspTransformDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        # A single coordinate scale derived solely from the supplied full-demo normalizer.
        s = torch.tensor(n['std'], dtype=torch.float32)
        self.register_buffer('xyz_scale', torch.stack((s[18:21], s[25:28], s[32:35])).amax(0))
        self.register_buffer('position_origin', torch.tensor(n['mean'][18:21], dtype=torch.float32))
        self.register_buffer('identity', torch.eye(2))
        self.horizon = int(spec['training']['horizon'])
        self.attachment = mlp(35, 64, 1)
        self.object_encoder = mlp(94, 128, 128)
        self.robot_encoder = mlp(72, 128, 128)
        self.fusion = nn.Sequential(nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, 256), nn.LayerNorm(256))
        self.denoiser = DiffusionBackbone(256, spec['training'])
        # This is a learned training-only dynamics predictor, not IK or robot FK.
        self.dynamics = mlp(256+self.horizon*8, 512, self.horizon*12)
        nn.init.normal_(self.dynamics[-1].weight, std=0.001)
        nn.init.zeros_(self.dynamics[-1].bias)

    def encode(self, raw):
        b = raw.shape[0]
        norm = (raw-self.obs_mean)/self.obs_scale
        tcp, rt, p, ro, goals, rel, local, rg = geometry(raw)
        dt = tcp[:, 1]-tcp[:, 0]
        dp = p[:, 1]-p[:, 0]
        dl = local[:, 1]-local[:, 0]
        length_scale = self.xyz_scale.max()
        fingers = norm[..., 7:9].reshape(b, 4)[:, None, :].expand(-1, 2, -1)
        fvel = norm[..., 16:18].reshape(b, 4)[:, None, :].expand(-1, 2, -1)
        local_history = (local/length_scale).transpose(1, 2).reshape(b, 2, 6)
        rg_change = (rg[:, 1]-rg[:, 0]).reshape(b, 2, 9)
        dt_objects = (dt/self.xyz_scale)[:, None, :].expand(-1, 2, -1)
        speeds = torch.stack((dp.square().sum(-1).sqrt(),
                              dt.square().sum(-1).sqrt()[:, None].expand(-1, 2),
                              dl.square().sum(-1).sqrt()), -1)/length_scale
        attach_features = torch.cat((fingers, fvel, local_history, rg_change,
                                     dp/self.xyz_scale, dt_objects, dl/length_scale, speeds), -1)
        logits = self.attachment(attach_features).squeeze(-1)
        eligible, target = evidence(raw)
        confidence = logits.sigmoid()*eligible

        goal_error = goals-p
        goal_local = matvec(rt.transpose(-1, -2).unsqueeze(-3), goal_error)
        # Persistent red->blue and blue->none relationship, not a goal-hit switch.
        next_rel = torch.stack((p[:, :, 1]-p[:, :, 0], torch.zeros_like(p[:, :, 1])), 2)
        identities = self.identity[None, None].expand(b, 2, -1, -1)
        next_present = self.identity[:, 0][None, None, :, None].expand(b, 2, -1, -1)
        base = torch.cat(((p-self.position_origin)/self.xyz_scale, ro.flatten(-2),
                          (goals-self.position_origin)/self.xyz_scale, goal_error/self.xyz_scale,
                          goal_local/length_scale, rel/self.xyz_scale,
                          next_rel/self.xyz_scale, identities, next_present), -1)
        base_history = base.transpose(1, 2).reshape(b, 2, 60)
        grasp = torch.cat((local/length_scale, rg.flatten(-2)), -1)
        grasp_history = grasp.transpose(1, 2).reshape(b, 2, 24)*confidence[..., None]
        motion = torch.cat((dp/self.xyz_scale, dt_objects, (dp-dt[:, None])/self.xyz_scale), -1)
        tokens = self.object_encoder(torch.cat((base_history, grasp_history, confidence[..., None], motion), -1))
        global_history = torch.cat((norm[..., :18], norm[..., 18:21], rt.flatten(-2)), -1).reshape(b, 60)
        global_motion = torch.cat((dt/self.xyz_scale, (rt[:, 1]-rt[:, 0]).reshape(b, 9)), -1)
        robot = self.robot_encoder(torch.cat((global_history, global_motion), -1))
        condition = self.fusion(torch.cat((robot, tokens[:, 0], tokens[:, 1]), -1))
        return condition, logits, target

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _ = self.encode(raw_history)
        return self.denoiser(noisy_action, timestep, condition)

    def predict_future(self, condition, encoded_actions, raw):
        b = raw.shape[0]
        out = self.dynamics(torch.cat((condition, encoded_actions.reshape(b, -1)), -1)).reshape(b, self.horizon, 12)
        tcp, rt, p, ro, goals, rel, local, rg = geometry(raw[:, -1])
        tcp_pred = tcp[:, None]+out[..., :3]*self.xyz_scale
        rt_pred = matmat(rt[:, None], rotation_exp(out[..., 3:6]))
        free_pred = p[:, None]+out[..., 6:12].reshape(b, self.horizon, 2, 3)*self.xyz_scale
        held_pred = tcp_pred[:, :, None]+matvec(rt_pred[:, :, None], local[:, None])
        held_rotation = matmat(rt_pred[:, :, None], rg[:, None])
        return tcp_pred, rt_pred, free_pred, held_pred, held_rotation


def robust_vector_error(pred, target):
    # Huber in globally normalized coordinates; no fitted local scales.
    return torch.nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=0.05).mean(-1)


def masked_average(error, mask, weights=None):
    if weights is not None:
        while weights.ndim < error.ndim:
            weights = weights.unsqueeze(-1)
        error = error*weights
    return (error*mask).sum()/mask.sum().clamp_min(1.0)


def state_losses(model, predictions, raw, future, fmask, start_label, weights=None):
    tcp_pred, rt_pred, free_pred, held_pred, held_rotation = predictions
    tf, rf, pf, rof, goals, relf, localf, rgf = geometry(future)
    _, _, _, _, _, _, local0, _ = geometry(raw[:, -1])
    valid = fmask[..., 0]
    pos_err = robust_vector_error(tcp_pred/model.xyz_scale, tf/model.xyz_scale)
    rot_err = (rt_pred-rf).square().mean(dim=(-1, -2))
    free_err = robust_vector_error(free_pred/model.xyz_scale, pf/model.xyz_scale)
    state_loss = masked_average(pos_err+0.1*rot_err, valid, weights)
    state_loss = state_loss+0.5*masked_average(free_err, valid[..., None].expand_as(free_err), weights)
    # Future observations are targets/masks only, never condition inputs.
    # Prefix closure excludes release and every later state from the rigid loss.
    closed = (future[..., 7:9].sum(-1) < 0.055).to(future.dtype)
    uninterrupted = torch.cumprod(closed, dim=1)
    stable = torch.exp(-(localf-local0[:, None]).square().sum(-1)/(0.004**2))
    near = (relf.square().sum(-1) < 0.045**2).to(future.dtype)
    held_mask = (valid[..., None]*uninterrupted[..., None]*near*stable*start_label[:, None]).detach()
    held_err = robust_vector_error(held_pred/model.xyz_scale, pf/model.xyz_scale)
    held_rot_err = (held_rotation-rof).square().mean(dim=(-1, -2))
    rigid_loss = masked_average(held_err+0.1*held_rot_err, held_mask, weights)
    return state_loss+rigid_loss


def build_model(spec):
    return GraspTransformDiffusion(spec)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    condition, logits, labels = model.encode(raw)
    eps = model.denoiser(batch['noisy_action'], batch['timesteps'], condition)
    diffusion = epsilon_loss(eps, batch['noise'], batch['mask'])
    attachment_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    a = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    x0 = (batch['noisy_action']-(1-a).sqrt()*eps)/a.sqrt()
    # The bound is only for the auxiliary predictor; it does not change DDPM.
    x0 = x0.clamp(-2.0, 2.0)
    true_predictions = model.predict_future(condition, batch['encoded_action'], raw)
    noisy_predictions = model.predict_future(condition, x0, raw)
    valid_future = batch['future_mask']*batch['mask']
    true_dynamics = state_losses(model, true_predictions, raw, batch['future_obs'], valid_future, labels)
    reconstructed_dynamics = state_losses(model, noisy_predictions, raw, batch['future_obs'], valid_future, labels,
                                         weights=batch['alpha_bar'].reshape(-1).square())
    prior = 0.02*attachment_loss+0.10*true_dynamics+0.05*reconstructed_dynamics
    return {'loss': diffusion+prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'attachment_loss': attachment_loss, 'demonstrated_dynamics_loss': true_dynamics,
            'reconstructed_dynamics_loss': reconstructed_dynamics}
