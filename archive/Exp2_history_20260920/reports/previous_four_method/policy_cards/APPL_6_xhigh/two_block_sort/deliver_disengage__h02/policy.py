import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


def unit(q):
    return F.normalize(q, dim=-1, eps=1e-8)


def rotation(q):
    w, x, y, z = unit(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def poses(raw):
    return torch.stack((raw[..., 18:25], raw[..., 25:32], raw[..., 32:39]), -2)


def goals(raw):
    return torch.stack((raw[..., 41:44], raw[..., 44:47]), -2)


def margins(pose, goal):
    # All eight rotated cube corners, in world metres. Height is a centre test.
    vertices = pose.new_tensor([[-.02,-.02,-.02], [-.02,-.02,.02],
                               [-.02,.02,-.02], [-.02,.02,.02],
                               [.02,-.02,-.02], [.02,-.02,.02],
                               [.02,.02,-.02], [.02,.02,.02]])
    corners = pose[..., :3].unsqueeze(-2) + torch.einsum('...ij,kj->...ki', rotation(pose[..., 3:]), vertices)
    xy = .06 - (corners[..., :2] - goal[..., None, :2]).abs().amax(-2)
    z = .011 - (pose[..., 2] - goal[..., 2]).abs()
    return torch.cat((xy, z.unsqueeze(-1)), -1)


def ready_proxy(raw):
    obj = poses(raw)[..., 1:, :]
    valid = (margins(obj, goals(raw)) >= 0).all(-1)
    displacement = raw[..., None, 18:21] - obj[..., :3]
    separated = (displacement.norm(dim=-1) > .06) & (displacement[..., 2] > .05)
    opened = (raw[..., 7:9] > .035).all(-1)
    return (valid & separated & opened.unsqueeze(-1)).to(raw.dtype)


def observed_status(history):
    now = history[:, -1]
    obj = poses(now)[:, 1:]
    m = margins(obj, goals(now))
    valid = (m >= 0).all(-1)
    distance = (now[:, None, 18:21] - obj[..., :3]).norm(dim=-1)
    obj_motion = (obj[..., :3] - poses(history[:, -2])[:, 1:, :3]).norm(dim=-1)
    # This is observed release support, not proof of contact or a success rule.
    release_support = ((now[:, 7:9].sum(-1, keepdim=True) > .065) | (distance > .08))
    protected = valid & release_support & (obj_motion < .003)
    role = (distance + 4.0 * protected.to(now.dtype)).argmin(-1)
    role = torch.where(protected.all(-1), torch.full_like(role, -1), role)
    onehot = F.one_hot(role.clamp_min(0), 2).to(now.dtype) * (role >= 0).unsqueeze(-1)
    return m, valid, protected.to(now.dtype), role, onehot


def average(value, mask):
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class Forecast(nn.Module):
    """Recurrent action-conditioned pose predictor; not a physics simulator."""
    def __init__(self, normalizer):
        super().__init__()
        scale = torch.tensor(normalizer['std'], dtype=torch.float32)
        self.register_buffer('position_scale', torch.stack((scale[18:21], scale[25:28], scale[32:35])))
        self.register_buffer('finger_scale', scale[7:9].clone())
        self.register_buffer('action_min', torch.tensor(normalizer['action_min'], dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(normalizer['action_scale'], dtype=torch.float32))
        self.initial = mlp(256, 128, 128)
        self.context = nn.Linear(256, 64)
        self.cell = nn.GRUCell(81, 128)
        self.output = mlp(128, 128, 23)
        # A persistence initialization is useful, but trainable nonzero weights
        # preserve an initial action-to-outcome gradient path.
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, context, action, raw_last):
        b, horizon, _ = action.shape
        initial_pose = poses(raw_last)
        initial_quat = unit(initial_pose[..., 3:])
        finger = raw_last[:, 7:9]
        command = ((finger.mean(-1, keepdim=True) - .015) / .025).clamp(-1, 1)
        native_reference = torch.cat((raw_last[:, :7], command), -1)
        reference = 2 * (native_reference-self.action_min)/self.action_scale - 1
        hidden = torch.tanh(self.initial(context))
        cond = torch.tanh(self.context(context))
        pose_list, finger_list = [], []
        for j in range(horizon):
            phase = action.new_full((b, 1), j / max(horizon-1, 1))
            features = torch.cat((action[:, j], action[:, j]-reference, phase, cond), -1)
            hidden = self.cell(features, hidden)
            residual = self.output(hidden)
            dp = residual[:, :9].reshape(b, 3, 3)
            dq = residual[:, 9:21].reshape(b, 3, 4)
            predicted_p = initial_pose[..., :3] + dp * self.position_scale
            predicted_q = unit(initial_quat + dq)
            pose_list.append(torch.cat((predicted_p, predicted_q), -1))
            finger_list.append(finger + residual[:, 21:23] * self.finger_scale)
        return torch.stack(pose_list, 1), torch.stack(finger_list, 1)


class ContractDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.horizon = int(spec['training']['horizon'])
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scale = torch.tensor(n['std'], dtype=torch.float32)
        # Common original-data spatial scales, not per-slice fitted scales.
        self.register_buffer('relative_scale', torch.stack((scale[18:21], scale[25:28], scale[32:35])).amax(0))
        self.register_buffer('tolerance', torch.tensor([.06, .06, .011], dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(164, 256), nn.LayerNorm(256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU())
        self.event_head = mlp(256, 128, self.horizon * 2)
        self.readiness_head = mlp(256, 128, (self.horizon+1) * 2)
        nn.init.constant_(self.event_head[-1].bias, -2.0)
        self.backbone = DiffusionBackbone(256 + self.horizon*2 + 2, spec['training'])
        self.forecast = Forecast(n)

    def encode(self, history):
        normalized = (history-self.obs_mean)/self.obs_scale
        now = history[:, -1]
        m, _, protected, _, role = observed_status(history)
        obj = poses(now)[:, 1:]
        relative_goal = (goals(now)-obj[..., :3])/self.relative_scale
        relative_tcp = (now[:, None, 18:21]-obj[..., :3])/self.relative_scale
        width = ((now[:, 7:9]-self.obs_mean[7:9])/self.obs_scale[7:9]).mean(-1, keepdim=True)
        features = torch.cat((normalized.flatten(1), normalized[:, 1]-normalized[:, 0],
                              relative_goal.flatten(1), relative_tcp.flatten(1),
                              (m/self.relative_scale).flatten(1), protected, role, width), -1)
        context = self.encoder(features)
        event = self.event_head(context).reshape(-1, self.horizon, 2)
        readiness = self.readiness_head(context).reshape(-1, self.horizon+1, 2)
        return context, event, readiness

    def denoise(self, noisy, timestep, context, event, readiness):
        condition = torch.cat((context, event.sigmoid().flatten(1), readiness[:, 0].sigmoid()), -1)
        return self.backbone(noisy, timestep, condition)

    def forward(self, noisy_action, timestep, raw_history):
        context, event, readiness = self.encode(raw_history)
        return self.denoise(noisy_action, timestep, context, event, readiness)

    @torch.no_grad()
    def diagnostics(self, raw_history):
        _, event, readiness = self.encode(raw_history)
        m, at_goal, protected, role, _ = observed_status(raw_history)
        return {'live_margins_m': m, 'live_at_goal': at_goal,
                'live_task_complete': at_goal.all(-1),
                'observed_protection_support': protected,
                'nominated_object_id': role,
                'placement_window_probability': event.sigmoid(),
                'current_readiness_probability': readiness[:, 0].sigmoid(),
                'future_readiness_probability': readiness[:, 1:].sigmoid(),
                'observed_readiness_proxy': ready_proxy(raw_history[:, -1])}


def build_model(spec):
    return ContractDiffusion(spec)


def protection_error(predicted_pose, live_pose, predicted_margin, model):
    drift = ((predicted_pose[..., :3]-live_pose[:, None, :, :3]) /
             model.forecast.position_scale[1:]).square().mean(-1)
    dot = (unit(predicted_pose[..., 3:])*unit(live_pose[:, None, :, 3:])).sum(-1)
    rotation_drift = (1-dot.square()).clamp_min(0)
    contract = (F.relu(-predicted_margin)/model.tolerance).square().mean(-1)
    return drift + .1*rotation_drift + .1*contract


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    action = batch['encoded_action']
    future = batch['future_obs']
    valid = batch['future_mask'].to(action.dtype) * batch['mask'].to(action.dtype)
    context, event_logits, readiness_logits = model.encode(history)
    predicted_noise = model.denoise(batch['noisy_action'], batch['timesteps'], context,
                                    event_logits, readiness_logits)
    diffusion = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    now = history[:, -1]
    target_pose = poses(future)
    future_goal = goals(future)
    live_pose = poses(now)[:, 1:]
    _, _, protected, _, _ = observed_status(history)
    true_margin = margins(target_pose[:, :, 1:], future_goal)
    # These are synchronized placement-window labels, not synthetic failures.
    event_target = (true_margin >= 0).all(-1).to(action.dtype) * (1-protected[:, None])
    ready_current = ready_proxy(now)
    ready_future = ready_proxy(future)
    event_loss = average(F.binary_cross_entropy_with_logits(
        event_logits, event_target, reduction='none', pos_weight=action.new_tensor(3.0)), valid)
    current_ready_loss = F.binary_cross_entropy_with_logits(readiness_logits[:, 0], ready_current)
    future_ready_loss = average(F.binary_cross_entropy_with_logits(
        readiness_logits[:, 1:], ready_future, reduction='none'), valid)
    readiness_loss = .5*(current_ready_loss+future_ready_loss)

    pred_pose, pred_finger = model.forecast(context, action, now)
    position_error = ((pred_pose[..., :3]-target_pose[..., :3]) /
                      model.forecast.position_scale).square().mean(-1)
    qdot = (unit(pred_pose[..., 3:])*unit(target_pose[..., 3:])).sum(-1)
    rotation_error = (1-qdot.square()).clamp_min(0)
    finger_error = ((pred_finger-future[..., 7:9])/model.forecast.finger_scale).square()
    pair_mask = valid[:, 1:]*valid[:, :-1]
    predicted_change = pred_pose[:, 1:, :, :3]-pred_pose[:, :-1, :, :3]
    true_change = target_pose[:, 1:, :, :3]-target_pose[:, :-1, :, :3]
    change_error = ((predicted_change-true_change)/model.forecast.position_scale).square().mean(-1)
    rollout_loss = (average(position_error, valid) + .25*average(rotation_error, valid)
                    + .25*average(finger_error, valid) + .5*average(change_error, pair_mask))

    demo_margin = margins(pred_pose[:, :, 1:], future_goal)
    event_mask = event_target*valid
    demo_contract = average((F.relu(-demo_margin)/model.tolerance).square(), event_mask.unsqueeze(-1))
    protected_mask = protected[:, None, :]*valid
    demo_protection = average(protection_error(pred_pose[:, :, 1:], live_pose, demo_margin, model), protected_mask)

    # Amortized outcome guidance: the sampler stays fixed. Gradients from this
    # frozen action-conditioned rollout reach x0 -> epsilon, not its parameters.
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp(1e-5, 1)
    clean = (batch['noisy_action']-(1-alpha).sqrt()*predicted_noise)/alpha.sqrt()
    bounded_clean = clean.clamp(-1, 1)
    detached_state = {name: value.detach() for name, value in model.forecast.named_parameters()}
    detached_state.update({name: value.detach() for name, value in model.forecast.named_buffers()})
    guided_pose, _ = torch.func.functional_call(
        model.forecast, detached_state, (context.detach(), bounded_clean, now.detach()))
    guided_margin = margins(guided_pose[:, :, 1:], goals(now)[:, None])

    # Do not exploit the rollout far from demonstrations or at high DDPM noise.
    local_action_error = (bounded_clean-action).square().mean(-1, keepdim=True)
    locality = torch.exp(-local_action_error.detach()/.04)
    calibration = torch.exp(-position_error.detach().mean(-1, keepdim=True)/.02)
    reliable = (alpha >= .5).to(action.dtype) * alpha.square() * locality * calibration
    event_gate = (2*event_logits.detach().sigmoid()-1).clamp(0, 1)
    action_contract_values = (F.relu(-guided_margin)/model.tolerance).square()
    action_contract = average(action_contract_values * (reliable*event_gate).unsqueeze(-1), event_mask.unsqueeze(-1))
    action_protection = average(protection_error(guided_pose[:, :, 1:], live_pose,
                                                guided_margin, model)*reliable, protected_mask)
    trust = average((clean-action).square() * alpha.square() * (alpha >= .5).to(action.dtype), batch['mask'])
    prior = (.5*rollout_loss + .05*event_loss + .025*readiness_loss
             + .01*demo_contract + .05*demo_protection
             + .02*action_contract + .05*action_protection + .02*trust)
    return {'loss': diffusion+prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'rollout_loss': rollout_loss, 'placement_event_loss': event_loss,
            'readiness_loss': readiness_loss, 'demo_contract_loss': demo_contract,
            'action_contract_loss': action_contract, 'protected_action_loss': action_protection,
            'trust_loss': trust}
