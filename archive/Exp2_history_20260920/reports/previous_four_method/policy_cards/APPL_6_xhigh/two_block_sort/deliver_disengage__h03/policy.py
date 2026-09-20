import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


N_PHASE = 8


def masked_mean(value, mask):
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def interpolate(values, positions, queries):
    # Piecewise-linear resampling. Only interval selection is nondifferentiable;
    # interpolation weights train both the values and the duration coordinates.
    index = torch.searchsorted(positions.detach().contiguous(), queries.detach().contiguous(), right=True)
    index = index.clamp(1, positions.shape[1] - 1)
    lo = index - 1
    left = positions.gather(1, lo)
    right = positions.gather(1, index)
    weight = ((queries - left) / (right - left).clamp_min(1e-5)).clamp(0, 1)
    a = values.gather(1, lo.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    b = values.gather(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    return a + weight.unsqueeze(-1) * (b - a)


class EventDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('action_min', torch.tensor(n['action_min'], dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(n['action_scale'], dtype=torch.float32))
        self.register_buffer('phase_ids', torch.arange(8, dtype=torch.float32))
        self.step_encoder = nn.Sequential(nn.Linear(94, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.recurrent = nn.GRU(128, 128, batch_first=True)
        self.context = nn.Sequential(nn.Linear(222, 192), nn.SiLU(), nn.Linear(192, 128), nn.SiLU())
        self.role = nn.Linear(128, 2)
        self.events = nn.Sequential(nn.Linear(130, 128), nn.SiLU(), nn.Linear(128, 48))
        self.knots = nn.Sequential(nn.Linear(178, 256), nn.SiLU(), nn.Linear(256, 8 * 6 * 7))
        self.initial_open = nn.Linear(128, 1)
        self.hazards = nn.Sequential(nn.Linear(145, 96), nn.SiLU(), nn.Linear(96, 2))
        nn.init.constant_(self.hazards[-1].bias, -4.0)
        self.backbone = DiffusionBackbone(450, spec['training'])
        self.time_mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))
        self.refine_condition = nn.Linear(178 + 64, 64)
        self.arm_refine = nn.Sequential(nn.Conv1d(25 + 64, 128, 3, padding=1), nn.SiLU(),
                                        nn.Conv1d(128, 128, 3, padding=1), nn.SiLU(),
                                        nn.Conv1d(128, 7, 1))
        self.grip_score = nn.Sequential(nn.Conv1d(25 + 64, 96, 3, padding=1), nn.SiLU(),
                                       nn.Conv1d(96, 96, 3, padding=1), nn.SiLU(),
                                       nn.Conv1d(96, 1, 1))
        nn.init.zeros_(self.arm_refine[-1].weight)
        nn.init.zeros_(self.arm_refine[-1].bias)

    def encode_action(self, action):
        return 2 * (action - self.action_min) / self.action_scale - 1

    def belief(self, history):
        obs = (history - self.obs_mean) / self.obs_scale
        change = torch.cat((torch.zeros_like(obs[:, :1]), obs[:, 1:] - obs[:, :-1]), dim=1)
        step = self.step_encoder(torch.cat((obs, change), dim=-1))
        _, hidden = self.recurrent(step)
        ctx = self.context(torch.cat((hidden[-1], obs.flatten(1)), dim=-1))
        role_logits = self.role(ctx)
        role = role_logits.softmax(-1)
        out = self.events(torch.cat((ctx, role), dim=-1)).reshape(-1, 6, 8)
        logits = out[:, 0]
        phase = logits.softmax(-1)
        u = 0.005 + 0.99 * out[:, 1].sigmoid()
        duration = 0.10 + F.softplus(out[:, 2])
        remaining = 0.025 + F.softplus(out[:, 3])
        sigma_u = 0.03 + 0.47 * out[:, 4].sigmoid()
        sigma_r = 0.10 + 1.40 * out[:, 5].sigmoid()
        cond = torch.cat((ctx, role, phase, u, duration.log(), remaining.log(), sigma_u, sigma_r), dim=-1)
        return dict(ctx=ctx, role_logits=role_logits, phase_logits=logits, phase=phase, u=u,
                    duration=duration, remaining=remaining, sigma_u=sigma_u, sigma_r=sigma_r, cond=cond)

    def time_decode(self, belief, horizon):
        p, u, d, r = (belief[k] for k in ('phase', 'u', 'duration', 'remaining'))
        b = p.shape[0]
        cumulative = d.cumsum(-1)
        previous = cumulative - d
        effective_current = r / (1 - u).clamp_min(0.005)
        ids = self.phase_ids
        following = ids.view(1, 1, 8) >= ids.view(1, 8, 1)
        ends_after = cumulative[:, None, :] - cumulative[:, :, None] + r[:, :, None]
        ends_before = cumulative[:, None, :] - previous[:, :, None] - (u * effective_current)[:, :, None]
        ends = torch.where(following, ends_after, ends_before)
        first_duration = d[:, :1].expand(-1, 8)
        first_duration = torch.cat((effective_current[:, :1], first_duration[:, 1:]), dim=1)
        starts = torch.cat((ends[:, :, :1] - first_duration[:, :, None], ends[:, :, :-1]), dim=-1)
        times = (torch.arange(horizon, device=p.device, dtype=p.dtype) - 1) * 0.05
        query = times.view(1, 1, horizon, 1)
        gates = torch.sigmoid((query - ends[:, :, None, :7]) / 0.035)
        occupancy = torch.cat((1 - gates[..., :1], gates[..., :-1] - gates[..., 1:], gates[..., -1:]), dim=-1)
        within = ((query - starts[:, :, None, :]) / (ends - starts)[:, :, None, :].clamp_min(0.025)).clamp(0, 1)
        weight = occupancy * p[:, :, None, None]
        phase_path = weight.sum(1)
        u_path = (weight * within).sum(1) / phase_path.clamp_min(1e-6)
        coordinate = (weight * (ids.view(1, 1, 1, 8) + within)).sum((1, 3))
        relative = (coordinate - coordinate[:, :1]) / (coordinate[:, -1:] - coordinate[:, :1]).clamp_min(1e-4)
        linear = torch.linspace(0, 1, horizon, device=p.device, dtype=p.dtype)[None].expand(b, -1)
        # Physical-time floor prevents collapse in terminal dwell.
        coords = 0.9 * relative + 0.1 * linear
        coords = (coords - coords[:, :1]) / (coords[:, -1:] - coords[:, :1]).clamp_min(1e-5)
        return phase_path, u_path, coords, times

    def reference(self, history, belief, phase_path, u_path, times):
        b, h, _ = phase_path.shape
        raw_knots = self.knots(belief['cond']).reshape(b, 8, 6, 7)
        anchor = self.encode_action(torch.cat((history[:, -1, :7], history.new_zeros(b, 1)), dim=-1))[:, :7]
        knot_actions = anchor[:, None, None, :] + raw_knots
        position = (u_path * 5).clamp(0, 5)
        low = position.floor().long().clamp(0, 4)
        frac = position - low
        expanded = knot_actions[:, None].expand(-1, h, -1, -1, -1)
        left = expanded.gather(3, low[..., None, None].expand(-1, -1, -1, 1, 7)).squeeze(3)
        right = expanded.gather(3, (low + 1)[..., None, None].expand(-1, -1, -1, 1, 7)).squeeze(3)
        arm = ((left + frac[..., None] * (right - left)) * phase_path[..., None]).sum(2)
        hc = torch.cat((belief['ctx'][:, None].expand(-1, h, -1), phase_path, u_path,
                        times.view(1, h, 1).expand(b, -1, -1)), dim=-1)
        hazard_logits = self.hazards(hc)
        hazard = hazard_logits.sigmoid()
        open_probability = self.initial_open(belief['ctx']).sigmoid()
        probabilities = [open_probability]
        for j in range(1, h):
            close = hazard[:, j, :1]
            opening = hazard[:, j, 1:2]
            open_probability = open_probability * (1 - close) + (1 - open_probability) * opening
            probabilities.append(open_probability)
        grip = torch.cat(probabilities, dim=1).unsqueeze(-1)
        plan = torch.cat((arm, 2 * grip - 1), dim=-1)
        return plan, grip, hazard_logits, raw_knots

    def predict(self, noisy_action, timestep, history):
        belief = self.belief(history)
        h = noisy_action.shape[1]
        phases, progress, coords, times = self.time_decode(belief, h)
        plan, grip, hazard_logits, knots = self.reference(history, belief, phases, progress, times)
        condition = torch.cat((belief['cond'], plan.flatten(1), coords, phases.flatten(1)), dim=-1)
        grid = torch.linspace(0, 1, h, device=noisy_action.device, dtype=noisy_action.dtype)[None].expand(noisy_action.shape[0], -1)
        warped = interpolate(noisy_action, coords, grid)
        warped_epsilon = self.backbone(warped, timestep, condition)
        epsilon = interpolate(warped_epsilon, grid, coords)
        t = torch.as_tensor(timestep, device=noisy_action.device).reshape(-1).expand(noisy_action.shape[0]).to(noisy_action.dtype)
        frequency = torch.exp(torch.linspace(0, -math.log(10000), 16, device=t.device, dtype=t.dtype))
        angles = t[:, None] * frequency[None]
        te = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))
        local_condition = F.silu(self.refine_condition(torch.cat((belief['cond'], te), dim=-1)))
        features = torch.cat((noisy_action, epsilon, plan, coords[..., None], local_condition[:, None].expand(-1, h, -1)), dim=-1).transpose(1, 2)
        arm_epsilon = epsilon[..., :7] + self.arm_refine(features).transpose(1, 2)
        # Separate original-time gripper score avoids interpolating its output.
        gripper_epsilon = self.grip_score(features).transpose(1, 2)
        result = torch.cat((arm_epsilon, gripper_epsilon), dim=-1)
        aux = dict(belief=belief, phases=phases, progress=progress, plan=plan, grip=grip,
                   hazards=hazard_logits, knots=knots)
        return result, aux

    def forward(self, noisy_action, timestep, raw_history):
        return self.predict(noisy_action, timestep, raw_history)[0]


def build_model(spec):
    return EventDiffusion(spec)


@torch.no_grad()
def annotate(batch):
    # Training-only local event evidence. No trajectory IDs or source indices.
    history = batch['raw_obs']
    pre = torch.cat((history[:, :1], batch['future_obs'][:, :-1]), dim=1)
    post = batch['future_obs']
    current = history[:, -1]
    red_distance = torch.linalg.vector_norm(current[:, 25:27] - current[:, 41:43], dim=-1)
    blue_distance = torch.linalg.vector_norm(current[:, 32:34] - current[:, 44:46], dim=-1)
    red_done = (red_distance < 0.035) & ((current[:, 27] - current[:, 43]).abs() < 0.012)
    blue_done = (blue_distance < 0.035) & ((current[:, 34] - current[:, 46]).abs() < 0.012)
    width = current[:, 7:9].mean(-1)
    blue_near = torch.linalg.vector_norm(current[:, 18:20] - current[:, 32:34], dim=-1) < 0.09
    # Preserve red during open high blue approach, and do not switch to blue
    # simply because red reaches the pad while still grasped.
    blue_role = red_done & ((blue_near & ((width < 0.030) | (current[:, 20] < 0.18))) | (current[:, 34] > 0.05) | blue_done)
    choose = blue_role[:, None, None]
    obj = torch.where(choose, pre[:, :, 32:35], pre[:, :, 25:28])
    obj_after = torch.where(choose, post[:, :, 32:35], post[:, :, 25:28])
    goal = torch.where(choose, pre[:, :, 44:47], pre[:, :, 41:44])
    tcp = pre[:, :, 18:21]
    vz = (obj_after[..., 2] - obj[..., 2]) / 0.05
    tcp_vz = (post[:, :, 20] - pre[:, :, 20]) / 0.05
    horizontal = torch.linalg.vector_norm(obj[..., :2] - goal[..., :2], dim=-1)
    height = obj[..., 2] - goal[..., 2]
    close_command = batch['native_action'][..., 7] < 0
    near_goal = horizontal < 0.035
    down = near_goal & ((height < 0.245) | (vz < -0.012))
    close_dwell = (height < 0.018) & (vz < 0.018)
    lifting = (height < 0.245) | (vz > 0.035)
    phase = torch.zeros_like(horizontal, dtype=torch.long)
    phase = torch.where(close_command, torch.full_like(phase, 3), phase)
    phase = torch.where(close_command & lifting, torch.full_like(phase, 2), phase)
    phase = torch.where(close_command & close_dwell, torch.full_like(phase, 1), phase)
    phase = torch.where(close_command & down, torch.full_like(phase, 4), phase)
    placed_open = (~close_command) & near_goal & (height.abs() < 0.014)
    phase = torch.where(placed_open, torch.full_like(phase, 5), phase)
    phase = torch.where(placed_open & (tcp[..., 2] > 0.043), torch.full_like(phase, 6), phase)
    final = placed_open & (tcp[..., 2] > 0.25) & (tcp_vz < 0.018)
    phase = torch.where(final, torch.full_like(phase, 7), phase)
    finger = pre[:, :, 7:9].mean(-1)
    approach_u = 1 - (tcp[..., 2] - obj[..., 2]).clamp_min(0) / 0.26
    closing_u = (0.04 - finger) / (0.04 - 0.0183)
    lifting_u = height / 0.26
    transport_u = 1 - horizontal / 0.25
    lowering_u = 1 - height / 0.27
    opening_u = (finger - 0.0183) / (0.04 - 0.0183)
    retreat_u = (tcp[..., 2] - goal[..., 2]) / 0.28
    next_distance = torch.linalg.vector_norm(tcp[..., :2] - pre[:, :, 32:34], dim=-1)
    travel_length = torch.linalg.vector_norm(pre[:, :, 41:43] - pre[:, :, 32:34], dim=-1).clamp_min(0.1)
    next_u = torch.where(blue_role[:, None], torch.ones_like(next_distance), 1 - next_distance / travel_length)
    all_u = torch.stack((approach_u, closing_u, lifting_u, transport_u, lowering_u, opening_u, retreat_u, next_u), dim=-1).clamp(0.01, 0.99)
    progress = all_u.gather(-1, phase[..., None]).squeeze(-1)
    mask = (batch['mask'] * batch['future_mask']).squeeze(-1)
    onehot = F.one_hot(phase, 8).to(pre.dtype)
    soft = 0.8 * onehot * mask[..., None]
    mass = 0.8 * mask[..., None]
    soft[:, 1:] += 0.1 * onehot[:, :-1] * mask[:, :-1, None]
    mass[:, 1:] += 0.1 * mask[:, :-1, None]
    soft[:, :-1] += 0.1 * onehot[:, 1:] * mask[:, 1:, None]
    mass[:, :-1] += 0.1 * mask[:, 1:, None]
    soft = soft / mass.clamp_min(1e-6)
    return dict(role=blue_role.long(), phase=phase, soft=soft, u=progress, mask=mask)


def compute_loss(model, batch, spec):
    labels = annotate(batch)
    raw = batch['raw_obs']
    truncate = torch.rand(raw.shape[0], 1, 1, device=raw.device) < 0.15
    history = torch.where(truncate, raw[:, -1:].expand_as(raw), raw)
    prediction, aux = model.predict(batch['noisy_action'], batch['timesteps'], history)
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    belief = aux['belief']
    m = labels['mask']
    current_mask = m[:, 1]
    phase = labels['phase']
    current_phase = phase[:, 1]
    soft = labels['soft']
    phase_ce = masked_mean(-(soft[:, 1] * belief['phase_logits'].log_softmax(-1)).sum(-1), current_mask)
    path_ce = masked_mean(-(soft * aux['phases'].clamp_min(1e-6).log()).sum(-1), m)
    role_ce = masked_mean(F.cross_entropy(belief['role_logits'], labels['role'], reduction='none'), current_mask)
    selected_u = belief['u'].gather(1, current_phase[:, None]).squeeze(1)
    sigma_u = belief['sigma_u'].gather(1, current_phase[:, None]).squeeze(1)
    error_u = selected_u - labels['u'][:, 1]
    u_nll = masked_mean(0.5 * (error_u / sigma_u).square() + (sigma_u / 0.03).log(), current_mask)
    path_u = aux['progress'].gather(-1, phase[..., None]).squeeze(-1)
    progress_loss = masked_mean((path_u - labels['u']).square(), m)
    slots = torch.arange(m.shape[1], device=m.device)[None]
    later = (slots > 1) & (m > 0)
    advanced = later & (phase > current_phase[:, None])
    hit_index = torch.where(advanced, slots.expand_as(phase), torch.full_like(phase, m.shape[1])).min(-1).values
    hit = hit_index < m.shape[1]
    seconds = ((hit_index.to(m.dtype) - 1) * 0.05).clamp_min(0.025)
    observed_time = torch.where(later, (slots - 1).to(m.dtype) * 0.05, torch.zeros_like(m)).max(-1).values
    remaining = belief['remaining'].gather(1, current_phase[:, None]).squeeze(1)
    sigma_r = belief['sigma_r'].gather(1, current_phase[:, None]).squeeze(1)
    event_nll = 0.5 * ((remaining.log() - seconds.log()) / sigma_r).square() + (sigma_r / 0.1).log()
    censored = F.relu(observed_time.clamp_min(0.025).log() - remaining.log()).square()
    duration_loss = masked_mean(torch.where(hit, event_nll, censored), current_mask * (observed_time > 0).to(m.dtype))
    du = labels['u'] - labels['u'][:, 1:2]
    same = (phase == current_phase[:, None]) & later & (du > 0.04)
    elapsed = ((slots - 1).to(m.dtype) * 0.05).expand_as(m)
    duration_target = (elapsed / du.clamp_min(0.01)).clamp(0.15, 12.0)
    nominal = belief['duration'].gather(1, current_phase[:, None])
    rate_loss = masked_mean((nominal.log() - duration_target.log()).square(), same.to(m.dtype) * m)
    action_mask = batch['mask']
    arm_reference = masked_mean((aux['plan'][..., :7] - batch['encoded_action'][..., :7]).square(), action_mask)
    grip_target = (batch['native_action'][..., 7:8] > 0).to(raw.dtype)
    grip_bce = masked_mean(F.binary_cross_entropy(aux['grip'].clamp(1e-5, 1 - 1e-5), grip_target, reduction='none'), action_mask)
    previous_open = grip_target[:, :-1, 0]
    next_open = grip_target[:, 1:, 0]
    targets = torch.stack((1 - next_open, next_open), dim=-1)
    eligible = torch.stack((previous_open, 1 - previous_open), dim=-1)
    switch_mask = action_mask[:, 1:] * action_mask[:, :-1] * eligible
    switch_loss = masked_mean(F.binary_cross_entropy_with_logits(aux['hazards'][:, 1:], targets, reduction='none') * (1 + 5 * targets), switch_mask)
    pair8 = batch['future_obs'][:, 6:8]
    pair15 = batch['future_obs'][:, 13:15]
    future_belief = model.belief(torch.cat((pair8, pair15), dim=0))
    future_soft = torch.cat((soft[:, 8], soft[:, 15]), dim=0)
    future_valid = torch.cat((m[:, 7] * m[:, 8], m[:, 14] * m[:, 15]), dim=0)
    future_phase_ce = masked_mean(-(future_soft * future_belief['phase_logits'].log_softmax(-1)).sum(-1), future_valid)
    expected = (belief['phase'] * model.phase_ids).sum(-1)
    future_expected = (future_belief['phase'] * model.phase_ids).sum(-1)
    e8, e15 = future_expected.chunk(2)
    order1 = F.relu(expected - e8 - 0.15).square() + F.relu(e8 - expected - 2.0).square()
    order2 = F.relu(e8 - e15 - 0.15).square() + F.relu(e15 - e8 - 2.0).square()
    order_loss = masked_mean(order1, current_mask * m[:, 7] * m[:, 8]) + masked_mean(order2, m[:, 7] * m[:, 8] * m[:, 14] * m[:, 15])
    knot_curve = (aux['knots'][:, :, 2:] - 2 * aux['knots'][:, :, 1:-1] + aux['knots'][:, :, :-2]).square().mean()
    prior = (0.08 * phase_ce + 0.04 * path_ce + 0.04 * role_ce + 0.015 * u_nll
             + 0.08 * progress_loss + 0.02 * duration_loss + 0.01 * rate_loss
             + 0.35 * arm_reference + 0.12 * grip_bce + 0.035 * switch_loss
             + 0.025 * future_phase_ce + 0.005 * order_loss + 0.0005 * knot_curve)
    return dict(loss=diffusion + prior, diffusion_loss=diffusion, prior_loss=prior,
                phase_loss=phase_ce, duration_loss=duration_loss, reference_loss=arm_reference,
                gripper_loss=grip_bce)
