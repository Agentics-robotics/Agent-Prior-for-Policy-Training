import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


def objects(x):
    return torch.stack((x[..., 25:28], x[..., 32:35]), dim=-2)


def goals(x):
    return torch.stack((x[..., 41:44], x[..., 44:47]), dim=-2)


def containment(x):
    # For a cube, extrema of the eight rotated corners equal .02 * row L1 norm.
    q = torch.stack((x[..., 28:32], x[..., 35:39]), dim=-2)
    q = q / q.square().sum(-1, keepdim=True).sqrt().clamp_min(1.e-8)
    w, a, b, c = q.unbind(-1)
    rx = torch.stack((1 - 2*(b*b+c*c), 2*(a*b-w*c), 2*(a*c+w*b)), -1)
    ry = torch.stack((2*(a*b+w*c), 1 - 2*(a*a+c*c), 2*(b*c-w*a)), -1)
    ext = .02 * torch.stack((rx.abs().sum(-1), ry.abs().sum(-1)), -1)
    d = (objects(x) - goals(x)).abs()
    margins = torch.cat((.06 - d[..., :2] - ext, .011 - d[..., 2:3]), -1)
    return margins, (margins >= 0).all(-1)


def causal_labels(x, prev):
    p = objects(x)
    rel = p - x[..., None, 18:21]
    dist = rel.square().sum(-1).sqrt()
    near = dist < .045
    finger = x[..., 7:9].mean(-1)
    old_finger = prev[..., 7:9].mean(-1)
    closed = finger < .024
    _, contained = containment(x)
    settled = (p[..., 2] - goals(x)[..., 2]).abs() < .003
    attached = near & closed.unsqueeze(-1)
    phase = torch.zeros_like(dist, dtype=torch.long)
    phase = torch.where(attached, torch.ones_like(phase), phase)
    phase = torch.where(near & contained & (finger >= .024).unsqueeze(-1),
                        torch.full_like(phase, 3), phase)
    widening = ((finger - old_finger) > .0005) & (finger < .0375)
    release_area = ((p[..., :2] - goals(x)[..., :2]).abs() < .06).all(-1)
    phase = torch.where(near & release_area & widening.unsqueeze(-1),
                        torch.full_like(phase, 2), phase)
    phase = torch.where(contained & ~near, torch.full_like(phase, 4), phase)
    # Weak intended/pending role labels, not an action controller. Held identity
    # has priority even after containment becomes true.
    local_work = near & (closed.unsqueeze(-1) | ~settled | (finger < .037).unsqueeze(-1))
    unfinished = ~contained
    local_choice = dist.masked_fill(~local_work, 100).argmin(-1)
    todo_choice = dist.masked_fill(~unfinished, 100).argmin(-1)
    role = torch.where(local_work.any(-1), local_choice,
                       torch.where(unfinished.any(-1), todo_choice, torch.full_like(todo_choice, 2)))
    ids = torch.arange(2, device=x.device)
    pending_mask = unfinished & (ids != role.unsqueeze(-1))
    pending = dist.masked_fill(~pending_mask, 100).argmin(-1)
    pending = torch.where(pending_mask.any(-1), pending, torch.full_like(pending, 2))
    att_mask = attached | ~near | (finger > .034).unsqueeze(-1)
    return dict(phase=phase, attached=attached, att_mask=att_mask, contained=contained,
                settled=settled, near=near, dist=dist, finger=finger, role=role,
                pending=pending, rel=rel)


class ReleaseBeliefPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.horizon = spec['training']['horizon']
        self.register_buffer('obs_mean', torch.tensor(spec['normalizer']['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(spec['normalizer']['std'], dtype=torch.float32))
        self.register_buffer('forecast_indices', torch.tensor([18, 19, 20, 25, 26, 27, 32, 33, 34, 7, 8]))
        self.frame = nn.Sequential(nn.Linear(118, 160), nn.LayerNorm(160), nn.SiLU(),
                                   nn.Linear(160, 160), nn.SiLU())
        self.initial_hidden = nn.Parameter(torch.zeros(192))
        self.recurrent = nn.GRUCell(160, 192)
        self.initial_phase_logits = nn.Parameter(torch.zeros(2, 5))
        self.phase_emission = nn.Linear(192, 10)
        self.phase_update = nn.Linear(352, 2)
        self.attachment_head = nn.Linear(202, 2)
        self.containment_head = nn.Linear(192, 2)
        self.readiness_head = nn.Sequential(nn.Linear(202, 96), nn.SiLU(), nn.Linear(96, 3))
        self.role_head = nn.Linear(202, 3)
        self.pending_head = nn.Linear(202, 3)
        self.hazard_head = nn.Sequential(nn.Linear(202, 128), nn.SiLU(),
                                         nn.Linear(128, (self.horizon-1)*2))
        self.forecast_head = nn.Sequential(nn.Linear(202, 192), nn.SiLU(),
                                           nn.Linear(192, self.horizon*11))
        # h, current frame, normalized raw skip, phase, attachment, containment,
        # readiness, intended role, pending role and release hazards.
        dim = 192 + 160 + 94 + 10 + 2 + 2 + 3 + 3 + 3 + (self.horizon-1)*2
        self.condition = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.SiLU(),
                                       nn.Linear(256, 256))
        self.diffusion = DiffusionBackbone(256, spec['training'])

    def encode(self, history):
        norm = (history - self.obs_mean) / self.obs_scale
        change = torch.cat((torch.zeros_like(norm[:, :1]), norm[:, 1:] - norm[:, :-1]), dim=1)
        pos = objects(history)
        goal = goals(history)
        tcp = history[..., None, 18:21]
        scale = self.obs_scale[18:21]
        rel = torch.cat(((pos-tcp).flatten(-2), (goal-pos).flatten(-2),
                         (goal-tcp).flatten(-2)), dim=-1)
        rel = rel / scale.repeat(6)
        margins, _ = containment(history)
        geom = (margins / scale).flatten(-2)
        feat = self.frame(torch.cat((norm, change, rel, geom), dim=-1))
        hidden = self.initial_hidden.unsqueeze(0).expand(history.shape[0], -1)
        phase = self.initial_phase_logits.softmax(-1).unsqueeze(0).expand(history.shape[0], -1, -1)
        phase_seq = []
        for j in range(history.shape[1]):
            hidden = self.recurrent(feat[:, j], hidden)
            emission = self.phase_emission(hidden).reshape(-1, 2, 5).softmax(-1)
            update = self.phase_update(torch.cat((hidden, feat[:, j]), -1)).sigmoid().unsqueeze(-1)
            phase = (1-update)*phase + update*emission
            phase_seq.append(phase)
        state = torch.cat((hidden, phase.flatten(1)), -1)
        att = self.attachment_head(state)
        cont = self.containment_head(hidden)
        ready = self.readiness_head(state)
        role = self.role_head(state)
        pending = self.pending_head(state)
        hazard = self.hazard_head(state).reshape(-1, self.horizon-1, 2)
        forecast = self.forecast_head(state).reshape(-1, self.horizon, 11)
        cond = self.condition(torch.cat((hidden, feat[:, -1], norm.flatten(1), phase.flatten(1),
                                        att.sigmoid(), cont.sigmoid(), ready.sigmoid(),
                                        role.softmax(-1), pending.softmax(-1),
                                        hazard.sigmoid().flatten(1)), -1))
        aux = dict(phase=torch.stack(phase_seq, 1), attachment=att, containment=cont,
                   readiness=ready, role=role, pending=pending, hazard=hazard, forecast=forecast)
        return cond, aux

    def forward(self, noisy_action, timestep, raw_history):
        condition, _ = self.encode(raw_history)
        return self.diffusion(noisy_action, timestep, condition)

    def beliefs(self, raw_history):
        # Optional diagnostics. The mandatory forward returns only epsilon;
        # the framework does not automatically route these estimates to a caller.
        _, a = self.encode(raw_history)
        _, truth = containment(raw_history[:, -1])
        return dict(phase=a['phase'][:, -1], attachment=a['attachment'].sigmoid(),
                    containment_estimate=a['containment'].sigmoid(), contract_per_object=truth,
                    disengagement=a['readiness'][:, :2].sigmoid(),
                    empty_hand_readiness=a['readiness'][:, 2].sigmoid(),
                    intended_role=a['role'].softmax(-1), pending_role=a['pending'].softmax(-1),
                    release_hazard=a['hazard'].sigmoid())


def build_model(spec):
    return ReleaseBeliefPolicy(spec)


def masked_mean(x, mask):
    mask = mask.to(x.dtype).expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    # Missing-history training only. No mutable state survives repeated DDPM calls.
    history = raw
    if model.training:
        missing = (torch.rand(raw.shape[0], 1, 1, device=raw.device) < .15)
        first = torch.where(missing, raw[:, -1:], raw[:, :1])
        history = torch.cat((first, raw[:, -1:]), dim=1)
    condition, a = model.encode(history)
    eps = model.diffusion(batch['noisy_action'], batch['timesteps'], condition)
    diffusion_loss = epsilon_loss(eps, batch['noise'], batch['mask'])
    future = batch['future_obs']
    valid = batch['future_mask'].squeeze(-1) > 0
    # Slot zero is the current state (after action t-1), not an unseen future.
    with torch.no_grad():
        current = raw[:, -1]
        labels = causal_labels(current, raw[:, 0])
        old = causal_labels(raw[:, 0], raw[:, 0])
        phase_target = torch.stack((old['phase'], labels['phase']), 1)
        current_pos = objects(current)
        fp = objects(future)
        delta_obj = fp - current_pos[:, None]
        delta_tcp = future[..., 18:21] - current[:, None, 18:21]
        distance_obj = delta_obj.square().sum(-1).sqrt()
        persistence = (distance_obj < .004) | ~valid.unsqueeze(-1)
        persistence = persistence.all(1)
        enough = valid[:, 1:].sum(1) >= 3
        future_motion = (delta_tcp.square().sum(-1).sqrt() > .003) & valid
        decoupled = ((distance_obj < .004) & future_motion.unsqueeze(-1)).any(1)
        wide = labels['finger'] > .035
        relative_clear = labels['rel'].square().sum(-1).sqrt() > .006
        clear_now = ~labels['near'] | (wide.unsqueeze(-1) & relative_clear)
        confirm = ~labels['near'] | decoupled
        positive = labels['contained'] & labels['settled'] & persistence & clear_now & confirm
        negative = (~labels['contained'] | ~labels['settled'] | labels['attached'] | ~persistence)
        ready_mask = enough.unsqueeze(-1) & (positive | negative)
        protected_ok = (persistence | ~labels['contained']).all(-1)
        empty_positive = wide & ~labels['attached'].any(-1) & protected_ok
        empty_negative = (labels['finger'] < .026) | ~protected_ok
        ready_target = torch.cat((positive.float(), empty_positive.float().unsqueeze(-1)), -1)
        ready_valid = torch.cat((ready_mask, (enough & (empty_positive | empty_negative)).unsqueeze(-1)), -1)
        forecast_target = (future.index_select(-1, model.forecast_indices) -
                           current.index_select(-1, model.forecast_indices)[:, None])
        forecast_target = forecast_target / model.obs_scale.index_select(0, model.forecast_indices)
        # Survival likelihood of the first demonstrated open command; windows
        # without an event are right-censored, not labelled as a later event.
        grip = batch['native_action'][..., 7]
        event = ((grip[:, 1:] > 0) & (grip[:, :-1] <= 0)).float()
        before_or_at = (event.cumsum(1) - event) == 0
        am = batch['mask'].squeeze(-1) > 0
        at_risk = labels['attached'] & (grip[:, 0] <= 0).unsqueeze(-1)
        hazard_valid = ((am[:, 1:] & am[:, :-1] & before_or_at).unsqueeze(-1) & at_risk[:, None])
        hazard_target = event.unsqueeze(-1).expand_as(a['hazard'])
        _, future_contained = containment(future)
        future_rel = objects(future) - future[..., None, 18:21]
        future_near = future_rel.square().sum(-1).sqrt() < .045
        future_finger = future[..., 7:9].mean(-1)
        # Penalize re-closing only on the demonstrated detached/open branch.
        # This cannot inhibit closing at the next unplaced source.
        future_held = future_near & (future_finger < .024).unsqueeze(-1)
        detached_branch = (future_contained.any(-1) & ~future_held.any(-1) &
                           (future_finger > .030) & (grip > 0) & valid & am)

    phase_loss = F.nll_loss(a['phase'].clamp_min(1.e-6).log().reshape(-1, 5), phase_target.reshape(-1))
    attachment_loss = masked_mean(F.binary_cross_entropy_with_logits(
        a['attachment'], labels['attached'].float(), reduction='none'), labels['att_mask'])
    containment_loss = F.binary_cross_entropy_with_logits(a['containment'], labels['contained'].float())
    readiness_loss = masked_mean(F.binary_cross_entropy_with_logits(a['readiness'], ready_target,
                                                                   reduction='none'), ready_valid)
    calibration = (a['containment'].sigmoid()-labels['contained'].float()).square().mean()
    calibration = calibration + masked_mean((a['readiness'].sigmoid()-ready_target).square(), ready_valid)
    role_loss = .5*(F.cross_entropy(a['role'], labels['role']) +
                    F.cross_entropy(a['pending'], labels['pending']))
    hazard_loss = masked_mean(F.binary_cross_entropy_with_logits(a['hazard'], hazard_target,
                                                                reduction='none'), hazard_valid)
    fm = valid.unsqueeze(-1)
    forecast_loss = masked_mean(F.smooth_l1_loss(a['forecast'], forecast_target, reduction='none'), fm)
    # Explicit future TCP/object relative-motion forecast loss, with world-axis
    # scales inherited from the complete-demonstration normalizer.
    fs = model.obs_scale.index_select(0, model.forecast_indices)
    pred_native = a['forecast'] * fs
    target_native = forecast_target * fs
    pred_relative = torch.cat((pred_native[..., 3:6]-pred_native[..., :3],
                               pred_native[..., 6:9]-pred_native[..., :3]), -1)
    target_relative = torch.cat((target_native[..., 3:6]-target_native[..., :3],
                                 target_native[..., 6:9]-target_native[..., :3]), -1)
    rel_scale = model.obs_scale[18:21].repeat(2)
    motion_loss = masked_mean(F.smooth_l1_loss(pred_relative/rel_scale, target_relative/rel_scale,
                                             reduction='none'), fm)
    same = (phase_target[:, 0] == phase_target[:, 1]).unsqueeze(-1)
    persistence_loss = masked_mean((a['phase'][:, 1]-a['phase'][:, 0]).square(), same)
    alpha = batch['alpha_bar'].reshape(-1, 1, 1)
    x0 = (batch['noisy_action'] - (1-alpha).sqrt()*eps) / alpha.sqrt().clamp_min(1.e-5)
    # Native gripper [-1,1] has the identical shared normalized coordinate.
    open_error = F.relu(.8-x0[..., 7]).square()*alpha.squeeze(-1)
    open_mask = detached_branch & (alpha.reshape(-1, 1) > .1)
    no_reclose = masked_mean(open_error, open_mask)
    prior_loss = (.10*phase_loss + .05*attachment_loss + .05*containment_loss +
                  .08*readiness_loss + .02*calibration + .025*role_loss +
                  .03*hazard_loss + .10*forecast_loss + .05*motion_loss +
                  .02*persistence_loss + .05*no_reclose)
    return dict(loss=diffusion_loss+prior_loss, diffusion_loss=diffusion_loss,
                prior_loss=prior_loss)
