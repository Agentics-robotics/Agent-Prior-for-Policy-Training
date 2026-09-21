import torch
from appl.public import DiffusionBackbone
from recurrent import WindowGRU
nn = torch.nn
F = torch.nn.functional


def relative_rotation(tcp_q, obj_q):
    a = tcp_q/tcp_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = obj_q/obj_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    aw, av = a[..., :1], -a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = (aw*bw-(av*bv).sum(-1, keepdim=True)).squeeze(-1)
    x, y, z = (aw*bv+bw*av+torch.cross(av, bv, dim=-1)).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)


def geometry(raw):
    obj = torch.stack((raw[..., 25:28], raw[..., 32:35]), -2)
    rel = obj-raw[..., 18:21].unsqueeze(-2)
    rot = torch.stack((relative_rotation(raw[..., 21:25], raw[..., 28:32]),
                       relative_rotation(raw[..., 21:25], raw[..., 35:39])), -2)
    return obj, rel, rot


class ContactDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.mean_values = list(spec['normalizer']['mean'])
        self.scale_values = list(spec['normalizer']['std'])
        self.dt = 0.05
        self.frame_encoder = nn.Sequential(nn.Linear(91, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.initial_state = nn.Sequential(nn.Linear(128, 128), nn.Tanh())
        self.contact_gru = WindowGRU(128, 128)
        self.emission = nn.Linear(128, 6)
        self.support_head = nn.Linear(128, 2)
        self.transition = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 36))
        self.transition_base = nn.Parameter(2.0*torch.eye(6))
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)
        self.context = nn.Sequential(nn.Linear(222, 256), nn.SiLU(), nn.Linear(256, 192), nn.SiLU())
        self.backbone = DiffusionBackbone(200, spec['training'])
        self.backbone.net.final_conv[-1] = nn.Conv1d(spec['training']['down_dims'][0], 64, 1)
        self.arm_output = nn.Linear(64, 7)
        self.gripper_output = nn.Linear(64, 1)
        self.horizon = spec['training']['horizon']
        self.forecast_position = nn.Parameter(torch.randn(1, self.horizon, 16)*0.02)
        self.forecast_input = nn.Sequential(nn.Linear(24, 128), nn.SiLU())
        self.forecast_init = nn.Sequential(nn.Linear(200, 128), nn.Tanh())
        self.forecast_gru = WindowGRU(128, 128)
        self.forecast_output = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 16))

    def scales(self, raw):
        mean, scale = raw.new_tensor(self.mean_values), raw.new_tensor(self.scale_values)
        relative = torch.stack((scale[18:21]+scale[25:28], scale[18:21]+scale[32:35]))
        motion = torch.cat((scale[18:21].unsqueeze(0), relative), 0)
        return mean, scale, relative, motion

    def features(self, raw):
        mean, scale, rs, ms = self.scales(raw)
        norm = (raw-mean)/scale
        obj, rel, rot = geometry(raw)
        pos = torch.cat((raw[..., 18:21].unsqueeze(-2), obj), -2)
        pv = torch.cat((torch.zeros_like(pos[:, :1]), (pos[:, 1:]-pos[:, :-1])/self.dt), 1)
        rv = torch.cat((torch.zeros_like(rel[:, :1]), (rel[:, 1:]-rel[:, :-1])/self.dt), 1)
        fingers = raw[..., 7:9]
        fv = torch.cat((torch.zeros_like(fingers[:, :1]), (fingers[:, 1:]-fingers[:, :-1])/self.dt), 1)
        flag = torch.cat((torch.zeros_like(fingers[:, :1, :1]), torch.ones_like(fingers[:, 1:, :1])), 1)
        return torch.cat((norm, fingers, (rel/rs).flatten(-2), rot.flatten(-2), (pv/ms).flatten(-2),
                           (rv/rs).flatten(-2), fv/scale[7:9], flag), -1), norm

    def encode(self, raw):
        frame, norm = self.features(raw)
        embedded = self.frame_encoder(frame)
        hidden, _ = self.contact_gru(embedded, self.initial_state(embedded[:, 0]).unsqueeze(0))
        emissions = self.emission(hidden)
        first = emissions[:, 0].softmax(-1)
        logits = self.transition(torch.cat((hidden[:, 0], hidden[:, 1]), -1)).reshape(-1, 6, 6)+self.transition_base
        transitions = logits.softmax(-1)
        predicted = (first.unsqueeze(-1)*transitions).sum(1)
        last_logits = emissions[:, 1]+0.35*predicted.clamp_min(1e-6).log()
        support_logits = self.support_head(hidden)
        context = self.context(torch.cat((hidden[:, -1], norm.flatten(1)), -1))
        condition = torch.cat((context, last_logits.softmax(-1), support_logits[:, -1].sigmoid()), -1)
        return condition, {'emissions': emissions, 'first': first, 'last_logits': last_logits,
                           'predicted_belief': predicted, 'transitions': transitions,
                           'transition_logits': logits, 'support_logits': support_logits}

    def denoise(self, noisy_action, timestep, condition):
        features = self.backbone(noisy_action, timestep, condition)
        return torch.cat((self.arm_output(features), self.gripper_output(features)), -1)

    def forward(self, noisy_action, timestep, raw_history):
        condition, _ = self.encode(raw_history)
        return self.denoise(noisy_action, timestep, condition)

    def forecast(self, condition, action):
        pos = self.forecast_position[:, :action.shape[1]].expand(action.shape[0], -1, -1)
        x = self.forecast_input(torch.cat((action, pos), -1))
        hidden, _ = self.forecast_gru(x, self.forecast_init(condition).unsqueeze(0))
        return self.forecast_output(hidden)


def build_model(spec):
    return ContactDiffusion(spec)


def weighted_mean(value, weight):
    return (value*weight).sum()/weight.sum().clamp_min(1.0)


@torch.no_grad()
def training_targets(model, batch):
    raw, after, actions = batch['raw_obs'], batch['future_obs'], batch['native_action']
    mean, scale, rs, _ = model.scales(raw)
    valid = batch['mask'].squeeze(-1)*batch['future_mask'].squeeze(-1)
    pre = torch.cat((raw[:, :1], after[:, :-1]), 1)
    valid = valid*torch.cat((torch.ones_like(valid[:, :1]), batch['future_mask'][:, :-1, 0]), 1)
    obj, rel, rot = geometry(pre)
    nxt, nrel, nrot = geometry(after)
    near = rel.norm(dim=-1) < 0.055
    aperture = pre[..., 7:9].mean(-1)
    held = (aperture > 0.012) & (aperture < 0.028)
    close = actions[..., 7] < 0
    stable = ((nrel-rel).norm(dim=-1) < 0.006) & ((nrot-rot).norm(dim=-1) < 0.10)
    confirmed = (obj[..., 2] > 0.035) & stable
    for j in range(after.shape[1]):
        end = min(j+8, after.shape[1])
        lift = (nxt[:, j:end, :, 2] > 0.035) & ((nxt[:, j:end, :, 2]-obj[:, j:j+1, :, 2]) > 0.008)
        consistent = ((nrel[:, j:end]-rel[:, j:j+1]).norm(dim=-1) < 0.006)
        consistent = consistent & ((nrot[:, j:end]-rot[:, j:j+1]).norm(dim=-1) < 0.10)
        fap = after[:, j:end, 7:9].mean(-1)
        evidence = lift & consistent & ((fap > 0.012) & (fap < 0.028)).unsqueeze(-1) & (valid[:, j:end] > 0).unsqueeze(-1)
        confirmed[:, j] = confirmed[:, j] | evidence.any(1)
    attached = confirmed & near & held.unsqueeze(-1) & close.unsqueeze(-1)
    supported = (obj[..., 2] < 0.032) & ((nxt-obj).norm(dim=-1) < 0.010)
    buffered = (obj[..., 0, :2]-pre[..., 44:46]).norm(dim=-1) > 0.10
    releasing = supported[..., 0] & buffered & (rel[..., 0, :].norm(dim=-1) < 0.070)
    labels = torch.zeros_like(actions[..., 7], dtype=torch.long)
    labels = torch.where(near[..., 0] & close, torch.ones_like(labels), labels)
    labels = torch.where(attached[..., 0], torch.full_like(labels, 2), labels)
    labels = torch.where(releasing, torch.full_like(labels, 3), labels)
    labels = torch.where(near[..., 1] & close, torch.full_like(labels, 4), labels)
    labels = torch.where(attached[..., 1], torch.full_like(labels, 5), labels)
    now, nowrel, _ = geometry(raw[:, -1])
    regression = torch.cat((((nrel-nowrel.unsqueeze(1))/rs).flatten(-2),
                            ((nxt-now.unsqueeze(1))/rs).flatten(-2), (after[..., 7:9]-mean[7:9])/scale[7:9]), -1)
    nap = after[..., 7:9].mean(-1)
    attachment = (nxt[..., 2] > 0.035) & (nrel.norm(dim=-1) < 0.055) & stable
    attachment = attachment & ((nap > 0.012) & (nap < 0.028)).unsqueeze(-1)
    return labels, supported.float(), valid, regression, attachment.float()


def forecast_error(prediction, regression, attachment):
    motion = F.smooth_l1_loss(prediction[..., :14], regression, reduction='none').mean(-1)
    attach = F.binary_cross_entropy_with_logits(prediction[..., 14:], attachment, reduction='none').mean(-1)
    return motion+0.25*attach


def compute_loss(model, batch, spec):
    condition, details = model.encode(batch['raw_obs'])
    epsilon = model.denoise(batch['noisy_action'], batch['timesteps'], condition)
    labels, support, valid, regression, attachment = training_targets(model, batch)
    mask = batch['mask'].squeeze(-1)
    squared = (epsilon-batch['noise']).square()
    arm_loss = weighted_mean(squared[..., :7].mean(-1), mask)
    with torch.no_grad():
        grip = batch['native_action'][..., 7]
        switches = torch.zeros_like(grip)
        switches[:, 1:] = (grip[:, 1:]*grip[:, :-1] < 0).float()*mask[:, 1:]*mask[:, :-1]
        events = F.max_pool1d(switches.unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
        closing = ((labels == 1) | (labels == 4)).float()
        gripper_weight = mask*(1.0+3.0*events+1.5*closing+(labels == 3).float())
    grip_loss = weighted_mean(squared[..., 7], gripper_weight)
    diffusion_loss = (7.0*arm_loss+2.0*grip_loss)/9.0
    two_valid = valid[:, :2]
    cw = epsilon.new_tensor([1.0, 3.0, 1.3, 2.0, 3.0, 1.3])
    logp = torch.stack((details['emissions'][:, 0].log_softmax(-1), details['last_logits'].log_softmax(-1)), 1)
    belief_loss = weighted_mean(-logp.gather(-1, labels[:, :2, None]).squeeze(-1), two_valid*cw[labels[:, :2]])
    pair = two_valid[:, 0]*two_valid[:, 1]
    row = details['transition_logits'].gather(1, labels[:, :1, None].expand(-1, 1, 6)).squeeze(1)
    transition_loss = weighted_mean(F.cross_entropy(row, labels[:, 1], reduction='none'), pair*(1.0+3.0*(labels[:, 0] != labels[:, 1]).float()))
    p, q = details['predicted_belief'].clamp_min(1e-6), details['emissions'][:, 1].softmax(-1).clamp_min(1e-6)
    middle = 0.5*(p+q)
    js = 0.5*((p*(p.log()-middle.log())).sum(-1)+(q*(q.log()-middle.log())).sum(-1))
    consistency = weighted_mean(js, pair)
    bad = epsilon.new_tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1],
                              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
    persistence = weighted_mean((details['first'].unsqueeze(-1)*details['transitions']*bad).sum((-1, -2)), pair)
    support_loss = weighted_mean(F.binary_cross_entropy_with_logits(details['support_logits'], support[:, :2], reduction='none').mean(-1), two_valid)
    teacher = model.forecast(condition, batch['encoded_action'])
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    estimate = (batch['noisy_action']-(1.0-alpha).sqrt()*epsilon)/alpha.sqrt()
    generated = model.forecast(condition, estimate.clamp(-2.0, 2.0))
    short = torch.zeros_like(valid)
    short[:, :8] = 1.0
    fm = valid*short
    teacher_loss = weighted_mean(forecast_error(teacher, regression, attachment), fm)
    generated_loss = (forecast_error(generated, regression, attachment)*fm*alpha.reshape(-1, 1)).sum()/fm.sum().clamp_min(1.0)
    prior = 0.12*belief_loss+0.04*transition_loss+0.02*consistency+0.01*persistence+0.04*support_loss+0.08*teacher_loss+0.04*generated_loss
    return {'loss': diffusion_loss+prior, 'diffusion_loss': diffusion_loss, 'prior_loss': prior}
