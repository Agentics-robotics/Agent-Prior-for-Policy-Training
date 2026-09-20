import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


class GripBlock(nn.Module):
    def __init__(self, width, cond_dim, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2*width)

    def forward(self, x, c):
        scale, shift = self.film(c).chunk(2, dim=-1)
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.norm2(y) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x + self.conv2(F.silu(y))


class PhaseHybridPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('act_min', torch.tensor(n['action_min'], dtype=torch.float32))
        self.register_buffer('act_scale', torch.tensor(n['action_scale'], dtype=torch.float32))
        self.horizon = spec['training']['horizon']
        # Constant drawer fields are not part of the recurrent input.
        self.obs_embed = nn.Sequential(nn.Linear(45, 128), nn.SiLU(), nn.Linear(128, 128))
        self.history_cell = nn.GRUCell(128, 128)
        self.context = nn.Sequential(nn.Linear(128+90, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU())
        self.slot_embedding = nn.Parameter(torch.randn(self.horizon, 32) * 0.02)
        self.phase_head = nn.Sequential(nn.Linear(224, 128), nn.SiLU(), nn.Linear(128, 15))
        self.phase_embedding = nn.Parameter(torch.randn(15, 8) * 0.05)
        cond_dim = 192 + self.horizon * 8
        # The public U-Net is eight-wide. Only its seven arm outputs are used.
        self.arm = DiffusionBackbone(cond_dim, spec['training'])
        self.register_buffer('frequencies', torch.exp(torch.arange(64, dtype=torch.float32) * (-math.log(10000.0)/63)))
        self.time_mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))
        self.grip_input = nn.Conv1d(8+15+1, 128, 3, padding=1)
        self.grip_blocks = nn.ModuleList([GripBlock(128, cond_dim+128, d) for d in (1, 2, 4, 1)])
        self.grip_norm = nn.GroupNorm(8, 128)
        self.grip_output = nn.Conv1d(128, 2, 1)
        # Predict normalized seven-joint displacement and two next finger positions.
        self.response = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 9))
        ids = torch.arange(15)
        old, new = ids[:, None], ids[None, :]
        self.register_buffer('transition_cost', ((new < old) | (new > old+1)).float())

    def normalize_obs(self, x):
        return (x-self.obs_mean)/self.obs_scale

    def native_action(self, x):
        return (x+1)*self.act_scale/2+self.act_min

    def encode_history(self, raw):
        x = self.normalize_obs(raw)
        x = torch.cat((x[..., :39], x[..., 41:]), dim=-1)
        h = x.new_zeros((x.shape[0], 128))
        for j in range(2):
            h = self.history_cell(self.obs_embed(x[:, j]), h)
        c = self.context(torch.cat((h, x[:, -1], x[:, -1]-x[:, -2]), dim=-1))
        slot = self.slot_embedding.unsqueeze(0).expand(c.shape[0], -1, -1)
        phase_logits = self.phase_head(torch.cat((c.unsqueeze(1).expand(-1, self.horizon, -1), slot), dim=-1))
        phase = phase_logits.softmax(dim=-1)
        local = phase @ self.phase_embedding
        condition = torch.cat((c, local.flatten(1)), dim=-1)
        return condition, phase_logits, phase

    def predict(self, noisy_action, timestep, raw_history):
        cond, phase_logits, phase = self.encode_history(raw_history)
        b, h, _ = noisy_action.shape
        ts = torch.as_tensor(timestep, device=noisy_action.device).reshape(-1).expand(b).to(noisy_action.dtype)
        angles = ts[:, None]*self.frequencies[None, :]
        time = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))
        slot_pos = torch.linspace(-1, 1, h, device=noisy_action.device, dtype=noisy_action.dtype).view(1, h, 1).expand(b, -1, -1)
        grip_h = self.grip_input(torch.cat((noisy_action, phase, slot_pos), dim=-1).transpose(1, 2))
        grip_cond = torch.cat((cond, time), dim=-1)
        for block in self.grip_blocks:
            grip_h = block(grip_h, grip_cond)
        grip_out = self.grip_output(F.silu(self.grip_norm(grip_h))).transpose(1, 2)
        arm_eps = self.arm(noisy_action, timestep, cond)[..., :7]
        eps = torch.cat((arm_eps, grip_out[..., :1]), dim=-1)
        return eps, phase_logits, phase, grip_out[..., 1]

    def forward(self, noisy_action, timestep, raw_history):
        return self.predict(noisy_action, timestep, raw_history)[0]

    def response_prediction(self, before, encoded_command):
        native = self.native_action(encoded_command)
        finger_target = (0.025*native[..., 7:8]+0.015).expand(-1, -1, 2)
        target = torch.cat((native[..., :7], finger_target), dim=-1)
        tracking_error = (target-before[..., :9])/self.obs_scale[:9]
        inp = torch.cat((self.normalize_obs(before), encoded_command, tracking_error), dim=-1)
        return self.response(inp)


def build_model(spec):
    return PhaseHybridPolicy(spec)


def masked_mean(value, mask):
    # value is already reduced over feature dimensions.
    return (value*mask).sum()/mask.sum().clamp_min(1.0)


def phase_targets(before, after, native_action):
    # Training-only weak event labels. No source index, clock, or future input
    # enters the deployed encoder. Distances below are semantic soft-label
    # construction choices, not success predicates or runtime constraints.
    tcp = before[..., 18:21]
    red = before[..., 25:28]
    blue = before[..., 32:35]
    rg = before[..., 41:44]
    bg = before[..., 44:47]
    tcp_vel = (after[..., 18:21]-tcp)/0.05
    red_goal_xy = torch.linalg.vector_norm(red[..., :2]-rg[..., :2], dim=-1) < 0.035
    red_low = (red[..., 2]-rg[..., 2]).abs() < 0.014
    red_done = red_goal_xy & red_low
    opened = native_action[..., 7] > 0
    red_gap_xy = torch.linalg.vector_norm(tcp[..., :2]-red[..., :2], dim=-1)
    towards_blue = (tcp_vel[..., :2]*(blue[..., :2]-tcp[..., :2])).sum(-1) > 0.002
    departing_red = opened & (tcp[..., 2] > 0.24) & towards_blue
    use_blue = red_done & ((red_gap_xy > 0.085) | departing_red)
    obj = torch.where(use_blue.unsqueeze(-1), blue, red)
    goal = torch.where(use_blue.unsqueeze(-1), bg, rg)
    future_obj = torch.where(use_blue.unsqueeze(-1), after[..., 32:35], after[..., 25:28])
    obj_vel = (future_obj-obj)/0.05
    goal_xy = torch.linalg.vector_norm(obj[..., :2]-goal[..., :2], dim=-1) < 0.04
    placed = goal_xy & ((obj[..., 2]-goal[..., 2]).abs() < 0.014)
    source_height = torch.where(use_blue, torch.full_like(obj[..., 2], 0.02), torch.full_like(obj[..., 2], 0.06))
    lifted = (obj[..., 2] > source_height+0.018) | (obj_vel[..., 2] > 0.01)
    moving_xy = torch.linalg.vector_norm(obj_vel[..., :2], dim=-1) > 0.025
    # Source y is close to zero in these demonstrations. This is a weak label,
    # not an assumed invariant workspace transform.
    in_transport = (moving_xy & (obj[..., 2] > source_height+0.025)) | (obj[..., 1].abs() > 0.045)
    descending = goal_xy & ((obj_vel[..., 2] < -0.008) | (obj[..., 2] < goal[..., 2]+0.23))
    close = torch.ones_like(use_blue, dtype=torch.long)
    held_phase = torch.where(lifted, close*2, close)
    held_phase = torch.where(in_transport, close*3, held_phase)
    held_phase = torch.where(descending, close*4, held_phase)
    clearance = tcp[..., 2]-obj[..., 2]
    open_phase = torch.where(placed, torch.where(clearance < 0.03, close*5, close*6), close*0)
    local = torch.where(opened, open_phase, held_phase)
    stage = local + use_blue.long()*7
    terminal = use_blue & placed & opened & (tcp[..., 2] > 0.28) & (torch.linalg.vector_norm(tcp_vel, dim=-1) < 0.025)
    stage = torch.where(terminal, close*14, stage)
    return F.one_hot(stage, 15).to(before.dtype)*0.96 + 0.04/15.0


def response_error(pred, target):
    error = F.smooth_l1_loss(pred, target, reduction='none')
    # Full-data units are unchanged. Joint displacements are smaller than
    # absolute normalized finger positions, so weight that response task.
    return 16.0*error[..., :7].mean(-1) + error[..., 7:9].mean(-1)


def compute_loss(model, batch, spec):
    eps, phase_logits, phase, grip_logits = model.predict(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diff = epsilon_loss(eps, batch['noise'], batch['mask'])
    mask = batch['mask'].squeeze(-1)
    fm = batch['future_mask'].squeeze(-1)
    valid = mask*fm
    before = torch.cat((batch['raw_obs'][:, :1], batch['future_obs'][:, :-1]), dim=1)
    pre_valid = torch.cat((torch.ones_like(fm[:, :1]), fm[:, :-1]), dim=1)
    valid = valid*pre_valid
    future = batch['future_obs']
    with torch.no_grad():
        labels = phase_targets(before, future, batch['native_action'])
    phase_ce = masked_mean(-(labels*phase_logits.log_softmax(-1)).sum(-1), valid)
    pairs = valid[:, :-1]*valid[:, 1:]
    order_cost = ((phase[:, :-1] @ model.transition_cost)*phase[:, 1:]).sum(-1)
    phase_order = masked_mean(order_cost, pairs)

    g = (batch['native_action'][..., 7] > 0).to(eps.dtype)
    grip_ce = masked_mean(F.binary_cross_entropy_with_logits(grip_logits, g, reduction='none'), mask)
    a = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    clean = (batch['noisy_action']-(1-a).sqrt()*eps)/a.sqrt()
    confidence = a[:, 0, 0].unsqueeze(1)
    # Soft bounding is deliberately not applied to the actual DDPM output.
    # Auxiliary clipping merely prevents extreme early response inputs.
    bounded = clean.clamp(-2, 2)
    grip_expected = 2*grip_logits.sigmoid()-1
    grip_consistency = masked_mean(confidence*F.smooth_l1_loss(clean[..., 7], grip_expected, reduction='none'), mask)
    grip_support = masked_mean(confidence*(bounded[..., 7].abs()-1).square(), mask)
    apairs = mask[:, :-1]*mask[:, 1:]
    pred_delta = bounded[:, 1:]-bounded[:, :-1]
    expert_delta = batch['encoded_action'][:, 1:]-batch['encoded_action'][:, :-1]
    delta_error = F.smooth_l1_loss(pred_delta, expert_delta, reduction='none')
    grip_events = masked_mean(confidence*delta_error[..., 7], apairs)
    arm_continuity = masked_mean(confidence*delta_error[..., :7].mean(-1), apairs)
    p = grip_logits.sigmoid()
    stable = (g[:, 2:] == g[:, 1:-1]) & (g[:, 1:-1] == g[:, :-2])
    triples = mask[:, 2:]*mask[:, 1:-1]*mask[:, :-2]*stable.to(mask.dtype)
    persistence = masked_mean((p[:, 2:]-2*p[:, 1:-1]+p[:, :-2]).square(), triples)

    response_target = torch.cat(((future[..., :7]-before[..., :7])/model.obs_scale[:7],
                                  (future[..., 7:9]-model.obs_mean[7:9])/model.obs_scale[7:9]), dim=-1)
    expert_response = model.response_prediction(before, batch['encoded_action'])
    clean_response = model.response_prediction(before, bounded)
    response_fit = masked_mean(response_error(expert_response, response_target), valid)
    response_coupling = masked_mean(confidence*response_error(clean_response, response_target), valid)

    prior = (0.08*phase_ce + 0.01*phase_order + 0.08*grip_ce
             + 0.025*grip_consistency + 0.01*grip_support
             + 0.02*grip_events + 0.01*arm_continuity + 0.01*persistence
             + 0.10*response_fit + 0.05*response_coupling)
    return {'loss': diff+prior, 'diffusion_loss': diff, 'prior_loss': prior,
            'phase_ce': phase_ce, 'phase_order': phase_order, 'gripper_ce': grip_ce,
            'gripper_consistency': grip_consistency, 'gripper_support': grip_support,
            'gripper_event_delta': grip_events, 'arm_delta': arm_continuity,
            'gripper_persistence': persistence, 'response_fit': response_fit,
            'response_coupling': response_coupling}
