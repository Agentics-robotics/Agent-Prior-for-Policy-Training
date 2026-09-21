"""Pick-place-wall designs, implemented only after the isolated proposal save.

The immutable proposal and sole padding feasibility amendment live under
round3/design_records/pick-place-wall. No simulator/controller source is used.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from relative_dp.model import DiffusionPolicy, masked_epsilon_loss
from .plugins import BaselinePlugin


RHO = float(np.sqrt(0.02**2 + 0.02**2))
POSITION_INDICES = tuple(
    i for start in (0, 4, 11, 18, 22, 29, 36, 39) for i in range(start, start + 3)
)


def base_scale(raw):
    """Fixed units only; preserve all45 raw fields, including absent zeros."""
    value = np.asarray(raw, np.float32).copy()
    if value.shape[-1] != 45:
        raise ValueError(f"Pickwall expects raw45, got {value.shape}")
    value[..., POSITION_INDICES] /= np.float32(0.30)
    value[..., 42:45] /= np.float32(0.10)
    for start in (7, 14, 25, 32):
        q = value[..., start:start + 4]
        ordered = q[..., [3, 0, 1, 2]]
        first = np.argmax(ordered != 0, axis=-1)
        sign_component = np.take_along_axis(ordered, first[..., None], axis=-1)[..., 0]
        q *= np.where(sign_component < 0, -1.0, 1.0)[..., None]
    return value


def relation_features(raw):
    """Per-row world45 + grasp15 + goal6 + wall14 =80 fixed features."""
    raw = np.asarray(raw, np.float32)
    scaled = base_scale(raw)
    h, o = raw[..., :3], raw[..., 4:7]
    hp, op = raw[..., 18:21], raw[..., 22:25]
    g, c, b = raw[..., 36:39], raw[..., 39:42], raw[..., 42:45]
    grasp = np.concatenate((
        (h - o) / 0.10, raw[..., 3:4], scaled[..., 7:11],
        (h - hp) / 0.01, (o - op) / 0.01,
        raw[..., 3:4] - raw[..., 21:22],
    ), axis=-1)
    goal = np.concatenate(((g - o) / 0.10, (g - h) / 0.10), axis=-1)
    faces = np.stack((
        o[..., 2] - (c[..., 2] + b[..., 2]) - RHO,
        o[..., 1] - (c[..., 1] + b[..., 1]) - RHO,
        c[..., 1] - b[..., 1] - o[..., 1] - RHO,
        o[..., 0] - (c[..., 0] + b[..., 0]) - RHO,
        c[..., 0] - b[..., 0] - o[..., 0] - RHO,
    ), axis=-1) / 0.10
    wall = np.concatenate(((h - c) / 0.10, (o - c) / 0.10, b / 0.10, faces), axis=-1)
    return np.concatenate((scaled, grasp, goal, wall), axis=-1).astype(np.float32)


def _branch(dimension):
    return nn.Sequential(nn.Linear(dimension, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())


class RelationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.world, self.grasp = _branch(45), _branch(15)
        self.goal, self.wall = _branch(6), _branch(14)
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.SiLU())

    def forward(self, history):
        per_row = torch.cat((
            self.world(history[..., :45]), self.grasp(history[..., 45:60]),
            self.goal(history[..., 60:66]), self.wall(history[..., 66:80]),
        ), dim=-1)
        return self.fusion(per_row.flatten(start_dim=1))


def persistent_events(obs):
    """Training-only geometry proxies; no use in runtime policy routing."""
    obs = np.asarray(obs, np.float32)
    total_actions = len(obs) - 1
    h, o = obs[:, :3], obs[:, 4:7]

    def persist(mask, start):
        for t in range(start, total_actions - 1):
            if bool(np.all(mask[t:t + 3])):
                return t
        return total_actions + 1

    lifted = (o[:, 2] >= 0.04) & (np.linalg.norm(h - o, axis=-1) <= 0.08) & (obs[:, 3] <= 0.65)
    e1 = persist(lifted, 0)
    e2 = persist(o[:, 2] - RHO >= obs[:, 41] + obs[:, 44] + 0.01, e1)
    e3 = persist(o[:, 1] - RHO >= obs[:, 40] + obs[:, 43] + 0.01, e2)
    return e1, e2, e3


def phase_labels(obs):
    return np.searchsorted(persistent_events(obs), np.arange(len(obs) - 1), side="right").astype(np.int64)


class PickwallPolicy(DiffusionPolicy):
    """One shared denoiser with a learned conditioner, including joint P3."""
    def __init__(self, config, candidate):
        condition_dim = 160 if candidate == "P2" else 128
        cfg = dict(config, action_dim=10 if candidate == "P3" else 4)
        # Base constructor multiplies this argument by the two outer history rows.
        super().__init__(condition_dim // cfg["n_obs_steps"], cfg)
        self.obs_dim = config["obs_dim"]
        self.candidate = candidate
        if candidate == "P2":
            self.encoder = nn.Sequential(nn.Linear(90, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
            self.router = nn.Linear(128, 4)
            self.phase_embedding = nn.Parameter(torch.randn(4, 32))
        else:
            self.encoder = RelationEncoder()

    def encode_condition(self, history):
        if self.candidate == "P2":
            state = self.encoder(history.flatten(start_dim=1))
            logits = self.router(state)
            probability = logits.softmax(dim=-1)
            return torch.cat((state, probability @ self.phase_embedding), dim=-1), logits
        return self.encoder(history), None

    def forward(self, noisy_actions, timesteps, obs_history):
        condition, logits = self.encode_condition(obs_history)
        epsilon = self.net(noisy_actions, timesteps, global_cond=condition)
        if self.candidate == "P2":
            return {"epsilon": epsilon, "phase_logits": logits}
        return epsilon

    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        # Condition/anchors are fixed for all16 positions and DDIM invocations.
        condition, _ = self.encode_condition(obs_history)
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config["inference_steps"], device=obs_history.device)
        sample = torch.randn(
            (len(obs_history), self.config["prediction_horizon"], self.config["action_dim"]),
            generator=generator, device=generator.device, dtype=obs_history.dtype,
        ).to(obs_history.device)
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self.net(sample, timestep, global_cond=condition)
            else:
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(sample, timestep, global_cond=condition)
            sample = scheduler.step(
                epsilon, timestep, sample, eta=0.0, use_clipped_model_output=False,
                generator=generator, return_dict=True,
            ).prev_sample
        # P3's extra channels are discarded by action_decode. No clean clipping.
        return sample


def sphere_clearance(position, center, half_size):
    """Signed box distance minus the sourced enclosing-sphere radius."""
    v = (position - center).abs() - half_size
    outside = torch.linalg.vector_norm(v.clamp_min(0), dim=-1)
    inside = v.amax(dim=-1).clamp_max(0)
    return outside + inside - RHO


class PickwallPlugin(BaselinePlugin):
    def __init__(self, task, schema, config):
        super().__init__(task, schema, config)
        self.candidate = config["candidate_id"]
        if task != "pick-place-wall" or self.candidate not in ("P1", "P2", "P3"):
            raise ValueError("PickwallPlugin requires pick-place-wall and P1/P2/P3")

    def observation(self, raw):
        return base_scale(raw) if self.candidate == "P2" else relation_features(raw)

    def passthrough(self, dimension):
        # The shared normalizer must not refit any of the declared fixed scales.
        return list(range(dimension))

    def model_config(self, cfg):
        return {"action_dim": 10 if self.candidate == "P3" else 4}

    def build_policy(self, cfg):
        return PickwallPolicy(cfg, self.candidate)

    def supervision(self, episodes, window_index):
        if self.candidate == "P1":
            return {}
        if self.candidate == "P2":
            labels = [phase_labels(episode["obs"]) for episode in episodes]
            return {"phase": np.asarray([labels[e][t] for e, t in window_index], np.int64)}
        future_delta, future_position = [], []
        for episode_index, t in window_index:
            episode = episodes[episode_index]
            obs = np.asarray(episode["obs"], np.float32)
            valid = min(16, len(episode["actions"]) - t)
            # Sole feasibility amendment: all invalid sequence target channels zero.
            delta = np.zeros((16, 6), np.float32)
            position = np.zeros((16, 6), np.float32)
            position[:valid, :3] = obs[t + 1:t + valid + 1, :3]
            position[:valid, 3:] = obs[t + 1:t + valid + 1, 4:7]
            delta[:valid, :3] = (position[:valid, :3] - obs[t, :3]) / 0.10
            delta[:valid, 3:] = (position[:valid, 3:] - obs[t, 4:7]) / 0.10
            future_delta.append(delta)
            future_position.append(position)
        return {
            "future_delta": np.stack(future_delta),
            "future_position": np.stack(future_position),
        }

    def training_actions(self, encoded_actions, extra_targets):
        if self.candidate != "P3":
            return np.asarray(encoded_actions, np.float32)
        return np.concatenate((encoded_actions, extra_targets["future_delta"]), axis=-1).astype(np.float32)

    def action_decode(self, actions, current_raw):
        # The shared runtime applies [-1,1] world/native clipping after this call.
        actions = np.asarray(actions, np.float32)
        expected = 10 if self.candidate == "P3" else 4
        if actions.shape[-1] != expected:
            raise ValueError(f"Expected {expected} sampled channels, got {actions.shape}")
        return actions[..., :4].copy()

    def extra_loss(self, output, batch, step):
        if self.candidate == "P1":
            return None, {}
        if self.candidate == "P2":
            ce = F.cross_entropy(output["phase_logits"], batch["targets"]["phase"])
            return 0.2 * ce, {"phase_cross_entropy": ce}
        epsilon = output["epsilon"] if isinstance(output, dict) else output
        mask, noise = batch["mask"], batch["noise"]
        future = 0.5 * masked_epsilon_loss(epsilon[..., 4:7], noise[..., 4:7], mask)
        future = future + 0.5 * masked_epsilon_loss(epsilon[..., 7:10], noise[..., 7:10], mask)
        noisy, raw = batch["noisy"], batch["raw_current"]
        alpha = batch["policy"].train_scheduler.alphas_cumprod.to(noisy.device)[batch["timesteps"]]
        alpha3 = alpha[:, None, None]
        clean = (noisy - (1 - alpha3).sqrt() * epsilon) / alpha3.sqrt()
        h_pred = raw[:, None, :3] + 0.10 * clean[..., 4:7]
        o_pred = raw[:, None, 4:7] + 0.10 * clean[..., 7:10]
        truth = batch["targets"]["future_position"]
        h_true, o_true = truth[..., :3], truth[..., 3:]
        center, half_size = raw[:, None, 39:42], raw[:, None, 42:45]
        clearance_error = (sphere_clearance(o_pred, center, half_size) - sphere_clearance(o_true, center, half_size)) / 0.05
        separation_error = (torch.linalg.vector_norm(h_pred - o_pred, dim=-1) - torch.linalg.vector_norm(h_true - o_true, dim=-1)) / 0.05
        geometric_mask = mask * (alpha[:, None] >= 0.5).to(mask.dtype)
        entry_loss = 0.5 * F.smooth_l1_loss(clearance_error, torch.zeros_like(clearance_error), reduction="none")
        entry_loss = entry_loss + 0.5 * F.smooth_l1_loss(separation_error, torch.zeros_like(separation_error), reduction="none")
        geometry = (entry_loss * geometric_mask).sum() / geometric_mask.sum().clamp_min(1)
        weight = 0.05 * min(1.0, (step + 1) / 1000.0)
        return 0.25 * future + weight * geometry, {
            "future_epsilon_MSE": future, "geometry_consistency": geometry,
            "geometry_weight": weight,
        }

    @torch.no_grad()
    def diagnostics(self, raw_history, policy):
        result = {"route": "soft_event" if self.candidate == "P2" else "flat", "execution_horizon": 4}
        if self.candidate == "P2":
            device = next(policy.parameters()).device
            history = torch.as_tensor(self.observation(raw_history), device=device)[None]
            _, logits = policy.encode_condition(history)
            result["phase_probabilities"] = logits.softmax(dim=-1)[0].cpu().tolist()
        if self.candidate == "P3":
            result["joint_sample_channels"] = 10
            result["executed_action_channels"] = 4
        return result
