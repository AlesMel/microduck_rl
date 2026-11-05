"""Shared pieces of the gait evaluation scripts: checkpoint → MLP policy, a headless play
env, the real robot's action pipeline, and the left-right mirror of a policy.

Ported 2026-10-28 from the VelStandFast session scratchpads (calib_vx / battery /
reward_acct / gait_asym), where each script carried its own copy of these.
"""
from __future__ import annotations

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

# 61D actor obs layout (symmetry.py): last_action, then the command block.
OBS_LAST_ACTION = slice(34, 48)
OBS_VX, OBS_VY, OBS_WZ = 48, 49, 50
OBS_HEAD_BODY = slice(51, 61)

DEFAULT_WALK_TASK = "Mjlab-WalkV2-Flat-Backlash-MicroDuck"


def load_policy(ckpt: str, device: str = "cpu"):
    """rsl_rl checkpoint → deterministic actor, normalizer applied as rsl_rl does
    ((x − mean) / (std + 1e-2)), 512-256-128 ELU MLP."""
    sd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
    mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"] + 1e-2
    layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    @torch.no_grad()
    def policy(obs: torch.Tensor) -> torch.Tensor:
        x = (obs - mean) / std
        for k, (w, b) in enumerate(layers):
            x = x @ w.T + b
            if k < len(layers) - 1:
                x = torch.nn.functional.elu(x)
        return x

    return policy


def zero_policy(obs: torch.Tensor) -> torch.Tensor:
    """Stand-still probe: hold the home pose (zero joint offsets)."""
    return torch.zeros(obs.shape[0], 14, device=obs.device)


def mirror_policy(policy):
    """π_m(o) = M_a(π(M_o(o))): the policy's left-right mirror image, from the 61D tables in
    symmetry.py. A symmetric reward must score π and π_m the same."""
    from mjlab_microduck.tasks.symmetry import _get_tensors

    @torch.no_grad()
    def mirrored(obs: torch.Tensor) -> torch.Tensor:
        obs_perm, obs_sign, act_perm, act_sign = _get_tensors(obs.device)
        a = policy(obs[:, obs_perm] * obs_sign)
        return a[:, act_perm] * act_sign

    return mirrored


def make_play_env(task: str, n: int, device: str = "cpu", *, keep_pushes: bool = False,
                  endless: bool = True, cfg_hook=None):
    """Headless play-cfg env: pushes off, episodes effectively endless (falls still terminate
    through the task's own fall termination). Returns (wrapped env, raw env)."""
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = n
    if not keep_pushes:
        for name in list(cfg.events):
            if "push" in name:
                cfg.events.pop(name)
    if endless:
        cfg.episode_length_s = 1e6
    if cfg_hook is not None:
        cfg_hook(cfg)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device))
    return env, env.unwrapped


class RobotChain:
    """The real robot's action pipeline (microduck robotd defaults), for A/B against training.

    robotd: target = HOME + action_scale · (7.4 / V_ema) · action  (voltage_adapt, V clamped
    6.0-9.5), then a first-order low-pass on the targets, y += α (x − y), legs α 0.7, head
    α 0.5 (control.rs:716-738, main.rs:3528-3532). The policy's last_action observation stays
    the RAW output (control.rs:665). Training applies target = HOME + 1.0 · action, so the env
    is fed the filtered offset y and the obs' last_action is overwritten with the raw action.

    V is each env's sampled BAM battery voltage (no load sag: the robot's V is an EMA)."""

    LEGS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
    HEAD = (5, 6, 7, 8)

    def __init__(self, raw: ManagerBasedRlEnv | None, n: int, device: str = "cpu", *,
                 action_scale: float = 0.9, nominal_v: float = 7.4, v_clamp=(6.0, 9.5),
                 legs_alpha: float = 0.7, head_alpha: float = 0.5, vin: torch.Tensor | None = None):
        if vin is None:
            vin = torch.full((n, 1), nominal_v, device=device)
            if raw is not None:
                for act in raw.scene["robot"].actuators:
                    v = getattr(act, "vin_tensor", None)
                    if v is not None:
                        vin = v.reshape(n, 1).to(device)
                        break
        self.vin = vin.reshape(n, 1)
        self.gain = action_scale * nominal_v / self.vin.clamp(*v_clamp)
        self.alpha = torch.ones(14, device=device)
        self.alpha[list(self.LEGS)] = legs_alpha
        self.alpha[list(self.HEAD)] = head_alpha
        self.y = torch.zeros(n, 14, device=device)
        self.last_raw = torch.zeros(n, 14, device=device)

    def __call__(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Raw policy output → the action to hand the training env."""
        x = self.gain * raw_action
        self.y = self.y + self.alpha * (x - self.y)
        self.last_raw = raw_action.clone()
        return self.y.clone()

    def fix_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Put the raw last action back into the observation, as the robot does."""
        obs = obs.clone()
        obs[:, OBS_LAST_ACTION] = self.last_raw
        return obs

    def reset(self, env_ids) -> None:
        self.y[env_ids] = 0.0
        self.last_raw[env_ids] = 0.0


def command_obs(obs: torch.Tensor, vx, vy=0.0, wz=0.0) -> torch.Tensor:
    """The observation with the twist slots overwritten and head/body slots zeroed (the
    scripts drive the policy through its observation, not the env's command manager)."""
    o = obs.clone()
    o[:, OBS_VX] = vx
    o[:, OBS_VY] = vy
    o[:, OBS_WZ] = wz
    o[:, OBS_HEAD_BODY] = 0.0
    return o


def step(env, policy, obs: torch.Tensor, vx, vy=0.0, wz=0.0, chain: RobotChain | None = None):
    """One control step through the optional robot chain. Returns the next actor obs."""
    o = command_obs(obs if chain is None else chain.fix_obs(obs), vx, vy, wz)
    a = policy(o)
    obs, _, dones, _ = env.step(a if chain is None else chain(a))
    if chain is not None and dones.any():
        chain.reset(dones.nonzero().flatten())
    return obs["actor"]
