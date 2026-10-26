"""Action terms: a twist and a kick trigger in, the frozen walker's (or kick's) joint targets out.

`WalkerCommandAction` sits on one robot. The match policy hands it four numbers per decision
(10 Hz): a twist and a kick trigger. It holds that twist and runs `velstand.onnx` at 50 Hz
underneath — every fourth physics substep — writing joint targets exactly as `JointPositionAction`
would have for velstand itself (default pose + 1.0 x action). That is the robot's own split:
`robot.move` at 20-50 Hz, robotd's 50 Hz loop below it.

**Kicks are robotd's one-shots.** A positive trigger with the ball in front of a foot swaps in the
shipped `ball_kick_left/right.onnx` for its manifest window (0.5 s), fed the all-zero command they
were trained on, then hands back to the walker — what `robot.do kick_left` does on a robot. A
trigger with the ball anywhere else does nothing, as does one mid-kick: the policy cannot kick the
air, so it has to learn to bring the ball to its feet first. Pushing the ball by walking into it
moves it at ~0.12 m/s; a kick is what makes a shot possible (the first runs ended in 84% draws).

With `source="opponent"` the term takes no slice of the policy's action. Each step it reads the
`opponent` observation group — red's view, laid out exactly like the learner's — and asks the
current self-play snapshot for red's twist and trigger. With no snapshot yet, red stands still:
the first opponent is a statue, which is the easiest thing there is to score against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from mjlab_microduck.soccer.frozen_walker import FrozenMlp, action_to_twist, walker_path
from mjlab_microduck.soccer.mdp import KICK_LEFT, KICK_NONE, KICK_RIGHT, ball_in_frame, kick_side

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Head and body slots of the 61D obs carry velstand's idle command: all zero (manifest `idle`).
_HEAD_BODY_DIM = 10
ACTION_DIM = 4  # vx, vy, yaw rate, kick trigger

_policies: dict[tuple[str, str], FrozenMlp] = {}


def _shared(path: str, device: str) -> FrozenMlp:
    """One copy of each frozen policy per device, shared by both ducks' terms."""
    key = (path, str(device))
    if key not in _policies:
        _policies[key] = FrozenMlp.from_onnx(path, device=device)
    return _policies[key]


@dataclass(kw_only=True)
class WalkerCommandActionCfg(JointPositionActionCfg):
    actuator_names: tuple[str, ...] = (".*",)
    scale: float = 1.0  # velstand's action scale (the kicks share it)
    use_default_offset: bool = True
    source: str = "policy"  # "policy": a slice of the action; "opponent": the self-play snapshot
    opponent_obs_group: str = "opponent"
    walker_file: str | None = None  # default: velstand.onnx from the pinned Hub set
    walker_period_s: float = 0.02  # 50 Hz, the rate velstand was trained at
    kick_files: tuple[str | None, str | None] = (None, None)  # default: ball_kick_left/right.onnx
    kick_duration_s: float = 0.5  # the set manifest's `duration_s` for both kicks

    def build(self, env: ManagerBasedRlEnv) -> WalkerCommandAction:
        return WalkerCommandAction(self, env)


class WalkerCommandAction(JointPositionAction):
    cfg: WalkerCommandActionCfg

    def __init__(self, cfg: WalkerCommandActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        n = self.num_envs
        self.walker = _shared(str(cfg.walker_file or walker_path()), self.device)
        left, right = cfg.kick_files
        self.kicks = {
            KICK_LEFT: _shared(str(left or walker_path("ball_kick_left.onnx")), self.device),
            KICK_RIGHT: _shared(str(right or walker_path("ball_kick_right.onnx")), self.device),
        }
        self._twist = torch.zeros(n, 3, device=self.device)
        self._hl_action = torch.zeros(n, ACTION_DIM if cfg.source == "policy" else 0, device=self.device)
        self._walker_action = torch.zeros(n, self._num_targets, device=self.device)
        self._head_body = torch.zeros(n, _HEAD_BODY_DIM, device=self.device)
        # The walker observes joints in joint order without passive joints, as velstand's
        # `joint_pos`/`joint_vel` terms did; its actions come out in actuator order (`_target_ids`).
        self._obs_joint_ids, _ = self._entity.find_joints(r"^(?!passive_).*")
        self._period = max(1, round(cfg.walker_period_s / env.physics_dt))
        self._substep = 0
        # Kick state, counted in walker ticks: which foot, and how many ticks of it are left.
        self._kick_ticks = max(1, round(cfg.kick_duration_s / cfg.walker_period_s))
        self._kick_left = torch.zeros(n, dtype=torch.long, device=self.device)
        self._kick_foot = torch.full((n,), KICK_NONE, dtype=torch.long, device=self.device)
        self.kicks_fired = torch.zeros((), device=self.device)  # since the runner last read it
        # Set by the self-play runner: (N, obs) -> (N, 4) in [-1, 1], or None for a standing opponent.
        self.opponent_policy: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def action_dim(self) -> int:
        return ACTION_DIM if self.cfg.source == "policy" else 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._hl_action

    @property
    def twist(self) -> torch.Tensor:
        return self._twist

    @property
    def kicking(self) -> torch.Tensor:
        return self._kick_left > 0

    @property
    def kick_fraction(self) -> torch.Tensor:
        """Fraction of a running kick still to go, 0 when walking: part of the observation."""
        return self._kick_left.float() / self._kick_ticks

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.source == "policy":
            self._hl_action[:] = actions
            decision = actions
        elif self.opponent_policy is not None:
            with torch.no_grad():
                decision = self.opponent_policy(self._env.obs_buf[self.cfg.opponent_obs_group])
        else:
            decision = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._twist[:] = action_to_twist(decision[:, :3])

        foot = kick_side(ball_in_frame(self._env, self.cfg.entity_name))
        fire = (decision[:, 3] > 0) & ~self.kicking & (foot != KICK_NONE)
        # In place, never rebound: these buffers were made outside the rollout's inference_mode,
        # and a tensor rebound inside it could not be reset in place outside it afterwards.
        self._kick_left[:] = torch.where(fire, torch.full_like(self._kick_left, self._kick_ticks), self._kick_left)
        self._kick_foot[:] = torch.where(fire, foot, self._kick_foot)
        self.kicks_fired += fire.sum()
        # A new decision starts a walker tick, so the twist lands without a 20 ms wait.
        self._substep = 0

    def _policy_obs(self) -> torch.Tensor:
        d = self._entity.data
        jid = self._obs_joint_ids
        # The kicks were trained on an all-zero command; a kicking duck's twist is not theirs.
        twist = torch.where(self.kicking.unsqueeze(-1), torch.zeros_like(self._twist), self._twist)
        return torch.cat(
            (
                d.root_link_ang_vel_b,  # == the imu gyro: the imu site is unrotated in trunk_base
                d.projected_gravity_b,
                d.joint_pos[:, jid] - d.default_joint_pos[:, jid],
                d.joint_vel[:, jid] - d.default_joint_vel[:, jid],
                self._walker_action,  # the last action, whichever policy produced it
                twist,
                self._head_body,
            ),
            dim=-1,
        )

    def apply_actions(self) -> None:
        if self._substep % self._period == 0:
            obs = self._policy_obs()
            action = self.walker(obs)
            if self.kicking.any():
                for foot, net in self.kicks.items():
                    use = (self.kicking & (self._kick_foot == foot)).unsqueeze(-1)
                    action = torch.where(use, net(obs), action)
                self._kick_left.sub_(1).clamp_(min=0)
            self._walker_action[:] = action
            self._processed_actions = self._walker_action * self._scale + self._offset
        self._substep += 1
        super().apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._hl_action[env_ids] = 0.0
        self._twist[env_ids] = 0.0
        self._walker_action[env_ids] = 0.0
        self._kick_left[env_ids] = 0
        self._kick_foot[env_ids] = KICK_NONE
