"""Action terms: a twist in, the frozen walker's joint targets out.

`WalkerCommandAction` sits on one robot. The match policy hands it three numbers per decision
(10 Hz); it holds that twist and runs `velstand.onnx` at 50 Hz underneath — every fourth physics
substep — writing joint targets exactly as `JointPositionAction` would have for velstand itself
(default pose + 1.0 x action). That is the robot's own split: `robot.move` at 20-50 Hz, robotd's
50 Hz loop below it.

With `source="opponent"` the term takes no slice of the policy's action. Each step it reads the
`opponent` observation group — red's view, laid out exactly like the learner's — and asks the
current self-play snapshot for red's twist. With no snapshot yet, red stands still: the first
opponent is a statue, which is the easiest thing there is to score against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from mjlab_microduck.soccer.frozen_walker import TWIST_LIMITS, FrozenMlp, walker_path

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Head and body slots of the 61D obs carry velstand's idle command: all zero (manifest `idle`).
_HEAD_BODY_DIM = 10

_walkers: dict[tuple[str, str], FrozenMlp] = {}


def _shared_walker(path: str, device: str) -> FrozenMlp:
    """One copy of the walker per device, shared by both ducks' terms."""
    key = (path, str(device))
    if key not in _walkers:
        _walkers[key] = FrozenMlp.from_onnx(path, device=device)
    return _walkers[key]


@dataclass(kw_only=True)
class WalkerCommandActionCfg(JointPositionActionCfg):
    actuator_names: tuple[str, ...] = (".*",)
    scale: float = 1.0  # velstand's action scale
    use_default_offset: bool = True
    source: str = "policy"  # "policy": a 3D slice of the action; "opponent": the self-play snapshot
    opponent_obs_group: str = "opponent"
    walker_file: str | None = None  # default: velstand.onnx from the pinned Hub set
    walker_period_s: float = 0.02  # 50 Hz, the rate velstand was trained at

    def build(self, env: ManagerBasedRlEnv) -> WalkerCommandAction:
        return WalkerCommandAction(self, env)


class WalkerCommandAction(JointPositionAction):
    cfg: WalkerCommandActionCfg

    def __init__(self, cfg: WalkerCommandActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        n = self.num_envs
        self.walker = _shared_walker(str(cfg.walker_file or walker_path()), self.device)
        self._limits = torch.tensor(TWIST_LIMITS, device=self.device)
        self._twist = torch.zeros(n, 3, device=self.device)
        self._hl_action = torch.zeros(n, 3 if cfg.source == "policy" else 0, device=self.device)
        self._walker_action = torch.zeros(n, self._num_targets, device=self.device)
        self._head_body = torch.zeros(n, _HEAD_BODY_DIM, device=self.device)
        # The walker observes joints in joint order without passive joints, as velstand's
        # `joint_pos`/`joint_vel` terms did; its actions come out in actuator order (`_target_ids`).
        self._obs_joint_ids, _ = self._entity.find_joints(r"^(?!passive_).*")
        self._period = max(1, round(cfg.walker_period_s / env.physics_dt))
        self._substep = 0
        # Set by the self-play runner: (N, obs) -> (N, 3) in [-1, 1], or None for a standing opponent.
        self.opponent_policy: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def action_dim(self) -> int:
        return 3 if self.cfg.source == "policy" else 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._hl_action

    @property
    def twist(self) -> torch.Tensor:
        return self._twist

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.source == "policy":
            self._hl_action[:] = actions
            self._twist[:] = actions.clamp(-1.0, 1.0) * self._limits
        elif self.opponent_policy is not None:
            obs = self._env.obs_buf[self.cfg.opponent_obs_group]
            with torch.no_grad():
                self._twist[:] = self.opponent_policy(obs).clamp(-1.0, 1.0) * self._limits
        else:
            self._twist.zero_()
        # A new decision starts a walker tick, so the twist lands without a 20 ms wait.
        self._substep = 0

    def _walker_obs(self) -> torch.Tensor:
        d = self._entity.data
        jid = self._obs_joint_ids
        return torch.cat(
            (
                d.root_link_ang_vel_b,  # == the imu gyro: the imu site is unrotated in trunk_base
                d.projected_gravity_b,
                d.joint_pos[:, jid] - d.default_joint_pos[:, jid],
                d.joint_vel[:, jid] - d.default_joint_vel[:, jid],
                self._walker_action,
                self._twist,
                self._head_body,
            ),
            dim=-1,
        )

    def apply_actions(self) -> None:
        if self._substep % self._period == 0:
            self._walker_action[:] = self.walker(self._walker_obs())
            self._processed_actions = self._walker_action * self._scale + self._offset
        self._substep += 1
        super().apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._hl_action[env_ids] = 0.0
        self._twist[env_ids] = 0.0
        self._walker_action[env_ids] = 0.0
