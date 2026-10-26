"""Action terms: a twist and a kick trigger in, the frozen walker's (or kick's) joint targets out.

`WalkerCommandAction` sits on one robot. The match policy hands it four numbers per decision
(10 Hz): a twist and a kick trigger. It holds that twist and runs `velstand.onnx` at 50 Hz
underneath — every fourth physics substep — writing joint targets exactly as `JointPositionAction`
would have for velstand itself (default pose + 1.0 x action). That is the robot's own split:
`robot.move` at 20-50 Hz, robotd's 50 Hz loop below it.

**Kicks are robotd's one-shots, from a standstill.** A positive trigger with the ball within reach
(the REACH zone) hands the duck to the routine: it walks in slowly and stops with the ball at the
FIRE band's depth, settles for `settle_s`, and then, if it is standing still and the ball sits in
that foot's FIRE band — where a kick from rest was measured to connect — the shipped
`ball_kick_left/right.onnx` runs for its manifest window (0.5 s), fed the all-zero command it was
trained on, and hands back to the walker. Otherwise the routine cancels and the duck walks on. This
is how `robot.do kick_left` is used on a robot too: you stop the duck before you ask it to kick.
A trigger with the ball out of reach, or mid-routine, does nothing.

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
from mjlab_microduck.soccer.mdp import ROUTINE, KICK_LEFT, KICK_NONE, KICK_RIGHT, ball_in_frame, fire_side, kick_side, routine_steer

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
    kick_duration_s: float = ROUTINE["kick_s"]  # the set manifest's `duration_s` for both kicks
    # Stop-then-kick: a trigger zeroes the twist this long, then kicks only if the duck is still and
    # the ball sits in the FIRE band. Kicked mid-stride the BallKick policies (trained from rest)
    # barely touch the ball: 0-12% of walk-in shots scored against 83% after a 0.6 s settle.
    settle_s: float = ROUTINE["settle_s"]
    still_speed: float = ROUTINE["still_speed"]  # m/s: a trunk slower than this counts as standing
    # Before settling, the routine walks the last few centimetres itself and stops on time: velstand
    # stands still below ~0.2 m/s commanded (measured: 0.15 -> 0 m/s, 0.25 -> 0.08 m/s), so the duck
    # cannot creep or side-step into place, but from `approach_cmd` it stops a consistent ~1.5 cm
    # (sd 0.3 cm) after the twist goes to zero. Stopping when the ball is `stop_x` ahead therefore
    # leaves it at the FIRE band's centre. Stopping where the ball first came into reach left it
    # short of the 3 cm-deep band: 4 kicks in 276 tries.
    approach_cmd: float = ROUTINE["approach_cmd"]
    stop_x: float = ROUTINE["stop_x"]
    align_timeout_s: float = ROUTINE["timeout_s"]
    # Lateral placement by steering on the way in: velstand turns while walking (0.3 -> 0.27 rad/s
    # at 0.25 m/s) though not standing (< 0.8 rad/s does nothing), and it cannot side-step below
    # 0.4 m/s. A ball dead centre is a dud kick (0.2-0.6 m/s, 32-39 degrees off), so the duck turns
    # to bring it `foot_y` toward the chosen foot.

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
        self._settle_ticks = max(1, round(cfg.settle_s / cfg.walker_period_s))
        self._settle_left = torch.zeros(n, dtype=torch.long, device=self.device)
        self._align_ticks = max(1, round(cfg.align_timeout_s / cfg.walker_period_s))
        self._align_left = torch.zeros(n, dtype=torch.long, device=self.device)
        self._align_foot = torch.full((n,), KICK_NONE, dtype=torch.long, device=self.device)
        self._align_twist = torch.zeros(n, 3, device=self.device)

        self.kicks_fired = torch.zeros((), device=self.device)  # since the runner last read it
        self.kicks_cancelled = torch.zeros((), device=self.device)  # settled, but no kickable ball
        self.fired_now = torch.zeros(n, dtype=torch.bool, device=self.device)  # this decision's kicks
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
    def aligning(self) -> torch.Tensor:
        return self._align_left > 0

    @property
    def settling(self) -> torch.Tensor:
        return self._settle_left > 0

    @property
    def busy(self) -> torch.Tensor:
        """In the kick routine (placing the ball, settling or kicking): the twist is not the policy's."""
        return (self._align_left > 0) | (self._settle_left > 0) | (self._kick_left > 0)

    @property
    def kick_fraction(self) -> torch.Tensor:
        """Where the kick routine is, for the observation: 0 walking, 1 placing the ball, then
        falling from 1 to 0 through the settle and the kick."""
        rest = (self._settle_left + self._kick_left).float() / (self._settle_ticks + self._kick_ticks)
        return torch.where(self.aligning, torch.ones_like(rest), rest)

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

        # A trigger with the ball within reach starts the routine: place the ball on the foot on its
        # side (`_align_step`), stop and settle, then kick or cancel (`_end_settle`).
        foot = kick_side(ball_in_frame(self._env, self.cfg.entity_name))
        start = (decision[:, 3] > 0) & ~self.busy & (foot != KICK_NONE)
        # In place, never rebound: these buffers were made outside the rollout's inference_mode,
        # and a tensor rebound inside it could not be reset in place outside it afterwards.
        self._align_left[:] = torch.where(start, torch.full_like(self._align_left, self._align_ticks), self._align_left)
        self._align_foot[:] = torch.where(start, foot, self._align_foot)
        self.fired_now[:] = False
        # A new decision starts a walker tick, so the twist lands without a 20 ms wait.
        self._substep = 0

    def _align_step(self) -> None:
        """Each walker tick while closing in: walk at `approach_cmd`, steering the ball toward the
        chosen foot, until it is `stop_x` ahead; then hand over to the settle. Gives up after the timeout or if the ball goes
        out of reach (pushed away, taken by the opponent)."""
        rel = ball_in_frame(self._env, self.cfg.entity_name)
        cfg = self.cfg
        self._align_twist[:, 0] = cfg.approach_cmd
        self._align_twist[:, 1] = 0.0
        self._align_twist[:, 2] = routine_steer(rel, self._align_foot)
        aligning = self.aligning
        placed = aligning & (rel[:, 0] <= cfg.stop_x)
        lost = aligning & ((kick_side(rel) == KICK_NONE) | (self._align_left == 1))
        self._settle_left[:] = torch.where(placed, torch.full_like(self._settle_left, self._settle_ticks), self._settle_left)
        self.kicks_cancelled += (lost & ~placed).sum()
        self._align_left[:] = torch.where(placed | lost, torch.zeros_like(self._align_left), (self._align_left - 1).clamp(min=0))

    def _end_settle(self) -> None:
        """At the walker tick a settle runs out: kick if the duck is still and the ball is in a
        foot's FIRE band, otherwise cancel back to walking."""
        ending = self._settle_left == 1
        foot = fire_side(ball_in_frame(self._env, self.cfg.entity_name))
        still = self._entity.data.root_link_lin_vel_w[:, :2].norm(dim=-1) < self.cfg.still_speed
        go = ending & still & (foot != KICK_NONE)
        self._kick_left[:] = torch.where(go, torch.full_like(self._kick_left, self._kick_ticks), self._kick_left)
        self._kick_foot[:] = torch.where(go, foot, self._kick_foot)
        self.fired_now |= go
        self.kicks_fired += go.sum()
        self.kicks_cancelled += (ending & ~go).sum()
        self._settle_left.sub_(1).clamp_(min=0)

    def _policy_obs(self) -> torch.Tensor:
        d = self._entity.data
        jid = self._obs_joint_ids
        # Placing the ball follows the routine's own twist; settling stands still, and the kicks were
        # trained on an all-zero command.
        twist = torch.where(self.busy.unsqueeze(-1), torch.zeros_like(self._twist), self._twist)
        twist = torch.where(self.aligning.unsqueeze(-1), self._align_twist, twist)
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
            self._align_step()
            self._end_settle()
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
        self._settle_left[env_ids] = 0
        self._align_left[env_ids] = 0
        self._align_foot[env_ids] = KICK_NONE
        self._kick_foot[env_ids] = KICK_NONE
