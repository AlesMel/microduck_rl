"""Observations, rewards and terminations for the 1v1 match.

**One policy plays both ends.** Every quantity is expressed in a *team frame*: blue's is the field
frame (attack +x), red's is the field frame turned 180° about z. A duck's observation therefore
reads the same whichever end it defends, which is what lets the frozen opponent be a snapshot of
the learner itself — Unity's soccer agents get the same property from their mirrored raycasts.

**Ground truth, not perception.** Positions and velocities are read from the simulator. A robot on
a real pitch has none of this (no ball detector, no field localization); `soccer_referee.py` feeds
the same vectors from duck-sim's world state, and on hardware they would have to come from an
overhead camera. The policy is a sim tool until then.

Everything here takes entity names as parameters — the microduck helpers in `tasks/mdp.py`
hard-code `env.scene["robot"]` and write qpos[0:7] directly, and there are two robots here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_microduck.soccer.field import FIELD

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

BLUE, RED = "robot_blue", "robot_red"
BALL = "ball"
BALL_RADIUS = 0.035

# A duck lying on its side reads gravity z ~ 0; upright is -1. Past 60° it is not playing.
FALLEN_GRAVITY_Z = -0.5

OBS_DIM = 26


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rotate(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors by -angle: world/team frame -> a frame yawed by `angle`."""
    c, s = torch.cos(angle), torch.sin(angle)
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack((c * x + s * y, -s * x + c * y), dim=-1)


def side_obs(
    own_pos: torch.Tensor,
    own_quat: torch.Tensor,
    own_vel: torch.Tensor,
    own_yaw_rate: torch.Tensor,
    own_gravity_z: torch.Tensor,
    opp_pos: torch.Tensor,
    opp_quat: torch.Tensor,
    opp_vel: torch.Tensor,
    opp_gravity_z: torch.Tensor,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    time_left: torch.Tensor,
    team: int,
) -> torch.Tensor:
    """The match as one duck sees it, 26 values. Positions are env-local (origin = centre spot).

    `team` is +1 for blue and -1 for red: red's world is turned 180° first, so both see their own
    goal at -x. Shared with the duck-sim referee, which is why it takes tensors and not an env.
    """
    flip = float(team)
    own_xy, opp_xy, ball_xy = own_pos[..., :2] * flip, opp_pos[..., :2] * flip, ball_pos[..., :2] * flip
    own_v, opp_v, ball_v = own_vel[..., :2] * flip, opp_vel[..., :2] * flip, ball_vel[..., :2] * flip
    turn = 0.0 if team > 0 else torch.pi
    own_yaw = _yaw(own_quat) + turn
    opp_yaw = _yaw(opp_quat) + turn

    hl, hw = FIELD.half_length, FIELD.half_width
    scale = own_xy.new_tensor((1 / hl, 1 / hw))
    attack_goal = own_xy.new_tensor((hl, 0.0)).expand_as(own_xy)
    defend_goal = own_xy.new_tensor((-hl, 0.0)).expand_as(own_xy)
    rel_heading = opp_yaw - own_yaw

    return torch.cat(
        (
            own_xy * scale,  # 2: where I am on the pitch
            torch.stack((torch.sin(own_yaw), torch.cos(own_yaw)), dim=-1),  # 2: which way I face
            _rotate(own_v, own_yaw),  # 2: my velocity, my frame
            own_yaw_rate.unsqueeze(-1),  # 1
            _rotate(ball_xy - own_xy, own_yaw),  # 2: the ball, my frame
            _rotate(ball_v, own_yaw),  # 2
            ball_xy * scale,  # 2: the ball on the pitch
            _rotate(opp_xy - own_xy, own_yaw),  # 2: the opponent, my frame
            torch.stack((torch.sin(rel_heading), torch.cos(rel_heading)), dim=-1),  # 2
            _rotate(opp_v, own_yaw),  # 2
            _rotate(attack_goal - own_xy, own_yaw),  # 2: where to shoot
            _rotate(defend_goal - own_xy, own_yaw),  # 2: what to protect
            own_gravity_z.unsqueeze(-1),  # 1: am I up
            opp_gravity_z.unsqueeze(-1),  # 1: is the opponent up
            time_left.unsqueeze(-1),  # 1: fraction of the match left
        ),
        dim=-1,
    )


def _local(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    return env.scene[name].data.root_link_pos_w - env.scene.env_origins


def match_obs(env: ManagerBasedRlEnv, team: int) -> torch.Tensor:
    """`side_obs` for blue (team=+1) or red (team=-1), read from the scene."""
    own, opp = (BLUE, RED) if team > 0 else (RED, BLUE)
    own_d, opp_d, ball_d = env.scene[own].data, env.scene[opp].data, env.scene[BALL].data
    time_left = 1.0 - env.episode_length_buf.float() / env.max_episode_length
    return side_obs(
        _local(env, own),
        own_d.root_link_quat_w,
        own_d.root_link_lin_vel_w,
        own_d.root_link_ang_vel_w[:, 2],
        own_d.projected_gravity_b[:, 2],
        _local(env, opp),
        opp_d.root_link_quat_w,
        opp_d.root_link_lin_vel_w,
        opp_d.projected_gravity_b[:, 2],
        _local(env, BALL),
        ball_d.root_link_lin_vel_w,
        time_left,
        team,
    )


# ── the referee ───────────────────────────────────────────────────────────────


def goal_flags(ball_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(blue scored, red scored): the whole ball past a goal line, between the posts, under the bar."""
    x, y, z = ball_local.unbind(-1)
    mouth = (y.abs() < FIELD.goal_width / 2) & (z < FIELD.goal_height)
    past = FIELD.half_length + BALL_RADIUS
    return mouth & (x > past), mouth & (x < -past)


def ball_out(ball_local: torch.Tensor) -> torch.Tensor:
    """The ball left the pitch some way the walls should have prevented (over a wall, through one)."""
    x, y, z = ball_local.unbind(-1)
    return (x.abs() > FIELD.half_length + FIELD.goal_depth + 0.1) | (y.abs() > FIELD.half_width + 0.1) | (z < -0.1)


class MatchStats:
    """Outcome counts since the self-play runner last read them: blue wins, red wins, draws.

    Updated out of place, never `+=`/`zero_()`: the rollout runs under `torch.inference_mode()`
    and the runner reads outside it, and an in-place op on an inference tensor outside that mode
    is an error.
    """

    def __init__(self, device):
        self._device = device
        self._step = -1
        self.blue_scored = self.red_scored = None
        self._zero()

    def _zero(self) -> None:
        self.blue = self.red = self.draws = torch.zeros((), device=self._device)

    def take(self) -> tuple[float, float, float]:
        out = (float(self.blue), float(self.red), float(self.draws))
        self._zero()
        return out


def match_stats(env: ManagerBasedRlEnv) -> MatchStats:
    stats = getattr(env, "_soccer_stats", None)
    if stats is None:
        stats = env._soccer_stats = MatchStats(env.device)
    return stats


def _referee(env: ManagerBasedRlEnv) -> MatchStats:
    """Decide this step's goals once, however many terms ask, and count the outcomes."""
    stats = match_stats(env)
    if stats._step != env.common_step_counter:
        stats._step = env.common_step_counter
        stats.blue_scored, stats.red_scored = goal_flags(_local(env, BALL))
        stats.blue = stats.blue + stats.blue_scored.sum()
        stats.red = stats.red + stats.red_scored.sum()
        timed_out = env.episode_length_buf >= env.max_episode_length
        stats.draws = stats.draws + (timed_out & ~stats.blue_scored & ~stats.red_scored).sum()
    return stats


def goal_scored(env: ManagerBasedRlEnv, by: str = "blue") -> torch.Tensor:
    """Termination and reward: 1 on the step `by` puts the ball in the other goal."""
    stats = _referee(env)
    return stats.blue_scored if by == "blue" else stats.red_scored


def goal_scored_reward(env: ManagerBasedRlEnv, by: str = "blue") -> torch.Tensor:
    return goal_scored(env, by).float()


def ball_left_pitch(env: ManagerBasedRlEnv) -> torch.Tensor:
    return ball_out(_local(env, BALL))


def any_state_is_nan(env: ManagerBasedRlEnv, names: tuple[str, ...] = (BLUE, RED, BALL)) -> torch.Tensor:
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in names:
        d = env.scene[name].data
        bad |= ~torch.isfinite(d.root_link_pos_w).all(-1) | ~torch.isfinite(d.root_link_lin_vel_w).all(-1)
    return bad


# ── shaping (annealed away by the curriculum; the match itself is the goal reward) ──


def ball_approach_speed(env: ManagerBasedRlEnv, asset_name: str = BLUE, max_speed: float = 0.5) -> torch.Tensor:
    """How fast the duck closes on the ball, m/s, clamped. The rate form of a distance potential,
    so it needs no state across resets and standing next to the ball pays nothing."""
    duck, ball = env.scene[asset_name].data, env.scene[BALL].data
    to_ball = (_local(env, BALL) - _local(env, asset_name))[:, :2]
    direction = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    closing = ((duck.root_link_lin_vel_w[:, :2] - ball.root_link_lin_vel_w[:, :2]) * direction).sum(-1)
    return closing.clamp(-max_speed, max_speed)


def ball_to_goal_speed(env: ManagerBasedRlEnv, team: int = 1, max_speed: float = 1.0) -> torch.Tensor:
    """Ball velocity toward the goal `team` attacks, m/s, clamped."""
    ball_xy = _local(env, BALL)[:, :2]
    goal = ball_xy.new_tensor((team * FIELD.half_length, 0.0))
    to_goal = goal - ball_xy
    direction = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    speed = (env.scene[BALL].data.root_link_lin_vel_w[:, :2] * direction).sum(-1)
    return speed.clamp(-max_speed, max_speed)


def fallen(env: ManagerBasedRlEnv, asset_name: str = BLUE) -> torch.Tensor:
    """1 while the duck is down. A cost (>= 0), so its weight is negative."""
    return (env.scene[asset_name].data.projected_gravity_b[:, 2] > FALLEN_GRAVITY_Z).float()


def existential(env: ManagerBasedRlEnv) -> torch.Tensor:
    """1 every step. With a negative weight, Unity's existential penalty: a draw is a small loss."""
    return torch.ones(env.num_envs, device=env.device)
