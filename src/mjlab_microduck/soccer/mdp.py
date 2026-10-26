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
from mjlab_microduck.soccer.scenarios import APPROACH, KICKOFF, SHOOT, TRAIN_PROBS, yaw_quat
from mjlab_microduck.soccer.scenarios import NAMES as SCENARIO_NAMES
from mjlab_microduck.soccer.scenarios import sample as sample_layout

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

BLUE, RED = "robot_blue", "robot_red"
BALL = "ball"
BALL_RADIUS = 0.035
TRUNK_Z = 0.125  # velstand's reset height
DRILL_SECONDS = 10.0

# A duck lying on its side reads gravity z ~ 0; upright is -1. Past 60° it is not playing.
FALLEN_GRAVITY_Z = -0.5

OBS_DIM = 27

# Where the ball must be, in the duck's yaw frame, for a kick to fire: the BallKick policies were
# trained with the ball at (0.09, -/+0.042) m +-0.015 (`reset_ball_in_front_of_foot`), so this is
# that placement with some margin. y > 0 is the left foot's side.
KICK_ZONE_X = (0.05, 0.25)  # REACH: far enough out that the routine has room to steer
KICK_ZONE_Y = 0.08
KICK_NONE, KICK_LEFT, KICK_RIGHT = 0, 1, 2
# Which action term drives which team, for the kick state in each side's observation.
ACTION_TERMS = {1: "blue_walk", -1: "red_walk"}


def kick_side(ball_rel_xy: torch.Tensor) -> torch.Tensor:
    """REACH zone: KICK_LEFT / KICK_RIGHT if the ball (in the duck's yaw frame) is near enough in
    front of that foot to start the stop-then-kick routine, else KICK_NONE. Shared by the training
    action term and the duck-sim referee."""
    x, y = ball_rel_xy[..., 0], ball_rel_xy[..., 1]
    inside = (x >= KICK_ZONE_X[0]) & (x <= KICK_ZONE_X[1]) & (y.abs() <= KICK_ZONE_Y)
    side = torch.where(y >= 0, torch.full_like(x, KICK_LEFT), torch.full_like(x, KICK_RIGHT))
    return torch.where(inside, side, torch.full_like(x, KICK_NONE)).long()


# FIRE band: where a kick from a standing duck actually connects, measured 2026-10-26 by kicking
# from rest with the ball on a grid. 8-10 cm ahead and 2-6.5 cm toward the kicking foot sends it at
# 1.0-1.3 m/s within 8-17 degrees of the heading; elsewhere in the REACH zone the kick misses it
# (0-0.3 m/s) or slices it 30+ degrees. BallKick was trained at (0.09, 0.042) +-0.015.
FIRE_X = (0.075, 0.105)
FIRE_Y = (0.015, 0.07)


# The stop-then-kick routine's numbers, shared by the training action term and the duck-sim referee
# (which must not import mjlab). Measured 2026-10-26 on velstand v8 — see `WalkerCommandActionCfg`.
ROUTINE = {
    "approach_cmd": 0.25,  # slowest forward command velstand actually walks at (0.08 m/s)
    "stop_x": 0.105,  # stop with the ball this far ahead: ~1.5 cm of drift lands it mid-FIRE
    "settle_s": 0.5,
    "still_speed": 0.03,
    "foot_y": 0.042,  # steer the ball this far toward the kicking foot
    "steer_gain": 2.0,
    "steer_max": 0.5,
    "timeout_s": 3.0,
    "kick_s": 0.5,  # the set manifest's duration_s
    # An opponent trunk this near the ball calls the routine off (or never starts it): two ducks
    # that both stop to kick a ball between them freeze there, 86% of oracle-vs-oracle points
    # ended drawn. Contested balls are dribbled: walked through at full speed, which scores 62%
    # of 0.7 m shots in ~3.7 s.
    "contest_dist": 0.25,
}


def routine_steer(ball_rel_xy: torch.Tensor, foot: torch.Tensor) -> torch.Tensor:
    """Yaw command (rad/s) bringing the ball toward `foot`'s side while walking in."""
    side = torch.where(foot == KICK_RIGHT, -1.0, 1.0)
    bearing = (ball_rel_xy[..., 1] - side * ROUTINE["foot_y"]) / ball_rel_xy[..., 0].clamp(min=0.08)
    return (ROUTINE["steer_gain"] * bearing).clamp(-ROUTINE["steer_max"], ROUTINE["steer_max"])


def contested(ball_xy: torch.Tensor, opponent_xy: torch.Tensor) -> torch.Tensor:
    """True where the opponent is close enough to the ball that stopping to kick it would freeze."""
    return (opponent_xy - ball_xy).norm(dim=-1) < ROUTINE["contest_dist"]


def fire_side(ball_rel_xy: torch.Tensor) -> torch.Tensor:
    """KICK_LEFT / KICK_RIGHT if the ball is in that foot's FIRE band, else KICK_NONE."""
    x, y = ball_rel_xy[..., 0], ball_rel_xy[..., 1]
    ahead = (x >= FIRE_X[0]) & (x <= FIRE_X[1])
    band = (y.abs() >= FIRE_Y[0]) & (y.abs() <= FIRE_Y[1])
    side = torch.where(y >= 0, torch.full_like(x, KICK_LEFT), torch.full_like(x, KICK_RIGHT))
    return torch.where(ahead & band, side, torch.full_like(x, KICK_NONE)).long()


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
    own_kick: torch.Tensor,
    team: int,
) -> torch.Tensor:
    """The match as one duck sees it, 27 values. Positions are env-local (origin = centre spot).

    `own_kick` is the fraction of a running kick still to go (0 when walking): while it is
    non-zero the duck's twist is ignored, and the value function needs to know that.

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
            own_kick.unsqueeze(-1),  # 1: fraction of my kick still running
        ),
        dim=-1,
    )


def to_frame(own_pos: torch.Tensor, own_quat: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """`point`'s planar position in the yaw frame of a body at `own_pos` / `own_quat`."""
    return _rotate((point - own_pos)[..., :2], _yaw(own_quat))


def ball_in_frame(env: ManagerBasedRlEnv, asset_name: str) -> torch.Tensor:
    """The ball's planar position in `asset_name`'s yaw frame."""
    return to_frame(_local(env, asset_name), env.scene[asset_name].data.root_link_quat_w, _local(env, BALL))


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
        env.action_manager.get_term(ACTION_TERMS[team]).kick_fraction,
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
    """What the self-play runner and the competence gate read about how play is going.

    - blue / red / draws: outcomes of KICKOFF points only — real matches, what Elo is about. The
      practice situations are not a match against the opponent and would skew it.
    - points / blue_goals / red_goals (4,): per practice situation, for `Scenario/*` logging.
    - approach_rate: running fraction of APPROACH drills in which blue reached the ball.
    - shoot_rate: running fraction of SHOOT drills blue scored from.

    Updated out of place, never `+=`/`zero_()`: the rollout runs under `torch.inference_mode()`
    and the runner reads outside it, and an in-place op on an inference tensor outside that mode
    is an error.
    """

    def __init__(self, device):
        self._device = device
        self._step = -1
        self.blue_scored = self.red_scored = None
        self.approach_rate = torch.zeros((), device=device)
        self.shoot_rate = torch.zeros((), device=device)
        self._zero()

    def _zero(self) -> None:
        z = torch.zeros((), device=self._device)
        self.blue = self.red = self.draws = z
        z4 = torch.zeros(len(SCENARIO_NAMES), device=self._device)
        self.points = self.blue_goals = self.red_goals = z4
        # Kickoff outcomes by opponent kind: rows statue / oracle / snapshot, columns win / loss / draw.
        self.vs = torch.zeros(len(OPP_NAMES), 3, device=self._device)

    def take(self) -> tuple[float, float, float]:
        """Kickoff outcomes since the last call (blue wins, red wins, draws). Also clears the
        per-situation counts: read `take_scenarios` first."""
        out = (float(self.blue), float(self.red), float(self.draws))
        self._zero()
        return out

    def take_vs(self) -> dict[str, tuple[float, float, float]]:
        """Kickoff (wins, losses, draws) per opponent kind since the last `take`. Read before it."""
        return {name: tuple(float(v) for v in self.vs[i]) for i, name in enumerate(OPP_NAMES)}

    def take_scenarios(self) -> dict[str, float]:
        out = {}
        for i, name in enumerate(SCENARIO_NAMES):
            n = float(self.points[i])
            if n > 0:
                out[f"{name}_score"] = float(self.blue_goals[i]) / n
                out[f"{name}_concede"] = float(self.red_goals[i]) / n
        out["approach_rate"] = float(self.approach_rate)
        out["shoot_rate"] = float(self.shoot_rate)
        return out


# Who red is in each env, set by the self-play runner: a statue, the scripted oracle, or a frozen
# snapshot of the learner. Kickoff outcomes are counted per kind so each yardstick stays clean.
OPP_STATUE, OPP_ORACLE, OPP_SNAPSHOT = 0, 1, 2
OPP_NAMES = ("statue", "oracle", "snapshot")


def opponent_kind(env: ManagerBasedRlEnv) -> torch.Tensor:
    kinds = getattr(env, "_soccer_opp_kind", None)
    if kinds is None:
        kinds = env._soccer_opp_kind = torch.full((env.num_envs,), OPP_SNAPSHOT, dtype=torch.long, device=env.device)
    return kinds


# Each finished drill moves its running rate this much toward its own outcome: ~1000 drills of memory,
# about one PPO iteration's worth at 1024 envs.
RATE_STEP = 0.001
# "Reached the ball" in an approach drill: trunk within this of the ball's centre.
TOUCH_DIST = 0.15


def match_stats(env: ManagerBasedRlEnv) -> MatchStats:
    stats = getattr(env, "_soccer_stats", None)
    if stats is None:
        stats = env._soccer_stats = MatchStats(env.device)
    return stats


def scenario(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Each env's practice situation (`scenarios.KICKOFF`, ...), set by `reset_scenario`."""
    kinds = getattr(env, "_soccer_scenario", None)
    if kinds is None:
        kinds = env._soccer_scenario = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._soccer_touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return kinds


def drill_timed_out(env: ManagerBasedRlEnv, seconds: float = DRILL_SECONDS) -> torch.Tensor:
    """A practice situation's clock ran out. A drill is a few seconds of one skill, not a match."""
    return (scenario(env) != KICKOFF) & (env.episode_length_buf >= round(seconds / env.step_dt))


def _referee(env: ManagerBasedRlEnv) -> MatchStats:
    """Decide this step's goals once, however many terms ask, and count the outcomes."""
    stats = match_stats(env)
    if stats._step == env.common_step_counter:
        return stats
    stats._step = env.common_step_counter
    stats.blue_scored, stats.red_scored = goal_flags(_local(env, BALL))
    kinds = scenario(env)
    touched = env._soccer_touched
    touched |= (_local(env, BALL) - _local(env, BLUE))[:, :2].norm(dim=-1) < TOUCH_DIST

    timed_out = env.episode_length_buf >= env.max_episode_length
    ended = stats.blue_scored | stats.red_scored | timed_out | drill_timed_out(env)
    goal = stats.blue_scored | stats.red_scored
    kickoff = kinds == KICKOFF
    stats.blue = stats.blue + (stats.blue_scored & kickoff).sum()
    stats.red = stats.red + (stats.red_scored & kickoff).sum()
    stats.draws = stats.draws + (timed_out & ~goal & kickoff).sum()
    outcome = torch.stack((stats.blue_scored, stats.red_scored, timed_out & ~goal), -1) & kickoff.unsqueeze(-1)
    by_opp = opponent_kind(env).unsqueeze(-1) == torch.arange(len(OPP_NAMES), device=env.device)
    stats.vs = stats.vs + (by_opp.unsqueeze(-1) & outcome.unsqueeze(1)).sum(0)

    per_kind = ended.unsqueeze(-1) & (kinds.unsqueeze(-1) == torch.arange(len(SCENARIO_NAMES), device=env.device))
    stats.points = stats.points + per_kind.sum(0)
    stats.blue_goals = stats.blue_goals + (per_kind & stats.blue_scored.unsqueeze(-1)).sum(0)
    stats.red_goals = stats.red_goals + (per_kind & stats.red_scored.unsqueeze(-1)).sum(0)

    done_approach, done_shoot = per_kind[:, APPROACH], per_kind[:, SHOOT]
    stats.approach_rate = stats.approach_rate + RATE_STEP * (
        (done_approach & touched).sum() - stats.approach_rate * done_approach.sum()
    )
    stats.shoot_rate = stats.shoot_rate + RATE_STEP * (
        (done_shoot & stats.blue_scored).sum() - stats.shoot_rate * done_shoot.sum()
    )
    return stats


def reset_scenario(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, probs: tuple[float, ...] = TRAIN_PROBS) -> None:
    """Reset event: draw each env's practice situation and place both ducks and the ball for it.

    After `reset_scene_to_default`, which has already put every joint at the default pose.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    kinds = scenario(env)
    weights = torch.tensor(probs, device=env.device, dtype=torch.float)
    drawn = torch.multinomial(weights, len(env_ids), replacement=True)
    kinds[env_ids] = drawn
    env._soccer_touched[env_ids] = False
    layout = sample_layout(drawn)
    origin = env.scene.env_origins[env_ids]
    n = len(env_ids)
    zeros6 = torch.zeros(n, 6, device=env.device)

    def state(xy, z, quat):
        pos = torch.cat((xy, torch.full((n, 1), z, device=env.device)), dim=-1) + origin
        return torch.cat((pos, quat, zeros6), dim=-1)

    env.scene[BLUE].write_root_state_to_sim(state(layout.blue_xy, TRUNK_Z, yaw_quat(layout.blue_yaw)), env_ids)
    env.scene[RED].write_root_state_to_sim(state(layout.red_xy, TRUNK_Z, yaw_quat(layout.red_yaw)), env_ids)
    upright = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(n, 4)
    env.scene[BALL].write_root_state_to_sim(state(layout.ball_xy, BALL_RADIUS, upright), env_ids)


def competence_gate(env: ManagerBasedRlEnv, env_ids: torch.Tensor, terms: dict) -> dict:
    """Curriculum: scale each shaping term by how well its skill is already done, not by the clock.

    `terms` maps a reward name to (base weight, signal, lo, hi, floor) with signal "approach" or
    "shoot": full weight while the skill's success rate is below `lo`, falling linearly to
    `floor` x base at `hi`. Every run annealed on a fixed iteration count, and every one collapsed
    into draws when shaping went before scoring had been learned.
    """
    del env_ids
    stats = match_stats(env)
    signal = {"approach": float(stats.approach_rate), "shoot": float(stats.shoot_rate)}
    out = {"approach_rate": signal["approach"], "shoot_rate": signal["shoot"]}
    for name, (base, which, lo, hi, floor) in terms.items():
        progress = min(1.0, max(0.0, (signal[which] - lo) / (hi - lo)))
        scale = 1.0 - progress * (1.0 - floor)
        env.reward_manager.get_term_cfg(name).weight = base * scale
        out[f"{name}_scale"] = scale
    return out


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


def _heading(env: ManagerBasedRlEnv, asset_name: str) -> torch.Tensor:
    yaw = _yaw(env.scene[asset_name].data.root_link_quat_w)
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def approach_facing(env: ManagerBasedRlEnv, asset_name: str = BLUE, max_speed: float = 0.5) -> torch.Tensor:
    """Closing speed on the ball (m/s, >= 0, clamped) times cos(heading, ball): paid only for
    walking *at* the ball face first. Negative when closing on it backwards.

    Replaces a plain "face the ball" term, which paid w per second for staring at the ball and
    nothing inside 12 cm: the kick run (2026-10-26) learned to hover 20-25 cm away facing it — the
    kick zone is 5-14 cm — and stopped kicking altogether (0 kicks, 100% draws by iteration 600).
    This one pays nothing standing still and nothing for keeping a distance, so the only way to
    collect it is to arrive at the ball facing it, which is where a kick starts.
    """
    duck, ball = env.scene[asset_name].data, env.scene[BALL].data
    to_ball = _unit((_local(env, BALL) - _local(env, asset_name))[:, :2])
    closing = ((duck.root_link_lin_vel_w[:, :2] - ball.root_link_lin_vel_w[:, :2]) * to_ball).sum(-1)
    facing = (_heading(env, asset_name) * to_ball).sum(-1)
    return closing.clamp(0.0, max_speed) * facing


def drive_through(
    env: ManagerBasedRlEnv, asset_name: str = BLUE, team: int = 1, max_speed: float = 0.4, near: float = 0.25
) -> torch.Tensor:
    """The duck's velocity toward the goal it attacks (m/s, >= 0, clamped), scaled by how squarely it
    stands behind the ball on that line: duck -> ball -> goal. Running through the ball at goal pays;
    standing still, standing beside the ball, or between the ball and the goal pays nothing.

    Paid on the duck's motion, but only where that motion carries the ball goalward, so it cannot be
    collected without bringing the ball along. It replaces a reward for firing a kick aimed at goal,
    which the aim run (2026-10-26) farmed: both ducks within 20 cm of the ball 88% of the time,
    kicking it into each other nonstop — 832 kicks in 16 matches and the ball never moved.

    Only within `near` of the ball, which is dribbling range: without it the drive run collected it
    from a median 0.39 m behind the ball, a third of the time from beyond 0.5 m, never touching it.
    """
    ball_xy = _local(env, BALL)[:, :2]
    duck_xy = _local(env, asset_name)[:, :2]
    to_goal = _unit(ball_xy.new_tensor((team * FIELD.half_length, 0.0)) - ball_xy)
    behind = (_unit(ball_xy - duck_xy) * to_goal).sum(-1)
    goalward = (env.scene[asset_name].data.root_link_lin_vel_w[:, :2] * to_goal).sum(-1)
    close = ((ball_xy - duck_xy).norm(dim=-1) <= near).float()
    return goalward.clamp(0.0, max_speed) * behind.clamp(0.0, 1.0) * close


def potential(env: ManagerBasedRlEnv, kind: str, team: int = 1) -> torch.Tensor:
    """Phi for potential shaping, in metres (<= 0, higher is better).

    "approach": minus blue's distance to the ball. "ball": minus the ball's distance to the centre
    of the goal `team` attacks.
    """
    ball = _local(env, BALL)[:, :2]
    if kind == "approach":
        return -(ball - _local(env, BLUE if team > 0 else RED)[:, :2]).norm(dim=-1)
    goal = ball.new_tensor((team * FIELD.half_length, 0.0))
    return -(goal - ball).norm(dim=-1)


class PotentialShaping:
    """Reward term: Phi(s') - Phi(s), Ng et al.'s potential-based shaping.

    Summed over a point it telescopes to Phi(end) - Phi(start): circling, hovering, pushing back and
    forth or standing still earn exactly nothing, only being closer at the end than at the start
    does. That is the property every rate-based shaping term in this task lacked, and each was
    farmed in turn. Undiscounted on purpose: with gamma < 1 the (gamma - 1) Phi term pays a little
    every step for being far away.

    The first step after a reset pays 0 and records Phi, so a point never inherits the jump from
    the previous point's final state.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.prev = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRlEnv, kind: str = "approach", team: int = 1) -> torch.Tensor:
        phi = potential(env, kind, team)
        out = torch.where(self.fresh, torch.zeros_like(phi), phi - self.prev)
        self.prev[:] = phi
        self.fresh[:] = False
        return out


class KickOutcome:
    """Reward term: what blue's kick did, `delay_s` after it fired — the ball's speed toward the goal
    blue attacks (m/s, clamped). A kick into the opponent, the air or a wall pays nothing; one at
    the own goal costs.

    Replaces paying the kick itself (`kick_aim`), which the aim run collected 832 times in 16
    matches without the ball ever moving.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.countdown = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.countdown[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self, env: ManagerBasedRlEnv, term: str = "blue_walk", team: int = 1, delay_s: float = 0.5, max_speed: float = 1.0
    ) -> torch.Tensor:
        due = self.countdown == 1
        speed = ball_to_goal_speed(env, team=team, max_speed=max_speed)
        out = torch.where(due, speed, torch.zeros_like(speed))
        self.countdown.sub_(1).clamp_(min=0)
        fired = env.action_manager.get_term(term).fired_now
        self.countdown[:] = torch.where(fired, torch.full_like(self.countdown, round(delay_s / env.step_dt) + 1), self.countdown)
        return out


def fallen(env: ManagerBasedRlEnv, asset_name: str = BLUE) -> torch.Tensor:
    """1 while the duck is down. A cost (>= 0), so its weight is negative."""
    return (env.scene[asset_name].data.projected_gravity_b[:, 2] > FALLEN_GRAVITY_Z).float()


def existential(env: ManagerBasedRlEnv) -> torch.Tensor:
    """1 every step. With a negative weight, Unity's existential penalty: a draw is a small loss."""
    return torch.ones(env.num_envs, device=env.device)
