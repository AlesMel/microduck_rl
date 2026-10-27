"""A scripted striker: the teacher the match policy imitates first, and a graded opponent.

It plays from ground truth like the policy does, and outputs the policy's own 4D action, so it can
stand in for either side. It is written around what velstand and the kicks can physically do,
all measured 2026-10-26:

- Forward commands below ~0.25 m/s don't move the duck at all, and it can't side-step, so it
  walks at full speed or turns in place.
- Turning in place needs |yaw| >= ~0.8 rad/s.
- A kick only connects from a standstill with the ball in a 3 cm band, so the last 20 cm belong to
  the action term's stop-then-kick routine: the oracle stops on a spot behind the ball, turns to
  face the goal, and triggers.

Attack: go round the ball to the spot behind it on the ball -> goal line, face the goal, trigger.
The goal point is the middle of the goal, unless the opponent stands between the ball and the
goal: then it is `AIM_Y` toward the post away from them. A defender holding the shot line ~0.55 m
ahead of the ball stops nearly every shot at its middle (the trained policy blocked 96% of the
oracle's on-target shots at the 0.6 m goal, the ball meeting its front between the feet), so the
goal is 1.2 m wide (`field.Field`) and the oracle shoots inside the open post: 57% of its goalward
kicks then scored against that same defender.
Dribble: with the opponent at the ball (`contested`) the routine won't run, so from anywhere
behind the ball the oracle walks it toward the goal at full speed instead — which also makes it
the tackle: the defender is goal-side, so it pushes the ball away from its own goal.
Defend: when the opponent is nearer the ball and the ball is in the oracle's own half, stand
between the ball and its own goal, facing the ball.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab_microduck.soccer.field import FIELD
from mjlab_microduck.soccer.mdp import BALL, KICK_NONE, _local, _unit, _yaw, ball_in_frame, contested, kick_side

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

SETUP_DIST = 0.20  # stand this far behind the ball: inside REACH, so the routine takes over
ROUND_DIST = 0.30  # how wide to go round a ball approached from the wrong side
BEHIND_MARGIN = 0.12  # counts as behind the ball once this far back along the aim line
ARRIVE = 0.06  # close enough to a target point to stop walking
TURN_IN_PLACE = 0.6  # rad of heading error above which the oracle stops and turns
AIM_TOL = 0.12  # rad: aimed well enough to trigger
GUARD_DIST = 0.35  # defending: this far from the ball toward its own goal
AIM_Y = FIELD.goal_width / 2 - 0.15  # shooting past a defender: this far from the goal's centre toward the open post
SIDE_HYSTERESIS = 0.04  # m of the defender's offset from the shot line before the aim switches post
DRIBBLE_BEHIND = 0.05  # dribbling: walk at a point this far behind the ball on its push line
ESCAPE_ANGLE = 0.9  # rad: with the opponent in front of the ball, push it this far round them
BLOCK_WIDTH = 0.2  # opponent within this of the ball -> goal line counts as in front of the ball


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class Oracle:
    """Scripted player for entity `me` attacking the goal at x = team * half_length.

    `skill` in (0, 1], per env or scalar: the chance of acting on a fresh decision each step; the
    rest of the time it repeats its last action. 1 is the full oracle; lower is a sluggish one.
    """

    def __init__(self, env: ManagerBasedRlEnv, me: str, opp: str, team: int, skill: float | torch.Tensor = 1.0):
        self.env, self.me, self.opp, self.team = env, me, opp, team
        n = env.num_envs
        self.skill = torch.as_tensor(skill, dtype=torch.float, device=env.device).expand(n).clone()
        self.last = torch.zeros(n, 4, device=env.device)
        self.last[:, 3] = -1.0
        self.post = torch.ones(n, device=env.device)  # which post to aim inside, +y or -y, when blocked

    def __call__(self, obs: torch.Tensor | None = None) -> torch.Tensor:
        fresh = self.decide()
        keep = torch.rand(self.env.num_envs, device=self.env.device) >= self.skill
        out = torch.where(keep.unsqueeze(-1), self.last, fresh)
        self.last.copy_(out)
        return out

    def decide(self) -> torch.Tensor:
        env, team = self.env, self.team
        pos = _local(env, self.me)[:, :2]
        opp = _local(env, self.opp)[:, :2]
        ball = _local(env, BALL)[:, :2]
        yaw = _yaw(env.scene[self.me].data.root_link_quat_w)
        centre = ball.new_tensor((team * FIELD.half_length, 0.0))
        own_goal = -centre
        to_centre = _unit(centre - ball)
        # Shoot past the opponent when they stand between the ball and the goal: inside the post
        # on the far side from them. The post only switches once they are clearly across the line,
        # or the aim (and the whole approach) would flicker as they shuffle.
        opp_rel = opp - ball
        across = (opp_rel * torch.stack((-to_centre[:, 1], to_centre[:, 0]), -1)).sum(-1)
        in_way = (opp_rel * to_centre).sum(-1) > 0
        # Rebound, not in place: the teacher's oracle can be built inside the rollout's inference_mode.
        self.post = torch.where(across > SIDE_HYSTERESIS, -1.0, torch.where(across < -SIDE_HYSTERESIS, 1.0, self.post))
        goal = centre + torch.stack((torch.zeros_like(across), self.post * AIM_Y * in_way), -1)
        aim = _unit(goal - ball)
        perp = torch.stack((-aim[:, 1], aim[:, 0]), -1)

        # Attack target: the setup spot behind the ball, or a waypoint round it first.
        rel = pos - ball
        behind = (rel * aim).sum(-1) < -BEHIND_MARGIN
        side = torch.sign((rel * perp).sum(-1))
        side = torch.where(side == 0, torch.ones_like(side), side)
        spot = ball - aim * SETUP_DIST
        around = ball - aim * ROUND_DIST + perp * side.unsqueeze(-1) * ROUND_DIST
        attack_target = torch.where(behind.unsqueeze(-1), spot, around)

        # Defend when beaten to the ball in the own half: stand goal-side of it.
        my_dist = (pos - ball).norm(dim=-1)
        opp_dist = (opp - ball).norm(dim=-1)
        own_half = (ball[:, 0] * team) < 0
        defend = own_half & (opp_dist + 0.05 < my_dist)
        guard = ball + _unit(own_goal - ball) * GUARD_DIST
        target = torch.where(defend.unsqueeze(-1), guard, attack_target)

        # Dribble a contested ball from any side behind it: no stop, no kick, just push. With the
        # opponent in front of the ball, pushing at the goal pins it between the two ducks, so push
        # it diagonally round the opponent's side away from them.
        dribble = contested(ball, opp) & ((rel * aim).sum(-1) < 0)
        defend = defend & ~dribble
        lateral = (opp_rel * perp).sum(-1)
        blocked = ((opp_rel * aim).sum(-1) > 0) & (lateral.abs() < BLOCK_WIDTH)
        away = torch.where(lateral > 0, -1.0, 1.0) * ESCAPE_ANGLE * blocked
        c, s_ = torch.cos(away).unsqueeze(-1), torch.sin(away).unsqueeze(-1)
        push = torch.cat((aim[:, :1] * c - aim[:, 1:] * s_, aim[:, :1] * s_ + aim[:, 1:] * c), -1)
        target = torch.where(dribble.unsqueeze(-1), ball - push * DRIBBLE_BEHIND, target)

        to_t = target - pos
        dist = to_t.norm(dim=-1)
        arrived = (dist < ARRIVE) & ~dribble  # a dribble walks on through its target
        # Heading to walk on, or once arrived the heading to hold: the goal when attacking, the
        # ball when guarding.
        face = torch.where(defend, torch.atan2(-rel[:, 1], -rel[:, 0]), torch.atan2(aim[:, 1], aim[:, 0]))
        want = torch.where(arrived, face, torch.atan2(to_t[:, 1], to_t[:, 0]))
        err = _wrap(want - yaw)

        # velstand: full speed or not at all; turn in place only with a strong yaw command.
        turning = err.abs() > torch.where(arrived, torch.full_like(err, AIM_TOL), torch.full_like(err, TURN_IN_PLACE))
        vx = torch.where(arrived | turning, torch.zeros_like(err), torch.ones_like(err))
        yaw_cmd = torch.where(turning, torch.sign(err), (2.0 * err).clamp(-1.0, 1.0))
        yaw_cmd = torch.where(arrived & ~turning, torch.zeros_like(err), yaw_cmd)

        # Trigger once on the spot and aimed, or whenever the ball is already in reach and the duck
        # roughly faces the goal; the routine walks the last centimetres.
        aim_err = _wrap(torch.atan2(aim[:, 1], aim[:, 0]) - yaw).abs()
        in_reach = kick_side(ball_in_frame(env, self.me)) != KICK_NONE
        ready = ((~defend & behind & arrived & (aim_err < AIM_TOL)) | (in_reach & (aim_err < 2 * AIM_TOL))) & ~dribble
        kick = torch.where(ready, torch.ones_like(err), -torch.ones_like(err))
        return torch.stack((vx, torch.zeros_like(vx), yaw_cmd, kick), -1)


def teacher_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Observation term: the oracle's action for blue — the label `PpoWithTeacher` imitates."""
    oracle = getattr(env, "_soccer_teacher", None)
    if oracle is None:
        oracle = env._soccer_teacher = Oracle(env, "robot_blue", "robot_red", 1)
    return oracle.decide()
