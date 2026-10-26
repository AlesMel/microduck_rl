"""The practice situations put the ball, and the ducks, where each drill says — on CPU, no env."""

import math

import torch

from mjlab_microduck.soccer.field import FIELD, KICKOFF_X
from mjlab_microduck.soccer.mdp import KICK_NONE, kick_side, to_frame
from mjlab_microduck.soccer.scenarios import (
    APPROACH, BALL_CLEARANCE, DEFEND, DUCK_CLEARANCE, KICKOFF, SHOOT, X_MAX, Y_MAX, sample, yaw_quat,
)

N = 4000


def _layout(kind):
    return sample(torch.full((N,), kind, dtype=torch.long), generator=torch.Generator().manual_seed(kind))


def _ball_in_frame(xy, yaw, ball):
    pos = torch.cat((xy, torch.zeros(len(xy), 1)), -1)
    return to_frame(pos, yaw_quat(yaw), torch.cat((ball, torch.zeros(len(ball), 1)), -1))


def _angle_to(target, origin, yaw):
    d = target - origin
    return torch.remainder(torch.atan2(d[:, 1], d[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi


def test_everyone_on_the_pitch():
    for kind in (KICKOFF, SHOOT, APPROACH, DEFEND):
        lay = _layout(kind)
        for xy in (lay.blue_xy, lay.red_xy):
            assert (xy[:, 0].abs() <= X_MAX + 1e-6).all() and (xy[:, 1].abs() <= Y_MAX + 1e-6).all(), kind
        assert (lay.ball_xy[:, 0].abs() < FIELD.half_length).all() and (lay.ball_xy[:, 1].abs() < FIELD.half_width).all()


def test_kickoff_is_the_match():
    lay = _layout(KICKOFF)
    assert (lay.blue_xy[:, 0] < 0).all() and (lay.red_xy[:, 0] > 0).all()
    assert torch.allclose(lay.blue_xy[:, 0].mean(), torch.tensor(-KICKOFF_X), atol=0.02)
    assert (lay.ball_xy.norm(dim=-1) < 0.08).all()


def test_shoot_puts_the_ball_at_blues_foot_facing_goal():
    lay = _layout(SHOOT)
    assert (kick_side(_ball_in_frame(lay.blue_xy, lay.blue_yaw, lay.ball_xy)) != KICK_NONE).float().mean() > 0.99
    goal = torch.tensor([FIELD.half_length, 0.0]).expand(N, 2)
    assert (_angle_to(goal, lay.blue_xy, lay.blue_yaw).abs() < 0.35).float().mean() > 0.99
    assert ((lay.red_xy - lay.ball_xy).norm(dim=-1) >= BALL_CLEARANCE - 1e-4).all()


def test_approach_starts_away_from_the_ball():
    lay = _layout(APPROACH)
    dist = (lay.ball_xy - lay.blue_xy).norm(dim=-1)
    assert (dist >= BALL_CLEARANCE - 1e-4).float().mean() > 0.999 and dist.median() > 0.6


def test_defend_gives_red_the_ball_and_blue_the_goal():
    lay = _layout(DEFEND)
    assert (kick_side(_ball_in_frame(lay.red_xy, lay.red_yaw, lay.ball_xy)) != KICK_NONE).float().mean() > 0.99
    own_goal = torch.tensor([-FIELD.half_length, 0.0]).expand(N, 2)
    assert (_angle_to(own_goal, lay.red_xy, lay.red_yaw).abs() < 0.35).float().mean() > 0.99
    # Blue starts goal-side of the ball, and not inside red.
    assert (lay.blue_xy[:, 0] < lay.ball_xy[:, 0]).float().mean() > 0.95
    assert ((lay.blue_xy - lay.red_xy).norm(dim=-1) >= DUCK_CLEARANCE - 1e-4).float().mean() > 0.99
