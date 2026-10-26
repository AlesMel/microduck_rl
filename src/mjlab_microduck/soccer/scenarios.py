"""Practice situations: where a point starts, so that every skill is practised from iteration 0.

Every run that started each point from kickoff spent most of its experience walking to the ball;
a kick that ended in a goal was too rare to learn from, and the policies settled on whatever the
dense shaping paid for. Each reset now draws one of four situations, the same reverse-curriculum
idea AGENTS.md recommends for tricks whose last mile never gets on-policy data:

- KICKOFF:  the match as it is played — each duck on its own half, ball on the centre spot.
- SHOOT:    the ball at blue's foot (BallKick's placement), blue facing the red goal give or take,
            anywhere from just inside its own half to 0.4 m from goal. Red is back in blue's half.
- APPROACH: blue anywhere facing anywhere, the ball 0.5-1.5 m away in any direction.
- DEFEND:   red has the ball at its foot in blue's half and faces blue's goal; blue starts between
            it and the goal, the way a defender would be found.

Pure tensors, no env: `tests/test_soccer_scenarios.py` checks the geometry directly. Positions are
env-local (origin = centre spot); blue always attacks +x.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab_microduck.soccer.field import FIELD, KICKOFF_X

KICKOFF, SHOOT, APPROACH, DEFEND = range(4)
NAMES = ("kickoff", "shoot", "approach", "defend")
TRAIN_PROBS = (0.4, 0.2, 0.2, 0.2)

# Where the ball sits in front of a foot in the BallKick policies' training (`mdp.KICK_ZONE_*`).
FOOT_BALL = (0.09, 0.042)
# Where a duck may be placed: inside the walls with a body's margin, clear of the cut corners.
X_MAX, Y_MAX = FIELD.half_length - 0.15, FIELD.half_width - 0.15
DUCK_CLEARANCE = 0.35  # between the two ducks' trunks
BALL_CLEARANCE = 0.30  # between a duck and a ball that is not meant to be at its feet


@dataclass
class Layout:
    blue_xy: torch.Tensor  # (n, 2)
    blue_yaw: torch.Tensor  # (n,)
    red_xy: torch.Tensor
    red_yaw: torch.Tensor
    ball_xy: torch.Tensor


def _u(n: int, lo: float, hi: float, device, g) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device, generator=g)


def _xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, y), dim=-1)


def _heading(target: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    d = target - origin
    return torch.atan2(d[..., 1], d[..., 0])


def _behind_ball(ball: torch.Tensor, yaw: torch.Tensor, foot: torch.Tensor) -> torch.Tensor:
    """Where a duck with heading `yaw` stands so `ball` is in front of its left (+1) or right (-1)
    foot, exactly as BallKick placed it."""
    fx, fy = FOOT_BALL[0], FOOT_BALL[1] * foot
    c, s = torch.cos(yaw), torch.sin(yaw)
    return ball - _xy(c * fx - s * fy, s * fx + c * fy)


def _push_apart(p: torch.Tensor, q: torch.Tensor, d: float, mask: torch.Tensor) -> torch.Tensor:
    """Move `p` straight away from `q` until they are `d` apart, where `mask` and too close."""
    gap = p - q
    dist = gap.norm(dim=-1, keepdim=True)
    away = torch.where(dist > 1e-6, gap / dist.clamp_min(1e-6), torch.tensor([1.0, 0.0], device=p.device))
    moved = q + away * d
    return torch.where((mask & (dist.squeeze(-1) < d)).unsqueeze(-1), moved, p)


def _clip(p: torch.Tensor) -> torch.Tensor:
    return torch.stack((p[..., 0].clamp(-X_MAX, X_MAX), p[..., 1].clamp(-Y_MAX, Y_MAX)), dim=-1)


def sample(kinds: torch.Tensor, generator: torch.Generator | None = None) -> Layout:
    """A starting layout for each env, by its situation in `kinds` (KICKOFF / SHOOT / ...)."""
    n, dev, g = kinds.shape[0], kinds.device, generator
    u = lambda lo, hi: _u(n, lo, hi, dev, g)
    foot = torch.where(torch.rand(n, device=dev, generator=g) < 0.5, 1.0, -1.0)
    red_goal = torch.tensor([FIELD.half_length, 0.0], device=dev)
    blue_goal = -red_goal

    # KICKOFF — as the original reset: ±0.15/±0.3 m and ±20° around the kickoff spots.
    k_blue, k_blue_yaw = _xy(-KICKOFF_X + u(-0.15, 0.15), u(-0.3, 0.3)), u(-0.35, 0.35)
    k_red, k_red_yaw = _xy(KICKOFF_X + u(-0.15, 0.15), u(-0.3, 0.3)), torch.pi + u(-0.35, 0.35)
    k_ball = _xy(u(-0.05, 0.05), u(-0.05, 0.05))

    # SHOOT — ball at blue's foot, blue aimed at the red goal within ±15°.
    s_ball = _xy(u(-0.2, FIELD.half_length - 0.4), u(-0.6, 0.6))
    s_blue_yaw = _heading(red_goal, s_ball) + u(-0.25, 0.25)
    s_blue = _behind_ball(s_ball, s_blue_yaw, foot)
    s_red, s_red_yaw = _xy(u(-X_MAX, -0.4), u(-Y_MAX, Y_MAX)), u(-torch.pi, torch.pi)

    # APPROACH — blue anywhere, the ball 0.5-1.5 m away in any direction.
    a_blue, a_blue_yaw = _xy(u(-X_MAX, 0.6), u(-Y_MAX, Y_MAX)), u(-torch.pi, torch.pi)
    angle, dist = u(-torch.pi, torch.pi), u(0.5, 1.5)
    offset = _xy(torch.cos(angle), torch.sin(angle)) * dist.unsqueeze(-1)
    # Aimed into a wall, clipping would pull the ball back onto blue; aim it the other way instead.
    a_ball = _clip(a_blue + offset)
    flipped = _clip(a_blue - offset)
    too_close = (a_ball - a_blue).norm(dim=-1) < (flipped - a_blue).norm(dim=-1)
    a_ball = torch.where(too_close.unsqueeze(-1) & ((a_ball - a_blue).norm(dim=-1, keepdim=True) < 0.5), flipped, a_ball)
    a_red, a_red_yaw = _xy(u(0.5, X_MAX), u(-Y_MAX, Y_MAX)), torch.pi + u(-0.5, 0.5)

    # DEFEND — red has the ball at its foot in blue's half, aimed at blue's goal; blue goal-side.
    d_ball = _xy(u(-1.0, -0.2), u(-0.6, 0.6))
    d_red_yaw = _heading(blue_goal, d_ball) + u(-0.25, 0.25)
    d_red = _behind_ball(d_ball, d_red_yaw, foot)
    d_blue, d_blue_yaw = _xy(u(-X_MAX, -0.9), u(-0.5, 0.5)), u(-0.5, 0.5)

    def pick(k, s, a, d):
        out = k.clone()
        for kind, value in ((SHOOT, s), (APPROACH, a), (DEFEND, d)):
            m = kinds == kind
            out[m] = value[m]
        return out

    blue, blue_yaw = pick(k_blue, s_blue, a_blue, d_blue), pick(k_blue_yaw, s_blue_yaw, a_blue_yaw, d_blue_yaw)
    red, red_yaw = pick(k_red, s_red, a_red, d_red), pick(k_red_yaw, s_red_yaw, a_red_yaw, d_red_yaw)
    ball = pick(k_ball, s_ball, a_ball, d_ball)

    # Keep bodies from spawning inside each other. The duck a situation puts at the ball keeps its
    # place; everyone else moves out of the way.
    every = torch.ones(n, dtype=torch.bool, device=dev)
    blue = _clip(_push_apart(blue, ball, BALL_CLEARANCE, kinds != SHOOT))
    red = _clip(_push_apart(red, ball, BALL_CLEARANCE, kinds != DEFEND))
    red = _clip(_push_apart(red, blue, DUCK_CLEARANCE, every & (kinds != DEFEND)))
    blue = _clip(_push_apart(blue, red, DUCK_CLEARANCE, kinds == DEFEND))
    return Layout(blue, blue_yaw, red, red_yaw, ball)


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), dim=-1)
