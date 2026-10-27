"""The pitch: walls, two goals and the markings, as geoms on the world body.

One source for both worlds a match is played in. Training adds these geoms through
`SceneCfg.spec_fn`; duck-sim loads `robot/microduck/soccer_field.xml`, which this module writes
(`python -m mjlab_microduck.soccer.field`) and `tests/test_soccer_cfg.py` checks is current — a
goal that is 2 cm wider in duck-sim than in training is a different game, and nobody would notice.

Sized for a 25 cm duck that walks at 0.4 m/s: 3 x 2 m is about twelve body lengths end to end, so a
run from the centre spot to a goal takes ~4 s. The walls are there for the reason Unity's are — a
ball that cannot leave the pitch removes throw-ins from the problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco


@dataclass(frozen=True)
class Field:
    length: float = 3.0  # goal line to goal line (x)
    width: float = 2.0  # touchline to touchline (y)
    wall_height: float = 0.15
    wall_thickness: float = 0.05
    # Between the posts (y). 0.6 m was a goal one duck could shut: standing on the shot line ~0.55 m
    # ahead of the ball it shadows +-10 degrees, and from the centre spot that goal spans +-11. At
    # 1.2 m the oracle's goalward kicks past the trained defender scored 57% (12% at 0.6 m).
    goal_width: float = 1.2
    goal_height: float = 0.25  # crossbar, from the floor
    goal_depth: float = 0.2  # behind the goal line (x)
    post_radius: float = 0.015
    line_width: float = 0.02
    corner_cut: float = 0.15  # each corner cut off at 45°, this far along both walls

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2


FIELD = Field()

# Each duck kicks off this far into its own half, facing the other goal — in training and in
# duck-sim alike.
KICKOFF_X = 0.6
# Kickoff possession, as in football: the two teams take turns to kick off. The kicker stands
# `KICKER_BALL_DIST` behind the ball, facing the goal it attacks; the other duck starts back at
# `DEFENDER_X` in its own half. Symmetric kickoffs had both ducks reach the ball together, and a
# contested ball can only be shoved: 23-25 of 32 oracle-vs-oracle points were draws. From here
# the kicker has ~4 s for one clean stop-then-kick shot before the defender can reach the ball.
KICKER_BALL_DIST = 0.30
DEFENDER_X = 0.9

FIELD_XML = Path(__file__).resolve().parents[1] / "robot" / "microduck" / "soccer_field.xml"

_WALL_RGBA = (0.85, 0.85, 0.9, 1.0)
_NET_RGBA = (1.0, 1.0, 1.0, 0.35)
_LINE_RGBA = (1.0, 1.0, 1.0, 1.0)
# Blue defends -x and attacks +x; the goal a team defends wears its colour.
_GOAL_RGBA = {-1: (0.2, 0.4, 1.0, 1.0), +1: (1.0, 0.25, 0.2, 1.0)}


def _box(body: mujoco.MjsBody, name: str, center, half_size, rgba, collide: bool = True, yaw: float = 0.0) -> None:
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(center),
        quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        size=list(half_size),
        rgba=list(rgba),
        contype=1 if collide else 0,
        conaffinity=1 if collide else 0,
        group=0 if collide else 1,
    )


def add_field(spec: mujoco.MjSpec, field: Field = FIELD) -> None:
    """Add the pitch to `spec`'s world body. Usable as (part of) a `SceneCfg.spec_fn`."""
    wb = spec.worldbody
    hl, hw, t, h = field.half_length, field.half_width, field.wall_thickness, field.wall_height
    gw, gh, gd = field.goal_width / 2, field.goal_height, field.goal_depth

    # Touchlines run the full length, past the goal boxes, so the corners are closed.
    for side, sign in (("left", 1), ("right", -1)):
        _box(wb, f"wall_{side}", (0, sign * (hw + t / 2), h / 2), (hl + gd + t, t / 2, h / 2), _WALL_RGBA)

    for end in (-1, 1):
        tag = "blue" if end < 0 else "red"
        x_line = end * (hl + t / 2)
        # End wall in two pieces, leaving the goal mouth open.
        seg = (hw - gw) / 2
        for part, sign in (("left", 1), ("right", -1)):
            _box(wb, f"wall_end_{tag}_{part}", (x_line, sign * (gw + seg), h / 2), (t / 2, seg, h / 2), _WALL_RGBA)
        # The goal: posts and bar in the team colour, a net box behind the line.
        x_box = end * (hl + gd / 2)
        x_back = end * (hl + gd + t / 2)
        r = field.post_radius
        for part, sign in (("left", 1), ("right", -1)):
            _box(wb, f"goal_{tag}_post_{part}", (end * hl, sign * (gw + r), gh / 2), (r, r, gh / 2), _GOAL_RGBA[end])
            _box(wb, f"goal_{tag}_net_{part}", (x_box, sign * (gw + r), gh / 2), (gd / 2, r / 2, gh / 2), _NET_RGBA)
        _box(wb, f"goal_{tag}_bar", (end * hl, 0, gh + r), (r, gw + 2 * r, r), _GOAL_RGBA[end])
        _box(wb, f"goal_{tag}_net_back", (x_back, 0, gh / 2), (t / 2, gw + 2 * r, gh / 2), _NET_RGBA)
        _box(wb, f"goal_{tag}_net_top", (x_box, 0, gh + r), (gd / 2, gw + 2 * r, r / 2), _NET_RGBA)
        # Goal line, drawn only: the referee decides a goal from the ball's position, not a contact.
        _box(wb, f"line_goal_{tag}", (end * hl, 0, 0.001), (field.line_width / 2, hw, 0.001), _LINE_RGBA, collide=False)

    # Corners cut at 45°: a ball run along a wall into a square corner stays there, and the drive run
    # had it within 20 cm of a touchline 41% of the time. A diagonal sends it back toward the middle.
    c = field.corner_cut
    for sx in (-1, 1):
        for sy in (-1, 1):
            # Inner face on the line between (sx(hl - c), sy hw) and (sx hl, sy(hw - c)); the box sits
            # just outside it, its long side along that line and long enough to reach into both walls.
            out = t / 2 / math.sqrt(2)
            center = (sx * (hl - c / 2 + out), sy * (hw - c / 2 + out), h / 2)
            half = (c / math.sqrt(2) + t, t / 2, h / 2)
            tag = f"{'red' if sx > 0 else 'blue'}_{'left' if sy > 0 else 'right'}"
            _box(wb, f"corner_{tag}", center, half, _WALL_RGBA, yaw=math.atan2(-sy, sx))

    _box(wb, "line_half", (0, 0, 0.001), (field.line_width / 2, hw, 0.001), _LINE_RGBA, collide=False)


def field_spec(field: Field = FIELD) -> mujoco.MjSpec:
    """A model holding only the pitch, for `<include>` in a duck-sim scene."""
    spec = mujoco.MjSpec()
    spec.modelname = "soccer_field"
    add_field(spec, field)
    return spec


def field_xml(field: Field = FIELD) -> str:
    return field_spec(field).to_xml()


if __name__ == "__main__":
    FIELD_XML.write_text(field_xml())
    print(f"wrote {FIELD_XML}")
