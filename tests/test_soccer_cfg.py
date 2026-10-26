"""Soccer 1v1: the cfg's invariants and the referee's geometry, on CPU without building an env."""

import math

import pytest
import torch

from mjlab_microduck.soccer import mdp as soccer_mdp
from mjlab_microduck.soccer.field import FIELD, FIELD_XML, field_xml
from mjlab_microduck.soccer.mdp import BALL_RADIUS, OBS_DIM, goal_flags, side_obs
from mjlab_microduck.tasks.microduck_soccer_env_cfg import (
    DECISION_DT,
    GOAL_WEIGHT,
    make_microduck_soccer_env_cfg,
)


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_soccer_env_cfg()


def test_builds_train_and_play():
    make_microduck_soccer_env_cfg()
    play = make_microduck_soccer_env_cfg(play=True)
    assert play.scene.num_envs == 1
    assert play.rewards["ball_approach"].weight == 0.0 and not play.curriculum


def test_entities_and_actions(cfg):
    assert set(cfg.scene.entities) == {"robot_blue", "robot_red", "ball"}
    assert cfg.actions["blue_walk"].entity_name == "robot_blue" and cfg.actions["blue_walk"].source == "policy"
    assert cfg.actions["red_walk"].entity_name == "robot_red" and cfg.actions["red_walk"].source == "opponent"
    # The walker runs at the rate it was trained at, under a 10 Hz decision.
    assert math.isclose(cfg.sim.mujoco.timestep * cfg.decimation, DECISION_DT)
    # World geometry sits at every world's origin, so the entities must too.
    assert cfg.scene.env_spacing == 0.0


def test_reward_signs(cfg):
    r = cfg.rewards
    assert r["goal"].weight == GOAL_WEIGHT > 0
    assert r["conceded"].weight == -GOAL_WEIGHT
    # Cost functions return >= 0, so their weights must be negative (AGENTS.md sign convention).
    for cost in ("existential", "fallen"):
        assert r[cost].weight < 0, cost
    # Shaping that pays for doing the right thing is positive, and none of it may pay a duck for
    # standing still (the kick run learned to hover when one did).
    for shaping in ("ball_approach", "ball_to_goal", "approach_facing", "drive_through"):
        assert r[shaping].weight > 0, shaping
    # Rewards a duck could collect without moving the ball (both were farmed): gone.
    assert "face_ball" not in r and "kick_aim" not in r
    # A full match of existential penalty is worth one conceded goal, no more.
    assert math.isclose(r["existential"].weight * cfg.episode_length_s, -1.0)


def test_shaping_anneals_to_zero(cfg):
    for name in ("ball_approach_weight", "approach_facing_weight", "drive_through_weight"):
        stages = cfg.curriculum[name].params["weight_stages"]
        assert stages[0]["weight"] > 0 and stages[-1]["weight"] == 0.0
    # Ball-to-goal measures an outcome and keeps a floor: every run that annealed it to zero drew.
    stages = cfg.curriculum["ball_to_goal_weight"].params["weight_stages"]
    assert math.isclose(stages[-1]["weight"], 0.25 * stages[0]["weight"])


def test_field_xml_is_current():
    """duck-sim plays on the XML; training on `add_field`. They must be the same pitch."""
    assert FIELD_XML.read_text() == field_xml(), "run: python -m mjlab_microduck.soccer.field"


def _state(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g) * 2 - 1
    yaw = rnd(n) * math.pi

    def quat(y):
        return torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.zeros_like(y), torch.sin(y / 2)), -1)

    return dict(
        a_pos=rnd(n, 3), a_quat=quat(yaw), a_vel=rnd(n, 3), a_rate=rnd(n), a_gz=-torch.rand(n, generator=g),
        b_pos=rnd(n, 3), b_quat=quat(yaw.flip(0)), b_vel=rnd(n, 3), b_rate=rnd(n), b_gz=-torch.rand(n, generator=g),
        ball_pos=rnd(n, 3), ball_vel=rnd(n, 3), t=torch.rand(n, generator=g),
        a_kick=torch.rand(n, generator=g), b_kick=torch.rand(n, generator=g),
    )


def _turned(s):
    """The same world rotated 180 degrees about the centre spot."""
    rot = lambda v: v * torch.tensor([-1.0, -1.0, 1.0])

    def qmul(q):  # (yaw pi) * q
        w, x, y, z = q.unbind(-1)
        return torch.stack((-z, -y, x, w), -1)

    return {**s, **{k: rot(s[k]) for k in ("a_pos", "a_vel", "b_pos", "b_vel", "ball_pos", "ball_vel")},
            "a_quat": qmul(s["a_quat"]), "b_quat": qmul(s["b_quat"])}


def _obs(s, team, own="a"):
    opp = "b" if own == "a" else "a"
    return side_obs(s[f"{own}_pos"], s[f"{own}_quat"], s[f"{own}_vel"], s[f"{own}_rate"], s[f"{own}_gz"],
                    s[f"{opp}_pos"], s[f"{opp}_quat"], s[f"{opp}_vel"], s[f"{opp}_gz"],
                    s["ball_pos"], s["ball_vel"], s["t"], s[f"{own}_kick"], team)


def test_mirroring_makes_one_policy_play_both_ends():
    """Red seeing the world turned 180 degrees must see exactly what blue sees in the original:
    that is what lets a snapshot of blue drive red."""
    s = _state()
    blue_view = _obs(s, team=+1)
    red_view_of_turned_world = _obs(_turned(s), team=-1)
    assert blue_view.shape[-1] == OBS_DIM
    torch.testing.assert_close(blue_view, red_view_of_turned_world, atol=1e-5, rtol=1e-5)


def test_goal_flags():
    hl, gw, gh = FIELD.half_length, FIELD.goal_width / 2, FIELD.goal_height
    past = hl + BALL_RADIUS + 0.01
    balls = torch.tensor([
        [past, 0.0, 0.035],  # in red's goal: blue scores
        [-past, 0.1, 0.035],  # in blue's goal: red scores
        [hl, 0.0, 0.035],  # on the line, not over it
        [past, gw + 0.05, 0.035],  # wide of the post
        [past, 0.0, gh + 0.05],  # over the bar
    ])
    blue, red = goal_flags(balls)
    assert blue.tolist() == [True, False, False, False, False]
    assert red.tolist() == [False, True, False, False, False]


def test_twist_mapping_caps_reverse():
    """0 stands still, +1 is velstand's full forward, -1 only a slow reverse — torch and numpy alike."""
    import numpy as np

    from mjlab_microduck.soccer.frozen_walker import TWIST_LIMITS, action_to_twist

    actions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [-3.0, 0.5, 2.0]])
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.3, 1.0], [-0.1, -0.3, -1.0], [-0.1, 0.15, 1.0]])
    torch.testing.assert_close(action_to_twist(actions), expected)
    np.testing.assert_allclose(action_to_twist(actions.numpy()), expected.numpy(), atol=1e-6)
    # Never beyond what velstand was trained on.
    assert (action_to_twist(actions).abs() <= torch.tensor(TWIST_LIMITS)).all()


def test_kick_zone():
    """A kick fires only with the ball where BallKick trained it: just ahead of a foot."""
    from mjlab_microduck.soccer.mdp import KICK_LEFT, KICK_NONE, KICK_RIGHT, kick_side

    balls = torch.tensor([
        [0.09, 0.042],  # BallKick's left-foot placement
        [0.09, -0.042],  # and the right
        [0.30, 0.0],  # too far ahead
        [-0.09, 0.04],  # behind the duck
        [0.09, 0.2],  # off to the side
    ])
    assert kick_side(balls).tolist() == [KICK_LEFT, KICK_RIGHT, KICK_NONE, KICK_NONE, KICK_NONE]


def test_ball_out():
    inside = torch.tensor([[0.0, 0.0, 0.035], [FIELD.half_length + 0.1, 0.0, 0.035]])
    outside = torch.tensor([[0.0, FIELD.half_width + 0.3, 0.1], [0.0, 0.0, -1.0]])
    assert not soccer_mdp.ball_out(inside).any()
    assert soccer_mdp.ball_out(outside).all()
