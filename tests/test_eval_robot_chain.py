"""The eval scripts' emulation of the real robot's action pipeline (scripts/eval/_common.py)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "eval"))
from _common import OBS_LAST_ACTION, RobotChain, mirror_policy  # noqa: E402


def _chain(vin):
    return RobotChain(None, len(vin), "cpu", vin=torch.tensor(vin).reshape(-1, 1))


def test_voltage_scaled_gain_with_clamp():
    c = _chain([7.4, 8.4, 5.0, 10.0])
    expected = 0.9 * 7.4 / torch.tensor([7.4, 8.4, 6.0, 9.5])              # V clamped to 6.0..9.5
    assert torch.allclose(c.gain.flatten(), expected)


def test_per_joint_low_pass_from_home():
    c = _chain([7.4])
    a = torch.ones(1, 14)
    y1 = c(a)
    legs, head = list(RobotChain.LEGS), list(RobotChain.HEAD)
    assert torch.allclose(y1[0, legs], torch.full((10,), 0.7 * 0.9))       # first step from HOME (0)
    assert torch.allclose(y1[0, head], torch.full((4,), 0.5 * 0.9))
    y2 = c(a)
    assert torch.allclose(y2[0, legs], torch.full((10,), 0.9 * (1 - 0.3 ** 2)))
    for _ in range(50):
        y = c(a)
    assert torch.allclose(y, torch.full((1, 14), 0.9), atol=1e-6)          # settles at the scaled target


def test_reset_only_the_given_envs():
    c = _chain([7.4, 7.4])
    c(torch.ones(2, 14))
    c.reset(torch.tensor([1]))
    assert (c.y[1] == 0).all() and (c.last_raw[1] == 0).all()
    assert (c.y[0] != 0).all() and (c.last_raw[0] == 1).all()


def test_observation_keeps_the_raw_action():
    c = _chain([8.0])
    raw = torch.linspace(-1, 1, 14).reshape(1, 14)
    sent = c(raw)
    assert not torch.allclose(sent, raw)                                   # the env gets the filtered offset
    obs = torch.zeros(1, 61)
    obs[:, OBS_LAST_ACTION] = sent                                          # what the env would report
    fixed = c.fix_obs(obs)
    assert torch.equal(fixed[:, OBS_LAST_ACTION], raw)
    assert torch.equal(obs[:, OBS_LAST_ACTION], sent)                       # input not modified


def test_mirror_policy_is_an_involution_on_a_symmetric_policy():
    from mjlab_microduck.tasks.symmetry import _get_tensors
    obs_perm, obs_sign, act_perm, act_sign = _get_tensors(torch.device("cpu"))
    w = torch.randn(61, 14)

    def pol(o):
        return o @ w

    m = mirror_policy(pol)
    o = torch.randn(5, 61)
    # π_m(M_o o) = M_a π(o): mirroring the input of the mirrored policy gives the mirrored action
    assert torch.allclose(m(o[:, obs_perm] * obs_sign), pol(o)[:, act_perm] * act_sign, atol=1e-5)
    assert torch.allclose(mirror_policy(m)(o), pol(o), atol=1e-5)          # mirror twice = original


@pytest.mark.parametrize("v", [6.5, 8.2])
def test_effective_scale_at_pack_extremes(v):
    # robotd docs: a full 8.4 V pack runs ≈ 0.79, 7.4 V runs 0.9
    assert _chain([v]).gain.item() == pytest.approx(0.9 * 7.4 / v)
