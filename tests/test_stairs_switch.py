"""The stairs switch (tasks/stairs_switch.py; docs/stairs.md, docs/archive/stairs_runlog.md §29):
the downhill test on synthetic 327-D observations, its scan layout against the stairs cfg, and the
routing — flat ground and steps up go to the stairs-up policy, a drop ahead to the stairs-down
policy — directly and through ``scripts/eval/_common.load_policy("switch:<down>+<up>")``.

Ported 2026-11-11 from the run-13 tests of the retired climb anchor, whose downhill mask this is."""
import sys
from pathlib import Path

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_stairs_up_env_cfg as su
from mjlab_microduck.tasks import microduck_terrains as mt
from mjlab_microduck.tasks import stairs_switch as sw

CFG = sw.SCAN_CFG
XS = CFG["scan_x0"] + CFG["scan_res"] * torch.arange(CFG["scan_nx"])
FLAT = torch.zeros(19, 7)
DROP = torch.where(XS[:, None] > 0.2, 0.015, 0.0).repeat(1, 7)        # ground 15 mm LOWER ahead → reads higher
STEP_UP = -DROP                                                       # ground 15 mm HIGHER ahead


def _obs(vals: torch.Tensor, known: torch.Tensor | None = None) -> torch.Tensor:
    """(1, 327) actor obs: 61 base dims, then the 19×7 scan values and known mask (x-major)."""
    known = torch.ones(19, 7) if known is None else known
    return torch.cat([torch.zeros(61), vals.flatten(), known.flatten()])[None]


def test_the_scan_cfg_matches_the_stairs_obs():
    xs, ys = microduck_mdp.scan_grid(su.SCAN_PARAMS["x_range"], su.SCAN_PARAMS["y_range"], su.SCAN_PARAMS["res"])
    assert (CFG["scan_nx"], CFG["scan_ny"]) == (len(xs), len(ys)) == (19, 7)
    assert CFG["scan_offset"] == 61                                   # after the 61-D family block
    assert CFG["scan_offset"] + 2 * CFG["scan_nx"] * CFG["scan_ny"] == 327
    assert (CFG["scan_x0"], CFG["scan_res"]) == (su.SCAN_PARAMS["x_range"][0], su.SCAN_PARAMS["res"])
    assert float(xs[0]) == pytest.approx(CFG["scan_x0"])
    assert CFG["drop_thresh"] < mt.STEP_UP_HEIGHT_RANGE[0]             # sees the lowest trained riser
    assert 0 < CFG["min_row_known"] <= CFG["scan_ny"]


def test_downhill_mask_sees_drops_ahead_only():
    assert not sw.downhill_ahead_mask(_obs(FLAT))
    assert sw.downhill_ahead_mask(_obs(DROP))
    assert not sw.downhill_ahead_mask(_obs(STEP_UP))                                # a step UP ahead
    assert not sw.downhill_ahead_mask(_obs(FLAT + 0.004 * torch.randn(19, 7).clamp(-1, 1)))   # map noise
    unseen = torch.where(XS[:, None] > 0.2, 0.0, 1.0).repeat(1, 7)
    assert not sw.downhill_ahead_mask(_obs(DROP, unseen))                           # the drop not mapped yet
    bob = FLAT + 0.02                                                               # trunk bobbing: all cells shift
    assert not sw.downhill_ahead_mask(_obs(bob))
    assert sw.downhill_ahead_mask(_obs(bob + DROP))
    batch = torch.cat([_obs(FLAT), _obs(DROP), _obs(STEP_UP)])
    assert sw.downhill_ahead_mask(batch).tolist() == [False, True, False]
    spike = FLAT.clone()
    spike[12, 3] = 0.03                                                             # one bad cell (a dropout's neighbour)
    assert not sw.downhill_ahead_mask(_obs(spike))
    g = torch.Generator().manual_seed(0)
    noisy = torch.cat([_obs(0.004 * torch.randn(19, 7, generator=g)) for _ in range(500)])  # σ 4 mm per cell
    assert sw.downhill_ahead_mask(noisy).float().mean() < 0.01
    assert sw.downhill_ahead_mask(noisy + _obs(DROP) - _obs(FLAT)).float().mean() > 0.99


def test_downhill_mask_refuses_an_obs_without_the_scan():
    with pytest.raises(AssertionError):
        sw.downhill_ahead_mask(torch.zeros(2, 61))                                  # a blind 61-D policy's obs


def _const(value: float):
    calls = []

    def policy(obs: torch.Tensor) -> torch.Tensor:
        calls.append(obs.shape[0])
        return torch.full((obs.shape[0], 14), value)

    return policy, calls


def test_switch_routes_flat_and_up_to_the_up_policy_and_a_drop_to_the_down_policy():
    down, _ = _const(-1.0)
    up, _ = _const(1.0)
    pi = sw.switch_policies(down, up)
    batch = torch.cat([_obs(FLAT), _obs(STEP_UP), _obs(DROP), _obs(DROP, torch.zeros(19, 7))])
    a = pi(batch)
    assert a.shape == (4, 14)
    assert a[:, 0].tolist() == [1.0, 1.0, -1.0, 1.0]       # flat, step up → up; drop → down; unmapped drop → up


def _checkpoint(path: Path, bias: float) -> str:
    """An rsl_rl-shaped checkpoint whose actor outputs ``bias`` everywhere (zero weights)."""
    sizes = [327, 512, 256, 128, 14]
    sd = {"obs_normalizer._mean": torch.zeros(1, 327), "obs_normalizer._std": torch.ones(1, 327)}
    for k, i in enumerate((0, 2, 4, 6)):
        sd[f"mlp.{i}.weight"] = torch.zeros(sizes[k + 1], sizes[k])
        sd[f"mlp.{i}.bias"] = torch.full((sizes[k + 1],), bias if i == 6 else 0.0)
    torch.save({"actor_state_dict": sd}, path)
    return str(path)


def test_load_policy_builds_the_switch_from_two_checkpoints(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))
    from _common import load_policy

    down = _checkpoint(tmp_path / "down.pt", -1.0)
    up = _checkpoint(tmp_path / "up.pt", 1.0)
    pi = load_policy(f"switch:{down}+{up}")
    batch = torch.cat([_obs(FLAT), _obs(STEP_UP), _obs(DROP)])
    assert pi(batch)[:, 0].tolist() == [1.0, 1.0, -1.0]
    assert load_policy(up)(batch)[:, 0].tolist() == [1.0, 1.0, 1.0]            # a plain checkpoint: no switch
