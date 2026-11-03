"""The ToF-built height map: with perfect sensing and odometry it reproduces the exact terrain
where the sensor looked, it knows nothing it never saw, and drift moves what it remembers."""
import math

import pytest
import torch

from mjlab_microduck.tasks.tof_map import TofHeightMap

STEP_X, STEP_H = 0.5, 0.03


def _terrain(x):
    return torch.where(x >= STEP_X, torch.full_like(x, STEP_H), torch.zeros_like(x))


def _frame(n=1, xs=None, ys=None, sensor=(0.0, 0.0, 0.24)):
    """Hits of a fake sensor at ``sensor`` on the step terrain, at the given ground points."""
    xs = torch.linspace(0.15, 0.9, 40) if xs is None else xs
    ys = torch.linspace(-0.12, 0.12, 9) if ys is None else ys
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    hits = torch.stack([gx.flatten(), gy.flatten(), _terrain(gx.flatten())], dim=1)[None].repeat(n, 1, 1)
    spos = torch.tensor([sensor]).repeat(n, 1)
    dist = (hits - spos[:, None, :]).norm(dim=-1)
    return spos, hits, dist


def _perfect(n=1, **kw):
    return TofHeightMap(n, "cpu", noise_rel=0.0, noise_abs=0.0, dropout=0.0, odom_scale=0.0,
                        odom_yaw_per_m_deg=0.0, odom_walk=0.0, odom_z_per_m=0.0, max_range=1.5, **kw)


def _pose(x=0.0, z=0.115, n=1):
    return torch.tensor([[x, 0.0, z]]).repeat(n, 1), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)


def test_unknown_before_any_frame():
    m = _perfect()
    pos, quat = _pose()
    m.reset(torch.tensor([0]), pos, quat)
    vals, known = m.scan(torch.tensor([[0.2, 0.0], [0.6, 0.0]]), pos, quat, nominal=0.115)
    assert not known.any() and torch.isnan(vals).all()


def test_perfect_map_reproduces_the_terrain_where_it_looked():
    m = _perfect()
    pos, quat = _pose()
    m.reset(torch.tensor([0]), pos, quat)
    spos, hits, dist = _frame()
    m.phase[:] = 0.99
    assert m.integrate(spos, hits, dist, pos, step_dt=0.02).all()
    offs = torch.tensor([[0.3, 0.0], [0.45, 0.06], [0.55, 0.0], [0.8, -0.06]])
    vals, known = m.scan(offs, pos, quat, nominal=0.115)
    assert known.all()
    expected = torch.tensor([0.0, 0.0, -STEP_H, -STEP_H])          # trunk at nominal height
    assert vals[0].tolist() == pytest.approx(expected.tolist(), abs=1e-6)


def test_memory_after_walking_on():
    """Seen from x = 0, the step stays in the map when the robot has walked to 0.4 m and the
    sensor no longer sees it (no new frames): the point of having a map."""
    m = _perfect()
    pos, quat = _pose()
    m.reset(torch.tensor([0]), pos, quat)
    spos, hits, dist = _frame()
    m.phase[:] = 0.99
    m.integrate(spos, hits, dist, pos, step_dt=0.02)
    pos2, _ = _pose(x=0.4)
    m.advance_odometry(pos2)
    vals, known = m.scan(torch.tensor([[0.05, 0.0], [0.15, 0.0]]), pos2, quat, nominal=0.115)
    assert known.all() and vals[0].tolist() == pytest.approx([0.0, -STEP_H], abs=1e-6)   # edge 0.1 m ahead


def test_odometry_scale_error_shifts_the_remembered_edge():
    m = TofHeightMap(1, "cpu", noise_rel=0.0, noise_abs=0.0, dropout=0.0, odom_walk=0.0, odom_z_per_m=0.0,
                     odom_yaw_per_m_deg=0.0, max_range=1.5)
    pos, quat = _pose()
    m.reset(torch.tensor([0]), pos, quat)
    m.scale[:] = 0.05                                                # odometry over-reads distance by 5 %
    spos, hits, dist = _frame()
    m.phase[:] = 0.99
    m.integrate(spos, hits, dist, pos, step_dt=0.02)
    pos2, _ = _pose(x=0.4)
    m.advance_odometry(pos2)                                         # believes it walked 0.42 m
    assert m.err[0, 0].item() == pytest.approx(0.02, abs=1e-6)
    # true edge is 0.10 m ahead; the robot believes it is 0.08 m ahead
    vals, _ = m.scan(torch.tensor([[0.07, 0.0], [0.09, 0.0]]), pos2, quat, nominal=0.115)
    assert vals[0].tolist() == pytest.approx([0.0, -STEP_H], abs=1e-6)


def test_frames_arrive_at_15_hz():
    m = _perfect()
    pos, quat = _pose()
    m.reset(torch.tensor([0]), pos, quat)
    m.phase[:] = 0.0
    spos, hits, dist = _frame()
    ticks = sum(int(m.integrate(spos, hits, dist, pos, step_dt=0.02)[0]) for _ in range(150))   # 3 s
    assert ticks == 45
