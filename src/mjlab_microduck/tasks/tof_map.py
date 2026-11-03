"""A body-frame height map built ONLY from the head ToF — what the robot can actually have.

Run 8 showed that terrain memory is the whole gap between kicking the riser and stepping on it,
but its actor read the simulator's exact terrain (an oracle). This module builds the same 19×7
scan from what a real duck has: ToF frames at 15 Hz (noise, dropouts, 0.12 m minimum range), the
sensor pose from kinematics, and the trunk pose from odometry that drifts.

Per env, a grid anchored at the episode's start (odometry frame, x along the start heading)
stores the running mean of the ground heights measured in each cell (``fuse="mean"``; or the
latest one) — every new ToF frame writes its hits into it, optionally spread to the lateral
neighbours (``splat_y``) to fill the gaps between the sensor's zone columns.
A query projects the scan points from the trunk's ESTIMATED yaw frame into the grid; a cell never
seen is unknown (NaN). The robot only knows its own pose up to the drift, and both the map and
the query use the same estimate — so drift shows up exactly as it would on the robot: as an
offset between where a step was seen and where the robot thinks it is now.

Drift model (zero-centred, per episode): a scale error on the travelled displacement, a yaw
error growing with distance, and a small random walk; z drifts slowly. Roll/pitch come from the
IMU and are taken as exact.
"""
from __future__ import annotations

import math

import torch


def _yaw(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _rot2(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * x - s * y, s * x + c * y


class TofHeightMap:
    def __init__(self, num_envs: int, device, res: float = 0.02, x_range=(-0.6, 3.0), y_range=(-0.9, 0.9),
                 max_range: float = 1.2, min_range: float = 0.12, hz: float = 15.0,
                 noise_rel: float = 0.01, noise_abs: float = 0.004, dropout: float = 0.05,
                 odom_scale: float = 0.03, odom_yaw_per_m_deg: float = 2.0, odom_walk: float = 0.01,
                 odom_z_per_m: float = 0.01, splat: int = 1, splat_y: int = 0, fuse: str = "mean"):
        self.N, self.device, self.res = num_envs, device, res
        self.x0, self.y0 = x_range[0], y_range[0]
        self.nx = int(round((x_range[1] - x_range[0]) / res)) + 1
        self.ny = int(round((y_range[1] - y_range[0]) / res)) + 1
        self.max_range, self.min_range, self.hz = max_range, min_range, hz
        self.noise_rel, self.noise_abs, self.dropout = noise_rel, noise_abs, dropout
        self.odom_scale, self.odom_yaw = odom_scale, math.radians(odom_yaw_per_m_deg)
        self.odom_walk, self.odom_z = odom_walk, odom_z_per_m
        self.splat, self.splat_y = splat, splat_y
        assert fuse in ("mean", "latest")
        self.fuse = fuse
        n = num_envs
        self.grid = torch.full((n, self.nx, self.ny), float("nan"), device=device)
        self.sum = torch.zeros((n, self.nx, self.ny), device=device)      # fuse="mean": per-cell running mean
        self.cnt = torch.zeros((n, self.nx, self.ny), device=device)
        self.anchor = torch.zeros(n, 3, device=device)          # odom pose at reset (x, y, z)
        self.anchor_yaw = torch.zeros(n, device=device)
        self.err = torch.zeros(n, 4, device=device)             # odometry error: dx, dy, dz, dyaw
        self.scale = torch.zeros(n, device=device)              # per-episode displacement scale error
        self.yaw_rate = torch.zeros(n, device=device)           # per-episode yaw error per metre
        self.z_rate = torch.zeros(n, device=device)
        self.last_xy = torch.zeros(n, 2, device=device)
        self.phase = torch.rand(n, device=device)

    # ── episode bookkeeping ──────────────────────────────────────────────────
    def reset(self, env_ids, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        k = len(env_ids) if not isinstance(env_ids, slice) else self.N
        dev = self.device
        self.grid[env_ids] = float("nan")
        self.sum[env_ids] = 0.0
        self.cnt[env_ids] = 0.0
        self.err[env_ids] = 0.0
        self.scale[env_ids] = (torch.rand(k, device=dev) * 2 - 1) * self.odom_scale
        self.yaw_rate[env_ids] = (torch.rand(k, device=dev) * 2 - 1) * self.odom_yaw
        self.z_rate[env_ids] = (torch.rand(k, device=dev) * 2 - 1) * self.odom_z
        self.anchor[env_ids] = pos_w[env_ids]
        self.anchor_yaw[env_ids] = _yaw(quat_w[env_ids])
        self.last_xy[env_ids] = pos_w[env_ids, :2]
        self.phase[env_ids] = torch.rand(k, device=dev)

    def advance_odometry(self, pos_w: torch.Tensor) -> None:
        """Grow the odometry error with the distance travelled since the last call."""
        d = pos_w[:, :2] - self.last_xy
        dist = d.norm(dim=1)
        # scale error on the displacement + a random walk of odom_walk metres per √metre travelled
        self.err[:, 0:2] += d * self.scale[:, None] + torch.randn_like(d) * self.odom_walk * torch.sqrt(dist)[:, None]
        self.err[:, 2] += dist * self.z_rate
        self.err[:, 3] += dist * self.yaw_rate
        self.last_xy = pos_w[:, :2].clone()

    # ── frames: world → (estimated) odometry → anchored grid ────────────────
    def _world_to_est(self, p_w: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        """A world point as the robot places it: rotated about the trunk by the yaw error and
        offset by the position error (the robot's pose estimate is the true pose + error)."""
        rel = p_w - pos_w[:, None, :]
        rx, ry = _rot2(rel[..., 0], rel[..., 1], self.err[:, None, 3])
        est_trunk = pos_w + self.err[:, :3]
        return torch.stack([rx, ry, rel[..., 2]], dim=-1) + est_trunk[:, None, :]

    def _est_to_cell(self, p_est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = p_est[..., :2] - self.anchor[:, None, :2]
        ax, ay = _rot2(rel[..., 0], rel[..., 1], -self.anchor_yaw[:, None])
        i = torch.round((ax - self.x0) / self.res).long()
        j = torch.round((ay - self.y0) / self.res).long()
        inside = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return i, j, inside

    def integrate(self, sensor_pos_w: torch.Tensor, hit_pos_w: torch.Tensor, dist: torch.Tensor,
                  pos_w: torch.Tensor, step_dt: float) -> torch.Tensor:
        """Advance the 15 Hz clock; for envs with a new frame, write its noisy hits into the grid.
        Returns the (N,) mask of envs that measured this step."""
        self.phase += step_dt * self.hz
        new = self.phase >= 1.0
        if not bool(new.any()):
            return new
        self.phase[new] -= 1.0
        valid = (dist >= self.min_range) & (dist <= self.max_range)
        if self.dropout > 0:
            valid &= torch.rand_like(dist) >= self.dropout
        valid &= new[:, None]
        d = dist.clamp(min=1e-6)
        noisy = dist + torch.randn_like(dist) * (self.noise_rel * dist + self.noise_abs)
        direction = (hit_pos_w - sensor_pos_w[:, None, :]) / d[..., None]
        p_w = sensor_pos_w[:, None, :] + direction * noisy[..., None]
        p_est = self._world_to_est(p_w, pos_w)
        h = p_est[..., 2] - self.anchor[:, None, 2]
        i, j, inside = self._est_to_cell(p_est)
        ok = valid & inside
        env = torch.arange(self.N, device=self.device)[:, None].expand_as(i)
        # splat: square (splat − 1 cells around) and/or lateral only (splat_y cells to each side,
        # which fills the gaps between the ToF's zone columns without blurring an edge along x)
        offs = {(di, dj) for di in range(-self.splat + 1, self.splat) for dj in range(-self.splat + 1, self.splat)}
        offs |= {(0, dj) for dj in range(-self.splat_y, self.splat_y + 1)}
        for di, dj in sorted(offs):
            ii, jj = (i + di).clamp(0, self.nx - 1), (j + dj).clamp(0, self.ny - 1)
            if self.fuse == "latest":
                self.grid[env[ok], ii[ok], jj[ok]] = h[ok]
            else:
                idx = (env[ok], ii[ok], jj[ok])
                self.sum.index_put_(idx, h[ok], accumulate=True)
                self.cnt.index_put_(idx, torch.ones_like(h[ok]), accumulate=True)
        if self.fuse == "mean":
            self.grid = torch.where(self.cnt > 0, self.sum / self.cnt.clamp(min=1), torch.full_like(self.sum, float("nan")))
        return new

    def scan(self, offsets_b: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor,
             nominal: float) -> tuple[torch.Tensor, torch.Tensor]:
        """The 19×7-style scan from the map: ``est_trunk_z − h − nominal`` at ``offsets_b`` (P, 2)
        in the trunk's ESTIMATED yaw frame. Returns (values (N, P) with NaN where unknown, known)."""
        yaw_est = _yaw(quat_w) + self.err[:, 3]
        ox, oy = _rot2(offsets_b[None, :, 0], offsets_b[None, :, 1], yaw_est[:, None])
        est_trunk = pos_w + self.err[:, :3]
        p_est = torch.stack([est_trunk[:, None, 0] + ox, est_trunk[:, None, 1] + oy,
                             torch.zeros_like(ox)], dim=-1)
        i, j, inside = self._est_to_cell(p_est)
        env = torch.arange(self.N, device=self.device)[:, None].expand_as(i)
        h = self.grid[env, i.clamp(0, self.nx - 1), j.clamp(0, self.ny - 1)]
        h = torch.where(inside, h, torch.full_like(h, float("nan")))
        z_rel = est_trunk[:, 2:3] - self.anchor[:, 2:3]
        values = z_rel - h - nominal
        return values, ~torch.isnan(values)
