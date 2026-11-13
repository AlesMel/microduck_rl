"""The stairs switch: stairs up and stairs down combined into one policy by a hard switch.

Both policies read the same 327-D actor observation (microduck_stairs_up_env_cfg: the 61-D family
block, then the ToF-map scan). Every control step the scan decides which one acts: the stairs-down
policy while the map shows a drop ahead, the stairs-up policy otherwise (flat ground, steps up,
treads, a drop not mapped yet). No training: the two networks run as trained. Measured in
``docs/archive/stairs_runlog.md`` §29 ("The hard switch"); overview in ``docs/stairs.md``.

The test reads only the observation. The scan values are ``est_trunk_z − h − nominal``
(tof_map.tof_map_scan): ground LOWER than under the robot reads LARGER. A frame is "downhill" when
the mean of the known cells of any scan row ahead (x > ``under_x_max``, at least ``min_row_known``
of its cells known) reads more than ``drop_thresh`` above the mean of the known cells under the
robot. Row means, not single cells: with per-cell map noise the maximum over ~100 cells ahead
crossed 8 mm on 29 % of flat frames; the trained risers span the whole tile width.

Eval scripts load the switch as ``--checkpoint switch:<down.pt>+<up.pt>``
(``scripts/eval/_common.py:load_policy``).
"""

from __future__ import annotations

from typing import Callable

import torch

# The scan block of the actor obs: 61 base dims, then [values (P), known (P)] on an nx × ny grid,
# x-major (index = ix * ny + iy), x = x0 + ix * res (microduck_stairs_up_env_cfg.SCAN_PARAMS).
SCAN_CFG = {
    "scan_offset": 61,
    "scan_nx": 19,
    "scan_ny": 7,
    "scan_x0": -0.09,
    "scan_res": 0.03,
    "under_x_max": 0.03,          # cells with x ≤ this are "under the robot" (the reference level)
    "drop_thresh": 0.008,         # m; the lowest riser trained is 15 mm
    "min_row_known": 3,           # of the scan_ny cells of a row
}

Policy = Callable[[torch.Tensor], torch.Tensor]


def _rows_ahead(flat: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, rows ahead) scan row mean minus the mean under the robot (> 0: ground lower ahead), and
    whether the row has enough known cells to count."""
    nx, ny, off = cfg["scan_nx"], cfg["scan_ny"], cfg["scan_offset"]
    P = nx * ny
    assert flat.shape[1] >= off + 2 * P, f"actor obs {flat.shape[1]}-D has no {nx}x{ny} scan at {off}"
    vals = flat[:, off:off + P].reshape(-1, nx, ny)
    known = flat[:, off + P:off + 2 * P].reshape(-1, nx, ny) > 0.5
    xs = cfg["scan_x0"] + cfg["scan_res"] * torch.arange(nx, device=flat.device)
    k = int((xs <= cfg["under_x_max"] + 1e-6).sum())
    ku = known[:, :k].float()
    ref = (vals[:, :k] * ku).sum(dim=(1, 2)) / ku.sum(dim=(1, 2)).clamp(min=1.0)       # 0 when nothing known
    ka = known[:, k:].float()
    n_row = ka.sum(dim=2)                                                               # (B, rows ahead)
    row = (vals[:, k:] * ka).sum(dim=2) / n_row.clamp(min=1.0)
    return row - ref[:, None], n_row >= cfg.get("min_row_known", 3)


def downhill_ahead_mask(flat: torch.Tensor, cfg: dict = SCAN_CFG) -> torch.Tensor:
    """(B,) bool: the scan in the actor obs shows ground lower than under the robot, ahead of it."""
    d, ok = _rows_ahead(flat, cfg)
    return (ok & (d > cfg["drop_thresh"])).any(dim=1)


def switch_policies(down: Policy, up: Policy, cfg: dict = SCAN_CFG) -> Policy:
    """obs (B, 327) → action (B, 14): ``down`` where :func:`downhill_ahead_mask`, ``up`` elsewhere."""

    @torch.no_grad()
    def switched(obs: torch.Tensor) -> torch.Tensor:
        return torch.where(downhill_ahead_mask(obs, cfg)[:, None], down(obs), up(obs))

    return switched
