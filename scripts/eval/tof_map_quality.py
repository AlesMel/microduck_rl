"""How close does a height map built ONLY from the head ToF get to the oracle map of run 8?

Drives the run 9 ToF policy (head pitched down, walking into the step-up tiles) and, alongside,
builds three ToF height maps (tasks/tof_map.py) from the sensor's frames:

  ideal    exact ranges, no dropouts, perfect odometry  → what the 8×8 geometry alone allows
  sensor   ToF noise + dropouts, perfect odometry        → + the sensor
  robot    ToF noise + dropouts + odometry drift          → what the robot would have

Each step, for robots on a step tile before any fall and within 0.6 m before / 0.2 m past the
first riser, it compares the map's 19×7 scan with the oracle scan at the same pose:

  coverage     share of scan cells in the foot zone (x ≤ 0.21 m ahead of the trunk) that are known
  error        |map − oracle| over known cells (median, p95, share > 1 cm)
  edge         along the centre column, where each scan says the next riser starts (first cell
               higher than half the riser); error in cm when both see it 0-0.3 m ahead, and how
               often the map has it within ±1.5 cm when the riser is ≤ 0.1 m ahead (imminent)

    uv run python scripts/eval/tof_map_quality.py --checkpoint <run 9>/model_3999.pt --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from step_up_failures import STEP_TILES, _risers  # noqa: E402
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks import microduck_step_up_env_cfg as su  # noqa: E402
from mjlab_microduck.tasks import tof  # noqa: E402
from mjlab_microduck.tasks.tof_map import TofHeightMap  # noqa: E402

TASK = "Mjlab-StepUp-Tof-Backlash-MicroDuck"   # any 125-D ToF checkpoint (run 9's loads here too)
VARIANTS = {}  # filled in run() from --fuse / --splat-y
_BASE = {
    "ideal": dict(noise_rel=0.0, noise_abs=0.0, dropout=0.0, odom_scale=0.0, odom_yaw_per_m_deg=0.0, odom_walk=0.0, odom_z_per_m=0.0),
    "sensor": dict(odom_scale=0.0, odom_yaw_per_m_deg=0.0, odom_walk=0.0, odom_z_per_m=0.0),
    "robot": dict(),
}


def run(a) -> None:
    env = build_env(a.task, a.num_envs, a.seed, a.device, fixed_heading=False)
    policy = WalkPolicy(a.checkpoint, a.vx, 0.0, 0.0, onnx=False, device=a.device)
    robot = env.scene["robot"]
    sensor = env.scene.sensors[tof.TOF_SENSOR_NAME]
    names, types, levels, _ = _terrain_layout(env)
    origins = env.scene.env_origins.cpu().numpy()
    n = env.num_envs
    on_step = np.array([names[types[i]] in STEP_TILES for i in range(n)])
    risers = {i: _risers(env.sim.mj_model, origins[i]) for i in range(n) if on_step[i]}
    xr = torch.tensor([risers[i][0][0] if on_step[i] and risers[i] else np.nan for i in range(n)], device=env.device)
    hr = torch.tensor([risers[i][0][1] - origins[i, 2] if on_step[i] and risers[i] else np.nan for i in range(n)], device=env.device)
    P = su.SCAN_PARAMS
    xs, ys = microduck_mdp.scan_grid(P["x_range"], P["y_range"], P["res"])
    nx, ny = len(xs), len(ys)
    gx, gy = torch.meshgrid(torch.tensor(xs, device=env.device), torch.tensor(ys, device=env.device), indexing="ij")
    offsets = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    foot_rows = [k for k, x in enumerate(xs) if x <= 0.21 + 1e-9]
    xs_t = torch.tensor(xs, device=env.device)
    maps = {k: TofHeightMap(n, env.device, splat=a.splat, splat_y=a.splat_y, fuse=a.fuse, **v) for k, v in _BASE.items()}

    obs, _ = env.reset()
    for m in maps.values():
        m.reset(torch.arange(n, device=env.device), robot.data.root_link_pos_w, robot.data.root_link_quat_w)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    acc = {k: {"cov": [], "err": [], "edge": [], "imminent_ok": [], "imminent_ok3": [], "imminent_n": 0} for k in maps}
    with torch.no_grad():
        for _ in range(round(a.seconds / env.step_dt)):
            pos, quat = robot.data.root_link_pos_w, robot.data.root_link_quat_w
            for m in maps.values():
                m.advance_odometry(pos)
                m.integrate(sensor.data.pos_w, sensor.data.hit_pos_w[:, :64], sensor.data.distances[:, :64], pos, env.step_dt)
            oracle = microduck_mdp.terrain_height_scan(env, **P, noise=0.0)               # (n, P)
            dist_to_riser = xr - pos[:, 0]
            window = torch.as_tensor(on_step, device=env.device) & ~fell & (dist_to_riser < 0.6) & (dist_to_riser > -0.2)
            if window.any():
                o = oracle[window].reshape(-1, nx, ny)
                thr = -0.5 * hr[window][:, None]
                o_hi = o[:, :, ny // 2] < thr
                o_edge = torch.where(o_hi.any(1), xs_t[o_hi.int().argmax(1)], torch.full_like(thr[:, 0], float("nan")))
                for k, m in maps.items():
                    vals, known = m.scan(offsets, pos, quat, P["nominal"])
                    v = vals[window].reshape(-1, nx, ny)
                    kn = known[window].reshape(-1, nx, ny)
                    acc[k]["cov"].append(kn[:, foot_rows].float().mean(dim=(1, 2)).cpu())
                    e = (v - o).abs()[kn]
                    acc[k]["err"].append(e.cpu())
                    col, colk = v[:, :, ny // 2], kn[:, :, ny // 2]
                    m_hi = (col < thr) & colk
                    m_edge = torch.where(m_hi.any(1), xs_t[m_hi.int().argmax(1)], torch.full_like(thr[:, 0], float("nan")))
                    ahead = (o_edge >= 0.0) & (o_edge <= 0.3)
                    both = ahead & ~torch.isnan(m_edge)
                    acc[k]["edge"].append((m_edge - o_edge)[both].cpu())
                    imminent = (o_edge >= 0.0) & (o_edge <= 0.1)
                    acc[k]["imminent_n"] += int(imminent.sum())
                    acc[k]["imminent_ok"].append(((m_edge - o_edge).abs() <= 0.015)[imminent].cpu())
                    acc[k]["imminent_ok3"].append(((m_edge - o_edge).abs() <= 0.03 + 1e-6)[imminent].cpu())
            obs, _, term, trunc, _ = env.step(policy(obs))
            fell |= term.bool()
            done = (term | trunc).bool()
            if done.any():
                ids = torch.nonzero(done).flatten()
                for m in maps.values():
                    m.reset(ids, robot.data.root_link_pos_w, robot.data.root_link_quat_w)
    print(f"\nToF height map vs oracle — {Path(a.checkpoint).name}, {n} robots ({int(on_step.sum())} on step tiles), "
          f"{a.seconds:.0f} s, vx {a.vx}, grid {maps['robot'].res * 100:.0f} cm, splat {a.splat}, splat_y {a.splat_y}, fuse {a.fuse}")
    print(f"{'map':8s} {'foot-zone coverage':>20s} {'|err| median / p95 / >1cm':>28s} {'edge err median / p90':>24s} {'imminent edge ±1.5 / ±3 cm':>28s}")
    for k in maps:
        cov = torch.cat(acc[k]["cov"]).numpy()
        err = torch.cat(acc[k]["err"]).numpy() * 100
        edge = torch.cat(acc[k]["edge"]).numpy() * 100
        imm = torch.cat(acc[k]["imminent_ok"]).numpy()
        imm3 = torch.cat(acc[k]["imminent_ok3"]).numpy()
        print(f"{k:8s} {100 * cov.mean():19.0f}% "
              f"{np.median(err):9.2f} / {np.percentile(err, 95):5.2f} / {100 * (err > 1).mean():4.1f}% cm "
              f"{np.median(np.abs(edge)):9.1f} / {np.percentile(np.abs(edge), 90):5.1f} cm "
              f"{100 * imm.mean():14.0f}% / {100 * imm3.mean():3.0f}% of {acc[k]['imminent_n']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default=TASK)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--vx", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splat", type=int, default=1, help="1 = a hit fills its own cell; 2 = also its 8 neighbours")
    p.add_argument("--splat-y", type=int, default=0, help="also fill this many cells to each side laterally")
    p.add_argument("--fuse", default="mean", choices=("mean", "latest"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
