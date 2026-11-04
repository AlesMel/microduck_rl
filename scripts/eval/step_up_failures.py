"""Step-up failure modes: WHY a robot stumbles or falls at a step, robot by robot.

For every robot on a step tile (play grid, approach platform → first riser) it records, at 50 Hz:
the world bounding box of each foot's collision mesh (toe x, heel x, sole z), foot contact and
net contact force (global frame), the ToF hit points the policy can see (≤ 1 m), trunk speed and
the beam pitch. From that, per robot:

  toe catch      a foot in contact with its sole BELOW a riser's top and its toe at that riser's
                 face — the foot is pushing into the step instead of standing on a level
  edge landing   a foot in contact on a riser's top while straddling its edge (heel before it)
  blind gap      how long before the first foot reaches the step the riser left the ToF frame:
                 the policy is a feed-forward MLP, so after that it has no record of the step
  slowdown       mean forward speed 0.35-0.6 m before the riser vs 0.05-0.2 m before it
  fall context   distance to the riser at the fall, and whether a toe catch preceded it (0.5 s)

    uv run python scripts/eval/step_up_failures.py --checkpoint <run>/model_5999.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402
from mjlab_microduck.tasks import tof  # noqa: E402

TASK = "Mjlab-StepUp-Tof-Backlash-MicroDuck"
STEP_TILES = ("single_step", "short_flight", "high_step")   # high_step (run 11a): the right answer is to STOP
HIGH_TILES = ("high_step",)
VIEW_M = 1.0          # the policy's clip: beyond this a zone reads "nothing"
TOL_Z = 0.003         # m, sole "below the top"
CATCH_X = (-0.004, 0.015)   # m, toe within this of a riser face (before, past)


def _risers(mjm: mujoco.MjModel, origin: np.ndarray) -> list[tuple[float, float]]:
    """(edge x, top z) of every riser of the tile whose spawn origin is ``origin`` (world)."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    x0 = origin[0] - 0.5
    out = {}
    for g in range(mjm.ngeom):
        if mjm.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        body = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, mjm.geom_bodyid[g]) or ""
        if "terrain" not in body:
            continue
        c, s = d.geom_xpos[g], mjm.geom_size[g]
        top = c[2] + s[2]
        if not (x0 - 0.01 <= c[0] <= x0 + 4.01 and abs(c[1] - origin[1]) < 1.9 and top > origin[2] + 0.004):
            continue
        out[round(float(c[0] - s[0]), 4)] = float(top)
    return sorted(out.items())


def run(checkpoint: str, device: str, n: int, seconds: float, vx: float, seed: int, task: str = TASK) -> dict:
    env = build_env(task, n, seed, device, fixed_heading=False)
    policy = WalkPolicy(checkpoint, vx, 0.0, 0.0, onnx=False, device=device)
    mjm = env.sim.mj_model
    robot = env.scene["robot"]
    sensor = env.scene.sensors.get(tof.TOF_SENSOR_NAME)       # None on the oracle task (no ToF)
    contact = env.scene.sensors["feet_ground_contact"]
    names, types, levels, _ = _terrain_layout(env)
    origins = env.scene.env_origins.cpu().numpy()
    gid = [mjm.geom("robot/left_foot_collision").id, mjm.geom("robot/right_foot_collision").id]
    aabb = torch.as_tensor(mjm.geom_aabb[gid], dtype=torch.float32, device=env.device)   # (2, 6)
    signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=torch.float32, device=env.device)                       # (8, 3)
    local = aabb[:, None, :3] + signs[None] * aabb[:, None, 3:]                        # (2, 8, 3)
    jn = robot.joint_names
    hid = [jn.index("neck_pitch"), jn.index("head_pitch")]
    on_step = [i for i in range(n) if names[types[i]] in STEP_TILES]
    risers = {i: _risers(mjm, origins[i]) for i in on_step}

    rec = defaultdict(list)
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(round(seconds / env.step_dt)):
            obs, _, term, _, _ = env.step(policy(obs))
            xpos = env.sim.data.geom_xpos[:, gid]                                     # (B, 2, 3)
            xmat = env.sim.data.geom_xmat[:, gid].reshape(-1, 2, 3, 3)
            corners = xpos[:, :, None, :] + torch.einsum("bfij,fkj->bfki", xmat, local)  # (B, 2, 8, 3)
            rec["toe"].append(corners[..., 0].amax(-1).cpu().numpy())
            rec["heel"].append(corners[..., 0].amin(-1).cpu().numpy())
            rec["sole"].append(corners[..., 2].amin(-1).cpu().numpy())
            rec["found"].append((contact.data.found[:, :2] > 0).cpu().numpy())
            rec["force"].append(contact.data.force[:, :2].cpu().numpy())
            if sensor is not None:
                d = sensor.data.distances[:, :64]
                hx = sensor.data.hit_pos_w[:, :64, 0]
                visible = (d >= tof.RANGE_MIN_M) & (d <= VIEW_M)
                rec["near"].append(torch.where(visible, hx, torch.full_like(hx, 1e9)).amin(1).cpu().numpy())
                rec["far"].append(torch.where(visible, hx, torch.full_like(hx, -1e9)).amax(1).cpu().numpy())
            else:
                rec["near"].append(np.full(n, 1e9))
                rec["far"].append(np.full(n, -1e9))
            rec["tx"].append(robot.data.root_link_pos_w[:, 0].cpu().numpy())
            rec["tz"].append((robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy())
            rec["vx"].append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
            rec["fell"].append(term.cpu().numpy().astype(bool))
            q = robot.data.joint_pos[:, hid]
            rec["pitch"].append((q[:, 1] - q[:, 0]).cpu().numpy())
    R = {k: np.stack(v) for k, v in rec.items()}
    dt = env.step_dt
    T = R["tx"].shape[0]

    rows = []
    for i in on_step:
        rs = risers[i]
        if not rs:
            continue
        xr, top1 = rs[0]
        riser_total = rs[-1][1] - origins[i, 2]
        fell_t = int(np.argmax(R["fell"][:, i])) if R["fell"][:, i].any() else T
        end = fell_t
        z0 = R["tz"][0, i]
        on_top = fell_t == T and (R["tz"][-25:, i].mean() - z0) >= 0.8 * riser_total
        # run 11a: on a too-high tile a robot that survived, never got on top and ended still
        # (< 0.05 m/s over the last 0.5 s) before the riser has STOPPED — the wanted outcome
        stopped = (names[types[i]] in HIGH_TILES and fell_t == T and not on_top
                   and abs(R["vx"][-25:, i].mean()) < 0.05 and R["tx"][-1, i] < xr)
        catches, edges = [], []
        for f in range(2):
            prev_c = prev_e = False
            for k in range(end):
                c_now = e_now = False
                if R["found"][k, i, f]:
                    toe, heel, sole = R["toe"][k, i, f], R["heel"][k, i, f], R["sole"][k, i, f]
                    lower = origins[i, 2]
                    for xe, zt in rs:
                        if sole < zt - TOL_Z and sole > lower - 0.01 and CATCH_X[0] <= toe - xe <= CATCH_X[1]:
                            c_now = True
                        if sole >= zt - TOL_Z and heel < xe - 0.005 < toe - 0.01:
                            e_now = True
                        lower = zt
                if c_now and not prev_c:
                    F = R["force"][k, i, f]
                    catches.append((k, f, float(abs(F[0]) / (abs(F[2]) + 1e-6))))
                if e_now and not prev_e:
                    edges.append((k, f))
                prev_c, prev_e = c_now, e_now
        # first foot standing on the first riser's top
        step_on = next((k for k in range(end)
                        if any(R["found"][k, i, f] and R["toe"][k, i, f] > xr + 0.01 and R["sole"][k, i, f] >= top1 - TOL_Z
                               for f in range(2))), None)
        in_view = (R["near"][:end, i] <= xr) & (R["far"][:end, i] >= xr)
        seen = np.nonzero(in_view)[0]
        ref = step_on if step_on is not None else end - 1
        last_seen = int(seen[seen <= ref].max()) if len(seen) and (seen <= ref).any() else None
        lead_toe = R["toe"][:end, i].max(axis=1)
        gap_ahead = R["near"][:end, i] - lead_toe                         # visible ground starts this far ahead of the toes
        dist = xr - R["tx"][:end, i]
        far_m = (dist > 0.35) & (dist < 0.6)
        near_m = (dist > 0.05) & (dist < 0.2)
        row = {
            "robot": i, "tile": names[types[i]], "row": int(levels[i]), "riser_cm": round(100 * (top1 - origins[i, 2]), 2),
            "outcome": "top" if on_top else ("fell" if fell_t < T else ("stopped" if stopped else "stalled")),
            "stop_dist_m": float(xr - R["tx"][-1, i]) if stopped else None,
            "toe_catches": len(catches), "hard_catches": sum(c[2] > 0.5 for c in catches),
            "catch_fx_over_fz": [c[2] for c in catches],
            "edge_landings": len(edges),
            "fall_dist_to_riser_m": float(dist[fell_t - 1]) if fell_t < T and fell_t > 0 else None,
            "fall_after_catch": bool(fell_t < T and any(fell_t - 25 <= c[0] <= fell_t for c in catches)),
            "fall_after_edge": bool(fell_t < T and any(fell_t - 25 <= e[0] <= fell_t for e in edges)),
            "step_on_s": None if step_on is None else round(step_on * dt, 2),
            "riser_ever_seen": len(seen) > 0,
            "blind_s": None if (step_on is None or last_seen is None) else round((step_on - last_seen) * dt, 2),
            "blind_travel_m": None if (step_on is None or last_seen is None) else float(R["tx"][step_on, i] - R["tx"][last_seen, i]),
            "view_gap_ahead_of_toes_m": float(np.median(gap_ahead[dist > 0.2])) if (dist > 0.2).any() else None,
            "vx_far": float(R["vx"][:end, i][far_m].mean()) if far_m.any() else None,
            "vx_near": float(R["vx"][:end, i][near_m].mean()) if near_m.any() else None,
            "beam_pitch_deg_approach": float(np.degrees(np.median(R["pitch"][:end, i][dist > 0.05]))) if (dist > 0.05).any() else None,
        }
        rows.append(row)

    def med(key, rs):
        v = [r[key] for r in rs if r[key] is not None]
        return float(np.median(v)) if v else float("nan")

    print(f"\n{Path(checkpoint).name}  vx {vx} m/s, {n} robots, {seconds:.0f} s, {len(rows)} on step tiles")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tile"], r["row"])].append(r)
    for (tile, row), rs in sorted(groups.items()):
        oc = {o: sum(r["outcome"] == o for r in rs) for o in ("top", "fell", "stalled", "stopped")}
        stop_txt = (f" stopped {oc['stopped']} (clean {sum(r['outcome'] == 'stopped' and r['toe_catches'] == 0 and r['edge_landings'] == 0 for r in rs)},"
                    f" {100 * med('stop_dist_m', [r for r in rs if r['outcome'] == 'stopped']):.0f} cm before)"
                    if tile in HIGH_TILES else "")
        print(f"  {tile:12s} row {row} riser {rs[0]['riser_cm']:.1f} cm: top {oc['top']} fell {oc['fell']} stalled {oc['stalled']}{stop_txt} / {len(rs)}"
              f" | toe catches {sum(r['toe_catches'] for r in rs)} (robots with ≥1: {sum(r['toe_catches'] > 0 for r in rs)})"
              f", edge landings {sum(r['edge_landings'] for r in rs)}"
              f" | blind {med('blind_s', rs):.2f} s / {100 * med('blind_travel_m', rs):.0f} cm"
              f" | vx far {med('vx_far', rs):.2f} → near {med('vx_near', rs):.2f}")
    falls = [r for r in rows if r["outcome"] == "fell"]
    print(f"  ALL: top {sum(r['outcome'] == 'top' for r in rows)}, fell {len(falls)}, stalled {sum(r['outcome'] == 'stalled' for r in rows)} / {len(rows)}")
    print(f"  toe catches: {sum(r['toe_catches'] for r in rows)} on {sum(r['toe_catches'] > 0 for r in rows)} robots"
          f" (of the robots that ended on top: {sum(r['toe_catches'] > 0 for r in rows if r['outcome'] == 'top')}); "
          f"hard (|Fx| > 0.5 Fz): {sum(r['hard_catches'] for r in rows)} on {sum(r['hard_catches'] > 0 for r in rows)} robots")
    print(f"  CLEAN climbs (on top, no toe catch, no edge landing): "
          f"{sum(r['outcome'] == 'top' and r['toe_catches'] == 0 and r['edge_landings'] == 0 for r in rows)}/{len(rows)}")
    print(f"  falls preceded (0.5 s) by a toe catch: {sum(r['fall_after_catch'] for r in falls)}/{len(falls)}, "
          f"by an edge landing: {sum(r['fall_after_edge'] for r in falls)}/{len(falls)}; "
          f"fall distance to riser median {med('fall_dist_to_riser_m', falls):+.2f} m (− = past the riser)")
    print(f"  riser never in the ToF frame: {sum(not r['riser_ever_seen'] for r in rows)}/{len(rows)}")
    print(f"  visible ground starts {100 * med('view_gap_ahead_of_toes_m', rows):.0f} cm ahead of the toes (median, approach)")
    print(f"  riser left the frame {med('blind_s', rows):.2f} s / {100 * med('blind_travel_m', rows):.0f} cm before the first foot stood on it")
    print(f"  forward speed 0.35-0.6 m before the riser {med('vx_far', rows):.2f} m/s, 0.05-0.2 m before {med('vx_near', rows):.2f} m/s")
    print(f"  beam pitch on approach {med('beam_pitch_deg_approach', rows):.0f}° below the head's level pose")
    return {"vx": vx, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--vx", type=float, nargs="+", default=[0.25])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default=TASK)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    out = [run(a.checkpoint, a.device, a.num_envs, a.seconds, v, a.seed, a.task) for v in a.vx]
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
        print("json →", a.json)


if __name__ == "__main__":
    main()
