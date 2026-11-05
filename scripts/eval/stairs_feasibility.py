"""Step-up feasibility for the stairs work: what the walk's feet actually do, and whether the
leg servos have the torque to lift the body over a 2-3 cm riser.

Two measurements with one walking policy (walk-v2 family, rsl_rl .pt or published ONNX):

  1. FLAT, per commanded speed 0.1-0.5 m/s: the per-swing foot apex (height above the ground
     from the foot ray sensor at the moment of landing, max over the swing) — median / p90 —
     and the hip-pitch / knee / ankle torque demand as a fraction of the voltage-limited BAM
     torque at each robot's own battery voltage (p99 and the fraction of steps ≥ 0.95).
  2. STAIRS (the straight small steps, rows 0-3 = 0.5-3 cm risers), every robot walking +x at
     0.3 m/s: outcome, apex on the stairs, torque demand on the approach vs on the steps, and
     the peak demand in the 0.3 s before each fall — if the servos sit at their limit when the
     robot goes down, the ceiling is torque; if they do not, it is clearance/balance.

The torque limit is MuJoCo's forcerange for the actuator (= kt·Vmax/R, what BAM installs),
scaled to the env's sampled battery voltage; it ignores the back-EMF loss at speed, so the
utilisation is a LOWER bound on how close the servo really is to its limit.

usage: uv run python scripts/eval/stairs_feasibility.py --checkpoint model.pt [--device cuda]
       [--num-envs 32] [--onnx policy.onnx] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from _common import OBS_VX
from view_terrain_env import STAIRS, WalkPolicy, _actor, _terrain_layout, build_env

FLAT_TASK = "Mjlab-WalkV2-Flat-Backlash-MicroDuck"
STAIRS_TASK = "Mjlab-WalkV2-Stairs-Backlash-MicroDuck"
JOINTS = {"hip_pitch": (2, 11), "knee": (3, 12), "ankle": (4, 13)}   # (left, right) ctrl ids on the walk model
SPEEDS = (0.1, 0.2, 0.3, 0.4, 0.5)
ROBOT_MASS_KG = 0.8


def torque_limits(env) -> tuple[torch.Tensor, str]:
    """Per-env, per-actuator torque cap, N·m: the tighter of BAM's firmware current limit
    (kt · max_current, 1.75 A → 0.64 N·m for the XL330, voltage-independent) and the voltage
    limit kt · vin / R at the env's sampled battery voltage (MuJoCo's forcerange is the latter
    at Vmax). For the XL330 the current limit binds at every voltage in the DR range."""
    robot = env.scene["robot"]
    hi = torch.as_tensor(env.sim.mj_model.actuator_forcerange[:, 1], device=env.device, dtype=torch.float32)
    hi = hi[robot.indexing.ctrl_ids] if hasattr(robot, "indexing") else hi
    limit = hi.unsqueeze(0).repeat(env.num_envs, 1)
    note = "MuJoCo forcerange"
    for act in robot.actuators:
        bam = getattr(act, "_bam_model", None)
        vin, rng = getattr(act, "vin_tensor", None), getattr(getattr(act, "cfg", None), "vin_range", None)
        if bam is None:
            continue
        kt, r = float(bam.kt.value), float(bam.R.value)
        volt = (vin.reshape(-1, 1) * kt / r) if vin is not None else torch.full((env.num_envs, 1), hi.max().item(), device=env.device)
        i_max = getattr(bam.actuator, "max_current", None)
        cur = torch.full_like(volt, kt * float(i_max)) if i_max else volt
        limit = torch.minimum(volt, cur).repeat(1, limit.shape[1])
        note = (f"min(kt·I_max = {kt * float(i_max):.2f} N·m, kt·vin/R = {volt.min().item():.2f}-{volt.max().item():.2f} N·m)"
                if i_max else f"kt·vin/R = {volt.min().item():.2f}-{volt.max().item():.2f} N·m")
        break
    return limit, note


class SwingTracker:
    """Per foot: running max of foot height while in the air; at landing, emit (apex, air time)."""

    def __init__(self, env, dt: float):
        low = env.cfg.rewards["low_swing"].params
        self.contact_sensor, self.height_sensor = env.scene[low["sensor_name"]], env.scene[low["height_sensor_name"]]
        self.dt = dt
        n = env.num_envs
        self.peak = torch.zeros(n, 2, device=env.device)
        self.air = torch.zeros(n, 2, device=env.device)
        self.prev_contact = torch.ones(n, 2, dtype=torch.bool, device=env.device)
        self.apex = [[] for _ in range(n)]         # per env: list of (apex, air_time, step)
        self.k = 0

    def step(self, mask: torch.Tensor | None = None) -> None:
        contact = self.contact_sensor.data.found > 0
        height = torch.nan_to_num(self.height_sensor.data.heights, nan=0.0)
        in_air = ~contact
        self.peak = torch.where(in_air, torch.maximum(self.peak, height), self.peak)
        self.air = torch.where(in_air, self.air + self.dt, self.air)
        landing = contact & ~self.prev_contact
        if mask is not None:
            landing = landing & mask.unsqueeze(1)
        for i, f in landing.nonzero().tolist():
            self.apex[i].append((float(self.peak[i, f]), float(self.air[i, f]), self.k))
        self.peak = torch.where(contact, torch.zeros_like(self.peak), self.peak)
        self.air = torch.where(contact, torch.zeros_like(self.air), self.air)
        self.prev_contact = contact
        self.k += 1


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def run_flat(policy_path: str, onnx: bool, n: int, device: str, seconds: float, seed: int) -> dict:
    env = build_env(FLAT_TASK, n, seed, device, fixed_heading=True)
    policy = WalkPolicy(policy_path, 0.0, 0.0, 0.0, onnx=onnx, device=device)
    speeds = torch.tensor([SPEEDS[i % len(SPEEDS)] for i in range(n)], device=device)
    limits, note = torque_limits(env)
    robot = env.scene["robot"]
    obs, _ = env.reset()
    tracker = SwingTracker(env, env.step_dt)
    util = []                                                            # (steps, N, nu) utilisation
    warm = round(1.0 / env.step_dt)
    with torch.no_grad():
        for k in range(round(seconds / env.step_dt) + warm):
            o = _actor(obs).clone()
            o[:, OBS_VX] = speeds
            o[:, OBS_VX + 1:OBS_VX + 3] = 0.0
            o[:, 51:61] = 0.0
            a = policy.policy(o) if policy.policy is not None else policy(o)
            obs, _, _, _, _ = env.step(a)
            if k >= warm:
                tracker.step()
                util.append((robot.data.actuator_force.abs() / limits).cpu().numpy())
    util = np.stack(util)                                                 # (T, N, nu)
    vin = next((a.vin_tensor.reshape(-1).cpu().numpy() for a in robot.actuators if getattr(a, "vin_tensor", None) is not None), None)
    out = {"per_speed": {}, "vin_range": [float(vin.min()), float(vin.max())] if vin is not None else None,
           "limit_nm_range": [float(limits.min()), float(limits.max())], "limit_note": note}
    for v in SPEEDS:
        ids = [i for i in range(n) if abs(float(speeds[i]) - v) < 1e-6]
        apex = [ap for i in ids for ap, air, _ in tracker.apex[i] if air >= 0.06]
        row = {"robots": len(ids), "swings": len(apex), "apex_median_cm": 100 * _pct(apex, 50), "apex_p10_cm": 100 * _pct(apex, 10),
               "apex_p90_cm": 100 * _pct(apex, 90)}
        for j, (l, r) in JOINTS.items():
            u = util[:, ids][:, :, [l, r]]
            row[f"{j}_util_p99"] = float(np.percentile(u, 99))
            row[f"{j}_sat_frac"] = float((u >= 0.95).mean())
        out["per_speed"][str(v)] = row
    env.close()
    return out


def run_stairs(policy_path: str, onnx: bool, n: int, device: str, seconds: float, seed: int, vx: float) -> dict:
    env = build_env(STAIRS_TASK, n, seed, device, fixed_heading=True)
    policy = WalkPolicy(policy_path, vx, 0.0, 0.0, onnx=onnx, device=device)
    limits, _note = torque_limits(env)
    robot = env.scene["robot"]
    names, types, levels, _ = _terrain_layout(env)
    gen = env.scene.terrain.cfg.terrain_generator
    st = gen.sub_terrains[STAIRS]
    on_stairs = np.nonzero(types == names.index(STAIRS))[0]
    obs, _ = env.reset()
    x0 = robot.data.root_link_pos_w[:, 0].clone()
    tracker = SwingTracker(env, env.step_dt)
    util, xrel, falls = [], [], []
    with torch.no_grad():
        for k in range(round(seconds / env.step_dt)):
            obs, _, term, _, _ = env.step(policy(obs))
            x = (robot.data.root_link_pos_w[:, 0] - x0)
            tracker.step()
            util.append((robot.data.actuator_force.abs() / limits).cpu().numpy())
            xrel.append(x.cpu().numpy())
            falls.append(term.cpu().numpy().astype(bool))
    util, xrel, falls = np.stack(util), np.stack(xrel), np.stack(falls)
    pre = round(0.3 / env.step_dt)
    # flight geometry (walk-v2 stairs cfg): spawn 0.5 m before the first riser, 4 treads up,
    # 0.6 m landing, 4 treads down. x_rel bands: approach / up / landing / down / after.
    up0 = st.start_length - 0.5
    bands = [("approach", -9.0, up0), ("up", up0, up0 + st.num_steps * st.step_width),
             ("landing", up0 + st.num_steps * st.step_width, up0 + st.num_steps * st.step_width + st.top_length),
             ("down", up0 + st.num_steps * st.step_width + st.top_length, up0 + 2 * st.num_steps * st.step_width + st.top_length),
             ("after", up0 + 2 * st.num_steps * st.step_width + st.top_length, 99.0)]
    rows = {}
    for i in on_stairs:
        r = int(levels[i])
        row = rows.setdefault(r, {"steps_cm": [100 * st.step_height(r / gen.num_rows), 100 * st.step_height((r + 1) / gen.num_rows)],
                                  "robots": 0, "fell": 0, "x_max_m": [], "apex_on_stairs_cm": [], "approach": [], "stairs": [], "pre_fall": [],
                                  "fall_where": [], "fall_last_apex_cm": [], "fall_knee_cap_frac": []})
        row["robots"] += 1
        fk = np.nonzero(falls[:, i])[0]
        row["fell"] += int(len(fk) > 0)
        row["x_max_m"].append(float(xrel[:, i].max()))
        appr = xrel[:, i] < 0.45
        on = (xrel[:, i] >= 0.45) & (xrel[:, i] <= 2.95)
        row["apex_on_stairs_cm"] += [100 * ap for ap, air, kk in tracker.apex[i] if air >= 0.06 and on[min(kk, len(on) - 1)]]
        for key, m in (("approach", appr), ("stairs", on)):
            if m.any():
                row[key].append(util[m][:, i])
        for f in fk:
            row["pre_fall"].append(util[max(0, f - pre):f + 1, i])
            xf = float(xrel[max(0, f - 1), i])
            row["fall_where"].append(next(name for name, lo, hi in bands if lo <= xf < hi))
            last = [ap for ap, air, kk in tracker.apex[i] if kk <= f and air >= 0.06]
            row["fall_last_apex_cm"].append(100 * last[-1] if last else float("nan"))
            w = util[max(0, f - pre):f + 1, i][:, [JOINTS["knee"][0], JOINTS["knee"][1]]]
            row["fall_knee_cap_frac"].append(float((w >= 0.95).mean()))
    out = {}
    for r, row in sorted(rows.items()):
        o = {"steps_cm": row["steps_cm"], "robots": row["robots"], "fell": row["fell"],
             "x_max_median_m": float(np.median(row["x_max_m"])),
             "apex_on_stairs_p10_cm": _pct(row["apex_on_stairs_cm"], 10),
             "apex_on_stairs_median_cm": _pct(row["apex_on_stairs_cm"], 50), "apex_on_stairs_p90_cm": _pct(row["apex_on_stairs_cm"], 90),
             "falls_where": {b: row["fall_where"].count(b) for b, _, _ in bands if row["fall_where"].count(b)},
             "fall_last_apex_median_cm": _pct([a for a in row["fall_last_apex_cm"] if a == a], 50),
             "fall_knee_at_cap_frac": float(np.mean(row["fall_knee_cap_frac"])) if row["fall_knee_cap_frac"] else float("nan")}
        for key in ("approach", "stairs", "pre_fall"):
            if row[key]:
                u = np.concatenate(row[key], axis=0)                      # (steps, nu)
                for j, (l, rr) in JOINTS.items():
                    uj = u[:, [l, rr]]
                    o[f"{key}_{j}_p99"] = float(np.percentile(uj, 99))
                    o[f"{key}_{j}_max"] = float(uj.max())
                    o[f"{key}_{j}_sat"] = float((uj >= 0.95).mean())
        out[str(r)] = o
    env.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--onnx", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--vx", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-flat", action="store_true")
    a = p.parse_args()
    path, onnx = (a.onnx, True) if a.onnx else (a.checkpoint, False)
    assert path, "--checkpoint or --onnx"

    flat = None if a.skip_flat else run_flat(path, onnx, a.num_envs, a.device, a.seconds, a.seed)
    if flat is not None:
      print(f"\nFLAT ({Path(path).name}, {a.num_envs} robots, battery {flat['vin_range'][0]:.2f}-{flat['vin_range'][1]:.2f} V); "
          f"torque cap {flat['limit_nm_range'][0]:.2f}-{flat['limit_nm_range'][1]:.2f} N·m = {flat['limit_note']}; utilisation = |τ| / cap")
      print(f"{'vx':>4s} {'swings':>6s} {'apex p10/med/p90 cm':>22s} | {'hip p99 sat':>12s} {'knee p99 sat':>13s} {'ankle p99 sat':>14s}")
      for v, r in flat["per_speed"].items():
        print(f"{float(v):4.1f} {r['swings']:6d} {r['apex_p10_cm']:6.2f} {r['apex_median_cm']:6.2f} {r['apex_p90_cm']:6.2f}        | "
              f"{r['hip_pitch_util_p99']:5.2f} {100*r['hip_pitch_sat_frac']:5.1f}% {r['knee_util_p99']:6.2f} {100*r['knee_sat_frac']:5.1f}% "
              f"{r['ankle_util_p99']:7.2f} {100*r['ankle_sat_frac']:5.1f}%")

    stairs = run_stairs(path, onnx, a.num_envs, a.device, a.seconds + 2.0, a.seed, a.vx)
    print(f"\nSTAIRS (straight small steps, vx {a.vx} m/s along the flight): utilisation p99 / max / % of steps ≥ 0.95")
    for r, o in stairs.items():
        print(f"row {r}: steps {o['steps_cm'][0]:.1f}-{o['steps_cm'][1]:.1f} cm, {o['fell']}/{o['robots']} fell, x progress median {o['x_max_median_m']:.2f} m, "
              f"apex on stairs p10/med/p90 {o['apex_on_stairs_p10_cm']:.2f}/{o['apex_on_stairs_median_cm']:.2f}/{o['apex_on_stairs_p90_cm']:.2f} cm")
        if o["falls_where"]:
            print(f"     falls at: {o['falls_where']}; last swing apex before a fall median {o['fall_last_apex_median_cm']:.2f} cm; "
                  f"knee at the cap in the 0.3 s before: {100 * o['fall_knee_at_cap_frac']:.0f}% of steps")
        for key in ("approach", "stairs", "pre_fall"):
            if f"{key}_knee_p99" in o:
                print(f"     {key:9s} " + "  ".join(f"{j} {o[f'{key}_{j}_p99']:.2f}/{o[f'{key}_{j}_max']:.2f}/{100*o[f'{key}_{j}_sat']:.0f}%" for j in JOINTS))
    g = 9.81 * ROBOT_MASS_KG
    print(f"\nquasi-static reference: body weight {g:.1f} N; lifting it on one knee with a 3 / 5 cm horizontal moment arm needs "
          f"{0.03*g:.2f} / {0.05*g:.2f} N·m.")
    if a.json:
        Path(a.json).write_text(json.dumps({"flat": flat, "stairs": stairs}, indent=1))
        print(f"json → {a.json}")


if __name__ == "__main__":
    main()
