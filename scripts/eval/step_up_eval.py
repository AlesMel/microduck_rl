"""Step-up skill eval: does the policy climb, and does it USE the ToF?

Runs a checkpoint on the step-up play grid (4 rows × flat / single_step / short_flight) from the
approach platform at a forward command, and reports per (tile, row): the fraction that ends ON
TOP (trunk rose by ≥ 80 % of the riser total and did not fall), the fraction that fell, and the
median time to the top. ``--blind`` replaces the ToF frame with "nothing in range" (1.0) for the
whole rollout: a policy that climbs as well blind as sighted is not using the sensor.

    uv run python scripts/eval/step_up_eval.py --checkpoint <run>/model_2999.pt --device cuda
    uv run python scripts/eval/step_up_eval.py --checkpoint ... --device cuda --blind
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402
from mjlab_microduck.tasks import tof  # noqa: E402

TASK = "Mjlab-StepUp-Tof-Backlash-MicroDuck"
STEP_TILES = ("single_step", "short_flight")


def set_blind(env, blind: bool) -> None:
    for term_cfg in env.observation_manager._group_obs_class_term_cfgs["actor"]:
        if isinstance(term_cfg.func, tof.tof_depth):
            term_cfg.func.blank = blind


def grid_of(env, i: int) -> np.ndarray:
    d = env.scene.sensors[tof.TOF_SENSOR_NAME].data.distances[i, :64].cpu().numpy()
    return np.where(d < 0, np.nan, d).reshape(8, 8)


def run(checkpoint: str, device: str, n: int, seconds: float, vx: float, seed: int, blind: bool,
        snapshots: bool) -> dict:
    env = build_env(TASK, n, seed, device, fixed_heading=False)
    policy = WalkPolicy(checkpoint, vx, 0.0, 0.0, onnx=False, device=device)
    set_blind(env, blind)
    robot = env.scene["robot"]
    names, types, levels, _ = _terrain_layout(env)
    gen = env.scene.terrain.cfg.terrain_generator
    obs, _ = env.reset()
    origins_z = env.scene.env_origins[:, 2]
    z0 = (robot.data.root_link_pos_w[:, 2] - origins_z).clone()
    x0 = robot.data.root_link_pos_w[:, 0].clone()
    zs, xs, falls, heads = [], [], [], []
    jn = robot.joint_names
    head_ids = [jn.index("neck_pitch"), jn.index("head_pitch")]
    snap_at = {round(t / env.step_dt) for t in (0.5, 1.5, 2.5, 4.0)}
    focus = next((int(i) for i in np.argsort(-levels) if names[types[i]] == "single_step"), 0)
    with torch.no_grad():
        for k in range(round(seconds / env.step_dt)):
            obs, _, term, _, _ = env.step(policy(obs))
            zs.append((robot.data.root_link_pos_w[:, 2] - origins_z).cpu().numpy())
            xs.append((robot.data.root_link_pos_w[:, 0] - x0).cpu().numpy())
            falls.append(term.cpu().numpy().astype(bool))
            heads.append(robot.data.joint_pos[:, head_ids].cpu().numpy())
            if snapshots and k in snap_at:
                np.set_printoptions(precision=2, suppress=True, linewidth=120)
                print(f"\nToF of robot {focus} ({names[types[focus]]} row {int(levels[focus])}) at t = {k * env.step_dt:.1f} s, "
                      f"x {xs[-1][focus]:.2f} m, trunk rise {100 * (zs[-1][focus] - z0[focus].item()):.1f} cm:")
                print(grid_of(env, focus))
    zs, xs, falls, heads = np.stack(zs), np.stack(xs), np.stack(falls), np.stack(heads)
    z0 = z0.cpu().numpy()
    # Does the policy move the head? Per-robot std of the sensor pitch (neck + head pitch deltas
    # both pitch the beam, with opposite signs) over the rollout, on the step tiles.
    step_mask = np.array([names[types[i]] in STEP_TILES for i in range(n)])
    beam_pitch = heads[:, :, 1] - heads[:, :, 0]                      # + = down (head +, neck −)
    head_std = float(np.median(beam_pitch[:, step_mask].std(axis=0)))
    head_range = float(np.median(beam_pitch[:, step_mask].max(axis=0) - beam_pitch[:, step_mask].min(axis=0)))
    print(f"head motion on step tiles: beam pitch std median {np.degrees(head_std):.1f}°, range median {np.degrees(head_range):.1f}°")
    out: dict[str, dict] = {}
    for i in range(n):
        tile = names[types[i]]
        if tile not in STEP_TILES:
            continue
        st = gen.sub_terrains[tile]
        r = int(levels[i])
        riser = st.num_steps * st.step_height((r + 0.5) / gen.num_rows)
        key = f"{tile} row {r}"
        o = out.setdefault(key, {"tile": tile, "row": r, "riser_cm": round(100 * riser, 2), "robots": 0, "fell": 0,
                                 "on_top": 0, "t_top_s": [], "rise_cm": [], "x_max_m": []})
        o["robots"] += 1
        fell = bool(falls[:, i].any())
        o["fell"] += int(fell)
        rise = zs[:, i] - z0[i]
        first_fall = int(np.argmax(falls[:, i])) if fell else len(rise)
        reached = np.nonzero(rise[:first_fall] >= 0.8 * riser)[0]
        top = (not fell) and rise[-25:].mean() >= 0.8 * riser
        o["on_top"] += int(top)
        if len(reached):
            o["t_top_s"].append(float(reached[0] * env.step_dt))
        o["rise_cm"].append(float(100 * rise[:first_fall].max()))
        o["x_max_m"].append(float(xs[:first_fall, i].max()) if first_fall else 0.0)
    mode = "BLIND (ToF = nothing in range)" if blind else "SIGHTED"
    print(f"\n{mode}: {Path(checkpoint).name}, {n} robots, {seconds:.0f} s, vx {vx} m/s")
    tot_top = tot_fell = tot = 0
    for key in sorted(out, key=lambda k: (out[k]["tile"], out[k]["row"])):
        o = out[key]
        tt = np.median(o["t_top_s"]) if o["t_top_s"] else float("nan")
        print(f"  {key:20s} riser {o['riser_cm']:4.1f} cm: on top {o['on_top']}/{o['robots']}, fell {o['fell']}/{o['robots']}, "
              f"rise median {np.median(o['rise_cm']):4.1f} cm, x max median {np.median(o['x_max_m']):.2f} m, t_top median {tt:.1f} s")
        tot_top += o["on_top"]; tot_fell += o["fell"]; tot += o["robots"]
        o["t_top_median_s"] = float(tt); o["rise_median_cm"] = float(np.median(o["rise_cm"]))
        o["x_max_median_m"] = float(np.median(o["x_max_m"]))
        for k in ("t_top_s", "rise_cm", "x_max_m"):
            o.pop(k)
    print(f"  TOTAL on top {tot_top}/{tot}, fell {tot_fell}/{tot}")
    out["total"] = {"on_top": tot_top, "fell": tot_fell, "robots": tot, "blind": blind,
                    "head_pitch_std_deg": float(np.degrees(head_std)), "head_pitch_range_deg": float(np.degrees(head_range))}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--vx", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--snapshots", action="store_true", help="print the ToF grid of one single_step robot over time")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    out = run(a.checkpoint, a.device, a.num_envs, a.seconds, a.vx, a.seed, a.blind, a.snapshots)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print("json →", a.json)


if __name__ == "__main__":
    main()
