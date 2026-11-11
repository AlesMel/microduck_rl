"""Technical video of a stairs policy going up long staircases: the test of
``scripts/eval/stairs_battery.py``, filmed.

One column per staircase (``--stairs 4@20 3@15 …`` = riser cm @ tread cm, 20 risers each), the
robots walked up at a constant vx with no heading correction, exactly as in the battery. For each
column one hero is picked by an archetype (``--plan``), chosen as the TYPICAL robot of that kind,
not the luckiest:

  clean    reached the top with every riser clean; median time to the top among those
  catch    reached the top with ≥ 1 toe catch / edge landing; clean run nearest the column median
  fall     fell; risers climbed nearest the median among the fallers
  typical  clean run nearest the column median (any outcome but leaving the tile sideways)

The HUD counts what the battery counts (same analysis code): step k / 20, the clean run so far, toe
catches and edge landings, each flashed when it happens; a foot's halo is red while the battery
calls it a catch or an edge landing. ``--battery`` JSONs of the 200-robot runs add each column's
measured numbers (Kaplan-Meier median clean run, all 20 clean, no fall) to the HUD.

    # on the GPU box
    uv run python scripts/render_staircase.py record --checkpoint switch:<down.pt>+<up.pt> \\
        --stairs 4@20 3@15 5@24 5@15 6@15 --plan clean catch clean fall fall --out stairs.npz
    MUJOCO_GL=egl uv run python scripts/render_staircase.py render --states stairs.npz \\
        --battery logs/staircase/up_t15.json logs/staircase/up_t20.json logs/staircase/up_t24.json \\
        --video stairs.mp4 [--stills 0:5,2:12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "eval"))
from render_showcase import FPS, Showcase, _compose, _scan_points_w  # noqa: E402
from stairs_battery import (FootRecorder, _first_unclean, _km, analyze, riser_table,  # noqa: E402
                            staircase_hook)
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402

from mjlab_microduck.tasks import tof  # noqa: E402

TASK = "Mjlab-StairsUp-Backlash-MicroDuck"
HOLD_S = 1.5                      # s a fallen robot's last live frame is held
TOP_S = 2.0                       # s shown after reaching the top
FLASH_S = 1.2                     # s an event label stays up
RED = (235, 70, 60, 255)
KIND_LABEL = {"clean": "a typical clean climb", "catch": "a typical climb with slips", "fall": "a typical fall",
              "typical": "the typical robot"}
FEET = ("robot/left_foot_collision", "robot/right_foot_collision")


def _tile(h: float, t: float, one_tread: bool) -> str:
    """The sub-terrain name ``staircase_hook`` gives a column."""
    return f"stairs_{100 * h:.1f}cm" + ("" if one_tread else f"_t{100 * t:.0f}")


def _columns(specs: list[str]) -> list[tuple[float, float]]:
    out = []
    for s in specs:
        h, t = s.split("@")
        out.append((float(h) / 100.0, float(t) / 100.0))
    return out


def _hook(meta: dict):
    return staircase_hook([tuple(c) for c in meta["columns"]], meta["steps"], meta["approach"], meta["top"],
                          meta["width"], 1, meta["map_x"], 0.0)


# ── record ───────────────────────────────────────────────────────────────────

def _pick(rows: list[dict], kind: str, steps: int) -> dict | None:
    def nearest(cands, key, target):
        return min(cands, key=lambda r: (abs(key(r) - target), r["robot"])) if cands else None
    live = [r for r in rows if r["outcome"] != "off_tile"]
    if kind == "clean":
        c = [r for r in live if r["outcome"] == "top" and r["clean_run"] == steps]
        return nearest(c, lambda r: r["top_s"], float(np.median([r["top_s"] for r in c]))) if c else None
    if kind == "catch":
        c = [r for r in live if r["outcome"] == "top" and r["catches"] + r["edges"] > 0]
        return nearest(c, lambda r: r["clean_run"], float(np.median([r["clean_run"] for r in live]))) if c else None
    if kind == "fall":
        c = [r for r in live if r["outcome"] == "fell"]
        return nearest(c, lambda r: r["climbed"], float(np.median([r["climbed"] for r in c]))) if c else None
    if kind == "typical":
        return nearest(live, lambda r: r["clean_run"], float(np.median([r["clean_run"] for r in live])))
    raise ValueError(kind)


def record(a) -> None:
    columns = _columns(a.stairs)
    plan = a.plan or ["typical"] * len(columns)
    assert len(plan) == len(columns), "one --plan entry per --stairs column"
    length = a.approach + (a.steps - 1) * max(t for _, t in columns) + a.top
    meta = {"columns": columns, "plan": plan, "steps": a.steps, "approach": a.approach, "top": a.top,
            "width": a.width, "map_x": a.map_x or length, "task": a.task, "checkpoint": a.checkpoint,
            "seed": a.seed, "vx": a.vx, "num_envs": a.num_envs}
    hook, length = _hook(meta)
    env = build_env(a.task, a.num_envs, a.seed, a.device, fixed_heading=False, cfg_hook=hook)
    policy = WalkPolicy(a.checkpoint, a.vx, 0.0, 0.0, onnx=a.checkpoint.endswith(".onnx"), device=a.device)
    sensor = env.scene.sensors.get(tof.TOF_SENSOR_NAME)
    om = env.observation_manager
    assert list(om._group_obs_term_names["actor"])[-1] == "height_scan" and sensor is not None, \
        f"{a.task}: not a ToF-map task"
    scan_params = dict(om._group_obs_term_cfgs["actor"][-1].params)
    from mjlab_microduck.tasks import mdp as microduck_mdp
    xs, ys = microduck_mdp.scan_grid(scan_params["x_range"], scan_params["y_range"], scan_params["res"])
    npts = len(xs) * len(ys)
    names, types, _, _ = _terrain_layout(env)
    origins = env.scene.env_origins.cpu().numpy()
    risers = riser_table(env, length, a.width)
    feet = FootRecorder(env)
    nq = env.sim.mj_model.nq
    stride = max(1, round(1.0 / (a.fps * env.step_dt)))
    V = {k: [] for k in ("qpos", "spos", "beams", "dists", "scan_w", "frames_in")}
    obs, _ = env.reset()
    with torch.no_grad():
        for k in range(round(a.seconds / env.step_dt)):
            if k % stride == 0:
                d = sensor.data
                V["qpos"].append(env.sim.data.qpos.cpu().numpy().reshape(-1, nq).copy())
                V["spos"].append(d.pos_w.cpu().numpy().copy())
                V["beams"].append(d.hit_pos_w[:, :64].cpu().numpy().copy())
                V["dists"].append(d.distances[:, :64].cpu().numpy().copy())
                V["scan_w"].append(_scan_points_w(env, scan_params).cpu().numpy())
                V["frames_in"].append(obs["actor"][:, -2 * npts:].cpu().numpy().copy())
            obs, _, term, trunc, _ = env.step(policy(obs))
            feet(term, trunc)
    R = feet.arrays()
    dt = env.step_dt
    rows, details = [], {}
    for i in range(a.num_envs):
        if len(risers[i]) != a.steps:
            continue
        row, det = analyze(R, i, risers[i], origins[i], a.steps, a.width, dt, a.approach, names[types[i]])
        rows.append(row)
        details[i] = det
    heroes = []
    for c, ((h, t), kind) in enumerate(zip(columns, plan)):
        tile = _tile(h, t, len({tt for _, tt in columns}) == 1)
        col_rows = [r for r in rows if r["tile"] == tile]
        hero = _pick(col_rows, kind, a.steps) or _pick(col_rows, "typical", a.steps)
        got = kind if _pick(col_rows, kind, a.steps) else "typical (no " + kind + ")"
        oc = {o: sum(r["outcome"] == o for r in col_rows) for o in ("top", "fell", "stalled", "off_tile")}
        print(f"column {tile}: {len(col_rows)} robots {oc}; hero {hero['robot']} ({got}): {hero['outcome']}, "
              f"climbed {hero['climbed']}, clean run {hero['clean_run']}, catches {hero['catches']}, edges {hero['edges']}")
        heroes.append({"column": c, "tile": tile, "kind": got, "riser": h, "tread": t, "row": hero,
                       "n_column": len(col_rows), "outcomes": oc})
    out = {"meta": json.dumps({**meta, "heroes": heroes, "dt": dt, "stride": stride, "fps": a.fps, "npts": npts,
                               "scan_shape": [len(xs), len(ys)]})}
    for j, hh in enumerate(heroes):
        i, det = hh["row"]["robot"], details[hh["row"]["robot"]]
        for k, v in V.items():
            out[f"h{j}_{k}"] = np.stack([x[i] for x in v])
        out[f"h{j}_run_lvl"] = det["run_lvl"]
        out[f"h{j}_bad"] = det["bad"]
        out[f"h{j}_found"] = R["found"][:, i]                       # (T at 50 Hz, 2): foot contact
        out[f"h{j}_events"] = np.array(det["events"], dtype=object).astype(str) if det["events"] else np.zeros((0, 3), str)
    np.savez_compressed(a.out, **out)
    print("states →", a.out)


# ── render ───────────────────────────────────────────────────────────────────

def _battery_stats(paths: list[str]) -> dict:
    """(riser cm, tread cm) → (KM median clean run, all-clean share, no-fall share, n)."""
    stats = {}
    for p in paths or []:
        d = json.load(open(p))
        args = d["args"]
        for tile in {r["tile"] for r in d["rows"]}:
            rr = [r for r in d["rows"] if r["tile"] == tile]
            n = args["steps"]
            clean = _km(rr, _first_unclean, lambda r: _first_unclean(r) or r["climbed"], n)
            fell_at = lambda r: r["climbed"] + 1 if r["outcome"] == "fell" else None   # noqa: E731
            nofall = _km(rr, fell_at, lambda r: fell_at(r) or r["climbed"], n)
            med = next((k for k, v in enumerate(clean, 1) if v <= 0.5), None)
            stats[(round(rr[0]["riser_cm"]), round(100 * args["tread"]))] = (med, clean[-1], nofall[-1], len(rr), n)
    return stats


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Two-sided moving average along axis 0 (edges padded): the camera path, not the bobbing trunk."""
    pad = np.concatenate([np.repeat(x[:1], w, 0), x, np.repeat(x[-1:], w, 0)])
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    return np.stack([np.convolve(pad[:, c], ker, mode="valid") for c in range(x.shape[1])], axis=1)


def _events(st, j) -> list[tuple[int, str, int]]:
    e = st[f"h{j}_events"]
    return [(int(k), str(kind), int(r)) for k, kind, r in e] if len(e) else []


def render(a) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    from render_showcase import _font
    st = np.load(a.states, allow_pickle=False)
    meta = json.loads(str(st["meta"]))
    hook, _ = _hook(meta)
    env = build_env(meta["task"], 1, meta["seed"], "cpu", fixed_heading=False, cfg_hook=hook)
    shw = Showcase(env, a.width, a.height)
    dt, stride, steps, npts = meta["dt"], meta["stride"], meta["steps"], meta["npts"]
    fdt = dt * stride
    stats = _battery_stats(a.battery)
    stills = None
    if a.stills:
        stills = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in a.stills.split(",")]
    writer = None if stills else imageio.get_writer(a.video, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    ckpt = Path(meta["checkpoint"]).stem
    foot_ids = [shw.mjm.geom(g).id for g in FEET]
    try:
        for j, hh in enumerate(meta["heroes"]):
            row, h, t = hh["row"], hh["riser"], hh["tread"]
            qpos = st[f"h{j}_qpos"]
            run_lvl, bad, events = st[f"h{j}_run_lvl"], st[f"h{j}_bad"], _events(st, j)
            T = qpos.shape[0]
            if row["outcome"] == "fell":
                fall_k = int(round(row["fell_s"] / dt))
                last = min(T - 1, fall_k // stride - 1)       # the last live frame (the env resets on termination)
            elif row["outcome"] == "top":
                last = min(T - 1, int((row["top_s"] + TOP_S) / fdt))
            else:
                last = min(T - 1, len(run_lvl) // stride)
            n_frames = last + 1 + (int(HOLD_S / fdt) if row["outcome"] == "fell" else 0)
            trunk = _smooth(qpos[:last + 1, :3], int(0.5 / fdt))
            key = (round(100 * h), round(100 * t))
            if row["outcome"] == "top" and row["clean_run"] == steps:
                outcome = f"ON TOP · all {steps} risers clean"
            elif row["outcome"] == "top":
                outcome = f"ON TOP · clean run {row['clean_run']} · {row['catches']} toe catch(es), {row['edges']} edge landing(s)"
            elif row["outcome"] == "fell":
                outcome = f"FELL on riser {row['climbed'] + 1} · clean run {row['clean_run']}"
            else:
                outcome = f"{row['outcome'].upper()} at riser {row['climbed']}"
            print(f"segment {j}: {hh['tile']} hero {row['robot']} ({hh['kind']}), {n_frames} frames — {outcome}")
            for f in range(n_frames):
                k = min(f, last)
                if stills is not None and not any(s == j and abs(sec - k * fdt) < fdt / 2 for s, sec in stills):
                    continue
                kr = k * stride - 1                                          # 50 Hz analysis row of this frame
                ka = max(0, min(kr, len(run_lvl) - 1))
                u = k / max(last, 1)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = trunk[k] + np.array([0.14, 0.0, -0.06])
                cam.distance = a.cam_dist
                cam.azimuth = a.cam_az[0] + (a.cam_az[1] - a.cam_az[0]) * (3 * u * u - 2 * u * u * u)
                cam.elevation = a.cam_el
                # halo under each foot (Showcase puts it on the ground there); contact and red = the battery's
                shw._load(shw.extra, qpos[k])
                fp = shw.extra.geom_xpos[foot_ids].copy()
                found = st[f"h{j}_found"][max(0, kr)]
                flag = bad[kr] if 0 <= kr < len(bad) else np.zeros(2, bool)
                feet = (fp, found, flag & found)
                frame = shw.render(qpos[k][None], 0, cam, st[f"h{j}_spos"][k], st[f"h{j}_scan_w"][k],
                                   st[f"h{j}_dists"][k], mode="tof_map", scan_values=st[f"h{j}_frames_in"][k, :npts],
                                   feet=feet, beams=st[f"h{j}_beams"][k],
                                   known=st[f"h{j}_frames_in"][k, npts:] > 0.5)
                lvl = int(run_lvl[ka]) if len(run_lvl) else 0
                past = [e for e in events if e[0] <= ka]
                clean_now = lvl if not past else min(past[0][2], lvl)
                lines = [
                    f"Microduck · {steps}-step staircase · {ckpt}",
                    f"riser {100 * h:.0f} cm · tread {100 * t:.0f} cm · robot {row['robot']}: "
                    f"{KIND_LABEL.get(hh['kind'], hh['kind'])} of the {hh['n_column']} on this staircase",
                    f"t = {k * fdt:4.1f} s · riser {lvl} / {steps} · clean run {clean_now}"
                    f" · toe catches {sum(e[1] == 'catch' for e in past)} · edge landings {sum(e[1] == 'edge' for e in past)}",
                ]
                if key in stats:
                    med, allc, nof, n, ns = stats[key]
                    lines.append(f"battery, {n} robots: median clean run {('>' + str(ns)) if med is None else med}"
                                 f" · all {ns} clean {100 * allc:.0f} % · no fall {100 * nof:.0f} %")
                footer = ("beams: the head ToF  ·  dots: the 19×7 map it built (hollow = never seen)  ·  "
                          "feet: green = clean contact, red = toe catch / edge landing")
                img = _compose(frame, st[f"h{j}_frames_in"][k], lines, footer, mode="tof_map",
                               scan_shape=meta["scan_shape"])
                pil = Image.fromarray(img).convert("RGBA")
                draw = ImageDraw.Draw(pil, "RGBA")
                flash = [e for e in past if (ka - e[0]) * dt < FLASH_S]
                y = 18 + 34 + 26 * (len(lines) - 1) + 10
                for e in flash[-2:]:
                    label = ("TOE CATCH" if e[1] == "catch" else "EDGE LANDING") + f" · riser {e[2] + 1}"
                    draw.text((24, y), "▲ " + label, font=_font(22), fill=RED, stroke_width=2, stroke_fill=(0, 0, 0, 200))
                    y += 30
                if f >= last or (row["outcome"] == "top" and lvl >= steps):
                    col = (90, 220, 110, 255) if row["outcome"] == "top" and row["clean_run"] == steps else \
                        (240, 190, 70, 255) if row["outcome"] == "top" else RED
                    W, H = pil.size
                    fnt = _font(30)
                    tw = draw.textlength(outcome, font=fnt)
                    draw.rectangle(((W - tw) / 2 - 16, H - 110, (W + tw) / 2 + 16, H - 64), fill=(0, 0, 0, 170))
                    draw.text(((W - tw) / 2, H - 104), outcome, font=fnt, fill=col)
                img = np.asarray(pil.convert("RGB"))
                if stills is not None:
                    out = Path(a.video).with_suffix("").as_posix() + f"_s{j}_{k * fdt:04.1f}.png"
                    Image.fromarray(img).save(out)
                    print("still →", out)
                else:
                    writer.append_data(img)
    finally:
        if writer is not None:
            writer.close()
        shw.close()
    if writer is not None:
        print("video →", a.video)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--task", default=TASK)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--stairs", nargs="+", default=["4@20", "3@15", "5@24", "5@15", "6@15"], help="riser cm @ tread cm")
    r.add_argument("--plan", nargs="+", default=None, help="clean / catch / fall / typical, one per column")
    r.add_argument("--num-envs", type=int, default=120)
    r.add_argument("--seconds", type=float, default=40.0)
    r.add_argument("--vx", type=float, default=0.25)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, default=20)
    r.add_argument("--approach", type=float, default=1.2)
    r.add_argument("--top", type=float, default=1.2)
    r.add_argument("--width", type=float, default=2.0)
    r.add_argument("--map-x", type=float, default=None, help="default: the tile length")
    r.add_argument("--fps", type=int, default=FPS)
    r.add_argument("--out", default="stairs.npz")
    v = sub.add_parser("render")
    v.add_argument("--states", required=True)
    v.add_argument("--video", default="stairs.mp4")
    v.add_argument("--battery", nargs="*", default=None, help="stairs_battery.py JSONs for the HUD's numbers")
    v.add_argument("--width", type=int, default=1280)
    v.add_argument("--height", type=int, default=720)
    v.add_argument("--cam-az", type=float, nargs=2, default=(100.0, 80.0), help="start, end (90 = from the robot's right)")
    v.add_argument("--cam-el", type=float, default=-14.0)
    v.add_argument("--cam-dist", type=float, default=1.0)
    v.add_argument("--stills", default=None, help="segment:seconds,… → PNGs next to --video instead of the mp4")
    a = p.parse_args()
    record(a) if a.cmd == "record" else render(a)


if __name__ == "__main__":
    main()
