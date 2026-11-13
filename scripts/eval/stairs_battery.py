"""The stairs battery — how many steps in a row: a long straight staircase per riser height, robot
by robot, up or (``--down``) down.

The play grid of a stairs task is replaced by one column per riser height (``--heights``), each
tile a single straight staircase along +x: a flat run-up (``--approach``), ``--steps`` risers of
that height with ``--tread`` treads, then a top platform. Every robot spawns facing the stairs
(yaw 0 unless ``--yaw-deg``) and is driven like the clean-climb battery (``step_up_failures.py``):
a constant ``--vx`` through its observation, head/body slots zero.

The ToF map the policy reads is anchored at the episode start and, in training, covers only
x ∈ [−0.6, 3.0] m from there; past that every cell reads "unknown". That is a limit of the sim's
map buffer, not of the skill (the robot's map would follow it), so the map is extended here to
cover the whole tile (``--map-x``).

Per robot, until it falls or reaches the top (later rows are ignored: a robot walking on off the
top platform drops into the next tile):
  climbed      risers climbed: the highest tread a foot stood on fully (heel past the edge,
               sole at that tread's height)
  clean run    risers climbed before the first toe catch or edge landing (the definitions of
               ``step_up_failures.py``: a foot in contact below a riser's top with its toe at the
               face; a foot standing on a top while straddling its edge)
  outcome      top (stood on the top platform), fell (at its climbed count), or stalled
  catches / edge landings per riser climbed

Per height it prints the Kaplan-Meier clean-run curve (median, S(k) = share with ≥ k clean risers)
and no-fall curve, censoring robots that leave the tile sideways (no heading correction here, and
the neighbouring column is another height), plus the hazard of the first unclean riser by riser
band — flat = a per-riser error rate, rising = the policy degrades along the flight.
``--from-json A.json B.json`` re-prints that summary from saved runs without simulating.

``--down`` turns every staircase around: the robot spawns on the upper platform and walks DOWN
``--steps`` risers (BoxSingleStepTerrainCfg ``down=True``). Per robot then: risers descended (the
lowest level a foot stood on wholly), the clean run (risers descended before the first edge contact —
a foot in contact while the ground under its footprint is not one level, the training cost
``foot_edge_contact``: toe hanging over an edge, heel caught on one), and the outcome ("top" in the
tables = reached the bottom).

The checkpoint is any 327-D stairs policy, or the switch of the two (``switch:<down.pt>+<up.pt>``,
tasks/stairs_switch.py). The batteries of docs/archive/stairs_runlog.md §28-29 ran on the stairs-up
task (the default), 20 risers, 20 cm treads, head down:

    uv run python scripts/eval/stairs_battery.py --checkpoint switch:<down.pt>+<up.pt> --device cuda \\
        --num-envs 1000 --heights 2 3 4 5 6 --tread 0.20 --seconds 45 --head down [--down] --json up_switch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402

from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks.microduck_terrains import BoxSingleStepTerrainCfg  # noqa: E402

TOL_Z = 0.003               # m, sole "below the top" (step_up_failures.py)
CATCH_X = (-0.004, 0.015)   # m, toe within this of a riser face (before, past)
ON_TOL_Z = 0.008            # m, a sole this close to a tread's height stands on it
ORIGIN_X = 0.5              # m, BoxSingleStepTerrainCfg's spawn origin from the tile start
STRADDLE_MM = 5.0           # mm, --down: an edge contact reaching this far across the edge on both
                            # sides is a straddle (an edge landing); less = a toe
                            # or heel just over the edge
# The head_pose command slot: "zero" = what every step-up battery used before 2026-11-08 (WalkPolicy
# zeroes it); "down" = the command the stairs policies were trained with and the playground
# orchestrator sends while climbing (neck −0.22 / head +0.22, HEAD_LOOK_DOWN).
HEADS = {"zero": None, "down": (-0.22, 0.22, 0.0, 0.0)}


def _risers(mjm: mujoco.MjModel, origin: np.ndarray, length: float, width: float,
            down: bool = False) -> list[tuple[float, float]]:
    """(edge x, top z) of every riser of the tile whose spawn origin is ``origin`` (world).
    ``down``: (edge x, level z after the drop) of every drop, the levels below the spawn platform."""
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    x0 = origin[0] - ORIGIN_X
    out = {}
    for g in range(mjm.ngeom):
        if mjm.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        body = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, mjm.geom_bodyid[g]) or ""
        if "terrain" not in body:
            continue
        c, s = d.geom_xpos[g], mjm.geom_size[g]
        top = c[2] + s[2]
        level_ok = top < origin[2] - 0.004 if down else top > origin[2] + 0.004
        if not (x0 - 0.01 <= c[0] <= x0 + length + 0.01 and abs(c[1] - origin[1]) < width / 2 - 0.05 and level_ok):
            continue
        out[round(float(c[0] - s[0]), 4)] = float(top)
    return sorted(out.items())


def staircase_hook(columns: list[tuple[float, float]], steps: int, approach: float, top: float, width: float,
                   rows: int, map_x: float, yaw_deg: float, map_y: float | None = None, down: bool = False):
    """(cfg hook, tile length): one straight staircase per column, ``columns`` = [(riser m, tread m)].
    Every tile is as long as the longest staircase (a shorter one gets a longer top platform)."""
    length = approach + (steps - 1) * max(t for _, t in columns) + top
    one_tread = len({t for _, t in columns}) == 1

    def hook(cfg):
        gen = cfg.scene.terrain.terrain_generator
        gen.size = (length, width)
        gen.num_rows = rows
        gen.curriculum = True
        gen.sub_terrains = {
            f"stairs_{100 * h:.1f}cm" + ("" if one_tread else f"_t{100 * t:.0f}"): BoxSingleStepTerrainCfg(
                proportion=1.0 / len(columns), step_height_range=(h, h), num_steps=steps,
                step_width=t, approach_range=(approach, approach), down=down)
            for h, t in columns}
        gen.num_cols = len(columns)
        scan = cfg.observations["actor"].terms["height_scan"]
        if "map_kwargs" in scan.params:           # the ToF map: cover the whole staircase
            scan.params["map_kwargs"] = {**scan.params["map_kwargs"], "x_range": (-0.6, map_x),
                                         **({} if map_y is None else {"y_range": (-map_y, map_y)})}
        yaw = float(np.radians(yaw_deg))        # a numpy scalar here turns the reset pose into float64
        cfg.events["reset_base"].params["pose_range"]["yaw"] = (-yaw, yaw)
    return hook, length


def riser_table(env, length: float, width: float, down: bool = False) -> list[list[tuple[float, float]]]:
    """Per robot: (edge x, top z) of the risers of its tile (``down``: the drops, see :func:`_risers`)."""
    origins = env.scene.env_origins.cpu().numpy()
    by_tile = {o: _risers(env.sim.mj_model, np.array(o), length, width, down)
               for o in {tuple(o) for o in origins.round(4)}}
    return [by_tile[tuple(o)] for o in origins.round(4)]


class FootRecorder:
    """Per control step: each foot's toe / heel x, sole z (collision AABB corners), contact; the
    trunk x, y; terminations. ``arrays()`` → dict of (T, N[, 2]) arrays for :func:`analyze`."""

    def __init__(self, env):
        mjm = env.sim.mj_model
        self.env, self.robot = env, env.scene["robot"]
        self.contact = env.scene.sensors["feet_ground_contact"]
        self.gid = [mjm.geom("robot/left_foot_collision").id, mjm.geom("robot/right_foot_collision").id]
        aabb = torch.as_tensor(mjm.geom_aabb[self.gid], dtype=torch.float32, device=env.device)
        signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                             dtype=torch.float32, device=env.device)
        self.local = aabb[:, None, :3] + signs[None] * aabb[:, None, 3:]
        self.hf = microduck_mdp.terrain_height_field(env)
        self.rec = {k: [] for k in ("toe", "heel", "sole", "found", "tx", "ty", "fell", "trunc", "edgeflag", "cx")}

    def __call__(self, term: torch.Tensor, trunc: torch.Tensor | None = None) -> None:
        """``term`` / ``trunc``: env.step's terminated / truncated. A truncation is a reset too
        (``out_of_terrain_bounds`` is one): the rows after it are the next episode's."""
        d = self.env.sim.data
        xmat = d.geom_xmat[:, self.gid].reshape(-1, 2, 3, 3)
        corners = d.geom_xpos[:, self.gid][:, :, None, :] + torch.einsum("bfij,fkj->bfki", xmat, self.local)
        self.rec["toe"].append(corners[..., 0].amax(-1).cpu().numpy())
        self.rec["heel"].append(corners[..., 0].amin(-1).cpu().numpy())
        self.rec["sole"].append(corners[..., 2].amin(-1).cpu().numpy())
        found = self.contact.data.found[:, :2] > 0
        self.rec["found"].append(found.cpu().numpy())
        # the training cost's definition: in contact while the ground under the footprint is not one level
        self.rec["edgeflag"].append(microduck_mdp.foot_edge_flags(self.hf, corners, found, 0.005).cpu().numpy())
        self.rec["cx"].append(corners[..., 0].mean(-1).cpu().numpy())
        self.rec["tx"].append(self.robot.data.root_link_pos_w[:, 0].cpu().numpy())
        self.rec["ty"].append(self.robot.data.root_link_pos_w[:, 1].cpu().numpy())
        self.rec["fell"].append(term.cpu().numpy().astype(bool))
        self.rec["trunc"].append(np.zeros_like(self.rec["fell"][-1]) if trunc is None
                                 else trunc.cpu().numpy().astype(bool))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.rec.items()}


def analyze(R: dict, i: int, rs: list[tuple[float, float]], origin: np.ndarray, steps: int, width: float,
            dt: float, approach: float, tile: str) -> tuple[dict, dict]:
    """One robot's staircase: (row for the JSON, detail) where detail holds the per-step level
    climbed ``run_lvl``, the unclean ``events`` [(step, kind, riser idx)], ``bad`` (end, 2) = the
    foot is in a catch / edge landing at that step, and ``end`` (steps analysed)."""
    T = R["tx"].shape[0]
    edges = np.array([x for x, _ in rs])
    tops = np.array([z for _, z in rs])
    fall_t = int(np.argmax(R["fell"][:, i])) if R["fell"][:, i].any() else T
    off = np.abs(R["ty"][:, i] - origin[1]) > width / 2 - 0.1      # walked into the next column
    if "trunc" in R:
        off |= R["trunc"][:, i]                                       # or out of the terrain: reset
    off_t = int(np.argmax(off)) if off.any() else T
    fell_t = min(fall_t, off_t)
    # the level each foot stands on fully (0 = the run-up), per row
    lvl = np.zeros((fell_t, 2), dtype=int)
    for f in range(2):
        c = R["found"][:fell_t, i, f]
        sole, heel = R["sole"][:fell_t, i, f], R["heel"][:fell_t, i, f]
        for k in range(steps):
            on = c & (np.abs(sole - tops[k]) < ON_TOL_Z) & (heel > edges[k] - 0.005)
            lvl[on, f] = np.maximum(lvl[on, f], k + 1)
    run_lvl = np.maximum.accumulate(lvl.max(axis=1)) if fell_t else np.zeros(0, int)
    top_t = int(np.argmax(run_lvl >= steps)) if (run_lvl >= steps).any() else None
    end = fell_t if top_t is None else min(fell_t, top_t + round(1.0 / dt))   # 1 s on top to see the last landing
    climbed = int(run_lvl[:end].max()) if end else 0
    # unclean events (onsets), each with the riser index it happened at
    lower = np.concatenate([[origin[2]], tops[:-1]])
    events, bad = [], np.zeros((end, 2), dtype=bool)
    for f in range(2):
        c = R["found"][:end, i, f, None]
        toe, heel, sole = (R[k][:end, i, f, None] for k in ("toe", "heel", "sole"))
        catch = c & (sole < tops - TOL_Z) & (sole > lower - 0.01) \
            & (toe - edges >= CATCH_X[0]) & (toe - edges <= CATCH_X[1])
        edge = c & (sole >= tops - TOL_Z) & (heel < edges - 0.005) & (edges - 0.005 < toe - 0.01)
        for kind, m in (("catch", catch), ("edge", edge)):
            on = m.any(axis=1)
            bad[:, f] |= on
            onset = np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))
            events += [(int(k), kind, int(np.flatnonzero(m[k])[-1])) for k in onset]
    events.sort()
    first_bad = events[0][2] if events else None
    clean_run = climbed if first_bad is None else min(first_bad, climbed)
    outcome = ("top" if top_t is not None else "off_tile" if off_t < fall_t
               else "fell" if fall_t < T else "stalled")
    row = {
        "robot": i, "tile": tile, "riser_cm": round(100 * float(tops[0] - origin[2]), 2),
        "outcome": outcome, "climbed": climbed, "clean_run": clean_run,
        "catches": sum(e[1] == "catch" for e in events), "edges": sum(e[1] == "edge" for e in events),
        "first_bad": None if not events else {"riser": events[0][2] + 1, "kind": events[0][1],
                                              "t": round(events[0][0] * dt, 2)},
        "top_s": None if top_t is None else round(top_t * dt, 2),
        "fell_s": round(fall_t * dt, 2) if fall_t < T else None,
        "x_end_m": float(R["tx"][max(end - 1, 0), i] - (origin[0] - ORIGIN_X + approach)),
    }
    return row, {"run_lvl": run_lvl[:end], "events": events, "bad": bad, "end": end, "top_t": top_t,
                 "fall_t": fall_t if fall_t < T else None}


def analyze_down(R: dict, i: int, rs: list[tuple[float, float]], origin: np.ndarray, steps: int, width: float,
                 dt: float, approach: float, tile: str) -> tuple[dict, dict]:
    """:func:`analyze` for a staircase going DOWN (``rs`` = (edge x, level z after the drop)). The row
    uses the same keys so :func:`summarize` reads it: ``climbed`` = risers descended, ``outcome``
    "top" = reached the bottom; the unclean events are edge contacts (``edgeflag``), each at the
    drop whose edge is nearest the foot."""
    T = R["tx"].shape[0]
    edges = np.array([x for x, _ in rs])
    lows = np.array([z for _, z in rs])
    fall_t = int(np.argmax(R["fell"][:, i])) if R["fell"][:, i].any() else T
    off = (np.abs(R["ty"][:, i] - origin[1]) > width / 2 - 0.1) | R["trunc"][:, i]
    off_t = int(np.argmax(off)) if off.any() else T
    fell_t = min(fall_t, off_t)
    E = np.concatenate([edges, [np.inf]])
    lvl = np.zeros((fell_t, 2), dtype=int)
    for f in range(2):
        c = R["found"][:fell_t, i, f]
        sole, heel, toe = R["sole"][:fell_t, i, f], R["heel"][:fell_t, i, f], R["toe"][:fell_t, i, f]
        for k in range(steps):
            on = c & (np.abs(sole - lows[k]) < ON_TOL_Z) & (heel > edges[k] - 0.005) & (toe < E[k + 1] + 0.005)
            lvl[on, f] = np.maximum(lvl[on, f], k + 1)
    run_lvl = np.maximum.accumulate(lvl.max(axis=1)) if fell_t else np.zeros(0, int)
    top_t = int(np.argmax(run_lvl >= steps)) if (run_lvl >= steps).any() else None
    end = fell_t if top_t is None else min(fell_t, top_t + round(1.0 / dt))
    descended = int(run_lvl[:end].max()) if end else 0
    events, bad, over = [], np.zeros((end, 2), dtype=bool), {}
    for f in range(2):
        on = R["edgeflag"][:end, i, f]
        bad[:, f] = on
        onset = np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))
        offset = np.flatnonzero(on & ~np.concatenate([on[1:], [False]]))
        for k0, k1 in zip(onset, offset):
            e = int(np.argmin(np.abs(edges - R["cx"][k0, i, f])))
            # how far the footprint reaches across the edge on its shorter side (mm, max over the
            # event): a toe 2 mm over the drop vs a foot half on the nose. ≤ 0 = the flag fired on
            # the height field's 1 cm cells, no real straddle.
            heel, toe = R["heel"][k0:k1 + 1, i, f], R["toe"][k0:k1 + 1, i, f]
            reach = np.minimum(toe - edges[e], edges[e] - heel)
            # where the sole is then, against the level above this edge (mm): ≈ 0 = flat on the
            # tread with part of the foot over the drop; < 0 = tipped over the nose (or on the way
            # down onto it); ≈ −riser = on the tread below
            upper = origin[2] if e == 0 else lows[e - 1]
            km = int(np.argmax(reach))
            over[(int(k0), f)] = (float(1000 * reach[km]), float(1000 * (R["sole"][k0 + km, i, f] - upper)))
            events.append((int(k0), "edge", e, f))
    events.sort()
    over_mm = [over[(k, f)][0] for k, _, _, f in events]
    sole_mm = [over[(k, f)][1] for k, _, _, f in events]
    first_bad = events[0][2] if events else None
    clean_run = descended if first_bad is None else min(first_bad, descended)
    strad = [ev for ev, mm in zip(events, over_mm) if mm >= STRADDLE_MM]
    clean_run_strad = descended if not strad else min(strad[0][2], descended)
    outcome = ("top" if top_t is not None else "off_tile" if off_t < fall_t
               else "fell" if fall_t < T else "stalled")
    row = {
        "robot": i, "tile": tile, "riser_cm": round(100 * float(origin[2] - lows[0]), 2), "direction": "down",
        "outcome": outcome, "climbed": descended, "clean_run": clean_run, "catches": 0,
        "edges": len(events), "edge_over_mm": [round(mm, 1) for mm in over_mm],
        "edge_sole_mm": [round(mm, 1) for mm in sole_mm],
        "first_bad": None if not events else {"riser": events[0][2] + 1, "kind": "edge",
                                              "t": round(events[0][0] * dt, 2)},
        # the same with only the straddles of ≥ STRADDLE_MM on both sides counted as unclean
        "clean_run_strad": clean_run_strad, "edges_strad": len(strad),
        "first_bad_strad": None if not strad else {"riser": strad[0][2] + 1, "kind": "edge",
                                                   "t": round(strad[0][0] * dt, 2)},
        "top_s": None if top_t is None else round(top_t * dt, 2),
        "fell_s": round(fall_t * dt, 2) if fall_t < T else None,
        "x_end_m": float(R["tx"][max(end - 1, 0), i] - (origin[0] - ORIGIN_X + approach)),
    }
    return row, {"run_lvl": run_lvl[:end], "events": events, "bad": bad, "end": end, "top_t": top_t,
                 "fall_t": fall_t if fall_t < T else None}


def run(a) -> dict:
    hook, length = staircase_hook([(float(h) / 100.0, a.tread) for h in a.heights], a.steps, a.approach, a.top,
                                  a.width, a.rows, a.map_x, a.yaw_deg, down=a.down)
    env = build_env(a.task, a.num_envs, a.seed, a.device, fixed_heading=False, cfg_hook=hook)
    policy = WalkPolicy(a.checkpoint, a.vx, 0.0, 0.0, onnx=False, device=a.device, head=HEADS[a.head])
    names, types, _, _ = _terrain_layout(env)
    origins = env.scene.env_origins.cpu().numpy()
    risers = riser_table(env, length, a.width, down=a.down)
    feet = FootRecorder(env)
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(round(a.seconds / env.step_dt)):
            obs, _, term, trunc, _ = env.step(policy(obs))
            feet(term, trunc)
    R = feet.arrays()
    rows = []
    for i in range(a.num_envs):
        if len(risers[i]) != a.steps:
            print(f"robot {i}: found {len(risers[i])} risers, expected {a.steps} — skipped")
            continue
        rows.append((analyze_down if a.down else analyze)(R, i, risers[i], origins[i], a.steps, a.width,
                                                          env.step_dt, a.approach, names[types[i]])[0])

    out = {"args": vars(a), "summary": summarize(rows, vars(a)), "rows": rows}
    if a.down:
        out["summary_strad"] = summarize_down_straddles(rows, vars(a))
    return out


def summarize_down_straddles(rows: list[dict], args: dict) -> list[dict]:
    """--down: how far the edge contacts reach across the edge, and the clean-run table again with
    only the straddles (≥ STRADDLE_MM on both sides) counted as unclean."""
    print(f"\n  edge contacts by how far the foot reaches across the edge (shorter side, max over the contact):")
    print(f"  {'riser':>6} {'events':>6} {'/riser':>6} | {'≤0 mm':>6} {'0-5':>6} {'5-15':>6} {'15-27':>6} {'>27':>6} (% of events)")
    for tile in sorted({r["tile"] for r in rows}, key=lambda t: float(t.split("_")[1][:-2])):
        rr = [r for r in rows if r["tile"] == tile]
        mm = np.array([m for r in rr for m in r.get("edge_over_mm", [])])
        risers = max(sum(r["climbed"] for r in rr), 1)
        bins = [(-1e9, 0), (0, 5), (5, 15), (15, 27), (27, 1e9)]
        pct = [100 * float(((mm > lo) & (mm <= hi)).mean()) if len(mm) else float("nan") for lo, hi in bins]
        line = f"  {rr[0]['riser_cm']:5.1f}c {len(mm):6d} {len(mm) / risers:6.2f} | " + " ".join(f"{p:5.0f}%" for p in pct)
        if "edge_sole_mm" in rr[0]:          # the straddles (≥ STRADDLE_MM): where the sole was
            sz = np.array([z for r in rr for m, z in zip(r["edge_over_mm"], r["edge_sole_mm"]) if m >= STRADDLE_MM])
            rise = 10 * rr[0]["riser_cm"]
            flat, low = np.abs(sz) <= 3, np.abs(sz + rise) <= 3
            line += (f" | straddles: flat on the upper tread {100 * flat.mean():3.0f}%, tipped/between "
                     f"{100 * (~flat & ~low).mean():3.0f}%, on the lower tread {100 * low.mean():3.0f}%" if len(sz) else "")
        print(line)
    print(f"\n  the same runs, unclean = a straddle ≥ {STRADDLE_MM:.0f} mm on both sides:")
    alt = [dict(r, clean_run=r["clean_run_strad"], first_bad=r["first_bad_strad"], edges=r["edges_strad"])
           for r in rows]
    return summarize(alt, args)


def _first_unclean(r: dict) -> int | None:
    """1-based riser of a robot's first unclean event (catch, edge landing, or a fall without one)."""
    if r["first_bad"] and r["clean_run"] == r["first_bad"]["riser"] - 1:
        return r["first_bad"]["riser"]
    return r["clean_run"] + 1 if r["outcome"] == "fell" else None


def _km(rows: list[dict], event_at, last_seen, n: int) -> list[float]:
    """Kaplan-Meier over risers: S(k) = P(no event through riser k). A robot is at risk at riser k
    while ``last_seen(r) >= k``; one that leaves the tile sideways or reaches the top is censored."""
    S, out = 1.0, []
    for k in range(1, n + 1):
        risk = sum(1 for r in rows if last_seen(r) >= k)
        ev = sum(1 for r in rows if event_at(r) == k)
        S *= 1 - ev / risk if risk else 1.0
        out.append(S)
    return out


BANDS = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 20), (21, 30), (31, 50))


def summarize(rows: list[dict], args: dict) -> list[dict]:
    """Per riser height: outcomes, the Kaplan-Meier clean run and no-fall curves (sideways exits
    censored), and the hazard of the first unclean riser per riser band."""
    n = args["steps"]
    print(f"\n{Path(args['checkpoint']).name} on {args['task']}: {n} risers, tread {100 * args['tread']:.0f} cm, "
          f"vx {args['vx']} m/s, yaw ±{args['yaw_deg']:.0f}°, {args['seconds']:.0f} s, {len(rows)} robots, "
          f"map x to {args['map_x']} m, head {args.get('head', 'zero')} (KM: robots leaving the tile sideways are censored)"
          + ("\n  DOWN: 'top' = reached the bottom, risers = drops descended, unclean = edge contact" if args.get("down") else ""))
    bands = [(lo, min(hi, n)) for lo, hi in BANDS if lo <= n]
    ks = [k for k in (5, 10, 20, 30, 50) if k <= n]
    print(f"  {'riser':>6} {'n':>4} {'top':>4} {'fell':>4} {'stl':>3} {'off':>3} | {'clean med':>9} "
          + " ".join(f"{'S' + str(k):>4}" for k in ks) + f" {'nofall':>6} | hazard %/riser "
          + "|".join(f"{lo}-{hi}" for lo, hi in bands))
    summary = []
    for tile in sorted({r["tile"] for r in rows}, key=lambda t: float(t.split("_")[1][:-2])):
        rr = [r for r in rows if r["tile"] == tile]
        oc = {o: sum(r["outcome"] == o for r in rr) for o in ("top", "fell", "stalled", "off_tile")}
        clean = _km(rr, _first_unclean, lambda r: _first_unclean(r) or r["climbed"], n)
        fell_at = lambda r: r["climbed"] + 1 if r["outcome"] == "fell" else None   # noqa: E731
        nofall = _km(rr, fell_at, lambda r: fell_at(r) or r["climbed"], n)
        med = next((k for k, v in enumerate(clean, 1) if v <= 0.5), None)
        hz = []
        for lo, hi in bands:
            risk = ev = 0
            for r in rr:
                f = _first_unclean(r)
                for k in range(lo, hi + 1):
                    if r["clean_run"] >= k - 1 and (f == k or r["climbed"] >= k):
                        risk += 1
                        ev += f == k
            hz.append(100 * ev / risk if risk else float("nan"))
        # FALL hazard per riser (§28): at risk on riser k while it got that far without leaving the
        # tile, event = fell on riser k. The column above is the FIRST UNCLEAN riser (edge or fall).
        fz = []
        for lo, hi in bands:
            risk = ev = 0
            for r in rr:
                fk = fell_at(r)
                last = fk or r["climbed"]
                for k in range(lo, hi + 1):
                    if last >= k:
                        risk += 1
                        ev += fk == k
            fz.append(100 * ev / risk if risk else float("nan"))
        print(f"  {rr[0]['riser_cm']:5.1f}c {len(rr):4d} {oc['top']:4d} {oc['fell']:4d} {oc['stalled']:3d} "
              f"{oc['off_tile']:3d} | {('>' + str(n)) if med is None else med:>9} "
              + " ".join(f"{100 * clean[k - 1]:3.0f}%" for k in ks) + f" {100 * nofall[-1]:5.0f}% | "
              + " | ".join(f"{h:4.1f}" for h in hz))
        print(f"  {'':>6} {'':>4} {'':>4} {'':>4} {'':>3} {'':>3}   {'':>9} " + " ".join(f"{'':>4}" for _ in ks)
              + f" {'':>6} | fall hazard    " + " | ".join(f"{h:4.1f}" for h in fz))
        summary.append({"tile": tile, "riser_cm": rr[0]["riser_cm"], "n": len(rr), **oc,
                        "clean_median_km": med, "clean_km": clean, "nofall_km": nofall,
                        "hazard_pct": {f"{lo}-{hi}": h for (lo, hi), h in zip(bands, hz)},
                        "fall_hazard_pct": {f"{lo}-{hi}": h for (lo, hi), h in zip(bands, fz)},
                        "climbed_p50": float(np.median([r["climbed"] for r in rr])),
                        "events": {k: sum(r[k] for r in rr) for k in ("catches", "edges")},
                        "first_bad_kind": {k: sum(1 for r in rr if r["first_bad"] and r["first_bad"]["kind"] == k)
                                           for k in ("catch", "edge")}})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Mjlab-StairsUp-Backlash-MicroDuck")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heights", nargs="+", default=["2", "3", "4", "5", "6"], help="riser heights, cm (one column each)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tread", type=float, default=0.15)
    p.add_argument("--approach", type=float, default=1.2, help="m from the tile start to the first riser (spawn at 0.5)")
    p.add_argument("--top", type=float, default=1.2, help="m of top platform")
    p.add_argument("--width", type=float, default=2.0, help="m, tile width")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--vx", type=float, default=0.25)
    p.add_argument("--yaw-deg", type=float, default=0.0, help="spawn yaw drawn from ±this")
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--map-x", type=float, default=6.0, help="m ahead of the spawn the ToF map covers")
    p.add_argument("--head", choices=sorted(HEADS), default="zero", help="the head_pose command slot (see HEADS)")
    p.add_argument("--down", action="store_true", help="spawn on top, walk DOWN the staircase")
    p.add_argument("--json", default=None)
    p.add_argument("--from-json", nargs="+", default=None, help="re-summarize saved runs instead of simulating")
    a = p.parse_args()
    if a.from_json:
        for path in a.from_json:
            d = json.load(open(path))
            summarize(d["rows"], d["args"])
            if d["args"].get("down") and "clean_run_strad" in d["rows"][0]:
                summarize_down_straddles(d["rows"], d["args"])
        return
    if not a.checkpoint:
        p.error("--checkpoint is required (or --from-json)")
    out = run(a)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
