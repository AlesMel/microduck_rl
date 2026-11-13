"""The contacts battery: what each edge contact on the way down a staircase IS, and how the feet
cross each edge (docs/archive/stairs_runlog.md §20 and on).

The descent's edge contacts (``stairs_battery.py --down``: 0.4-1.2 per riser, the sole mostly
tipped between the two levels) could be two different things, and each wants another fix:
  landing   the foot touches down already across the edge: the step that crosses the edge is too
            short (or starts too far back), the foot comes down on the nose
  stance    the foot stood clean on the tread and then went over the nose while loaded: it rolls
            (pitches toe-down about the nose) or slides, typically the trailing foot as the body
            moves on and down
and each is the LEADING foot (the other one still on the level above this edge) or the TRAILING one
(the other one already below it). Per touchdown after a step-off, and per touchdown on a tread away
from any edge, the impact: peak foot force, peak trunk |a_z|, trunk v_z just before.

Per robot and edge contact (the training cost's definition, ``foot_edge_flags``) it records that
class, the duration, how far the foot reaches across, its pitch (toe down < 0) and load. Per
foot crossing of an edge (two touchdowns of one foot on either side of it): how far behind the
edge the foot left (toe at lift-off) and how far beyond it it landed (heel at touchdown) — a clean
step-off needs both ≥ 0, i.e. a same-foot stride of at least the 54 mm foot plus both gaps. And
the pitch of a loaded foot (normal force ≥ ``--load-n``) on flat ground, near an edge, and in an
edge contact: does the walk keep its soles flat anyway?

``--hard-terrain`` undoes ``_soften_terrain_contacts`` (terrain solref 0.04 → MuJoCo's default 0.02,
solimp back to the default): does rolling over the nose still work on a stiff edge? NaN
terminations (``nan_state``, the reason for the softening) are counted separately.

Same staircase, spawn, commands and recorder as ``stairs_battery.py --down``. The checkpoint is any
327-D stairs policy or the switch (``switch:<down.pt>+<up.pt>``, tasks/stairs_switch.py).

    uv run python scripts/eval/stairs_contacts.py --checkpoint switch:<down.pt>+<up.pt> --device cuda:0 \\
        --num-envs 600 --heights 2 3 4 --json logs/staircase/contacts_switch_t20.json [--hard-terrain]
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
from stairs_battery import (HEADS, FootRecorder, analyze_down, riser_table,  # noqa: E402
                            staircase_hook)
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402

LOAD_N = 2.0          # N: a foot carrying ≥ this (a quarter of the 7.8 N weight) is loaded
SWING_STEPS = 2       # a touchdown = contact after ≥ this many control steps without
NEAR_EDGE = 0.03      # m: a loaded foot whose heel or toe is this close to an edge is "near" it
DEFAULT_SOLREF = [0.02, 1.0]
DEFAULT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]


class ContactRecorder(FootRecorder):
    """FootRecorder + each foot's pitch (deg, toe down < 0; the foot box's long axis, oriented +x)
    and normal force (N), and the ``nan_state`` termination."""

    def __init__(self, env):
        super().__init__(env)
        half = env.sim.mj_model.geom_aabb[self.gid][:, 3:5]          # local x / y half sizes
        self.axis = [int(np.argmax(h)) for h in half]                 # the long axis per foot
        self.rec.update(pitch=[], fz=[], nan=[], vz=[], tz=[])

    def __call__(self, term: torch.Tensor, trunc: torch.Tensor | None = None) -> None:
        super().__call__(term, trunc)
        d = self.env.sim.data
        xmat = d.geom_xmat[:, self.gid].reshape(-1, 2, 3, 3)
        ax = torch.stack([xmat[:, f, :, self.axis[f]] for f in range(2)], dim=1)   # (B, 2, 3) world
        ax = torch.where(ax[..., :1] < 0, -ax, ax)                        # point it along +x (forward)
        self.rec["pitch"].append(torch.rad2deg(torch.asin(ax[..., 2].clamp(-1, 1))).cpu().numpy())
        force = self.contact.data.force[:, :2].reshape(len(term), 2, -1)
        self.rec["fz"].append(force[..., 2].abs().cpu().numpy() if force.shape[-1] >= 3
                              else force.norm(dim=-1).cpu().numpy())
        self.rec["vz"].append(self.robot.data.root_link_lin_vel_w[:, 2].cpu().numpy())
        self.rec["tz"].append(self.robot.data.root_link_pos_w[:, 2].cpu().numpy())
        try:
            nan = self.env.termination_manager.get_term("nan_state")
            self.rec["nan"].append(nan.cpu().numpy().astype(bool))
        except Exception:                                             # noqa: BLE001 - no such term
            self.rec["nan"].append(np.zeros(len(term), dtype=bool))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(first, last)] of every run of True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def touchdowns(c: np.ndarray, swing: int = SWING_STEPS) -> list[tuple[int, int]]:
    """[(touchdown, lift-off)] of every stance of one foot, ``c`` = contact per step; gaps shorter
    than ``swing`` steps are contact flicker, not a swing."""
    runs = _runs(c)
    out: list[list[int]] = []
    for a, b in runs:
        if out and a - out[-1][1] - 1 < swing:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _last_sole(R: dict, i: int, f: int, k: int) -> float:
    """Sole z of foot ``f`` at its last contact at or before step ``k`` (a foot in the air counts
    at the level it last stood on); +inf if it never touched down."""
    idx = np.flatnonzero(R["found"][:k + 1, i, f])
    return float(R["sole"][idx[-1], i, f]) if len(idx) else float("inf")


IMPACT_AFTER = 5      # control steps (0.1 s) after a touchdown searched for the force / |a_z| peak
IMPACT_BEFORE = 2     # control steps before it searched for |a_z| (the trunk may decelerate first)


def _impact(R: dict, i: int, f: int, k: int, end: int, dt: float) -> dict:
    """The touchdown of foot ``f`` at step ``k``: peak foot normal force in the next 0.1 s, peak trunk
    |a_z| (finite difference of v_z, as ``trunk_vertical_accel_penalty``) from 0.04 s before to 0.1 s
    after, and the trunk's v_z just before (< 0 = falling onto it). NaN without the trunk record."""
    k1 = min(k + IMPACT_AFTER, end)
    out = {"impact_fz_n": round(float(R["fz"][k:k1, i, f].max()), 2) if k1 > k else float("nan")}
    if "vz" in R and k >= 1:
        vz = R["vz"][:end, i]
        az = np.abs(np.diff(vz)) / dt                                   # az[j] = a_z between steps j and j+1
        lo, hi = max(k - IMPACT_BEFORE, 0), min(k1, len(az))
        out["impact_az"] = round(float(az[lo:hi].max()), 2) if hi > lo else float("nan")
        out["vz_td"] = round(float(vz[k - 1]), 3)
    else:
        out["impact_az"], out["vz_td"] = float("nan"), float("nan")
    return out


def classify(R: dict, i: int, edges: np.ndarray, lows: np.ndarray, top_z: float, end: int,
             load_n: float = LOAD_N, dt: float = 0.02) -> dict:
    """One robot's descent (steps < ``end``): its edge contacts, foot crossings and loaded-foot
    pitch samples (see the module docstring). ``lows[k]`` = the level after drop ``k``, ``top_z``
    the platform's. The trailing foot of an edge is the one whose other foot last stood below it."""
    uppers = np.concatenate([[top_z], lows[:-1]])
    events, crossings, flat_td = [], [], []
    pitch = {"flat": [], "near": [], "edge": []}
    for f in range(2):
        g = 1 - f
        c = R["found"][:end, i, f].astype(bool)
        flag = R["edgeflag"][:end, i, f].astype(bool)
        heel, toe, cx = R["heel"][:end, i, f], R["toe"][:end, i, f], R["cx"][:end, i, f]
        pz, fz = R["pitch"][:end, i, f], R["fz"][:end, i, f]
        stances = touchdowns(c)
        for k0, k1 in _runs(flag):
            e = int(np.argmin(np.abs(edges - cx[k0])))
            td = next((a for a, _ in reversed(stances) if a <= k0), k0)
            kind = "landing" if td >= k0 - 1 else "stance"
            reach = np.minimum(toe[k0:k1 + 1] - edges[e], edges[e] - heel[k0:k1 + 1])
            km = k0 + int(np.argmax(reach))
            other_below = _last_sole(R, i, g, k0) < uppers[e] - 0.5 * (uppers[e] - lows[e])
            events.append({
                "t": int(k0), "foot": f, "edge": e, "kind": kind,
                "role": "trailing" if other_below else "leading",
                "steps": int(k1 - k0 + 1),
                "reach_mm": round(1000 * float(reach.max()), 1),
                "pitch_deg": round(float(pz[km]), 1), "load_n": round(float(fz[km]), 2),
                # stance: where the foot had touched down and how it got across (slide vs roll)
                "td_toe_mm": round(1000 * float(toe[td] - edges[e]), 1),
                "slide_mm": round(1000 * float(cx[k0] - cx[td]), 1),
                "dpitch_deg": round(float(pz[km] - pz[td]), 1),
            })
        for (_, b0), (a1, _) in zip(stances, stances[1:]):
            crossed = np.flatnonzero((cx[b0] < edges) & (cx[a1] > edges))
            if len(crossed) != 1:
                continue
            e = int(crossed[0])
            other_below = _last_sole(R, i, g, a1) < uppers[e] - 0.5 * (uppers[e] - lows[e])
            crossings.append({
                "t": int(a1), "foot": f, "edge": e, "role": "trailing" if other_below else "leading",
                "left_behind_mm": round(1000 * float(edges[e] - toe[b0]), 1),   # ≥ 0: lifted off wholly behind
                "landed_beyond_mm": round(1000 * float(heel[a1] - edges[e]), 1),  # ≥ 0: landed wholly beyond
                "stride_mm": round(1000 * float(cx[a1] - cx[b0]), 1),
                # the trunk's height above the upper tread when this foot lifts off (a crouch
                # at the edge shows as a lower trunk than on a flat tread; ``flat_trunk_mm`` below)
                "trunk_mm": round(1000 * float(R["tz"][b0, i] - uppers[e]), 1) if "tz" in R else float("nan"),
                **_impact(R, i, f, a1, end, dt),
            })
        for (_, b0), (a1, _) in zip(stances, stances[1:]):                # touchdowns on a tread, away from edges
            if np.abs(np.concatenate([heel[[a1]] - edges, toe[[a1]] - edges])).min() >= NEAR_EDGE and \
                    not ((heel[a1] < edges) & (toe[a1] > edges)).any() and not ((cx[b0] < edges) & (cx[a1] > edges)).any():
                lvl = _last_sole(R, i, f, a1)
                flat_td.append({"t": int(a1), "foot": f, **_impact(R, i, f, a1, end, dt),
                                "trunk_mm": round(1000 * float(R["tz"][a1, i] - lvl), 1) if "tz" in R else float("nan")})
        loaded = c & (fz >= load_n)
        near = (np.abs(heel[:, None] - edges[None]).min(1) < NEAR_EDGE) | (np.abs(toe[:, None] - edges[None]).min(1) < NEAR_EDGE)
        between = ((heel[:, None] < edges[None]) & (toe[:, None] > edges[None])).any(1)
        pitch["edge"] += pz[loaded & flag].tolist()
        pitch["near"] += pz[loaded & ~flag & (near | between)].tolist()
        pitch["flat"] += pz[loaded & ~flag & ~near & ~between].tolist()
    return {"events": events, "crossings": crossings, "pitch": pitch, "flat_td": flat_td}


def _pct(x, q) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize(per: list[dict], label: str) -> dict:
    """Per riser height: shares of the four edge-contact classes, the stance ones split into
    rolled / slid, the crossings, and the loaded-foot pitch per zone."""
    print(f"\n{label}")
    print(f"  {'riser':>6} {'n':>4} {'bottom':>6} {'fell':>4} {'nan':>3} | {'ev/riser':>8} | "
          f"{'land lead':>9} {'land trail':>10} {'stnc lead':>9} {'stnc trail':>10} | "
          f"{'stance: td clean':>16} {'slide p50':>9} {'dpitch p50':>10} | {'pitch at edge p50/p10':>21}")
    out = []
    for h in sorted({p["riser_cm"] for p in per}):
        pp = [p for p in per if p["riser_cm"] == h]
        ev = [e for p in pp for e in p["events"]]
        ris = max(sum(p["descended"] for p in pp), 1)
        share = {f"{k}_{r}": 100 * sum(e["kind"] == k and e["role"] == r for e in ev) / max(len(ev), 1)
                 for k in ("landing", "stance") for r in ("leading", "trailing")}
        st = [e for e in ev if e["kind"] == "stance"]
        td_clean = 100 * sum(e["td_toe_mm"] <= 0 for e in st) / max(len(st), 1)
        slide = _pct([e["slide_mm"] for e in st], 50)
        dpitch = _pct([e["dpitch_deg"] for e in st], 50)
        pe = [e["pitch_deg"] for e in ev]
        oc = {o: sum(p["outcome"] == o for p in pp) for o in ("top", "fell", "stalled", "off_tile")}
        nan = sum(p["nan"] for p in pp)
        print(f"  {h:5.1f}c {len(pp):4d} {oc['top']:6d} {oc['fell']:4d} {nan:3d} | {len(ev) / ris:8.2f} | "
              f"{share['landing_leading']:8.0f}% {share['landing_trailing']:9.0f}% {share['stance_leading']:8.0f}% "
              f"{share['stance_trailing']:9.0f}% | {td_clean:15.0f}% {slide:8.1f}mm {dpitch:9.1f}° | "
              f"{_pct(pe, 50):9.1f}° / {_pct(pe, 10):6.1f}°")
        out.append({"riser_cm": h, "n": len(pp), **oc, "nan": nan, "events": len(ev),
                    "events_per_riser": len(ev) / ris, "shares_pct": share, "stance_td_clean_pct": td_clean,
                    "stance_slide_mm_p50": slide, "stance_dpitch_deg_p50": dpitch,
                    "edge_pitch_deg": {"p50": _pct(pe, 50), "p10": _pct(pe, 10)}})
    print(f"\n  crossings (one foot's lift-off before an edge → its touchdown after it):")
    print(f"  {'riser':>6} {'role':>8} {'n':>5} | {'clean':>6} | {'left behind p10/p50':>19} | "
          f"{'landed beyond p10/p50':>21} | {'stride p50':>10} | {'trunk at lift-off p10/p50':>25} (flat tread p50)")
    for h in sorted({p["riser_cm"] for p in per}):
        for role in ("leading", "trailing"):
            cr = [c for p in per if p["riser_cm"] == h for c in p["crossings"] if c["role"] == role]
            lb, la = [c["left_behind_mm"] for c in cr], [c["landed_beyond_mm"] for c in cr]
            clean = 100 * sum(c["left_behind_mm"] >= 0 and c["landed_beyond_mm"] >= 0 for c in cr) / max(len(cr), 1)
            print(f"  {h:5.1f}c {role:>8} {len(cr):5d} | {clean:5.0f}% | {_pct(lb, 10):8.0f} / {_pct(lb, 50):4.0f} mm | "
                  f"{_pct(la, 10):10.0f} / {_pct(la, 50):4.0f} mm | {_pct([c['stride_mm'] for c in cr], 50):7.0f} mm | "
                  f"{_pct([c.get('trunk_mm', np.nan) for c in cr], 10):11.0f} / {_pct([c.get('trunk_mm', np.nan) for c in cr], 50):4.0f} mm "
                  f"({_pct([t.get('trunk_mm', np.nan) for p in per if p['riser_cm'] == h for t in p.get('flat_td', [])], 50):.0f})")
            next(o for o in out if o["riser_cm"] == h)[f"crossings_{role}"] = {
                "n": len(cr), "clean_pct": clean, "left_behind_mm": {"p10": _pct(lb, 10), "p50": _pct(lb, 50)},
                "landed_beyond_mm": {"p10": _pct(la, 10), "p50": _pct(la, 50)},
                "stride_mm_p50": _pct([c["stride_mm"] for c in cr], 50),
                "trunk_mm": {"p10": _pct([c.get("trunk_mm", np.nan) for c in cr], 10),
                             "p50": _pct([c.get("trunk_mm", np.nan) for c in cr], 50)}}
    print(f"\n  touchdown impact: peak foot force (N) p50/p90 | peak trunk |a_z| (m/s²) p50/p90 | trunk v_z before (m/s) p50")
    for h in sorted({p["riser_cm"] for p in per}):
        rowout = next(o for o in out if o["riser_cm"] == h)
        groups = {"flat tread": [t for p in per if p["riser_cm"] == h for t in p.get("flat_td", [])]}
        for role in ("leading", "trailing"):
            cr = [c for p in per if p["riser_cm"] == h for c in p["crossings"] if c["role"] == role]
            groups[f"step-off {role}"] = cr
            if role == "leading":
                groups["  leading, stride < 150"] = [c for c in cr if c["stride_mm"] < 150]
                groups["  leading, stride ≥ 180"] = [c for c in cr if c["stride_mm"] >= 180]
        for name, g in groups.items():
            fz = [c["impact_fz_n"] for c in g if np.isfinite(c.get("impact_fz_n", np.nan))]
            az = [c["impact_az"] for c in g if np.isfinite(c.get("impact_az", np.nan))]
            vz = [c["vz_td"] for c in g if np.isfinite(c.get("vz_td", np.nan))]
            print(f"  {h:5.1f}c {name:<24} n {len(g):5d} | {_pct(fz, 50):5.1f} / {_pct(fz, 90):5.1f} N | "
                  f"{_pct(az, 50):5.1f} / {_pct(az, 90):5.1f} | {_pct(vz, 50):6.3f}")
            rowout.setdefault("impact", {})[name.strip()] = {
                "n": len(g), "fz_p50": _pct(fz, 50), "fz_p90": _pct(fz, 90),
                "az_p50": _pct(az, 50), "az_p90": _pct(az, 90), "vz_p50": _pct(vz, 50)}
    print(f"\n  pitch of a loaded foot (deg, toe down < 0): p10 / p50 / p90")
    for h in sorted({p["riser_cm"] for p in per}):
        row = []
        for z in ("flat", "near", "edge"):
            x = [v for p in per if p["riser_cm"] == h for v in p["pitch"][z]]
            row.append(f"{z} {_pct(x, 10):5.1f} / {_pct(x, 50):5.1f} / {_pct(x, 90):5.1f}")
            next(o for o in out if o["riser_cm"] == h)[f"pitch_{z}"] = {
                "n": len(x), "p10": _pct(x, 10), "p50": _pct(x, 50), "p90": _pct(x, 90)}
        print(f"  {h:5.1f}c  " + "  |  ".join(row))
    return {"label": label, "per_height": out}


def harden(cfg) -> None:
    """Undo ``_soften_terrain_contacts`` after the cfg's own spec_fn ran."""
    prev = cfg.scene.spec_fn

    def spec_fn(spec):
        if prev is not None:
            prev(spec)
        for geom in spec.body("terrain").geoms:
            geom.solref = DEFAULT_SOLREF
            geom.solimp = DEFAULT_SOLIMP
        print(f"[stairs_contacts] terrain contacts back to solref {DEFAULT_SOLREF}, solimp {DEFAULT_SOLIMP}")
    cfg.scene.spec_fn = spec_fn


def run(a) -> dict:
    hook, length = staircase_hook([(float(h) / 100.0, a.tread) for h in a.heights], a.steps, a.approach, a.top,
                                  a.width, a.rows, a.map_x, 0.0, down=True)
    if a.hard_terrain:
        def hook(cfg, _base=hook):
            _base(cfg)
            harden(cfg)
    env = build_env(a.task, a.num_envs, a.seed, a.device, fixed_heading=False, cfg_hook=hook)
    policy = WalkPolicy(a.checkpoint, a.vx, 0.0, 0.0, onnx=False, device=a.device, head=HEADS[a.head])
    names, types, _, _ = _terrain_layout(env)
    origins = env.scene.env_origins.cpu().numpy()
    risers = riser_table(env, length, a.width, down=True)
    rec = ContactRecorder(env)
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(round(a.seconds / env.step_dt)):
            obs, _, term, trunc, _ = env.step(policy(obs))
            rec(term, trunc)
    R = rec.arrays()
    per = []
    for i in range(a.num_envs):
        if len(risers[i]) != a.steps:
            continue
        row, det = analyze_down(R, i, risers[i], origins[i], a.steps, a.width, env.step_dt, a.approach,
                                names[types[i]])
        edges = np.array([x for x, _ in risers[i]])
        lows = np.array([z for _, z in risers[i]])
        cl = classify(R, i, edges, lows, float(origins[i][2]), det["end"], a.load_n, env.step_dt)
        T = R["nan"].shape[0]
        k_end = det["end"] if det["end"] < T else T - 1
        per.append({"robot": i, "riser_cm": row["riser_cm"], "outcome": row["outcome"],
                    "descended": row["climbed"], "nan": bool(R["nan"][:k_end + 1, i].any()), **cl})
    label = (f"{Path(a.checkpoint).name} on {a.task}: DOWN {a.steps} risers, tread {100 * a.tread:.0f} cm, "
             f"vx {a.vx}, {a.seconds:.0f} s, head {a.head}, "
             f"terrain {'HARD (solref 0.02)' if a.hard_terrain else 'softened (solref 0.04, training)'}")
    return {"args": vars(a), "summary": summarize(per, label), "robots": per}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Mjlab-StairsUp-Backlash-MicroDuck")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heights", nargs="+", default=["2", "3", "4"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tread", type=float, default=0.20)
    p.add_argument("--approach", type=float, default=1.2)
    p.add_argument("--top", type=float, default=2.0)
    p.add_argument("--width", type=float, default=3.0)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--map-x", type=float, default=7.0)
    p.add_argument("--vx", type=float, default=0.25)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--head", choices=sorted(HEADS), default="down")
    p.add_argument("--load-n", type=float, default=LOAD_N)
    p.add_argument("--hard-terrain", action="store_true")
    p.add_argument("--json", default=None)
    p.add_argument("--from-json", default=None, help="re-print the summary of a saved run")
    a = p.parse_args()
    if a.from_json:
        d = json.load(open(a.from_json))
        summarize(d["robots"], d["summary"]["label"])
        return
    out = run(a)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w"))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
