"""Build the run-18a spawn banks (docs/archive/stairs_runlog.md §26) with the static solver of §22:

- ``tasks/data/edge_crouch_poses.json``: double support crouched on one level — both soles flat,
  side by side at STAND's width, the CoM over the soles (sagittal), trunk 6.5-12 cm over the soles,
  pitch -20..0° (+ = nose down). ``h`` = 0.12 - trunk z: the riser this crouch is the step-down
  target for (§22's highest working trunk is 12 cm - the riser, and the run-18a teacher tracks a
  trunk at that height before it steps).
- ``tasks/data/edge_stand_poses.json``: the same solve at the walking height (12.0-12.5 cm),
  pitch 0, ``h`` = 0: a robot standing before the edge.

Same format as step_down_poses.json (mdp.reset_at_down_edge): trunk pose relative to the left
(stance) sole's centre, leg joints by name, the right-stance mirror = the values negated.

    uv run python scripts/make_edge_crouch_bank.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import step_down_static as sds  # noqa: E402

DATA = ROOT / "src/mjlab_microduck/tasks/data"
JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
STAND_H = 0.12          # m: §22's stand; the crouch for a riser h is STAND_H - h


def _poses(r: sds.Robot, sols: list[dict], adr: dict, h_of) -> list[dict]:
    m, d = r.m, r.d
    out = []
    for x in sols:
        q = np.array(x["qpos"])
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        sole, up = r.foot("left")
        other, up2 = r.foot("right")
        assert abs(sole[2]) < 1e-3 and abs(other[2]) < 1e-3 and up[2] > 0.998 and up2[2] > 0.998, (sole, other)
        assert abs(other[0] - sole[0]) < 1e-3, (sole, other)                     # side by side
        tz = float(q[2] - sole[2])
        out.append({"h": round(h_of(tz), 4), "trunk_x": float(q[0] - sole[0]), "trunk_z": tz,
                    "pitch_deg": x["pitch_deg"], "d_mm": 0.0, "sole_y": float(sole[1] - q[1]),
                    "q": [float(q[adr[n]]) for n in JOINTS]})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--z-min", type=float, default=0.065)
    p.add_argument("--z-max", type=float, default=0.125)
    p.add_argument("--z-step", type=float, default=0.0025)
    p.add_argument("--pitch-min", type=float, default=-20.0)
    p.add_argument("--pitch-step", type=float, default=5.0)
    a = p.parse_args()
    r = sds.Robot()
    adr = {n: int(r.m.jnt_qposadr[r.m.joint(n).id]) for n in JOINTS}
    zs = np.arange(a.z_min, a.z_max + 1e-9, a.z_step)
    pitches = np.radians(np.arange(a.pitch_min, 1e-9, a.pitch_step))
    sols = sds.solve_case(r, 0.0, 0.0, zs, pitches)                            # flat: the free sole beside, d 0
    crouch = [s for s in sols if s["z"] <= STAND_H - 0.01 + 1e-9]
    stand = [s for s in sols if s["z"] >= STAND_H - 1e-9 and abs(s["pitch_deg"]) < 1e-6]
    for name, sel, h_of in (("edge_crouch_poses.json", crouch, lambda tz: STAND_H - tz),
                            ("edge_stand_poses.json", stand, lambda tz: 0.0)):
        poses = _poses(r, sel, adr, h_of)
        sole_y = float(np.median([q["sole_y"] for q in poses]))
        bank = {"source": "scripts/make_edge_crouch_bank.py", "joints": JOINTS, "mirror_sign": [-1.0] * len(JOINTS),
                "stance_sole_y": sole_y, "foot_half_len": r.half_len, "poses": poses}
        json.dump(bank, open(DATA / name, "w"), indent=0)
        zr = sorted({round(q["trunk_z"] * 100, 2) for q in poses})
        print(f"{len(poses)} poses → {DATA / name}; trunk z {zr[0]}-{zr[-1]} cm, "
              f"pitches {sorted({round(q['pitch_deg']) for q in poses})}, sole y {sole_y * 1000:.1f} mm")
    got = {(round(s["z"], 4), round(s["pitch_deg"])) for s in sols}
    miss = [(round(z * 100, 2), round(np.degrees(pp))) for z in zs for pp in pitches
            if (round(z, 4), round(np.degrees(pp))) not in got]
    print(f"no pose for {len(miss)} of {len(zs) * len(pitches)} (z cm, pitch°):", miss[:20])


if __name__ == "__main__":
    main()
