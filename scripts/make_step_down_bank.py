"""Build the run-15 spawn bank (src/mjlab_microduck/tasks/data/step_down_poses.json) from the
static step-down scan (scripts/eval/step_down_static.py --directions down --keep-poses).

Each pose of the scan is a left-stance robot: the rear (stance) foot flat at the origin on the
upper tread, the front foot flat on the lower tread `h` below, the CoM over the rear foot, nothing
sinking into the step. The bank stores what mdp.reset_at_down_edge needs: the trunk pose relative
to the stance sole's centre (x forward, z up; pitch + = nose down) and the leg joints by name.
The right-stance mirror is the same values negated (checked against the STAND keyframe here).

    uv run python scripts/eval/step_down_static.py --directions down --keep-poses \\
        --heights 1.5 2 2.5 3 3.5 4 4.5 --json logs/staircase/step_down_bank.json
    uv run python scripts/make_step_down_bank.py logs/staircase/step_down_bank.json
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

OUT = ROOT / "src/mjlab_microduck/tasks/data/step_down_poses.json"
JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("scan", help="step_down_static.py --keep-poses JSON")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--max-per-height", type=int, default=120)
    a = p.parse_args()
    scan = json.load(open(a.scan))
    r = sds.Robot()
    m, d = r.m, r.d
    adr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS}
    for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
        lv, rv = r.stand[adr[f"left_{j}"]], r.stand[adr[f"right_{j}"]]
        assert abs(lv + rv) < 1e-3, f"STAND is not mirror-negated for {j}: {lv} vs {rv}"
    rng = np.random.default_rng(0)
    poses = []
    for c in scan["cases"]:
        if c["direction"] != "down":
            continue
        ps = c.get("poses", [])
        if len(ps) > a.max_per_height:
            ps = [ps[i] for i in sorted(rng.choice(len(ps), a.max_per_height, replace=False))]
        for x in ps:
            q = np.array(x["qpos"])
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            sole, up = r.foot(sds.STANCE)
            # the sagittal solve levels the sole in pitch only: STAND's ~2.7° foot roll stays (≤ 2 mm
            # across the 4 cm sole, inside the spawn's 2 mm lift)
            assert abs(sole[2]) < 1e-3 and up[2] > 0.998, (sole, up)
            poses.append({"h": c["h_cm"] / 100.0, "trunk_x": float(q[0] - sole[0]), "trunk_z": float(q[2] - sole[2]),
                          "pitch_deg": x["pitch_deg"], "d_mm": x["d_mm"], "sole_y": float(sole[1] - q[1]),
                          "q": [float(q[adr[n]]) for n in JOINTS]})
    sole_y = float(np.median([p_["sole_y"] for p_ in poses]))
    bank = {"source": a.scan, "joints": JOINTS, "mirror_sign": [-1.0] * len(JOINTS),
            "stance_sole_y": sole_y, "foot_half_len": r.half_len, "poses": poses}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(bank, open(a.out, "w"), indent=0)
    hs = sorted({p_["h"] for p_ in poses})
    print(f"{len(poses)} poses → {a.out}; per height:",
          {f"{h * 100:.1f}": sum(p_["h"] == h for p_ in poses) for h in hs}, f"stance sole y {sole_y * 1000:.1f} mm")


if __name__ == "__main__":
    main()
