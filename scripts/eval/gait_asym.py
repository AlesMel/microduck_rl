"""Where does straight-walk yaw drift come from? Left/right gait asymmetry per robot vs its drift.

Open loop (wz command 0) at a fixed forward command, play cfg, no pushes (default task: flat + backlash). Over
the measuring window, per robot:
  drift            mean body yaw rate (rad/s; + = left/CCW)
  yaw by phase     trunk yaw accumulated (deg per second of walking) while only the LEFT foot is
                   down, only the RIGHT, both, none — a symmetric gait cancels L against R
Per foot (R mirrored so a symmetric gait gives L == R):
  stance_s, swing_s    mean contact / air duration
  step_m               foot displacement along the trunk heading between its touchdowns
  width_m              lateral foot offset from the trunk at touchdown (outward +)
  toe_out_deg          foot yaw vs trunk yaw at touchdown (outward +)
  pivot_deg            foot yaw change on the ground during one stance (toward the body's left +
                       for both feet: + on either foot turns the robot left)
  push_Ns              contact force impulse per stance (sensor net force norm)
  joint means          hip_yaw/hip_roll/hip_pitch/knee/ankle (mirrored), ankle min/max
Then the population mean of L, R, L−R and the across-robot correlation of each L−R (and of the
robot's randomized trunk CoM y and hip-yaw encoder biases) with drift.

usage: uv run scripts/eval/gait_asym.py MODEL.pt OUT.json [--task ...] [--vx 0.4] [--n 64]
       [--robot-chain] [--device cuda:0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import DEFAULT_WALK_TASK, RobotChain, load_policy, make_play_env, step  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ckpt"); p.add_argument("out")
p.add_argument("--task", default=DEFAULT_WALK_TASK)
p.add_argument("--vx", type=float, default=0.4)
p.add_argument("--n", type=int, default=64)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--settle", type=float, default=4.0)
p.add_argument("--robot-chain", action="store_true")
p.add_argument("--device", default="cpu")
a = p.parse_args()

policy = load_policy(a.ckpt, a.device)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


env, raw = make_play_env(a.task, a.n, a.device)
chain = RobotChain(raw, a.n, a.device) if a.robot_chain else None
robot = raw.scene["robot"]
sensor = raw.scene.sensors["feet_ground_contact"]
assert sensor.cfg.primary.pattern.index("left") < sensor.cfg.primary.pattern.index("right")
jn = list(robot.joint_names)
J = {f"{s}_{j}": jn.index(f"{s}_{j}") for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")}
sites = [robot.site_names.index("left_foot"), robot.site_names.index("right_foot")]
trunk = robot.body_names.index("trunk_base")
dt = raw.step_dt
steps, settle = int(a.seconds / dt), int(a.settle / dt)

# Randomized draws (per world): trunk CoM y, hip-yaw encoder biases.
import mujoco  # noqa: E402
trunk_bid = mujoco.mj_name2id(raw.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
col = raw.sim.model.body_ipos[:, trunk_bid, 1]
com_y = (col - col.mean()).cpu().numpy() if col.numel() == a.n and col.std() > 0 else None
eb = getattr(robot.data, "encoder_bias", None)
enc = {k: eb[:, J[k]].cpu().numpy() for k in ("left_hip_yaw", "right_hip_yaw")} if eb is not None and eb.shape[1] == len(jn) else None

L = {k: [] for k in ("found", "force", "foot_pos", "root_xy", "foot_yaw", "trunk_yaw", "wz", "q", "up")}
obs = env.get_observations()["actor"]
for k in range(steps):
    obs = step(env, policy, obs, a.vx if k >= 50 else 0.0, chain=chain)
    if k >= settle:
        d = robot.data
        L["found"].append((sensor.data.found > 0).cpu().numpy().reshape(a.n, -1)[:, :2])
        L["force"].append(sensor.data.force.reshape(a.n, 2, -1).norm(dim=-1).cpu().numpy())
        L["foot_pos"].append(d.site_pos_w[:, sites, :2].cpu().numpy())
        L["root_xy"].append(d.root_link_pos_w[:, :2].cpu().numpy())
        L["foot_yaw"].append(yaw_of(d.site_quat_w[:, sites]).cpu().numpy())
        L["trunk_yaw"].append(yaw_of(d.root_link_quat_w).cpu().numpy())
        L["wz"].append(d.root_link_ang_vel_b[:, 2].cpu().numpy())
        L["q"].append(d.joint_pos.cpu().numpy())
        L["up"].append((-d.projected_gravity_b[:, 2]).cpu().numpy())
X = {k: np.stack(v, 1) for k, v in L.items()}   # (n, T, ...)
T = X["wz"].shape[1]
ok = (X["up"] > 0.8).all(1)


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


rows = []
for e in np.where(ok)[0]:
    c = X["found"][e]                       # (T, 2) left, right
    ty = np.unwrap(X["trunk_yaw"][e])
    dyaw = np.diff(ty, prepend=ty[0])
    phase = {"L_only": c[:, 0] & ~c[:, 1], "R_only": ~c[:, 0] & c[:, 1], "both": c[:, 0] & c[:, 1], "none": ~c[:, 0] & ~c[:, 1]}
    r = {"drift": float(X["wz"][e].mean())}
    secs = T * dt
    for ph, m in phase.items():
        r[f"yaw_{ph}"] = float(np.degrees(dyaw[m].sum()) / secs)
        r[f"frac_{ph}"] = float(m.mean())
    for f, side in ((0, "L"), (1, "R")):
        s = c[:, f].astype(int)
        td = np.where(np.diff(s) == 1)[0] + 1
        lo = np.where(np.diff(s) == -1)[0] + 1
        mir = 1.0 if side == "L" else -1.0
        # stances fully inside the window
        st = [(t0, t1) for t0 in td for t1 in lo[lo > t0][:1]]
        r[f"{side}_n_stance"] = len(st)
        r[f"{side}_stance_s"] = float(np.mean([t1 - t0 for t0, t1 in st]) * dt) if st else np.nan
        sw = [(t0, t1) for t0 in lo for t1 in td[td > t0][:1]]
        r[f"{side}_swing_s"] = float(np.mean([t1 - t0 for t0, t1 in sw]) * dt) if sw else np.nan
        fwd = np.stack([np.cos(ty), np.sin(ty)], -1)
        lat = np.stack([-np.sin(ty), np.cos(ty)], -1)          # body-left
        fp = X["foot_pos"][e, :, f]
        if len(td) > 1:
            steps_m = [float((fp[t1] - fp[t0]) @ fwd[t1]) for t0, t1 in zip(td[:-1], td[1:])]
            r[f"{side}_step_m"] = float(np.mean(steps_m))
        else:
            r[f"{side}_step_m"] = np.nan
        trunk_xy = X["root_xy"][e]
        r[f"{side}_width_m"] = float(np.mean([mir * (fp[t] - trunk_xy[t]) @ lat[t] for t in td])) if len(td) else np.nan
        fy = X["foot_yaw"][e, :, f]
        r[f"{side}_toe_out_deg"] = float(np.degrees(np.mean([mir * wrap(fy[t] - ty[t]) for t in td]))) if len(td) else np.nan
        r[f"{side}_pivot_deg"] = float(np.degrees(np.mean([wrap(fy[t1 - 1] - fy[t0]) for t0, t1 in st]))) if st else np.nan
        r[f"{side}_push_Ns"] = float(np.mean([X["force"][e, t0:t1, f].sum() * dt for t0, t1 in st])) if st else np.nan
        sgn = {"hip_yaw": mir, "hip_roll": mir, "hip_pitch": mir, "knee": mir, "ankle": mir}
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
            qq = X["q"][e, :, J[f"{'left' if side == 'L' else 'right'}_{j}"]] * sgn[j]
            r[f"{side}_{j}_mean_deg"] = float(np.degrees(qq.mean()))
            if j == "ankle":
                r[f"{side}_ankle_max_deg"] = float(np.degrees(np.percentile(qq, 98)))
                r[f"{side}_ankle_min_deg"] = float(np.degrees(np.percentile(qq, 2)))
    if com_y is not None:
        r["dr_trunk_com_y_mm"] = float(com_y[e] * 1000)
    if enc is not None:
        r["dr_enc_hip_yaw_L_deg"] = float(np.degrees(enc["left_hip_yaw"][e]))
        r["dr_enc_hip_yaw_R_deg"] = float(np.degrees(-enc["right_hip_yaw"][e]))
    rows.append(r)

drift = np.array([r["drift"] for r in rows])
print(f"{a.ckpt.rsplit('/', 1)[-1]}  vx cmd {a.vx}: {len(rows)}/{a.n} upright; drift mean {drift.mean():+.3f} rad/s, "
      f"|drift| {np.abs(drift).mean():.3f}, robots turning right {np.mean(drift < 0):.0%}")
print("trunk yaw accumulated per phase (deg per s of walking; time share):")
print("   " + "   ".join(f"{ph} {np.mean([r[f'yaw_{ph}'] for r in rows]):+6.2f} ({np.mean([r[f'frac_{ph}'] for r in rows]):.0%})"
                         for ph in ("L_only", "R_only", "both", "none")))
summary = {}
print(f"{'metric (R mirrored)':22s} {'L':>8s} {'R':>8s} {'L-R':>8s}  corr(L-R, drift)")
for m in ("stance_s", "swing_s", "step_m", "width_m", "toe_out_deg", "pivot_deg", "push_Ns", "hip_yaw_mean_deg",
          "hip_roll_mean_deg", "hip_pitch_mean_deg", "knee_mean_deg", "ankle_mean_deg", "ankle_max_deg", "ankle_min_deg"):
    lv = np.array([r[f"L_{m}"] for r in rows]); rv = np.array([r[f"R_{m}"] for r in rows])
    good = np.isfinite(lv) & np.isfinite(rv)
    dlr = lv[good] - rv[good]
    cc = float(np.corrcoef(dlr, drift[good])[0, 1]) if good.sum() > 3 and dlr.std() > 0 else np.nan
    summary[m] = {"L": float(lv[good].mean()), "R": float(rv[good].mean()), "L_minus_R": float(dlr.mean()), "corr": cc}
    print(f"{m:22s} {lv[good].mean():8.3f} {rv[good].mean():8.3f} {dlr.mean():+8.3f}  {cc:+.2f}")
for m in ("dr_trunk_com_y_mm", "dr_enc_hip_yaw_L_deg", "dr_enc_hip_yaw_R_deg"):
    if m in rows[0]:
        v = np.array([r[m] for r in rows])
        cc = float(np.corrcoef(v, drift)[0, 1]) if v.std() > 0 else np.nan
        summary[m] = {"corr": cc, "std": float(v.std())}
        print(f"{m:22s} randomized draw, sd {v.std():.3f}: corr with drift {cc:+.2f}")
if "dr_enc_hip_yaw_L_deg" in rows[0]:
    v = np.array([r["dr_enc_hip_yaw_L_deg"] - r["dr_enc_hip_yaw_R_deg"] for r in rows])
    print(f"{'dr_enc_hip_yaw L-R':22s} corr with drift {np.corrcoef(v, drift)[0, 1]:+.2f}")
json.dump({"ckpt": a.ckpt, "task": a.task, "robot_chain": a.robot_chain, "vx": a.vx, "summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
