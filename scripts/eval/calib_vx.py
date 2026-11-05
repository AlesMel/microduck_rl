"""Forward-speed response of a checkpoint: fed vx command → achieved mean body vx, upright
count and open-loop yaw drift (play cfg, no pushes, wz command 0).

usage: uv run scripts/eval/calib_vx.py MODEL.pt OUT.json [--task ...] [--fed 0.1,...]
       [--per 16] [--seconds 12] [--settle 4] [--robot-chain] [--device cpu]

--robot-chain runs the policy through the real robot's action pipeline (scale 0.9 × 7.4/V,
target low-pass 0.7 legs / 0.5 head; see _common.RobotChain) instead of the training one.
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
p.add_argument("--fed", default="0.1,0.2,0.3,0.4,0.5")
p.add_argument("--per", type=int, default=16)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--settle", type=float, default=4.0)
p.add_argument("--robot-chain", action="store_true")
p.add_argument("--device", default="cpu")
a = p.parse_args()

policy = load_policy(a.ckpt, a.device)
fed = [float(f) for f in a.fed.split(",")]
n = len(fed) * a.per
env, raw = make_play_env(a.task, n, a.device)
robot = raw.scene["robot"]
chain = RobotChain(raw, n, a.device) if a.robot_chain else None
fed_t = torch.tensor(np.repeat(fed, a.per), dtype=torch.float32, device=a.device)
steps, settle = int(a.seconds / raw.step_dt), int(a.settle / raw.step_dt)
V, W, U = [], [], []
obs = env.get_observations()["actor"]
for k in range(steps):
    obs = step(env, policy, obs, fed_t if k >= 50 else 0.0, chain=chain)
    if k >= settle:
        V.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy().copy())
        W.append(robot.data.root_link_ang_vel_b[:, 2].cpu().numpy().copy())
        U.append((-robot.data.projected_gravity_b[:, 2]).cpu().numpy().copy())
V, W, U = np.stack(V, 1), np.stack(W, 1), np.stack(U, 1)
rows = []
for f in fed:
    idx = np.where(fed_t.cpu().numpy() == f)[0]
    ok = [e for e in idx if (U[e] > 0.8).all()]
    drift = W[ok].mean(1) if ok else np.array([])
    rows.append({"fed": f, "n_up": len(ok), "n": len(idx),
                 "vx": float(V[ok].mean()) if ok else None,
                 "vx_sd_across_robots": float(V[ok].mean(1).std()) if ok else None,
                 "yaw_drift_abs": float(np.abs(drift).mean()) if ok else None,
                 "yaw_drift_mean": float(drift.mean()) if ok else None})
    r = rows[-1]
    print(f"fed {f:.2f}: up {r['n_up']}/{r['n']}  vx {r['vx']:.3f} ± {r['vx_sd_across_robots']:.3f}  "
          f"drift {r['yaw_drift_mean']:+.3f} (|·| {r['yaw_drift_abs']:.3f}) rad/s"
          if ok else f"fed {f:.2f}: up 0/{len(idx)}")
json.dump({"ckpt": a.ckpt, "task": a.task, "robot_chain": a.robot_chain, "rows": rows}, open(a.out, "w"), indent=1)
