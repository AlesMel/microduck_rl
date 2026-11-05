"""Regression battery: walk falls/speed, or get-up from fallen spawns.

walk: play cfg on --terrain, N robots per forward command, no pushes: mean body vx after
  settling (robots that never fell), % of robots that ever fell (tilt > 40°), % upright at
  the end. --robot-chain runs the real robot's action pipeline (see _common.RobotChain).
getup: per spawn family, N robots spawned fallen through velstand's own reset event
  (random_prone_init) on --getup-task, zero command, fallen_too_long/time_out terminations
  removed so a robot that never rises stays down. recovered = tilt < 25° and trunk z > 0.09 m
  held for the last 0.5 s; t_rec = first time recovered; |w| p50 = median trunk angular speed
  while fallen. Needs a task with random_prone_init (the velstand family), not WalkV2.

usage: uv run scripts/eval/battery.py MODEL.pt [MODEL.pt ...] --mode walk|getup [--per 32]
       [--family WalkV2] [--terrain rough|flat] [--cmds 0,0.2,0.3,0.4] [--robot-chain]
       [--getup-task Mjlab-VelStand-Flat-Backlash-MicroDuck] [--device cuda:0]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import RobotChain, load_policy, make_play_env, step  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="+")
p.add_argument("--mode", choices=("getup", "walk"), required=True)
p.add_argument("--per", type=int, default=32)
p.add_argument("--device", default="cuda:0")
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--family", default="WalkV2", help="walk task = Mjlab-<family>-<Terrain>-Backlash-MicroDuck")
p.add_argument("--terrain", choices=("rough", "flat"), default="rough")
p.add_argument("--cmds", default="0.0,0.2,0.3,0.4")
p.add_argument("--robot-chain", action="store_true")
p.add_argument("--getup-task", default="Mjlab-VelStand-Flat-Backlash-MicroDuck")
a = p.parse_args()

FAMILIES = {  # random_prone_init params per family
    "home face-down": dict(prone_prob=1.0, face_down_prob=1.0, side_prob=0.0, joint_random_prob=0.0, so3_prob=0.0, sit_prob=0.0),
    "home face-up":   dict(prone_prob=1.0, face_down_prob=0.0, side_prob=0.0, joint_random_prob=0.0, so3_prob=0.0, sit_prob=0.0),
    "home side":      dict(prone_prob=1.0, face_down_prob=0.5, side_prob=1.0, joint_random_prob=0.0, so3_prob=0.0, sit_prob=0.0),
    "rj face-down":   dict(prone_prob=1.0, face_down_prob=1.0, side_prob=0.0, joint_random_prob=1.0, so3_prob=0.0, sit_prob=0.0),
    "rj face-up":     dict(prone_prob=1.0, face_down_prob=0.0, side_prob=0.0, joint_random_prob=1.0, so3_prob=0.0, sit_prob=0.0),
    "so3 random":     dict(prone_prob=1.0, face_down_prob=0.5, side_prob=0.0, joint_random_prob=1.0, so3_prob=1.0, sit_prob=0.0),
    "sit":            dict(prone_prob=0.0, face_down_prob=0.5, side_prob=0.0, joint_random_prob=0.0, so3_prob=0.0, sit_prob=1.0),
}


def make_env(task, n, prone=None):
    def hook(cfg):
        for t in ("fallen_too_long", "time_out"):
            cfg.terminations.pop(t, None)
        # The prone_init_prob curriculum rewrites the spawn mix at every reset (play cfg keeps it).
        cfg.curriculum.pop("prone_init_prob", None)
    env, raw = make_play_env(task, n, a.device, cfg_hook=hook)
    if prone is not None:
        # managers deepcopy their cfg: mutate the live term, then reset
        raw.event_manager.get_term_cfg("random_prone_init").params.update(prone)
        raw.reset(env_ids=torch.arange(n, device=a.device))
        tilt = -raw.scene["robot"].data.projected_gravity_b[:, 2]
        if not prone["so3_prob"] and not prone["sit_prob"]:  # uniform SO3 can land near upright; sitting is ≤ 20° tilt
            assert (tilt < 0.9).float().mean() > 0.9, f"spawns not fallen: cos tilt {tilt.tolist()}"
    return env, raw


def run(env, raw, policy, vx, steps, chain=None):
    robot = raw.scene["robot"]
    obs = env.get_observations()["actor"]
    cos_t, z, w, v = [], [], [], []
    for _ in range(steps):
        obs = step(env, policy, obs, vx, chain=chain)
        d = robot.data
        cos_t.append((-d.projected_gravity_b[:, 2]).cpu().numpy())
        z.append((d.root_link_pos_w[:, 2] - raw.scene.terrain.env_origins[:, 2]).cpu().numpy())
        w.append(d.root_link_ang_vel_b.norm(dim=1).cpu().numpy())
        v.append(d.root_link_lin_vel_b[:, 0].cpu().numpy())
    return np.stack(cos_t, 1), np.stack(z, 1), np.stack(w, 1), np.stack(v, 1), raw.step_dt


policies = {c.rsplit("/", 1)[-1]: load_policy(c, a.device) for c in a.ckpts}
if a.mode == "getup":
    up_cos = math.cos(math.radians(25.0))
    print(f"{'family':16s}" + "".join(f"{name:>34s}" for name in policies))
    print(f"{'':16s}" + "".join(f"{'rec% / stuck% / t_rec p50 / |w| p50':>34s}" for _ in policies))
    for fam, prone in FAMILIES.items():
        row = f"{fam:16s}"
        for name, pol in policies.items():
            env, raw = make_env(a.getup_task, a.per, prone)
            c, z, w, _, dt = run(env, raw, pol, 0.0, int(a.seconds / 0.02))
            env.close()
            up = (c > up_cos) & (z > 0.09)
            rec_end = up[:, -25:].all(1)
            t_rec = np.array([np.argmax(u) * dt if u.any() else np.nan for u in up])
            started_fallen = ~up[:, 0]
            fallen_w = w[~up]
            row += (f"{100 * rec_end.mean():>12.0f} /{100 * (~rec_end).mean():>5.1f} /"
                    f"{np.nanmedian(t_rec[started_fallen]) if started_fallen.any() else float('nan'):>6.2f}s /"
                    f"{np.median(fallen_w) if fallen_w.size else float('nan'):>5.1f}")
        print(row, flush=True)
else:
    task = f"Mjlab-{a.family}-{a.terrain.capitalize()}-Backlash-MicroDuck"
    fall_cos = math.cos(math.radians(40.0))
    print(f"{task}{'  [robot chain]' if a.robot_chain else ''}")
    print(f"{'cmd vx':8s}" + "".join(f"{name:>36s}" for name in policies))
    print(f"{'':8s}" + "".join(f"{'vx m/s / fell % / up at end %':>36s}" for _ in policies))
    for cmd in (float(x) for x in a.cmds.split(",")):
        row = f"{cmd:<8.2f}"
        for name, pol in policies.items():
            env, raw = make_env(task, a.per)
            chain = RobotChain(raw, a.per, a.device) if a.robot_chain else None
            c, z, w, vx, dt = run(env, raw, pol, cmd, int(a.seconds / 0.02), chain)
            env.close()
            settle = int(4.0 / dt)
            fell = (c < fall_cos).any(1)
            ok = ~fell
            v = vx[ok, settle:].mean() if ok.any() else float("nan")
            row += f"{v:>17.3f} /{100 * fell.mean():>6.1f} /{100 * (c[:, -1] > fall_cos).mean():>6.1f}"
        print(row, flush=True)
