"""Are the down-edge spawns (mdp.reset_at_down_edge) of the stairs tasks physically sane?

Every robot on a downhill tile spawns at an edge in a bank pose (prob 1.0); a checkpoint (or zero
actions = hold the default pose) then runs ``--seconds``. Per spawned robot it reports the first
step's trunk vertical speed (a pop-out from interpenetration shows as a jump), the trunk pitch at
spawn and after 0.2 s, how many leg geoms touch the terrain at spawn, falls within the window, and
whether the trunk went down the riser (the spawn's job is to put the robot mid-step-down).

    uv run python scripts/eval/edge_spawn_check.py --checkpoint <up.pt> --device cuda:0

Stairs down chains several spawn events: ``--event`` picks the one under test (prob 1; the spawn
events chained after it are switched off), e.g. the double-support crouch at the edge:

    uv run python scripts/eval/edge_spawn_check.py --task Mjlab-StairsDown-Backlash-MicroDuck \
        --event reset_edge_crouch --hold --seconds 3 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_policy  # noqa: E402

from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks import microduck_terrains as mt  # noqa: E402

TASK = "Mjlab-StairsUp-Backlash-MicroDuck"


def _pitch_deg(g: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.asin(g[:, 0].clamp(-1, 1)))          # + = nose down


def ground0_pre(hf, robot):
    return hf.heights(robot.data.root_link_pos_w[:, :2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=TASK)
    p.add_argument("--checkpoint", default=None, help="none: zero actions (the default pose), or --hold")
    p.add_argument("--hold", action="store_true", help="no policy: hold each robot's spawn joint pose")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--event", default="reset_at_down_edge", help="the spawn event under test (prob 1)")
    a = p.parse_args()

    cfg = load_env_cfg(a.task, play=False)
    cfg.scene.num_envs = a.num_envs
    cfg.scene.terrain.max_init_terrain_level = None
    spawns = [k for k, e in cfg.events.items() if e.func is microduck_mdp.reset_at_down_edge]
    assert a.event in spawns, (a.event, spawns)
    for k in spawns[spawns.index(a.event) + 1:]:
        cfg.events[k].params["prob"] = 0.0                                   # nothing overrides the one under test
    cfg.events[a.event].params["prob"] = 1.0
    print("spawn events:", {k: cfg.events[k].params["prob"] for k in spawns})
    env = ManagerBasedRlEnv(cfg, device=a.device)
    robot = env.scene["robot"]
    names = list(cfg.scene.terrain.terrain_generator.sub_terrains)
    obs, _ = env.reset()
    spawned = env._edge_spawned.clone()
    types = env.scene.terrain.terrain_types
    print(f"spawned at an edge: {int(spawned.sum())} / {a.num_envs} "
          f"(down tiles: {int(sum((types == names.index(t)).sum() for t in mt.DOWN_TILES if t in names))})")

    hf = microduck_mdp.terrain_height_field(env)
    # the spawn state itself, before any step: each sole's gap to the ground under it, the CoM vs the feet
    env.sim.forward()
    mjm0 = env.sim.mj_model
    sole_z = microduck_mdp._SoleZ(env)(env)                                         # (N, 2)
    fg = [mjm0.geom(n).id for n in ("robot/left_foot_collision", "robot/right_foot_collision")]
    fxy = env.sim.data.geom_xpos[:, fg, :2]                                          # (N, 2, 2)
    gap = (sole_z - hf.heights(fxy.reshape(-1, 2)).reshape(-1, 2)) * 1000
    q = robot.data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
    root = [b for b in range(mjm0.nbody) if mjm0.body_parentid[b] == 0 and mjm0.body(b).name.startswith("robot/")][0]
    com = env.sim.data.subtree_com[:, root, :2]
    fx = (fxy * fwd[:, None, :]).sum(-1) * 1000                                      # along the heading, mm
    rear = fx.argmin(dim=1)
    ar = torch.arange(a.num_envs, device=a.device)
    cx = (com * fwd).sum(-1) * 1000
    g = lambda x: np.percentile(x[spawned].cpu().numpy(), [5, 50, 95]).round(1).tolist()  # noqa: E731
    print("spawn state (before any step), p5/50/95:")
    print(f"  rear sole gap mm {g(gap[ar, rear])}   front sole gap mm {g(gap[ar, 1 - rear])}")
    print(f"  front − rear foot centre mm {g(fx[ar, 1 - rear] - fx[ar, rear])}   CoM − rear foot centre mm {g(cx - fx[ar, rear])}")
    print(f"  root z − ground under the trunk mm {g((robot.data.root_link_pos_w[:, 2] - ground0_pre(hf, robot)) * 1000)}")

    # leg geoms touching the terrain right after the reset (before any step)
    mjm = env.sim.mj_model
    leg = {g for g in range(mjm.ngeom) if mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) and
           any(s in mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) for s in ("shin_collision", "thigh_collision"))}
    policy = load_policy(a.checkpoint, device=a.device) if a.checkpoint else None
    jt = env.action_manager.get_term("joint_pos")
    hold = ((robot.data.joint_pos[:, jt.target_ids] - jt._offset) / jt._scale).clone() if a.hold else None
    z0 = robot.data.root_link_pos_w[:, 2].clone()
    ground0 = hf.heights(robot.data.root_link_pos_w[:, :2])
    pitch0 = _pitch_deg(robot.data.projected_gravity_b)
    T = round(a.seconds / env.step_dt)
    done = torch.zeros(a.num_envs, dtype=torch.bool, device=a.device)
    fell = torch.zeros_like(done)
    vz1 = pitch02 = None
    leg_hits = torch.zeros(a.num_envs, device=a.device)
    min_z_rel = torch.full((a.num_envs,), 9.0, device=a.device)
    max_tilt = torch.zeros(a.num_envs, device=a.device)
    with torch.no_grad():
        for t in range(T):
            if policy is not None:
                act = policy(obs["actor"])
            elif hold is not None:
                act = hold
            else:
                act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=a.device)
            obs, _, term, trunc, _ = env.step(act)
            live = ~done
            if t == 0:
                vz1 = robot.data.root_link_lin_vel_w[:, 2].clone()
                # leg ↔ terrain contacts after the first step, from the GPU data (contact geoms)
                d = env.sim.data
                try:
                    geom = d.contact.geom.cpu().numpy()
                    wid = d.contact.worldid.cpu().numpy()
                    n = int(d.nacon.cpu().numpy()[0]) if hasattr(d, "nacon") else len(geom)
                    for k in range(n):
                        g1, g2 = geom[k]
                        if (g1 in leg) != (g2 in leg):
                            leg_hits[wid[k]] += 1
                except Exception as e:  # noqa: BLE001
                    print("contact read failed:", e)
            if t == round(0.2 / env.step_dt) - 1:
                pitch02 = _pitch_deg(robot.data.projected_gravity_b)
                if hold is not None:                                       # servo sag: |q − target| per joint
                    err = (robot.data.joint_pos[:, jt.target_ids] - (hold * jt._scale + jt._offset)).abs()
                    names_t = jt.target_names
                    sp = spawned & ~done
                    for jn in ("left_hip_pitch", "left_knee", "left_ankle", "right_hip_pitch", "right_knee", "right_ankle"):
                        e = np.rad2deg(err[sp, names_t.index(jn)].cpu().numpy())
                        print(f"  sag at 0.2 s {jn:>16}: p50 {np.median(e):5.1f}°  p95 {np.percentile(e, 95):5.1f}°")
            z_rel = robot.data.root_link_pos_w[:, 2] - ground0
            min_z_rel = torch.where(live, torch.minimum(min_z_rel, z_rel), min_z_rel)
            tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))
            max_tilt = torch.where(live, torch.maximum(max_tilt, tilt), max_tilt)
            fell |= live & term
            done |= term | trunc
    m = spawned
    zt = robot.data.root_link_pos_w[:, 2]
    went_down = (zt < z0 - 0.01) & ~done          # still in the first episode and lower than at spawn
    f = lambda x: np.percentile(x[m].cpu().numpy(), [5, 50, 95]).round(3).tolist()  # noqa: E731
    print(f"checkpoint: {a.checkpoint or ('hold the spawn pose' if a.hold else 'zero actions')}, {a.seconds:.1f} s")
    print(f"  first-step trunk vz m/s p5/50/95: {f(vz1)}  (|vz| > 0.3: {int(((vz1.abs() > 0.3) & m).sum())})")
    print(f"  trunk pitch at spawn deg (− = nose up): {f(pitch0)}   after 0.2 s: {f(pitch02)}")
    print(f"  trunk z above the spawn ground, min over the window, m: {f(min_z_rel)}")
    alive = m & ~done
    print(f"  max trunk tilt over the window deg (alive at the end): "
          f"{np.percentile(max_tilt[alive].cpu().numpy(), [5, 50, 95, 99]).round(1).tolist() if alive.any() else '-'}"
          f"   tilt > 20°: {int(((max_tilt > 20) & m).sum())}")
    print(f"  trunk z end − spawn mm (alive): "
          f"{np.percentile(((zt - z0) * 1000)[alive].cpu().numpy(), [5, 50, 95]).round(1).tolist() if alive.any() else '-'}")
    print(f"  leg-geom terrain contacts after step 1: envs with any {int(((leg_hits > 0) & m).sum())}")
    print(f"  fell within {a.seconds:.1f} s: {100 * float((fell & m).sum() / m.sum()):.1f} %"
          f"   trunk ≥ 1 cm below spawn at the end (alive): {100 * float((went_down & m).sum() / m.sum()):.1f} %")


if __name__ == "__main__":
    main()
