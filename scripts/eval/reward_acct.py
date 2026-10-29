"""Counterfactual reward accounting: score a policy's gait under a task's reward.

The policy is fed a forward command (--fed, a sweep, so it really walks at different speeds),
but every reward term is scored at the TRUE command (--true), written into the env's command
manager before each step. Per-term weighted per-step values (reward_manager._step_reward,
not dt-scaled) are averaged over upright robots after settling, next to gait stats (achieved
vx, swing time, stride frequency).

Weights: the play cfg never runs curricula, so curriculum-weighted terms would sit at stage 0
(action_rate −0.1 instead of −1.0, ...). Terms are scored with the TRAIN cfg's weights, with
reward_weight curricula at their final stage.

Probes: a checkpoint, --zero-action (stand still at home pose), and --mirror (the policy's
left-right mirror image; a symmetric reward scores it the same).

usage: uv run scripts/eval/reward_acct.py MODEL.pt OUT.json [--task ...] [--true 0.4]
       [--fed 0.4,0.5,0.6] [--per 8] [--seconds 12] [--settle 4] [--mirror]
       uv run scripts/eval/reward_acct.py --zero-action OUT.json [--true 0.1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_policy, make_play_env, mirror_policy, step, zero_policy  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ckpt", nargs="?", help="checkpoint (omit with --zero-action)")
p.add_argument("out")
p.add_argument("--task", default="Mjlab-WalkV2-Rough-Backlash-MicroDuck")
p.add_argument("--true", type=float, default=0.4)
p.add_argument("--fed", default=None, help="default: the true command only")
p.add_argument("--per", type=int, default=8)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--settle", type=float, default=4.0)
p.add_argument("--zero-action", action="store_true")
p.add_argument("--mirror", action="store_true")
p.add_argument("--device", default="cpu")
a = p.parse_args()
if a.zero_action == bool(a.ckpt):
    p.error("give a checkpoint or --zero-action, not both")

policy = zero_policy if a.zero_action else load_policy(a.ckpt, a.device)
if a.mirror:
    policy = mirror_policy(policy)
fed = [float(f) for f in (a.fed or str(a.true)).split(",")]

train = load_env_cfg(a.task)
final = {c.params["reward_name"]: c.params["weight_stages"][-1]["weight"]
         for c in train.curriculum.values() if "weight_stages" in c.params and "reward_name" in c.params}


def hook(cfg):
    for n, term in cfg.rewards.items():
        term.weight = final.get(n, train.rewards[n].weight)


env, raw = make_play_env(a.task, len(fed) * a.per, a.device, cfg_hook=hook)
robot = raw.scene["robot"]
rm, cm = raw.reward_manager, raw.command_manager
names = list(rm.active_terms)
sensor = raw.scene.sensors["feet_ground_contact"]
fed_t = torch.tensor(np.repeat(fed, a.per), dtype=torch.float32, device=a.device)
twist = cm.get_term("twist")
others = [n for n in cm.active_terms if n != "twist"]


def pin_commands():
    twist.vel_command_b[:] = torch.tensor([a.true, 0.0, 0.0], device=a.device)
    for flag in ("is_standing_env", "is_heading_env"):
        if hasattr(twist, flag):
            getattr(twist, flag)[:] = False
    for n in others:
        cm.get_command(n).zero_()


steps, settle = int(a.seconds / raw.step_dt), int(a.settle / raw.step_dt)
R, C, V, U = [], [], [], []
obs = env.get_observations()["actor"]
for k in range(steps):
    pin_commands()
    obs = step(env, policy, obs, fed_t if k >= 50 else 0.0)
    if k >= settle:
        R.append(rm._step_reward.clone().cpu().numpy())
        C.append((sensor.data.found > 0).cpu().numpy().copy())
        V.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy().copy())
        U.append((-robot.data.projected_gravity_b[:, 2]).cpu().numpy().copy())
R, C, V, U = np.stack(R, 1), np.stack(C, 1).astype(int), np.stack(V, 1), np.stack(U, 1)
dt = raw.step_dt


def swings(c):
    out, s = [], 0
    for t in range(1, len(c) + 1):
        if t == len(c) or c[t] != c[s]:
            if c[s] == 0 and s > 0 and t < len(c):
                out.append((t - s) * dt)
            s = t
    return out


def is_penalty(name):
    """Both penalty styles (AGENTS.md): mjlab costs (≥ 0, negative weight) and self-negating
    microduck terms (`*_penalty` / `*_l1`, ≤ 0, positive weight). Either must log ≤ 0."""
    t = train.rewards[name]
    fn = getattr(t.func, "__name__", type(t.func).__name__)
    return t.weight < 0 or fn.endswith(("_penalty", "_l1"))


res = {"task": a.task, "ckpt": a.ckpt, "zero_action": a.zero_action, "mirror": a.mirror,
       "true": a.true, "terms": names, "rows": []}
for f in fed:
    idx = np.where(fed_t.cpu().numpy() == f)[0]
    ok = [e for e in idx if (U[e] > 0.8).all()]
    sw = [s for e in ok for foot in range(C.shape[2]) for s in swings(C[e, :, foot])]
    tds = [np.diff(np.where(np.diff(C[e, :, 0]) == 1)[0]).mean() * dt for e in ok if (np.diff(C[e, :, 0]) == 1).sum() > 2]
    row = {"fed": f, "n_up": len(ok), "n": len(idx), "vx": float(V[ok].mean()) if ok else None,
           "swing": float(np.mean(sw)) if sw else None, "stride_hz": float(1 / np.mean(tds)) if tds else None,
           "terms": {n: float(R[ok][:, :, i].mean()) for i, n in enumerate(names)} if ok else {}}
    row["total"] = float(sum(row["terms"].values())) if ok else None
    row["penalties_le_0"] = all(v <= 1e-9 for n, v in row["terms"].items() if is_penalty(n)) if ok else None
    res["rows"].append(row)
json.dump(res, open(a.out, "w"), indent=1)

rows = [r for r in res["rows"] if r["n_up"]]
probe = "zero action" if a.zero_action else a.ckpt.rsplit("/", 1)[-1]
print(f"{probe}{' (mirrored)' if a.mirror else ''} on {a.task}, scored at TRUE vx = {a.true}; weighted per-step values")
if not rows:
    print("no robot stayed upright")
    sys.exit(0)
print(f"{'fed':>22s}" + "".join(f"{r['fed']:>9.2f}" for r in rows))
for lab, key, fmt in (("upright envs", "n_up", "{:>9d}"), ("achieved vx", "vx", "{:>9.3f}"),
                      ("stride Hz", "stride_hz", "{:>9.2f}"), ("swing s", "swing", "{:>9.3f}"),
                      ("TOTAL", "total", "{:>9.3f}")):
    print(f"{lab:>22s}" + "".join(fmt.format(r[key]) if r[key] is not None else f"{'-':>9s}" for r in rows))
order = sorted(names, key=lambda n: -max(abs(r["terms"][n]) for r in rows))
for n in order:
    vals = [r["terms"][n] for r in rows]
    if max(abs(v) for v in vals) < 1e-4:
        continue
    print(f"{n:>22s}" + "".join(f"{v:>9.3f}" for v in vals))
print(f"{'penalties all <= 0':>22s}" + "".join(f"{str(r['penalties_le_0']):>9s}" for r in rows))
