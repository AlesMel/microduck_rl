"""Show a terrain task with the microduck on it — no training.

Builds the task's PLAY cfg, resets every robot onto its tile (one column per terrain type,
rows = difficulty) and either opens mjlab's native MuJoCo viewer, renders a screenshot, or
records a video. Without a policy the robots hold a zero action (the HOME pose). NOTE: on this
robot a zero action is NOT a balanced stance — the low-gain BAM servos sag within a second and
the trunk leans 15-40° even on flat ground (a walking policy balances actively). So the viewer
re-poses the robots every ``--repose-s`` seconds (0 disables) and ``--settle`` judges each
spawn at ``--settle-s`` (default 0.3 s: feet have found the ground, nothing has sagged): trunk
at its reset height above the tile origin and tilt < 10° means the spawn sits on solid ground,
not on a step edge or inside a pit.

With a policy — ``--onnx policy.onnx`` (a published walk ONNX, normalizer baked in) or
``--checkpoint model_XXXX.pt`` (a walk-v2 rsl_rl checkpoint) — every robot is spawned facing
+x and walks at ``--vx``: straight into the small steps on the straight_stairs column.

``--video out.mp4`` records ``--video-s`` seconds: left, a fixed view of the straight-stairs
column; right, a camera following the robot on its hardest row. ``--save-states f.npz`` keeps
the rollout (all robots' qpos per frame) and ``--replay f.npz`` renders it on another machine
(same task / num-envs / seed rebuild the same terrain) — for a GPU box with no working EGL.

usage:
    uv run mjpython scripts/view_terrain_env.py                              # viewer
    uv run mjpython scripts/view_terrain_env.py --task Mjlab-WalkV2-Stairs-MicroDuck
    uv run python scripts/view_terrain_env.py --settle --screenshot out.png --no-viewer
    MUJOCO_GL=egl uv run python scripts/view_terrain_env.py --task Mjlab-WalkV2-Stairs-MicroDuck \
        --onnx walk_v2_model_5000.onnx --device cuda --video stairs.mp4 --no-viewer

``mjpython`` (in the venv) is what macOS needs for MuJoCo's passive viewer; ``--no-viewer``
runs (settle, screenshot, video) work under plain ``python`` too.

In the viewer: scroll zooms, left-drag orbits, right-drag pans; [ and ] switch the tracked
robot; mjlab's key help (F1) lists the "show all envs" toggle that draws every robot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, str(Path(__file__).parent / "eval"))
from _common import command_obs, load_policy

DEFAULT_TASK = "Mjlab-WalkV2-Stairs-MicroDuck"
NUM_ACTIONS = 14
STAIRS = "straight_stairs"
STEP_TILES = (STAIRS, "single_step", "short_flight")   # tiles with a step profile worth a side camera


def _tile_name(names: list[str]) -> str | None:
    """The step tile of this menu (stairs grids: the straight stairs; step-up grids: the single step)."""
    return next((t for t in STEP_TILES if t in names), None)


def build_env(task: str, num_envs: int, seed: int, device: str = "cpu",
              fixed_heading: bool = False, cfg_hook=None) -> ManagerBasedRlEnv:
    """``cfg_hook(cfg)``: an optional edit of the play cfg before the env is built (e.g. other riser heights)."""
    cfg = load_env_cfg(task, play=True)
    if cfg_hook is not None:
        cfg_hook(cfg)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 1e6                       # no time-out resets while we look
    for name in list(cfg.events):                    # no pushes: we want to see the ground
        if "push" in name:
            cfg.events.pop(name)
    # mjlab's promotion rule demotes a robot that did not walk half a tile — i.e. every robot
    # at the first reset here. Drop it so the preview keeps robots spread over all rows.
    cfg.curriculum.pop("terrain_levels", None)
    if fixed_heading:                                # walk along +x, into the straight stairs
        cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    gen = cfg.scene.terrain.terrain_generator
    if gen is not None and gen.seed is None:
        gen.seed = seed
    return ManagerBasedRlEnv(cfg, device=device)


def _actor(obs) -> torch.Tensor:
    return obs["actor"] if hasattr(obs, "keys") else obs


class HomePolicy:
    """Zero action (= HOME pose), re-posing every ``repose_steps`` env steps (0: never)."""

    def __init__(self, env: ManagerBasedRlEnv, repose_steps: int):
        self.env, self.repose_steps, self.t = env, repose_steps, 0

    def __call__(self, obs) -> torch.Tensor:
        self.t += 1
        if self.repose_steps and self.t % self.repose_steps == 0:
            self.env.reset()
        o = _actor(obs)
        return torch.zeros(o.shape[0], NUM_ACTIONS, device=o.device)


class WalkPolicy:
    """A walking policy driven through its observation: constant twist, head/body zero.
    ``onnx``: a published ONNX (normalizer baked in, 61D in → 14D out); else an rsl_rl .pt."""

    def __init__(self, path: str, vx: float, vy: float, wz: float, onnx: bool, device: str = "cpu"):
        self.cmd = (vx, vy, wz)
        if onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.in_name = self.session.get_inputs()[0].name
            self.out_name = self.session.get_outputs()[0].name
            self.policy = None
        else:
            self.session, self.policy = None, load_policy(path, device=device)

    def __call__(self, obs) -> torch.Tensor:
        o = command_obs(_actor(obs), *self.cmd)
        if self.policy is not None:
            return self.policy(o)
        # Published ONNX files are exported with a fixed batch of 1: one run per robot.
        x = o.cpu().numpy().astype(np.float32)
        a = np.concatenate([self.session.run([self.out_name], {self.in_name: x[i:i + 1]})[0] for i in range(x.shape[0])])
        return torch.as_tensor(a, device=o.device)


def settle_report(env: ManagerBasedRlEnv, policy, seconds: float) -> dict[str, dict]:
    """Reset, run ``policy`` for ``seconds`` and report per terrain type: robots upright
    (tilt < 10°), max tilt, trunk height above the tile origin (reset puts it at 0.12-0.13 m;
    0.115 m once the feet carry the weight; far below = fell through or off a step)."""
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    types = terrain.terrain_types.cpu().numpy()
    levels = terrain.terrain_levels.cpu().numpy()
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(round(seconds / env.step_dt)):
            obs, *_ = env.step(policy(obs))
    up = -robot.data.projected_gravity_b[:, 2].cpu().numpy()           # cos(tilt)
    tilt_deg = np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))
    z_rel = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
    report = {}
    for t, name in enumerate(names):
        m = types == t
        if not m.any():
            continue
        report[name] = {
            "robots": int(m.sum()),
            "rows": sorted({int(r) for r in levels[m]}),
            "upright": int((tilt_deg[m] < 10.0).sum()),
            "max_tilt_deg": float(tilt_deg[m].max()),
            "z_rel_min": float(z_rel[m].min()),
            "z_rel_max": float(z_rel[m].max()),
        }
    return report


# ── rendering ────────────────────────────────────────────────────────────────

class GridRenderer:
    """Offscreen render of the whole grid with every robot: one robot is the MjData, the
    others are added as extra scene geoms, as mjlab's native viewer does in "show all envs"."""

    def __init__(self, env: ManagerBasedRlEnv, width: int, height: int):
        self.env = env
        self.mjm = env.sim.mj_model
        self.mjd = mujoco.MjData(self.mjm)
        self.extra = mujoco.MjData(self.mjm)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.cat = mujoco.mjtCatBit.mjCAT_DYNAMIC
        self.mjm.vis.global_.offwidth = max(self.mjm.vis.global_.offwidth, width)
        self.mjm.vis.global_.offheight = max(self.mjm.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(self.mjm, height=height, width=width, max_geom=200_000)
        self.nq = self.mjm.nq

    def qpos_now(self) -> np.ndarray:
        return self.env.sim.data.qpos.cpu().numpy().copy()        # (N, nq)

    def _load(self, mjd: mujoco.MjData, qpos: np.ndarray) -> None:
        mjd.qpos[:] = qpos
        mujoco.mj_forward(self.mjm, mjd)

    def render(self, qpos_all: np.ndarray, focus: int, lookat, distance, azimuth, elevation) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        self._load(self.mjd, qpos_all[focus])
        self.renderer.update_scene(self.mjd, cam, self.opt)
        for i in range(qpos_all.shape[0]):
            if i != focus:
                self._load(self.extra, qpos_all[i])
                mujoco.mjv_addGeoms(self.mjm, self.extra, self.opt, self.pert, self.cat, self.renderer.scene)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _terrain_layout(env: ManagerBasedRlEnv):
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    types = terrain.terrain_types.cpu().numpy()
    levels = terrain.terrain_levels.cpu().numpy()
    origins = terrain.terrain_origins.cpu().numpy()                 # (rows, cols, 3)
    return names, types, levels, origins


def _focus_robot(env: ManagerBasedRlEnv) -> int:
    """The robot on the straight stairs' hardest occupied row (else robot 0)."""
    names, types, levels, _ = _terrain_layout(env)
    tile = _tile_name(names)
    if tile is not None:
        on = np.nonzero(types == names.index(tile))[0]
        if len(on):
            return int(on[np.argmax(levels[on])])
    return 0


def screenshot(env: ManagerBasedRlEnv, path: str, width: int = 1600, height: int = 1000) -> None:
    r = GridRenderer(env, width, height)
    origins = _terrain_layout(env)[3]
    q = r.qpos_now()
    flat = origins.reshape(-1, 3)
    extent = float(np.ptp(flat[:, :2], axis=0).max()) + 4.0
    wide = r.render(q, 0, flat.mean(axis=0), 0.9 * extent, 150.0, -38.0)
    focus = _focus_robot(env)
    close = r.render(q, focus, q[focus, :3] + np.array([0.5, 0.0, -0.05]), 1.6, 160.0, -22.0)
    r.close()
    import imageio.v2 as imageio
    imageio.imwrite(path, np.concatenate([wide, close], axis=0))
    print(f"screenshot → {path}")


def rollout(env: ManagerBasedRlEnv, policy, seconds: float, fps: int) -> dict:
    """Run ``policy`` for ``seconds`` from a reset; keep every robot's qpos at ``fps`` plus the
    per-robot outcome on the stairs (x progress, min upright cos, falls)."""
    robot = env.scene["robot"]
    nq = env.sim.mj_model.nq
    stride = max(1, round(1.0 / (fps * env.step_dt)))
    obs, _ = env.reset()
    x0 = robot.data.root_link_pos_w[:, 0].clone()
    frames, up_min, falls, x_max = [], None, None, None
    n_steps = round(seconds / env.step_dt)
    with torch.no_grad():
        for k in range(n_steps):
            if k % stride == 0:
                frames.append(env.sim.data.qpos.cpu().numpy().reshape(-1, nq).copy())
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            dones = terminated | truncated
            up = -robot.data.projected_gravity_b[:, 2]
            x = robot.data.root_link_pos_w[:, 0] - x0
            up_min = up if up_min is None else torch.minimum(up_min, up)
            x_max = x if x_max is None else torch.maximum(x_max, x)
            d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones).bool()
            falls = d.long() if falls is None else falls + d.long()
    names, types, levels, _ = _terrain_layout(env)
    return {
        "qpos": np.stack(frames),                                   # (T, N, nq)
        "fps": fps, "dt": env.step_dt * stride,
        "names": names, "types": types, "levels": levels,
        "env_origins": env.scene.env_origins.cpu().numpy(),
        "x_max": x_max.cpu().numpy(), "up_min": up_min.cpu().numpy(), "falls": falls.cpu().numpy(),
    }


def outcome_lines(states: dict, env: ManagerBasedRlEnv) -> list[str]:
    """Per-stairs-robot summary. The straight stairs start 1.0 m past the tile's left edge
    (spawn at 0.5 m): x progress > 0.5 m = reached the first step, > 1.1 m = on the landing,
    > 1.7 m = down the far flight, > 2.9 m = on the end platform (walk-v2 cfg, 4 steps)."""
    names, types, levels = states["names"], states["types"], states["levels"]
    gen = env.scene.terrain.cfg.terrain_generator
    lines = []
    if STAIRS not in names:
        return lines
    st = gen.sub_terrains[STAIRS]
    for i in np.nonzero(types == names.index(STAIRS))[0]:
        r = int(levels[i])
        h_lo, h_hi = st.step_height(r / gen.num_rows), st.step_height((r + 1) / gen.num_rows)
        x = float(states["x_max"][i])
        stage = ("end platform" if x > 2.9 else "down the far flight" if x > 1.7 else
                 "on the landing" if x > 1.1 else "on the first flight" if x > 0.5 else "never reached the steps")
        lines.append(f"robot {i:2d} row {r} (steps {100*h_lo:.1f}-{100*h_hi:.1f} cm): "
                     f"x progress {x:4.2f} m → {stage}; min upright cos {states['up_min'][i]:.2f}; "
                     f"falls/resets {int(states['falls'][i])}")
    return lines


def _stairs_robots(states: dict) -> list[int]:
    """Robots on the straight stairs, easiest row first (ties: lowest id)."""
    names, types, levels = states["names"], states["types"], states["levels"]
    tile = _tile_name(names)
    if tile is None:
        return [0]
    on = np.nonzero(types == names.index(tile))[0]
    return [int(i) for i in sorted(on, key=lambda i: (levels[i], i))] or [0]


def render_video(env: ManagerBasedRlEnv, states: dict, path: str, label: str,
                 width: int = 1600, height: int = 720) -> None:
    """Two side-view cameras (robot walks left → right, the step profile in view): the stairs
    robot on the easiest occupied row and the one on the hardest row."""
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    # Layout from the recorded rollout: the robot → tile draw is a torch RNG draw and differs
    # between the cpu and cuda generators, so a replay must not re-derive it from this env.
    levels = states["levels"]
    env_origins = states["env_origins"]
    gen = env.scene.terrain.cfg.terrain_generator
    st = gen.sub_terrains.get(_tile_name(states["names"]) or "")
    robots = _stairs_robots(states)
    panes = [robots[0], robots[-1]] if len(robots) > 1 else [robots[0]]
    pane_w = width // len(panes)
    r = GridRenderer(env, pane_w, height)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

    def pane_title(i: int) -> str:
        row = int(levels[i])
        if st is None:
            return f"robot {i}, row {row}"
        return (f"robot {i}, row {row}/{gen.num_rows - 1}: steps "
                f"{100 * st.step_height(row / gen.num_rows):.1f}-{100 * st.step_height((row + 1) / gen.num_rows):.1f} cm")

    writer = imageio.get_writer(path, fps=states["fps"], codec="libx264", quality=8, macro_block_size=8)
    qpos = states["qpos"]
    for k in range(qpos.shape[0]):
        q = qpos[k]
        imgs = []
        for i in panes:
            p = q[i, :3]
            # Camera on the -y side looking +y: +x (the walking direction) runs left → right.
            img = r.render(q, i, p + np.array([0.15, 0.0, -0.03]), 1.0, 90.0, -10.0)
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, im.width, 34), fill=(0, 0, 0))
            d.text((10, 7), f"{pane_title(i)}   x +{p[0] - env_origins[i, 0]:4.2f} m", fill=(255, 255, 255), font=font)
            imgs.append(np.asarray(im))
        frame = Image.fromarray(np.concatenate(imgs, axis=1))
        d = ImageDraw.Draw(frame)
        d.rectangle((0, frame.height - 34, frame.width, frame.height), fill=(0, 0, 0))
        d.text((10, frame.height - 27), f"{label}   t = {k * states['dt']:5.2f} s   {env.cfg.__class__.__name__.replace('Cfg', '')}",
               fill=(255, 220, 120), font=font)
        writer.append_data(np.asarray(frame))
    writer.close()
    r.close()
    print(f"video → {path}  ({qpos.shape[0]} frames @ {states['fps']} fps, robots {panes})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--num-envs", type=int, default=28, help="robots (spread over all tiles)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--onnx", default=None, help="published walk ONNX: walk instead of holding HOME")
    p.add_argument("--checkpoint", default=None, help="walk-v2 rsl_rl .pt: walk instead of holding HOME")
    p.add_argument("--vx", type=float, default=0.3, help="m/s along +x with a policy")
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--repose-s", type=float, default=3.0, help="HOME mode: re-pose period in the viewer (0: never)")
    p.add_argument("--settle", action="store_true", help="report the stance per terrain type")
    p.add_argument("--settle-s", type=float, default=0.3)
    p.add_argument("--screenshot", default=None, help="write a PNG (wide grid + close-up)")
    p.add_argument("--video", default=None, help="record an MP4 of a rollout")
    p.add_argument("--video-s", type=float, default=14.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--save-states", default=None, help="dump the rollout (qpos per frame) to .npz")
    p.add_argument("--replay", default=None, help="render --video from a saved .npz instead of simulating")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    has_policy = bool(args.onnx or args.checkpoint)
    env = build_env(args.task, args.num_envs, args.seed, args.device, fixed_heading=has_policy)
    gen = env.scene.terrain.cfg.terrain_generator
    print(f"{args.task}: {len(gen.sub_terrains)} terrain types × {gen.num_rows} rows, "
          f"{gen.size[0]} m tiles, {env.sim.mj_model.ngeom} geoms, {args.num_envs} robots, {args.device}")
    if has_policy:
        path = args.onnx or args.checkpoint
        policy = WalkPolicy(path, args.vx, args.vy, args.wz, onnx=bool(args.onnx), device=args.device)
        settle_policy = policy
        label = f"{Path(path).name}  vx {args.vx:+.2f} m/s"
    else:
        policy = HomePolicy(env, round(args.repose_s / env.step_dt))
        settle_policy = HomePolicy(env, 0)
        label = "HOME (zero action)"
    env.reset()

    if args.settle:
        rep = settle_report(env, settle_policy, args.settle_s)
        print(f"\nstance after {args.settle_s:.1f} s ({label}):")
        for name, r in rep.items():
            ok = r["upright"] == r["robots"] and 0.10 < r["z_rel_min"] and r["z_rel_max"] < 0.14
            print(f"  {'OK ' if ok else 'BAD'} {name:20s} {r['upright']:2d}/{r['robots']:<2d} upright  max tilt {r['max_tilt_deg']:5.1f}°  "
                  f"trunk z above origin {r['z_rel_min']:.3f}-{r['z_rel_max']:.3f} m  rows {r['rows']}")
        env.reset()

    if args.screenshot:
        screenshot(env, args.screenshot)

    states = None
    if args.replay:
        z = np.load(args.replay, allow_pickle=True)
        states = {k: z[k] for k in z.files}
        states["names"] = list(states["names"]); states["fps"] = int(states["fps"]); states["dt"] = float(states["dt"])
        states["label"] = str(states["label"])
        assert states["qpos"].shape[1] == env.num_envs, "replay with the same --num-envs as the recording"
        print(f"replaying {args.replay}: {states['qpos'].shape[0]} frames")
    elif args.video or args.save_states:
        states = rollout(env, policy, args.video_s, args.fps)
        states["label"] = label
        for line in outcome_lines(states, env):
            print("  " + line)
        if args.save_states:
            np.savez_compressed(args.save_states, **{k: np.asarray(v) for k, v in states.items()})
            print(f"states → {args.save_states}")
    if args.video and states is not None:
        render_video(env, states, args.video, states["label"])

    if not args.no_viewer:
        from mjlab.viewer import NativeMujocoViewer
        print("opening the native viewer (Ctrl+C to quit)…", flush=True)
        # The viewer drives the env through the rsl_rl wrapper API (get_observations).
        NativeMujocoViewer(RslRlVecEnvWrapper(env), policy).run()
    env.close()


if __name__ == "__main__":
    main()
