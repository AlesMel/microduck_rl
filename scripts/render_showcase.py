"""Showcase video of the ToF step-up policy: the hero robot in colour with its ToF beams and hit
points drawn from the head, every other robot of the batch grey and translucent, an 8×8 inset of
the frame the policy actually receives, a HUD, and a slow camera orbit.

    # on the GPU box (records the rollout, then renders offscreen with EGL):
    MUJOCO_GL=egl uv run python scripts/render_showcase.py record --checkpoint <run>/model_2999.pt \\
        --device cuda --num-envs 32 --seconds 8 --out showcase.npz
    MUJOCO_GL=egl uv run python scripts/render_showcase.py render --states showcase.npz --video showcase.mp4

Heroes: for each step tile (single_step, short_flight) the robot on the highest row that ended on
top of its step, earliest first; one segment per hero, concatenated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402
from mjlab_microduck.tasks import tof  # noqa: E402

TASK = "Mjlab-StepUp-Tof-Backlash-MicroDuck"
STEP_TILES = ("single_step", "short_flight")
FPS = 25


# ── record ───────────────────────────────────────────────────────────────────

def record(a) -> None:
    env = build_env(TASK, a.num_envs, a.seed, a.device, fixed_heading=False)
    policy = WalkPolicy(a.checkpoint, a.vx, 0.0, 0.0, onnx=False, device=a.device)
    robot = env.scene["robot"]
    sensor = env.scene.sensors[tof.TOF_SENSOR_NAME]
    names, types, levels, _ = _terrain_layout(env)
    gen = env.scene.terrain.cfg.terrain_generator
    nq = env.sim.mj_model.nq
    stride = max(1, round(1.0 / (FPS * env.step_dt)))
    obs, _ = env.reset()
    origins = env.scene.env_origins.cpu().numpy()
    z0 = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
    qpos, hits, dists, spos, frames_in, rise, fallen = [], [], [], [], [], [], []
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.no_grad():
        for k in range(round(a.seconds / env.step_dt)):
            if k % stride == 0:
                qpos.append(env.sim.data.qpos.cpu().numpy().reshape(-1, nq).copy())
                d = sensor.data
                hits.append(d.hit_pos_w[:, :64].cpu().numpy().copy())
                dists.append(d.distances[:, :64].cpu().numpy().copy())
                spos.append(d.pos_w.cpu().numpy().copy())
                frames_in.append(obs["actor"][:, -64:].cpu().numpy().copy())
                rise.append((robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy() - z0)
                fallen.append(fell.cpu().numpy().copy())
            obs, _, term, _, _ = env.step(policy(obs))
            fell |= term.bool()
    rise = np.stack(rise)
    riser = np.zeros(env.num_envs)
    for i in range(env.num_envs):
        t = names[types[i]]
        if t in STEP_TILES:
            st = gen.sub_terrains[t]
            riser[i] = st.num_steps * st.step_height((int(levels[i]) + 0.5) / gen.num_rows)
    heroes = []
    for t in STEP_TILES:
        cand = []
        for i in range(env.num_envs):
            if names[types[i]] != t or fallen[-1][i]:
                continue
            top = np.nonzero(rise[:, i] >= 0.8 * riser[i])[0]
            if len(top) and rise[-FPS:, i].mean() >= 0.8 * riser[i]:
                cand.append((-int(levels[i]), int(top[0]), i))
        if cand:
            heroes.append(int(sorted(cand)[0][2]))
    print("heroes:", [(h, names[types[h]], int(levels[h]), f"riser {100 * riser[h]:.1f} cm") for h in heroes])
    np.savez_compressed(a.out, qpos=np.stack(qpos), hits=np.stack(hits), dists=np.stack(dists), spos=np.stack(spos),
                        frames_in=np.stack(frames_in), rise=rise, fallen=np.stack(fallen), riser=riser,
                        names=np.array(names), types=types, levels=levels, env_origins=origins,
                        heroes=np.array(heroes), fps=FPS, dt=env.step_dt * stride)
    print("states →", a.out)


# ── render ───────────────────────────────────────────────────────────────────

def _turbo(x: np.ndarray) -> np.ndarray:
    """Near-warm → far-cool colours for values in [0, 1] (0 = near)."""
    from matplotlib import colormaps
    return colormaps["turbo"](1.0 - np.clip(x, 0, 1))[..., :3]


class Showcase:
    def __init__(self, env, width: int, height: int):
        self.mjm = env.sim.mj_model
        self.mjd = mujoco.MjData(self.mjm)
        self.extra = mujoco.MjData(self.mjm)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.mjm.vis.global_.offwidth = max(self.mjm.vis.global_.offwidth, width)
        self.mjm.vis.global_.offheight = max(self.mjm.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(self.mjm, height=height, width=width, max_geom=200_000)
        self.scene = self.renderer.scene

    def _load(self, mjd, q):
        mjd.qpos[:] = q
        mujoco.mj_forward(self.mjm, mjd)

    def _add_marker(self, kind, pos, size, rgba, frm=None, to=None, width=0.0):
        if self.scene.ngeom >= self.scene.maxgeom:
            return
        g = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        if frm is not None:
            mujoco.mjv_connector(g, kind, width, np.asarray(frm, dtype=np.float64), np.asarray(to, dtype=np.float64))
        self.scene.ngeom += 1

    def render(self, qpos_all, hero, cam, spos, hits, dists, grey=(0.55, 0.55, 0.6, 0.22),
               ghost_min_dist: float = 0.6) -> np.ndarray:
        self._load(self.mjd, qpos_all[hero])
        self.renderer.update_scene(self.mjd, cam, self.opt)
        for i in range(qpos_all.shape[0]):
            # Ghosts sharing the hero's tile would walk through the shot: keep only the far ones.
            if i == hero or np.linalg.norm(qpos_all[i, :2] - qpos_all[hero, :2]) < ghost_min_dist:
                continue
            self._load(self.extra, qpos_all[i])
            n0 = self.scene.ngeom
            mujoco.mjv_addGeoms(self.mjm, self.extra, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, self.scene)
            for g in range(n0, self.scene.ngeom):
                self.scene.geoms[g].rgba[:] = grey
        # ToF beams from the sensor to the hit points, coloured by range (near warm → far cool).
        col = _turbo(np.clip(dists, 0, 1.0) * 0.85)          # 1 m → blue, not the map's near-black end
        for z in range(64):
            if dists[z] < 0:
                continue
            far = dists[z] > 1.0
            self._add_marker(mujoco.mjtGeom.mjGEOM_CAPSULE, spos, (0, 0, 0), (*col[z], 0.18 if far else 0.45),
                             frm=spos, to=hits[z], width=0.0005)
            if not far:                                         # hit points only where the policy can see them
                self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, hits[z], (0.0045, 0, 0), (*col[z], 0.95))
        self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, spos, (0.006, 0, 0), (1, 1, 1, 1))
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def _font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _compose(frame: np.ndarray, grid: np.ndarray, lines: list[str], footer: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    # HUD (top-left)
    f1, f2 = _font(26), _font(18)
    y = 18
    for j, t in enumerate(lines):
        draw.text((24, y), t, font=f1 if j == 0 else f2, fill=(255, 255, 255, 235),
                  stroke_width=2, stroke_fill=(0, 0, 0, 180))
        y += 34 if j == 0 else 26
    draw.text((24, H - 34), footer, font=_font(16), fill=(220, 220, 220, 220), stroke_width=2, stroke_fill=(0, 0, 0, 160))
    # ToF inset (top-right): the frame the policy receives, near = warm, 1.0 = nothing within 1 m.
    cell, pad = 22, 2
    size = 8 * cell + 9 * pad
    x0, y0 = W - size - 24, 18
    draw.rectangle((x0 - 8, y0 - 30, x0 + size + 8, y0 + size + 8), fill=(0, 0, 0, 150))
    draw.text((x0, y0 - 26), "ToF 8×8 · policy input", font=f2, fill=(255, 255, 255, 235))
    col = (_turbo(grid.reshape(8, 8)) * 255).astype(np.uint8)
    for r in range(8):
        for c in range(8):
            v = grid.reshape(8, 8)[r, c]
            rgb = tuple(int(u) for u in col[r, c]) if v < 0.999 else (40, 40, 48)
            xx, yy = x0 + pad + c * (cell + pad), y0 + pad + r * (cell + pad)
            draw.rectangle((xx, yy, xx + cell, yy + cell), fill=(*rgb, 255))
    draw.text((x0, y0 + size + 10), "near ■ warm  ·  far ■ cool  ·  ■ nothing < 1 m", font=_font(14), fill=(230, 230, 230, 230),
              stroke_width=1, stroke_fill=(0, 0, 0, 160))
    return np.asarray(img.convert("RGB"))


def render(a) -> None:
    import imageio.v2 as imageio
    st = np.load(a.states, allow_pickle=False)
    names = [str(n) for n in st["names"]]
    types, levels, heroes = st["types"], st["levels"], [int(h) for h in st["heroes"]]
    qpos, hits, dists, spos, frames_in, rise, riser = (st[k] for k in ("qpos", "hits", "dists", "spos", "frames_in", "rise", "riser"))
    T, N = qpos.shape[:2]
    dt = float(st["dt"])
    env = build_env(TASK, N, a.seed, "cpu", fixed_heading=False)   # same grid: seeded generator
    shw = Showcase(env, a.width, a.height)
    writer = imageio.get_writer(a.video, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for seg, hero in enumerate(heroes if a.hero is None else [a.hero]):
            tile = names[types[hero]]
            label = {"single_step": "single step", "short_flight": "three-riser flight"}.get(tile, tile)
            print(f"segment {seg}: hero {hero} on {tile} row {int(levels[hero])}, riser {100 * riser[hero]:.1f} cm")
            for k in range(T):
                trunk = qpos[k, hero, :3]
                u = k / max(T - 1, 1)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = trunk + np.array([0.28, 0.0, -0.04])   # the step ahead stays in frame
                cam.distance = 1.15
                cam.azimuth = -75.0 + 45.0 * u          # over the left shoulder, slowly swinging round to the side
                cam.elevation = -14.0
                frame = shw.render(qpos[k], hero, cam, spos[k, hero], hits[k, hero], dists[k, hero])
                lines = [
                    "Microduck · step-up with ToF perception",
                    f"{label} · riser {100 * riser[hero]:.1f} cm",
                    f"t = {k * dt:4.1f} s · trunk rise {100 * rise[k, hero]:+.1f} cm",
                ]
                footer = "beams: the head ToF's 64 zones by range  ·  grey: other robots of the same batch  ·  policy: 61-D + 64 ToF → 14 servos"
                writer.append_data(_compose(frame, frames_in[k, hero], lines, footer))
    finally:
        writer.close()
        shw.close()
    print("video →", a.video)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--num-envs", type=int, default=32)
    r.add_argument("--seconds", type=float, default=8.0)
    r.add_argument("--vx", type=float, default=0.25)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="showcase.npz")
    v = sub.add_parser("render")
    v.add_argument("--states", required=True)
    v.add_argument("--video", default="showcase.mp4")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--width", type=int, default=1280)
    v.add_argument("--height", type=int, default=720)
    v.add_argument("--hero", type=int, default=None, help="force one robot as the hero")
    a = p.parse_args()
    record(a) if a.cmd == "record" else render(a)


if __name__ == "__main__":
    main()
