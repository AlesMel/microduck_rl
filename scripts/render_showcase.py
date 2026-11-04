"""Showcase video of a step-up policy: the hero robot in colour with what it perceives, every other
robot of the batch grey and translucent, an inset of the perception input as the policy receives
it, the feet marked at every contact (green = on one level, red = on an edge / against a riser),
a HUD with the running count of edge contacts, and a slow camera orbit.

Three perception modes, picked from the task's actor observation:
  ToF      the head sensor's 64 beams and hit points; inset = the 8×8 frame
  scan     the oracle height map's 133 points on the ground; inset = the 19×7 grid (forward = up)
  tof_map  the ToF-built map (run 10): the beams faint, the 133 scan points as the policy reads them
           from its map, cells it has never seen hollow grey; inset = the map with unknown cells dark

    # on the GPU box (records the rollout, then renders offscreen with EGL):
    MUJOCO_GL=egl uv run python scripts/render_showcase.py record --checkpoint <run>/model_2999.pt \\
        --device cuda --num-envs 32 --seconds 8 --out showcase.npz
    MUJOCO_GL=egl uv run python scripts/render_showcase.py render --states showcase.npz --video showcase.mp4
    # the oracle: add --task Mjlab-StepUp-Oracle-Clean-Backlash-MicroDuck to BOTH commands

Heroes: for each step tile (single_step, short_flight) the robot on the highest row that ended on
top of its step — a clean one (no edge contact) if there is one, else the earliest; one segment
per hero, concatenated. The HUD shows the hero's edge contacts, so a dirty climb is visible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from types import SimpleNamespace  # noqa: E402

from view_terrain_env import WalkPolicy, _terrain_layout, build_env  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks import tof  # noqa: E402

TASK = "Mjlab-StepUp-Tof-Backlash-MicroDuck"
STEP_TILES = ("single_step", "short_flight", "high_step")   # high_step: the envelope grid (run 11a)
FPS = 25


# ── record ───────────────────────────────────────────────────────────────────

def _scan_points_w(env, params: dict) -> torch.Tensor:
    """World (x, y, terrain z) of every oracle-scan point, in the obs order."""
    robot = env.scene["robot"]
    hf = microduck_mdp.terrain_height_field(env)
    xs, ys = microduck_mdp.scan_grid(params["x_range"], params["y_range"], params["res"])
    gx, gy = torch.meshgrid(torch.tensor(xs, device=env.device), torch.tensor(ys, device=env.device), indexing="ij")
    off = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    pos, q = robot.data.root_link_pos_w, robot.data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s_ = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    wx = pos[:, 0:1] + c * off[None, :, 0] - s_ * off[None, :, 1]
    wy = pos[:, 1:2] + s_ * off[None, :, 0] + c * off[None, :, 1]
    xy = torch.stack([wx, wy], dim=-1)
    return torch.cat([xy, hf.heights(xy)[..., None]], dim=-1)


def record(a) -> None:
    env = build_env(a.task, a.num_envs, a.seed, a.device, fixed_heading=False)
    policy = WalkPolicy(a.checkpoint, a.vx, 0.0, 0.0, onnx=False, device=a.device)
    robot = env.scene["robot"]
    sensor = env.scene.sensors.get(tof.TOF_SENSOR_NAME)
    om = env.observation_manager
    actor_names = list(om._group_obs_term_names["actor"])
    if sensor is not None and actor_names[-1] == "height_scan":
        mode = "tof_map"      # the map the ToF built: obs = [scan values (P), known mask (P)]
    elif sensor is not None:
        mode = "tof"
    else:
        mode = "scan"
    scan_params = None
    npts = 0
    if mode != "tof":
        assert actor_names[-1] == "height_scan", f"{a.task}: neither a ToF nor a height-scan task"
        scan_params = dict(om._group_obs_term_cfgs["actor"][-1].params)
        xs_, ys_ = microduck_mdp.scan_grid(scan_params["x_range"], scan_params["y_range"], scan_params["res"])
        npts = len(xs_) * len(ys_)
    width = {"tof": 64, "scan": npts, "tof_map": 2 * npts}[mode]
    feet = microduck_mdp.foot_edge_contact_cost(SimpleNamespace(params={}), env)      # corners only
    contact = env.scene.sensors["feet_ground_contact"]
    names, types, levels, _ = _terrain_layout(env)
    gen = env.scene.terrain.cfg.terrain_generator
    nq = env.sim.mj_model.nq
    stride = max(1, round(1.0 / (FPS * env.step_dt)))
    obs, _ = env.reset()
    origins = env.scene.env_origins.cpu().numpy()
    z0 = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
    qpos, hits, dists, spos, frames_in, rise, fallen, foot_pos, foot_found, foot_edge, beams = ([] for _ in range(11))
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    edge_count = torch.zeros(env.num_envs, device=env.device)
    hf = microduck_mdp.terrain_height_field(env)
    with torch.no_grad():
        for k in range(round(a.seconds / env.step_dt)):
            corners = feet.corners(env)                                               # (B, 2, 8, 3)
            found = contact.data.found[:, :2] > 0
            edge = microduck_mdp.foot_edge_flags(hf, corners, found)
            edge_count += (edge & ~fell[:, None]).float().sum(dim=1)
            if k % stride == 0:
                qpos.append(env.sim.data.qpos.cpu().numpy().reshape(-1, nq).copy())
                if sensor is not None:
                    d = sensor.data
                    (hits if mode == "tof" else beams).append(d.hit_pos_w[:, :64].cpu().numpy().copy())
                    dists.append(d.distances[:, :64].cpu().numpy().copy())
                    spos.append(d.pos_w.cpu().numpy().copy())
                if mode != "tof":
                    hits.append(_scan_points_w(env, scan_params).cpu().numpy())
                frames_in.append(obs["actor"][:, -width:].cpu().numpy().copy())
                rise.append((robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy() - z0)
                fallen.append(fell.cpu().numpy().copy())
                sole = torch.stack([corners[..., 0].mean(-1), corners[..., 1].mean(-1), corners[..., 2].amin(-1)], dim=-1)
                foot_pos.append(sole.cpu().numpy())
                foot_found.append(found.cpu().numpy())
                foot_edge.append(edge_count.cpu().numpy().copy())
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
                dirty = int(edge_count[i].item() > 0)
                cand.append((-int(levels[i]), dirty, int(top[0]), i))
        if cand:
            heroes.append(int(sorted(cand)[0][3]))
    print("heroes:", [(h, names[types[h]], int(levels[h]), f"riser {100 * riser[h]:.1f} cm",
                       f"edge contacts {int(edge_count[h].item())}") for h in heroes])
    empty = np.zeros((len(qpos), env.num_envs, 0))
    np.savez_compressed(a.out, mode=mode, qpos=np.stack(qpos), hits=np.stack(hits),
                        dists=np.stack(dists) if dists else empty, spos=np.stack(spos) if spos else empty,
                        frames_in=np.stack(frames_in), rise=rise, fallen=np.stack(fallen), riser=riser,
                        foot_pos=np.stack(foot_pos), foot_found=np.stack(foot_found), foot_edge=np.stack(foot_edge),
                        beams=np.stack(beams) if beams else empty,
                        scan_shape=np.array([0, 0] if scan_params is None else [
                            len(microduck_mdp.scan_grid(scan_params["x_range"], scan_params["y_range"], scan_params["res"])[0]),
                            len(microduck_mdp.scan_grid(scan_params["x_range"], scan_params["y_range"], scan_params["res"])[1])]),
                        names=np.array(names), types=types, levels=levels, env_origins=origins,
                        heroes=np.array(heroes), fps=FPS, dt=env.step_dt * stride, task=a.task)
    print("states →", a.out)


# ── render ───────────────────────────────────────────────────────────────────

def _turbo(x: np.ndarray) -> np.ndarray:
    """Near-warm → far-cool colours for values in [0, 1] (0 = near)."""
    from matplotlib import colormaps
    return colormaps["turbo"](1.0 - np.clip(x, 0, 1))[..., :3]


def _height_colour(scan_value: np.ndarray) -> np.ndarray:
    """Oracle scan values (trunk_z − h − nominal: 0 = the ground the robot stands on, −0.04 = 4 cm
    higher) → cool for the robot's own level, warm for terrain above it."""
    from matplotlib import colormaps
    return colormaps["turbo"](0.15 + 0.85 * np.clip(-scan_value / 0.10, 0, 1))[..., :3]


class Showcase:
    def __init__(self, env, width: int, height: int):
        self.hf = microduck_mdp.terrain_height_field(env)     # halo height = the ground under the foot
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
               ghost_min_dist: float = 0.6, mode: str = "tof", scan_values=None,
               feet=None, beams=None, known=None) -> np.ndarray:
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
        if mode == "tof":
            # ToF beams from the sensor to the hit points, coloured by range (near warm → far cool).
            col = _turbo(np.clip(dists, 0, 1.0) * 0.85)      # 1 m → blue, not the map's near-black end
            for z in range(64):
                if dists[z] < 0:
                    continue
                far = dists[z] > 1.0
                self._add_marker(mujoco.mjtGeom.mjGEOM_CAPSULE, spos, (0, 0, 0), (*col[z], 0.18 if far else 0.45),
                                 frm=spos, to=hits[z], width=0.0005)
                if not far:                                     # hit points only where the policy can see them
                    self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, hits[z], (0.0045, 0, 0), (*col[z], 0.95))
            self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, spos, (0.006, 0, 0), (1, 1, 1, 1))
        else:
            if mode == "tof_map" and beams is not None:
                # What is being measured: the ToF beams, faint, and the hit points within the map's reach.
                col = _turbo(np.clip(dists, 0, 1.0) * 0.85)
                for z in range(64):
                    if dists[z] < 0:
                        continue
                    far = dists[z] > 1.2
                    self._add_marker(mujoco.mjtGeom.mjGEOM_CAPSULE, spos, (0, 0, 0), (*col[z], 0.08 if far else 0.22),
                                     frm=spos, to=beams[z], width=0.0004)
                    if not far:
                        self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, beams[z], (0.003, 0, 0), (*col[z], 0.7))
                self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, spos, (0.006, 0, 0), (1, 1, 1, 1))
            # The height map the policy reads: one dot per scan point, on the terrain surface;
            # on the ToF-built map a cell never seen is a small hollow grey dot.
            col = _height_colour(scan_values)
            for z in range(hits.shape[0]):
                if known is not None and not known[z]:
                    self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, hits[z] + np.array([0, 0, 0.003]), (0.0028, 0, 0),
                                     (0.45, 0.45, 0.5, 0.6))
                    continue
                self._add_marker(mujoco.mjtGeom.mjGEOM_SPHERE, hits[z] + np.array([0, 0, 0.003]), (0.0042, 0, 0),
                                 (*col[z], 0.95))
        if feet is not None:
            # Feet at contact: green = on one level, red = against a riser / straddling an edge.
            pos, found, edge = feet
            for f in range(2):
                if found[f]:
                    rgba = (0.95, 0.12, 0.1, 0.95) if edge[f] else (0.15, 0.85, 0.35, 0.85)
                    ground = float(self.hf.heights(torch.tensor(pos[f][None, :2], dtype=torch.float32))[0])
                    self._add_marker(mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([pos[f][0], pos[f][1], ground + 0.0015]),
                                     (0.04, 0.0012, 0), rgba)             # wider than the sole: a halo
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


def _compose(frame: np.ndarray, grid: np.ndarray, lines: list[str], footer: str, mode: str = "tof",
             scan_shape=(0, 0)) -> np.ndarray:
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
    if mode in ("scan", "tof_map"):
        # Height-map inset (top-right): the 19×7 grid the policy receives, forward = up, left = left.
        nx, ny = int(scan_shape[0]), int(scan_shape[1])
        known = None
        if mode == "tof_map":                                   # [values (P), known (P)]
            grid, known = grid[:nx * ny], grid[nx * ny:].reshape(nx, ny)[::-1, ::-1] > 0.5
        g = grid.reshape(nx, ny)[::-1, ::-1]                  # far row on top, +y (left) on the left
        cell, pad = 14, 1
        wpx, hpx = ny * (cell + pad) + pad, nx * (cell + pad) + pad
        x0, y0 = W - wpx - 24, 46
        draw.rectangle((x0 - 8, y0 - 30, x0 + wpx + 8, y0 + hpx + 30), fill=(0, 0, 0, 150))
        draw.text((x0 - 2, y0 - 26), "height map" if mode == "scan" else "ToF map", font=f2, fill=(255, 255, 255, 235))
        col = (_height_colour(g) * 255).astype(np.uint8)
        for r in range(nx):
            for c in range(ny):
                xx, yy = x0 + pad + c * (cell + pad), y0 + pad + r * (cell + pad)
                rgb = (40, 40, 48) if known is not None and not known[r, c] else tuple(int(u) for u in col[r, c])
                draw.rectangle((xx, yy, xx + cell, yy + cell), fill=(*rgb, 255))
        draw.text((x0 - 2, y0 + hpx + 6), "▲ forward", font=_font(14), fill=(230, 230, 230, 230))
        if known is not None:
            draw.rectangle((x0 - 8, y0 + hpx + 30, x0 + wpx + 8, y0 + hpx + 50), fill=(0, 0, 0, 150))
            draw.text((x0 - 2, y0 + hpx + 32), "■ unseen", font=_font(14), fill=(230, 230, 230, 230))
        return np.asarray(img.convert("RGB"))
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
    mode = str(st["mode"]) if "mode" in st else "tof"
    npts = int(st["scan_shape"][0] * st["scan_shape"][1]) if "scan_shape" in st else 0
    task = str(st["task"]) if "task" in st else TASK
    has_feet = "foot_pos" in st
    T, N = qpos.shape[:2]
    dt = float(st["dt"])
    env = build_env(task, N, a.seed, "cpu", fixed_heading=False)   # same grid: seeded generator
    shw = Showcase(env, a.width, a.height)
    writer = imageio.get_writer(a.video, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for seg, hero in enumerate(heroes if a.hero is None else [a.hero]):
            tile = names[types[hero]]
            label = {"single_step": "single step", "short_flight": "three-riser flight", "high_step": "tall single step"}.get(tile, tile)
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
                feet = (st["foot_pos"][k, hero], st["foot_found"][k, hero], None) if has_feet else None
                if feet is not None:
                    edges_now = st["foot_edge"][k, hero] > (st["foot_edge"][k - 1, hero] if k else 0)
                    feet = (feet[0], feet[1], np.array([edges_now and feet[1][f] for f in range(2)]))
                frame = shw.render(qpos[k], hero, cam, spos[k, hero] if mode != "scan" else None, hits[k, hero],
                                   dists[k, hero] if mode != "scan" else None, mode=mode,
                                   scan_values=frames_in[k, hero, :npts] if mode != "tof" else None, feet=feet,
                                   beams=st["beams"][k, hero] if mode == "tof_map" else None,
                                   known=frames_in[k, hero, npts:] > 0.5 if mode == "tof_map" else None)
                title = {"tof": "Microduck · step-up with ToF perception",
                         "scan": "Microduck · step-up with a terrain height map (oracle)",
                         "tof_map": "Microduck · step-up with a height map built from the ToF"}[mode]
                lines = [
                    title,
                    f"{label} · riser {100 * riser[hero]:.1f} cm",
                    f"t = {k * dt:4.1f} s · trunk rise {100 * rise[k, hero]:+.1f} cm"
                    + (f" · edge contacts {int(st['foot_edge'][k, hero])}" if has_feet else ""),
                ]
                if mode == "tof":
                    footer = "beams: the head ToF's 64 zones  ·  feet: green = clean contact, red = edge  ·  grey: other robots"
                elif mode == "scan":
                    footer = "dots: the 19×7 height map the policy reads  ·  feet: green = clean contact, red = edge  ·  grey: other robots"
                else:
                    footer = "beams: the head ToF  ·  dots: the 19×7 map it built (hollow = never seen)  ·  feet: green = clean, red = edge"
                writer.append_data(_compose(frame, frames_in[k, hero], lines, footer, mode=mode,
                                            scan_shape=st["scan_shape"] if "scan_shape" in st else (0, 0)))
    finally:
        writer.close()
        shw.close()
    print("video →", a.video)


# ── compare: two policies side by side on the same step ────────────────────

def _feet_at(st, k: int, hero: int):
    if "foot_pos" not in st:
        return None
    found = st["foot_found"][k, hero]
    edges_now = st["foot_edge"][k, hero] > (st["foot_edge"][k - 1, hero] if k else 0)
    return st["foot_pos"][k, hero], found, np.array([bool(edges_now and found[f]) for f in range(2)])


def _card(width: int, height: int, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (12, 12, 16))
    draw = ImageDraw.Draw(img)
    y = height // 2 - 26 * len(lines)
    for j, t in enumerate(lines):
        f = _font(40 if j == 0 else 28)
        w = draw.textlength(t, font=f)
        draw.text(((width - w) / 2, y), t, font=f, fill=(255, 255, 255) if j == 0 else (210, 210, 220))
        y += 64 if j == 0 else 46
    return np.asarray(img)


def compare(a) -> None:
    """Left / right panes, one policy each, the same tile and row of the same grid, a fixed camera
    over the hero's left shoulder. Heroes are NOT the best robots: the ``--pick``-th robot on that
    tile and row that still had its approach ahead of it (trunk rise < 30 % of the riser at 0.6 s).
    A hero that falls is frozen at its last frame with FELL over it (the env resets it after)."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    sides = []
    for path, label in ((a.left, a.left_label), (a.right, a.right_label)):
        st = np.load(path, allow_pickle=False)
        N = st["qpos"].shape[1]
        env = build_env(str(st["task"]), N, a.seed, "cpu", fixed_heading=False)
        sides.append(dict(st=st, mode=str(st["mode"]), shw=Showcase(env, a.width // 2, a.height), label=label,
                          names=[str(n) for n in st["names"]], N=N, dt=float(st["dt"])))
    T = min(s_["st"]["qpos"].shape[0] for s_ in sides)
    writer = imageio.get_writer(a.video, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for _ in range(2 * FPS):
            writer.append_data(_card(a.width + 4, a.height, [a.title, f"left: {a.left_label}", f"right: {a.right_label}",
                                                         "same step · feet: green = clean contact, red = edge"]))
        for seg in a.segments.split(","):
            tile, row = seg.split(":")
            row = int(row)
            label = {"single_step": "single step", "short_flight": "three-riser flight", "high_step": "tall single step"}.get(tile, tile)
            heroes = []
            for sd in sides:
                st, names = sd["st"], sd["names"]
                k06 = min(int(round(0.6 / sd["dt"])), T - 1)
                cands = [i for i in range(sd["N"]) if names[st["types"][i]] == tile and int(st["levels"][i]) == row
                         and st["rise"][k06, i] < 0.3 * st["riser"][i]]
                assert cands, f"no robot with its approach ahead on {tile} row {row} in {sd['label']}"
                hero = cands[a.pick % len(cands)]
                fk = np.nonzero(st["fallen"][:, hero])[0]
                heroes.append((hero, int(fk[0]) if len(fk) else None))
                print(f"{sd['label']}: {tile} row {row} hero {hero}, riser {100 * st['riser'][hero]:.1f} cm, "
                      f"fell at frame {heroes[-1][1]}, edge contacts {int(st['foot_edge'][-1, hero])}")
            for k in range(T):
                imgs = []
                for sd, (hero, fk) in zip(sides, heroes):
                    st, mode = sd["st"], sd["mode"]
                    fell = fk is not None and k >= fk
                    kk = max(fk - 1, 0) if fell else k
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    cam.lookat[:] = st["qpos"][kk, hero, :3] + np.array([0.25, 0.0, -0.04])
                    cam.distance, cam.azimuth, cam.elevation = 1.1, -62.0, -14.0
                    npts = int(st["scan_shape"][0] * st["scan_shape"][1]) if "scan_shape" in st else 0
                    frame = sd["shw"].render(st["qpos"][kk], hero, cam, st["spos"][kk, hero] if mode != "scan" else None,
                                             st["hits"][kk, hero], st["dists"][kk, hero] if mode != "scan" else None,
                                             mode=mode, scan_values=st["frames_in"][kk, hero, :npts] if mode != "tof" else None,
                                             feet=_feet_at(st, kk, hero),
                                             beams=st["beams"][kk, hero] if mode == "tof_map" else None,
                                             known=st["frames_in"][kk, hero, npts:] > 0.5 if mode == "tof_map" else None)
                    lines = [sd["label"], f"{label} · riser {100 * st['riser'][hero]:.1f} cm",
                             f"t = {k * sd['dt']:4.1f} s · edge contacts {int(st['foot_edge'][kk, hero])}"]
                    img = _compose(frame, st["frames_in"][kk, hero], lines, "", mode=mode,
                                   scan_shape=st["scan_shape"] if "scan_shape" in st else (0, 0))
                    if fell:
                        pim = Image.fromarray(img)
                        d = ImageDraw.Draw(pim)
                        f = _font(72)
                        w = d.textlength("FELL", font=f)
                        d.text(((pim.width - w) / 2, pim.height / 2 - 40), "FELL", font=f, fill=(235, 40, 30),
                               stroke_width=3, stroke_fill=(0, 0, 0))
                        img = np.asarray(pim)
                    imgs.append(img)
                    if len(imgs) == 1:
                        imgs.append(np.full((img.shape[0], 4, 3), 255, dtype=np.uint8))   # divider
                writer.append_data(np.concatenate(imgs, axis=1))
        if a.summary:
            for _ in range(4 * FPS):
                writer.append_data(_card(a.width + 4, a.height, a.summary))
    finally:
        writer.close()
        for sd in sides:
            sd["shw"].close()
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
    r.add_argument("--task", default=TASK)
    v = sub.add_parser("render")
    v.add_argument("--states", required=True)
    v.add_argument("--video", default="showcase.mp4")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--width", type=int, default=1280)
    v.add_argument("--height", type=int, default=720)
    v.add_argument("--hero", type=int, default=None, help="force one robot as the hero")
    c = sub.add_parser("compare")
    c.add_argument("--left", required=True, help="states .npz of the left policy")
    c.add_argument("--right", required=True)
    c.add_argument("--left-label", default="left")
    c.add_argument("--right-label", default="right")
    c.add_argument("--title", default="Same step, two policies")
    c.add_argument("--segments", default="single_step:3,short_flight:3", help="tile:row,...")
    c.add_argument("--pick", type=int, default=0, help="which qualifying robot (0 = the first by index)")
    c.add_argument("--summary", nargs="*", default=None, help="lines of an end card")
    c.add_argument("--video", default="compare.mp4")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=600)
    a = p.parse_args()
    {"record": record, "render": render, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
