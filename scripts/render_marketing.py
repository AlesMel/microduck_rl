"""Marketing cut of the step-up skill: a storyboard of recorded rollouts (``render_showcase.py record``)
rendered in a restyled look — warm dark terrain fading into fog, a sky gradient, a sun that follows
the hero for shadows, no ghost robots, the ToF beams / height-map dots / foot halos of the showcase,
a perception inset, lower-third captions, cream title / end cards (palette and fonts at the top).

    # on a GPU box, after recording (render_showcase.py record --fps 50: 50 states/s for slow motion):
    MUJOCO_GL=egl uv run python scripts/render_marketing.py pair --left mk/alpha_walking_lo.npz --right mk/now_lo.npz
    MUJOCO_GL=egl uv run python scripts/render_marketing.py still --board mk/board.json --clip 2 --at 0.5 --out s.png
    MUJOCO_GL=egl uv run python scripts/render_marketing.py cut --board mk/board.json --video microduck_steps.mp4

``pair`` lines up two recordings made on the same grid and seed (robot i = the same tile, row and
start in both) and prints per-height clean-climb and fall counts — the end card's numbers come from it.

The board is JSON (scripts/marketing/step_up_board.json): ``{"width", "height", "items": [...]}``:

    {"type": "card", "kind": "title" | "text" | "end", "seconds": 3.5, "lines": [...],
     "stats": [{"value": "79 %", "label": "clean climbs\nnew policy", "color": "good"}, ...]}   # end card
    {"type": "clip", "states": "mk/now_hi.npz", "hero": 29,
     "center": "climb" | "edge" | <seconds>, "pre": 1.0, "post": 2.2,     # source window around the centre
     "slow": [-0.5, 0.5, 0.4],                # 0.4× from 0.5 s before to 0.5 s after the centre
     "cam": {"az": [196, 180], "el": [-6, -4], "dist": [0.72, 0.64], "ahead": 0.04},   # [start, end] eased
     "beams": false, "tag": "NOW", "tag_color": "good", "caption": "...", "sub": "...", "badge": "7.7 cm"}
    {"type": "split", "left": "mk/alpha_walking_hi.npz", "right": "mk/now_hi.npz", "hero": 29,
     "height": "7.7 cm", "left_label": "BEFORE", "left_sub": "...", "right_label": "NOW", "right_sub": "...",
     "left_perception": false,                # a blind policy: no beams / map dots drawn for it
     "align": true, "center_left": "edge"}    # each pane around its own moment (left: first riser hit)

A split pane counts edge hits (onsets of a foot touching a riser edge) and freezes a fallen robot.
Everything shown is simulation; the cards say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from render_showcase import Showcase, riser_hook  # noqa: E402
from view_terrain_env import build_env  # noqa: E402

FPS = 25
FONT_DIRS = (Path.home() / ".fonts", Path("/usr/share/fonts/truetype"), Path("/usr/share/fonts"))


# Palette: warm ink / paper / clay, in the spirit of Anthropic's colours but shifted (deeper clay,
# warmer paper, lighter sage and sky) so it is a cousin, not a copy.
def _hex(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


INK = _hex("#181715")          # near-black, warm
PAPER = _hex("#EFEBE2")        # cream
STONE = _hex("#CFC8BA")        # secondary text on dark
DUSK = _hex("#5E584F")         # secondary text on paper
CLAY = _hex("#D2734F")         # the accent
KRAFT = _hex("#D6A780")
SAGE = _hex("#8BA46E")
SKY = _hex("#7BA4CF")
BRICK = _hex("#C2503A")
ACCENT = {"good": SAGE, "bad": BRICK, "info": SKY, "warm": CLAY}
SKY_TOP = np.array([17, 16, 14], dtype=np.float32)
SKY_HORIZON = np.array([50, 46, 41], dtype=np.float32)
TERRAIN_DARK = np.array([0.20, 0.19, 0.175])
TERRAIN_LIGHT = np.array([0.50, 0.47, 0.43])
EDGE_RGBA = (0.95, 0.80, 0.62, 0.95)                     # riser edges: pale kraft glow
HALO_CLEAN = (*(np.array(SAGE) / 255 * 1.1).clip(0, 1), 0.9)
HALO_EDGE = (*(np.array(BRICK) / 255), 0.95)


def _ramp(x, stops) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    pos = np.linspace(0.0, 1.0, len(stops))
    cols = np.array(stops, dtype=np.float64) / 255.0
    return np.stack([np.interp(x, pos, cols[:, c]) for c in range(3)], axis=-1)


def beam_cmap(x) -> np.ndarray:
    """ToF range in [0, 1] (0 = near): clay → kraft → pale → sky."""
    return _ramp(x, [CLAY, KRAFT, (226, 214, 196), SKY])


def height_cmap(v) -> np.ndarray:
    """Height-map value (0 = the robot's own level, −0.04 = 4 cm higher): sky → pale → kraft → clay."""
    return _ramp(np.clip(-np.asarray(v, dtype=np.float64) / 0.09, 0.0, 1.0), [SKY, (205, 198, 182), KRAFT, CLAY])


# ── fonts ────────────────────────────────────────────────────────────────────

def _find_font(name: str) -> str | None:
    for d in FONT_DIRS:
        if d.exists():
            for p in d.rglob(name):
                return str(p)
    return None


_FONT_CACHE: dict = {}


def font(size: int, weight: str = "Regular", display: bool = False):
    from PIL import ImageFont
    key = (size, weight, display)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    f = None
    path = _find_font("BebasNeue-Regular.ttf") if display else _find_font("Jost-VariableFont_wght.ttf")
    if path:
        f = ImageFont.truetype(path, size)
        if not display:
            try:
                f.set_variation_by_name(weight)
            except Exception:  # noqa: BLE001 — a static fallback font has no named instances
                pass
    if f is None:
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
            try:
                f = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
    _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


def serif(size: int, bold: bool = False):
    """Headlines and numbers: Bitstream Charter (a sturdy book serif), else a fallback serif."""
    from PIL import ImageFont
    key = ("serif", size, bold)
    if key not in _FONT_CACHE:
        f = None
        for name in (("bchb8a.pfb", "c0632bt_.pfb") if bold else ("bchr8a.pfb", "c0648bt_.pfb")) + ("DejaVuSerif.ttf",):
            path = _find_font(name)
            if path:
                try:
                    f = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        _FONT_CACHE[key] = f or font(size)
    return _FONT_CACHE[key]


def _spaced(draw, xy, text, fnt, fill, spacing: float):
    """Letter-spaced text (wordmarks); returns the width drawn."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + spacing
    return x - xy[0] - spacing


def _spaced_width(draw, text, fnt, spacing: float) -> float:
    return sum(draw.textlength(ch, font=fnt) for ch in text) + spacing * (len(text) - 1)


# ── the restyled scene ───────────────────────────────────────────────────────

class Stage:
    """A Showcase renderer with the marketing look applied to the model and scene."""

    def __init__(self, env, width: int, height: int):
        self.env = env
        self.W, self.H = width, height
        m = self.mjm = env.sim.mj_model
        # Fog and shadow-map size are baked into the GL context when it is made: set them first.
        m.vis.map.fogstart, m.vis.map.fogend = 1.0, 3.5           # × extent (2 m): fade from 2 m to 7 m
        m.vis.rgba.fog[:] = (*(SKY_HORIZON / 255.0), 1.0)
        m.vis.headlight.ambient[:] = (0.30, 0.31, 0.34)
        m.vis.headlight.diffuse[:] = (0.34, 0.34, 0.36)
        m.vis.headlight.specular[:] = (0.1, 0.1, 0.1)
        m.vis.quality.shadowsize = 8192
        if m.nlight:
            m.light_diffuse[0] = (0.8, 0.77, 0.72)
            m.light_dir[0] = np.array([0.55, -0.35, -1.0]) / np.linalg.norm([0.55, -0.35, -1.0])
        self.sun_offset = -m.light_dir[0] * 1.5 if m.nlight else None
        self.shw = Showcase(env, width, height)
        self.shw.beam_cmap, self.shw.height_cmap = beam_cmap, height_cmap
        self.shw.halo_clean, self.shw.halo_edge = HALO_CLEAN, HALO_EDGE
        # Terrain: dark slate, shaded by the height of each box's top so every level reads apart.
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for g in range(m.ngeom):
            if m.body_weldid[m.geom_bodyid[g]] != 0 or m.geom_rgba[g, 3] == 0:
                continue
            top = d.geom_xpos[g, 2] + (m.geom_size[g, 2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0.0)
            u = float(np.clip(top / 0.08, 0.0, 1.0))
            m.geom_rgba[g, :3] = TERRAIN_DARK + (TERRAIN_LIGHT - TERRAIN_DARK) * u
            m.geom_matid[g] = -1
        self.edges: list[tuple[np.ndarray, np.ndarray]] = []
        r = self.shw.renderer
        orig = r.update_scene

        def update_scene(*a, **kw):
            orig(*a, **kw)
            sc = r.scene
            sc.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            for frm, to in self.edges:                              # glowing riser edges
                self.shw._add_marker(mujoco.mjtGeom.mjGEOM_CAPSULE, frm, (0, 0, 0), EDGE_RGBA,
                                     frm=frm, to=to, width=0.0016)
        r.update_scene = update_scene
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None] ** 1.4
        self.sky = (SKY_TOP + (SKY_HORIZON - SKY_TOP) * ys) * np.ones((1, width, 1), dtype=np.float32)

    def mark_risers(self, trunk: np.ndarray, back: float = 0.6, ahead: float = 1.6, half_width: float = 0.3) -> None:
        """Find the risers on the hero's line (+x, the tiles' stair direction) in the exact height
        field and outline each top edge with a thin glowing line."""
        xs = np.arange(trunk[0] - back, trunk[0] + ahead, 0.004)
        pts = torch.tensor(np.stack([xs, np.full_like(xs, trunk[1])], axis=1), dtype=torch.float32)
        h = self.shw.hf.heights(pts).numpy()
        self.edges = []
        for j in np.nonzero(np.diff(h) > 0.004)[0]:
            x, top = xs[j] + 0.002, float(h[j + 1]) + 0.0012
            self.edges.append((np.array([x, trunk[1] - half_width, top]), np.array([x, trunk[1] + half_width, top])))

    def frame(self, st, k: int, hero: int, cam, beams: bool = True, perception: bool = True) -> np.ndarray:
        """``perception=False``: no beams, no map dots (a blind policy recorded on a perceptive task)."""
        mode = str(st["mode"])
        npts = int(st["scan_shape"][0] * st["scan_shape"][1])
        if self.sun_offset is not None:
            self.mjm.light_pos[0] = st["qpos"][k, hero, :3] + self.sun_offset
        fe = st["foot_edge"]
        edges_now = fe[k, hero] > (fe[k - 1, hero] if k else 0)
        found = st["foot_found"][k, hero]
        feet = (st["foot_pos"][k, hero], found, np.array([bool(edges_now and found[f]) for f in range(2)]))
        if not perception:
            rgb = self.shw.render(st["qpos"][k], hero, cam, None, np.zeros((0, 3)), None, ghost_min_dist=1e9,
                                  mode="scan", scan_values=np.zeros(0), feet=feet)
        else:
            rgb = self.shw.render(st["qpos"][k], hero, cam, st["spos"][k, hero] if mode != "scan" else None,
                                  st["hits"][k, hero], st["dists"][k, hero] if mode != "scan" else None,
                                  ghost_min_dist=1e9, mode=mode,
                                  scan_values=st["frames_in"][k, hero, :npts] if mode != "tof" else None, feet=feet,
                                  beams=st["beams"][k, hero] if mode == "tof_map" and beams else None,
                                  known=st["frames_in"][k, hero, npts:] > 0.5 if mode == "tof_map" else None)
        r = self.shw.renderer
        r.enable_depth_rendering()
        depth = r.render()
        r.disable_depth_rendering()
        far = self.mjm.vis.map.zfar * self.mjm.stat.extent
        bg = depth > 0.9 * far
        out = rgb.astype(np.float32)
        out[bg] = self.sky[bg]
        return out.clip(0, 255).astype(np.uint8)

    def close(self):
        self.shw.close()


# ── clips ────────────────────────────────────────────────────────────────────

def load_states(path: str) -> dict:
    st = np.load(path, allow_pickle=False)
    return {k: st[k] for k in st.files}


def build_stage(st: dict, width: int, height: int, seed: int) -> Stage:
    rr = st.get("riser_range", np.zeros(0))
    env = build_env(str(st["task"]), st["qpos"].shape[1], seed, "cpu", fixed_heading=False, cfg_hook=riser_hook(rr))
    return Stage(env, width, height)


def event_time(st: dict, hero: int, what) -> float:
    dt = float(st["dt"])
    if isinstance(what, (int, float)):
        return float(what)
    if what == "climb":
        k = np.nonzero(st["rise"][:, hero] >= 0.5 * st["riser"][hero])[0]
    elif what == "edge":
        k = np.nonzero(np.diff(st["foot_edge"][:, hero]) > 0)[0] + 1
    else:
        raise ValueError(what)
    assert len(k), f"hero {hero}: no '{what}' event"
    return float(k[0] * dt)


def source_times(c: dict, st: dict) -> np.ndarray:
    """Output frame j (25 fps) → source time, with an eased slow-motion window around the centre."""
    tc = event_time(st, c["hero"], c.get("center", "climb"))
    t0 = max(0.0, tc - c.get("pre", 2.5))
    t1 = min(st["qpos"].shape[0] * float(st["dt"]) - float(st["dt"]), tc + c.get("post", 3.0))
    s0, s1, speed = (c.get("slow") or [0, 0, 1.0])
    s0, s1 = tc + s0, tc + s1
    ramp = 0.25
    out, t = [], t0
    while t <= t1:
        out.append(t)
        if t < s0 - ramp or t > s1 + ramp:
            v = 1.0
        elif s0 <= t <= s1:
            v = speed
        else:                                                   # ease in / out of the slow window
            u = (s0 - t) / ramp if t < s0 else (t - s1) / ramp
            v = speed + (1.0 - speed) * (0.5 - 0.5 * np.cos(np.pi * u))
        t += v / FPS
    return np.array(out)


def smooth_trunk(st: dict, hero: int, window_s: float = 0.35) -> np.ndarray:
    p = st["qpos"][:, hero, :3].astype(np.float64)
    fell = np.nonzero(st["fallen"][:, hero])[0]
    if len(fell):                                               # the env teleports a fallen robot: hold
        p[fell[0]:] = p[max(fell[0] - 1, 0)]
    n = max(1, int(round(window_s / float(st["dt"]))))
    pad = np.pad(p, ((n, n), (0, 0)), mode="edge")
    ker = np.ones(2 * n + 1) / (2 * n + 1)
    return np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(3)], axis=1)


def camera(c: dict, trunk: np.ndarray, u: float):
    cm = c.get("cam", {})

    def lerp(key, default):
        v = cm.get(key, default)
        if isinstance(v, (list, tuple)):
            e = 0.5 - 0.5 * np.cos(np.pi * u)                   # ease in-out across the clip
            return v[0] + (v[1] - v[0]) * e
        return v
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = trunk + np.array([lerp("ahead", 0.15), 0.0, lerp("up", -0.03)])
    cam.distance = lerp("dist", 0.85)
    cam.azimuth = lerp("az", -95.0)
    cam.elevation = lerp("el", -8.0)
    return cam


# ── overlays ─────────────────────────────────────────────────────────────────

def _alpha(j: int, n: int, fade: int = 8) -> float:
    return float(min(1.0, (j + 1) / fade, (n - j) / fade))


def overlay(img: np.ndarray, c: dict, st: dict, k: int, j: int, n: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    W, H = img.shape[1], img.shape[0]
    s = H / 1080.0
    base = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a_txt = min(1.0, max(0.0, (j - 6) / 10.0)) * min(1.0, (n - j) / 8.0)
    # Top-left: the wordmark and the honest tag.
    fw = font(int(30 * s), "SemiBold")
    _spaced(d, (int(48 * s), int(40 * s)), "MICRODUCK", fw, (*PAPER, 235), 6 * s)
    ft = font(int(17 * s), "Medium")
    tag = "SIMULATION"
    tw = _spaced_width(d, tag, ft, 3 * s)
    x0, y0 = int(48 * s), int(84 * s)
    d.rounded_rectangle((x0, y0, x0 + tw + 24 * s, y0 + 30 * s), radius=int(15 * s), outline=(*STONE, 130), width=max(1, int(1.5 * s)))
    _spaced(d, (x0 + 12 * s, y0 + 4 * s), tag, ft, (*STONE, 190), 3 * s)
    # Top-right: the perception inset, exactly what the policy reads.
    _inset(d, st, k, c["hero"], W, s)
    # Lower third: tag pill, caption, sub-caption.
    if c.get("caption"):
        acc = ACCENT.get(c.get("tag_color", "info"), ACCENT["info"])
        ya = int(H - 230 * s)
        xa = int(48 * s)
        if c.get("tag"):
            fp = font(int(22 * s), "SemiBold")
            pw = _spaced_width(d, c["tag"], fp, 3 * s)
            d.rounded_rectangle((xa, ya, xa + pw + 32 * s, ya + 40 * s), radius=int(8 * s), fill=(*acc, int(235 * a_txt)))
            _spaced(d, (xa + 16 * s, ya + 6 * s), c["tag"], fp, (*INK, int(255 * a_txt)), 3 * s)
            ya += int(56 * s)
        d.rectangle((xa, ya + 6 * s, xa + 5 * s, ya + 104 * s), fill=(*acc, int(255 * a_txt)))
        d.text((xa + 24 * s, ya - 4 * s), c["caption"], font=serif(int(56 * s)), fill=(*PAPER, int(255 * a_txt)))
        if c.get("sub"):
            d.text((xa + 24 * s, ya + 68 * s), c["sub"], font=font(int(28 * s), "Light"), fill=(*STONE, int(235 * a_txt)))
    # Right: the big number.
    if c.get("badge"):
        fb = serif(int(170 * s))
        bw = d.textlength(c["badge"], font=fb)
        xb, yb = W - 64 * s - bw, H - 330 * s
        d.text((xb, yb), c["badge"], font=fb, fill=(*PAPER, int(250 * a_txt)))
        if c.get("badge_sub"):
            fs = font(int(26 * s), "Regular")
            for i, line in enumerate(c["badge_sub"].split("\n")):
                lw = d.textlength(line, font=fs)
                d.text((W - 64 * s - lw, yb + 192 * s + i * 36 * s), line, font=fs, fill=(*STONE, int(235 * a_txt)))
    out = Image.alpha_composite(base, layer).convert("RGB")
    fade = _alpha(j, n)
    arr = np.asarray(out).astype(np.float32) * fade
    return arr.astype(np.uint8)


def _inset(d, st: dict, k: int, hero: int, W: int, s: float) -> None:
    mode = str(st["mode"])
    grid = st["frames_in"][k, hero]
    if mode == "tof":
        g = grid.reshape(8, 8)
        rows, cols, cell = 8, 8, int(20 * s)
        col = (beam_cmap(g) * 255).astype(np.uint8)
        known = g < 0.999
        title, foot = "ToF FRAME", "8×8 · no memory"
    else:
        nx, ny = int(st["scan_shape"][0]), int(st["scan_shape"][1])
        P = nx * ny
        vals = grid[:P].reshape(nx, ny)[::-1, ::-1]
        known = (grid[P:].reshape(nx, ny)[::-1, ::-1] > 0.5) if mode == "tof_map" else np.ones((nx, ny), bool)
        rows, cols, cell = nx, ny, int(13 * s)
        col = (height_cmap(vals) * 255).astype(np.uint8)
        title, foot = "HEIGHT MAP", "built from ToF"
    pad = max(1, int(2 * s))
    gw, gh = cols * (cell + pad) - pad, rows * (cell + pad) - pad
    x0 = int(W - 48 * s - gw - 32 * s)
    y0 = int(40 * s)
    ft, ff = font(int(18 * s), "SemiBold"), font(int(16 * s), "Light")
    box_w = max(gw, d.textlength(title, font=ft) + 30 * s, d.textlength(foot, font=ff)) + 32 * s
    x0 = int(W - 48 * s - box_w)
    d.rounded_rectangle((x0, y0, x0 + box_w, y0 + gh + 92 * s), radius=int(14 * s), fill=(*INK, 175),
                        outline=(*STONE, 45), width=1)
    d.ellipse((x0 + 16 * s, y0 + 20 * s, x0 + 26 * s, y0 + 30 * s), fill=(*CLAY, 235))     # live dot
    d.text((x0 + 34 * s, y0 + 13 * s), title, font=ft, fill=(*PAPER, 235))
    gx = x0 + (box_w - gw) / 2
    gy = y0 + 46 * s
    for r in range(rows):
        for cc in range(cols):
            xx, yy = gx + cc * (cell + pad), gy + r * (cell + pad)
            rgb = tuple(int(u) for u in col[r, cc]) if known[r, cc] else (52, 48, 43)
            d.rectangle((xx, yy, xx + cell - 1, yy + cell - 1), fill=(*rgb, 240))
    d.text((x0 + 16 * s, gy + gh + 10 * s), foot, font=ff, fill=(*STONE, 220))


# ── cards ────────────────────────────────────────────────────────────────────

def _card_bg(W: int, H: int, t: float):
    """Cream paper, a touch darker at the edges, with a faint kraft dot grid (the height map's dots)
    drifting slowly."""
    from PIL import Image, ImageDraw
    ys = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    vig = np.clip(1.0 - 0.06 * (xs ** 2 + ys ** 2), 0, 1)[..., None]
    arr = np.array(PAPER, dtype=np.float32) * vig
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    s = H / 1080.0
    step = 36 * s
    off = (t * 18 * s) % step
    for yy in np.arange(-step, H + step, step):
        for xx in np.arange(-step, W + step, step):
            cx, cy = xx + off, yy
            dist = np.hypot((cx - W / 2) / W, (cy - H / 2) / H)
            a = int(max(0.0, 0.5 - dist) * 150)
            if a > 2:
                d.ellipse((cx - 2 * s, cy - 2 * s, cx + 2 * s, cy + 2 * s), fill=(*KRAFT, a))
    return Image.alpha_composite(img, layer)


def card(c: dict, W: int, H: int, j: int, n: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    s = H / 1080.0
    img = _card_bg(W, H, j / FPS)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = min(1.0, max(0.0, (j - 4) / 14.0)) * min(1.0, (n - j) / 10.0)
    kind = c.get("kind", "text")
    lines = c.get("lines", [])
    if kind == "end" and c.get("stats"):
        fw = font(int(96 * s), "SemiBold")
        sp = 18 * s
        ww = _spaced_width(d, "MICRODUCK", fw, sp)
        y = H * 0.17
        _spaced(d, ((W - ww) / 2, y), "MICRODUCK", fw, (*INK, int(255 * a)), sp)
        y += 150 * s
        d.rectangle((W / 2 - 36 * s, y - 14 * s, W / 2 + 36 * s, y - 10 * s), fill=(*CLAY, int(255 * a)))
        f = serif(int(46 * s))
        lw = d.textlength(lines[0], font=f)
        d.text(((W - lw) / 2, y + 4 * s), lines[0], font=f, fill=(*INK, int(240 * a)))
        y += 130 * s
        stats = c["stats"]
        colw = 420 * s
        x0 = (W - colw * len(stats)) / 2
        for i, stt in enumerate(stats):
            cx = x0 + colw * (i + 0.5)
            fn = serif(int(92 * s))
            col = ACCENT.get(stt.get("color", ""), INK)
            vw = d.textlength(stt["value"], font=fn)
            d.text((cx - vw / 2, y), stt["value"], font=fn, fill=(*col, int(255 * a)))
            for li, lab in enumerate(stt["label"].split("\n")):
                fl = font(int(24 * s), "Regular")
                lw = d.textlength(lab, font=fl)
                d.text((cx - lw / 2, y + 112 * s + li * 32 * s), lab, font=fl, fill=(*DUSK, int(235 * a)))
            if i:
                d.line((x0 + colw * i, y + 10 * s, x0 + colw * i, y + 160 * s), fill=(*KRAFT, int(160 * a)), width=max(1, int(2 * s)))
        y += 250 * s
        for line in lines[1:]:
            fl = font(int(22 * s), "Light")
            lw = d.textlength(line, font=fl)
            d.text(((W - lw) / 2, y), line, font=fl, fill=(*DUSK, int(220 * a)))
            y += 34 * s
    elif kind in ("title", "end"):
        fw = font(int((140 if kind == "title" else 112) * s), "SemiBold")
        sp = 20 * s
        ww = _spaced_width(d, "MICRODUCK", fw, sp)
        y = H / 2 - (130 if kind == "title" else 150) * s
        _spaced(d, ((W - ww) / 2, y), "MICRODUCK", fw, (*INK, int(255 * a)), sp)
        y += 215 * s if kind == "title" else 185 * s
        d.rectangle((W / 2 - 36 * s, y - 14 * s, W / 2 + 36 * s, y - 10 * s), fill=(*CLAY, int(255 * a)))
        for i, line in enumerate(lines):
            f = serif(int(48 * s)) if i == 0 else font(int(26 * s), "Regular")
            lw = d.textlength(line, font=f)
            d.text(((W - lw) / 2, y + (0 if i == 0 else 34 * s + i * 40 * s)), line, font=f,
                   fill=(*INK, int(240 * a)) if i == 0 else (*DUSK, int(230 * a)))
    else:
        y = H / 2 - 50 * s * len(lines)
        for i, line in enumerate(lines):
            ff = serif(int(84 * s)) if i == 0 else font(int(32 * s), "Light")
            lw = d.textlength(line, font=ff)
            d.text(((W - lw) / 2, y), line, font=ff, fill=(*INK, int(250 * a)) if i == 0 else (*DUSK, int(230 * a)))
            y += 110 * s if i == 0 else 46 * s
    out = np.asarray(Image.alpha_composite(img, layer).convert("RGB")).astype(np.float32)
    return (out * _alpha(j, n, fade=6)).astype(np.uint8)       # from / to black, like the clips


# ── commands ─────────────────────────────────────────────────────────────────

def pick(a) -> None:
    st = load_states(a.states)
    names = [str(x) for x in st["names"]]
    dt = float(st["dt"])
    rows = []
    for i in range(st["qpos"].shape[1]):
        t = names[st["types"][i]]
        if a.tile and t != a.tile:
            continue
        riser = st["riser"][i]
        if riser <= 0:
            continue
        fell = np.nonzero(st["fallen"][:, i])[0]
        top = np.nonzero(st["rise"][:, i] >= 0.5 * riser)[0]
        on_top = st["rise"][-25:, i].mean() >= 0.8 * riser
        fe = np.nonzero(np.diff(st["foot_edge"][:, i]) > 0)[0]
        rows.append((riser, i, t, int(st["levels"][i]), int(st["foot_edge"][-1, i]),
                     f"{top[0] * dt:4.1f}" if len(top) else "  - ", f"{fe[0] * dt:4.1f}" if len(fe) else "  - ",
                     f"{fell[0] * dt:4.1f}" if len(fell) else "  - ", on_top))
    print(f"{'riser':>6} {'hero':>4} {'tile':<13} row edges climb@ edge@ fell@ on_top")
    for r in sorted(rows):
        print(f"{100 * r[0]:5.1f}c {r[1]:4d} {r[2]:<13} {r[3]:3d} {r[4]:5d} {r[5]:>6} {r[6]:>5} {r[7]:>5} {r[8]}")


def edge_hits(st: dict, hero: int) -> np.ndarray:
    """Cumulative count of discrete edge hits (onsets of an edge contact), per state."""
    inc = np.diff(st["foot_edge"][:, hero], prepend=0) > 0
    onset = inc & ~np.concatenate([[False], inc[:-1]])
    return np.cumsum(onset)


def fall_index(st: dict, hero: int) -> int | None:
    fk = np.nonzero(st["fallen"][:, hero])[0]
    return int(fk[0]) if len(fk) else None


def clean_climb(st: dict, i: int) -> tuple[bool, bool, bool]:
    """(on top at the end, clean = on top with no edge contact, fell)."""
    riser = st["riser"][i]
    fell = fall_index(st, i) is not None
    on_top = (not fell) and st["rise"][-25:, i].mean() >= 0.8 * riser
    return on_top, on_top and st["foot_edge"][-1, i] == 0, fell


class Stages:
    """Recordings and restyled renderers, cached by (path, width)."""

    def __init__(self, seed: int):
        self.seed, self.states, self.stages = seed, {}, {}

    def get(self, path: str, W: int, H: int) -> tuple[dict, Stage]:
        if path not in self.states:
            self.states[path] = load_states(path)
        if (path, W) not in self.stages:
            self.stages[(path, W)] = build_stage(self.states[path], W, H, self.seed)
        return self.states[path], self.stages[(path, W)]

    def close(self):
        for stg in self.stages.values():
            stg.close()


def _pane_overlay(img: np.ndarray, st: dict, hero: int, k: int, label: str, sub: str, acc, fell: bool,
                  a_txt: float) -> np.ndarray:
    from PIL import Image, ImageDraw
    W, H = img.shape[1], img.shape[0]
    s = H / 1080.0
    base = Image.fromarray(img).convert("RGBA")
    if fell:                                                    # dim the frozen pane
        base = Image.alpha_composite(base, Image.new("RGBA", base.size, (*INK, 120)))
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    xa, ya = int(40 * s), int(H - 250 * s)
    fp = font(int(22 * s), "SemiBold")
    pw = _spaced_width(d, label, fp, 3 * s)
    d.rounded_rectangle((xa, ya, xa + pw + 32 * s, ya + 40 * s), radius=int(8 * s), fill=(*acc, int(235 * a_txt)))
    _spaced(d, (xa + 16 * s, ya + 6 * s), label, fp, (*INK, int(255 * a_txt)), 3 * s)
    d.text((xa + pw + 48 * s, ya + 6 * s), sub, font=font(int(24 * s), "Light"), fill=(*STONE, int(230 * a_txt)))
    hits = int(edge_hits(st, hero)[k])
    num = serif(int(96 * s))
    col = BRICK if hits else SAGE
    d.text((xa, ya + 54 * s), str(hits), font=num, fill=(*col, int(255 * a_txt)))
    nw = d.textlength(str(hits), font=num)
    d.text((xa + nw + 18 * s, ya + 108 * s), "edge hit" if hits == 1 else "edge hits", font=font(int(30 * s), "Regular"),
           fill=(*PAPER, int(235 * a_txt)))
    if fell:
        f = serif(int(120 * s))
        w = d.textlength("fell", font=f)
        d.text(((W - w) / 2, H * 0.36), "fell", font=f, fill=(*BRICK, 255))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def _split_chrome(img: np.ndarray, c: dict, a_txt: float) -> np.ndarray:
    """Wordmark, the honest tag and the step height over the divider."""
    from PIL import Image, ImageDraw
    W, H = img.shape[1], img.shape[0]
    s = H / 1080.0
    base = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    _spaced(d, (int(40 * s), int(40 * s)), "MICRODUCK", font(int(26 * s), "SemiBold"), (*PAPER, 235), 5 * s)
    ft = font(int(16 * s), "Medium")
    tw = _spaced_width(d, "SIMULATION", ft, 3 * s)
    x0, y0 = W - 40 * s - tw - 24 * s, int(40 * s)
    d.rounded_rectangle((x0, y0, x0 + tw + 24 * s, y0 + 28 * s), radius=int(14 * s), outline=(*STONE, 130), width=max(1, int(1.5 * s)))
    _spaced(d, (x0 + 12 * s, y0 + 4 * s), "SIMULATION", ft, (*STONE, 190), 3 * s)
    if c.get("height"):
        f = serif(int(54 * s))
        tw = d.textlength(c["height"], font=f)
        cap = font(int(16 * s), "Medium")
        cw = _spaced_width(d, "STEP HEIGHT", cap, 3 * s)
        bw = max(tw, cw) + 64 * s
        x0, y0 = (W - bw) / 2, 30 * s
        d.rounded_rectangle((x0, y0, x0 + bw, y0 + 104 * s), radius=int(18 * s), fill=(*PAPER, int(245 * a_txt)))
        _spaced(d, ((W - cw) / 2, y0 + 14 * s), "STEP HEIGHT", cap, (*DUSK, int(255 * a_txt)), 3 * s)
        d.text(((W - tw) / 2, y0 + 30 * s), c["height"], font=f, fill=(*INK, int(255 * a_txt)))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def item_frames(c: dict, stg: Stages, W: int, H: int, only: int | None = None):
    """Yield the frames of one board item; ``only`` = render just that output frame (stills)."""
    if c["type"] == "card":
        n = int(round(c.get("seconds", 3) * FPS))
        for j in range(n):
            if only is None or j == only:
                yield card(c, W, H, j, n)
        return
    if c["type"] == "clip":
        st, stage = stg.get(c["states"], W, H)
        hero = c["hero"]
        ts = source_times(c, st)
        trunk = smooth_trunk(st, hero)
        dt = float(st["dt"])
        stage.mark_risers(trunk[int(round(event_time(st, hero, c.get("center", "climb")) / dt))])
        n, fk = len(ts), fall_index(st, hero)
        for j, t in enumerate(ts):
            if only is not None and j != only:
                continue
            k = min(int(round(t / dt)), st["qpos"].shape[0] - 1)
            if fk is not None:                                  # the env teleports a fallen robot: hold
                k = min(k, fk - 1)
            img = stage.frame(st, k, hero, camera(c, trunk[k], j / max(n - 1, 1)), beams=c.get("beams", True))
            yield overlay(img, c, st, k, j, n)
        return
    # split: the same step and start, the left policy | the right policy, centred on the right's climb
    PW = (W - 4) // 2
    sides = []
    for side in ("left", "right"):
        st, stage = stg.get(c[side], PW, H)
        hero = c.get(f"hero_{side}", c["hero"])
        trunk = smooth_trunk(st, hero)
        sides.append(dict(st=st, stage=stage, hero=hero, trunk=trunk, fk=fall_index(st, hero),
                          label=c[f"{side}_label"], sub=c.get(f"{side}_sub", ""),
                          acc=ACCENT[c.get(f"{side}_color", "bad" if side == "left" else "good")],
                          perception=c.get(f"{side}_perception", True),
                          center=c.get(f"center_{side}", c.get("center", "climb"))))
    R = sides[1]
    ts = source_times({**c, "hero": R["hero"]}, R["st"])
    tc = event_time(R["st"], R["hero"], c.get("center", "climb"))
    for sd in sides:
        # ``align``: each pane plays around its own key moment (e.g. a slower policy's first riser hit
        # lined up with the other's climb); otherwise both share the right pane's clock.
        tcs = event_time(sd["st"], sd["hero"], sd["center"]) if c.get("align") else tc
        sd["offset"] = tcs - tc
        dt = float(sd["st"]["dt"])
        sd["stage"].mark_risers(sd["trunk"][int(round(tcs / dt))])
    n = len(ts)
    divider = np.full((H, W - 2 * PW, 3), np.array(INK, dtype=np.uint8), dtype=np.uint8)
    for j, t in enumerate(ts):
        if only is not None and j != only:
            continue
        a_txt = min(1.0, max(0.0, (j - 6) / 10.0)) * min(1.0, (n - j) / 8.0)
        panes = []
        for sd in sides:
            st, dt = sd["st"], float(sd["st"]["dt"])
            k = min(max(int(round((t + sd["offset"]) / dt)), 0), st["qpos"].shape[0] - 1)
            fell = sd["fk"] is not None and k >= sd["fk"]
            kk = k if not fell else max(sd["fk"] - 1, 0)       # a fallen robot freezes, dimmed, "fell"
            img = sd["stage"].frame(st, kk, sd["hero"], camera(c, sd["trunk"][kk], j / max(n - 1, 1)),
                                    perception=sd["perception"])
            panes.append(_pane_overlay(img, st, sd["hero"], kk, sd["label"], sd["sub"], sd["acc"], fell, a_txt))
        img = _split_chrome(np.concatenate([panes[0], divider, panes[1]], axis=1), c, a_txt)
        yield (img.astype(np.float32) * _alpha(j, n)).astype(np.uint8)


def item_length(c: dict, stg: Stages) -> int:
    if c["type"] == "card":
        return int(round(c.get("seconds", 3) * FPS))
    path, hero = (c["states"], c["hero"]) if c["type"] == "clip" else (c["right"], c.get("hero_right", c["hero"]))
    if path not in stg.states:
        stg.states[path] = load_states(path)
    return len(source_times({**c, "hero": hero}, stg.states[path]))


def pair(a) -> None:
    """Robots of two recordings made on the same grid (same index = same tile, row and start), side by
    side, plus clean-climb rates per riser height."""
    L, R = load_states(a.left), load_states(a.right)
    names = [str(x) for x in L["names"]]
    dt = float(R["dt"])
    by_h: dict = {}
    print(f"{'riser':>6} {'hero':>4} | {'L hits':>6} {'L top':>5} {'L fell@':>7} | {'R hits':>6} {'R climb@':>8} {'R fell@':>7}")
    for i in np.argsort(L["riser"]):
        if names[L["types"][i]] != a.tile or L["riser"][i] <= 0:
            continue
        lt, lc, lf = clean_climb(L, i)
        rt, rc, rf = clean_climb(R, i)
        h = round(100 * float(R["riser"][i]), 1)
        b = by_h.setdefault(h, [0, 0, 0, 0, 0])
        b[0] += 1
        b[1] += lc
        b[2] += rc
        b[3] += lf
        b[4] += rf
        top = np.nonzero(R["rise"][:, i] >= 0.5 * R["riser"][i])[0]
        flk, frk = fall_index(L, i), fall_index(R, i)
        print(f"{h:5.1f}c {i:4d} | {int(edge_hits(L, i)[-1]):6d} {str(lt):>5} {f'{flk * dt:.1f}' if flk else '-':>7} | "
              f"{int(edge_hits(R, i)[-1]):6d} {f'{top[0] * dt:.1f}' if len(top) else '-':>8} {f'{frk * dt:.1f}' if frk else '-':>7}")
    print("\nper height: n, clean left, clean right, falls left, falls right")
    for h, b in sorted(by_h.items()):
        print(f"  {h:4.1f} cm  n={b[0]:3d}  clean {b[1]:3d} ({100 * b[1] / b[0]:3.0f} %) vs {b[2]:3d} ({100 * b[2] / b[0]:3.0f} %)"
              f"   falls {b[3]} vs {b[4]}")
    tot = np.array(list(by_h.values())).sum(axis=0)
    print(f"  all      n={tot[0]:3d}  clean {100 * tot[1] / tot[0]:.0f} % vs {100 * tot[2] / tot[0]:.0f} %   falls {tot[3]} vs {tot[4]}")


def still(a) -> None:
    from PIL import Image
    board = json.load(open(a.board))
    W, H = board.get("width", 1920), board.get("height", 1080)
    c = board["items"][a.clip]
    stg = Stages(a.seed)
    n = item_length(c, stg)
    j = int(a.at * (n - 1))
    for img in item_frames(c, stg, W, H, only=j):
        Image.fromarray(img).save(a.out)
    stg.close()
    print("still →", a.out, f"(item {a.clip}, frame {j}/{n})")


def cut(a) -> None:
    import imageio.v2 as imageio
    board = json.load(open(a.board))
    W, H = board.get("width", 1920), board.get("height", 1080)
    stg = Stages(a.seed)
    writer = imageio.get_writer(a.video, fps=FPS, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    total = 0
    try:
        for i, c in enumerate(board["items"]):
            t0 = total
            for img in item_frames(c, stg, W, H):
                writer.append_data(img)
                total += 1
            print(f"item {i}: {c['type']} {c.get('kind', c.get('height', c.get('badge', '')))} {(total - t0) / FPS:.1f} s")
    finally:
        writer.close()
        stg.close()
    print(f"video → {a.video} ({total / FPS:.1f} s, no audio)")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("pick", help="list the robots of a recording: riser, edge contacts, climb / edge / fall times")
    k.add_argument("--states", required=True)
    k.add_argument("--tile", default=None)
    q = sub.add_parser("pair", help="two recordings on the same grid side by side, with clean-climb rates per height")
    q.add_argument("--left", required=True)
    q.add_argument("--right", required=True)
    q.add_argument("--tile", default="single_step")
    s = sub.add_parser("still", help="one frame of one board item, to check framing")
    s.add_argument("--board", required=True)
    s.add_argument("--clip", type=int, required=True)
    s.add_argument("--at", type=float, default=0.5, help="position in the item, 0..1")
    s.add_argument("--out", default="still.png")
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("cut", help="render the whole board to a video (no audio)")
    c.add_argument("--board", required=True)
    c.add_argument("--video", default="microduck_steps.mp4")
    c.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"pick": pick, "pair": pair, "still": still, "cut": cut}[a.cmd](a)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
