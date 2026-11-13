"""Static step test: can the Microduck's joints place the leading foot flat on the next tread
while the trailing leg still carries all the weight? Step DOWN against step UP.

The moment compared is the same in both directions: the leading foot has just reached the other
tread, flat, and the trailing leg holds all the weight with the CoM over its foot.
- DOWN: the trailing foot is on the upper tread (toe 3 mm behind the edge), the leading foot
  reaches `h` DOWN and ahead. The trailing leg must crouch so that the other foot gets down.
- UP: the trailing foot is on the lower tread (toe 3 mm before the riser), the leading foot is
  placed `h` UP and ahead. The trailing leg can stay tall.
If no such pose exists, a step without a drop is impossible: the leading foot can only get flat
on the lower tread by letting the body fall the rest of the way.

For every riser height `h` the script scans trunk height and trunk pitch. For each it solves the
rest by least squares (8 unknowns: trunk x, hip pitch, knee and ankle of each leg, and the
foot-centre distance `d`).

The constraints:
- both feet flat in pitch;
- the stance sole at x = z = 0;
- the free sole at (d, ∓h), with the whole free foot past the edge: d ≥ 60.4 mm (two foot
  half-lengths plus 3 mm on each side of the edge), d ≤ 120 mm;
- the CoM over the stance foot in x.

A pose counts only if no part of the robot (meshes, convex hulls) sinks more than 1 mm into the
step (the foot that stands on it excepted). This matters going down: the duck's knee points
BACKWARD, toward the riser.

The test is SAGITTAL: trunk roll 0, hip roll at STAND. The duck cannot stand on one leg
statically. It has no ankle roll, so a flat foot cannot carry the CoM sideways over it (a
full-3-D solve found no pose even on flat ground). In walking, lateral balance is dynamic, and
the single support lasts a fraction of a second. So the stance foot takes the whole weight here,
and only the pitch joints (hip pitch, knee, ankle) are judged. Those carry the lowering. The
roll torques in the output are not meaningful.

The static joint torques follow from the floating-base equation with the stance foot as the only
contact. With q̇ = q̈ = 0: J_baseᵀ f = g_base gives the foot wrench f, and τ = g_joints − J_jointsᵀ f.

Per (direction, h) it reports:
- the most upright pose (highest trunk), which the video shows;
- the pose that needs the least torque (the minimum over the scan of the largest |τ| / limit over
  the pitch joints) and the worst pose;
- the range of trunk heights at which any pose holds;
- how many poses were dropped because they hit the step, and with which part.

Limits:
- 0.96 N·m: the actuators' forcerange, BAM's kt·vin/R at 7.4 V. Training samples 6.5–8.2 V,
  which gives 0.85–1.07.
- 0.64 N·m: the XL330's firmware current limit, kt·1.75 A (see stairs_feasibility.py).

Model: robot_walk.xml, head at the STAND keyframe, hip yaw 0. It is a static bound: no dynamics,
no backlash, no friction, no self-collision (a deep knee bend could press the shin into the thigh).

    uv run python scripts/eval/step_down_static.py --json logs/staircase/step_static.json
    MUJOCO_GL=egl uv run python scripts/eval/step_down_static.py --from-json logs/staircase/step_static.json \\
        --video logs/staircase/video/step_static.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

XML = Path(__file__).resolve().parents[2] / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
STANCE, FREE = "left", "right"
# qpos addresses on robot_walk (after the 7-D freejoint): roll, pitch, knee, ankle per leg
LEG = {"left": (8, 9, 10, 11), "right": (17, 18, 19, 20)}
HIP_YAW = (7, 16)
FOOT = {"left": "left_foot_collision", "right": "right_foot_collision"}
LIM_SIM, LIM_FW = 0.96, 0.64            # N·m
EDGE_GAP = 0.003                        # m, stance toe (down) / heel (up) to the edge
JOINT_MARGIN = 0.03                     # rad inside the joint range
D_MAX = 0.12                            # m, longest foot-centre distance tried
PEN_TOL = 0.001                         # m, allowed penetration of a mesh into the step


class Robot:
    def __init__(self, m: mujoco.MjModel | None = None, on_step: str = STANCE):
        self.m = m if m is not None else mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("STAND").id)
        self.stand = self.d.qpos.copy()
        self.gid = {s: self.m.geom(n).id for s, n in FOOT.items()}
        self.half_thick = float(self.m.geom_size[self.gid["left"], 0])     # box x axis = the foot's up
        self.half_len = float(self.m.geom_size[self.gid["left"], 2])       # box z axis = the foot's forward
        self.act_dof = np.array([self.m.jnt_dofadr[self.m.actuator_trnid[i, 0]] for i in range(self.m.nu)])
        self.act_names = [self.m.actuator(i).name for i in range(self.m.nu)]
        lo, hi = self.m.jnt_range[:, 0], self.m.jnt_range[:, 1]
        self.qlo = np.full(self.m.nq, -np.inf)
        self.qhi = np.full(self.m.nq, np.inf)
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                a = self.m.jnt_qposadr[j]
                self.qlo[a], self.qhi[a] = lo[j] + JOINT_MARGIN, hi[j] - JOINT_MARGIN
        self.d_min = 2 * self.half_len + 2 * EDGE_GAP
        names = [self.m.geom(g).name for g in range(self.m.ngeom)]
        self.step_gid = names.index("step") if "step" in names else None
        rest = self.m.body(f"ankle_{on_step}").id                         # the foot standing on the step
        self.check_gids = [g for g in range(self.m.ngeom)
                           if self.m.geom_bodyid[g] > 0 and self.m.geom_bodyid[g] != rest]

    def clearance(self) -> tuple[float, str]:
        """Smallest signed distance (m) from any robot geom but the resting foot's to the step."""
        if self.step_gid is None:
            return np.inf, ""
        ft = np.zeros(6)
        best = (np.inf, "")
        for g in self.check_gids:
            dist = mujoco.mj_geomDistance(self.m, self.d, g, self.step_gid, 0.05, ft)
            if dist < best[0]:
                best = (dist, self.m.body(self.m.geom_bodyid[g]).name)
        return best

    def foot(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        g = self.gid[side]
        R = self.d.geom_xmat[g].reshape(3, 3)
        up = R[:, 0]
        return self.d.geom_xpos[g] - self.half_thick * up, up

    def set(self, z: float, pitch: float, u: np.ndarray) -> np.ndarray:
        """u = [x, stance pitch/knee/ankle, free pitch/knee/ankle, d] → qpos (trunk roll 0, y 0, hip roll at STAND)."""
        q = self.stand.copy()
        q[0], q[1], q[2] = u[0], 0.0, z
        q[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]   # pitch about y; + = nose down
        q[list(HIP_YAW)] = 0.0
        q[list(LEG[STANCE][1:])] = u[1:4]
        q[list(LEG[FREE][1:])] = u[4:7]
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return q

    def residual(self, u, z, pitch, hz) -> np.ndarray:
        """hz: the free sole's target height (−h going down, +h going up)."""
        self.set(z, pitch, u)
        dx = u[7]
        s, s_up = self.foot(STANCE)
        f, f_up = self.foot(FREE)
        com = self.d.subtree_com[1]
        return np.array([s[0], s[2], s_up[0], f[0] - dx, f[2] - hz, f_up[0], com[0] - s[0]]) * 100.0

    def torques(self) -> tuple[np.ndarray, np.ndarray]:
        """Static single support on the stance foot: actuator torques (N·m) and the CoP offset in
        the foot (m, forward / left) for the current qpos."""
        m, d = self.m, self.d
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        sole, _ = self.foot(STANCE)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, sole, m.geom_bodyid[self.gid[STANCE]])
        J = np.vstack([jacp, jacr])                                   # (6, nv)
        bias = d.qfrc_bias.copy()                                     # gravity at q̇ = 0
        f = np.linalg.solve(J[:, :6].T, bias[:6])                     # foot wrench (world, at the sole)
        tau = bias - J.T @ f
        cop = np.array([-f[4] / f[2], f[3] / f[2]]) if abs(f[2]) > 1e-6 else np.full(2, np.nan)
        return tau[self.act_dof], cop


def u0(r: Robot) -> np.ndarray:
    q = r.stand
    return np.concatenate([[0.0], q[list(LEG[STANCE][1:])], q[list(LEG[FREE][1:])], [0.0]])


def _ls(r: Robot, g, z, p, h, dlo, dhi, lo, hi) -> tuple[np.ndarray, bool]:
    lo, hi = np.r_[lo, dlo], np.r_[hi, dhi]
    g = np.clip(g, lo + 1e-9, hi - 1e-9)
    sol = least_squares(r.residual, g, args=(z, p, h), bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=300)
    return sol.x, bool(np.abs(sol.fun).max() <= 0.05)                # 0.5 mm / 0.0005 rad


def solve_case(r: Robot, h: float, sign: float, zs: np.ndarray, pitches: np.ndarray, n_ramp: int = 8,
               n_rand: int = 8, seed: int = 0) -> list[dict]:
    """Every (z, pitch) of the scan that has a pose, for the free sole at height `h` (−: down, +: up)
    and ahead (sign +1) or beside (0, flat baseline, d = 0). Least squares is local, so each pose is tried
    from three kinds of start, in order, until one converges:
    1. the pose found at the previous (higher) z of the same pitch;
    2. a continuation: the feet-together pose at this (z, pitch), then the free foot moved to
       (±d_min + 1.5 cm, h) in `n_ramp` steps, then d set free;
    3. `n_rand` random joint guesses around STAND."""
    lo = np.concatenate([[-0.2], r.qlo[list(LEG[STANCE][1:])], r.qlo[list(LEG[FREE][1:])]])
    hi = np.concatenate([[0.2], r.qhi[list(LEG[STANCE][1:])], r.qhi[list(LEG[FREE][1:])]])
    if sign > 0:
        dlo, dhi = r.d_min, D_MAX
    else:
        dlo, dhi = -1e-6, 1e-6
    d_ramp = 0.0 if sign == 0 else sign * (r.d_min + 0.015)
    rng = np.random.default_rng(seed)
    judged = [i for i, n in enumerate(r.act_names) if n.endswith(("hip_pitch", "knee", "ankle"))]
    out = []
    for p in pitches:
        base = u0(r)
        prev = None
        for z in zs[::-1]:                                            # high → low, warm-started
            base, _ = _ls(r, base, z, p, 0.0, -1e-6, 1e-6, lo, hi)    # feet together at this (z, pitch)
            x, ok = _ls(r, prev, z, p, h, dlo, dhi, lo, hi) if prev is not None else (None, False)
            if not ok:
                g = base
                for t in np.linspace(0.0, 1.0, n_ramp + 1)[1:]:
                    dt = t * d_ramp
                    g, _ = _ls(r, g, z, p, t * h, dt - 1e-6, dt + 1e-6, lo, hi)
                x, ok = _ls(r, g, z, p, h, dlo, dhi, lo, hi)
            for _ in range(n_rand):
                if ok:
                    break
                g0 = u0(r) + np.r_[0.0, rng.uniform(-0.8, 0.8, 6), 0.0]
                g0[7] = rng.uniform(dlo, dhi)
                x, ok = _ls(r, g0, z, p, h, dlo, dhi, lo, hi)
            if not ok:
                continue
            r.set(z, p, x)
            clear, part = r.clearance()
            if clear < -PEN_TOL and sign > 0:                          # slide along the family: other d
                for dpin in np.linspace(r.d_min + 0.01, D_MAX, 6):
                    x2, ok2 = _ls(r, x, z, p, h, dpin - 1e-6, dpin + 1e-6, lo, hi)
                    if ok2:
                        r.set(z, p, x2)
                        c2, p2 = r.clearance()
                        if c2 >= -PEN_TOL:
                            x, clear, part = x2, c2, p2
                            break
            prev = x
            q = r.set(z, p, x)
            tau, cop = r.torques()
            util = np.abs(tau) / LIM_SIM
            k = max(judged, key=lambda i: util[i])
            out.append({"z": float(z), "pitch_deg": float(np.degrees(p)), "d_mm": float(abs(x[7]) * 1000),
                        "qpos": q.tolist(), "tau": tau.tolist(), "max_util": float(util[k]),
                        "max_joint": r.act_names[k], "cop_mm": (1000 * cop).tolist(),
                        "clear_mm": float(clear * 1000), "clear_part": part, "hits": bool(clear < -PEN_TOL)})
    return out


def summarize(r: Robot, sols: list[dict]) -> dict:
    from collections import Counter

    n_hit = sum(s["hits"] for s in sols)
    hit_parts = dict(Counter(s["clear_part"] for s in sols if s["hits"]))
    sols = [s for s in sols if not s["hits"]]
    if not sols:
        return {"n": 0, "n_hit": n_hit, "hit_parts": hit_parts}
    best = min(sols, key=lambda s: s["max_util"])
    tallest = max(sols, key=lambda s: (round(s["z"], 4), -s["max_util"]))     # least crouch
    i = {n: r.act_names.index(n) for n in r.act_names}
    t = best["tau"]
    ok = [s for s in sols if s["max_util"] <= 1.0]
    okfw = [s for s in sols if s["max_util"] * LIM_SIM <= LIM_FW]
    q = np.array(best["qpos"])
    return {
        "n": len(sols), "best": best, "tallest": tallest, "hit_parts": hit_parts,
        "util_range": [min(s["max_util"] for s in sols), max(s["max_util"] for s in sols)],
        "knee_nm": abs(t[i[f"{STANCE}_knee"]]), "hip_pitch_nm": abs(t[i[f"{STANCE}_hip_pitch"]]),
        "ankle_nm": abs(t[i[f"{STANCE}_ankle"]]), "hip_roll_nm": abs(t[i[f"{STANCE}_hip_roll"]]),
        "knee_deg": float(np.degrees(q[LEG[STANCE][2]])),
        "z_ok_range": [min(s["z"] for s in ok), max(s["z"] for s in ok)] if ok else None,
        "z_okfw_range": [min(s["z"] for s in okfw), max(s["z"] for s in okfw)] if okfw else None,
        "n_ok": len(ok), "n_okfw": len(okfw), "n_hit": n_hit,
        "z_range": [min(s["z"] for s in sols), max(s["z"] for s in sols)],
        "d_range_mm": [min(s["d_mm"] for s in sols), max(s["d_mm"] for s in sols)],
    }


def _case(job: tuple) -> dict:
    direction, h, zs, pitches, n_rand, keep = job
    sign, hz, on_step = {"flat": (0.0, 0.0, STANCE), "down": (1.0, -h / 100.0, STANCE), "up": (1.0, h / 100.0, FREE)}[direction]
    r = Robot(_scene(h / 100.0, direction), on_step=on_step)
    sols = solve_case(r, hz, sign, zs, pitches, n_rand=n_rand)
    s = summarize(r, sols)
    out = {"direction": direction, "h_cm": h, **s}
    if keep:                                      # every valid pose, for the run-15 spawn bank
        out["poses"] = [{k: v for k, v in x.items() if k in ("z", "pitch_deg", "d_mm", "qpos", "max_util", "clear_mm")}
                        for x in sols if not x["hits"]]
    return out


def run(a) -> dict:
    from multiprocessing import Pool

    zs = np.arange(a.z_min, a.z_max + 1e-9, a.z_step)
    pitches = np.radians(np.arange(a.pitch_min, a.pitch_max + 1e-9, a.pitch_step))
    r = Robot()
    res = {"args": vars(a), "limits_nm": {"sim": LIM_SIM, "firmware": LIM_FW}, "act_names": r.act_names,
           "foot_half_len": r.half_len, "d_min_mm": r.d_min * 1000, "cases": []}
    # baseline: stand on one leg on flat ground, the free foot beside it just touching
    jobs = [("flat", 0.0, zs, pitches, a.n_rand, False)] if "flat" in a.directions else []
    jobs += [(dr, h, zs, pitches, a.n_rand, a.keep_poses) for h in a.heights for dr in ("down", "up") if dr in a.directions]
    with Pool(min(a.workers, len(jobs))) as pool:
        for c in pool.imap(_case, jobs):
            res["cases"].append(c)
            tag = f"  {c['direction']:>4} h {c['h_cm']:3.0f} cm"
            if c["n"]:
                b = c["best"]
                print(f"{tag} | poses {c['n']:3d} (+{c['n_hit']} hit the step) | best: max {b['max_util'] * LIM_SIM:.2f} N·m "
                      f"({b['max_util']:4.0%}, {b['max_joint']}) knee {c['knee_nm']:.2f} hip {c['hip_pitch_nm']:.2f} "
                      f"ankle {c['ankle_nm']:.2f} | trunk z {b['z'] * 100:4.1f} cm pitch {b['pitch_deg']:+5.1f}° d {b['d_mm']:3.0f} mm "
                      f"knee {c['knee_deg']:+5.0f}° | trunk z range {c['z_range'][0] * 100:.1f}-{c['z_range'][1] * 100:.1f} cm "
                      f"| under 0.64: {c['n_okfw']:3d} | tallest: z {c['tallest']['z'] * 100:.1f} pitch {c['tallest']['pitch_deg']:+.0f} "
                      f"d {c['tallest']['d_mm']:.0f} max {c['tallest']['max_util'] * LIM_SIM:.2f} N·m | worst pose {c['util_range'][1] * LIM_SIM:.2f} N·m "
                      f"| hits {c['hit_parts']}", flush=True)
            else:
                print(f"{tag} | no pose ({c['n_hit']} hit the step: {c['hit_parts']}): out of reach or joint limits", flush=True)
    return res


# ── video ────────────────────────────────────────────────────────────────────

def _scene(h: float, direction: str) -> mujoco.MjModel:
    """The stance sole sits at the origin. DOWN: it stands on the step, whose edge is 3 mm ahead of
    its toe, and the floor is h below. UP: it stands on the floor and the riser is 3 mm ahead."""
    spec = mujoco.MjSpec.from_file(str(XML))
    edge = 0.0272 + EDGE_GAP
    if h > 0 and direction == "down":
        next(g for g in spec.geoms if g.name == "floor").pos = [0, 0, -h]
        spec.worldbody.add_geom(name="step", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.3, h / 2],
                                pos=[edge - 0.25, 0, -h / 2], rgba=[0.45, 0.55, 0.85, 1], contype=0, conaffinity=0)
    elif h > 0 and direction == "up":
        spec.worldbody.add_geom(name="step", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.3, h / 2],
                                pos=[edge + 0.25, 0, h / 2], rgba=[0.45, 0.55, 0.85, 1], contype=0, conaffinity=0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    return spec.compile()


PART = {f"upper_leg_{FREE}": "front thigh (knee)", "leg_2": "front shin", "leg": "rear shin",
        f"upper_leg_{STANCE}": "rear thigh (knee)"}


def render(a, res: dict) -> None:
    """Per height: STEP DOWN | STEP UP, the most upright pose that works, seen from the side (the
    front leg nearest the camera), with the rear (loaded) leg's torques. Then an end card."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
            if Path(f).exists():
                return ImageFont.truetype(f, sz)
        return ImageFont.load_default()

    def txt(dr, xy, t, sz, fill=(255, 255, 255)):
        dr.text(xy, t, font=font(sz), fill=fill, stroke_width=2, stroke_fill=(0, 0, 0))

    W, H, fps = 1280, 720, 25
    pane_w = W // 2
    cases = {(c["direction"], c["h_cm"]): c for c in res["cases"]}
    names = res["act_names"]
    writer = imageio.get_writer(a.video, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    stills = []
    try:
        for h in a.video_heights:
            panes = []
            for direction in ("down", "up"):
                c = cases.get((direction, h))
                m = _scene(h / 100.0, direction)
                d = mujoco.MjData(m)
                if c and c.get("n"):
                    d.qpos[:] = c["tallest"]["qpos"]
                else:
                    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)       # just stands at the step
                mujoco.mj_forward(m, d)
                panes.append((direction, c, m, d, mujoco.Renderer(m, H, pane_w)))
            nfr = int(a.seconds_per_height * fps)
            for k in range(nfr):
                imgs = []
                for direction, c, m, d, rend in panes:
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = [0.03, 0.0, 0.045]
                    cam.distance = 0.75
                    cam.elevation = -6.0
                    cam.azimuth = 90.0 - 14.0 * np.sin(2 * np.pi * k / nfr)       # side view, forward = right
                    rend.update_scene(d, camera=cam)
                    pil = Image.fromarray(rend.render())
                    dr = ImageDraw.Draw(pil)
                    up = direction == "up"
                    txt(dr, (20, 14), f"{'STEP UP' if up else 'STEP DOWN'} · riser {h:.0f} cm", 32)
                    txt(dr, (20, 54), f"all weight on the rear foot ({'lower' if up else 'upper'} tread)", 18, (230, 230, 230))
                    txt(dr, (20, 78), f"front foot placed flat on the {'upper' if up else 'lower'} tread", 18, (230, 230, 230))
                    if c and c.get("n"):
                        b = c["tallest"]
                        lean = b["pitch_deg"]
                        lean_s = "upright" if abs(lean) < 2.5 else f"leans {'back' if lean < 0 else 'forward'} {abs(lean):.0f}°"
                        txt(dr, (20, H - 196), f"trunk {b['z'] * 100:.1f} cm high (stands at 12) · {lean_s}", 20,
                            (255, 230, 120) if b["z"] < 0.105 else (255, 255, 255))
                        y = H - 160
                        for lab, jn in (("knee", f"{STANCE}_knee"), ("hip", f"{STANCE}_hip_pitch"),
                                        ("ankle", f"{STANCE}_ankle")):
                            t = abs(b["tau"][names.index(jn)])
                            frac = min(t / LIM_SIM, 1.2)
                            col = (90, 200, 90) if t <= LIM_FW else (235, 180, 60) if t <= LIM_SIM else (235, 70, 60)
                            txt(dr, (20, y), f"rear {lab}", 18)
                            x0, wbar = 150, 300
                            dr.rectangle((x0, y + 3, x0 + wbar, y + 21), outline=(255, 255, 255), width=1)
                            dr.rectangle((x0, y + 3, x0 + int(wbar * frac / 1.0), y + 21), fill=col)
                            x64 = x0 + int(wbar * LIM_FW / LIM_SIM)
                            dr.line((x64, y - 2, x64, y + 26), fill=(255, 255, 255), width=2)
                            dr.line((x0 + wbar, y - 2, x0 + wbar, y + 26), fill=(255, 80, 80), width=3)
                            txt(dr, (x0 + wbar + 12, y), f"{t:.2f} N·m", 18)
                            y += 30
                        txt(dr, (20, y + 6), "white line 0.64 N·m (servo current limit) · red 0.96 (max)", 15, (220, 220, 220))
                    else:
                        txt(dr, (20, H - 170), "NO POSE", 44, (235, 70, 60))
                        if c and c.get("n_hit", 0) >= 5:
                            part = max(c["hit_parts"], key=c["hit_parts"].get)
                            why = [f"the foot gets there, but the", f"{PART.get(part, part)} hits the step"]
                        elif c and c.get("n_hit"):
                            part = max(c["hit_parts"], key=c["hit_parts"].get)
                            why = ["the front leg is too short (the one", f"pose that reaches puts the {PART.get(part, part).split(' ')[1]} into the step)"]
                        else:
                            why = ["the front leg is too short:", "it cannot reach the lower tread flat"]
                        txt(dr, (20, H - 110), why[0], 22)
                        txt(dr, (20, H - 80), why[1], 22)
                    imgs.append(np.asarray(pil.convert("RGB")))
                frame = np.concatenate(imgs, axis=1)
                frame[:, pane_w - 2:pane_w + 2] = 0
                writer.append_data(frame)
                if k == 0:
                    p = Path(a.video).with_name(Path(a.video).stem + f"_h{h:.0f}.png")
                    Image.fromarray(frame).save(p)
                    stills.append(str(p))
            for *_, rend in panes:
                rend.close()
        # end card: the measured table
        card = Image.new("RGB", (W, H), (22, 30, 44))
        dr = ImageDraw.Draw(card)
        txt(dr, (60, 40), "Static test: front foot flat on the next tread, all weight on the rear leg", 30)
        txt(dr, (60, 120), "riser", 26, (200, 200, 200))
        txt(dr, (60, 175), "STEP DOWN: highest trunk", 24)
        txt(dr, (60, 225), "STEP UP: highest trunk", 24)
        txt(dr, (60, 275), "worst rear-leg torque", 24)
        for j, h in enumerate(a.video_heights):
            x = 470 + j * 125
            txt(dr, (x, 120), f"{h:.0f} cm", 26, (200, 200, 200))
            for row, direction in ((175, "down"), (225, "up")):
                c = cases.get((direction, h))
                ok = bool(c and c.get("n"))
                v = f"{c['tallest']['z'] * 100:.1f}" if ok else "none"
                txt(dr, (x, row), v, 24, ((255, 230, 120) if direction == "down" else (120, 220, 120)) if ok else (235, 70, 60))
            ws = [cases[(dr_, h)]["util_range"][1] * LIM_SIM for dr_ in ("down", "up") if cases.get((dr_, h), {}).get("n")]
            txt(dr, (x, 275), f"{max(ws):.2f}" if ws else "–", 24)
        lines = ["trunk height in cm, standing = 12 · torques in N·m, servo current limit 0.64, max 0.96",
                 "· torque is not the problem: every pose that exists needs under half the current limit",
                 "· going down the rear leg must crouch and lean back, deeper with every cm; at 6 cm there is no pose",
                 "· going up the rear leg stays tall up to 5 cm; at 6-7 cm the front shin hits the step nose",
                 "· static only: no dynamics, no backlash; in sim the step-up policy climbs 7.7 cm with momentum"]
        for i, l in enumerate(lines):
            txt(dr, (60, 360 + i * 48), l, 20 if i else 18, (230, 230, 230) if i else (170, 170, 170))
        arr = np.asarray(card)
        for _ in range(int(5 * fps)):
            writer.append_data(arr)
        p = Path(a.video).with_name(Path(a.video).stem + "_card.png")
        card.save(p)
        stills.append(str(p))
    finally:
        writer.close()
    print("video →", a.video, "| stills:", " ".join(stills))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heights", type=float, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--z-min", type=float, default=0.04)
    p.add_argument("--z-max", type=float, default=0.13)
    p.add_argument("--z-step", type=float, default=0.0025)
    p.add_argument("--pitch-min", type=float, default=-30.0)
    p.add_argument("--pitch-max", type=float, default=30.0)
    p.add_argument("--pitch-step", type=float, default=5.0)
    p.add_argument("--n-rand", type=int, default=8)
    p.add_argument("--workers", type=int, default=13)
    p.add_argument("--directions", nargs="+", default=["flat", "down", "up"])
    p.add_argument("--keep-poses", action="store_true", help="store every valid pose (run-15 spawn bank)")
    p.add_argument("--json", default=None)
    p.add_argument("--from-json", default=None, help="skip the scan, render from a saved result")
    p.add_argument("--video", default=None)
    p.add_argument("--video-heights", type=float, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--seconds-per-height", type=float, default=3.0)
    a = p.parse_args()
    if a.from_json:
        res = json.load(open(a.from_json))
    else:
        res = run(a)
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(a.json, "w"))
            print("wrote", a.json)
    if a.video:
        render(a, res)


if __name__ == "__main__":
    main()
