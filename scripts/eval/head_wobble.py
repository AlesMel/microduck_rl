"""Head wobble on the stairs: what the head does, and why, per policy.

Records, per control step and robot, on the stairs battery's terrain (``stairs_battery.py``: one
straight staircase per riser height, up or ``--down``; a 0 cm column is a flat baseline):
  - the 14 actions (the head = servo idx 5-8: neck_pitch, head_pitch, head_yaw, head_roll) and,
    for the 4 head joints, the servo qpos and the backlash qpos (offsets from the default pose);
  - the head (``tof`` site: x forward, y left, z up) and the trunk: world pitch / roll and the
    angular velocity in the trunk's heading frame (roll rate about forward, pitch rate about left,
    + = nose down), the head's position;
  - the feet: contact, site z; the trunk position; falls / resets;
  - the ToF scan in the obs: whether the stairs switch would pick the down policy
    (``stairs_switch.downhill_ahead_mask``), the deepest row-mean drop ahead, and the mean |Δ| of
    the known scan cells from the previous step.
Then analyses each robot's window on the stairs (from 25 cm before the first edge to 25 cm past
the last one; 2 s → end on the flat column; cut 1 s before a fall or at leaving the tile):
  - band RMS of the 1 s high-passed signals (the wobble, ≥ ~1 Hz), Welch PSDs, dominant frequency;
  - the step rate and the share of the wobble's variance locked to the stride phase (left-foot
    touchdown to touchdown) and to the staircase phase (trunk x within a tread);
  - commanded vs passive: per head joint the RMS of the action, of the output angle (servo +
    backlash), of the backlash alone, and their correlation; the head's world pitch split into the
    trunk's and the neck chain's share;
  - ToF-driven: switch flips and the head action change at them, |Δscan| vs |Δaction|.

    uv run python scripts/eval/head_wobble.py --checkpoint ckpts/stairs/stairs_down_18d.pt --down \\
        --heights 0 2 3 4 --num-envs 600 --label down18d --out head_down18d
    uv run python scripts/eval/head_wobble.py --analyze head_down18d.npz [more.npz ...] --json summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

HEAD = (5, 6, 7, 8)
HEAD_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
LEGS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
HP_S = 1.0              # s, the high-pass's moving-average window (the "wobble" = what is faster)
EDGE_PAD = 0.25         # m before the first / past the last edge
FLAT_T0 = 2.0           # s, flat column window start
PRE_FALL = 1.0          # s cut before a fall
MIN_WIN = 3.0           # s, shorter windows are dropped
NPERSEG = 128           # Welch segment (2.56 s at 50 Hz)
PHASE_BINS = 20


# ── recording (GPU box) ─────────────────────────────────────────────────────

def _rotm(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _pitch_roll_yaw(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fx, ly = R[..., :, 0], R[..., :, 1]
    pitch = torch.atan2(-fx[..., 2], torch.hypot(fx[..., 0], fx[..., 1]))       # + = nose down
    roll = torch.asin(ly[..., 2].clamp(-1, 1))                                   # + = left side up
    yaw = torch.atan2(fx[..., 1], fx[..., 0])
    return pitch, roll, yaw


def _heading_frame(w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * w[:, 0] + s * w[:, 1], -s * w[:, 0] + c * w[:, 1], w[:, 2]], -1)


def record(a) -> None:
    from _common import command_obs, load_policy
    from stairs_battery import riser_table, staircase_hook
    from view_terrain_env import _terrain_layout, build_env

    from mjlab_microduck.tasks import mdp as microduck_mdp
    from mjlab_microduck.tasks.stairs_switch import SCAN_CFG, _rows_ahead, downhill_ahead_mask

    hook, length = staircase_hook([(float(h) / 100.0, a.tread) for h in a.heights], a.steps, a.approach, a.top,
                                  a.width, a.rows, a.map_x, 0.0, down=a.down)
    env = build_env(a.task, a.num_envs, a.seed, a.device, fixed_heading=False, cfg_hook=hook)
    policy = load_policy(a.checkpoint, device=a.device)
    head_cmd = {"zero": None, "down": (-0.22, 0.22, 0.0, 0.0)}[a.head]
    robot = env.scene["robot"]
    servo = microduck_mdp._servo_joint_ids(env, robot)
    names = robot.joint_names
    head_j = [servo[i] for i in HEAD]
    assert [names[j] for j in head_j] == list(HEAD_NAMES), [names[j] for j in head_j]
    bl_j = [names.index(f"passive_{names[j]}_backlash") if f"passive_{names[j]}_backlash" in names else -1
            for j in head_j]
    tof_site = robot.find_sites("tof")[0][0]
    feet = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
    head_body = robot.find_bodies("jaw_soft")[0][0]
    default = robot.data.default_joint_pos
    contact = env.scene.sensors["feet_ground_contact"]
    col_names, types, _, _ = _terrain_layout(env)
    origins = env.scene.env_origins.cpu().numpy()
    risers = riser_table(env, length, a.width, down=a.down)
    T = round(a.seconds / env.step_dt)
    off = SCAN_CFG["scan_offset"]
    P = SCAN_CFG["scan_nx"] * SCAN_CFG["scan_ny"]
    xs = SCAN_CFG["scan_x0"] + SCAN_CFG["scan_res"] * np.arange(SCAN_CFG["scan_nx"])
    n_under = int((xs <= SCAN_CFG["under_x_max"] + 1e-6).sum())

    override = override_mask = None
    if a.head_override:
        v = [float(x) if x != "p" else float("nan") for x in a.head_override.split(",")]
        assert len(v) == 4, "--head-override: 4 comma-separated values (rad offsets) or p = policy's own"
        override = torch.tensor([0.0 if np.isnan(x) else x for x in v], device=a.device)
        override_mask = torch.tensor([not np.isnan(x) for x in v], device=a.device)
    lp_alpha = lp_prev = None
    if a.head_lowpass_hz:
        hz = [float(x) if x != "p" else float("nan") for x in a.head_lowpass_hz.split(",")]
        assert len(hz) == 4, "--head-lowpass-hz: 4 comma-separated cutoffs (Hz) or p = unfiltered"
        # y += α (x − y), α = 1 − exp(−2π fc dt); α = 1 → unfiltered
        lp_alpha = torch.tensor([1.0 if np.isnan(f) else 1.0 - float(np.exp(-2 * np.pi * f * env.step_dt)) for f in hz],
                                device=a.device)
        print("head low-pass α", lp_alpha.tolist())
    keys =("known_ahead", "act", "q_servo", "q_bl", "head_pr", "head_w", "head_pos", "trunk_pr", "trunk_w", "trunk_pos",
            "found", "foot_z", "mask", "dmax", "dscan", "fell", "trunc")
    rec = {k: [] for k in keys}
    obs, _ = env.reset()
    prev_scan = prev_known = None
    with torch.no_grad():
        for t in range(T):
            o = command_obs(obs["actor"], a.vx)
            if head_cmd is not None:
                o[:, 51:55] = torch.tensor(head_cmd, dtype=o.dtype, device=o.device)
            wide = o.shape[1] >= off + 2 * P
            if wide:
                mask = downhill_ahead_mask(o)
                d, ok = _rows_ahead(o, SCAN_CFG)
                dmax = torch.where(ok, d, torch.full_like(d, -1.0)).amax(1)
                scan, known = o[:, off:off + P], o[:, off + P:off + 2 * P] > 0.5
                if prev_scan is None:
                    dscan = torch.zeros(o.shape[0], device=o.device)
                else:
                    both = (known & prev_known).float()
                    dscan = ((scan - prev_scan).abs() * both).sum(1) / both.sum(1).clamp(min=1)
                prev_scan, prev_known = scan.clone(), known.clone()
            else:
                mask = torch.zeros(o.shape[0], dtype=torch.bool, device=o.device)
                dmax = dscan = torch.zeros(o.shape[0], device=o.device)
                known = None
            act = policy(o[:, :61] if a.obs61 else o)
            if lp_alpha is not None:                  # ablation: the head targets low-passed
                act = act.clone()
                x = act[:, list(HEAD)]
                if lp_prev is None:
                    lp_prev, lp_fresh = x.clone(), torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
                lp_prev = torch.where(lp_fresh[:, None], x, lp_prev + lp_alpha * (x - lp_prev))
                lp_fresh[:] = False
                act[:, list(HEAD)] = lp_prev
            if override is not None:                 # ablation: the head joints held, or some of them
                act = act.clone()
                act[:, list(HEAD)] = torch.where(override_mask, override, act[:, list(HEAD)])
            obs, _, term, trunc, _ = env.step(act)
            if trunc.any() or term.any():           # a reset happened: the scan diff is meaningless
                prev_scan = None
                if lp_prev is not None:             # restart the filter on the reset robots
                    lp_fresh |= term | trunc

            jp = robot.data.joint_pos
            q_servo = jp[:, head_j] - default[:, head_j]
            q_bl = torch.stack([jp[:, b] if b >= 0 else torch.zeros_like(jp[:, 0]) for b in bl_j], -1)
            Rh = _rotm(robot.data.site_quat_w[:, tof_site])
            Rt = _rotm(robot.data.root_link_quat_w)
            hp, hr, _ = _pitch_roll_yaw(Rh)
            tp, tr, ty = _pitch_roll_yaw(Rt)
            wh = _heading_frame(robot.data.body_link_ang_vel_w[:, head_body], ty)
            wt = _heading_frame(robot.data.root_link_ang_vel_w, ty)
            vals = {
                "act": act, "q_servo": q_servo, "q_bl": q_bl,
                "head_pr": torch.stack([hp, hr], -1), "head_w": wh,
                "head_pos": robot.data.site_pos_w[:, tof_site], "trunk_pr": torch.stack([tp, tr], -1),
                "trunk_w": wt, "trunk_pos": robot.data.root_link_pos_w,
                "found": contact.data.found[:, :2] > 0, "foot_z": robot.data.site_pos_w[:, feet, 2],
                "mask": mask, "dmax": dmax, "dscan": dscan, "fell": term, "trunc": trunc,
                "known_ahead": (known.reshape(-1, SCAN_CFG["scan_nx"], SCAN_CFG["scan_ny"])[:, n_under:].float().mean((1, 2))
                                if known is not None else torch.zeros(o.shape[0], device=o.device)),
            }
            for k, v in vals.items():
                rec[k].append(v.detach().cpu().numpy())
    R = {k: np.stack(v) for k, v in rec.items()}
    n_r = max(len(r) for r in risers)
    edges = np.full((a.num_envs, n_r), np.nan)
    tops = np.full((a.num_envs, n_r), np.nan)
    for i, r in enumerate(risers):
        for k, (x, z) in enumerate(r):
            edges[i, k], tops[i, k] = x, z
    col = np.array([col_names[t] for t in types])
    np.savez_compressed(a.out + ".npz", **R, edges=edges, tops=tops, origins=origins, col=col,
                        dt=env.step_dt, meta=json.dumps({**vars(a), "label": a.label}))
    print("wrote", a.out + ".npz", {k: v.shape for k, v in R.items()})


# ── analysis (numpy only) ───────────────────────────────────────────────────

def _hp(x: np.ndarray, n: int) -> np.ndarray:
    """x minus its centered n-sample moving average, trimmed to the 'valid' part (axis 0)."""
    ma = np.convolve(x, np.ones(n) / n, "valid")              # n odd: centered on x[n // 2 + t]
    return x[n // 2: n // 2 + len(ma)] - ma


def _psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Welch (Hann, 50 % overlap, linear detrend per segment); summed over segments, and the count."""
    w = np.hanning(NPERSEG)
    f = np.fft.rfftfreq(NPERSEG, 1 / fs)
    acc, n = np.zeros(len(f)), 0
    t = np.arange(NPERSEG)
    for s in range(0, len(x) - NPERSEG + 1, NPERSEG // 2):
        seg = x[s:s + NPERSEG]
        seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
        p = np.abs(np.fft.rfft(w * seg)) ** 2 / (fs * (w ** 2).sum())
        p[1:-1] *= 2
        acc += p
        n += 1
    return f, acc, n


def _locked_share(sig: np.ndarray, phase: np.ndarray) -> float:
    ok = np.isfinite(phase)
    if ok.sum() < 5 * PHASE_BINS:
        return np.nan
    s, b = sig[ok] - sig[ok].mean(), np.minimum((phase[ok] * PHASE_BINS).astype(int), PHASE_BINS - 1)
    num = sum((b == k).sum() * s[b == k].mean() ** 2 for k in range(PHASE_BINS) if (b == k).any())
    return float(num / max((s ** 2).sum(), 1e-12))


def _since(events: np.ndarray, n: int, cap: int) -> np.ndarray:
    """Steps since the last event (nan before the first one or beyond ``cap``)."""
    out = np.full(n, np.nan)
    ev = np.sort(np.asarray(events, dtype=int))
    for a, b in zip(ev, list(ev[1:]) + [n]):
        k = np.arange(a, min(b, a + cap + 1))
        out[k] = k - a
    return out


def _bin_share(sig: np.ndarray, bins: np.ndarray) -> float:
    """Share of the variance of ``sig`` explained by the mean per integer bin (nan = unbinned)."""
    ok = np.isfinite(bins)
    if ok.sum() < 200:
        return np.nan
    s, b = sig[ok] - sig[ok].mean(), bins[ok].astype(int)
    num = sum((b == k).sum() * s[b == k].mean() ** 2 for k in np.unique(b))
    return float(num / max((s ** 2).sum(), 1e-12))


def _phase_wave(sig: np.ndarray, phase: np.ndarray) -> np.ndarray:
    ok = np.isfinite(phase)
    b = np.minimum((phase[ok] * PHASE_BINS).astype(int), PHASE_BINS - 1)
    s = sig[ok]
    return np.array([s[b == k].mean() if (b == k).any() else np.nan for k in range(PHASE_BINS)])


def _touchdowns(found: np.ndarray, air: int = 3) -> np.ndarray:
    """Contact onsets after ≥ ``air`` steps without contact."""
    on = []
    for t in range(air, len(found)):
        if found[t] and not found[t - air:t].any():
            on.append(t)
    return np.array(on, dtype=int)


def _stride_phase(td: np.ndarray, n: int, lo: int, hi: int) -> np.ndarray:
    ph = np.full(n, np.nan)
    for a, b in zip(td[:-1], td[1:]):
        if lo <= b - a <= hi:
            ph[a:b] = (np.arange(a, b) - a) / (b - a)
    return ph


def _best_lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int = 8) -> tuple[float, int, float]:
    """corr(x(t), y(t+lag)) at the best lag ≥ 0, the lag, and the slope of y on x there."""
    best = (np.nan, 0, np.nan)
    for L in range(max_lag + 1):
        xa, ya = x[: len(x) - L], y[L:]
        if xa.std() < 1e-9 or ya.std() < 1e-9:
            continue
        c = float(np.corrcoef(xa, ya)[0, 1])
        if not np.isfinite(best[0]) or c > best[0]:
            best = (c, L, float(np.cov(xa, ya)[0, 1] / xa.var()))
    return best


def analyze_file(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:          # decompress every array once
        D = {k: z[k] for k in z.files}
    meta = json.loads(str(D["meta"]))
    dt = float(D["dt"])
    fs = 1.0 / dt
    T, N = D["fell"].shape
    nhp = int(round(HP_S / dt)) | 1
    tread = meta["tread"]
    out = {"label": meta["label"], "checkpoint": meta["checkpoint"], "down": meta["down"], "columns": {}}
    for c in sorted(set(D["col"].tolist()), key=lambda s: float(s.split("_")[1][:-2])):
        idx = np.where(D["col"] == c)[0]
        per = []
        psd = {}
        nseg = 0
        waves = {"head_pitch_rate": [], "head_roll_rate": [], "neck_pitch_act": [], "trunk_pitch_rate": []}
        flips_total, flip_da, nonflip_da = 0, [], []
        outcome = {"bottom": 0, "fell": 0, "other": 0}
        td_waves = {}
        for i in idx:
            xe = D["edges"][i]
            xe = xe[np.isfinite(xe)]
            if len(xe):
                stop = D["fell"][:, i] | D["trunc"][:, i] | (np.abs(D["trunk_pos"][:, i, 1] - D["origins"][i, 1]) > meta["width"] / 2 - 0.1)
                t_stop = int(np.argmax(stop)) if stop.any() else T
                reached = np.where(D["trunk_pos"][:, i, 0] > xe[-1] + EDGE_PAD)[0]
                if len(reached) and reached[0] < t_stop:
                    outcome["bottom"] += 1
                elif D["fell"][:, i].any() and int(np.argmax(D["fell"][:, i])) == t_stop:
                    outcome["fell"] += 1
                else:
                    outcome["other"] += 1
            fell = D["fell"][:, i] | D["trunc"][:, i]
            ty = D["trunk_pos"][:, i, 1]
            off = np.abs(ty - D["origins"][i, 1]) > meta["width"] / 2 - 0.1
            end = T
            if fell.any():
                end = min(end, int(np.argmax(fell)) - int(PRE_FALL * fs))
            if off.any():
                end = min(end, int(np.argmax(off)))
            x = D["trunk_pos"][:, i, 0]
            e = D["edges"][i]
            e = e[np.isfinite(e)]
            if len(e):
                after = np.where(x > e[0] - EDGE_PAD)[0]
                if not len(after):
                    continue
                t0 = int(after[0])
                past = np.where(x > e[-1] + EDGE_PAD)[0]
                if len(past):
                    end = min(end, int(past[0]))
            else:
                t0 = int(FLAT_T0 * fs)
            if end - t0 < MIN_WIN * fs:
                continue
            sl = slice(t0, end)
            sig = {
                "head_pitch": np.degrees(D["head_pr"][sl, i, 0]), "head_roll": np.degrees(D["head_pr"][sl, i, 1]),
                "trunk_pitch": np.degrees(D["trunk_pr"][sl, i, 0]), "trunk_roll": np.degrees(D["trunk_pr"][sl, i, 1]),
                "head_pitch_rate": np.degrees(D["head_w"][sl, i, 1]), "head_roll_rate": np.degrees(D["head_w"][sl, i, 0]),
                "head_yaw_rate": np.degrees(D["head_w"][sl, i, 2]),
                "trunk_pitch_rate": np.degrees(D["trunk_w"][sl, i, 1]), "trunk_roll_rate": np.degrees(D["trunk_w"][sl, i, 0]),
                "head_z": 1000 * D["head_pos"][sl, i, 2], "trunk_z": 1000 * D["trunk_pos"][sl, i, 2],
            }
            sig["headrel_pitch"] = sig["head_pitch"] - sig["trunk_pitch"]
            sig["headrel_roll"] = sig["head_roll"] - sig["trunk_roll"]
            for j, nm in enumerate(HEAD_NAMES):
                sig[f"{nm}_act"] = np.degrees(D["act"][sl, i, HEAD[j]])
                sig[f"{nm}_out"] = np.degrees(D["q_servo"][sl, i, j] + D["q_bl"][sl, i, j])
                sig[f"{nm}_servo"] = np.degrees(D["q_servo"][sl, i, j])
                sig[f"{nm}_bl"] = np.degrees(D["q_bl"][sl, i, j])
            hp = {k: _hp(v, nhp) for k, v in sig.items()}
            trim = slice(nhp // 2, nhp // 2 + len(hp["head_pitch"]))       # the hp signals' time base
            r = {"robot": int(i), "seconds": (end - t0) * dt}
            for k, v in hp.items():
                r[f"rms_{k}"] = float(np.sqrt(np.mean(v ** 2)))
            for k in ("head_pitch", "head_roll", "neck_pitch_act", "head_pitch_act", "trunk_pitch", "trunk_z"):
                r[f"mean_{k}"] = float(sig[k].mean())
            r["max_abs_head_pitch_rate"] = float(np.abs(sig["head_pitch_rate"]).max())
            # rates: raw signals are already ~zero-mean; the rms of the raw rate too
            for k in ("head_pitch_rate", "head_roll_rate", "trunk_pitch_rate", "trunk_roll_rate"):
                r[f"rmsraw_{k}"] = float(np.sqrt(np.mean(sig[k] ** 2)))
            # action rate (the reward's Σ Δa², rad²) head vs legs, per step
            A = D["act"][sl, i]
            dA = np.diff(A, axis=0)
            r["arate_head"] = float((dA[:, HEAD] ** 2).sum(1).mean())
            r["arate_legs"] = float((dA[:, LEGS] ** 2).sum(1).mean())
            # gait
            found = D["found"][sl, i]
            tdl, tdr = _touchdowns(found[:, 0]), _touchdowns(found[:, 1])
            r["step_rate"] = (len(tdl) + len(tdr)) / r["seconds"]
            ph = _stride_phase(tdl, end - t0, int(0.2 * fs), int(2.5 * fs))[trim]
            for k in ("head_pitch_rate", "head_roll_rate", "head_pitch", "head_roll", "neck_pitch_act",
                      "head_pitch_act", "trunk_pitch_rate", "headrel_pitch"):
                r[f"stride_locked_{k}"] = _locked_share(hp[k], ph)
            for k in waves:
                if np.isfinite(ph).sum() > 5 * PHASE_BINS:
                    waves[k].append(_phase_wave(hp[k], ph))
            # locked to any touchdown (irregular steps): the variance explained by "time since the
            # last touchdown of either foot", 0-0.5 s; and to the switch mask's flips
            td_any = np.concatenate([tdl, tdr])
            since_td = _since(td_any, end - t0, int(0.5 * fs))[trim]
            m_win = D["mask"][sl, i]
            fl_win = np.where(m_win[1:] != m_win[:-1])[0] + 1
            since_fl = _since(fl_win, end - t0, int(0.5 * fs))[trim]
            for k in ("head_pitch_rate", "head_roll_rate", "head_yaw_rate", "neck_pitch_act", "head_yaw_act",
                      "head_roll_act", "trunk_pitch_rate"):
                r[f"td_locked_{k}"] = _bin_share(hp[k], since_td)
                r[f"flip_locked_{k}"] = _bin_share(hp[k], since_fl)
                if k in ("head_pitch_rate", "head_yaw_act", "head_roll_act"):
                    td_waves.setdefault(k, []).append(
                        [float(hp[k][since_td == j].mean()) if (since_td == j).any() else np.nan
                         for j in range(int(0.5 * fs) + 1)])
            if len(e):
                xs = x[sl][trim]
                rph = np.where((xs >= e[0]) & (xs <= e[-1]), np.mod((xs - e[0]) / tread, 1.0), np.nan)
                for k in ("head_pitch_rate", "head_roll_rate", "neck_pitch_act"):
                    r[f"riser_locked_{k}"] = _locked_share(hp[k], rph)
            # commanded vs passive, per head joint
            for nm in HEAD_NAMES:
                c_, L, g = _best_lag_corr(hp[f"{nm}_act"], hp[f"{nm}_out"])
                r[f"corr_act_out_{nm}"], r[f"lag_{nm}"], r[f"gain_{nm}"] = c_, L, g
                r[f"corr_bl_out_{nm}"] = float(np.corrcoef(hp[f"{nm}_bl"], hp[f"{nm}_out"])[0, 1]) \
                    if hp[f"{nm}_bl"].std() > 1e-9 else np.nan
            r["corr_trunk_headrel_pitch"] = float(np.corrcoef(hp["trunk_pitch"], hp["headrel_pitch"])[0, 1])
            r["corr_trunk_headrel_roll"] = float(np.corrcoef(hp["trunk_roll"], hp["headrel_roll"])[0, 1])
            # ToF / switch
            m = D["mask"][sl, i]
            fl = np.where(m[1:] != m[:-1])[0] + 1
            r["mask_share"] = float(m.mean())
            r["flips_per_s"] = len(fl) / r["seconds"]
            dah = np.sqrt((dA[:, HEAD] ** 2).sum(1))      # |Δa_head| between step t-1 and t → index t-1
            flips_total += len(fl)
            isflip = np.zeros(len(dah), bool)
            isflip[fl - 1] = True
            flip_da += dah[isflip].tolist()
            nonflip_da += dah[~isflip].tolist()
            ds = D["dscan"][sl, i][1:]
            r["corr_dscan_dahead"] = float(np.corrcoef(ds, dah)[0, 1]) if ds.std() > 0 else np.nan
            if "known_ahead" in D:
                r["known_ahead"] = float(D["known_ahead"][sl, i].mean())
            per.append(r)
            for k in ("head_pitch_rate", "head_roll_rate", "head_pitch", "head_roll", "trunk_pitch_rate",
                      "trunk_roll_rate", "neck_pitch_act", "head_pitch_act", "head_roll_act", "neck_pitch_out",
                      "head_pitch_out", "neck_pitch_bl", "head_pitch_bl", "headrel_pitch", "trunk_pitch"):
                f, p, n = _psd(sig[k] - sig[k].mean(), fs)
                psd[k] = psd.get(k, 0) + p
                if k == "head_pitch_rate":
                    nseg += n
        if not per:
            continue
        f = np.fft.rfftfreq(NPERSEG, dt)
        psd = {k: v / max(nseg, 1) for k, v in psd.items()}
        dom = {k: float(f[1:][np.argmax(v[1:] * (f[1:] >= 0.5))]) for k, v in psd.items()}
        band = lambda v, lo, hi: float(np.sqrt(v[(f >= lo) & (f < hi)].sum() * (f[1] - f[0])))
        bands = {k: {"0.5-1.5": band(v, 0.5, 1.5), "1.5-4": band(v, 1.5, 4), "4-8": band(v, 4, 8),
                     "8-25": band(v, 8, 25.01), "13-17": band(v, 13, 17)} for k, v in psd.items()}
        keys = [k for k in per[0] if k != "robot"]
        med = {k: float(np.nanmedian([p.get(k, np.nan) for p in per])) for k in keys}
        q = {k: [float(np.nanpercentile([p.get(k, np.nan) for p in per], 25)),
                 float(np.nanpercentile([p.get(k, np.nan) for p in per], 75))] for k in keys}
        out["columns"][c] = {
            "robots": len(per), "of": int(len(idx)), "outcome": outcome, "median": med, "iqr": q, "dominant_hz": dom, "bands": bands,
            "psd_f": f.tolist(), "psd": {k: v.tolist() for k, v in psd.items()},
            "stride_wave": {k: np.nanmean(np.stack(v), 0).tolist() if v else None for k, v in waves.items()},
            "td_wave": {k: np.nanmean(np.array(v), 0).tolist() for k, v in td_waves.items()},
            "flips": flips_total, "dahead_at_flip": float(np.mean(flip_da)) if flip_da else np.nan,
            "dahead_else": float(np.mean(nonflip_da)) if nonflip_da else np.nan,
        }
    return out


def print_summary(res: list[dict]) -> None:
    rows = [
        ("robots", lambda c: f"{c['robots']}/{c['of']}"),
        ("end: bottom/fell/other", lambda c: f"{c['outcome']['bottom']}/{c['outcome']['fell']}/{c['outcome']['other']}"),
        ("known share ahead", lambda c: f"{c['median'].get('known_ahead', float('nan')):.2f}"),
        ("step rate Hz", lambda c: f"{c['median']['step_rate']:.2f}"),
        ("HEAD pitch rms °", lambda c: f"{c['median']['rms_head_pitch']:.2f}"),
        ("HEAD roll rms °", lambda c: f"{c['median']['rms_head_roll']:.2f}"),
        ("HEAD pitch rate rms °/s", lambda c: f"{c['median']['rmsraw_head_pitch_rate']:.0f}"),
        ("HEAD roll rate rms °/s", lambda c: f"{c['median']['rmsraw_head_roll_rate']:.0f}"),
        ("head yaw rate rms °/s", lambda c: f"{c['median']['rms_head_yaw_rate']:.0f}"),
        ("trunk pitch rms °", lambda c: f"{c['median']['rms_trunk_pitch']:.2f}"),
        ("trunk roll rms °", lambda c: f"{c['median']['rms_trunk_roll']:.2f}"),
        ("trunk pitch rate rms °/s", lambda c: f"{c['median']['rmsraw_trunk_pitch_rate']:.0f}"),
        ("trunk roll rate rms °/s", lambda c: f"{c['median']['rmsraw_trunk_roll_rate']:.0f}"),
        ("head-rel-trunk pitch rms °", lambda c: f"{c['median']['rms_headrel_pitch']:.2f}"),
        ("head-rel-trunk roll rms °", lambda c: f"{c['median']['rms_headrel_roll']:.2f}"),
        ("corr(trunk, rel) pitch", lambda c: f"{c['median']['corr_trunk_headrel_pitch']:+.2f}"),
        ("corr(trunk, rel) roll", lambda c: f"{c['median']['corr_trunk_headrel_roll']:+.2f}"),
        ("dom Hz head pitch rate", lambda c: f"{c['dominant_hz']['head_pitch_rate']:.2f}"),
        ("dom Hz head roll rate", lambda c: f"{c['dominant_hz']['head_roll_rate']:.2f}"),
        ("dom Hz neck_pitch act", lambda c: f"{c['dominant_hz']['neck_pitch_act']:.2f}"),
        ("stride-locked pitch rate", lambda c: f"{c['median']['stride_locked_head_pitch_rate']:.2f}"),
        ("stride-locked roll rate", lambda c: f"{c['median']['stride_locked_head_roll_rate']:.2f}"),
        ("stride-locked neck act", lambda c: f"{c['median']['stride_locked_neck_pitch_act']:.2f}"),
        ("td-locked pitch rate", lambda c: f"{c['median']['td_locked_head_pitch_rate']:.2f}"),
        ("td-locked roll rate", lambda c: f"{c['median']['td_locked_head_roll_rate']:.2f}"),
        ("td-locked yaw rate", lambda c: f"{c['median']['td_locked_head_yaw_rate']:.2f}"),
        ("td-locked trunk pitch rate", lambda c: f"{c['median']['td_locked_trunk_pitch_rate']:.2f}"),
        ("td-locked yaw act", lambda c: f"{c['median']['td_locked_head_yaw_act']:.2f}"),
        ("td-locked roll act", lambda c: f"{c['median']['td_locked_head_roll_act']:.2f}"),
        ("flip-locked pitch rate", lambda c: f"{c['median']['flip_locked_head_pitch_rate']:.2f}"),
        ("flip-locked roll act", lambda c: f"{c['median']['flip_locked_head_roll_act']:.2f}"),
        ("riser-locked pitch rate", lambda c: f"{c['median'].get('riser_locked_head_pitch_rate', float('nan')):.2f}"),
        ("riser-locked neck act", lambda c: f"{c['median'].get('riser_locked_neck_pitch_act', float('nan')):.2f}"),
    ]
    for nm in HEAD_NAMES:
        rows += [
            (f"{nm}: act rms °", lambda c, nm=nm: f"{c['median'][f'rms_{nm}_act']:.2f}"),
            (f"{nm}: out rms °", lambda c, nm=nm: f"{c['median'][f'rms_{nm}_out']:.2f}"),
            (f"{nm}: backlash rms °", lambda c, nm=nm: f"{c['median'][f'rms_{nm}_bl']:.2f}"),
            (f"{nm}: corr act→out", lambda c, nm=nm: f"{c['median'][f'corr_act_out_{nm}']:+.2f}"),
            (f"{nm}: lag steps", lambda c, nm=nm: f"{c['median'][f'lag_{nm}']:.0f}"),
        ]
    rows += [
        ("mean neck_pitch act °", lambda c: f"{c['median']['mean_neck_pitch_act']:+.1f}"),
        ("mean head_pitch act °", lambda c: f"{c['median']['mean_head_pitch_act']:+.1f}"),
        ("mean head pitch (world) °", lambda c: f"{c['median']['mean_head_pitch']:+.1f}"),
        ("action rate head (Σ Δa²)", lambda c: f"{c['median']['arate_head']:.4f}"),
        ("action rate legs (Σ Δa²)", lambda c: f"{c['median']['arate_legs']:.4f}"),
        ("neck act band 8-25 Hz °", lambda c: f"{c['bands']['neck_pitch_act']['8-25']:.2f}"),
        ("neck act band 1.5-4 Hz °", lambda c: f"{c['bands']['neck_pitch_act']['1.5-4']:.2f}"),
        ("neck act band 13-17 Hz °", lambda c: f"{c['bands']['neck_pitch_act']['13-17']:.2f}"),
        ("head pitch rate 1.5-4 °/s", lambda c: f"{c['bands']['head_pitch_rate']['1.5-4']:.0f}"),
        ("head pitch rate 4-8 °/s", lambda c: f"{c['bands']['head_pitch_rate']['4-8']:.0f}"),
        ("head pitch rate 8-25 °/s", lambda c: f"{c['bands']['head_pitch_rate']['8-25']:.0f}"),
        ("switch: down share", lambda c: f"{c['median']['mask_share']:.2f}"),
        ("switch flips /s", lambda c: f"{c['median']['flips_per_s']:.2f}"),
        ("|Δa_head| at flip / else", lambda c: f"{c['dahead_at_flip']:.3f}/{c['dahead_else']:.3f}"),
        ("corr(|Δscan|, |Δa_head|)", lambda c: f"{c['median']['corr_dscan_dahead']:+.2f}"),
    ]
    cols = [(r["label"], cn, c) for r in res for cn, c in r["columns"].items()]
    w = 15
    print(f"{'':30s}" + "".join(f"{lab[:w - 1]:>{w}s}" for lab, _, _ in cols))
    print(f"{'':30s}" + "".join(f"{cn.replace('stairs_', '')[:w - 1]:>{w}s}" for _, cn, _ in cols))
    for name, fn in rows:
        line = f"{name:30s}"
        for _, _, c in cols:
            try:
                line += f"{fn(c):>{w}s}"
            except (KeyError, TypeError, ValueError):
                line += f"{'-':>{w}s}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Mjlab-StairsUp-Backlash-MicroDuck")
    p.add_argument("--checkpoint")
    p.add_argument("--label", default="run")
    p.add_argument("--out", default="head_wobble")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heights", nargs="+", default=["0", "2", "3", "4"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tread", type=float, default=0.20)
    p.add_argument("--approach", type=float, default=1.2)
    p.add_argument("--top", type=float, default=1.2)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--vx", type=float, default=0.25)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--map-x", type=float, default=7.0)
    p.add_argument("--head", choices=("zero", "down"), default="down")
    p.add_argument("--down", action="store_true")
    p.add_argument("--head-override", default=None,
                   help="ablation: 4 comma-separated head action offsets (rad) replacing the policy's, p = keep the policy's")
    p.add_argument("--head-lowpass-hz", default=None,
                   help="ablation: 4 comma-separated first-order low-pass cutoffs (Hz) on the head actions, p = unfiltered "
                        "(applied before --head-override; the policy sees the filtered action as its last action)")
    p.add_argument("--obs61", action="store_true", help="feed the policy the 61-D prefix (a VelStand / walk .pt)")
    p.add_argument("--analyze", nargs="+", default=None, help="npz files to analyse instead of recording")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    if a.analyze:
        res = [analyze_file(f) for f in a.analyze]
        print_summary(res)
        if a.json:
            json.dump(res, open(a.json, "w"))
            print("wrote", a.json)
        return
    record(a)
    res = [analyze_file(a.out + ".npz")]
    print_summary(res)
    json.dump(res, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
