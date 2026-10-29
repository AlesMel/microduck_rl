"""Microduck VelStandFast: velstand v8 fine-tuned to actually reach its commanded speed.

MEASURED (2026-10-27, v8 = 0045i2si @ 2749 in Mjlab-VelStand-Flat-Backlash, play cfg, no
pushes, 4 envs × 9 s per command): a 0.4 m/s command gives 0.18 m/s (0.5 → 0.23, 0.2 →
0.07, 0.1 → stands still). Stride frequency is locked at ~2 Hz with ~0.25 s swings at every
command; speed only comes from stride length (4.4 cm sweep at 0.4). No crouch, |vy| ≈
0.13 m/s of sway. Peak joint speeds at 0.4 are ≤ 40 % of the XL330 no-load speed, so the
servos are not the limit.

WHY (reward accounting at weight, same rollouts): air_time (w 3, +1/step per foot while its
current air time is inside [0.125, 0.3] s) pays 1.50/step for that slow gait, more than
track_linear_velocity (0.94 at 0.4; its std 0.316 still pays 47 % for 0.18 vs 0.4). A real
0.4 m/s gait at ~3.5 Hz (0.13 s swings) earns ~0.3 from air_time and pays more action
rate: ≈ 0.7/step LESS than the slow walk, so PPO is right to keep it. On top of that the
walk anchor (alpha_walking, coef-1.0 BC on every upright frame) has the same ceiling.

WHAT CHANGES vs velstand (everything else is velstand verbatim, so the policy is a drop-in
replacement with the same 61D obs):
  - air_time → feet_air_time_speed_window: the window slides with |vx command| from
    [0.125, 0.30] s at ≤ 0.1 m/s (unchanged) to [0.09, 0.15] s at ≥ 0.4 m/s.
  - track_vx_fine: vx-only Gaussian (std 0.15) ADDED next to the stock term, like
    track_yaw_fine. 0.15 rather than ~0.1 so the current 0.22 m/s error at 0.4 still
    scores (3.5 % → visible gradient); the stock term keeps the far-error gradient.
  - walk anchor faded out with forward speed (bc_cfg["anchor_vx_fade"]): full weight at
    |vx| ≤ 0.1, zero at ≥ 0.2. Standing, turning in place and the slow walk stay anchored.
Expected per-step totals at a 0.4 command (same accounting, weighted): slow walk ≈ 2.3
(track 0.94 + air 0.70 + vx_fine 0.23 + pose 0.61 − action rate 0.18); a real 0.4 m/s gait
at ~3.5 Hz ≈ 4.0-4.4 if its vx stays within 0.05-0.08 m/s rms of the command (track 1.80 +
air 0.83 + vx_fine 1.5-1.8 + pose 0.55 − action rate 0.55). Before: 2.9 vs 2.1 the other
way round. The fast-gait side is an estimate, not a measurement.

HOW TO TRAIN: plain resume from v8 (no MICRODUCK_WARM_START, so velstand's own curricula
stay at their final stage — fine-tuning, as for v8b). This task adds no curricula.
    uv run train Mjlab-VelStandFast-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 \
      --agent.resume True --wandb-run-path pollen-robotics/mjlab_microduck/0045i2si \
      --wandb-checkpoint-name model_2749.pt --agent.max_iterations 1000
STAGE 2 (2026-10-28): run 1 plateaued at 0.25 m/s for a 0.4 command; see the stage-2 block
below for the accounting and the three changes (EMA tracking of vx and yaw, vx weight 4, the
pose running tier). Launched as a plain resume of run 1's model_3748.
STAGE 3 (2026-10-28): stage 2 fixed the reward but not the gait; a frozen model_4747 fed a
calibrated larger vx command teaches the forward frames (see the stage-3 block). Plain resume
of stage 2's model_4747.
STAGE 4 (2026-10-28): stage 3 trips on rough ground at speed; a one-sided low-swing landing
cost and the vx teacher eased 1.0 → 0.25 (see the stage-4 block). Plain resume of stage 3's
model_5746.
STAGE 5 (2026-10-28): heading drift at speed; fine yaw std 0.35 → 0.15 on the EMA, vx teacher
held at 0.25 (see the stage-5 block). Plain resume of stage 4's model_6745.
STAGE 6 (2026-10-28): stage 5 regressed (more drift, less speed) and is switched off; heading-
hold envs in training and a straight-walking vx teacher instead (see the stage-6 block). Plain
resume of stage 4's model_6745.
STAGE 7 (2026-10-28): stage 6 flipped the drift to the other side instead of removing it; the
gait itself is lopsided. Left-right mirror loss on the policy + mirrored BC samples (see the
stage-7 block). Plain resume of stage 6's model_7000. It lost speed and was stopped.
STAGE 7b (2026-10-28): the same mirror loss + mirrored BC, ramped in over 600 iterations (see
the stage-7b block). Plain resume of stage 6's model_7000. Symmetric gait, get-up eroded.
STAGE 7c (2026-10-28): mirror only upright frames, coefficient 0.4 (see the stage-7c block).
Plain resume of stage 7b's model_7500.
SUCCESS = re-measured with the 2026-10-27 method (scratchpad measure_gait.py): 0.35-0.4 m/s
achieved at a 0.4 command, stride frequency rising with the command, cmd 0.1 actually
walks, no regression in falls / get-up. Watch Episode_Reward/air_time not collapsing to 0
(no stepping) and Train/expert_bc_vx_free_frac ≈ the share of |vx| > 0.2 commands.
"""
import math
from copy import deepcopy

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, UPRIGHT_TILT_DEG
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    MicroduckVelStandRlCfg,
    REWARD_GATE_TILT_DEG,
    make_microduck_velstand_env_cfg,
)

ENABLE_SPEED_AIR_WINDOW = True
AIR_SPEED_KNOTS = (0.1, 0.4)        # |vx command| m/s
AIR_MIN_KNOTS = (0.125, 0.09)       # s, window low edge at the two knots
AIR_MAX_KNOTS = (0.30, 0.15)        # s, window high edge

ENABLE_VX_FINE = True
TRACK_VX_FINE_STD = 0.15
TRACK_VX_FINE_WEIGHT = 2.0

ENABLE_ANCHOR_VX_FADE = True
ANCHOR_VX_FADE = (0.1, 0.2)         # anchor weight 1 at |vx| ≤ 0.1 → 0 at ≥ 0.2

# ── Stage 2 (2026-10-28): make going faster actually pay ──────────────────────
# Run 1 (an5hpzz1 → model_3748, 4 robots/cmd: 0.25 m/s at a 0.4 command) and its 2000-iter
# continuation (iot15ve7, no gain) plateaued. Counterfactual accounting on model_3748 (fed
# 0.4/0.5/0.6, scored at a true 0.4, 16 robots): 0.246 → 0.328 m/s earns only +0.20/step —
# vx_fine +0.67 and track_lin +0.16 are eaten by the yaw terms (−0.27: in-stride yaw-rate
# oscillation 0.7 → 1.0 rad/s, drift only 0.06), action rate (−0.12), pose (−0.12), upright
# (−0.08), head (−0.06). The three changes below re-scored on the same rollouts: +1.25.
ENABLE_EMA_TRACKING = True
EMA_TAU_S = 0.5                     # ≈ one stride at 2-2.5 Hz
TRACK_VX_EMA_WEIGHT = 4.0           # was 2 (instantaneous): alone +0.20 → +0.87
TRACK_YAW_EMA_STDS = {"track_angular_velocity": math.sqrt(0.5), "track_yaw_fine": 0.35}  # same stds/weights as before
ENABLE_POSE_RUNNING_TIER = True
# mjlab's variable_posture has a running tier (std_running above running_threshold, in
# |v_xy| + |wz| command units) but the velocity recipe set it to the walking stds at 1.5,
# which a ±0.4 m/s command never reaches. Also applies to turning in place ≥ 0.3 rad/s.
POSE_RUNNING_THRESHOLD = 0.3
POSE_STD_RUNNING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.10,
    r".*hip_pitch.*": 0.6,
    r".*knee.*": 0.6,
    r".*ankle.*": 0.4,
}

# ── Stage 3 (2026-10-28): teach the faster gait the policy already has ─────────
# Stage 2 (pf256qqb → model_4747) fixed the reward — at a true 0.4 command, fed 0.4/0.5/0.6
# scored 11.45/11.93/12.07 for 0.27/0.34/0.40 m/s, no falls — but did not move the 0.4
# behaviour (4 robots: 0.229 m/s). A frozen copy of model_4747 is the BC teacher for
# forward frames, fed the command that makes IT reach the commanded speed (distill
# remap_vx). Calibrated on model_4747 (claude scratchpad calib_vx.py, flat + backlash, 16
# robots per fed command, 4 s settle, mean body vx): fed 0.10/0.15/0.20/0.25/0.30/0.35/0.40/
# 0.50/0.60/0.70/0.80/0.90 → 0.009/0.053/0.124/0.158/0.198/0.243/0.268/0.357/0.428/0.448/
# 0.483/0.500 m/s (± 0.02-0.06 across robots, all upright up to 0.8, 15/16 at 0.9).
ENABLE_VX_TEACHER = True
VX_TEACHER_COEF = 1.0               # same as the walk anchor it takes over from
VX_TEACHER_WANDB_RUN = "alesmelichar-slovak-university-of-technology-in-bratislava/mjlab_microduck/pf256qqb"
VX_TEACHER_CHECKPOINT = "model_4747.pt"
VX_TEACHER_REMAP = {                # inverse of the curve above at the trained command range
    "targets": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "fed": [0.183, 0.238, 0.302, 0.364, 0.436, 0.492, 0.561],
}

# ── Stage 4 (2026-10-28): stop tripping at speed on rough ground ─────────────
# Stage 3 (iq06yjt5 → model_5746): 0.39 m/s at a 0.4 command, flat falls ≤ 1.6 % in 20 s
# (64 robots; v8 1.6-4.7 %), but rough 12.5-14 % vs v8 3 %, and its own teacher 6-16 %.
# 128 robots at 0.4 on rough: 12 of 20 falls forward; last swings before a fall p10 apex 2 mm
# above the terrain (14 mm for robots that never fell), all at ≥ 8 mm stair/box edges.
ENABLE_LOW_SWING_COST = True
LOW_SWING_MIN_APEX = 0.010          # m; the never-fell robots' p10 apex is 14 mm
# A full scuff (apex 0) costs ≈ a quarter-second of speed tracking at speed: (track_vx_fine
# ≈ 3.4 + track_linear_velocity ≈ 1.3)/step × 12.5 steps ≈ 60. A 2 mm scuff: 0.64 × 60 ≈ 38.
LOW_SWING_WEIGHT = -60.0
# Charged by forward speed only: 0 at |vx| ≤ 0.15 → full at ≥ 0.3. Unscaled, model_5746 on
# rough paid −1.35/step at 0.1 m/s (≈ what standing still would save → the 0.1 dead zone
# again), −0.81 sideways, −0.48 turning at 0.5 rad/s, vs −0.38/−0.41 at 0.3/0.4.
LOW_SWING_VX_RAMP = (0.15, 0.30)
VX_TEACHER_COEF_RAMP = (VX_TEACHER_COEF, 0.25, 400)   # (start, end, iters of this run)

# ── Stage 5 (2026-10-28): drift ────────────────────────────────────────────────
# Straight-walk heading drift, mean |wz| over 32 robots on flat (calib_vx.py): stage-4
# model_6745 0.07/0.11/0.16 rad/s at 0.2/0.3/0.4 vs v8 0.06/0.04/0.07 — visible curving.
# The yaw terms score a 0.5 s EMA since stage 2, so the in-stride wobble no longer counts and
# a tighter std prices only real drift: fine std 0.35 → 0.15 (a steady 0.2 rad/s drift costs
# 1.7/step instead of 0.6). The vx teacher (which itself drifts ≈ 0.2 at fed 0.5-0.6) stays
# at stage 4's 0.25 — a new run would otherwise restart the stage-4 ramp at 1.0.
# RESULT (32 robots per row, heading_hold.py, flat): final model_7744 drifts MORE than stage 4
# (0.21 vs 0.17 rad/s at 0.4) and walks slower (0.32 vs 0.36 m/s). Off; stage 6 resumes stage 4.
ENABLE_DRIFT_FIX = False
TRACK_YAW_FINE_EMA_STD = 0.15
VX_TEACHER_COEF_STAGE5 = 0.25

# ── Stage 6 (2026-10-28): hold a heading ──────────────────────────────────────
# The policy has no absolute heading in its obs, so heading is held OUTSIDE it: the runtime
# feeds wz = k·wrap(target − heading). Measured on stage-4 model_6745 (heading_hold.py, flat,
# 32 robots, 0.4 m/s): a k=2/ki=1 PI loop cuts drift 0.17 → 0.015 rad/s at unchanged speed,
# but leaves 0.32 m mean (2.3 m worst) sideways over 3.6 m, because at speed the walker barely
# obeys LEFT yaw commands: fed +0.1/+0.2/+0.3 it turns −0.07/−0.02/0.00 rad/s, fed −0.1/−0.3 it
# turns −0.16/−0.36 (v8 is symmetric: ±0.3 → −0.24/+0.35, and holds 9 cm over 1.9 m). Source:
# the vx teacher (stage-2 model_4747, BC on forward frames) turns right at speed by itself —
# zero yaw command → +0.02 (fed 0.18), −0.07 (0.36), −0.12 (0.44), −0.20 (0.56) rad/s.
# Two changes:
#   - heading envs: mjlab's heading mode (wz command = stiffness · heading error, recomputed
#     every step inside the env, obs contract unchanged) on a share of the envs, targets drawn
#     near the current heading so the experience is small corrections both ways at speed.
#   - the vx teacher fed a yaw offset that makes IT walk straight (calibrated on it, same
#     32-robot runs: forward residual |mean wz| ≤ 0.045 rad/s with these offsets).
# The stage-5 yaw std (0.15) is dropped with stage 5; yaw tracking is stage 4's.
ENABLE_HEADING_ENVS = True
REL_HEADING_ENVS = 0.5
HEADING_CONTROL_STIFFNESS = 2.0     # the runtime loop's k; ±1 rad/s clip from ranges.ang_vel_z
HEADING_TARGET_REL_RANGE = (-0.5, 0.5)
VX_TEACHER_COEF_STAGE6 = 0.25       # stage 4's end value, held (no re-ramp from 1.0)
VX_TEACHER_WZ_OFFSET = {            # rad/s added to the teacher's yaw command, by fed |vx|
    "fed": [0.18, 0.24, 0.30, 0.36, 0.44, 0.49, 0.56],
    "fwd": [-0.02, -0.02, 0.05, 0.08, 0.14, 0.25, 0.24],
    "back": [0.10, 0.20, 0.27, 0.28, 0.28, 0.33, 0.37],   # 1.4× the first guess, which left −0.05..−0.075
}

# RESULT (16 robots, 0.4 m/s): model_6750 (5 iters) drift −0.18 → +0.02, +0.2 command now obeyed;
# model_7000 drift +0.17 (LEFT, 92 % of robots) and −0.2 commands ignored — the bias changed
# side instead of going away. With the k=2/ki=1 heading loop model_7000 is the best so far:
# drift 0.014 rad/s, 0.36 m/s, worst heading error 0.30 rad, 0.39 m mean / 0.74 m worst
# sideways over ~3.6 m. Stopped at 7089.
# 32 robots (heading_hold.py, GPU), model_7000 + the k=2/ki=1 loop: cmd 0.2/0.3/0.4 → 0.18/0.27/
# 0.35 m/s, drift 0.004/0.015/0.014 rad/s, worst heading error 0.05/0.17/0.24 rad, sideways 0.18/
# 0.33/0.32 m mean over 1.8/2.6/3.5 m (worst 0.30/2.06/0.67). battery.py (64 robots) vs stage-4
# model_6745: get-up 97-100 % in every family for both; walk falls flat 0.0-6.2 % (stage 4
# 1.6-3.1 %), rough 0.0-7.8 % (stage 4 0.0-7.8 %, rough 0.4: 7.8 % vs 0 %).

# ── Stage 7 (2026-10-28): a symmetric gait ─────────────────────────────────────
# WHY (gait_asym.py, 64 robots at 0.4 m/s, open loop, R mirrored): all the drift builds up in
# single support (corr of single-support yaw with drift 0.99-1.00: the body yaws toward the
# stance side every step and drift is what does not cancel). The fast gait yaws twice as much
# per step as v8 (stance foot twists 8.8° vs 4.4°), so a small imbalance is a big drift. Not
# contact timing or push-off: L/R stance times within 2-17 ms, push impulse equal, ankle
# push-off peaks within 3° and uncorrelated with drift. The posture is lopsided instead: the
# teacher, stage 4 and stage 6 all carry the right hip ~8° more flexed (v8 −5° the other way),
# and per-robot randomization (trunk CoM, hip-yaw encoder bias) correlates |r| ≤ 0.3 with
# drift. Nothing in training prefers a symmetric gait, so the bias drifts from side to side.
#   - mirror loss on the policy (symmetry.py, coeff 0.5): π(mirror(o)) = mirror(π(o)).
#   - BC samples mirrored too (bc_mirror), so the teacher labels are symmetric.
# Heading envs and the teacher's yaw offset stay (stage 6).
# RESULT: stopped at 7508. The mirror loss fell 0.067 → 0.017 within ~20 iterations and speed
# went with it: track_vx_fine 0.89 → ~0.3 for 500 iterations, model_7250 (16 robots, 0.4 m/s)
# 0.33 m/s open loop, drift back to −0.07, 1-2 falls in 16, 0.30 m/s under the heading loop.
# The mirror tables are right (stage 4 and v8 run through them walk as fast, no extra falls,
# drift and yaw responses flip sign: stage 4 −0.166 → +0.129, v8 +0.080 → −0.050, 32 robots):
# stage 4 and its mirror are two different lopsided gaits and the loss collapsed the policy
# onto their average, which is not a fast gait. A retry would need the coefficient ramped
# from ~0 (AGENTS.md: introduce a constraint after the skill, not over it).
ENABLE_SYMMETRY = True

# ── Stage 7b (2026-10-28): the same constraint, introduced slowly ────────────────
# Both the mirror-loss coefficient (0 → 0.5) and the mirrored BC samples' weight (0 → 1) ramp
# linearly over the first MIRROR_RAMP_ITERS iterations of the run, then hold; the stage-7 run
# went from 0 to full in one step. 600 of 1000 iterations: stage 7 collapsed in ~20, so this
# is ~30× slower. Plain resume of stage 6's model_7000 again.
MIRROR_RAMP_ITERS = 600
# RESULT (run 2026-10-28_16-24-32, 32 robots per row unless noted). Speed term (log, 50-iter
# mean) 1.69 (7090) → 1.73 (7250) → 1.23 (7500) → 0.49 (7750) → 0.44 (7999); stage 6: 1.71.
#   model_7500 (coeff ≈ 0.42): open-loop |drift| at 0.4 0.123 (stage 6 0.182, v8 0.107), mean
#     +0.021 with 42 % of 64 robots turning right (stage 4: 91 %); hip-pitch L−R +1.1° (stage 4
#     +8.5°, stage 6 +7.2°); single-support yaw +10.4 / −9.2 °/s (balanced). Heading loop: 0.35 /
#     0.27 / 0.17 m/s, sideways 0.20 / 0.15 / 0.11 m mean (worst 1.29 / 1.25 / 0.90). Falls flat
#     0-6.2 %, rough 0-6.2 %. Get-up REGRESSED: home side 88 % (12.5 % stuck, 3.7 s vs 1.1 s),
#     home face-up 1.37 s vs 0.94 s; others 100 %.
#   model_7750 (coeff 0.5): most symmetric (±0.2 command → +0.18 / −0.15), sideways 0.07 m at
#     0.2/0.3 with the loop, but get-up BROKEN (face-up 0 %, side 0 %, so3 30 %, sit 48 %).
#   model_7999: asymmetric again (+0.10 / −0.32), falls 2-3 of 32.
# Get-up moves are one-sided (roll to a side); mirroring averages the two directions. A next
# step would mirror only upright frames (loss and BC), coefficient held below 0.5.

# ── Stage 7c (2026-10-28): mirror only while upright ──────────────────────────
# Same mirror loss and mirrored BC, but only on frames tilted < 25° (the walk-anchor gate):
# the mirror loss uses symmetry.microduck_vel_symmetry_upright (tilted rows unmirrored → zero
# loss), mirrored BC skips tilted samples (the stand expert's get-up labels stay one-sided).
# Coefficient held at 0.4 (≈ 7b's value at model_7500; speed fell once it sat at 0.5), no ramp.
# Plain resume of stage 7b's model_7500.
ENABLE_MIRROR_UPRIGHT_ONLY = True
MIRROR_UPRIGHT_COEFF = 0.4
# RESULT (run 2026-10-28_18-15-10_velstand_fast_1792_stage7c, 7500 → 7999). Speed term 1.71-1.74
# throughout (stage 6 1.71), mirror loss 0.011. Get-up (battery.py, 64 robots) 97-100 % in every
# family at every checkpoint, side-lying ~1.0 s again (7b model_7500: 88 %, 3.7 s).
# PICK = model_7625 (64 robots at 0.4 unless noted): open loop 0.33 m/s, drift mean +0.017,
# |drift| 0.074 (v8 0.107, stage 4 0.155), 45 % turn right; single-support yaw +8.9 / −8.0 °/s;
# hip-pitch L−R −1.7°; ±0.2 command → +0.15 / −0.11 (32 robots). Falls flat 0-4.7 %, rough
# 1.6-6.2 %. Runtime heading loop wz = clamp(2·e + 1·∫e, ±0.8): 0.32 m/s, drift 0.012 rad/s,
# worst heading error 0.26 rad, sideways 0.14 m mean / 0.58 m worst over ~3.3 m (0.3 m/s: 0.12 /
# 0.47 m). A ±0.5 clamp left one robot in 32-64 losing heading (worst 2.6 rad, 3.2 m sideways).
# Later checkpoints swing the population drift again (7750/7875 −0.21, 7999 +0.17 at 0.4).


def make_microduck_velstand_fast_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_velstand_env_cfg(play=play, rough=rough)

    if ENABLE_SPEED_AIR_WINDOW:
        at = cfg.rewards["air_time"]
        cfg.rewards["air_time"] = RewardTermCfg(
            func=microduck_mdp.feet_air_time_speed_window,
            weight=at.weight,
            params={
                "sensor_name": at.params["sensor_name"],
                "command_name": at.params["command_name"],
                "speed_knots": AIR_SPEED_KNOTS,
                "min_knots": AIR_MIN_KNOTS,
                "max_knots": AIR_MAX_KNOTS,
                "command_threshold": at.params["command_threshold"],
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            },
        )
    if ENABLE_VX_FINE:
        cfg.rewards["track_vx_fine"] = RewardTermCfg(
            func=microduck_mdp.track_lin_vel_x_fine,
            weight=TRACK_VX_FINE_WEIGHT,
            params={
                "std": TRACK_VX_FINE_STD,
                "command_name": cfg.rewards["track_linear_velocity"].params["command_name"],
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            },
        )
    if ENABLE_EMA_TRACKING:
        # Same term names as before so wandb curves continue; the functions change.
        cmd_name = cfg.rewards["track_linear_velocity"].params["command_name"]
        cfg.rewards["track_vx_fine"] = RewardTermCfg(
            func=microduck_mdp.track_velocity_ema,
            weight=TRACK_VX_EMA_WEIGHT,
            params={"std": TRACK_VX_FINE_STD, "command_name": cmd_name, "axis": "lin_x",
                    "tau_s": EMA_TAU_S, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
        )
        yaw_stds = dict(TRACK_YAW_EMA_STDS)
        if ENABLE_DRIFT_FIX:
            yaw_stds["track_yaw_fine"] = TRACK_YAW_FINE_EMA_STD
        for name, std in yaw_stds.items():
            # Ungated, like the instantaneous terms they replace (recovery economics unchanged).
            cfg.rewards[name] = RewardTermCfg(
                func=microduck_mdp.track_velocity_ema,
                weight=cfg.rewards[name].weight,
                params={"std": std, "command_name": cmd_name, "axis": "ang_z", "tau_s": EMA_TAU_S},
            )
    if ENABLE_POSE_RUNNING_TIER:
        cfg.rewards["pose"].params.update(
            {"running_threshold": POSE_RUNNING_THRESHOLD, "std_running": dict(POSE_STD_RUNNING)}
        )
    if ENABLE_HEADING_ENVS:
        tw = cfg.commands["twist"]
        assert tw.heading_command and tw.ranges.heading is not None, "velocity recipe keeps heading mode built"
        tw.rel_heading_envs = REL_HEADING_ENVS
        tw.heading_control_stiffness = HEADING_CONTROL_STIFFNESS
        tw.heading_target_rel_range = HEADING_TARGET_REL_RANGE
    if ENABLE_LOW_SWING_COST:
        sw = cfg.rewards["foot_swing_height"].params
        cfg.rewards["low_swing"] = RewardTermCfg(
            func=microduck_mdp.feet_low_swing_cost,
            weight=LOW_SWING_WEIGHT,
            params={
                "sensor_name": sw["sensor_name"],
                "height_sensor_name": sw["height_sensor_name"],
                "command_name": sw["command_name"],
                "min_apex": LOW_SWING_MIN_APEX,
                "command_threshold": sw["command_threshold"],
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
                "vx_ramp": LOW_SWING_VX_RAMP,
            },
        )
    return cfg


MicroduckVelStandFastRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckVelStandRlCfg)
MicroduckVelStandFastRlCfg.experiment_name = "velstand_fast"
MicroduckVelStandFastRlCfg.run_name = "velstand_fast"
MicroduckVelStandFastRlCfg.max_iterations = 1_000
if ENABLE_ANCHOR_VX_FADE and MicroduckVelStandFastRlCfg.algorithm.bc_cfg is not None:
    MicroduckVelStandFastRlCfg.algorithm.bc_cfg["anchor_vx_fade"] = ANCHOR_VX_FADE
if ENABLE_VX_TEACHER and MicroduckVelStandFastRlCfg.algorithm.bc_cfg is not None:
    assert ENABLE_ANCHOR_VX_FADE, "the vx teacher takes the frame share the anchor fades out of"
    MicroduckVelStandFastRlCfg.algorithm.bc_cfg.update({
        "vx_teacher_coef": VX_TEACHER_COEF,
        "vx_teacher_wandb_run_path": VX_TEACHER_WANDB_RUN,
        "vx_teacher_checkpoint_name": VX_TEACHER_CHECKPOINT,
        "vx_teacher_remap": VX_TEACHER_REMAP,
        "vx_teacher_coef_ramp": VX_TEACHER_COEF_RAMP if ENABLE_LOW_SWING_COST else None,
    })
    if ENABLE_DRIFT_FIX:
        MicroduckVelStandFastRlCfg.algorithm.bc_cfg.update(
            {"vx_teacher_coef": VX_TEACHER_COEF_STAGE5, "vx_teacher_coef_ramp": None})
    if ENABLE_HEADING_ENVS:
        MicroduckVelStandFastRlCfg.algorithm.bc_cfg.update({
            "vx_teacher_coef": VX_TEACHER_COEF_STAGE6, "vx_teacher_coef_ramp": None,
            "vx_teacher_remap": {**VX_TEACHER_REMAP, "wz_offset": VX_TEACHER_WZ_OFFSET},
        })
if ENABLE_SYMMETRY:
    MicroduckVelStandFastRlCfg.algorithm.symmetry_cfg = {**SYMMETRY_CFG, "mirror_ramp_iters": MIRROR_RAMP_ITERS}
    if MicroduckVelStandFastRlCfg.algorithm.bc_cfg is not None:
        MicroduckVelStandFastRlCfg.algorithm.bc_cfg["bc_mirror"] = True
    if ENABLE_MIRROR_UPRIGHT_ONLY:
        MicroduckVelStandFastRlCfg.algorithm.symmetry_cfg = {
            **SYMMETRY_CFG, "mirror_loss_coeff": MIRROR_UPRIGHT_COEFF,
            "data_augmentation_func": "mjlab_microduck.tasks.symmetry.microduck_vel_symmetry_upright"}
        if MicroduckVelStandFastRlCfg.algorithm.bc_cfg is not None:
            MicroduckVelStandFastRlCfg.algorithm.bc_cfg["bc_mirror_upright_tilt_deg"] = UPRIGHT_TILT_DEG

assert math.isclose(AIR_MIN_KNOTS[0], 0.125) and math.isclose(AIR_MAX_KNOTS[0], 0.30), \
    "slow end must match velstand's window so the slow walk / turning are unchanged"
