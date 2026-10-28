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
        for name, std in TRACK_YAW_EMA_STDS.items():
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
    })

assert math.isclose(AIR_MIN_KNOTS[0], 0.125) and math.isclose(AIR_MAX_KNOTS[0], 0.30), \
    "slow end must match velstand's window so the slow walk / turning are unchanged"
