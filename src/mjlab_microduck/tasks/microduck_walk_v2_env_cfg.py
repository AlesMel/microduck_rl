"""Microduck Walk v2: the walking gait retrained from scratch, fast and straight.

WHY FROM SCRATCH (2026-10-28). VelStandFast took velstand v8 from 0.18 to ≈ 0.35 m/s at a
0.4 m/s command over nine resumed fine-tunes (microduck_velstand_fast_env_cfg.py on branch
ales/velstand-fast), each trading one defect for another. Three causes no further stage can fix:
  - stage 1 resumed v8 REBUILT FROM ITS ONNX (no wandb access): action std hard-set to 0.2,
    critic and optimizer from a 5-iteration smoke run. PPO, which only searches near the
    current policy, then fine-tuned a low-noise policy against an untrained value function;
  - the speed came from behaviour cloning a frozen stage-2 policy that turns right on its
    own at speed (BC coef 1.0), and every later stage fought that cloned bias;
  - the mirror loss was off while v8's gait formed; added at stage 7 it averaged two
    lopsided gaits into a slower one.
The slow gait's root cause is known and its fix validated by counterfactual reward
accounting: v8's air_time window paid a 2 Hz gait (1.50/step) more than speed tracking
(0.94 at 0.4) — see feet_air_time_speed_window and track_velocity_ema in mdp.py.

PRINCIPLE: change only what the diagnosis points at. This is the proven velocity recipe
(robot, DR, obs noise and delays, 61D actor obs, head command, regularizers, curricula)
plus ONLY the VelStandFast reward fixes that accounting validated, with no behaviour
cloning of any kind and the mirror loss on from iteration 0. No new speculative terms
(no energy term, no gait clock) in run 1.

WHAT CHANGES vs the velocity recipe:
  - air_time → feet_air_time_speed_window (VSF stage 1): window [0.125, 0.30] s at
    |vx| ≤ 0.1 narrowing to [0.09, 0.15] s at ≥ 0.4, so long swings stop paying at speed.
  - tracking (VSF stage 2): stock track_linear_velocity kept for its far-error gradient;
    + track_vx_ema (0.5 s EMA of body vx, std 0.15, w 4); track_angular_velocity scored on
    the 0.5 s EMA of yaw rate, so in-stride surge and yaw wobble are not taxed.
  - pose running tier from 0.3 (VSF stage 2) — the recipe's tier sat at 1.5, unreachable.
  - low_swing (VSF stage 4): one-sided cost for swings below 10 mm, ramped by forward
    speed; fixed the rough-ground trips that came with speed.
  - commands: vx from ±0.3 widened by 0.05 to ±0.5 only once the edge is tracked
    (vx_range_performance_curriculum); vy, wz, turn-in-place, standing as the recipe.
    30 % heading envs so small left and right yaw corrections at speed are in the data.
  - mirror loss (symmetry.py, coef 0.5) from iteration 0.
  - critic only: EMA velocities, battery voltage + BAM friction scale, trunk CoM offset,
    foot friction, heading error. The actor is untouched (61D family contract).
  - walk_v2_gait_stats logs speed ratio and stride frequency per commanded-speed bucket,
    signed yaw bias and L/R air-time asymmetry as Curriculum/gait_stats/*.

DEPLOYMENT PARITY: the robot's defaults (robotd.toml) are action scale 0.9 × 7.4/V and
target low-pass 0.7 legs / 0.5 head; this env, like every env in this repo, trains scale
1.0, unfiltered, with voltage simulated by BAM. Deploy with action_scale = 1.0,
legs_lowpass = 1.0, head_lowpass = 1.0, voltage_adapt = false.

The robot has NO heading-hold loop (stick wz goes straight to the policy): straightness
is judged open loop.

GATES RUN BEFORE TRAINING (2026-10-28, CPU, Rough-Backlash play cfg, 16 robots per row,
scripts/eval/):
  Phase 0, deployment parity (calib_vx.py, fed 0.2/0.3/0.4, training pipeline → robot
  pipeline): v8 0.094/0.136/0.186 → 0.079/0.123/0.153 m/s; VSF 7c model_7625 0.162/0.283/
  0.347 → 0.156/0.269/0.326. The robot's defaults cost 4-18 % of speed, falls unchanged
  (≤ 1/16). Real, but not why v8 is slow — supports deploying with scale 1.0, filters off.
  Phase 2, reward audit (reward_acct.py, train-cfg final weights):
    G1 at a true 0.4, v8 driven faster: achieved 0.179/0.237/0.288/0.349 → 7.68/8.86/9.02/
       9.92 per step, monotonic; 0.24 → 0.35 m/s gains +1.06. PASS.
    G2 v8 at 0.4 (0.179 m/s) 7.68 < stage 3 model_5746 (0.361 m/s, 2.6 Hz) 10.22. PASS —
       v8's original failure (slow gait out-scoring a fast one) is inverted.
    G3 at a true 0.1: v8 (stands, 0.000 m/s) 11.06 < stage 3 (steps, 0.043) 11.89 and 7c
       (0.045) 12.08. PASS. The zero-action probe fell on rough backlash, so no score.
    G4 7c 9.94 vs its mirror image 10.57 (6 %): the mirror walked faster (0.330 vs 0.321 m/s)
       and the gap is in track_vx_ema (+0.45) and air_time (+0.09) — the gaits differ, not
       the reward; every term is a symmetric function of state. Not decisive as a rollout
       check.
    G5 every penalty ≤ 0 in every run. PASS.

PILOT (2026-10-28, urkui-3 RTX 4070 Ti, 2048 envs, 400 iters ≈ 16 min, wandb slgx1nys):
  action std 1.00 → 0.77 (it 50) → 0.81 (it 399), no collapse; air_time 0.02 → 0.72 and
  track_vx_ema 0.11 → 2.36 (it 300; 2.01 at 399 right after a widening); every penalty ≤ 0;
  vx max 0.30 → 0.35 (≈ it 200) → 0.45 (≈ it 300), edge shortfall 0.18 → 0.07-0.09 m/s;
  speed ratio at 0.3 m/s 0.39 → 0.87; yaw bias +0.012 rad/s, L/R air asymmetry 0.5 %;
  mirror loss 0.017 → 0.027 (low; slowly rising, not falling). Episode length 714/1000 at
  399. model_399, flat, 0.3 m/s command: median swing 0.18 s, contact 0.18 s ≈ 2.8 Hz
  (v8: 0.25 s / 0.27 s ≈ 1.9 Hz). The pilot's stride_hz stat read 7-11 Hz because it
  averaged 1/period (contact flicker); fixed to 1/mean(period) after the pilot.

RUN 1 (2026-10-28/29, urkui-3, 2048 envs, 6000 iters ≈ 4.2 h, wandb mg26yora). Final evals
on GPU, play cfg, backlash, 32 robots per row (calib_vx / gait_asym 64 / battery 20 s):
  speed at a 0.4 command, rough / flat:  m4000 0.430/0.393  m4500 0.411/0.390
    m5000 0.400/0.378  m5500 0.430/0.403  m5999 0.418/0.390  (v8 0.186, VSF 7c 0.347)
  |drift| at 0.4, rough / flat:  m4000 0.071/0.055  m4500 0.081/0.077  m5000 0.039/0.050
    m5500 0.061/0.045  m5999 0.093/0.075. Population mean drift swings sign between
    checkpoints (m5000 +0.041, 83 % left; m5500 −0.029; m5999 −0.063, 89 % right): the
    reward does not pin drift — yaw tracking at std √0.5 prices 0.08 rad/s at ≈ 0.02/step.
  PICK = model_5000 (lowest drift on both terrains, least lopsided posture):
    0.1 → 0.081-0.088 m/s (walks); 0.5 → 0.473-0.488; stride 2.7 → 4.0 Hz across commands
    (training stat); falls 0 % flat at every command, rough 3.1 % at 0.4 only; L/R stance
    6.2 %, swing 0.2 %, step 4.2 %, push 7.4 %; open-loop |drift| 0.039 rough / 0.050-0.054
    flat at 0.4 — every success criterion met except drift, which is on the line.
    Robot pipeline (--robot-chain): 0.344 flat / 0.368 rough at 0.4 (−9 / −7 %), rough falls
    3-6 %. Exported: microduck/walk_v2/walk_v2_model_5000.onnx (= actor to 4e-7).

HOW TO TRAIN (urkui-3, 12 GB → 2048 envs):
    uv run train Mjlab-WalkV2-Rough-Backlash-MicroDuck --env.scene.num-envs 2048
Gates before that (plan: reward audit with scripts/eval/reward_acct.py, then a 400-iter
pilot) and success criteria are in docs/superpowers/specs/2026-10-28-walk-v2-from-scratch-design.md.
"""
from copy import deepcopy
import math

from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

NUM_STEPS_PER_ENV = 24
REWARD_GATE_TILT_DEG = 40.0         # fallen gate of the gait/tracking terms (as velstand)

ENABLE_SPEED_AIR_WINDOW = True
AIR_SPEED_KNOTS = (0.1, 0.4)        # |vx command| m/s
AIR_MIN_KNOTS = (0.125, 0.09)       # s, window low edge at the two knots
AIR_MAX_KNOTS = (0.30, 0.15)        # s, window high edge

ENABLE_EMA_TRACKING = True
EMA_TAU_S = 0.5                     # ≈ one stride at 2-2.5 Hz
TRACK_VX_EMA_STD = 0.15
TRACK_VX_EMA_WEIGHT = 4.0
TRACK_YAW_EMA_STD = math.sqrt(0.5)  # the recipe's track_angular_velocity std, on the EMA

ENABLE_POSE_RUNNING_TIER = True
POSE_RUNNING_THRESHOLD = 0.3        # |v_xy| + |wz| command units
POSE_STD_RUNNING = {                # = VelStandFast stage 2
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.10,
    r".*hip_pitch.*": 0.6,
    r".*knee.*": 0.6,
    r".*ankle.*": 0.4,
}

ENABLE_LOW_SWING_COST = True
LOW_SWING_MIN_APEX = 0.010          # m
LOW_SWING_WEIGHT = -60.0
LOW_SWING_VX_RAMP = (0.15, 0.30)    # charged by forward speed only

ENABLE_VX_CURRICULUM = True
VX_START_MAX = 0.3                  # m/s
VX_FINAL_MAX = 0.5                  # leg-model ceiling ≈ 0.47-0.58 m/s across 6.5-8.2 V
VX_STEP = 0.05
VX_EDGE_FRAC = 0.8
VX_MAX_SHORTFALL = 0.08             # m/s mean signed shortfall at the edge
VX_UPDATE_EVERY_ITERS = 50

ENABLE_HEADING_ENVS = True
REL_HEADING_ENVS = 0.3
HEADING_CONTROL_STIFFNESS = 2.0
HEADING_TARGET_REL_RANGE = (-0.5, 0.5)

ENABLE_PRIVILEGED_CRITIC = True
ENABLE_GAIT_STATS = True
GAIT_STATS_WINDOW_ITERS = 50

ENABLE_SYMMETRY = True              # mirror loss, coef SYMMETRY_CFG["mirror_loss_coeff"]


def make_microduck_walk_v2_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cmd_name = cfg.rewards["track_linear_velocity"].params["command_name"]

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
    if ENABLE_EMA_TRACKING:
        cfg.rewards["track_vx_ema"] = RewardTermCfg(
            func=microduck_mdp.track_velocity_ema,
            weight=TRACK_VX_EMA_WEIGHT,
            params={"std": TRACK_VX_EMA_STD, "command_name": cmd_name, "axis": "lin_x",
                    "tau_s": EMA_TAU_S, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
        )
        cfg.rewards["track_angular_velocity"] = RewardTermCfg(
            func=microduck_mdp.track_velocity_ema,
            weight=cfg.rewards["track_angular_velocity"].weight,
            params={"std": TRACK_YAW_EMA_STD, "command_name": cmd_name, "axis": "ang_z",
                    "tau_s": EMA_TAU_S},
        )
    if ENABLE_POSE_RUNNING_TIER:
        cfg.rewards["pose"].params.update(
            {"running_threshold": POSE_RUNNING_THRESHOLD, "std_running": dict(POSE_STD_RUNNING)}
        )
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

    tw = cfg.commands[cmd_name]
    if ENABLE_VX_CURRICULUM:
        # The env starts at the curriculum's start range; the curriculum owns it from there.
        tw.ranges.lin_vel_x = (-VX_START_MAX, VX_START_MAX)
        cfg.curriculum["vx_range"] = CurriculumTermCfg(
            func=microduck_mdp.vx_range_performance_curriculum,
            params={
                "command_name": cmd_name,
                "start_max": VX_START_MAX,
                "final_max": VX_FINAL_MAX,
                "step": VX_STEP,
                "edge_frac": VX_EDGE_FRAC,
                "max_shortfall": VX_MAX_SHORTFALL,
                "update_every_steps": VX_UPDATE_EVERY_ITERS * NUM_STEPS_PER_ENV,
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            },
        )
        if play:
            # Eval at the full range; the curriculum only runs during training.
            cfg.curriculum.pop("vx_range")
            tw.ranges.lin_vel_x = (-VX_FINAL_MAX, VX_FINAL_MAX)
    if ENABLE_HEADING_ENVS:
        assert tw.heading_command and tw.ranges.heading is not None, "velocity recipe keeps heading mode built"
        tw.rel_heading_envs = REL_HEADING_ENVS
        tw.heading_control_stiffness = HEADING_CONTROL_STIFFNESS
        tw.heading_target_rel_range = HEADING_TARGET_REL_RANGE

    if ENABLE_PRIVILEGED_CRITIC:
        critic = cfg.observations["critic"].terms
        critic["velocity_ema"] = ObservationTermCfg(
            func=microduck_mdp.velocity_ema_obs, params={"tau_s": EMA_TAU_S}
        )
        critic["bam_dr"] = ObservationTermCfg(func=microduck_mdp.bam_dr_params_obs)
        critic["trunk_com"] = ObservationTermCfg(
            func=microduck_mdp.body_com_offset_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        )
        critic["foot_friction"] = ObservationTermCfg(
            func=microduck_mdp.geom_friction_obs,
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_foot_collision",))},
        )
        critic["heading_error"] = ObservationTermCfg(
            func=microduck_mdp.heading_error_obs, params={"command_name": cmd_name}
        )

    if ENABLE_GAIT_STATS:
        cfg.curriculum["gait_stats"] = CurriculumTermCfg(
            func=microduck_mdp.walk_v2_gait_stats,
            params={
                "command_name": cmd_name,
                "sensor_name": cfg.rewards["air_time"].params["sensor_name"],
                "window_steps": GAIT_STATS_WINDOW_ITERS * NUM_STEPS_PER_ENV,
                "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            },
        )
    return cfg


MicroduckWalkV2RlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckRlCfg)
MicroduckWalkV2RlCfg.experiment_name = "walk_v2"
MicroduckWalkV2RlCfg.run_name = "walk_v2"
MicroduckWalkV2RlCfg.max_iterations = 6_000
if ENABLE_SYMMETRY:
    MicroduckWalkV2RlCfg.algorithm.symmetry_cfg = dict(SYMMETRY_CFG)
