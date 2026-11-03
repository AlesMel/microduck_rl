"""Step-up skill with ToF perception — the first PERCEPTIVE microduck policy.

Why (docs/step_up_perception.md): five blind stairs runs put the blind walk's ceiling at ≈ 2 cm
risers; on 2.4-3.0 cm the knee sits at the XL330's 0.64 N·m current cap during every step-up
and pricing torque (runs 4-5) only lowered the swing. Quasi-statically the lift needs 0.24-0.39
N·m, so a DELIBERATE step is within budget — which needs to see the step. This env is the
stairs-mode recipe (run 4's slow walk with the apex reward) plus the head ToF as an input and a
small step-up stack:

  obs      61-D family contract + 64 ToF zones appended (actor: 15 Hz hold, latency, noise,
           dropouts; critic: the clean frame as well) → 125-D actor. See tasks/tof.py.
  terrain  ``step_up`` menu: flat / single_step / short_flight (3 risers), riser 1.5 → 4 cm by
           row, approach 0.8-1.6 m drawn per tile, top platform to the tile end (no step down).
  head     the head_pose command slot says "look down": neck −0.22 / head +0.22 rad (≈ 25°
           down, split over both joints) with a little jitter, tracked at std 0.25 — so the
           metre ahead is in the grid from step 0 and the runtime reproduces it by writing the
           same command. The policy may still move the head (active perception).
  commands twist vx ∈ [0.1, 0.3] forward only (the step is ahead), vy / wz small; head and
           body slots alive as always.
  rewards  inherited gait stack (apex, torque headroom) +
             trunk_height_progress   potential-based Δz, rising pays, hopping nets zero
             foot_on_step            1 per foot, once per episode, contact ≥ 1.2 cm above origin
             trunk_vertical_accel    |a_z| impact cost
  spawns   reset_root_on_terrain, yaw ±20° toward +x, 80 % on the approach BELOW the step
           (low_frac) so most episodes have to climb; 20 % anywhere usable (on top too) as
           the reverse curriculum. Run 6's first launch spawned anywhere: half the robots
           started on top, were promoted by the survival ladder without meeting a step and
           collected the foot-on-step bonus for free — hence low_frac and the bonus latch.
  warm     MICRODUCK_WARM_START_EXPAND_OBS=1 loads run 4's 61-D actor into the 125-D one
           (ToF columns zero): iteration 0 IS the blind stairs policy.

Verify before training (design doc §6): ``tests/test_tof.py`` (the grid sees the floor),
``tests/test_step_up_cfg.py``, then a 64-env smoke run.

RUN 6 (2026-11-02, Mjlab-StepUp-Tof-Backlash-MicroDuck from run 4's model_2999 via Patch 6,
2048 envs, 3000 it, wandb 5fiepb9r; a first launch yuhkuuw7 was stopped at it 138 — spawns
anywhere let half the robots start on top, farm foot_on_step and climb the ladder without a
step, hence low_frac and the bonus latch). Aggregates were FLAT: ladder 5.9 from it 250 on,
foot_on_step 0.021/step and height_progress ≈ 0 throughout, ep_len 364 — nothing in wandb
said the sensor was being learned. The eval said otherwise (scripts/eval/step_up_eval.py,
96 robots, 8 s, vx 0.25, play grid of 4 rows):
                            SIGHTED                 BLIND (ToF := 1.0)
  single_step 1.8 cm        on top 6/10, fell 0     5/10, fell 2
  single_step 2.4 cm        on top 7/10, fell 0     2/10, fell 6
  single_step 3.1 cm        on top 6/15, fell 1     2/15, fell 8
  single_step 3.7 cm        on top 7/13, fell 4     0/13, fell 13
  short_flight 5.4 cm       on top 6/6,  fell 0     4/6,  fell 2
  short_flight 7.3 cm       on top 12/14, fell 1    2/14, fell 12
  short_flight 9.2 cm       on top 2/6,  fell 2     0/6,  fell 6
  short_flight 11.1 cm      on top 4/12, fell 3     0/12, fell 12
  TOTAL                     50/86 on top, 11 fell   15/86 on top, 61 fell
The same network with the grid blanked collapses to the blind walk's ceiling (≈ 2 cm); with
the grid it climbs a 3.7 cm single step in half the attempts and an 11 cm three-riser flight
in a third, falling rarely — the sighted robots that do not make it mostly STOP before the
step (x max 1.26 m at 3.7 cm) rather than fall. The ToF columns were learned even though no
training curve moved: the ladder plateau hid a sighted/blind gap of 35 robots. Caveat: a
1.0 frame is out of distribution (training frames always show floor), so part of the blind
collapse may be OOD rather than lost information; a flat-floor frame ablation would be
the fairer control. Video: scripts/view_terrain_env.py on the step-up grid (run6_step_up_tof).
Next levers: longer training (the curve never saturated by any visible metric), the
flat-floor ablation, and the runtime side (obs_len 125 policy kind + ToF append).

RUNS 7 / 7b (2026-11-03, both from run 6's model_2999, 1024 envs × 6000 it in parallel on one
GPU, eval = step_up_eval.py, 96 robots, 8 s, vx 0.25, same seed; run 6 re-evaluated alongside):
                               run 6 (3000 it)   run 7 (+6000 it)   run 7b FREE HEAD (+6000 it)
  sighted, on top / fell       49/86 / 14        49/86 / 18          64/86 / 14
    single_step 3.1 cm         5/15              10/15               13/15
    single_step 3.7 cm         8/13              4/13 (9 fell)       8/13
    short_flight 11.1 cm       6/12              5/12                6/12
  blind (ToF := 1.0)           5/86 / 71         15/86 / 50          18/86 / 53
  beam-pitch std on step tiles 6.1°              4.6°                8.7°
  - Run 7 (wandb 9r89j05z): twice the samples bought nothing — the recipe is converged.
  - Run 7b (wandb 3fwrducu, head_pose_tracking 2.0 → 0.5, std 0.25 → 0.5): +15 robots on top
    at the same fall count; 3.1 cm single steps 13/15. The head moves more (std 8.7° vs 4.6°)
    but is not "looking at the step" in a deliberate way — the gain is more likely the policy
    being free to hold the sensor where it wants it (it still pitches down) than active scanning.
  - Blind baselines of all three sit at 5-18/86: the sensor is doing the work in every run.
  Keeper: run 7b model_5999 (logs/rsl_rl/step_up/2026-11-03_00-51-55_step_up_tof_freehead).
"""
from copy import deepcopy

from mjlab.envs.mdp.rewards import is_terminated as mjlab_is_terminated
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import tof
from mjlab_microduck.tasks.microduck_walk_v2_terrain_env_cfg import (
    MicroduckWalkV2StairsFinetuneRlCfg,
    make_microduck_walk_v2_terrain_env_cfg,
)

# ── tuned constants ────────────────────────────────────────────────────────
EPISODE_LENGTH_S = 8.0
VX_RANGE = (0.1, 0.3)            # m/s, forward only
VY_RANGE = (-0.05, 0.05)
WZ_RANGE = (-0.2, 0.2)
HEAD_DOWN = (-0.22, 0.22)        # (neck_pitch, head_pitch) deltas, rad: sensor ≈ 25° below level
HEAD_JITTER = 0.05               # rad, so the slot is a range (alive), not a constant
HEAD_TRACK_STD = 0.25            # velocity recipe: 0.5 (too loose to hold a 0.22 rad offset)
# free head (run 7b): the "look down" command stays as a weak prior and the policy may aim the
# sensor itself — run 6 held the head still and only ever saw the floor band 0.2-0.5 m ahead.
FREE_HEAD_TRACK_WEIGHT = 0.5     # recipe: 2.0
FREE_HEAD_TRACK_STD = 0.5        # recipe: 0.5
SPAWN_YAW = 0.35                 # rad, ±20° around +x (into the step)
SPAWN_LOW_FRAC = 0.8             # resets on the approach (below the step); the rest anywhere = reverse curriculum
PROGRESS_WEIGHT = 100.0          # × Δz [m]: a 3 cm step-up pays 3.0 once; a fall pays ≈ −7
PROGRESS_CLIP = 0.01             # m per step
FOOT_ON_STEP_WEIGHT = 5.0        # × 2 feet, once per episode
FOOT_ON_STEP_MIN_RISE = 0.012    # m, 80 % of the smallest riser
IMPACT_WEIGHT = 0.02             # trunk_vertical_accel_penalty is SELF-NEGATING (returns −|a_z|) → POSITIVE weight
                                 # (the smoke test caught −0.02 paying +0.025/step for impacts)
TOF_PARAMS = {"sensor_name": tof.TOF_SENSOR_NAME}

# ── oracle + clean stepping (run 8, docs/step_up_perception.md §10-11) ─────
# Rewards are scaled by dt in mjlab (weight = per-second rate). The step-up stack above was sized
# as if it were not: height_progress paid 100·Δz·dt ≈ 0.07 for a 3.7 cm climb and foot_on_step
# 0.1 per foot, against an episode return of ≈ 52 — runs 6-7b climbed for velocity tracking and
# the ladder, not for the step. The clean variant sizes everything in dt units:
SCAN_PARAMS = {"x_range": (-0.09, 0.45), "y_range": (-0.09, 0.09), "res": 0.03, "nominal": 0.115, "clip": 0.2}
SCAN_NOISE = 0.005               # m, uniform, actor only (the critic sees the exact field)
CLEAN_PROGRESS_WEIGHT = 5000.0   # 5000·dt = 100 per metre of rise → 1.0 per cm, paid back on the way down
CLEAN_FOOT_ON_STEP_WEIGHT = 100.0   # 100·dt = 2 per foot, once
EDGE_WEIGHT = -10.0              # per foot-second on a riser face / edge: the kicking climb (≈ 0.3-0.5 s) costs 3-5
EDGE_TOL = 0.005                 # m, height spread under a footprint that counts as "not one level"
FALL_WEIGHT = -500.0             # −500·dt = −10 once per fall (≈ 20 % of an episode's return)
MAX_EDGE_STEPS = 3               # foot-steps of edge contact per episode a promotion still allows


def make_microduck_step_up_env_cfg(play: bool = False, free_head: bool = False,
                                   perception: str = "tof", clean_feet: bool = False):
    """``perception``: "tof" (the head sensor, 125-D actor) or "oracle" (an exact height scan
    around the feet, 61 + 133-D — perfect terrain memory, the bound on what a ToF-built height
    map could give). ``clean_feet``: price edge contact and falls, size the step-up stack in dt
    units, promote on the ladder only for clean episodes."""
    if perception not in ("tof", "oracle"):
        raise ValueError(f"perception must be 'tof' or 'oracle', not {perception!r}")
    cfg = make_microduck_walk_v2_terrain_env_cfg(
        play=play, menu="step_up", finetune=True, v2=True, apex=True, stairs_mode=True,
    )
    cfg.episode_length_s = EPISODE_LENGTH_S

    if perception == "tof":
        # The ToF in the scene, the frame in both obs groups (actor last → 125-D).
        cfg.scene.sensors = tuple(cfg.scene.sensors) + (tof.make_tof_sensor_cfg(),)
        cfg.observations["actor"].terms["tof"] = ObservationTermCfg(func=tof.tof_depth, params=dict(TOF_PARAMS))
        cfg.observations["critic"].terms["tof_clean"] = ObservationTermCfg(
            func=tof.tof_depth, params={**TOF_PARAMS, "clean": True},
        )
    else:
        # Oracle: the exact terrain around and ahead of the feet (actor last → 61 + 133-D).
        cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
            func=microduck_mdp.terrain_height_scan, params={**SCAN_PARAMS, "noise": SCAN_NOISE},
        )
        cfg.observations["critic"].terms["height_scan_clean"] = ObservationTermCfg(
            func=microduck_mdp.terrain_height_scan, params={**SCAN_PARAMS, "noise": 0.0},
        )

    # Commands: forward into the step; the head slot says "look down".
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = VX_RANGE
    tw.ranges.lin_vel_y = VY_RANGE
    tw.ranges.ang_vel_z = WZ_RANGE
    for attr in ("rel_standing_envs", "zero_command_prob", "rel_turn_in_place_envs"):
        if hasattr(tw, attr):
            setattr(tw, attr, 0.0)
    if perception == "tof":
        # "Look down" so the floor ahead is in the frame; the oracle keeps the gait's own head.
        n, h = HEAD_DOWN
        cfg.commands["head_pose"].ranges = (
            (n - HEAD_JITTER, n + HEAD_JITTER),
            (h - HEAD_JITTER, h + HEAD_JITTER),
            (-0.07, 0.07),
            (-0.015, 0.015),
        )
        cfg.rewards["head_pose_tracking"].params["std"] = HEAD_TRACK_STD
        if free_head:
            cfg.rewards["head_pose_tracking"].weight = FREE_HEAD_TRACK_WEIGHT
            cfg.rewards["head_pose_tracking"].params["std"] = FREE_HEAD_TRACK_STD
        cfg.curriculum.pop("head_pose_range", None)   # the inherited widening would undo the offset

    # Spawn facing the step (reset_root_on_terrain is already wired by v2).
    if not play:
        cfg.events["reset_base"].params["yaw_range"] = (-SPAWN_YAW, SPAWN_YAW)
        cfg.events["reset_base"].params["low_frac"] = SPAWN_LOW_FRAC
    else:
        cfg.events["reset_base"].params["pose_range"]["yaw"] = (-SPAWN_YAW, SPAWN_YAW)

    # The step-up stack.
    contact_sensor = cfg.rewards["low_swing"].params["sensor_name"]
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=microduck_mdp.trunk_height_progress, weight=PROGRESS_WEIGHT, params={"clip": PROGRESS_CLIP},
    )
    cfg.rewards["foot_on_step"] = RewardTermCfg(
        func=microduck_mdp.foot_on_step_bonus, weight=FOOT_ON_STEP_WEIGHT,
        params={"sensor_name": contact_sensor, "min_rise": FOOT_ON_STEP_MIN_RISE,
                # in params so the manager resolves the two foot site ids (a default arg stays unresolved)
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    )
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty, weight=IMPACT_WEIGHT, params={},
    )
    if clean_feet:
        cfg.rewards["height_progress"].weight = CLEAN_PROGRESS_WEIGHT
        cfg.rewards["foot_on_step"].weight = CLEAN_FOOT_ON_STEP_WEIGHT
        cfg.rewards["foot_edge_contact"] = RewardTermCfg(
            func=microduck_mdp.foot_edge_contact_cost, weight=EDGE_WEIGHT,
            params={"sensor_name": contact_sensor, "tol": EDGE_TOL},
        )
        cfg.rewards["fall"] = RewardTermCfg(func=mjlab_is_terminated, weight=FALL_WEIGHT, params={})
        if not play:
            cfg.curriculum["terrain_levels"].params["max_edge_steps"] = MAX_EDGE_STEPS
    return cfg


MicroduckStepUpRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2StairsFinetuneRlCfg)
MicroduckStepUpRlCfg.experiment_name = "step_up"
MicroduckStepUpRlCfg.run_name = "step_up_tof"

MicroduckStepUpFreeHeadRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckStepUpRlCfg)
MicroduckStepUpFreeHeadRlCfg.run_name = "step_up_tof_freehead"

MicroduckStepUpOracleRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckStepUpRlCfg)
MicroduckStepUpOracleRlCfg.run_name = "step_up_oracle_clean"
