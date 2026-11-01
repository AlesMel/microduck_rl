"""Walk v2 on an IsaacLab-style terrain curriculum (2026-10-29): the walking recipe, harder ground.

The env is ``make_microduck_walk_v2_env_cfg(rough=True)`` — the full walk-v2 reward stack, DR,
obs noise and delays, the 61D actor contract, the rough-terrain solver settings (nconmax 200,
30 solver iterations, softened terrain contacts) and mjlab's ``terrain_levels`` promotion
curriculum — with the base rough mix replaced by one of the duck-scaled menus in
``microduck_terrains.py``:

  * ``menu="isaaclab"`` (Mjlab-WalkV2-Terrain-MicroDuck): flat, rubble, random grid, pyramid
    stairs up and down, straight small steps, pyramid slopes — one column each, NUM_ROWS
    difficulty rows, robots start on row 0 and climb when they cover half a tile.
  * ``menu="stairs"`` (Mjlab-WalkV2-Stairs-MicroDuck): straight small steps + pyramid stairs.

What changes vs walk v2 besides the ground (each behind an ``ENABLE_*`` toggle):
  - spawn offset ±0.2 m instead of the recipe's ±0.5 m: the straight stairs' start platform is
    1 m long, and the pyramids' top platform is 0.6 m — a ±0.5 m spawn would start robots on a
    step edge (or, on the inverted pyramid, over the pit).
  - foot swing targets 2 cm → 3.5 cm (foot_clearance / foot_swing_height): a 3 cm step needs a
    higher swing; the 2026-08 hostile-terrain runs validated 3.5 cm. The hip-roll pose
    tolerance is loosened 0.05 → 0.12 rad in the walking/running tiers for the same reason
    (no ankle roll: hip roll is the only sideways adaptation to uneven ground).

``finetune=True`` (the ``-Finetune-`` task ids) is the warm-start recipe of AGENTS.md: the walk-v2
checkpoint is loaded with ``MICRODUCK_WARM_START=1`` (step counter restarts at 0 so the terrain
ladder starts on row 0) and every step-keyed curriculum the walk inherited (action-rate tax,
standing fraction, head/body ranges, CoM ramps, head-pose-bias weight) is pinned at its final
stage — the loaded policy was trained under those conditions, letting them ramp again would hand
it back an untaxed, easier task. The vx performance curriculum is replaced by its final range
(±0.5 m/s); the only curriculum left to climb is the terrain ladder.

MEASURED BEFORE TRAINING (2026-10-29, urkui-3, walk_v2 model_5000 ONNX, stairs play cfg, 16
robots, vx 0.3 along the stairs, scripts/view_terrain_env.py --video): the walk crosses 0.5-1.1 cm
steps cleanly (2/2 robots to the end platform, no falls); 1.1-1.7 cm: 1/2 reaches the landing
after 4 falls; 1.7-3.0 cm: every robot trips on the first step (6-7 falls in 14 s). The policy
generalises from the 1.5 cm rough mix to ≈ 1 cm steps and no further.

HOW TO TRAIN (urkui-3, 12 GB → 2048 envs; model_5000 of walk-v2 run mg26yora)::

    MICRODUCK_WARM_START=1 uv run train Mjlab-WalkV2-Stairs-Finetune-Backlash-MicroDuck \
        --env.scene.num-envs 2048 --agent.resume True \
        --wandb-run-path <entity>/mjlab_microduck/mg26yora --wandb-checkpoint-name model_5000.pt

RUN 1 (2026-10-29, urkui-3, Mjlab-WalkV2-Stairs-Finetune-Backlash-MicroDuck, 2048 envs, 3000 it,
wandb f3u8hzan, 80 min): reward 1 → 113 by it 250 then flat (118 at the end); gait unchanged
(speed ratio 0.91 at 0.3 m/s, 3.1 Hz, no yaw bias); every penalty ≤ 0. Terrain ladder mean 0.85 →
1.5 of 9 rows and no further — even the FLAT column sat at 1.8, so the stock promotion rule
(end an episode > 2 m from the spawn) is what capped it, not the policy; out_of_terrain_bounds
ended as many episodes as falls (0.375 each). model_2999 on the stairs (16 robots, 0.3 m/s):
0.5-1.7 cm steps crossed without falls (1.1-1.7 cm was 4-5 falls before), 1.7-2.4 cm: stops
at the first step upright, 2.4-3.0 cm: falls or turns away. The gain is exactly the ladder's
reach (row 2 ≈ 1.25 cm).

``v2=True`` (the ``-StairsV2-`` task ids), the follow-up: (1) the ladder promotes on surviving
the episode while tracking the command and covering ≥ 30 % of the commanded distance, demotes
on a fall or on no progress (``terrain_levels_survival``, the hostile-terrain rule);
(2) robots spawn ANYWHERE usable on their tile, on the treads too, facing any direction
(``reset_root_on_terrain``, 4 cm ground lookup, flat window = foot = one tread), so the higher
rows get data from iteration 0 and the policy cannot memorise where the steps are; (3) a 6 m
flat apron and a bounds margin that lets robots roam 5 m into it, so walking off the grid stops
truncating episodes. Terrain seed fixed (V2_TERRAIN_SEED) for the lookup.

RUN 2 (2026-10-29/30, urkui-3, Mjlab-WalkV2-StairsV2-Finetune-Backlash-MicroDuck from run 1's
model_2999, 2048 envs, 3000 it, wandb 4y1g919r): reward 105-120, gait unchanged, every penalty
≤ 0, out_of_terrain_bounds 0.0 throughout. Survival ladder: promoted 0.35-0.5 / demoted
0.45-0.6 of resets, mean level 0 → 2.1 by it 1250 then an EQUILIBRIUM at ≈ 2 (robots reach
row 3+, fall, come back). model_2999 on the stairs: 0.5-2.4 cm steps crossed (1.7-2.4 cm: up,
over the landing and down with one fall — new), 2.4-3.0 cm: onto the first flight then falls.
RUN 3 (2026-10-30, wandb s8noq7vp) = run 2 + V2_MAX_INIT_TERRAIN_LEVEL 0 → 5, from run 2's
model_2999. Ladder started at 2.5, the robots dealt to rows 3-5 fell and were demoted to 1.9 by
it 100, equilibrium back at 2.2-2.3 by the end: spreading the start rows did NOT raise the
ceiling. model_2999 on the stairs (32 robots): 0.5-1.7 cm crossed without falls, 1.7-2.4 cm
5/5 fell on the first flight, 2.4-3.0 cm 2/2 fell.
WHY, in one number: Metrics/peak_height_mean read 0.018-0.019 m in ALL THREE runs — the swing
apex never moved, and a blind walker lifting 1.9 cm trips on a 2.4 cm riser whatever the
ladder does. foot_swing_height's weighted contribution is 0.007 per episode against a stack of
≈ 110: the 3.5 cm target is on paper only. Next lever is the gait (a per-swing apex term that
competes with the gait stack, trained as a separate rough-ground walk), not the curriculum and
not perception — see the actuator-torque feasibility check first.

FEASIBILITY (2026-11-01, scripts/eval/stairs_feasibility.py, run 3 model_2999, 64 robots, torque
measured against BAM's XL330 firmware current limit 1.75 A × kt = 0.64 N·m, which binds at every
battery voltage in the DR range):
  - the swing apex SCALES WITH SPEED on flat: p10/median 0.6/1.3 cm at 0.1 m/s, 1.8/2.0 at 0.2,
    2.2/2.6 at 0.3, 2.9/3.3 at 0.4, 3.5/4.0 at 0.5 (the logged peak_height_mean of 1.9 cm is the
    all-command average, not the walking apex);
  - on the stairs at 0.4-0.5 m/s the apex is 3.3-3.9 cm median, above every riser, and rows 2-3
    STILL fall (10/11, 6/6): the last swing before a fall cleared 3.3-3.9 cm. Clearance is not
    the limiter above 0.3 m/s;
  - the knee and hip are at the current cap: flat p99 0.86 (knee) at 0.3 m/s and 1.00 at 0.5;
    on the steps of rows 2-3 p99 = 1.00 with 8-15 % of steps saturated; in the 0.3 s before a
    fall the knee sits at the cap 20-30 % of the time. Falls are on the UP flight (toe-catch
    at the first riser at 0.3 m/s aside). The ceiling is torque during the step-up.
  - at 0.1-0.2 m/s the gait leaves 27-41 % knee reserve (p99 0.59-0.73) — enough for the
    0.24-0.39 N·m a quasi-static one-knee lift of the 0.8 kg body needs — but swings only
    1.3-2.0 cm. A stairs MODE therefore needs all three: slow (≤ 0.3 m/s), a high apex
    (feet_swing_apex_reward, the ``apex`` option) and torque-reserve pressure, as a separate
    hot-swapped policy; the fast flat walk cannot be both.

Preview the terrain with the robot standing on it, and check that every spawn is a stable
stance, with::

    uv run mjpython scripts/view_terrain_env.py --task Mjlab-WalkV2-Terrain-MicroDuck --settle
"""
from copy import deepcopy

import math

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

from mjlab_microduck.tasks.microduck_terrains import (
    BORDER_WIDTH,
    NUM_ROWS,
    PATCH_SIZE,
    make_microduck_stairs_terrains_cfg,
    make_microduck_terrains_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final
from mjlab_microduck.tasks.microduck_walk_v2_env_cfg import (
    VX_FINAL_MAX,
    MicroduckWalkV2RlCfg,
    make_microduck_walk_v2_env_cfg,
)

TERRAIN_MENUS = {
    "isaaclab": make_microduck_terrains_cfg,
    "stairs": make_microduck_stairs_terrains_cfg,
}
PLAY_ROWS = 4                   # play / preview: every column at 4 difficulties
MAX_INIT_TERRAIN_LEVEL = 0      # training starts every robot on the easiest row
FINETUNE_MAX_ITERATIONS = 3_000 # warm start from a walking policy: the gait exists, the ladder is new

# v2: survival ladder + spawn anywhere + wide apron (see the module docstring, RUN 1).
V2_TERRAIN_SEED = 20261029
V2_BORDER_WIDTH = 6.0           # m, flat apron (was 2.0)
V2_BOUNDS_MARGIN = -5.0         # m, negative = robots may roam this far INTO the apron before truncation
V2_MIN_TRACKING = 0.55          # fraction of the max track_linear_velocity reward (std² = 0.1) to be promoted
V2_MIN_PROGRESS = 0.3           # fraction of |v_cmd| × T that must be covered (anti-circling)
V2_MAX_INIT_TERRAIN_LEVEL = 5   # run 3: start rows spread over 0-5 (steps up to ≈ 2 cm) — run 2's ladder
                                # settled at row 2 with everyone starting on row 0; the tall rows only ever
                                # saw freshly promoted robots that fell straight back (see RUN 2 below)
V2_SPAWN = {"span": PATCH_SIZE / 2 - 0.3, "res": 0.04, "flat_tol": 0.02,
            "yaw_range": (-math.pi, math.pi), "z_clearance": (0.125, 0.135), "candidates": 8}

# apex: pay for swing height (runs 1-3 never lifted above 1.9 cm, see the module docstring).
APEX_TARGET = 0.035             # m, = FOOT_TARGET_HEIGHT: clears a 3 cm riser with margin
APEX_WEIGHT = 5.0               # ≈ 6 landings/s × ≤ 1 each → up to ≈ 0.6/step, the air_time term's mass
APEX_MIN_AIR_TIME = 0.06        # s, a landing counts only after this long in the air

ENABLE_TIGHT_SPAWN = True
SPAWN_XY = 0.2                  # m, ± offset from the tile's spawn origin

ENABLE_STAIR_FOOT_TARGET = True
FOOT_TARGET_HEIGHT = 0.035      # m (recipe: 0.02)
HIP_ROLL_STD = 0.12             # rad, walking/running tiers (recipe: 0.05); standing stays tight


def make_microduck_walk_v2_terrain_env_cfg(play: bool = False, menu: str = "isaaclab",
                                           finetune: bool = False, v2: bool = False, apex: bool = False):
    if menu not in TERRAIN_MENUS:
        raise ValueError(f"menu must be one of {sorted(TERRAIN_MENUS)}, got {menu!r}")
    cfg = make_microduck_walk_v2_env_cfg(play=play, rough=True)

    if finetune and not play:
        # Warm start from a trained walk: inherited step curricula at their final stage, speed
        # range at its final width; only the terrain ladder still climbs.
        _collapse_curricula_to_final(cfg)
        cfg.curriculum.pop("vx_range", None)
        cfg.commands["twist"].ranges.lin_vel_x = (-VX_FINAL_MAX, VX_FINAL_MAX)

    # Ground. The base rough branch mutates its generator for play (random 5×5); build ours
    # AFTER so a play cfg keeps the curriculum layout (one column per type, rows = difficulty).
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TERRAIN_MENUS[menu](
        num_rows=PLAY_ROWS if play else NUM_ROWS,
        seed=V2_TERRAIN_SEED if (v2 and not play) else None,
        border_width=V2_BORDER_WIDTH if (v2 and not play) else BORDER_WIDTH,
    )
    cfg.scene.terrain.max_init_terrain_level = None if play else MAX_INIT_TERRAIN_LEVEL
    # Stock rule, minus its iteration-0 "everyone up a row" artefact (see the mdp docstring).
    cfg.curriculum["terrain_levels"].func = microduck_mdp.terrain_levels_first_reset_guarded

    if v2 and not play:
        cfg.scene.terrain.max_init_terrain_level = V2_MAX_INIT_TERRAIN_LEVEL
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=microduck_mdp.terrain_levels_survival,
            params={"min_tracking": V2_MIN_TRACKING, "min_progress": V2_MIN_PROGRESS,
                    "demote_no_progress": True, "reward_name": "track_linear_velocity",
                    "command_name": "twist"},
        )
        cfg.events["reset_base"] = EventTermCfg(
            func=microduck_mdp.reset_root_on_terrain, mode="reset", params=dict(V2_SPAWN),
        )
        cfg.terminations["out_of_terrain_bounds"].params["margin"] = V2_BOUNDS_MARGIN

    if apex:
        low = cfg.rewards["low_swing"].params
        cfg.rewards["swing_apex"] = RewardTermCfg(
            func=microduck_mdp.feet_swing_apex_reward,
            weight=APEX_WEIGHT,
            params={"sensor_name": low["sensor_name"], "height_sensor_name": low["height_sensor_name"],
                    "command_name": low["command_name"], "target": APEX_TARGET,
                    "min_air_time": APEX_MIN_AIR_TIME, "command_threshold": low["command_threshold"],
                    "gate_tilt_above_deg": low["gate_tilt_above_deg"]},
        )

    if ENABLE_TIGHT_SPAWN and "pose_range" in cfg.events["reset_base"].params:
        pose_range = cfg.events["reset_base"].params["pose_range"]
        pose_range["x"] = (-SPAWN_XY, SPAWN_XY)
        pose_range["y"] = (-SPAWN_XY, SPAWN_XY)

    if ENABLE_STAIR_FOOT_TARGET:
        cfg.rewards["foot_clearance"].params["target_height"] = FOOT_TARGET_HEIGHT
        cfg.rewards["foot_swing_height"].params["target_height"] = FOOT_TARGET_HEIGHT
        for tier in ("std_walking", "std_running"):
            stds = dict(cfg.rewards["pose"].params[tier])
            stds[r".*hip_roll.*"] = HIP_ROLL_STD
            cfg.rewards["pose"].params[tier] = stds
    return cfg


MicroduckWalkV2TerrainRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2RlCfg)
MicroduckWalkV2TerrainRlCfg.experiment_name = "walk_v2_terrain"
MicroduckWalkV2TerrainRlCfg.run_name = "walk_v2_terrain"

MicroduckWalkV2StairsRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2RlCfg)
MicroduckWalkV2StairsRlCfg.experiment_name = "walk_v2_stairs"
MicroduckWalkV2StairsRlCfg.run_name = "walk_v2_stairs"

MicroduckWalkV2StairsFinetuneRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2StairsRlCfg)
MicroduckWalkV2StairsFinetuneRlCfg.run_name = "walk_v2_stairs_ft"
MicroduckWalkV2StairsFinetuneRlCfg.max_iterations = FINETUNE_MAX_ITERATIONS

MicroduckWalkV2StairsV2RlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2StairsFinetuneRlCfg)
MicroduckWalkV2StairsV2RlCfg.run_name = "walk_v2_stairs_v2"

MicroduckWalkV2StairsApexRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2StairsFinetuneRlCfg)
MicroduckWalkV2StairsApexRlCfg.run_name = "walk_v2_stairs_apex"
