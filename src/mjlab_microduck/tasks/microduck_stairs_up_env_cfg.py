"""Stairs UP: the perceptive step-up policy (run 15), the Microduck's stairs policy for flat ground,
steps up and treads. Overview, checkpoint and how it is combined with the stairs-down policy:
``docs/stairs.md``; history: ``docs/archive/stairs_runlog.md`` (§15-16, §19-23).

  obs      61-D family contract + the ToF height map (actor 61 + 2 × 133 = 327-D): a 19 × 7 scan
           (SCAN_PARAMS, 3 cm cells from 9 cm behind to 45 cm ahead of the trunk) read from a map
           built only from the head ToF and drifting odometry (tof_map.tof_map_scan: values, then
           the known mask). The critic reads the exact scan (mdp.terrain_height_scan).
  base     the walk-v2 stairs-mode walk (vx ≤ 0.3, apex reward, torque headroom), 8 s episodes;
           commands forward only (VX_RANGE), the head slot says "look down" (HEAD_DOWN, a weak
           prior: the policy may aim the sensor).
  terrain  ``step_updown`` menu (microduck_terrains): flat, single steps, 3-riser flights up, and
           the same going down (DOWN_TILES); risers 1.5-4 cm by row, tread depth 11-24 cm per tile
           in training (TREAD_DR_RANGE; the play grid keeps 15 cm).
  climb    Δ trunk height (potential-based), each foot on a higher level once, a clean placement
           on a tread mid-flight; edge contact and falls cost; the ladder promotes clean episodes.
  descent  on DOWN_TILES the climb potential is off and its mirrors pay: Δ ground under the feet,
           each foot below the platform once, each new lower level reached by a clean touchdown;
           edge contact costs double; the trunk falling faster than TRUNK_DROP_MAX_VEL costs.
  edge     inside the down-edge zone (mdp.down_edge_zone) a backward lean, the leg pitch chain
           (the crouch) and slowing to EDGE_V are free, the drop cap is EDGE_DROP_MAX_VEL;
           EDGE_SPAWN_PROB of the downhill resets start mid-step-down in a static-test pose
           (tasks/data/step_down_poses.json, scripts/make_step_down_bank.py), the map prefilled.
  robot    the walk model with thighs and shins colliding (MICRODUCK_WALK_LEGCOL_ROBOT_CFG; the
           ``-Backlash-`` twin: MICRODUCK_WALK_BACKLASH_LEGCOL_ROBOT_CFG).
"""
from copy import deepcopy
from dataclasses import replace

from mjlab.envs.mdp.rewards import is_terminated as mjlab_is_terminated
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_LEGCOL_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import tof
from mjlab_microduck.tasks.microduck_terrains import DOWN_TILES
from mjlab_microduck.tasks.microduck_walk_v2_terrain_env_cfg import (
    MicroduckWalkV2StairsFinetuneRlCfg,
    make_microduck_walk_v2_terrain_env_cfg,
)
from mjlab_microduck.tasks.tof_map import tof_map_scan

# ── episode, commands, head, spawn ──────────────────────────────────────────
EPISODE_LENGTH_S = 8.0
VX_RANGE = (0.1, 0.3)            # m/s, forward only
VY_RANGE = (-0.05, 0.05)
WZ_RANGE = (-0.2, 0.2)
HEAD_DOWN = (-0.22, 0.22)        # (neck_pitch, head_pitch) deltas, rad: sensor ≈ 25° below level
HEAD_JITTER = 0.05               # rad, so the slot is a range (alive), not a constant
HEAD_TRACK_WEIGHT = 0.5          # a weak prior: the policy may aim the sensor itself (walk recipe: 2.0)
HEAD_TRACK_STD = 0.5
SPAWN_YAW = 0.35                 # rad, ±20° around +x (into the steps)
SPAWN_LOW_FRAC = 0.8             # resets at the tile origin's level; the rest anywhere = reverse curriculum
SPAWN_MIN_DX = -0.45             # m from the origin (x 0.5 in the tile): spawns stay on their own tile

# ── perception ──────────────────────────────────────────────────────────────
SCAN_PARAMS = {"x_range": (-0.09, 0.45), "y_range": (-0.09, 0.09), "res": 0.03, "nominal": 0.115, "clip": 0.2}
TOF_MAP_KWARGS = {"res": 0.02, "fuse": "mean", "splat": 1, "splat_y": 1, "max_range": 1.2}

# ── climb (rewards are scaled by dt: weight = per-second rate) ──────────────
CLEAN_PROGRESS_WEIGHT = 5000.0   # × Δz: 5000·dt = 1.0 per cm of rise, paid back on the way down
PROGRESS_CLIP = 0.01             # m per step
CLEAN_FOOT_ON_STEP_WEIGHT = 100.0   # 100·dt = 2 per foot, once
FOOT_ON_STEP_MIN_RISE = 0.012    # m, 80 % of the smallest riser
IMPACT_WEIGHT = 0.02             # trunk_vertical_accel_penalty is SELF-NEGATING (returns −|a_z|) → POSITIVE weight
EDGE_WEIGHT = -10.0              # per foot-second on a riser face / edge
EDGE_TOL = 0.005                 # m, height spread under a footprint that counts as "not one level"
FALL_WEIGHT = -500.0             # −500·dt = −10 once per fall
MAX_EDGE_STEPS = 3               # foot-steps of edge contact per episode a promotion still allows
TREAD_DR_RANGE = (0.11, 0.24)    # m, microduck_terrains.STEP_UP_TREAD_RANGE (training only; play keeps 0.15)
PLACEMENT_WEIGHT = 50.0          # 50·dt = 1.0 per well-placed touchdown inside a flight
PLACEMENT_MARGIN = 0.02          # m, the footprint grown by this must sit on one level
PLACEMENT_MIN_STRIDE = 0.04      # m, from the same foot's previous touchdown

# ── descent (on DOWN_TILES) ─────────────────────────────────────────────────
DESCENT_PROGRESS_WEIGHT = CLEAN_PROGRESS_WEIGHT   # × Δ(ground under the feet), 1.0 per cm like the climb
DESCENT_PROGRESS_CLIP = 0.05     # m per step: a foot crossing an edge moves the feet's mean ground by h/2 at once
FOOT_BELOW_STEP_WEIGHT = CLEAN_FOOT_ON_STEP_WEIGHT  # 2 per foot, once
DOWN_EDGE_SCALE = 2.0            # the edge cost × this on the downhill columns
CLEAN_STEP_DOWN_WEIGHT = 25.0    # 0.5 per cm per foot reaching a new lower level with a clean touchdown
TRUNK_DROP_MAX_VEL = 0.3         # m/s: above the trailing foot and flat treads, below a leading drop
TRUNK_DROP_WEIGHT = 150.0        # self-negating → POSITIVE; W/100 per cm the trunk falls faster than the cap

# ── the down edge (mdp.down_edge_zone: an edge 8 cm behind to 14 cm ahead of the trunk) ────────
EDGE_MAX_BACK_DEG = 30.0         # a backward lean up to this is free
EDGE_V = 0.08                    # m/s, the forward target at an edge (commands are 0.1-0.3)
EDGE_DROP_MAX_VEL = 0.15         # m/s, the trunk drop cap inside the zone
EDGE_SHRINK = 0.008              # m off each side of the sole for the downhill edge test
EDGE_SPAWN_PROB = 0.3            # of the resets on a downhill column
EDGE_SPAWN_BANK = "step_down_poses.json"


def make_microduck_stairs_up_env_cfg(play: bool = False):
    cfg = make_microduck_walk_v2_terrain_env_cfg(
        play=play, menu="step_updown", finetune=True, v2=True, apex=True, stairs_mode=True,
    )
    cfg.episode_length_s = EPISODE_LENGTH_S
    if not play:
        # per-tile tread depth for the flights; the play grid keeps 15 cm so batteries stay comparable
        subs = cfg.scene.terrain.terrain_generator.sub_terrains
        for name in ("short_flight", "down_flight"):
            subs[name].step_width_range = TREAD_DR_RANGE

    # Perception: the actor reads the ToF-built map (values + known mask), the critic the exact scan.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (tof.make_tof_sensor_cfg(),)
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=tof_map_scan, params={**SCAN_PARAMS, "sensor_name": tof.TOF_SENSOR_NAME,
                                   "map_kwargs": dict(TOF_MAP_KWARGS)},
    )
    cfg.observations["critic"].terms["height_scan_clean"] = ObservationTermCfg(
        func=microduck_mdp.terrain_height_scan, params={**SCAN_PARAMS, "noise": 0.0},
    )

    # Commands: forward into the steps; the head slot says "look down".
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = VX_RANGE
    tw.ranges.lin_vel_y = VY_RANGE
    tw.ranges.ang_vel_z = WZ_RANGE
    for attr in ("rel_standing_envs", "zero_command_prob", "rel_turn_in_place_envs"):
        if hasattr(tw, attr):
            setattr(tw, attr, 0.0)
    n, h = HEAD_DOWN
    cfg.commands["head_pose"].ranges = (
        (n - HEAD_JITTER, n + HEAD_JITTER),
        (h - HEAD_JITTER, h + HEAD_JITTER),
        (-0.07, 0.07),
        (-0.015, 0.015),
    )
    cfg.rewards["head_pose_tracking"].params["std"] = HEAD_TRACK_STD
    cfg.rewards["head_pose_tracking"].weight = HEAD_TRACK_WEIGHT
    cfg.curriculum.pop("head_pose_range", None)   # the inherited widening would undo the offset

    # Spawn facing the steps (reset_root_on_terrain is wired by walk-v2).
    if not play:
        cfg.events["reset_base"].params["yaw_range"] = (-SPAWN_YAW, SPAWN_YAW)
        cfg.events["reset_base"].params["low_frac"] = SPAWN_LOW_FRAC
    else:
        cfg.events["reset_base"].params["pose_range"]["yaw"] = (-SPAWN_YAW, SPAWN_YAW)

    # The climb: rise, feet on the steps, no impacts, no edge contact, no fall, placed on the treads.
    r = cfg.rewards
    contact_sensor = r["low_swing"].params["sensor_name"]
    r["height_progress"] = RewardTermCfg(
        func=microduck_mdp.trunk_height_progress, weight=CLEAN_PROGRESS_WEIGHT,
        params={"clip": PROGRESS_CLIP, "skip_tile": DOWN_TILES},          # the descent has its own terms
    )
    r["foot_on_step"] = RewardTermCfg(
        func=microduck_mdp.foot_on_step_bonus, weight=CLEAN_FOOT_ON_STEP_WEIGHT,
        params={"sensor_name": contact_sensor, "min_rise": FOOT_ON_STEP_MIN_RISE,
                # in params so the manager resolves the two foot site ids (a default arg stays unresolved)
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
                "sole": True},                                            # measured on the sole, not the site
    )
    r["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty, weight=IMPACT_WEIGHT, params={},
    )
    r["foot_edge_contact"] = RewardTermCfg(
        func=microduck_mdp.foot_edge_contact_cost, weight=EDGE_WEIGHT,
        params={"sensor_name": contact_sensor, "tol": EDGE_TOL,
                "scale_tiles": DOWN_TILES, "tile_scale": DOWN_EDGE_SCALE,
                "shrink": EDGE_SHRINK, "shrink_tiles": DOWN_TILES},
    )
    r["fall"] = RewardTermCfg(func=mjlab_is_terminated, weight=FALL_WEIGHT, params={})
    if not play:
        cfg.curriculum["terrain_levels"].params["max_edge_steps"] = MAX_EDGE_STEPS
    r["foot_placement"] = RewardTermCfg(
        func=microduck_mdp.foot_placement_bonus, weight=PLACEMENT_WEIGHT,
        params={"sensor_name": contact_sensor, "min_rise": FOOT_ON_STEP_MIN_RISE, "margin": PLACEMENT_MARGIN,
                "tol": EDGE_TOL, "min_stride": PLACEMENT_MIN_STRIDE, "down_tiles": DOWN_TILES},
    )

    # The descent: the climb's mirrors on the downhill columns, clean and without a drop.
    r["descent_progress"] = RewardTermCfg(
        func=microduck_mdp.descent_progress, weight=DESCENT_PROGRESS_WEIGHT,
        params={"tiles": DOWN_TILES, "clip": DESCENT_PROGRESS_CLIP},
    )
    r["foot_below_step"] = RewardTermCfg(
        func=microduck_mdp.foot_below_step_bonus, weight=FOOT_BELOW_STEP_WEIGHT,
        params={"sensor_name": contact_sensor, "tiles": DOWN_TILES, "min_rise": FOOT_ON_STEP_MIN_RISE},
    )
    if not play:
        cfg.events["reset_base"].params["min_dx"] = SPAWN_MIN_DX
    r["clean_step_down"] = RewardTermCfg(
        func=microduck_mdp.clean_step_down_bonus, weight=CLEAN_STEP_DOWN_WEIGHT,
        params={"sensor_name": contact_sensor, "tiles": DOWN_TILES, "min_rise": FOOT_ON_STEP_MIN_RISE,
                "tol": EDGE_TOL, "shrink": EDGE_SHRINK},
    )
    r["trunk_drop"] = RewardTermCfg(
        func=microduck_mdp.trunk_drop_penalty, weight=TRUNK_DROP_WEIGHT,
        params={"max_down_vel": TRUNK_DROP_MAX_VEL, "tiles": DOWN_TILES, "zone_max_down_vel": EDGE_DROP_MAX_VEL},
    )

    # At a down edge the crouch, a backward lean and slowing down are free.
    r["upright"] = replace(r["upright"], func=microduck_mdp.upright_edge_lean,
                           params={**r["upright"].params, "tiles": DOWN_TILES, "max_back_deg": EDGE_MAX_BACK_DEG})
    r["pose"] = replace(r["pose"], func=microduck_mdp.variable_posture_edge,
                        params={**r["pose"].params, "tiles": DOWN_TILES, "free_joints": r".*(hip_pitch|knee|ankle).*"})
    r["track_vx_ema"] = replace(r["track_vx_ema"], func=microduck_mdp.track_velocity_ema_edge,
                                params={**r["track_vx_ema"].params, "tiles": DOWN_TILES, "v_edge": EDGE_V})
    lin = r["track_linear_velocity"]
    r["track_linear_velocity"] = replace(lin, func=microduck_mdp.track_linear_velocity_edge,
                                         params={"std": lin.params["std"], "command_name": lin.params["command_name"],
                                                 "tiles": DOWN_TILES, "v_edge": EDGE_V})
    # the walk model, legs colliding (the backlash twin passes ..._BACKLASH_LEGCOL_ROBOT_CFG)
    cfg.scene.entities = {**cfg.scene.entities, "robot": MICRODUCK_WALK_LEGCOL_ROBOT_CFG}
    if not play:
        spawn = EventTermCfg(func=microduck_mdp.reset_at_down_edge, mode="reset",
                             params={"bank": EDGE_SPAWN_BANK, "prob": EDGE_SPAWN_PROB, "tiles": DOWN_TILES,
                                     "gap": (0.003, 0.008), "yaw_noise": 0.05, "lateral": 0.05})
        events = {}
        for k, v in cfg.events.items():                      # right after the joints are reset
            events[k] = v
            if k == "reset_robot_joints":
                events["reset_at_down_edge"] = spawn
        assert "reset_at_down_edge" in events
        cfg.events = events
    return cfg


MicroduckStairsUpRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckWalkV2StairsFinetuneRlCfg)
MicroduckStairsUpRlCfg.experiment_name = "stairs_up"
MicroduckStairsUpRlCfg.run_name = "stairs_up"
MicroduckStairsUpRlCfg.max_iterations = 2000
