"""Stairs DOWN: the step-down specialist (run 18d), the policy that runs while the ToF map shows a
drop ahead (tasks/stairs_switch.py). Overview, checkpoint and the switch: ``docs/stairs.md``;
history: ``docs/archive/stairs_runlog.md`` (§26-28).

It is the stairs-up env (same 327-D obs and network, so it loads the up policy's weights and swaps
with it per step) turned into a descent teacher:

  terrain  down only: single drops (SINGLE_DROP_SHARE) and 20-riser flights on 10 m tiles, risers
           1.5-4.5 cm (RISERS) uniform over the rows (no promotion curriculum), the flights' tread
           drawn per tile as in stairs up; 15 s episodes, enough for the whole staircase.
  spawns   three chained reset events (mdp.reset_at_down_edge): standing on the upper tread with
           the toe 2-12 cm before an edge (tasks/data/edge_stand_poses.json, ORIGIN_SHARE start
           at the tile's platform instead), or crouched in double support right at the edge
           (edge_crouch_poses.json, scripts/make_edge_crouch_bank.py), or mid-step in a pose of
           the static test (step_down_poses.json, scripts/make_step_down_bank.py).
  rewards  stairs up's, plus the main term ``trunk_level``: the trunk height tracks a slewed target
           of 12 cm − one riser over the tread under the higher foot while an edge is within LOOK
           ahead (12 cm over the ground otherwise, mdp.trunk_ground_tracking) + its L1, and
           ``drop_free_step_down``: each new lower level reached with both feet clean, unless the
           trunk fell faster than DROP_GATE_V on the way (mdp.drop_free_level_bonus).
  commands forward only, with a heading target per robot (the stairs go down +x).
"""
from __future__ import annotations

from copy import deepcopy

from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_stairs_up_env_cfg import (
    EDGE_SHRINK,
    EDGE_SPAWN_BANK,
    TREAD_DR_RANGE,
    MicroduckStairsUpRlCfg,
    make_microduck_stairs_up_env_cfg,
)
from mjlab_microduck.tasks.microduck_terrains import BoxSingleStepTerrainCfg

# ── terrain and episode ─────────────────────────────────────────────────────
TILES = ("single_drop", "down_flight")
SINGLE_DROP_SHARE = 0.25                 # of the columns (1 of NUM_COLS); the rest are flights
NUM_COLS = 4
RISERS = (0.015, 0.045)                  # m by row: the static test's drop-free band, the banks' 1.5-4.5 cm
FLIGHT_STEPS = 20                        # risers, as the stairs battery (scripts/eval/stairs_battery.py)
APPROACH = (0.8, 1.3)                    # m: the origin 0.3-0.8 m before the first edge
TILE_X = 10.0                            # m along +x: ≥ 10 − 1.3 − 19 × 0.24 = 4.1 m of bottom
EPISODE_LENGTH_S = 15.0                  # ≤ 3.75 m at 0.25 m/s

# ── commands ────────────────────────────────────────────────────────────────
TWIST = {"lin_vel_x": (0.1, 0.25), "lin_vel_y": (-0.02, 0.02), "ang_vel_z": (-0.2, 0.2)}
HEADING = (-0.15, 0.15)                  # rad, world yaw targets for every robot

# ── spawns (chained after reset_robot_joints; a later one overrides an earlier one) ──
STAND_BANK, CROUCH_BANK = "edge_stand_poses.json", "edge_crouch_poses.json"
STAND_GAP = (0.02, 0.12)                 # m, toe behind the edge
CROUCH_GAP = (0.003, 0.03)
ORIGIN_SHARE = 0.25                      # start on the platform (reset_base), not at an edge
CROUCH_PROB = 0.2                        # of all resets (training only)
MIDSTEP_PROB = 0.15                      # of all resets (training only)
CROUCH_MATCH = 0.01                      # m: the bank crouch is 0-1 cm shallower than the riser asks

# ── the crouch (main term) and the drop-free completion ─────────────────────
STAND_H = 0.12                           # m: the trunk over the lower tread; over the upper = 12 cm − riser
LEVEL_STD = 0.025                        # m
LEVEL_RATE = 0.10                        # m/s: stairs at 0.25 m/s descend ≤ 0.10 (4.5 cm / 11 cm tread)
LOOK = 0.14                              # m ahead of the trunk: stairs up's down-edge zone
LEVEL_WEIGHT = 10.0
LEVEL_L1_WEIGHT = 100.0                  # per m of error
DROP_GATE_V = 0.25                       # m/s: the stairs-up policy steps off at 0.34-0.54
DROP_FREE_WEIGHT = 750.0                 # × dt = 15 per riser


def make_microduck_stairs_down_env_cfg(play: bool = False):
    cfg = make_microduck_stairs_up_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # terrain: down tiles only, every row from the start
    tg = cfg.scene.terrain.terrain_generator
    tg.sub_terrains = {
        "single_drop": BoxSingleStepTerrainCfg(proportion=SINGLE_DROP_SHARE, step_height_range=RISERS,
                                               num_steps=1, down=True, approach_range=APPROACH),
        "down_flight": BoxSingleStepTerrainCfg(proportion=1.0 - SINGLE_DROP_SHARE, step_height_range=RISERS,
                                               num_steps=FLIGHT_STEPS, down=True, approach_range=APPROACH,
                                               step_width_range=TREAD_DR_RANGE),
    }
    tg.num_cols = NUM_COLS
    tg.size = (TILE_X, tg.size[1])
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.pop("terrain_levels", None)

    # commands: forward, heading held
    tw = cfg.commands["twist"]
    for k, v in TWIST.items():
        setattr(tw.ranges, k, v)
    tw.heading_command = True
    tw.rel_heading_envs = 1.0
    tw.ranges.heading = HEADING

    # spawns: standing before an edge, or crouched at it, or mid-step (later ones override)
    def spawn(bank, prob, gap, hover, clear):
        return EventTermCfg(func=microduck_mdp.reset_at_down_edge, mode="reset",
                            params={"bank": bank, "prob": prob, "tiles": TILES, "gap": gap, "yaw_noise": 0.05,
                                    "lateral": 0.05, "max_hover": hover, "clear": clear})
    chain = {"reset_edge_stand": spawn(STAND_BANK, 1.0 - ORIGIN_SHARE, STAND_GAP, 1.0, True),
             "reset_edge_crouch": spawn(CROUCH_BANK, 0.0 if play else CROUCH_PROB, CROUCH_GAP, CROUCH_MATCH, False),
             "reset_at_down_edge": spawn(EDGE_SPAWN_BANK, 0.0 if play else MIDSTEP_PROB, (0.003, 0.008), 0.006, False)}
    events = {}
    for k, v in cfg.events.items():
        if k == "reset_at_down_edge":                        # stairs up's mid-step spawn: replaced by the chain
            continue
        events[k] = v
        if k == "reset_robot_joints":
            events.update(chain)
    assert all(k in events for k in chain)
    cfg.events = events

    # the crouch is the main term; a riser only counts without a drop
    r = cfg.rewards
    r["trunk_level"] = RewardTermCfg(func=microduck_mdp.trunk_ground_tracking, weight=LEVEL_WEIGHT,
                                     params={"std": LEVEL_STD, "rate": LEVEL_RATE, "stand_h": STAND_H,
                                             "look": LOOK, "tiles": TILES})
    r["trunk_level_l1"] = RewardTermCfg(func=microduck_mdp.trunk_level_l1, weight=LEVEL_L1_WEIGHT)
    c = r["clean_step_down"].params
    r["drop_free_step_down"] = RewardTermCfg(
        func=microduck_mdp.drop_free_level_bonus, weight=DROP_FREE_WEIGHT,
        params={"sensor_name": c["sensor_name"], "tiles": TILES, "tol": c["tol"], "shrink": EDGE_SHRINK,
                "v_gate": DROP_GATE_V})
    return cfg


MicroduckStairsDownRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckStairsUpRlCfg)
MicroduckStairsDownRlCfg.experiment_name = "stairs_down"
MicroduckStairsDownRlCfg.run_name = "stairs_down"
MicroduckStairsDownRlCfg.max_iterations = 1000
