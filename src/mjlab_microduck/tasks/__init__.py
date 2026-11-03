from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_walk_v2_env_cfg import (
    make_microduck_walk_v2_env_cfg,
    MicroduckWalkV2RlCfg,
)
from .microduck_walk_v2_terrain_env_cfg import (
    make_microduck_walk_v2_terrain_env_cfg,
    MicroduckWalkV2TerrainRlCfg,
    MicroduckWalkV2StairsRlCfg,
    MicroduckWalkV2StairsFinetuneRlCfg,
    MicroduckWalkV2StairsV2RlCfg,
    MicroduckWalkV2StairsApexRlCfg,
    MicroduckWalkV2StairsModeRlCfg,
    MicroduckWalkV2StairsModeHRRlCfg,
    MODE_HEADROOM_WEIGHT_HR,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_step_up_env_cfg import (
    MicroduckStepUpFreeHeadRlCfg,
    MicroduckStepUpOracleRlCfg,
    MicroduckStepUpTofCleanRlCfg,
    MicroduckStepUpRlCfg,
    make_microduck_step_up_env_cfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# WalkV2 — the walking gait retrained from scratch, fast and straight (no BC, mirror loss on).
register_mjlab_task(
    task_id="Mjlab-WalkV2-Flat-MicroDuck",
    env_cfg=make_microduck_walk_v2_env_cfg(),
    play_env_cfg=make_microduck_walk_v2_env_cfg(play=True),
    rl_cfg=MicroduckWalkV2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-WalkV2-Rough-MicroDuck",
    env_cfg=make_microduck_walk_v2_env_cfg(rough=True),
    play_env_cfg=make_microduck_walk_v2_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckWalkV2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# WalkV2 on the IsaacLab-style duck-scaled terrain grid (microduck_terrains.py):
# the full mix, and a stairs-only menu (straight small steps + pyramids). Untrained.
register_mjlab_task(
    task_id="Mjlab-WalkV2-Terrain-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True),
    rl_cfg=MicroduckWalkV2TerrainRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-WalkV2-Stairs-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(menu="stairs"),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs"),
    rl_cfg=MicroduckWalkV2StairsRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Warm start from the walk-v2 checkpoint (MICRODUCK_WARM_START=1 + --agent.resume True):
# inherited curricula pinned final, only the terrain ladder climbs.
register_mjlab_task(
    task_id="Mjlab-WalkV2-Stairs-Finetune-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs"),
    rl_cfg=MicroduckWalkV2StairsFinetuneRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StairsV2 — survival ladder, spawns on the steps, wide apron (warm start as above).
register_mjlab_task(
    task_id="Mjlab-WalkV2-StairsV2-Finetune-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True, v2=True),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs"),
    rl_cfg=MicroduckWalkV2StairsV2RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StairsApex — v2 + a per-swing apex reward (runs 1-3 never lifted the feet above 1.9 cm).
register_mjlab_task(
    task_id="Mjlab-WalkV2-StairsApex-Finetune-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True, v2=True, apex=True),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs", apex=True),
    rl_cfg=MicroduckWalkV2StairsApexRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StairsMode — the slow rough-ground walk: vx ≤ 0.3, apex reward, torque headroom (run 4).
register_mjlab_task(
    task_id="Mjlab-WalkV2-StairsMode-Finetune-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True, v2=True, apex=True, stairs_mode=True),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs", apex=True, stairs_mode=True),
    rl_cfg=MicroduckWalkV2StairsModeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StairsModeHR — stairs mode with a torque-headroom weight that competes (run 5: −12 vs −3).
register_mjlab_task(
    task_id="Mjlab-WalkV2-StairsModeHR-Finetune-MicroDuck",
    env_cfg=make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True, v2=True, apex=True, stairs_mode=True, headroom_weight=MODE_HEADROOM_WEIGHT_HR),
    play_env_cfg=make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs", apex=True, stairs_mode=True, headroom_weight=MODE_HEADROOM_WEIGHT_HR),
    rl_cfg=MicroduckWalkV2StairsModeHRRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StepUp-Tof — the perceptive step-up skill: stairs mode + the head ToF (125-D actor obs).
register_mjlab_task(
    task_id="Mjlab-StepUp-Tof-MicroDuck",
    env_cfg=make_microduck_step_up_env_cfg(),
    play_env_cfg=make_microduck_step_up_env_cfg(play=True),
    rl_cfg=MicroduckStepUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StepUp-Oracle-Clean — exact height scan around the feet + clean-stepping objective (run 8):
# bounds what perfect terrain memory could give before a ToF height map is built.
register_mjlab_task(
    task_id="Mjlab-StepUp-Oracle-Clean-MicroDuck",
    env_cfg=make_microduck_step_up_env_cfg(perception="oracle", clean_feet=True),
    play_env_cfg=make_microduck_step_up_env_cfg(play=True, perception="oracle", clean_feet=True),
    rl_cfg=MicroduckStepUpOracleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StepUp-Tof-Clean — run 9: the ToF (free head, as run 7b) with run 8's clean-stepping objective.
# Differs from StepUp-Oracle-Clean ONLY in perception, so the two runs split the oracle's gain
# into "the objective" and "perfect terrain memory".
register_mjlab_task(
    task_id="Mjlab-StepUp-Tof-Clean-MicroDuck",
    env_cfg=make_microduck_step_up_env_cfg(free_head=True, perception="tof", clean_feet=True),
    play_env_cfg=make_microduck_step_up_env_cfg(play=True, free_head=True, perception="tof", clean_feet=True),
    rl_cfg=MicroduckStepUpTofCleanRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# StepUp-Tof-FreeHead — the same skill with the head free to aim the sensor (run 7b).
register_mjlab_task(
    task_id="Mjlab-StepUp-Tof-FreeHead-MicroDuck",
    env_cfg=make_microduck_step_up_env_cfg(free_head=True),
    play_env_cfg=make_microduck_step_up_env_cfg(play=True, free_head=True),
    rl_cfg=MicroduckStepUpFreeHeadRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-WalkV2-Flat-Backlash-MicroDuck", make_microduck_walk_v2_env_cfg, {}, MicroduckWalkV2RlCfg, _BL_WALK),
    ("Mjlab-WalkV2-Rough-Backlash-MicroDuck", make_microduck_walk_v2_env_cfg, {"rough": True}, MicroduckWalkV2RlCfg, _BL_WALK),
    ("Mjlab-WalkV2-Terrain-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {}, MicroduckWalkV2TerrainRlCfg, _BL_WALK),
    ("Mjlab-WalkV2-Stairs-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {"menu": "stairs"}, MicroduckWalkV2StairsRlCfg, _BL_WALK),
    ("Mjlab-WalkV2-Stairs-Finetune-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {"menu": "stairs", "finetune": True}, MicroduckWalkV2StairsFinetuneRlCfg, _BL_WALK),
    ("Mjlab-WalkV2-StairsV2-Finetune-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {"menu": "stairs", "finetune": True, "v2": True}, MicroduckWalkV2StairsV2RlCfg, _BL_WALK),
    ("Mjlab-WalkV2-StairsApex-Finetune-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {"menu": "stairs", "finetune": True, "v2": True, "apex": True}, MicroduckWalkV2StairsApexRlCfg, _BL_WALK),
    ("Mjlab-WalkV2-StairsMode-Finetune-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {"menu": "stairs", "finetune": True, "v2": True, "apex": True, "stairs_mode": True}, MicroduckWalkV2StairsModeRlCfg, _BL_WALK),
    ("Mjlab-WalkV2-StairsModeHR-Finetune-Backlash-MicroDuck", make_microduck_walk_v2_terrain_env_cfg, {"menu": "stairs", "finetune": True, "v2": True, "apex": True, "stairs_mode": True, "headroom_weight": MODE_HEADROOM_WEIGHT_HR}, MicroduckWalkV2StairsModeHRRlCfg, _BL_WALK),
    ("Mjlab-StepUp-Tof-Backlash-MicroDuck", make_microduck_step_up_env_cfg, {}, MicroduckStepUpRlCfg, _BL_WALK),
    ("Mjlab-StepUp-Tof-FreeHead-Backlash-MicroDuck", make_microduck_step_up_env_cfg, {"free_head": True}, MicroduckStepUpFreeHeadRlCfg, _BL_WALK),
    ("Mjlab-StepUp-Oracle-Clean-Backlash-MicroDuck", make_microduck_step_up_env_cfg, {"perception": "oracle", "clean_feet": True}, MicroduckStepUpOracleRlCfg, _BL_WALK),
    ("Mjlab-StepUp-Tof-Clean-Backlash-MicroDuck", make_microduck_step_up_env_cfg, {"free_head": True, "perception": "tof", "clean_feet": True}, MicroduckStepUpTofCleanRlCfg, _BL_WALK),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
