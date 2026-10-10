"""Microduck ground GRASP task — the ground-pick slot, with an opening beak and a real object to pick up.

Design: docs/superpowers/specs/2026-10-10-ground-grasp-design.md (Rémi's brief: the beak must go lower, flatter than
today's straight-down head, with a vertical final approach that does not push the object away).

Built ON the ground-pick env (same 61-D obs, phase command [cos 2πφ, sin 2πφ, 0], DR / obs-noise / delay stack),
with:
  - the ground-grasp robot (all-collisions + opening jaw on ``passive_mouth`` + beak boxes, robot/ground_grasp.py);
  - a scripted mouth (MouthScheduleAction, action_dim 0): opens on the way down, closes at φ_close, squeezes after;
  - one free object per env, placed in front of the robot (the policy is BLIND to it, like the ball kick: robustness
    comes from placement DR), size / mass / friction randomised per episode;
  - episode = ONE cycle from φ = 0 (standing), 4 s, like robotd's A-button pick (which hands back to the stand net
    at φ = 0.7, so the move must be standing again, object held, by then).
Rewards: grasp held/lifted (main), a slewed approach target that goes above the object and then straight down,
object-pushed and head-on-floor penalties, plus the ground-pick standing/regularisation stack.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.ground_grasp import (
    GRASP_OBJECT_CFG,
    MICRODUCK_GROUND_GRASP_ROBOT_CFG,
    MOUTH_CLOSED_RAD,
    MOUTH_JOINT,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Timeline (phase units, period GG_PERIOD) ──────────────────────────────────
GG_PERIOD = 4.0            # robotd ground_pick_period default: no runtime config change
MOUTH_OPEN_RAD = 0.45      # 26° of the 30° the servo allows (margin for the DR scale up to x1.05)
MOUTH_OPEN_START, MOUTH_OPEN_END = 0.02, 0.10
CLOSE_START, CLOSE_END = 0.30, 0.36
APPROACH_START, HOVER_AT, DOWN_AT = 0.05, 0.20, 0.29
HOVER_HEIGHT = 0.06        # the last 6 cm of the approach are vertical
HELD_FROM = 0.40           # grasp_held pays from here (mouth closed + a little lift time)
RETURN_START, RETURN_END = 0.40, 0.62   # stand-up gates (robotd hands over at 0.70)

# ── Object placement (robot yaw frame) ────────────────────────────────────────
# Static pose study (notes/ground-grasp/poses): at a 60° beak pitch the lips land 9-13 cm in front of the feet.
OBJECT_NOMINAL_XY = (0.10, 0.0)      # from the standing root; the study's 60° lips span 8-11.5 cm from the feet
OBJECT_NOISE_STAGES = [
    {"step": 0, "noise_xy": (0.01, 0.01)},
    {"step": 600 * 24, "noise_xy": (0.015, 0.015)},
    {"step": 1200 * 24, "noise_xy": (0.02, 0.02)},
]
OBJECT_HALF_SCALE = (0.7, 1.3)       # x the nominal 1.0 x 1.0 x 0.8 cm half-extents: 1.4-2.6 cm wide, 1.1-2.1 cm tall
OBJECT_MASS_SCALE = (0.4, 2.0)       # x 12 g
OBJECT_FRICTION = (1.1, 1.6)

# Grasp point (pad midpoint) at HOME relative to the robot root, base frame (measured on the model).
GRASP_POINT_HOME_B = (0.061, 0.0, 0.101)

# ── Variant knobs ──────────────────────────────────────────────────────────────
BEAK_PITCH_PREF_WEIGHT = 0.0         # >0: Gaussian pull of the beak pitch toward BEAK_PITCH_TARGET in the grasp window
BEAK_PITCH_TARGET = 60.0


def make_microduck_ground_grasp_env_cfg(
    play: bool = False,
    beak_pitch_pref_weight: float = BEAK_PITCH_PREF_WEIGHT,
    beak_pitch_target: float = BEAK_PITCH_TARGET,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=False)

    # ── Scene: grasp robot + object; contact sensors ──────────────────────────
    cfg.scene.entities = {"robot": MICRODUCK_GROUND_GRASP_ROBOT_CFG, "object": GRASP_OBJECT_CFG}
    feet_ground_cfg, self_collision_cfg, _old_head = cfg.scene.sensors

    def obj_contact(name, robot_geom):
        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="geom", pattern=robot_geom, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="object", entity="object"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        )

    head_floor_cfg = ContactSensorCfg(
        name="head_floor_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",), reduce="netforce", num_slots=1,
    )
    jaw_floor_cfg = ContactSensorCfg(
        name="jaw_floor_contact",
        primary=ContactMatch(mode="body", pattern="jaw", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",), reduce="netforce", num_slots=1,
    )
    object_floor_cfg = ContactSensorCfg(
        name="object_floor_contact",
        primary=ContactMatch(mode="body", pattern="object", entity="object"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )
    cfg.scene.sensors = (
        feet_ground_cfg, self_collision_cfg, head_floor_cfg, jaw_floor_cfg, object_floor_cfg,
        obj_contact("object_jaw_contact", "jaw_plate_collision"),
        obj_contact("object_upper_contact", "upper_beak_collision"),
    )
    cfg.sim.nconmax = 60
    cfg.episode_length_s = GG_PERIOD

    # ── Actions: the 14 servos (the mouth actuator is passive_* -> excluded) + the scripted mouth ──
    jp = cfg.actions["joint_pos"]
    assert isinstance(jp, JointPositionActionCfg)
    jp.actuator_names = (r"^(?!passive_).*",)
    cfg.actions["mouth"] = microduck_mdp.MouthScheduleActionCfg(
        entity_name="robot", joint_name=MOUTH_JOINT, command_name="twist",
        closed_rad=MOUTH_CLOSED_RAD, open_rad=MOUTH_OPEN_RAD,
        open_start=MOUTH_OPEN_START, open_end=MOUTH_OPEN_END,
        close_start=CLOSE_START, close_end=CLOSE_END,
    )

    # ── Command: one cycle per episode from φ = 0 (standing), like the A button ──
    cfg.commands["twist"].randomize_phase = False
    cfg.commands["twist"].period = GG_PERIOD

    # ── Rewards ───────────────────────────────────────────────────────────────
    for name in ("mouth_ground_proximity", "mouth_perpendicular_to_ground", "mouth_payload_force",
                 "head_impact_penalty"):
        cfg.rewards.pop(name, None)

    cfg.rewards["grasp_held"] = RewardTermCfg(
        func=microduck_mdp.grasp_held, weight=15.0,
        params={"command_name": "twist", "phase_start": HELD_FROM},
    )
    cfg.rewards["grasp_lift"] = RewardTermCfg(
        func=microduck_mdp.grasp_lift, weight=6.0,
        params={"command_name": "twist", "phase_start": CLOSE_START + 0.03, "max_lift": 0.04},
    )
    cfg.rewards["approach_track"] = RewardTermCfg(
        func=microduck_mdp.grasp_approach_track, weight=10.0,
        params={"command_name": "twist", "std": 0.03, "std_wide": 0.08, "phase_start": APPROACH_START,
                "phase_end": CLOSE_END,
                "move_start": APPROACH_START, "hover_at": HOVER_AT, "down_at": DOWN_AT,
                "hover_height": HOVER_HEIGHT},
    )
    cfg.rewards["object_pushed"] = RewardTermCfg(
        func=microduck_mdp.object_push_penalty, weight=3.0,   # self-negating (<= 0): positive weight
        params={"command_name": "twist", "phase_end": CLOSE_END, "max_dist": 0.04},
    )
    cfg.rewards["head_floor_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost, weight=-0.5,
        params={"sensor_name": "head_floor_contact", "threshold": 1.0},
    )
    cfg.rewards["jaw_floor_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost, weight=-0.5,
        params={"sensor_name": "jaw_floor_contact", "threshold": 3.0},
    )
    if beak_pitch_pref_weight > 0:
        cfg.rewards["beak_pitch_pref"] = RewardTermCfg(
            func=microduck_mdp.beak_pitch_pref, weight=beak_pitch_pref_weight,
            params={"command_name": "twist", "target_deg": beak_pitch_target, "std_deg": 15.0,
                    "phase_start": HOVER_AT, "phase_end": CLOSE_END},
        )

    # Stand-up / return terms re-timed to the grasp timeline (robotd hands over at 0.70).
    for name in ("ground_pick_return_pose_legs", "ground_pick_return_pose_neck", "return_upright"):
        cfg.rewards[name].params["hold_end"] = RETURN_START
        cfg.rewards[name].params["rise_end"] = RETURN_END
    cfg.rewards["neck_vel_descent"].params["hold_end"] = CLOSE_START

    # ── Observations: actor unchanged (61-D, blind); critic sees the object, the mouth, the contacts ──
    critic = cfg.observations["critic"].terms
    critic["object_pos"] = ObservationTermCfg(func=microduck_mdp.ball_pos_in_base, params={"asset_name": "object"})
    critic["object_vel"] = ObservationTermCfg(func=microduck_mdp.ball_vel_in_base, params={"asset_name": "object"})
    critic["object_rel_grasp"] = ObservationTermCfg(func=microduck_mdp.object_pos_rel_grasp_point)
    critic["object_size"] = ObservationTermCfg(func=microduck_mdp.object_size_obs)
    critic["mouth_angle"] = ObservationTermCfg(func=microduck_mdp.mouth_angle_obs)
    critic["grasp_contacts"] = ObservationTermCfg(func=microduck_mdp.grasp_contacts_obs)

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events.pop("sample_mouth_payload", None)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)   # handed over from the stand net
    obj = SceneEntityCfg("object", geom_names=("object_geom",))
    cfg.events["object_size"] = EventTermCfg(
        func=dr.geom_size, mode="reset",
        params={"asset_cfg": obj, "operation": "scale", "ranges": OBJECT_HALF_SCALE},
    )
    cfg.events["object_mass"] = EventTermCfg(          # mass and inertia scaled together (e^(2 alpha))
        func=dr.pseudo_inertia, mode="reset",
        params={"asset_cfg": SceneEntityCfg("object", body_names=("object",)),
                "alpha_range": (math.log(OBJECT_MASS_SCALE[0]) / 2.0, math.log(OBJECT_MASS_SCALE[1]) / 2.0)},
    )
    cfg.events["object_friction"] = EventTermCfg(
        func=dr.geom_friction, mode="reset",
        params={"asset_cfg": obj, "operation": "abs", "axes": [0], "ranges": OBJECT_FRICTION},
    )
    # after the robot reset + the object size (events run in insertion order)
    cfg.events["reset_object"] = EventTermCfg(
        func=microduck_mdp.reset_grasp_object, mode="reset",
        params={"nominal_xy": OBJECT_NOMINAL_XY, "noise_xy": OBJECT_NOISE_STAGES[0]["noise_xy"],
                "asset_name": "object"},
    )
    cfg.events["record_grasp_start"] = EventTermCfg(
        func=microduck_mdp.record_grasp_start, mode="reset", params={"offset_b": GRASP_POINT_HOME_B},
    )

    # ── Terminations: a fall ends the episode (4 s episodes: nothing to recover for) ──
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(80.0)}, time_out=False,
    )

    # ── Curriculum: widen the object placement noise ──────────────────────────
    cfg.curriculum["object_noise"] = CurriculumTermCfg(
        func=microduck_mdp.grasp_object_noise_curriculum,
        params={"event_name": "reset_object", "stages": OBJECT_NOISE_STAGES},
    )
    return cfg


MicroduckGroundGraspRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="ground_grasp",
    run_name="ground_grasp",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=1500,
)
