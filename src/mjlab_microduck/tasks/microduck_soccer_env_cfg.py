"""Soccer 1v1: two ducks, one ball, two goals — a match the high-level policy learns by self-play.

The learner is blue. Its action is a twist (vx, vy, yaw rate in [-1, 1], mapped by
`frozen_walker.action_to_twist` inside velstand's training ranges, reverse capped at 0.1 m/s); the
frozen `velstand.onnx` walks it (`soccer/actions.py`). Red is the same walker driven by a frozen
snapshot of the learner, swapped in by `SelfPlayRunner` — until the first one exists red just
stands there.

One episode is one point, like Unity's soccer: it ends on a goal, on the clock (a draw), or if the
ball escapes the pitch. Reward is Unity's: +1 for a goal, -1 for conceding, and an existential
penalty summing to -1 over a full match so that a draw is worth less than a win. Two shaping terms
(close on the ball, move the ball toward the goal) bootstrap the first few hundred iterations and
are annealed to zero — a policy still paid for chasing the ball would chase rather than score.

Rewards are multiplied by step_dt (0.1 s) in mjlab, so an event that should be worth 1 has weight
1 / step_dt.

**Not a member of the 61D policy family.** The high-level obs is 26D match state; nothing here is
a policy robotd loads into a slot, and it must not be published as one.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    MICRODUCK_BALL_CFG,
)
from mjlab_microduck.soccer import mdp as soccer_mdp
from mjlab_microduck.soccer.actions import WalkerCommandActionCfg
from mjlab_microduck.soccer.field import KICKOFF_X, add_field
from mjlab_microduck.soccer.mdp import BALL, BALL_RADIUS, BLUE, RED
from mjlab_microduck.tasks import mdp as microduck_mdp

PHYSICS_DT = 0.005  # velstand's
DECISION_DT = 0.1  # the match policy decides at 10 Hz; the walker runs 5 ticks per decision
DECIMATION = round(DECISION_DT / PHYSICS_DT)
EPISODE_LENGTH_S = 45.0

TRUNK_Z = 0.125  # velstand's reset height
KICKOFF_NOISE = {"x": (-0.15, 0.15), "y": (-0.3, 0.3), "yaw": (-0.35, 0.35)}
BALL_NOISE = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}

GOAL_WEIGHT = 1.0 / DECISION_DT
APPROACH_WEIGHT = 0.5
BALL_TO_GOAL_WEIGHT = 1.0
# Facing the ball pays w per second and can be collected standing still, so it is kept small: at
# most 4.5 over a full point, against the speed terms that only pay while something moves.
FACE_BALL_WEIGHT = 0.1
# Shaping off by ~2000 iterations: long enough to find the ball and the goal, short enough that
# the match reward is what the late policy is shaped by.
SHAPING_STAGES = ((0, 1.0), (1000, 0.5), (1500, 0.25), (2000, 0.0))
NUM_STEPS_PER_ENV = 24


def _placed(spec_fn, root_body: str, pos, quat):
    """`spec_fn` with its root body moved to the kickoff pose.

    mjlab attaches every entity at the world origin and its `Simulation` runs the first
    `mj_forward` on qpos0, before any reset applies `init_state`. Two ducks whose trunk sits at
    the origin in their XML are then one duck inside another (727 contacts, nconmax overflow).
    Moving the body in the spec makes qpos0 itself a legal kickoff.
    """

    def build():
        spec = spec_fn()
        body = spec.body(root_body)
        body.pos, body.quat = list(pos), list(quat)
        return spec

    return build


def _robot(x: float, yaw_quat: tuple[float, float, float, float]):
    pos = (x, 0.0, TRUNK_Z)
    return replace(
        MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
        spec_fn=_placed(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG.spec_fn, "trunk_base", pos, yaw_quat),
        init_state=replace(HOME_FRAME, pos=pos, rot=yaw_quat),
    )


def _ball():
    pos = (0.0, 0.0, BALL_RADIUS)
    return replace(
        MICRODUCK_BALL_CFG,
        spec_fn=_placed(MICRODUCK_BALL_CFG.spec_fn, "ball", pos, (1.0, 0.0, 0.0, 0.0)),
        init_state=replace(MICRODUCK_BALL_CFG.init_state, pos=pos),
    )


def _shaping_curriculum(reward_name: str, weight: float) -> CurriculumTermCfg:
    return CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": reward_name,
            "weight_stages": [{"step": it * NUM_STEPS_PER_ENV, "weight": weight * k} for it, k in SHAPING_STAGES],
        },
    )


def make_microduck_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    match = {"match": ObservationTermCfg(func=soccer_mdp.match_obs, params={"team": 1})}
    observations = {
        "actor": ObservationGroupCfg(terms=dict(match)),
        "critic": ObservationGroupCfg(terms=dict(match)),
        # Red's view, same layout as the actor's: what the self-play snapshot is fed.
        "opponent": ObservationGroupCfg(
            terms={"match": ObservationTermCfg(func=soccer_mdp.match_obs, params={"team": -1})}
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
        scene=SceneCfg(
            num_envs=4096,
            # The pitch is world geometry, at the origin of every world, so the ducks and the ball
            # must be too: a non-zero spacing would place them away from their own field.
            env_spacing=0.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                BLUE: _robot(-KICKOFF_X, (1.0, 0.0, 0.0, 0.0)),
                RED: _robot(KICKOFF_X, (0.0, 0.0, 0.0, 1.0)),
                BALL: _ball(),
            },
            spec_fn=add_field,
        ),
        observations=observations,
        actions={
            "blue_walk": WalkerCommandActionCfg(entity_name=BLUE),
            "red_walk": WalkerCommandActionCfg(entity_name=RED, source="opponent"),
        },
        events={
            "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "expand_bam_friction_fields": EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup"),
            **{
                f"kickoff_{name}": EventTermCfg(
                    func=envs_mdp.reset_root_state_uniform,
                    mode="reset",
                    params={"pose_range": KICKOFF_NOISE, "asset_cfg": SceneEntityCfg(name)},
                )
                for name in (BLUE, RED)
            },
            "kickoff_ball": EventTermCfg(
                func=envs_mdp.reset_root_state_uniform,
                mode="reset",
                params={"pose_range": BALL_NOISE, "asset_cfg": SceneEntityCfg(BALL)},
            ),
        },
        rewards={
            "goal": RewardTermCfg(func=soccer_mdp.goal_scored_reward, weight=GOAL_WEIGHT, params={"by": "blue"}),
            "conceded": RewardTermCfg(func=soccer_mdp.goal_scored_reward, weight=-GOAL_WEIGHT, params={"by": "red"}),
            "existential": RewardTermCfg(func=soccer_mdp.existential, weight=-1.0 / EPISODE_LENGTH_S),
            "ball_approach": RewardTermCfg(func=soccer_mdp.ball_approach_speed, weight=APPROACH_WEIGHT),
            "ball_to_goal": RewardTermCfg(func=soccer_mdp.ball_to_goal_speed, weight=BALL_TO_GOAL_WEIGHT),
            "face_ball": RewardTermCfg(func=soccer_mdp.facing_ball, weight=FACE_BALL_WEIGHT),
            "fallen": RewardTermCfg(func=soccer_mdp.fallen, weight=-0.5),
        },
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "blue_scored": TerminationTermCfg(func=soccer_mdp.goal_scored, params={"by": "blue"}),
            "red_scored": TerminationTermCfg(func=soccer_mdp.goal_scored, params={"by": "red"}),
            "ball_out": TerminationTermCfg(func=soccer_mdp.ball_left_pitch),
            "nan_state": TerminationTermCfg(func=soccer_mdp.any_state_is_nan),
        },
        curriculum={
            "ball_approach_weight": _shaping_curriculum("ball_approach", APPROACH_WEIGHT),
            "ball_to_goal_weight": _shaping_curriculum("ball_to_goal", BALL_TO_GOAL_WEIGHT),
            "face_ball_weight": _shaping_curriculum("face_ball", FACE_BALL_WEIGHT),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.0),
            distance=3.6,
            elevation=-55.0,
            azimuth=90.0,
            max_extra_envs=0,  # every world is on the same pitch; neighbours would overlap
        ),
        sim=SimulationCfg(
            # Two all-collision ducks (velstand budgets 200 for one), the ball and the walls.
            nconmax=400,
            njmax=3000,
            mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20),
        ),
    )

    if play:
        cfg.scene.num_envs = 1
        # No shaping in play: the log should show the match, not the bootstrap.
        cfg.curriculum = {}
        cfg.rewards["ball_approach"].weight = 0.0
        cfg.rewards["ball_to_goal"].weight = 0.0
        cfg.rewards["face_ball"].weight = 0.0
    return cfg


MicroduckSoccerRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,  # a goal can be ~10 s (100 decisions) after the touch that made it
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="soccer_1v1",
    run_name="soccer_1v1",
    save_interval=100,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=5_000,
)
