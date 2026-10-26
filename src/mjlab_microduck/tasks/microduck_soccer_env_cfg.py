"""Soccer 1v1: two ducks, one ball, two goals — a match the high-level policy learns by self-play.

The learner is blue. Its action is a twist (vx, vy, yaw rate in [-1, 1], mapped by
`frozen_walker.action_to_twist` inside velstand's training ranges, reverse capped at 0.1 m/s); the
frozen `velstand.onnx` walks it (`soccer/actions.py`). Red is the same walker driven by a frozen
snapshot of the learner, swapped in by `SelfPlayRunner` — until the first one exists red just
stands there.

One episode is one point: it ends on a goal, on the clock (a draw), or if the ball escapes the
pitch. Each reset is either a kickoff or a practice situation — a shooting, approach or defending
drill of at most 10 s (`soccer/scenarios.py`) — so every skill is practised from the start.

Reward has Unity's core: +5 for a goal, -5 for conceding, and an existential penalty summing to -1
over a full match, plus two outcome terms — the ball's potential progress toward the goal and what
each kick did to it. Getting to the ball, stopping and aiming is taught by the scripted striker
(`soccer/oracle.py`) through `PpoWithTeacher`; the runner fades the teacher out once the learner
holds its own against it, then switches to self-play (`soccer/selfplay_runner.py`).

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
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
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
from mjlab_microduck.soccer.mdp import BALL, BALL_RADIUS, BLUE, RED, TRUNK_Z
from mjlab_microduck.soccer.oracle import teacher_action
from mjlab_microduck.soccer.teacher_ppo import PpoWithTeacherCfg
from mjlab_microduck.soccer.scenarios import KICKOFF, NAMES
from mjlab_microduck.tasks import mdp as microduck_mdp

PHYSICS_DT = 0.005  # velstand's
DECISION_DT = 0.1  # the match policy decides at 10 Hz; the walker runs 5 ticks per decision
DECIMATION = round(DECISION_DT / PHYSICS_DT)
EPISODE_LENGTH_S = 45.0

# A goal is worth 5 (an event on one step, so the weight carries 1 / step_dt). At 1 it was 0.08 per
# point in practice against ~2-4 of shaping, and the drive run optimised the shaping instead.
GOAL_WEIGHT = 5.0 / DECISION_DT

# One shaping term per skill, each paid only for an outcome (`soccer_mdp`). The potentials are in
# metres and mjlab multiplies by step_dt, so these weights make a metre of progress worth the
# number in front of `/ DECISION_DT`.
# Getting to the ball and setting up the shot is taught by the scripted teacher (`PpoWithTeacher`),
# so only the outcomes are rewarded here.
BALL_POTENTIAL_WEIGHT = 1.0 / DECISION_DT  # move the ball: +1 per metre nearer the red goal
KICK_OUTCOME_WEIGHT = 1.0 / DECISION_DT  # kick: +1 for one sending the ball at goal at 1 m/s

# Shaping is taken away by competence, not by the clock (`soccer_mdp.competence_gate`):
# reward name -> (base weight, skill signal, success rate where fading starts, where it ends, floor).
# Reaching the ball is measured in the APPROACH drill, everything about the ball in the SHOOT drill.
# The ball terms measure outcomes and keep a quarter: every run that took all shaping away drew.
SHAPING_GATE = {
    "ball_potential": (BALL_POTENTIAL_WEIGHT, "shoot", 0.2, 0.6, 0.25),
    "kick_outcome": (KICK_OUTCOME_WEIGHT, "shoot", 0.2, 0.6, 0.25),
}
# Kickoff is half of the experience: matches are what the policy is for. The drills keep each
# skill in the data (the teacher labels all of it).
SCENARIO_PROBS = (0.5, 0.2, 0.15, 0.15)
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


def make_microduck_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    match = {"match": ObservationTermCfg(func=soccer_mdp.match_obs, params={"team": 1})}
    observations = {
        "actor": ObservationGroupCfg(terms=dict(match)),
        "critic": ObservationGroupCfg(terms=dict(match)),
        # Red's view, same layout as the actor's: what the self-play snapshot is fed.
        "opponent": ObservationGroupCfg(
            terms={"match": ObservationTermCfg(func=soccer_mdp.match_obs, params={"team": -1})}
        ),
        # The scripted striker's action for blue: the label `PpoWithTeacher` imitates. Read by
        # neither actor nor critic; it rides along in the rollout storage.
        "teacher": ObservationGroupCfg(terms={"oracle": ObservationTermCfg(func=teacher_action)}),
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
            "blue_walk": WalkerCommandActionCfg(entity_name=BLUE, opponent_name=RED),
            "red_walk": WalkerCommandActionCfg(entity_name=RED, opponent_name=BLUE, source="opponent"),
        },
        events={
            "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "expand_bam_friction_fields": EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup"),
            # Kickoff or a practice situation (shoot / approach / defend), drawn per reset.
            "scenario": EventTermCfg(func=soccer_mdp.reset_scenario, mode="reset", params={"probs": SCENARIO_PROBS}),
        },
        rewards={
            "goal": RewardTermCfg(func=soccer_mdp.goal_scored_reward, weight=GOAL_WEIGHT, params={"by": "blue"}),
            "conceded": RewardTermCfg(func=soccer_mdp.goal_scored_reward, weight=-GOAL_WEIGHT, params={"by": "red"}),
            "existential": RewardTermCfg(func=soccer_mdp.existential, weight=-1.0 / EPISODE_LENGTH_S),
            "fallen": RewardTermCfg(func=soccer_mdp.fallen, weight=-0.5),
            "ball_potential": RewardTermCfg(
                func=soccer_mdp.PotentialShaping, weight=BALL_POTENTIAL_WEIGHT, params={"kind": "ball"}
            ),
            "kick_outcome": RewardTermCfg(func=soccer_mdp.KickOutcome, weight=KICK_OUTCOME_WEIGHT),
        },
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            # Practice situations are short: a truncation, not a failure.
            "drill_timeout": TerminationTermCfg(func=soccer_mdp.drill_timed_out, time_out=True),
            "blue_scored": TerminationTermCfg(func=soccer_mdp.goal_scored, params={"by": "blue"}),
            "red_scored": TerminationTermCfg(func=soccer_mdp.goal_scored, params={"by": "red"}),
            "ball_out": TerminationTermCfg(func=soccer_mdp.ball_left_pitch),
            "nan_state": TerminationTermCfg(func=soccer_mdp.any_state_is_nan),
        },
        curriculum={
            "competence_gate": CurriculumTermCfg(func=soccer_mdp.competence_gate, params={"terms": SHAPING_GATE}),
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
        # Matches only, and no shaping: play (and the 50 fps renders) should show the game itself.
        cfg.events["scenario"].params["probs"] = tuple(1.0 if i == KICKOFF else 0.0 for i in range(len(NAMES)))
        cfg.curriculum = {}
        for name in SHAPING_GATE:
            cfg.rewards[name].weight = 0.0
    return cfg


MicroduckSoccerRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithTeacherCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Low: the actions are clipped to [-1, 1], so a wider Gaussian costs the policy nothing it can
        # see, and at 0.005 the aim run's std climbed to 2.9 — random play — by iteration 400. At 0.001
        # the drive run's collapsed to 0.07 by iteration 1300 and it stopped exploring.
        entropy_coef=0.002,
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
