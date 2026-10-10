"""Ground grasp: model (opening beak), contract (61 obs / 14 actions), reward signs, mouth schedule."""

import re

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registry first: mjlab's plugin loader imports the task package)
from mjlab_microduck.robot.ground_grasp import (
    MOUTH_CLOSED_RAD,
    MOUTH_JOINT,
    get_ground_grasp_spec,
)
from mjlab_microduck.tasks import mdp as M
from mjlab_microduck.tasks.microduck_ground_grasp_env_cfg import (
    CLOSE_END,
    CLOSE_START,
    MOUTH_OPEN_RAD,
    make_microduck_ground_grasp_env_cfg,
)

SERVO_RE = re.compile(r"^(?!passive_).*")


@pytest.fixture(scope="module")
def model():
    return get_ground_grasp_spec().compile()


def _site(m, d, name):
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def test_mouth_is_passive_and_servos_stay_14(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    assert MOUTH_JOINT in names
    hinges = [n for j, n in enumerate(names) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    servos = [n for n in hinges if SERVO_RE.match(n)]
    assert len(servos) == 14 and MOUTH_JOINT not in servos


def test_underside_hulls_replaced_by_boxes(model):
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in (head, jaw) or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
            assert mesh not in ("bottom_head_shell", "soft_mouth_top", "jaw", "jaw_soft"), mesh
    for name, body in (("upper_beak_collision", head), ("head_rear_collision", head), ("jaw_plate_collision", jaw)):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0 and model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX


def test_jaw_opens_downward_and_closes_on_the_upper_beak(model):
    d = mujoco.MjData(model)
    q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MOUTH_JOINT)]
    d.qpos[3] = 1.0
    d.qpos[q] = 0.0
    mujoco.mj_forward(model, d)
    closed_gap = np.linalg.norm(_site(model, d, "lip_up") - _site(model, d, "lip_low"))
    d.qpos[q] = MOUTH_OPEN_RAD
    mujoco.mj_forward(model, d)
    up, low = _site(model, d, "lip_up"), _site(model, d, "lip_low")
    assert low[2] < up[2] - 0.015            # the jaw tip goes DOWN (head level at the identity pose)
    assert np.linalg.norm(up - low) > closed_gap + 0.015
    # closed (-5 deg): jaw and head are parent/child bodies, so MuJoCo never collides the two boxes; the joint stop
    # ends the travel and the actuator squeezes whatever sits between the pads
    lo, hi = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MOUTH_JOINT)]
    assert abs(lo - MOUTH_CLOSED_RAD) < 1e-6 and hi > MOUTH_OPEN_RAD


def test_mouth_schedule_fraction():
    ph = torch.tensor([0.0, 0.06, 0.2, CLOSE_START, 0.5 * (CLOSE_START + CLOSE_END), CLOSE_END, 0.9])
    f = M.mouth_schedule_fraction(ph, 0.02, 0.10, CLOSE_START, CLOSE_END)
    assert f[0] == 0.0 and 0.0 < f[1] < 1.0 and f[2] == 1.0 and f[3] == 1.0
    assert 0.0 < f[4] < 1.0 and f[5] == 0.0 and f[6] == 0.0


def test_reward_signs_and_contract_terms():
    cfg = make_microduck_ground_grasp_env_cfg()
    r = cfg.rewards
    for name in ("grasp_held", "grasp_lift", "approach_track"):
        assert r[name].weight > 0, name
    # self-negating penalty -> POSITIVE weight (AGENTS.md sign convention)
    assert r["object_pushed"].func is M.object_push_penalty and r["object_pushed"].weight > 0
    assert r["joint_overspeed"].func is M.joint_overspeed_penalty and r["joint_overspeed"].weight > 0
    # >= 0 costs -> negative weights
    for name in ("head_floor_impact", "jaw_floor_impact", "action_rate_l2", "self_collisions"):
        assert r[name].weight < 0, name
    for dead in ("mouth_ground_proximity", "mouth_perpendicular_to_ground", "mouth_payload_force"):
        assert dead not in r
    # the 14-servo action term never sees the mouth actuator; the mouth term is scripted (no policy dims)
    assert cfg.actions["joint_pos"].actuator_names == (r"^(?!passive_).*",)
    assert isinstance(cfg.actions["mouth"], M.MouthScheduleActionCfg)
    # actor obs: the unified 61-D layout, blind to the object
    actor = list(cfg.observations["actor"].terms)
    assert actor == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
                     "head_command", "body_command"]
    assert "object_pos" in cfg.observations["critic"].terms
    assert cfg.commands["twist"].randomize_phase is False
    # sticky friction (a pinched block creeps out of the wedge with pyramidal / impratio 1)
    assert cfg.sim.mujoco.cone == "elliptic" and cfg.sim.mujoco.impratio >= 10


def test_env_steps_on_cpu_with_61_obs_14_actions():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_microduck_ground_grasp_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 61)
    assert env.action_manager.total_action_dim == 14
    mouth = env.action_manager.get_term("mouth")
    act = torch.zeros(2, 14)
    angles, targets = [], []
    for _ in range(int(0.25 * 4.0 / env.step_dt)):        # to phase 0.25: mouth open
        env.step(act)
        angles.append(float(M.mouth_angle_obs(env)[0]))
        targets.append(float(mouth._target[0, 0]))
    # max, not last: with zero actions the robot may fall and reset (mouth closed again) before phase 0.25
    assert max(angles) > 0.3 and max(targets) > 0.3
    env.close()
