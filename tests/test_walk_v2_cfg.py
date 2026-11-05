"""Cfg invariants for Walk v2 (the walk retrained from scratch: velocity recipe + the validated
VelStandFast reward fixes, no BC, mirror loss on). CPU only, never steps an env."""
import math
from types import SimpleNamespace

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velocity_env_cfg as vel
from mjlab_microduck.tasks import microduck_walk_v2_env_cfg as w2
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

TASKS = (
    "Mjlab-WalkV2-Flat-MicroDuck",
    "Mjlab-WalkV2-Rough-MicroDuck",
    "Mjlab-WalkV2-Flat-Backlash-MicroDuck",
    "Mjlab-WalkV2-Rough-Backlash-MicroDuck",
)
PENALTIES = ("action_rate_l2", "angular_momentum", "body_ang_vel", "dof_pos_limits",
             "foot_clearance", "foot_slip", "foot_swing_height", "self_collisions", "low_swing")


# ── Cfg ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_tasks_registered(task):
    assert load_env_cfg(task) is not None and load_env_cfg(task, play=True) is not None
    assert load_rl_cfg(task).experiment_name == "walk_v2"


def test_backlash_variant_uses_walk_model():
    cfg = load_env_cfg("Mjlab-WalkV2-Rough-Backlash-MicroDuck")
    assert cfg.scene.entities["robot"].spec_fn is MICRODUCK_WALK_BACKLASH_ROBOT_CFG.spec_fn


@pytest.mark.parametrize("play", [False, True])
def test_only_the_diagnosed_terms_differ_from_velocity(play):
    base = vel.make_microduck_velocity_env_cfg(play=play)
    cfg = w2.make_microduck_walk_v2_env_cfg(play=play)
    assert set(cfg.rewards) == set(base.rewards) | {"track_vx_ema", "low_swing"}
    changed = {"air_time", "track_angular_velocity", "pose"}
    for name in base.rewards:
        if name in changed:
            continue
        assert cfg.rewards[name].func is base.rewards[name].func, name
        assert cfg.rewards[name].weight == base.rewards[name].weight, name
        assert cfg.rewards[name].params == base.rewards[name].params, name
    assert cfg.observations["actor"].terms.keys() == base.observations["actor"].terms.keys()
    added = {"vx_range", "gait_stats"} - ({"vx_range"} if play else set())
    assert set(cfg.curriculum) == set(base.curriculum) | added


def test_actor_obs_is_the_61d_family_layout_and_critic_is_privileged():
    cfg = w2.make_microduck_walk_v2_env_cfg()
    base = vel.make_microduck_velocity_env_cfg()
    actor = cfg.observations["actor"].terms
    assert list(actor) == list(base.observations["actor"].terms)          # same terms, same order
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    critic = cfg.observations["critic"].terms
    for k in ("velocity_ema", "bam_dr", "trunk_com", "foot_friction", "heading_error"):
        assert k in critic and k not in actor, k
    assert critic["velocity_ema"].params["tau_s"] == w2.EMA_TAU_S


def test_penalty_signs():
    cfg = w2.make_microduck_walk_v2_env_cfg()
    for name in PENALTIES:
        assert cfg.rewards[name].weight < 0, name                          # all ≥ 0 costs here
    for name in ("air_time", "track_linear_velocity", "track_vx_ema", "track_angular_velocity",
                 "upright", "pose", "head_pose_tracking"):
        assert cfg.rewards[name].weight > 0, name


def test_tracking_and_gait_terms_wired():
    r = w2.make_microduck_walk_v2_env_cfg().rewards
    at = r["air_time"]
    assert at.func is microduck_mdp.feet_air_time_speed_window
    assert at.params["min_knots"][1] < 0.13 < at.params["max_knots"][1] < 0.25   # 3.5 Hz swings paid, v8's 0.25 s not
    vx = r["track_vx_ema"]
    assert vx.func is microduck_mdp.track_velocity_ema and vx.params["axis"] == "lin_x"
    assert vx.params["tau_s"] == w2.EMA_TAU_S and vx.params["std"] < r["track_linear_velocity"].params["std"]
    yaw = r["track_angular_velocity"]
    assert yaw.func is microduck_mdp.track_velocity_ema and yaw.params["axis"] == "ang_z"
    assert math.isclose(yaw.params["std"], math.sqrt(0.5))
    ls = r["low_swing"]
    assert ls.func is microduck_mdp.feet_low_swing_cost and ls.params["vx_ramp"] == w2.LOW_SWING_VX_RAMP
    p = r["pose"].params
    # reachable from iteration 0 (threshold is |v_xy| + |wz| in command units)
    assert p["walking_threshold"] < p["running_threshold"] <= w2.VX_START_MAX
    for k, std in p["std_running"].items():
        assert std >= p["std_walking"][k], k                               # only ever looser


def test_no_bc_and_mirror_loss_from_iteration_0():
    rl = w2.MicroduckWalkV2RlCfg
    assert type(rl.algorithm).__name__ == "PpoWithSymmetryCfg"            # not PpoWithExpertBc
    assert getattr(rl.algorithm, "bc_cfg", None) is None
    assert rl.algorithm.symmetry_cfg == SYMMETRY_CFG and rl.algorithm.symmetry_cfg["use_mirror_loss"]
    assert "mirror_ramp_iters" not in rl.algorithm.symmetry_cfg
    assert vel.MicroduckRlCfg.algorithm.symmetry_cfg is None               # deepcopy, velocity untouched
    assert rl.max_iterations == 6000


def test_commands():
    cfg = w2.make_microduck_walk_v2_env_cfg()
    tw = cfg.commands["twist"]
    assert tw.ranges.lin_vel_x == (-w2.VX_START_MAX, w2.VX_START_MAX)
    assert tw.ranges.lin_vel_y == (-0.3, 0.3) and tw.ranges.ang_vel_z == (-1.0, 1.0)
    assert tw.rel_heading_envs == w2.REL_HEADING_ENVS and tw.heading_target_rel_range == w2.HEADING_TARGET_REL_RANGE
    assert tw.rel_turn_in_place_envs == vel.TURN_IN_PLACE_FRACTION
    c = cfg.curriculum["vx_range"].params
    assert c["start_max"] == w2.VX_START_MAX < c["final_max"] == w2.VX_FINAL_MAX
    play = w2.make_microduck_walk_v2_env_cfg(play=True).commands["twist"]
    assert play.ranges.lin_vel_x == (-w2.VX_FINAL_MAX, w2.VX_FINAL_MAX)


def _track(err_lin, err_vx, cfg):
    """Weighted per-step pay of the forward-speed tracking terms at a steady error."""
    r = cfg.rewards
    lin = r["track_linear_velocity"]
    vx = r["track_vx_ema"]
    return (lin.weight * math.exp(-err_lin ** 2 / lin.params["std"] ** 2)
            + vx.weight * math.exp(-err_vx ** 2 / vx.params["std"] ** 2))


def _air_time_ceiling(cmd, cfg):
    """Most the air-time term can pay per step in a walk. Duty factor ≥ 0.5 → on average at
    most one foot airborne; a swing of length T inside the window [lo, hi] is paid only for
    its last T − lo seconds, so the paid share is ≤ 1 − lo/hi (at T = hi). Check: v8's slow
    gait, which maximized the old [0.125, 0.30] window, measured 1.50/step at w 3 against
    this formula's 1.75."""
    p = cfg.rewards["air_time"].params
    lo, hi = microduck_mdp.air_time_window(torch.tensor([cmd]), p["speed_knots"], p["min_knots"], p["max_knots"])
    return cfg.rewards["air_time"].weight * (1.0 - (lo / hi).item())


@pytest.mark.parametrize("cmd", [0.25, 0.4, 0.5])
def test_closing_the_speed_gap_outpays_the_whole_air_time_term(cmd):
    """v8's failure: a slow gait collected more from air_time than it lost on tracking.
    Here walking at half the commanded speed forfeits more tracking pay than the air-time
    term can pay at all in a walk."""
    cfg = w2.make_microduck_walk_v2_env_cfg()
    gap = _track(0.0, 0.0, cfg) - _track(cmd / 2, cmd / 2, cfg)
    assert gap > _air_time_ceiling(cmd, cfg)


def test_standing_still_at_a_slow_command_forfeits_tracking():
    cfg = w2.make_microduck_walk_v2_env_cfg()
    assert _track(0.0, 0.0, cfg) - _track(0.1, 0.1, cfg) > 1.5          # v8's 0.1 m/s dead zone


# ── Fake env for the new MDP terms ───────────────────────────────────────────

def _fake_env(cmd, vel_b=None, ang_b=None, tilted=None, standing=None, heading=None,
              air=None, contact=None, step=0):
    n = cmd.shape[0]
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    if tilted is not None:
        s = math.sin(math.radians(60) / 2)
        quat[tilted] = torch.tensor([math.cos(math.radians(60) / 2), s, 0.0, 0.0])
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.12]]).repeat(n, 1),
        root_link_quat_w=quat,
        root_link_lin_vel_b=vel_b if vel_b is not None else torch.zeros(n, 3),
        root_link_ang_vel_b=ang_b if ang_b is not None else torch.zeros(n, 3),
        heading_w=torch.zeros(n),
    )
    robot = SimpleNamespace(data=data, actuators=[])
    term = SimpleNamespace(
        cfg=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=(-0.3, 0.3))),
        command=cmd,
        is_standing_env=standing if standing is not None else torch.zeros(n, dtype=torch.bool),
        is_heading_env=heading if heading is not None else torch.zeros(n, dtype=torch.bool),
        heading_target=torch.zeros(n),
        robot=robot,
    )
    feet = SimpleNamespace(data=SimpleNamespace(
        last_air_time=air if air is not None else torch.zeros(n, 2),
        last_contact_time=contact if contact is not None else torch.zeros(n, 2),
    ))

    class _Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))

    return SimpleNamespace(
        num_envs=n, device="cpu", step_dt=0.02, common_step_counter=step,
        scene=_Scene(robot=robot, feet=feet),
        command_manager=SimpleNamespace(get_term=lambda name: term, get_command=lambda name: cmd),
    )


def _curriculum(env, **kw):
    params = dict(command_name="twist", start_max=0.3, final_max=0.5, step=0.05, edge_frac=0.8,
                  max_shortfall=0.08, update_every_steps=10, min_samples=4)
    params.update(kw)
    cur = microduck_mdp.vx_range_performance_curriculum(SimpleNamespace(params=params), env)
    return cur, params


def _run(cur, env, params, steps):
    out = None
    for _ in range(steps):
        out = cur(env, slice(None), **params)
        env.common_step_counter += 1
    return out


def test_vx_curriculum_widens_when_the_edge_is_tracked_and_caps():
    cmd = torch.tensor([[0.3, 0, 0], [-0.3, 0, 0], [0.1, 0, 0]])
    vel = torch.tensor([[0.27, 0, 0], [-0.28, 0, 0], [0.0, 0, 0]])     # edge tracked; env 2 is below the edge
    env = _fake_env(cmd, vel_b=vel)
    cur, p = _curriculum(env)
    out = _run(cur, env, p, 11)
    assert out["vx_max"] == pytest.approx(0.35) and out["edge_shortfall"] == pytest.approx(0.025, abs=1e-6)
    cmd[:, 0] = torch.tensor([0.5, -0.5, 0.1])                         # stays at the edge as it widens
    vel[:, 0] = torch.tensor([0.48, -0.48, 0.0])
    out = _run(cur, env, p, 200)
    assert out["vx_max"] == pytest.approx(0.5)                          # capped at final_max
    assert env.command_manager.get_term("twist").cfg.ranges.lin_vel_x == pytest.approx((-0.5, 0.5))


def test_vx_curriculum_holds_when_short_and_never_narrows():
    cmd = torch.tensor([[0.3, 0, 0], [0.3, 0, 0]])
    vel = torch.tensor([[0.15, 0, 0], [0.15, 0, 0]])                   # 0.15 m/s short of the edge
    env = _fake_env(cmd, vel_b=vel)
    cur, p = _curriculum(env)
    out = _run(cur, env, p, 50)
    assert out["vx_max"] == pytest.approx(0.3) and out["edge_shortfall"] == pytest.approx(0.15)


def test_vx_curriculum_ignores_fallen_and_standing_envs():
    cmd = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0.3, 0, 0]])
    vel = torch.tensor([[0.29, 0, 0], [0.0, 0, 0], [0.0, 0, 0]])
    env = _fake_env(cmd, vel_b=vel, tilted=torch.tensor([False, True, False]),
                    standing=torch.tensor([False, False, True]))
    cur, p = _curriculum(env, min_samples=1)
    out = _run(cur, env, p, 11)
    assert out["edge_shortfall"] == pytest.approx(0.01, abs=1e-6) and out["vx_max"] == pytest.approx(0.35)


def test_vx_curriculum_needs_enough_samples():
    env = _fake_env(torch.tensor([[0.3, 0, 0]]), vel_b=torch.tensor([[0.3, 0, 0]]))
    cur, p = _curriculum(env, min_samples=10_000)
    out = _run(cur, env, p, 30)
    assert out["vx_max"] == pytest.approx(0.3) and math.isnan(out["edge_shortfall"])


def test_gait_stats_buckets_stride_and_yaw_bias():
    cmd = torch.tensor([[0.4, 0, 0], [0.4, 0, 0], [0.2, 0, 0.0], [0.3, 0, 0]])
    vel = torch.tensor([[0.36, 0, 0], [0.32, 0, 0], [0.1, 0, 0], [0.3, 0, 0]])
    ang = torch.tensor([[0, 0, 0.1], [0, 0, 0.1], [0, 0, 0.0], [0, 0, 0.5]])
    air = torch.tensor([[0.13, 0.13], [0.13, 0.13], [0.25, 0.25], [0.2, 0.1]])
    con = torch.tensor([[0.15, 0.15], [0.15, 0.15], [0.25, 0.25], [0.2, 0.3]])
    env = _fake_env(cmd, vel_b=vel, ang_b=ang, air=air, contact=con,
                    heading=torch.tensor([False, False, False, True]))
    stats = microduck_mdp.walk_v2_gait_stats(SimpleNamespace(params={}), env)
    out = {}
    for _ in range(3):
        out = stats(env, slice(None), command_name="twist", sensor_name="feet", window_steps=2)
        env.common_step_counter += 1
    assert out["vx_ratio_0.4"] == pytest.approx(0.85)                 # (0.9 + 0.8) / 2
    assert out["stride_hz_0.4"] == pytest.approx(1 / 0.28)
    assert out["vx_ratio_0.2"] == pytest.approx(0.5) and out["stride_hz_0.2"] == pytest.approx(2.0)
    assert math.isnan(out["vx_ratio_0.1"])
    assert out["yaw_bias"] == pytest.approx(0.1)                       # heading env 3 excluded; env 2 not > 0.2
    # walking envs 0, 1, 3: L air mean (0.13+0.13+0.2)/3, R (0.13+0.13+0.1)/3
    l, r = 0.46 / 3, 0.36 / 3
    assert out["lr_air_asym"] == pytest.approx(abs(l - r) / (0.5 * (l + r)))


def test_gait_stats_stride_is_robust_to_contact_flicker():
    # 1/mean(period), not mean(1/period): one 0.02 s flicker among 0.36 s strides must not
    # triple the reading (the pilot's mean of reciprocals read 7-11 Hz for a ≈ 2.9 Hz gait).
    cmd = torch.tensor([[0.3, 0, 0]] * 4)
    air = torch.tensor([[0.18, 0.18], [0.18, 0.18], [0.18, 0.18], [0.01, 0.18]])
    con = torch.tensor([[0.18, 0.18], [0.18, 0.18], [0.18, 0.18], [0.01, 0.18]])
    env = _fake_env(cmd, vel_b=torch.tensor([[0.3, 0, 0]] * 4), air=air, contact=con)
    stats = microduck_mdp.walk_v2_gait_stats(SimpleNamespace(params={}), env)
    out = {}
    for _ in range(3):
        out = stats(env, slice(None), command_name="twist", sensor_name="feet", window_steps=2)
        env.common_step_counter += 1
    assert out["stride_hz_0.3"] == pytest.approx(8 / (7 * 0.36 + 0.02))
    assert out["stride_hz_0.3"] < 3.5


# ── Critic obs ───────────────────────────────────────────────────────────────

def test_critic_velocity_ema_equals_the_reward_ema():
    cmd = torch.tensor([[0.4, 0, 0.3], [0.4, 0, 0.3]])
    vel = torch.zeros(2, 3)
    ang = torch.zeros(2, 3)
    env = _fake_env(cmd, vel_b=vel, ang_b=ang)
    obs = microduck_mdp.velocity_ema_obs(SimpleNamespace(params={"tau_s": 0.5}), env)
    rx = microduck_mdp.track_velocity_ema(SimpleNamespace(params={"tau_s": 0.5}), env)
    rz = microduck_mdp.track_velocity_ema(SimpleNamespace(params={"tau_s": 0.5}), env)
    for k in range(60):
        vel[:, 0] = torch.tensor([0.4, 0.2]) + 0.1 * math.sin(k)
        ang[:, 2] = torch.tensor([0.3, -0.2]) + 0.2 * math.cos(k)
        o = obs(env)
        rx(env, std=0.15, command_name="twist", axis="lin_x")
        rz(env, std=0.7, command_name="twist", axis="ang_z")
    assert torch.allclose(o[:, 0], rx.ema) and torch.allclose(o[:, 1], rz.ema)
    obs.reset(torch.tensor([0]))
    assert (obs.ema[0] == 0).all() and (obs.ema[1] != 0).any()


def test_bam_dr_obs_reads_the_actuator_and_defaults_to_nominal():
    env = _fake_env(torch.zeros(2, 3))
    assert torch.allclose(microduck_mdp.bam_dr_params_obs(env), torch.tensor([[7.4, 1.0], [7.4, 1.0]]))
    env.scene["robot"].actuators.append(SimpleNamespace(
        vin_tensor=torch.tensor([[6.6], [8.1]]), friction_scale=torch.tensor([[0.9], [1.1]])))
    assert torch.allclose(microduck_mdp.bam_dr_params_obs(env), torch.tensor([[6.6, 0.9], [8.1, 1.1]]))


def test_heading_error_obs_only_in_heading_envs():
    env = _fake_env(torch.zeros(3, 3), heading=torch.tensor([True, False, True]))
    term = env.command_manager.get_term("twist")
    term.heading_target = torch.tensor([0.3, 0.3, 3.0])
    term.robot.data.heading_w = torch.tensor([0.0, 0.0, -3.0])
    out = microduck_mdp.heading_error_obs(env)
    assert out.shape == (3, 1)
    assert out[0, 0] == pytest.approx(0.3) and out[1, 0] == 0.0
    assert out[2, 0] == pytest.approx(6.0 - 2 * math.pi, abs=1e-5)      # wrapped
