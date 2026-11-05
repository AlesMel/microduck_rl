"""Gait reward terms and heading-hold commands shared by the walking tasks (first written
for VelStandFast, used by Walk v2): the speed-scaled air-time window, EMA velocity tracking,
the low-swing landing cost, and heading targets drawn near the current heading. CPU only."""

import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


# ── Reward functions ─────────────────────────────────────────────────────────

def test_air_time_window_interpolates_and_clamps():
    v = torch.tensor([0.0, 0.1, 0.25, 0.4, 0.8])
    lo, hi = microduck_mdp.air_time_window(v, (0.1, 0.4), (0.125, 0.09), (0.30, 0.15))
    assert torch.allclose(lo, torch.tensor([0.125, 0.125, 0.1075, 0.09, 0.09]))
    assert torch.allclose(hi, torch.tensor([0.30, 0.30, 0.225, 0.15, 0.15]))


def _fake_env(cmd, air=None, vel_b=None, tilted=None):
    n = cmd.shape[0]
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    if tilted is not None:
        s = math.sin(math.radians(60) / 2)
        quat[tilted] = torch.tensor([math.cos(math.radians(60) / 2), s, 0.0, 0.0])
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.12]]).repeat(n, 1),
        root_link_quat_w=quat,
        root_link_lin_vel_b=vel_b if vel_b is not None else torch.zeros(n, 3),
    )
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=air if air is not None else torch.zeros(n, 2)))

    class _Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(n, 3))

    return SimpleNamespace(
        scene=_Scene(robot=SimpleNamespace(data=data), feet=sensor),
        command_manager=SimpleNamespace(get_command=lambda name: cmd),
        extras={"log": {}},
    )


def test_speed_window_pays_short_swings_only_at_speed():
    cmd = torch.tensor([[0.1, 0, 0], [0.4, 0, 0], [0.1, 0, 0], [0.4, 0, 0], [0.0, 0, 0], [-0.4, 0, 0], [0.0, 0.3, 0]])
    air = torch.tensor([
        [0.25, 0.0],   # slow cmd, long swing → paid (velstand's window)
        [0.25, 0.0],   # fast cmd, long swing → not paid
        [0.12, 0.0],   # slow cmd, short swing → not yet paid
        [0.12, 0.0],   # fast cmd, short swing → paid
        [0.25, 0.20],  # standing → never paid
        [0.12, 0.13],  # backward at speed → both feet paid (|vx|)
        [0.25, 0.0],   # lateral only → slow-end window
    ])
    out = microduck_mdp.feet_air_time_speed_window(_fake_env(cmd, air=air), sensor_name="feet", command_name="twist")
    assert out.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0]


def test_speed_window_zero_while_fallen():
    cmd = torch.tensor([[0.4, 0, 0], [0.4, 0, 0]])
    air = torch.tensor([[0.12, 0.12], [0.12, 0.12]])
    env = _fake_env(cmd, air=air, tilted=torch.tensor([False, True]))
    out = microduck_mdp.feet_air_time_speed_window(env, sensor_name="feet", command_name="twist")
    assert out.tolist() == [2.0, 0.0]


def _ema_term(n, std=0.15, tau_s=0.5, **params):
    term_cfg = SimpleNamespace(params={"std": std, "command_name": "twist", "tau_s": tau_s, **params})
    return microduck_mdp.track_velocity_ema(term_cfg, SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02))


def test_velocity_ema_converges_on_the_mean_and_cancels_oscillation():
    cmd = torch.tensor([[0.4, 0, 0], [0.4, 0, 0]])
    term = _ema_term(2, axis="lin_x")
    vel = torch.zeros(2, 3)
    env = _fake_env(cmd, vel_b=vel)
    ema_out, inst_out = [], []
    for k in range(200):  # 4 s at 50 Hz: env 0 holds 0.4; env 1 averages 0.4 with ±0.15 at 2.5 Hz
        vel[0, 0] = 0.4
        vel[1, 0] = 0.4 + 0.15 * math.sin(2 * math.pi * 2.5 * k * 0.02)
        ema_out.append(term(env, std=0.15, command_name="twist", axis="lin_x"))
        inst_out.append(math.exp(-((0.4 - vel[1, 0].item()) ** 2) / 0.15 ** 2))
    last = torch.stack(ema_out[-20:])                                      # the last stride (0.4 s)
    assert last[:, 0].min() > 0.99
    assert last[:, 1].mean() > 0.9 > 0.7 > sum(inst_out[-20:]) / 20       # the EMA pays the mean; instantaneous does not


def test_velocity_ema_yaw_axis_ignores_roll_pitch_and_resets():
    cmd = torch.tensor([[0.0, 0, 0.5], [0.0, 0, 0.5]])
    term = _ema_term(2, std=0.35, axis="ang_z")
    env = _fake_env(cmd)
    ang = torch.tensor([[0.0, 0.0, 0.5], [2.0, 2.0, 0.5]])              # env 1 rocks hard in roll/pitch
    env.scene["robot"].data.root_link_ang_vel_b = ang
    for _ in range(150):
        out = term(env, std=0.35, command_name="twist", axis="ang_z")
    assert torch.allclose(out[0], out[1]) and out[0] > 0.99
    term.reset(env_ids=torch.tensor([1]))
    assert term.ema[1] == 0.0 and term.ema[0] > 0.49
    term.reset(None)
    assert (term.ema == 0).all()


def test_velocity_ema_fallen_gate():
    cmd = torch.tensor([[0.0, 0, 0], [0.0, 0, 0]])
    term = _ema_term(2, axis="lin_x", gate_tilt_above_deg=40.0)
    env = _fake_env(cmd, tilted=torch.tensor([False, True]))
    out = term(env, std=0.15, command_name="twist", axis="lin_x", gate_tilt_above_deg=40.0)
    assert out.tolist() == [1.0, 0.0]


# ── Low-swing landing cost ───────────────────────────────────────────────────

def _swing_env(cmd, contact, heights, tilted=None):
    env = _fake_env(cmd, tilted=tilted)
    env.scene["feet"] = SimpleNamespace(data=SimpleNamespace(found=contact))
    env.scene["scan"] = SimpleNamespace(data=SimpleNamespace(heights=heights))
    return env


def _swing(term, env, contact_seq, height_seq):
    out = []
    for c, h in zip(contact_seq, height_seq):
        env.scene["feet"].data.found = torch.tensor(c)
        env.scene["scan"].data.heights = torch.tensor(h)
        out.append(term(env, sensor_name="feet", height_sensor_name="scan", command_name="twist", min_apex=0.010))
    return torch.stack(out)


def test_low_swing_cost_charges_only_low_swings_at_landing():
    n = 2
    cmd = torch.tensor([[0.4, 0, 0]] * n)
    term = microduck_mdp.feet_low_swing_cost(SimpleNamespace(params={}), SimpleNamespace(num_envs=n, device="cpu"))
    env = _swing_env(cmd, torch.ones(n, 2), torch.zeros(n, 2))
    # left foot swings: env 0 peaks at 25 mm (fine), env 1 at 2 mm (a scuff); right foot stays down
    contact = [[[1, 1], [1, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]
    height = [[[0, 0], [0, 0]], [[0.010, 0], [0.001, 0]], [[0.025, 0], [0.002, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    out = _swing(term, env, contact, height)
    assert out[:, 0].abs().sum() == 0                                   # high swing: free
    assert out[:3, 1].abs().sum() == 0 and math.isclose(out[3, 1].item(), 0.64, rel_tol=1e-5)  # charged once, at landing
    assert out[4, 1] == 0                                               # not charged again
    assert (out >= 0).all()                                              # cost → negative weight


def test_low_swing_cost_free_while_standing_or_fallen():
    cmd = torch.tensor([[0.0, 0, 0], [0.4, 0, 0]])
    term = microduck_mdp.feet_low_swing_cost(SimpleNamespace(params={}), SimpleNamespace(num_envs=2, device="cpu"))
    env = _swing_env(cmd, torch.ones(2, 2), torch.zeros(2, 2), tilted=torch.tensor([False, True]))
    out = _swing(term, env, [[[0, 1]] * 2, [[1, 1]] * 2], [[[0.001, 0]] * 2, [[0, 0]] * 2])
    assert out.abs().sum() == 0                                          # env 0 standing, env 1 fallen
    term.reset(None)
    assert (term.peak == 0).all() and term.prev_contact.all()


def test_low_swing_cost_scales_with_forward_speed_only():
    cmd = torch.tensor([[0.1, 0, 0], [0.225, 0, 0], [0.4, 0, 0], [0.0, 0.3, 0], [0.0, 0, 1.0]])
    n = cmd.shape[0]
    term = microduck_mdp.feet_low_swing_cost(SimpleNamespace(params={}), SimpleNamespace(num_envs=n, device="cpu"))
    env = _swing_env(cmd, torch.ones(n, 2), torch.zeros(n, 2))
    out = []
    for c, h in (([[0, 1]] * n, [[0.0, 0]] * n), ([[1, 1]] * n, [[0, 0]] * n)):   # a zero-clearance scuff, then land
        env.scene["feet"].data.found = torch.tensor(c, dtype=torch.float32)
        env.scene["scan"].data.heights = torch.tensor(h)
        out.append(term(env, sensor_name="feet", height_sensor_name="scan", command_name="twist",
                        min_apex=0.010, vx_ramp=(0.15, 0.30)))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0]))   # slow, sideways and turning: free


# ── Heading-hold commands ────────────────────────────────────────────────────

def _fake_root(heading, stale_heading_w=None):
    """Free-joint root whose qpos holds yaw `heading`; heading_w (xquat-derived) may be stale."""
    n = heading.shape[0]
    qpos = torch.zeros(n, 7)
    qpos[:, 3] = torch.cos(heading / 2)
    qpos[:, 6] = torch.sin(heading / 2)
    data = SimpleNamespace(
        data=SimpleNamespace(qpos=qpos),
        forward_vec_b=torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1),
        heading_w=heading if stale_heading_w is None else stale_heading_w,
    )
    return SimpleNamespace(data=data, indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)))


def _heading_cmd(n, heading, p_heading, p_turn):
    cfg = microduck_mdp.VelocityCommandCommandOnlyCfg(
        entity_name="robot", resampling_time_range=(3.0, 8.0), heading_command=True,
        rel_heading_envs=p_heading, rel_standing_envs=0.0, rel_turn_in_place_envs=p_turn,
        heading_target_rel_range=(-0.5, 0.5),
        ranges=microduck_mdp.VelocityCommandCommandOnlyCfg.Ranges(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)))
    term = microduck_mdp.VelocityCommandCommandOnly.__new__(microduck_mdp.VelocityCommandCommandOnly)
    term._env = SimpleNamespace(num_envs=n, device="cpu")
    term.cfg = cfg
    term.robot = _fake_root(heading)
    term.vel_command_b = torch.zeros(n, 3); term.vel_command_w = torch.zeros(n, 3)
    term.heading_target = torch.zeros(n); term.heading_error = torch.zeros(n)
    for f in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
        setattr(term, f, torch.zeros(n, dtype=torch.bool))
    return term


def test_heading_targets_drawn_near_current_heading():
    n = 2000
    heading = torch.linspace(-3.1, 3.1, n)
    term = _heading_cmd(n, heading, 1.0, 0.0)
    term._resample_command(torch.arange(n))
    from mjlab.utils.lab_api.math import wrap_to_pi
    d = wrap_to_pi(term.heading_target - heading)
    assert d.abs().max() <= 0.5 + 1e-5 and d.std() > 0.2 and term.is_heading_env.all()
    assert (term.heading_target.abs() <= math.pi + 1e-5).all()               # wrapped across ±π


def test_heading_targets_use_the_reset_pose_not_stale_xquat():
    # On an episode reset the command is resampled before sim.forward(): qpos already holds
    # the new random yaw, heading_w still the old one. Targets must follow qpos.
    n = 500
    heading = torch.linspace(-3.0, 3.0, n)
    term = _heading_cmd(n, heading, 1.0, 0.0)
    term.robot = _fake_root(heading, stale_heading_w=torch.zeros(n))
    term._resample_command(torch.arange(n))
    from mjlab.utils.lab_api.math import wrap_to_pi
    assert wrap_to_pi(term.heading_target - heading).abs().max() <= 0.5 + 1e-4


def test_turn_in_place_envs_leave_heading_mode():
    n = 2000
    term = _heading_cmd(n, torch.zeros(n), 1.0, 0.5)
    term._resample_command(torch.arange(n))
    turning = (term.vel_command_b[:, :2] == 0).all(1) & (term.vel_command_b[:, 2].abs() >= 0.4)
    assert 0.3 < turning.float().mean() < 0.7
    assert not term.is_heading_env[turning].any() and term.is_heading_env[~turning].all()
