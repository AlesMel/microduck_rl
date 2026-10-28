"""Cfg invariants for VelStandFast (velstand + speed-dependent air time, vx fine tracking,
walk anchor faded out with forward speed). CPU only, never steps an env."""
import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.robot.microduck_constants import get_allcollisions_backlash_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs
from mjlab_microduck.tasks import microduck_velstand_fast_env_cfg as vf
from mjlab_microduck.tasks.distill import PpoWithExpertBc, anchor_vx_fade_weight

TASKS = (
    "Mjlab-VelStandFast-Flat-MicroDuck",
    "Mjlab-VelStandFast-Rough-MicroDuck",
    "Mjlab-VelStandFast-Flat-Backlash-MicroDuck",
    "Mjlab-VelStandFast-Rough-Backlash-MicroDuck",
)


# ── Cfg ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_tasks_registered(task):
    assert load_env_cfg(task) is not None and load_env_cfg(task, play=True) is not None
    assert load_rl_cfg(task).experiment_name == "velstand_fast"


def test_backlash_variant_uses_velstand_model():
    cfg = load_env_cfg("Mjlab-VelStandFast-Rough-Backlash-MicroDuck")
    assert cfg.scene.entities["robot"].spec_fn is get_allcollisions_backlash_spec


@pytest.mark.parametrize("play", [False, True])
def test_only_the_speed_terms_differ_from_velstand(play):
    base = vs.make_microduck_velstand_env_cfg(play=play)
    cfg = vf.make_microduck_velstand_fast_env_cfg(play=play)
    assert set(cfg.rewards) == set(base.rewards) | {"track_vx_fine"} | ({"low_swing"} if vf.ENABLE_LOW_SWING_COST else set())
    changed = {"air_time"}
    if vf.ENABLE_EMA_TRACKING:
        changed |= set(vf.TRACK_YAW_EMA_STDS)
    for name in base.rewards:
        if name in changed:
            continue
        assert cfg.rewards[name].func is base.rewards[name].func, name
        assert cfg.rewards[name].weight == base.rewards[name].weight, name
        if name != "pose" or not vf.ENABLE_POSE_RUNNING_TIER:
            assert cfg.rewards[name].params == base.rewards[name].params, name
    assert set(cfg.curriculum) == set(base.curriculum)        # no new curricula: plain resume from v8
    assert cfg.commands["twist"].ranges == base.commands["twist"].ranges
    assert cfg.observations["actor"].terms.keys() == base.observations["actor"].terms.keys()


def test_air_time_term_wired():
    base = vs.make_microduck_velstand_env_cfg()
    at = vf.make_microduck_velstand_fast_env_cfg().rewards["air_time"]
    assert at.func is microduck_mdp.feet_air_time_speed_window
    assert at.weight == base.rewards["air_time"].weight > 0
    p = at.params
    assert p["sensor_name"] == base.rewards["air_time"].params["sensor_name"]
    assert p["gate_tilt_above_deg"] == vs.REWARD_GATE_TILT_DEG
    # slow end = velstand's window, so standing, slow walking and turning are unchanged
    assert p["min_knots"][0] == base.rewards["air_time"].params["threshold_min"]
    assert p["max_knots"][0] == base.rewards["air_time"].params["threshold_max"]
    # fast end: a ~0.13 s swing is inside, velstand's measured 0.25 s swing is not
    assert p["min_knots"][1] < 0.13 < p["max_knots"][1] < 0.25


def test_vx_fine_term_wired():
    cfg = vf.make_microduck_velstand_fast_env_cfg()
    base = vs.make_microduck_velstand_env_cfg()
    f = cfg.rewards["track_vx_fine"]
    if vf.ENABLE_EMA_TRACKING:
        assert f.func is microduck_mdp.track_velocity_ema and f.weight == vf.TRACK_VX_EMA_WEIGHT > vf.TRACK_VX_FINE_WEIGHT
        assert f.params["axis"] == "lin_x" and f.params["tau_s"] == vf.EMA_TAU_S
        assert f.params["gate_tilt_above_deg"] == vs.REWARD_GATE_TILT_DEG
    else:
        assert f.func is microduck_mdp.track_lin_vel_x_fine and f.weight > 0
    assert f.params["std"] < base.rewards["track_linear_velocity"].params["std"]
    assert cfg.rewards["track_linear_velocity"].params == base.rewards["track_linear_velocity"].params


def test_yaw_terms_on_ema():
    if not vf.ENABLE_EMA_TRACKING:
        return
    cfg = vf.make_microduck_velstand_fast_env_cfg()
    base = vs.make_microduck_velstand_env_cfg()
    for name in ("track_angular_velocity", "track_yaw_fine"):
        t = cfg.rewards[name]
        assert t.func is microduck_mdp.track_velocity_ema and t.params["axis"] == "ang_z", name
        assert t.weight == base.rewards[name].weight > 0, name          # same weight as the term it replaces
        assert math.isclose(t.params["std"], base.rewards[name].params["std"]), name
        assert t.params.get("gate_tilt_above_deg") is None, name         # ungated, like before (recovery untouched)


def test_pose_running_tier():
    pose = vf.make_microduck_velstand_fast_env_cfg().rewards["pose"]
    base = vs.make_microduck_velstand_env_cfg().rewards["pose"]
    if not vf.ENABLE_POSE_RUNNING_TIER:
        assert pose.params == base.params
        return
    p = pose.params
    # reachable: the twist command tops out at 0.4 m/s; walking tier stays below the threshold
    assert p["walking_threshold"] < p["running_threshold"] < max(vs.make_microduck_velstand_env_cfg().commands["twist"].ranges.lin_vel_x)
    assert set(p["std_running"]) == set(p["std_walking"])
    for k, std in p["std_running"].items():
        assert std >= p["std_walking"][k], k                              # only ever looser than walking
    assert p["std_standing"] == base.params["std_standing"] and p["std_walking"] == base.params["std_walking"]


def test_anchor_fade_only_in_fast_runner_cfg():
    fast = vf.MicroduckVelStandFastRlCfg.algorithm.bc_cfg
    assert fast["anchor_vx_fade"] == vf.ANCHOR_VX_FADE
    assert vs.MicroduckVelStandRlCfg.algorithm.bc_cfg.get("anchor_vx_fade") is None  # deepcopy, velstand untouched
    lo, hi = vf.ANCHOR_VX_FADE
    assert vf.AIR_SPEED_KNOTS[0] <= lo < hi < vf.AIR_SPEED_KNOTS[1]
    # everything else about the BC is velstand's
    ours = lambda k: k == "anchor_vx_fade" or k.startswith("vx_teacher_")  # noqa: E731
    assert {k: v for k, v in fast.items() if not ours(k)} == \
           {k: v for k, v in vs.MicroduckVelStandRlCfg.algorithm.bc_cfg.items() if not ours(k)}
    assert vf.MicroduckVelStandFastRlCfg.experiment_name != vs.MicroduckVelStandRlCfg.experiment_name


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


def test_vx_fine_ignores_sway_and_prices_speed_error():
    cmd = torch.tensor([[0.4, 0, 0], [0.4, 0, 0], [0.4, 0, 0], [0.4, 0, 0]])
    vel = torch.tensor([[0.4, 0.0, 0], [0.4, 0.15, 0], [0.18, 0.0, 0], [0.4, 0.0, 0]])
    env = _fake_env(cmd, vel_b=vel, tilted=torch.tensor([False, False, False, True]))
    out = microduck_mdp.track_lin_vel_x_fine(env, std=0.15, command_name="twist")
    assert out[0] == 1.0 and out[1] == 1.0                    # lateral sway is free
    assert math.isclose(out[2].item(), math.exp(-(0.22 / 0.15) ** 2), rel_tol=1e-5)
    assert out[3] == 0.0                                      # fallen


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


# ── BC anchor fade ───────────────────────────────────────────────────────────

def test_anchor_vx_fade_weight():
    obs = torch.zeros(5, 61)
    obs[:, 48] = torch.tensor([0.0, 0.1, 0.15, -0.2, 0.4])
    w = anchor_vx_fade_weight(obs, (48, 51), (0.1, 0.2))
    assert torch.allclose(w, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))


class _Actor(torch.nn.Module):
    obs_groups = ["actor"]

    def __init__(self, bias: float = 0.0, trainable: bool = True):
        super().__init__()
        self.lin = torch.nn.Linear(61, 14)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.constant_(self.lin.bias, bias)
        self.lin.requires_grad_(trainable)

    def forward(self, td):
        return self.lin(td["actor"])


def _bc_algo(vx: torch.Tensor):
    """PpoWithExpertBc with only the BC pieces: student outputs 0, anchor outputs 1."""
    n = vx.numel()
    obs = torch.zeros(n, 1, 61)
    obs[:, 0, 5] = -1.0            # projected gravity: upright
    obs[:, 0, 48] = vx
    algo = PpoWithExpertBc.__new__(PpoWithExpertBc)
    algo.actor = _Actor()
    algo.anchor = _Actor(bias=1.0, trainable=False)
    algo.expert = _Actor(bias=-1.0, trainable=False)
    algo.vx_teacher = None
    algo.device = "cpu"
    algo.max_grad_norm = 1.0
    algo.storage = SimpleNamespace(observations=TensorDict({"actor": obs}, batch_size=[n, 1]),
                                   actions=torch.zeros(n, 1, 14))
    algo.bc_optimizer = torch.optim.SGD(algo.actor.parameters(), lr=0.0)  # measure the loss, change nothing
    algo.bc_cfg = {**vf.MicroduckVelStandFastRlCfg.algorithm.bc_cfg, "anchor_yaw_remap": None,
                   "body_slice": None, "min_mini_batch": 1, "epochs": 1, "mini_batches": 1}
    return algo


def test_bc_update_skips_fast_frames_and_fades_mid_frames():
    lo, hi = vf.ANCHOR_VX_FADE
    s = _bc_algo(torch.tensor([0.0, lo, 0.5 * (lo + hi), hi, 0.4]))._bc_update()
    # anchored: 0, lo (w 1) and the midpoint (w 0.5); free: hi and 0.4
    assert math.isclose(s["expert_bc_anchor_frac"], 3 / 5, rel_tol=1e-6)
    assert math.isclose(s["expert_bc_vx_free_frac"], 2 / 5, rel_tol=1e-6)
    assert math.isclose(s["expert_bc"], (1.0 + 1.0 + 0.5) / 3, rel_tol=1e-5)  # MSE 1 per frame × weight
    s = _bc_algo(torch.tensor([0.3, -0.4]))._bc_update()
    assert s["expert_bc"] == 0.0 and s["expert_bc_anchor_frac"] == 0.0


# ── BC vx teacher (stage 3) ──────────────────────────────────────────────────

def test_remap_vx_table_lookup():
    from mjlab_microduck.tasks.distill import remap_vx
    obs = torch.zeros(6, 61)
    obs[:, 48] = torch.tensor([0.0, 0.1, 0.3, -0.4, 0.6, 0.2])
    obs[:, 49] = 0.1; obs[:, 50] = -0.5; obs[:, 0] = 7.0
    out = remap_vx(obs, (48, 51), {"targets": [0.2, 0.4], "fed": [0.3, 0.6]})
    assert torch.allclose(out[:, 48], torch.tensor([0.0, 0.15, 0.45, -0.6, 0.6, 0.3]))  # 0→0, linear, sign kept, clamped
    assert (out[:, 49:] == obs[:, 49:]).all() and (out[:, :48] == obs[:, :48]).all()  # only vx changes
    assert obs[1, 48] == 0.1                                                         # input untouched


class _EchoVx(torch.nn.Module):
    """Teacher whose every action equals the vx command it is fed."""
    obs_groups = ["actor"]

    def forward(self, td):
        return td["actor"][:, 48:49].expand(-1, 14)


def test_bc_update_vx_teacher_takes_the_faded_share():
    lo, hi = vf.ANCHOR_VX_FADE
    mid = 0.5 * (lo + hi)
    algo = _bc_algo(torch.tensor([0.0, mid, 0.3]))
    algo.vx_teacher = _EchoVx()
    algo.bc_cfg = {**algo.bc_cfg, "vx_teacher_coef": 1.0, "vx_teacher_remap": {"targets": [0.4], "fed": [0.8]}}  # fed = 2·vx
    s = algo._bc_update()
    # anchor (target 1): vx 0 at w 1, mid at w 0.5; teacher (target 2·vx): mid at w 0.5, 0.3 at w 1
    expected = (1.0 + 0.5 * 1.0 + 0.5 * (2 * mid) ** 2 + 1.0 * 0.6 ** 2) / 4
    assert math.isclose(s["expert_bc"], expected, rel_tol=1e-5)
    assert math.isclose(s["expert_bc_vx_teacher_frac"], 2 / 3, rel_tol=1e-6)


def test_bc_update_vx_teacher_skips_fallen_frames():
    algo = _bc_algo(torch.tensor([0.3, 0.3]))
    algo.storage.observations["actor"][1, 0, 3:6] = torch.tensor([0.0, -1.0, 0.0])  # lying on its side
    algo.vx_teacher = _EchoVx()
    algo.bc_cfg = {**algo.bc_cfg, "vx_teacher_coef": 1.0, "vx_teacher_remap": {"targets": [0.4], "fed": [0.8]}, "min_samples": 10}
    s = algo._bc_update()
    assert math.isclose(s["expert_bc_vx_teacher_frac"], 1 / 2, rel_tol=1e-6)
    assert math.isclose(s["expert_bc"], 0.6 ** 2, rel_tol=1e-5)


def test_vx_teacher_wired():
    bc = vf.MicroduckVelStandFastRlCfg.algorithm.bc_cfg
    if not vf.ENABLE_VX_TEACHER:
        assert not bc.get("vx_teacher_coef")
        return
    assert bc["vx_teacher_coef"] > 0 and bc["anchor_vx_fade"] == vf.ANCHOR_VX_FADE
    assert bc["vx_teacher_wandb_run_path"] and bc["vx_teacher_checkpoint_name"].endswith(".pt")
    t = bc["vx_teacher_remap"]
    assert len(t["targets"]) == len(t["fed"])
    assert list(t["targets"]) == sorted(t["targets"]) and list(t["fed"]) == sorted(t["fed"])  # monotone
    assert all(f > x for x, f in zip(t["targets"], t["fed"]))       # the teacher undershoots → always fed more
    assert t["targets"][0] <= vf.ANCHOR_VX_FADE[0]                  # covers the whole faded band
    lin_x = vf.make_microduck_velstand_fast_env_cfg().commands["twist"].ranges.lin_vel_x
    assert t["targets"][-1] >= max(abs(v) for v in lin_x)           # and the whole trained command range
    assert vs.MicroduckVelStandRlCfg.algorithm.bc_cfg.get("vx_teacher_coef", 0.0) == 0.0  # velstand untouched


# ── Stage 4: low-swing landing cost + vx-teacher ramp ────────────────────────

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


def test_low_swing_wired():
    cfg = vf.make_microduck_velstand_fast_env_cfg()
    if not vf.ENABLE_LOW_SWING_COST:
        assert "low_swing" not in cfg.rewards
        return
    t = cfg.rewards["low_swing"]
    assert t.func is microduck_mdp.feet_low_swing_cost and t.weight < 0      # ≥ 0 cost → negative weight
    assert 0 < t.params["min_apex"] < cfg.rewards["foot_swing_height"].params["target_height"]
    assert t.params["gate_tilt_above_deg"] == vs.REWARD_GATE_TILT_DEG
    bc = vf.MicroduckVelStandFastRlCfg.algorithm.bc_cfg
    start, end, iters = bc["vx_teacher_coef_ramp"]
    assert start == bc["vx_teacher_coef"] > end > 0 and iters > 0


def test_vx_teacher_coef_ramp():
    algo = _bc_algo(torch.tensor([0.3]))
    algo.vx_teacher = _EchoVx()
    algo.bc_cfg = {**algo.bc_cfg, "vx_teacher_coef": 1.0, "vx_teacher_remap": {"targets": [0.4], "fed": [0.8]},
                   "vx_teacher_coef_ramp": (1.0, 0.25, 4)}
    coefs = [algo._bc_update()["expert_bc_vx_teacher_coef"] for _ in range(6)]
    assert [round(c, 4) for c in coefs] == [1.0, 0.8125, 0.625, 0.4375, 0.25, 0.25]
    algo.bc_cfg["vx_teacher_coef_ramp"] = None
    assert algo._vx_teacher_coef() == 1.0


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


def test_low_swing_uses_the_vx_ramp():
    if vf.ENABLE_LOW_SWING_COST:
        lo, hi = vf.make_microduck_velstand_fast_env_cfg().rewards["low_swing"].params["vx_ramp"]
        assert 0.1 < lo < hi <= 0.4
