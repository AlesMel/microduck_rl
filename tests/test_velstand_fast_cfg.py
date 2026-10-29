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
        if name == "track_yaw_fine" and vf.ENABLE_DRIFT_FIX:
            assert t.params["std"] == vf.TRACK_YAW_FINE_EMA_STD < base.rewards[name].params["std"]  # stage 5: tighter
        else:
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
    ours = lambda k: k in ("anchor_vx_fade", "bc_mirror", "bc_mirror_upright_tilt_deg") or k.startswith("vx_teacher_")  # noqa: E731
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
    if vf.ENABLE_HEADING_ENVS:  # stage 6 holds the teacher at stage 4's end value instead of re-ramping from 1.0
        assert bc["vx_teacher_coef_ramp"] is None and bc["vx_teacher_coef"] == vf.VX_TEACHER_COEF_STAGE6 == vf.VX_TEACHER_COEF_RAMP[1]
    elif vf.ENABLE_DRIFT_FIX:   # stage 5 likewise
        assert bc["vx_teacher_coef_ramp"] is None and bc["vx_teacher_coef"] == vf.VX_TEACHER_COEF_STAGE5 == vf.VX_TEACHER_COEF_RAMP[1]
    else:
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



def test_drift_fix_tightens_only_the_fine_yaw_term():
    cfg = vf.make_microduck_velstand_fast_env_cfg()
    fine, coarse = cfg.rewards["track_yaw_fine"], cfg.rewards["track_angular_velocity"]
    assert fine.params["std"] < coarse.params["std"]
    assert fine.params["axis"] == coarse.params["axis"] == "ang_z"            # both on the EMA, roll/pitch excluded


# ── Heading hold (stage 6) ───────────────────────────────────────────────────

def test_remap_vx_wz_offset_by_fed_speed_and_direction():
    from mjlab_microduck.tasks.distill import remap_vx
    obs = torch.zeros(5, 61)
    obs[:, 48] = torch.tensor([0.0, 0.2, 0.4, -0.2, 0.6])
    obs[:, 50] = 0.3
    table = {"targets": [0.2, 0.4], "fed": [0.3, 0.6],
             "wz_offset": {"fed": [0.3, 0.6], "fwd": [0.1, 0.2], "back": [-0.1, 0.4]}}
    out = remap_vx(obs, (48, 51), table)
    assert torch.allclose(out[:, 48], torch.tensor([0.0, 0.3, 0.6, -0.3, 0.6]))
    # zero vx: no offset; fwd/back lists by sign; clamped past the last knot (fed 0.6 at vx 0.6)
    assert torch.allclose(out[:, 50], torch.tensor([0.3, 0.4, 0.5, 0.2, 0.5]))
    plain = remap_vx(obs, (48, 51), {"targets": [0.2, 0.4], "fed": [0.3, 0.6]})
    assert (plain[:, 50] == 0.3).all()                                    # no table → yaw untouched


def test_heading_envs_wired():
    for play in (False, True):
        tw = vf.make_microduck_velstand_fast_env_cfg(play=play).commands["twist"]
        if not vf.ENABLE_HEADING_ENVS:
            assert tw.rel_heading_envs == 0.0
            continue
        assert isinstance(tw, microduck_mdp.VelocityCommandCommandOnlyCfg) and tw.heading_command
        assert tw.rel_heading_envs == vf.REL_HEADING_ENVS > 0
        assert tw.heading_control_stiffness == vf.HEADING_CONTROL_STIFFNESS
        lo, hi = tw.heading_target_rel_range
        assert lo < 0 < hi <= 1.0                                            # small corrections, both ways
        assert tw.rel_turn_in_place_envs > 0                                 # turn-in-place bucket kept
    vel = vs.make_microduck_velstand_env_cfg().commands["twist"]
    assert vel.rel_heading_envs == 0.0 and vel.heading_target_rel_range is None   # velstand untouched
    bc = vf.MicroduckVelStandFastRlCfg.algorithm.bc_cfg
    if vf.ENABLE_HEADING_ENVS:
        off = bc["vx_teacher_remap"]["wz_offset"]
        assert len(off["fed"]) == len(off["fwd"]) == len(off["back"]) and off["fed"] == sorted(off["fed"])
        assert off["fwd"][-1] > 0 and off["back"][-1] > 0                     # the teacher turns right at speed


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


# ── Symmetric gait (stage 7) ─────────────────────────────────────────────────

def test_mirror_bc_samples_doubles_and_mirrors():
    from mjlab_microduck.tasks.distill import mirror_bc_samples
    from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(5, 61, generator=g)
    td = TensorDict({"actor": obs, "critic": torch.randn(5, 76, generator=g)}, batch_size=[5])
    act = torch.randn(5, 14, generator=g)
    w = torch.rand(5, generator=g)
    td2, act2, w2 = mirror_bc_samples(td, act, w, ["actor"])
    ref_td, ref_act = microduck_vel_symmetry(None, td, act)               # same tables as the mirror loss
    assert torch.equal(td2["actor"], ref_td["actor"]) and torch.equal(act2, ref_act)
    assert torch.equal(w2, torch.cat([w, w])) and torch.equal(td2["critic"][5:], td["critic"])
    back, back_act, _ = mirror_bc_samples(td2[5:], act2[5:], w, ["actor"])
    assert torch.allclose(back["actor"][5:], obs) and torch.allclose(back_act[5:], act)   # mirror ∘ mirror = id
    assert torch.equal(td["actor"], obs)                                                  # input untouched
    m = td2["actor"][5:]
    assert torch.equal(m[:, 48], obs[:, 48]) and torch.equal(m[:, 50], -obs[:, 50])       # vx kept, yaw command flipped


def test_symmetry_wired():
    alg = vf.MicroduckVelStandFastRlCfg.algorithm
    if not vf.ENABLE_SYMMETRY:
        assert alg.symmetry_cfg is None
        return
    assert alg.symmetry_cfg["use_mirror_loss"] and alg.symmetry_cfg["mirror_loss_coeff"] > 0
    assert "microduck_vel_symmetry" in alg.symmetry_cfg["data_augmentation_func"]
    assert alg.bc_cfg["bc_mirror"] is True
    if vf.ENABLE_MIRROR_UPRIGHT_ONLY:   # stage 7c: upright frames only, held coefficient, no ramp
        assert alg.symmetry_cfg["data_augmentation_func"].endswith("microduck_vel_symmetry_upright")
        assert alg.symmetry_cfg["mirror_loss_coeff"] == vf.MIRROR_UPRIGHT_COEFF and "mirror_ramp_iters" not in alg.symmetry_cfg
        assert alg.bc_cfg["bc_mirror_upright_tilt_deg"] == alg.bc_cfg["anchor_tilt_deg"]   # same upright gate
    else:
        assert alg.symmetry_cfg["mirror_ramp_iters"] == vf.MIRROR_RAMP_ITERS > 0
    assert vs.MicroduckVelStandRlCfg.algorithm.symmetry_cfg is None                      # velstand untouched
    assert not vs.MicroduckVelStandRlCfg.algorithm.bc_cfg.get("bc_mirror")


def test_mirror_ramp_schedule_and_bc_weight():
    from mjlab_microduck.tasks.distill import mirror_bc_samples
    algo = PpoWithExpertBc.__new__(PpoWithExpertBc)
    algo.symmetry = None
    assert algo._mirror_frac() == 1.0                                     # no symmetry → no ramp
    algo.symmetry = {"mirror_loss_coeff": 0.5}
    assert algo._mirror_frac() == 1.0                                     # no ramp key → full
    algo.symmetry["mirror_ramp_iters"] = 4
    fr = []
    for k in range(6):
        algo._ppo_updates = k
        fr.append(algo._mirror_frac())
    assert fr == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    td = TensorDict({"actor": torch.randn(3, 61)}, batch_size=[3])
    _, _, w = mirror_bc_samples(td, torch.randn(3, 14), torch.tensor([1.0, 2.0, 3.0]), ["actor"], 0.25)
    assert torch.allclose(w, torch.tensor([1.0, 2.0, 3.0, 0.25, 0.5, 0.75]))


def _tilted_obs(n_up, n_tilt, g):
    obs = torch.randn(n_up + n_tilt, 61, generator=g)
    obs[:, 3:6] = torch.tensor([0.0, 0.0, -1.0])
    obs[n_up:, 3:6] = torch.tensor([0.0, -0.7, -0.7])                  # 45° roll
    return obs


def test_upright_symmetry_leaves_tilted_rows_unmirrored():
    from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry, microduck_vel_symmetry_upright
    g = torch.Generator().manual_seed(1)
    obs = _tilted_obs(3, 2, g)
    td = TensorDict({"actor": obs, "critic": torch.randn(5, 76, generator=g)}, batch_size=[5])
    full, _ = microduck_vel_symmetry(None, td, None)
    aug, _ = microduck_vel_symmetry_upright(None, td, None)
    assert torch.equal(aug["actor"][:5], obs)                              # original half untouched
    assert torch.equal(aug["actor"][5:8], full["actor"][5:8])              # upright rows mirrored
    assert torch.equal(aug["actor"][8:], obs[3:])                          # tilted rows: identity
    act = torch.randn(5, 14, generator=g)
    _, full_a = microduck_vel_symmetry(None, None, act)
    _, aug_a = microduck_vel_symmetry_upright(None, None, act)              # reuses the obs call's mask
    assert torch.equal(aug_a[5:8], full_a[5:8]) and torch.equal(aug_a[8:], act[3:])
    with pytest.raises(AssertionError):
        microduck_vel_symmetry_upright(None, None, torch.randn(4, 14))      # batch mismatch caught


def test_mirror_bc_samples_upright_only():
    from mjlab_microduck.tasks.distill import mirror_bc_samples
    g = torch.Generator().manual_seed(2)
    obs = _tilted_obs(3, 2, g)
    td = TensorDict({"actor": obs}, batch_size=[5])
    tgt = torch.randn(5, 14, generator=g)
    td2, t2, w2 = mirror_bc_samples(td, tgt, torch.ones(5), ["actor"], 1.0, 25.0)
    assert td2.shape[0] == 8 and t2.shape[0] == 8 and w2.shape[0] == 8      # 5 originals + 3 upright mirrors
    full, ft, _ = mirror_bc_samples(td[:3], tgt[:3], torch.ones(3), ["actor"])
    assert torch.equal(td2["actor"][5:], full["actor"][3:]) and torch.equal(t2[5:], ft[3:])
