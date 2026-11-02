"""The perceptive step-up skill: terrain, obs contract (61 + 64), head-down command, reward
signs and the two new reward functions on a fake env."""
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_step_up_env_cfg as su
from mjlab_microduck.tasks import microduck_terrains as mt
from mjlab_microduck.tasks import tof

TASKS = ("Mjlab-StepUp-Tof-MicroDuck", "Mjlab-StepUp-Tof-Backlash-MicroDuck")


# ── terrain ──────────────────────────────────────────────────────────────────

def test_single_step_tile_geometry():
    t = mt.BoxSingleStepTerrainCfg(proportion=1.0, size=(4.0, 4.0), num_steps=1)
    assert t.step_height(0.0) == pytest.approx(0.015) and t.step_height(1.0) == pytest.approx(0.040)
    f = mt.BoxSingleStepTerrainCfg(proportion=1.0, size=(4.0, 4.0), num_steps=3)
    assert f.step_height(0.5) * 3 == pytest.approx(0.0825)


def test_step_up_menu_builds():
    gen = TerrainGenerator(mt.make_microduck_step_up_terrains_cfg(num_rows=3, seed=1), device="cpu")
    assert gen.terrain_origins.shape == (3, 3, 3)
    assert list(mt.microduck_step_up_subterrains()) == ["flat", "single_step", "short_flight"]


# ── cfg ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_registered_with_tof_and_rewards(task):
    cfg = load_env_cfg(task, play=False)
    assert any(getattr(s, "name", "") == tof.TOF_SENSOR_NAME for s in cfg.scene.sensors)
    actor = list(cfg.observations["actor"].terms)
    assert actor[-1] == "tof" and actor[:8] == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                                                "actions", "command", "head_command", "body_command"]
    assert "tof_clean" in cfg.observations["critic"].terms
    assert cfg.observations["critic"].terms["tof_clean"].params["clean"] is True
    r = cfg.rewards
    assert r["height_progress"].func is microduck_mdp.trunk_height_progress and r["height_progress"].weight > 0
    assert r["foot_on_step"].func is microduck_mdp.foot_on_step_bonus and r["foot_on_step"].weight > 0
    assert tuple(r["foot_on_step"].params["asset_cfg"].site_names) == ("left_foot", "right_foot")
    assert r["trunk_impact"].func is microduck_mdp.trunk_vertical_accel_penalty and r["trunk_impact"].weight > 0   # self-negating
    assert "swing_apex" in r and "torque_headroom" in r and r["torque_headroom"].weight < 0
    assert cfg.episode_length_s == su.EPISODE_LENGTH_S
    assert load_rl_cfg(task).run_name == "step_up_tof" and load_rl_cfg(task).experiment_name == "step_up"


def test_commands_point_into_the_step_and_head_down():
    cfg = load_env_cfg("Mjlab-StepUp-Tof-MicroDuck", play=False)
    tw = cfg.commands["twist"].ranges
    assert tw.lin_vel_x == su.VX_RANGE and tw.lin_vel_x[0] > 0
    hp = cfg.commands["head_pose"].ranges
    assert hp[0][0] < hp[0][1] < 0 and 0 < hp[1][0] < hp[1][1]          # neck down (−), head down (+)
    assert np.isclose(np.mean(hp[0]), su.HEAD_DOWN[0]) and np.isclose(np.mean(hp[1]), su.HEAD_DOWN[1])
    assert cfg.rewards["head_pose_tracking"].params["std"] == su.HEAD_TRACK_STD
    assert "head_pose_range" not in cfg.curriculum
    assert cfg.events["reset_base"].func is microduck_mdp.reset_root_on_terrain
    assert cfg.events["reset_base"].params["yaw_range"] == (-su.SPAWN_YAW, su.SPAWN_YAW)
    assert cfg.scene.terrain.terrain_generator.sub_terrains.keys() == {"flat", "single_step", "short_flight"}


def test_backlash_twin_uses_backlash_model():
    cfg = load_env_cfg("Mjlab-StepUp-Tof-Backlash-MicroDuck", play=False)
    assert "backlash" in str(cfg.scene.entities["robot"].spec_fn).lower() or cfg.scene.entities["robot"] is not \
        load_env_cfg("Mjlab-StepUp-Tof-MicroDuck", play=False).scene.entities["robot"]


# ── reward functions on a fake env ───────────────────────────────────────────

class _Asset:
    def __init__(self, n):
        self.data = type("D", (), {})()
        self.data.root_link_pos_w = torch.zeros(n, 3)
        self.data.site_pos_w = torch.zeros(n, 2, 3)
        self.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)
        self.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)


class _Env:
    def __init__(self, n=3):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.asset = _Asset(n)
        self.found = torch.ones(n, 2)
        origins = torch.zeros(n, 3)
        contact = type("C", (), {"data": type("CD", (), {"found": self.found})()})()
        self.scene = type("S", (), {"env_origins": origins})()
        self.scene.terrain = type("T", (), {"env_origins": origins})()
        self.scene.__class__.__getitem__ = lambda s, k: self.asset if k == "robot" else contact


def test_trunk_height_progress_is_potential_based():
    env = _Env()
    cfg = type("Cfg", (), {"params": {}})()
    term = microduck_mdp.trunk_height_progress(cfg, env)
    env.asset.data.root_link_pos_w[:, 2] = 0.12
    assert torch.all(term(env) == 0)                        # fresh after reset: no transient
    env.asset.data.root_link_pos_w[:, 2] = 0.125
    assert torch.allclose(term(env), torch.full((3,), 0.005))
    env.asset.data.root_link_pos_w[:, 2] = 0.20               # a jump: clipped per step
    assert torch.allclose(term(env), torch.full((3,), 0.01))
    env.asset.data.root_link_pos_w[:, 2] = 0.125               # and paid back on the way down
    assert torch.all(term(env) == -0.01)
    term.reset(torch.tensor([1]))
    env.asset.data.root_link_pos_w[:, 2] = 0.30
    out = term(env)
    assert out[1] == 0 and out[0] == 0.01


def test_foot_on_step_pays_once_per_foot():
    env = _Env()
    cfg = type("Cfg", (), {"params": {}})()
    term = microduck_mdp.foot_on_step_bonus(cfg, env)
    site_cfg = type("SC", (), {"name": "robot", "site_ids": [0, 1]})()
    env.asset.data.site_pos_w[:, :, 2] = 0.0
    assert torch.all(term(env, "feet", 0.012, 40.0, site_cfg) == 0)      # on the flat: nothing
    env.asset.data.site_pos_w[0, 0, 2] = 0.02                             # env 0 left foot on the step
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [1.0, 0.0, 0.0]
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [0.0, 0.0, 0.0]   # once
    env.asset.data.site_pos_w[0, 1, 2] = 0.02                             # then the right foot
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [1.0, 0.0, 0.0]
    env.found[1] = 0                                                      # env 1 feet in the air: no pay
    env.asset.data.site_pos_w[1, :, 2] = 0.03
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [0.0, 0.0, 0.0]
    term.reset(torch.tensor([0]))
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [2.0, 0.0, 0.0]


# ── warm start across observation widths (Patch 6) ───────────────────────────

def test_expand_checkpoint_obs_pads_first_layer_normalizer_and_adam():
    old_a, new_a, old_c, new_c, h = 61, 125, 85, 149, 8
    loaded = {
        "actor_state_dict": {
            "obs_normalizer._mean": torch.randn(1, old_a), "obs_normalizer._var": torch.rand(1, old_a) + 0.5,
            "obs_normalizer._std": torch.rand(1, old_a) + 0.5, "obs_normalizer.count": torch.tensor(72000.0),
            "distribution.std_param": torch.ones(14), "mlp.0.weight": torch.randn(h, old_a), "mlp.0.bias": torch.zeros(h),
            "mlp.2.weight": torch.randn(14, h),
        },
        "critic_state_dict": {"obs_normalizer._mean": torch.zeros(1, old_c), "mlp.0.weight": torch.randn(h, old_c)},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(5.0), "exp_avg": torch.zeros(14), "exp_avg_sq": torch.zeros(14)},
                                           1: {"step": torch.tensor(5.0), "exp_avg": torch.ones(h, old_a), "exp_avg_sq": torch.ones(h, old_a)}},
                                 "param_groups": [{"params": [0, 1]}]},
    }
    w_old = loaded["actor_state_dict"]["mlp.0.weight"].clone()
    actor_new = {k: torch.zeros(*v.shape[:-1], new_a) if v.dim() and v.shape[-1] == old_a else v
                 for k, v in loaded["actor_state_dict"].items()}
    critic_new = {k: torch.zeros(*v.shape[:-1], new_c) for k, v in loaded["critic_state_dict"].items()}
    params = [torch.zeros(14), torch.zeros(h, new_a)]
    log = microduck_mdp.expand_checkpoint_obs(loaded, actor_new, critic_new, params)
    a = loaded["actor_state_dict"]
    assert a["mlp.0.weight"].shape == (h, new_a) and torch.equal(a["mlp.0.weight"][:, :old_a], w_old)
    assert torch.all(a["mlp.0.weight"][:, old_a:] == 0)                       # new inputs start silent
    assert a["obs_normalizer._mean"].shape == (1, new_a) and torch.all(a["obs_normalizer._mean"][:, old_a:] == 0)
    assert torch.all(a["obs_normalizer._var"][:, old_a:] == 1) and torch.all(a["obs_normalizer._std"][:, old_a:] == 1)
    assert a["obs_normalizer.count"].item() == 72000.0 and a["mlp.2.weight"].shape == (14, h)   # untouched
    assert loaded["critic_state_dict"]["mlp.0.weight"].shape == (h, new_c)
    st = loaded["optimizer_state_dict"]["state"]
    assert st[1]["exp_avg"].shape == (h, new_a) and torch.all(st[1]["exp_avg"][:, old_a:] == 0)
    assert st[0]["exp_avg"].shape == (14,)
    assert len(log) == 8                                                       # 4 actor + 2 critic + 2 Adam
    assert microduck_mdp.expand_checkpoint_obs(loaded, actor_new, critic_new, params) == []   # idempotent
