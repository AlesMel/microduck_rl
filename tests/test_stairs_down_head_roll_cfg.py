"""Stairs down head roll (tasks/microduck_stairs_down_env_cfg.py; the stairs-down policy since 2026-11-13, a
continuation of the retired head-calm fine-tune of 18d that prices mostly head roll): the cfg is stairs down
+ three terms and two ramps, the signs hold, pitch stays free, the roll stop band resolves on the backlash
model, the runner saves every 50 iterations, and mdp.head_joint_error_l1 on a fake env."""
from types import SimpleNamespace

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_stairs_down_env_cfg as sd

DOWN, ROLL = "Mjlab-StairsDown-Backlash-MicroDuck", "Mjlab-StairsDown-HeadRoll-Backlash-MicroDuck"
NEW = {"head_roll_l1", "head_yaw_l1", "head_roll_stop"}
RAMPS = {"head_roll_l1_weight": "head_roll_l1", "head_roll_stop_weight": "head_roll_stop"}


@pytest.mark.parametrize("play", [False, True])
def test_head_roll_is_stairs_down_plus_three_terms(play):
    down, cfg = load_env_cfg(DOWN, play=play), load_env_cfg(ROLL, play=play)
    assert set(cfg.rewards) - set(down.rewards) == NEW
    for k, v in down.rewards.items():                        # everything else unchanged
        assert cfg.rewards[k].func is v.func and cfg.rewards[k].weight == v.weight
    assert set(cfg.curriculum) - set(down.curriculum) == set(RAMPS)
    for g in ("actor", "critic"):                            # same obs: loads 18d / head calm, swaps with stairs up
        assert list(cfg.observations[g].terms) == list(down.observations[g].terms)
    assert cfg.scene.entities["robot"].spec_fn is down.scene.entities["robot"].spec_fn


def test_signs_ramps_and_joints():
    cfg = load_env_cfg(ROLL)
    r = cfg.rewards
    # self-negating L1 (≤ 0) → positive weights; joint_pos_limit_proximity is a cost (≥ 0) → negative
    l1 = [s["weight"] for s in cfg.curriculum["head_roll_l1_weight"].params["weight_stages"]]
    stop = [s["weight"] for s in cfg.curriculum["head_roll_stop_weight"].params["weight_stages"]]
    assert all(w > 0 for w in l1) and l1 == sorted(l1) and r["head_roll_l1"].weight == l1[0]
    assert all(w < 0 for w in stop) and stop == sorted(stop, reverse=True) and r["head_roll_stop"].weight == stop[0]
    assert r["head_yaw_l1"].weight > 0
    for name, stages in (("head_roll_l1_weight", sd.HEAD_ROLL_L1_STAGES), ("head_roll_stop_weight", sd.HEAD_ROLL_STOP_STAGES)):
        c = cfg.curriculum[name]
        assert c.params["reward_name"] == RAMPS[name]
        assert [s["step"] for s in c.params["weight_stages"]] == [it * 24 for it, _ in stages]
    # roll priced harder than head calm priced yaw + roll, yaw lighter, pitch free (the descent's counterweight)
    assert l1[-1] > sd.HEAD_CALM_FINAL_L1_WEIGHT > sd.HEAD_ROLL_YAW_L1_WEIGHT
    assert r["head_roll_l1"].func is microduck_mdp.head_joint_error_l1
    assert tuple(r["head_roll_l1"].params["joints"]) == ("head_roll",)
    assert tuple(r["head_yaw_l1"].params["joints"]) == ("head_yaw",)
    assert r["head_roll_stop"].func is microduck_mdp.joint_pos_limit_proximity
    assert tuple(r["head_roll_stop"].params["asset_cfg"].joint_names) == ("head_roll",)
    assert r["head_roll_stop"].params["margin"] == sd.HEAD_ROLL_STOP_MARGIN > 0.10   # head calm's band was 0.10
    # every inherited curriculum sits at its final stage from step 0, so a warm start only ramps the new two
    for k, c in cfg.curriculum.items():
        if k in RAMPS:
            continue
        for key in ("weight_stages", "range_stages", "standing_stages"):
            if key in c.params:
                assert len(c.params[key]) == 1 and c.params[key][0]["step"] == 0, k


def test_rl_cfg():
    rl = load_rl_cfg(ROLL)
    assert rl.experiment_name == "stairs_down_head_roll" and rl.max_iterations == 500
    assert rl.save_interval == 50                             # pick a checkpoint by battery, not the last one


def test_retired_head_fine_tunes_are_not_registered():
    tasks = set(list_tasks())
    assert ROLL in tasks
    assert not any(t.startswith(("Mjlab-StairsDown-HeadCalm", "Mjlab-StairsDown-HeadSmooth")) for t in tasks)


def test_roll_stop_band_resolves_on_the_backlash_model():
    spec = load_env_cfg(ROLL).scene.entities["robot"].spec_fn()
    m = spec.compile()
    names = [m.joint(i).name for i in range(m.njnt)]
    for j in ("head_yaw", "head_roll"):
        assert j in names and f"passive_{j}_backlash" in names
    lo, hi = m.joint("head_roll").range
    # the band starts at ≈ 0.32 rad (18°): before the 0.42 rad "on the stop" line the eval counts, well off 0
    assert 0.28 < hi - sd.HEAD_ROLL_STOP_MARGIN < 0.42 and -0.42 < lo + sd.HEAD_ROLL_STOP_MARGIN < -0.28


def _fake(q_servo: torch.Tensor, q_bl: torch.Tensor, cmd: torch.Tensor):
    """4 neck servos at qpos ids 0..3 with their backlash hinges at 4..7, default pose 0."""
    n = q_servo.shape[0]
    jp = torch.cat([q_servo, q_bl], dim=-1)
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=jp, default_joint_pos=torch.zeros(n, 8)))
    env = SimpleNamespace(scene={"robot": robot}, device="cpu",
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    env._head_pose_neck_ids = torch.arange(4)
    env._head_pose_bl_ids = torch.arange(4, 8)
    env._head_pose_bl_mask = torch.ones(4)
    return env


def test_head_joint_error_l1_prices_yaw_roll_not_pitch():
    cmd = torch.tensor([[-0.22, 0.22, 0.0, 0.0]])
    q = torch.tensor([[0.5, -0.5, 0.3, -0.4]])               # pitch far off, yaw +0.3, roll −0.4
    bl = torch.tensor([[0.0, 0.0, 0.02, 0.01]])              # read through the play
    out = microduck_mdp.head_joint_error_l1(_fake(q, bl, cmd))
    assert out.shape == (1,)
    assert out.item() == pytest.approx(-(0.32 + 0.39))
    calm = microduck_mdp.head_joint_error_l1(_fake(torch.tensor([[0.5, -0.5, 0.0, 0.0]]), torch.zeros(1, 4), cmd))
    assert calm.item() == 0.0                                # pitch error alone costs nothing
    assert (out <= 0).all()
    # one joint at a time, as head roll prices them
    roll = microduck_mdp.head_joint_error_l1(_fake(q, bl, cmd), joints=("head_roll",))
    yaw = microduck_mdp.head_joint_error_l1(_fake(q, bl, cmd), joints=("head_yaw",))
    assert roll.item() == pytest.approx(-0.39) and yaw.item() == pytest.approx(-0.32)
