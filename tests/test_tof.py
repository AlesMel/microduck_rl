"""The simulated head ToF (VL53L8CX): beam geometry ported from the runtime's ``tof.rs``, the
site frame checked in the world, the observation encoding, and the 15 Hz sample-and-hold."""
import math

import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ObservationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import tof


# ── geometry (port of the Rust unit tests) ──────────────────────────────────

def test_beams_are_unit_and_in_wire_order():
    b = tof.tof_beams_sensor_frame()
    assert b.shape == (64, 3) and np.allclose(np.linalg.norm(b, axis=1), 1.0)
    inset = tof.FOV_DEG / 2 - tof.FOV_DEG / tof.COLS / 2           # 19.6875°
    assert math.degrees(math.asin(b[0, 2])) == pytest.approx(inset, abs=1e-9)   # row 0 = top
    assert b[0, 1] > 0 and b[7, 1] < 0                                # column 0 = left (+y)
    assert b[56, 2] < 0 and b[56, 2] == pytest.approx(-b[0, 2])       # row 7 = bottom, symmetric
    centre = b[3 * 8 + 3]                                             # "slightly up-left of the axis"
    assert centre[0] > 0.99 and centre[1] > 0 and centre[2] > 0


def test_site_frame_is_sensor_frame():
    assert np.array_equal(tof.tof_beams_site_frame(), tof.tof_beams_sensor_frame())


def test_pattern_cfg_generates_64_rays():
    off, d = tof.TofPatternCfg().generate_rays(None, "cpu")
    assert off.shape == (64, 3) and torch.all(off == 0) and d.shape == (64, 3)


def test_encode_ranges():
    d = torch.tensor([-1.0, 0.05, 0.3, 1.0, 2.5, 3.6])
    enc = tof.encode_ranges(d, clip_m=1.0)
    assert enc.tolist() == pytest.approx([1.0, 1.0, 0.3, 1.0, 1.0, 1.0])


# ── in the scene ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tof_env():
    cfg = load_env_cfg("Mjlab-WalkV2-Stairs-MicroDuck", play=True)
    cfg.scene.num_envs = 4
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (tof.make_tof_sensor_cfg(),)
    cfg.observations["actor"].terms["tof"] = ObservationTermCfg(func=tof.tof_depth, params={})
    cfg.observations["critic"].terms["tof_clean"] = ObservationTermCfg(func=tof.tof_depth, params={"clean": True})
    cfg.curriculum.pop("terrain_levels", None)
    for n in list(cfg.events):
        if "push" in n:
            cfg.events.pop(n)
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    cfg.scene.terrain.terrain_generator.seed = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def _zero_step(env, n=1):
    obs = {}
    with torch.no_grad():
        for _ in range(n):
            obs, *_ = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
    return obs


def test_obs_layout_is_61_plus_64(tof_env):
    obs, _ = tof_env.reset()
    assert obs["actor"].shape[1] == 61 + 64
    names = list(tof_env.observation_manager._group_obs_term_names["actor"])
    assert names[-1] == "tof" and names[:8] == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                                                 "actions", "command", "head_command", "body_command"]


def test_grid_sees_the_floor_below_the_horizon(tof_env):
    """Head at HOME is level: the bottom row ranges the floor ≈ h / sin(19.7°) ahead, the rows
    above the horizon return nothing, and the grid is left/right symmetric on a flat tile."""
    tof_env.reset()
    _zero_step(tof_env, 3)
    sen = tof_env.scene.sensors["tof"].data
    names = list(tof_env.scene.terrain.cfg.terrain_generator.sub_terrains)
    types = tof_env.scene.terrain.terrain_types.numpy()
    flat = [e for e in range(tof_env.num_envs) if names[types[e]] == "flat"]
    assert flat, "seed 0 should deal at least one robot onto the flat column"
    e = flat[0]
    d = sen.distances[e, :64].numpy().reshape(8, 8)
    h = sen.pos_w[e, 2].item() - tof_env.scene.env_origins[e, 2].item()
    assert 0.20 < h < 0.28                                           # sensor height on the standing duck
    expected = h / math.sin(math.radians(19.6875))
    assert np.all(d[7] > 0) and np.allclose(d[7], expected, rtol=0.12)   # bottom row = floor ahead
    assert np.all(d[:4] < 0)                                         # above the horizon: no hit
    assert np.allclose(d[7], d[7, ::-1], rtol=0.03)                  # left/right symmetric
    assert d[6].mean() > d[7].mean()                                 # rows further from the floor are longer


def test_actor_frame_is_bounded(tof_env):
    obs, _ = tof_env.reset()
    for _ in range(6):
        obs = _zero_step(tof_env)
    assert obs["actor"][:, -64:].min() >= 0 and obs["actor"][:, -64:].max() <= 1
    assert obs["critic"][:, -64:].min() >= 0 and obs["critic"][:, -64:].max() <= 1


class _FakeSensorEnv:
    """A 2-env scene whose ToF distances change every step, so a held frame is detectable."""
    def __init__(self):
        self.num_envs, self.device, self.step_dt = 2, "cpu", 0.02
        self.k = 0
        self.scene = type("S", (), {})()
        self.scene.sensors = {"tof": type("Sen", (), {"data": type("D", (), {})()})()}
        self.tick()

    def tick(self):
        self.k += 1
        self.scene.sensors["tof"].data.distances = torch.full((2, 64), 0.3 + 0.001 * self.k)


def test_actor_frame_is_sampled_and_held_at_15_hz():
    env = _FakeSensorEnv()
    cfg = type("Cfg", (), {"params": {"noise_rel": 0.0, "noise_abs": 0.0, "dropout_prob": 0.0, "scale_range": 0.0}})()
    term = tof.tof_depth(cfg, env)
    term.reset()
    frames = []
    for _ in range(40):
        env.tick()
        frames.append(term(env).clone())
    f = torch.stack(frames)                                           # [40, 2, 64]
    changed = (f[1:] != f[:-1]).any(dim=2)                            # [39, 2]
    # 40 steps × 0.02 s × 15 Hz = 12 ticks; the hold alternates 3 and 4 steps.
    assert torch.all(changed.sum(dim=0) >= 11) and torch.all(changed.sum(dim=0) <= 13), changed.sum(dim=0)
    for e in range(2):
        idx = torch.nonzero(changed[:, e]).flatten()
        gaps = (idx[1:] - idx[:-1]).tolist()
        assert set(gaps) <= {3, 4}, gaps
    # latency one frame: the value shown is the PREVIOUS measurement, not the one just taken.
    clean = tof.encode_ranges(env.scene.sensors["tof"].data.distances)
    assert torch.all(f[-1] <= clean) and not torch.equal(f[-1], clean)


def test_reset_blanks_the_frame_buffer(tof_env):
    term = tof_env.observation_manager._group_obs_class_term_cfgs["actor"][-1].func
    assert isinstance(term, tof.tof_depth)
    term.reset(torch.tensor([0, 1]))
    assert torch.all(term.frames[:2] == 1.0)
