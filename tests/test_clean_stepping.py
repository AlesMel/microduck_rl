"""Clean stepping (run 8): the exact terrain height field, the oracle height scan, the foot-edge
contact cost, the clean-episode ladder gate, the layout-aware mirror, and the oracle task cfg."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_step_up_env_cfg as su
from mjlab_microduck.tasks import symmetry

ORACLE = ("Mjlab-StepUp-Oracle-Clean-MicroDuck", "Mjlab-StepUp-Oracle-Clean-Backlash-MicroDuck")


def _step_field(edge_x=1.0, h=0.03, res=0.01):
    """A 2 m × 1 m field: ground 0 before ``edge_x``, ``h`` after it."""
    nx, ny = int(2.0 / res) + 1, int(1.0 / res) + 1
    grid = torch.zeros(nx, ny)
    grid[int(round(edge_x / res)):, :] = h
    return microduck_mdp.TerrainHeightField(0.0, -0.5, res, grid)


# ── height field ────────────────────────────────────────────────────────────

def test_height_field_lookup_and_outside_default():
    hf = _step_field()
    xy = torch.tensor([[0.5, 0.0], [0.99, 0.2], [1.01, -0.2], [1.8, 0.0], [5.0, 0.0], [0.5, 3.0]])
    assert hf.heights(xy).tolist() == pytest.approx([0.0, 0.0, 0.03, 0.03, 0.0, 0.0])


@pytest.fixture(scope="module")
def oracle_env():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg("Mjlab-StepUp-Oracle-Clean-MicroDuck", play=True)
    cfg.scene.num_envs = 6
    cfg.scene.terrain.terrain_generator.seed = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_height_field_matches_raycasts_on_the_real_terrain(oracle_env):
    hf = microduck_mdp.terrain_height_field(oracle_env)
    mjm = oracle_env.sim.mj_model
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    origins = oracle_env.scene.terrain.terrain_origins.reshape(-1, 3).cpu().numpy()
    rng = np.random.default_rng(0)
    mask = np.zeros(6, dtype=np.uint8)
    mask[0] = 1                                                    # geom group 0 = terrain only
    geomid = np.zeros(1, dtype=np.int32)
    n_checked = 0
    for o in origins:
        for _ in range(40):
            x, y = o[0] - 0.5 + rng.uniform(0.05, 3.95), o[1] + rng.uniform(-1.8, 1.8)
            dist = mujoco.mj_ray(mjm, d, np.array([x, y, 1.0]), np.array([0, 0, -1.0]), mask, 1, -1, geomid)
            if dist < 0:
                continue
            ray_h = 1.0 - dist
            lut = float(hf.heights(torch.tensor([[x, y]], dtype=torch.float32))[0])
            # within one cell of a riser the 1 cm raster may pick the neighbouring level
            near_edge = abs(float(hf.heights(torch.tensor([[x + 0.012, y]]))[0]) - float(hf.heights(torch.tensor([[x - 0.012, y]]))[0])) > 1e-4
            if not near_edge:
                assert lut == pytest.approx(ray_h, abs=1e-4), (x, y)
                n_checked += 1
    assert n_checked > 100


# ── oracle scan ─────────────────────────────────────────────────────────────

def test_scan_layout_and_values(oracle_env):
    obs, _ = oracle_env.reset()
    xs, ys = microduck_mdp.scan_grid(su.SCAN_PARAMS["x_range"], su.SCAN_PARAMS["y_range"], su.SCAN_PARAMS["res"])
    assert (len(xs), len(ys)) == (19, 7)
    assert obs["actor"].shape[1] == 61 + 19 * 7
    assert obs["critic"].shape[1] >= 85 + 19 * 7
    names = list(oracle_env.observation_manager._group_obs_term_names["actor"])
    assert names[-1] == "height_scan" and "tof" not in names
    scan = obs["actor"][:, 61:]
    assert scan.abs().max() <= su.SCAN_PARAMS["clip"] + 1e-6
    # robots on the flat column read flat ground around them (trunk bob + 5 mm noise)
    tnames = list(oracle_env.scene.terrain.cfg.terrain_generator.sub_terrains)
    types = oracle_env.scene.terrain.terrain_types.cpu().numpy()
    for i in range(oracle_env.num_envs):
        if tnames[types[i]] == "flat":
            assert scan[i].abs().max() < 0.04


def test_scan_sees_a_step_ahead():
    """A trunk 0.3 m before a 3 cm step at nominal height: the far scan rows read −0.03."""
    hf = _step_field(edge_x=1.0, h=0.03)
    env = SimpleNamespace(device="cpu", _terrain_hf=hf)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.7, 0.0, 0.115]]),
                                                 root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
    env.scene = {"robot": robot}
    out = microduck_mdp.terrain_height_scan(env, **su.SCAN_PARAMS).reshape(19, 7)
    xs, _ = microduck_mdp.scan_grid(su.SCAN_PARAMS["x_range"], su.SCAN_PARAMS["y_range"], su.SCAN_PARAMS["res"])
    for ix, x in enumerate(xs):
        expected = -0.03 if 0.7 + x >= 1.0 - 1e-6 else 0.0   # the riser cell belongs to the top
        assert out[ix].tolist() == pytest.approx([expected] * 7, abs=1e-6), x


# ── foot edge contact ───────────────────────────────────────────────────────

def _foot(x_heel, x_toe, z0=0.0):
    return torch.tensor([[x, y, z] for x in (x_heel, x_toe) for y in (-0.02, 0.02) for z in (z0, z0 + 0.02)])


def test_foot_edge_flags():
    hf = _step_field(edge_x=1.0, h=0.03)
    corners = torch.stack([
        _foot(0.90, 0.95),          # on the ground before the step: level
        _foot(0.96, 1.01),          # toe in the riser: not level
        _foot(0.98, 1.03, 0.03),    # straddling the edge on top: not level
        _foot(1.05, 1.10, 0.03),    # fully on the tread: level
    ])[None]                        # (1, 4, 8, 3)
    found = torch.tensor([[True, True, True, True]])
    assert microduck_mdp.foot_edge_flags(hf, corners, found).tolist() == [[False, True, True, False]]
    assert not microduck_mdp.foot_edge_flags(hf, corners, torch.zeros(1, 4, dtype=torch.bool)).any()   # swing: free


def test_foot_edge_cost_runs_and_counts(oracle_env):
    oracle_env.reset()
    with torch.no_grad():
        for _ in range(5):
            oracle_env.step(torch.zeros(oracle_env.num_envs, 14))
    term = oracle_env.reward_manager.get_term_cfg("foot_edge_contact").func
    assert isinstance(term, microduck_mdp.foot_edge_contact_cost)
    assert oracle_env._foot_edge_steps.shape == (oracle_env.num_envs,)
    assert torch.all(oracle_env._foot_edge_steps >= 0)
    term.reset(torch.tensor([0]))
    assert oracle_env._foot_edge_steps[0] == 0


# ── ladder gate ─────────────────────────────────────────────────────────────

def test_ladder_promotes_only_clean_episodes():
    n = 3
    calls = {}
    terrain = SimpleNamespace(update_env_origins=lambda ids, up, down: calls.update(up=up.clone(), down=down.clone()),
                              terrain_levels=torch.zeros(n, dtype=torch.long))
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        termination_manager=SimpleNamespace(get_term=lambda k: torch.zeros(n, dtype=torch.bool) if k == "fell_over" else torch.ones(n, dtype=torch.bool)),
        reward_manager=SimpleNamespace(_episode_sums={"track_linear_velocity": torch.full((n,), 0.9 * 2.0 * 20)},
                                       get_term_cfg=lambda k: SimpleNamespace(weight=2.0)),
        max_episode_length_s=20.0,
        _foot_edge_steps=torch.tensor([0.0, 3.0, 12.0]),
    )
    microduck_mdp.terrain_levels_survival(env, torch.arange(n), min_tracking=0.55, max_edge_steps=3)
    assert calls["up"].tolist() == [True, True, False]       # 12 edge steps: timed out, but dirty
    assert calls["down"].tolist() == [False, False, False]


# ── mirror with a height scan ───────────────────────────────────────────────

def test_mirror_flips_the_scan_in_y_and_is_an_involution(oracle_env):
    obs, _ = oracle_env.reset()
    actor = obs["actor"]
    obs_perm, obs_sign, _, _ = symmetry._get_tensors(torch.device("cpu"))
    layout = symmetry._actor_layout(oracle_env)
    assert layout == [(61, 133, ("grid", 19, 7))]
    sym = symmetry.mirror_actor_obs(actor, obs_perm, obs_sign, layout)
    assert torch.equal(sym[:, 61:].reshape(-1, 19, 7), actor[:, 61:].reshape(-1, 19, 7).flip(-1))
    assert torch.allclose(symmetry.mirror_actor_obs(sym, obs_perm, obs_sign, layout), actor)


# ── cfg ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ORACLE)
def test_oracle_cfg(task):
    cfg = load_env_cfg(task, play=False)
    r = cfg.rewards
    assert r["foot_edge_contact"].weight < 0 and r["fall"].weight < 0                 # ≥ 0 costs → negative
    assert r["height_progress"].weight == su.CLEAN_PROGRESS_WEIGHT > 0
    assert r["foot_on_step"].weight == su.CLEAN_FOOT_ON_STEP_WEIGHT > 0
    assert r["trunk_impact"].weight > 0                                              # self-negating
    assert cfg.curriculum["terrain_levels"].params["max_edge_steps"] == su.MAX_EDGE_STEPS
    assert "tof" not in cfg.observations["actor"].terms
    assert not any(getattr(s, "name", "") == "tof" for s in cfg.scene.sensors)
    assert cfg.commands["head_pose"].ranges[0] == (-0.05, 0.05)                      # the gait's own head
    assert load_rl_cfg(task).run_name == "step_up_oracle_clean"
    # the ToF tasks are untouched by the clean variant
    tof_cfg = load_env_cfg("Mjlab-StepUp-Tof-Backlash-MicroDuck", play=False)
    assert "foot_edge_contact" not in tof_cfg.rewards and tof_cfg.rewards["height_progress"].weight == su.PROGRESS_WEIGHT


@pytest.mark.parametrize("task", ("Mjlab-StepUp-Tof-Clean-MicroDuck", "Mjlab-StepUp-Tof-Clean-Backlash-MicroDuck"))
def test_tof_clean_differs_from_the_oracle_only_in_perception(task):
    cfg = load_env_cfg(task, play=False)
    oracle = load_env_cfg(task.replace("Tof-Clean", "Oracle-Clean"), play=False)
    for k in ("height_progress", "foot_on_step", "foot_edge_contact", "fall", "trunk_impact", "swing_apex", "torque_headroom"):
        assert cfg.rewards[k].weight == oracle.rewards[k].weight, k
    assert cfg.curriculum["terrain_levels"].params["max_edge_steps"] == su.MAX_EDGE_STEPS
    assert list(cfg.observations["actor"].terms)[-1] == "tof" and "height_scan" not in cfg.observations["actor"].terms
    assert cfg.rewards["head_pose_tracking"].weight == su.FREE_HEAD_TRACK_WEIGHT        # the free head of run 7b
    assert load_rl_cfg(task).run_name == "step_up_tof_clean"



# ── run 10: the deployable map ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def tofmap_env():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg("Mjlab-StepUp-TofMap-Clean-MicroDuck", play=True)
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator.seed = 0
    # Driven by constant actions, not a policy: a sagging HOME would trip fell_over and reset (and
    # rightly clear) the maps mid-test. Only the time-out may end an episode here.
    for name in [k for k in cfg.terminations if k != "time_out"]:
        cfg.terminations.pop(name)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_tofmap_obs_is_values_then_mask_and_fills_in(tofmap_env):
    obs, _ = tofmap_env.reset()
    assert obs["actor"].shape[1] == 61 + 2 * 133
    assert obs["critic"].shape[1] == 85 + 133            # = run 8's critic: it loads unpadded
    # Head pitched down as the policies hold it (neck −, head +): with a level head the bottom ToF
    # row lands 0.72 m ahead, beyond the scan (≤ 0.45 m), and the scan would rightly stay unknown.
    action = torch.zeros(tofmap_env.num_envs, 14)
    action[:, 5], action[:, 6] = -4.0, 4.0
    with torch.no_grad():
        for _ in range(40):                               # 0.8 s: ≈ 12 ToF frames
            obs, *_ = tofmap_env.step(action)
    vals, mask = obs["actor"][:, 61:194], obs["actor"][:, 194:]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.all(vals[mask == 0] == 0)                # unknown reads "level with me"
    assert mask.mean() > 0.05                             # the map has filled in ahead of the robots
    assert vals.abs().max() <= su.SCAN_PARAMS["clip"] + 1e-6


def test_tofmap_mirror_flips_values_and_mask(tofmap_env):
    obs, _ = tofmap_env.reset()
    obs_perm, obs_sign, _, _ = symmetry._get_tensors(torch.device("cpu"))
    layout = symmetry._actor_layout(tofmap_env)
    assert layout == [(61, 266, ("grid", 19, 7, 2))]
    a = obs["actor"]
    m = symmetry.mirror_actor_obs(a, obs_perm, obs_sign, layout)
    assert torch.equal(m[:, 61:].reshape(-1, 2, 19, 7), a[:, 61:].reshape(-1, 2, 19, 7).flip(-1))


@pytest.mark.parametrize("task", ("Mjlab-StepUp-TofMap-Clean-MicroDuck", "Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck"))
def test_tofmap_cfg(task):
    cfg = load_env_cfg(task, play=False)
    oracle = load_env_cfg(task.replace("TofMap-Clean", "Oracle-Clean"), play=False)
    for k in ("height_progress", "foot_on_step", "foot_edge_contact", "fall", "trunk_impact"):
        assert cfg.rewards[k].weight == oracle.rewards[k].weight
    assert cfg.observations["actor"].terms["height_scan"].params["map_kwargs"] == su.TOF_MAP_KWARGS
    assert cfg.observations["critic"].terms["height_scan_clean"].func is microduck_mdp.terrain_height_scan
    assert any(getattr(s_, "name", "") == "tof" for s_ in cfg.scene.sensors)
    assert "tof" not in cfg.observations["actor"].terms                # the frame reaches the actor only via the map
    assert load_rl_cfg(task).run_name == "step_up_tofmap_clean"
