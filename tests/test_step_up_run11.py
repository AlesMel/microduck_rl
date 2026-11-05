"""Run 11 (docs/step_up_perception.md §16): the envelope (11a), the hard sensor model for the map
(11c) and tread DR + foot placement (11b) — cfg invariants, terrain, the cone rays, and the stop
functions on a fake env with an exact height field."""
import math
from types import SimpleNamespace

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
from mjlab_microduck.tasks.tof_map import TofHeightMap, tof_map_scan

BASE = "Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck"
ENVELOPE = "Mjlab-StepUp-TofMap-Envelope-Backlash-MicroDuck"
HARD = "Mjlab-StepUp-TofMap-Hard-Backlash-MicroDuck"
FLIGHTS = "Mjlab-StepUp-TofMap-Flights-Backlash-MicroDuck"


# ── terrain ──────────────────────────────────────────────────────────────────

def test_envelope_menu_adds_the_high_column_only():
    assert list(mt.microduck_step_up_subterrains()) == ["flat", "single_step", "short_flight"]
    subs = mt.microduck_step_up_subterrains(high_step=True)
    assert list(subs) == ["flat", "single_step", "short_flight", "high_step"]
    hs = subs["high_step"]
    assert hs.num_steps == 1 and hs.step_height(0.0) == pytest.approx(0.045) and hs.step_height(1.0) == pytest.approx(0.065)
    assert hs.step_height(0.0) > su.MAX_FEASIBLE_RISE > mt.STEP_UP_HEIGHT_RANGE[1]
    assert sum(t.proportion for t in subs.values()) == pytest.approx(1.0)
    gen = TerrainGenerator(mt.make_microduck_step_up_envelope_terrains_cfg(num_rows=2, seed=1), device="cpu")
    assert gen.terrain_origins.shape == (2, 4, 3)


def test_tread_range_is_drawn_per_tile_and_holds_a_foot():
    import mujoco
    t = mt.BoxSingleStepTerrainCfg(proportion=1.0, size=(4.0, 4.0), num_steps=3, step_width_range=mt.STEP_UP_TREAD_RANGE)
    treads = []
    for seed in range(6):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        out = t.function(0.5, spec, np.random.default_rng(seed))
        xs = sorted(g.geom.size[0] * 2 for g in out.geometries)          # box lengths along x
        treads.append(xs[0])                                             # the intermediate treads are the shortest boxes
        assert xs[0] == pytest.approx(xs[1])
    assert min(treads) >= 2 * mt.FOOT_LENGTH and max(treads) - min(treads) > 0.03
    assert mt.STEP_UP_TREAD_RANGE[0] >= 2 * mt.FOOT_LENGTH


# ── the cone rays ────────────────────────────────────────────────────────────

def test_cone_rays_follow_the_64_centres_zone_major():
    pat = tof.TofPatternCfg(cone_rays=4)
    off, d = pat.generate_rays(None, "cpu")
    assert d.shape == (64 * 5, 3) and off.shape == (64 * 5, 3)
    centres = torch.as_tensor(tof.tof_beams_sensor_frame(), dtype=torch.float32)
    assert torch.allclose(d[:64], centres)
    half_zone = math.radians(tof.FOV_DEG / tof.COLS / 2.0)
    for z in (0, 27, 63):
        for k in range(4):
            cos = float(d[64 + z * 4 + k] @ centres[z])
            ang = math.acos(min(1.0, cos))
            assert 0 < ang <= half_zone * 0.6 * math.sqrt(2) + 1e-6
    assert tof.TofPatternCfg().generate_rays(None, "cpu")[1].shape == (64, 3)


def test_map_picks_one_ray_per_zone_from_its_cone():
    m = TofHeightMap(2, "cpu", cone_rays=4)
    hits = torch.arange(2 * 320 * 3, dtype=torch.float32).reshape(2, 320, 3)
    dist = torch.arange(2 * 320, dtype=torch.float32).reshape(2, 320)
    torch.manual_seed(0)
    pos, d = m.pick_cone(hits, dist)
    assert pos.shape == (2, 64, 3) and d.shape == (2, 64)
    for n in range(2):
        for z in range(64):
            allowed = {float(dist[n, z])} | {float(dist[n, 64 + z * 4 + k]) for k in range(4)}
            assert float(d[n, z]) in allowed
            assert torch.equal(pos[n, z], hits[n, int(d[n, z]) - n * 320])
    assert len(set(d.flatten().tolist()) - set(dist[:, :64].flatten().tolist())) > 0   # some extras were drawn
    # without cone rays the frame passes through untouched (the caller slices it)
    p2, d2 = TofHeightMap(2, "cpu").pick_cone(hits, dist)
    assert torch.equal(d2, dist) and torch.equal(p2, hits)


# ── cfgs ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task,run_name", [(ENVELOPE, "step_up_tofmap_envelope"), (HARD, "step_up_tofmap_hard"),
                                           (FLIGHTS, "step_up_tofmap_flights")])
def test_run11_tasks_keep_run10s_obs_contract(task, run_name):
    base, cfg = load_env_cfg(BASE, play=False), load_env_cfg(task, play=False)
    assert list(cfg.observations["actor"].terms) == list(base.observations["actor"].terms)
    assert cfg.observations["actor"].terms["height_scan"].func is tof_map_scan
    assert "height_scan_clean" in cfg.observations["critic"].terms
    for k in ("foot_edge_contact", "fall", "trunk_impact", "height_progress", "foot_on_step"):
        assert k in cfg.rewards and cfg.rewards[k].weight == base.rewards[k].weight
    assert load_rl_cfg(task).run_name == run_name and load_rl_cfg(task).experiment_name == "step_up"
    assert "backlash" in str(cfg.scene.entities["robot"].spec_fn).lower()


def test_envelope_cfg():
    cfg = load_env_cfg(ENVELOPE, play=False)
    assert list(cfg.scene.terrain.terrain_generator.sub_terrains) == ["flat", "single_step", "short_flight", "high_step"]
    tr = cfg.rewards["track_linear_velocity"]
    assert tr.func is microduck_mdp.track_linear_velocity_stop_gated and tr.weight > 0
    assert tr.params["max_rise"] == su.MAX_FEASIBLE_RISE and tr.params["stop_dist"] == su.STOP_DIST
    assert tr.params["command_name"] == "twist" and tr.params["std"] == pytest.approx(math.sqrt(0.1))
    st = cfg.rewards["stop_before_high_step"]
    assert st.func is microduck_mdp.stop_before_high_step and st.weight < 0           # mjlab-style cost
    assert st.params["max_rise"] == su.MAX_FEASIBLE_RISE and st.params["stop_dist"] == su.STOP_DIST
    assert cfg.rewards["height_progress"].params["skip_tile"] == "high_step"
    assert cfg.rewards["foot_on_step"].params["skip_tile"] == "high_step"
    assert cfg.curriculum["terrain_levels"].params["progress_exempt_attr"] == "_stop_seen"
    assert cfg.curriculum["terrain_levels"].params["max_edge_steps"] == su.MAX_EDGE_STEPS
    play = load_env_cfg(ENVELOPE, play=True)
    assert "high_step" in play.scene.terrain.terrain_generator.sub_terrains
    assert "progress_exempt_attr" not in play.curriculum.get("terrain_levels", SimpleNamespace(params={})).params
    base = load_env_cfg(BASE, play=False)
    assert "stop_before_high_step" not in base.rewards and "skip_tile" not in base.rewards["height_progress"].params


def test_hard_cfg_widens_the_sensor_model_only():
    base, cfg = load_env_cfg(BASE, play=False), load_env_cfg(HARD, play=False)
    kw = cfg.observations["actor"].terms["height_scan"].params["map_kwargs"]
    kb = base.observations["actor"].terms["height_scan"].params["map_kwargs"]
    assert kw["cone_rays"] == su.TOF_MAP_CONE_RAYS == 4 and "cone_rays" not in kb
    nominal_kw = {"noise_rel": 0.01, "noise_abs": 0.004, "dropout": 0.05, "odom_scale": 0.03,
                  "odom_yaw_per_m_deg": 2.0, "odom_walk": 0.01, "odom_z_per_m": 0.01}   # TofHeightMap defaults (docs §14)
    for k, v in nominal_kw.items():
        assert kw[k] == pytest.approx(2 * v) and k not in kb
    sensor = next(s for s in cfg.scene.sensors if getattr(s, "name", "") == tof.TOF_SENSOR_NAME)
    assert sensor.pattern.cone_rays == 4
    sensor_b = next(s for s in base.scene.sensors if getattr(s, "name", "") == tof.TOF_SENSOR_NAME)
    assert sensor_b.pattern.cone_rays == 0
    assert list(cfg.rewards) == list(base.rewards) and list(cfg.scene.terrain.terrain_generator.sub_terrains) == ["flat", "single_step", "short_flight"]
    nominal = TofHeightMap(1, "cpu")
    hard = TofHeightMap(1, "cpu", **{k: v for k, v in kw.items()})
    assert hard.noise_rel == 2 * nominal.noise_rel and hard.odom_scale == 2 * nominal.odom_scale and hard.dropout == 2 * nominal.dropout


def test_flights_cfg_randomises_treads_in_training_only_and_pays_placement():
    cfg, play = load_env_cfg(FLIGHTS, play=False), load_env_cfg(FLIGHTS, play=True)
    assert cfg.scene.terrain.terrain_generator.sub_terrains["short_flight"].step_width_range == su.TREAD_DR_RANGE
    assert cfg.scene.terrain.terrain_generator.sub_terrains["single_step"].step_width_range is None
    assert play.scene.terrain.terrain_generator.sub_terrains["short_flight"].step_width_range is None
    fp = cfg.rewards["foot_placement"]
    assert fp.func is microduck_mdp.foot_placement_bonus and fp.weight == su.PLACEMENT_WEIGHT > 0
    assert fp.params["sensor_name"] == cfg.rewards["foot_edge_contact"].params["sensor_name"]
    assert fp.params["margin"] == su.PLACEMENT_MARGIN and fp.params["min_stride"] == su.PLACEMENT_MIN_STRIDE
    assert "foot_placement" not in load_env_cfg(BASE, play=False).rewards


# ── the stop functions on a fake env with an exact height field ──────────────

def _field(edge_x: float, h: float):
    res = 0.01
    xs = torch.arange(-1.0, 2.0 + res / 2, res)
    grid = torch.where(xs[:, None] >= edge_x, torch.full((len(xs), 201), h), torch.zeros(len(xs), 201))
    return microduck_mdp.TerrainHeightField(-1.0, -1.0, res, grid)


def _env(edge_x: float, h: float, vx: float = 0.25, n: int = 1, yaw: float = 0.0):
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]).repeat(n, 1),
        root_link_quat_w=torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]).repeat(n, 1),
        root_link_lin_vel_b=torch.tensor([[vx, 0.0, 0.0]]).repeat(n, 1),
    ))
    scene = {"robot": robot}
    env = SimpleNamespace(num_envs=n, device="cpu", common_step_counter=0, _terrain_hf=_field(edge_x, h),
                          scene=SimpleNamespace(__getitem__=lambda self, k: scene[k], env_origins=torch.zeros(n, 3)),
                          command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.25, 0.0, 0.0]]).repeat(n, 1)))
    env.scene = type("S", (), {"__getitem__": lambda self, k: scene[k], "env_origins": torch.zeros(n, 3)})()
    return env


def test_high_riser_ahead_sees_only_too_high_risers_within_look():
    due, dist = microduck_mdp.high_riser_ahead(_env(0.25, 0.05), max_rise=0.042)
    assert bool(due[0]) and dist[0] == pytest.approx(0.25, abs=0.011)
    due, _ = microduck_mdp.high_riser_ahead(_env(0.25, 0.035), max_rise=0.042)
    assert not bool(due[0])                                   # feasible: no stop
    due, dist = microduck_mdp.high_riser_ahead(_env(0.9, 0.05), max_rise=0.042, look=0.6)
    assert not bool(due[0]) and dist[0] == pytest.approx(0.6)  # beyond the look
    due, _ = microduck_mdp.high_riser_ahead(_env(0.25, 0.05, yaw=math.pi / 2), max_rise=0.042)
    assert not bool(due[0])                                   # the riser is to the side of the heading


def test_stop_due_latches_and_the_gated_tracking_pays_standing_still():
    env = _env(0.25, 0.05, vx=0.25)
    assert bool(microduck_mdp.stop_due(env, 0.042, 0.30)[0]) and bool(env._stop_seen[0])
    far = _env(0.5, 0.05, vx=0.25)
    assert not bool(microduck_mdp.stop_due(far, 0.042, 0.30)[0]) and not bool(far._stop_seen[0])
    std = math.sqrt(0.1)
    moving = microduck_mdp.track_linear_velocity_stop_gated(env, std, "twist", 0.042, 0.30)
    still = microduck_mdp.track_linear_velocity_stop_gated(_env(0.25, 0.05, vx=0.0), std, "twist", 0.042, 0.30)
    assert still[0] == pytest.approx(1.0) and moving[0] == pytest.approx(math.exp(-0.0625 / 0.1))
    # no riser due: the stock behaviour — walking at the command pays 1, standing pays less
    assert microduck_mdp.track_linear_velocity_stop_gated(far, std, "twist", 0.042, 0.30)[0] == pytest.approx(1.0)
    assert microduck_mdp.track_linear_velocity_stop_gated(_env(0.5, 0.05, vx=0.0), std, "twist", 0.042, 0.30)[0] < 0.6
    # the cost term: speed while due, nothing otherwise; its reset clears the env latch
    term = microduck_mdp.stop_before_high_step(SimpleNamespace(params={}), env)
    assert term(env, 0.042, 0.30)[0] == pytest.approx(1.0) and term(far, 0.042, 0.30)[0] == 0.0
    env._stop_seen[:] = True
    term.reset(torch.tensor([0]))
    assert not bool(env._stop_seen[0])


def test_ladder_exempts_a_stopped_robot_from_the_progress_rule():
    n = 3
    calls = {}
    terrain = SimpleNamespace(update_env_origins=lambda ids, up, down: calls.update(up=up.clone(), down=down.clone()),
                              terrain_levels=torch.zeros(n, dtype=torch.long))
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.3, 0, 0.1], [0.3, 0, 0.1], [2.0, 0, 0.1]])))
    env = SimpleNamespace(
        scene=type("S", (), {"__getitem__": lambda self, k: robot, "terrain": terrain, "env_origins": torch.zeros(n, 3)})(),
        termination_manager=SimpleNamespace(get_term=lambda k: torch.zeros(n, dtype=torch.bool) if k == "fell_over" else torch.ones(n, dtype=torch.bool)),
        reward_manager=SimpleNamespace(_episode_sums={"track_linear_velocity": torch.full((n,), 0.9 * 2.0 * 8)},
                                       get_term_cfg=lambda k: SimpleNamespace(weight=2.0)),
        command_manager=SimpleNamespace(get_command=lambda k: torch.tensor([[0.25, 0.0, 0.0]]).repeat(n, 1)),
        max_episode_length_s=8.0,
        _stop_seen=torch.tensor([False, True, False]),
    )
    microduck_mdp.terrain_levels_survival(env, torch.arange(n), min_tracking=0.55, min_progress=0.3,
                                          demote_no_progress=True, progress_exempt_attr="_stop_seen")
    # 0: 0.3 m of the 0.6 m needed and no stop → demoted; 1: same distance but stopped → promoted; 2: walked → promoted
    assert calls["up"].tolist() == [False, True, True] and calls["down"].tolist() == [True, False, False]


# ── the placement bonus on the real env ──────────────────────────────────────

def test_foot_placement_runs_on_the_flights_env():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg("Mjlab-StepUp-TofMap-Flights-MicroDuck", play=True)
    cfg.scene.num_envs = 3
    cfg.scene.terrain.terrain_generator.seed = 0
    cfg.rewards["foot_placement"] = load_env_cfg("Mjlab-StepUp-TofMap-Flights-MicroDuck", play=False).rewards["foot_placement"]
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        for _ in range(5):
            env.step(torch.zeros(3, env.action_manager.total_action_dim))
        term = env.reward_manager._term_cfgs[env.reward_manager._term_names.index("foot_placement")]
        val = term.func(env, **term.params)
        assert val.shape == (3,) and bool((val >= 0).all()) and bool((val <= 2).all())
        assert env.observation_manager.group_obs_dim["actor"] == (327,)
    finally:
        env.close()
