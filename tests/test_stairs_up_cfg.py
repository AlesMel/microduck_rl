"""Stairs up (run 15, tasks/microduck_stairs_up_env_cfg.py; docs/stairs.md, history in
docs/archive/stairs_runlog.md §15-23): the terrain, the ToF cone rays, the 327-D obs contract, the
registered cfgs, every reward term / spawn on a fake env with an exact height field, and the real
env (map fill-in, mirror, edge cost, placement).

Ported 2026-11-11 from the per-run test files of the step-up line (runs 6-18); the assertions on
retired variants went with them. The fake-env helpers (_stairs_down, _env, _put) are shared with
test_stairs_down_cfg.py."""
import json
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab_microduck.robot import microduck_constants as mc
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_stairs_up_env_cfg as su
from mjlab_microduck.tasks import microduck_terrains as mt
from mjlab_microduck.tasks import symmetry, tof
from mjlab_microduck.tasks.tof_map import TofHeightMap, tof_map_scan

UP = "Mjlab-StairsUp-MicroDuck"
UP_BL = "Mjlab-StairsUp-Backlash-MicroDuck"
TASKS = (UP, UP_BL)
ACTOR_HEAD = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
              "head_command", "body_command"]                      # the 61-D family block
MENU = ["flat", "single_step", "short_flight", "single_drop", "down_flight"]


# ── terrain ──────────────────────────────────────────────────────────────────

def test_single_step_tile_geometry():
    t = mt.BoxSingleStepTerrainCfg(proportion=1.0, size=(4.0, 4.0), num_steps=1)
    assert t.step_height(0.0) == pytest.approx(0.015) and t.step_height(1.0) == pytest.approx(0.040)
    f = mt.BoxSingleStepTerrainCfg(proportion=1.0, size=(4.0, 4.0), num_steps=3)
    assert f.step_height(0.5) * 3 == pytest.approx(0.0825)


def test_step_up_menus_build():
    # the walk-v2 terrain menu offers all three (microduck_walk_v2_terrain_env_cfg); stairs up uses step_updown
    assert list(mt.microduck_step_up_subterrains()) == ["flat", "single_step", "short_flight"]
    gen = TerrainGenerator(mt.make_microduck_step_up_terrains_cfg(num_rows=3, seed=1), device="cpu")
    assert gen.terrain_origins.shape == (3, 3, 3)
    subs = mt.microduck_step_up_subterrains(high_step=True)
    assert list(subs) == ["flat", "single_step", "short_flight", "high_step"]
    hs = subs["high_step"]
    assert hs.num_steps == 1 and hs.step_height(0.0) == pytest.approx(0.045) and hs.step_height(1.0) == pytest.approx(0.065)
    assert hs.step_height(0.0) > mt.STEP_UP_HEIGHT_RANGE[1]
    assert sum(t.proportion for t in subs.values()) == pytest.approx(1.0)
    gen = TerrainGenerator(mt.make_microduck_step_up_envelope_terrains_cfg(num_rows=2, seed=1), device="cpu")
    assert gen.terrain_origins.shape == (2, 4, 3)


def test_tread_range_is_drawn_per_tile_and_holds_a_foot():
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
    assert su.TREAD_DR_RANGE == mt.STEP_UP_TREAD_RANGE


def _tile(cfg: mt.BoxSingleStepTerrainCfg, difficulty: float = 0.5, seed: int = 0):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    out = cfg.function(difficulty, spec, np.random.default_rng(seed))
    boxes = sorted((g.geom.pos[0] - g.geom.size[0], g.geom.pos[0] + g.geom.size[0], g.geom.pos[2] + g.geom.size[2])
                   for g in out.geometries)
    return out, boxes


@pytest.mark.parametrize("n", [1, 3])
def test_down_tile_mirrors_the_up_tile(n):
    kw = dict(proportion=1.0, size=(4.0, 4.0), num_steps=n, step_height_range=(0.03, 0.03))
    up_out, up = _tile(mt.BoxSingleStepTerrainCfg(**kw))
    dn_out, dn = _tile(mt.BoxSingleStepTerrainCfg(**kw, down=True))
    assert np.allclose([b[:2] for b in up], [b[:2] for b in dn])             # same x layout (same rng draws)
    assert [b[2] for b in up] == pytest.approx([0.03 * k for k in range(n + 1)])
    assert [b[2] for b in dn] == pytest.approx([0.03 * (n - k) for k in range(n + 1)])
    assert up_out.origin[2] == 0.0
    assert dn_out.origin[2] == pytest.approx(0.03 * n)                       # spawn origin on the upper platform
    assert dn_out.origin[0] < dn[0][1]                                        # ... before the first drop


def test_updown_menu_and_generator():
    subs = mt.microduck_step_updown_subterrains(tread_range=mt.STEP_UP_TREAD_RANGE)
    assert list(subs) == MENU
    assert sum(t.proportion for t in subs.values()) == pytest.approx(1.0)
    assert set(mt.DOWN_TILES) == {k for k, t in subs.items() if getattr(t, "down", False)}
    assert subs["down_flight"].num_steps == subs["short_flight"].num_steps == 3
    assert subs["single_drop"].step_height_range == subs["single_step"].step_height_range == mt.STEP_UP_HEIGHT_RANGE
    assert subs["down_flight"].step_width_range == mt.STEP_UP_TREAD_RANGE
    gen = TerrainGenerator(mt.make_microduck_step_updown_terrains_cfg(num_rows=3, seed=1), device="cpu")
    gen.compile(mujoco.MjSpec())                                              # origins are filled in here
    assert gen.terrain_origins.shape == (3, 5, 3)
    o = np.asarray(gen.terrain_origins)
    for c, name in enumerate(subs):
        if name in mt.DOWN_TILES:
            assert (o[:, c, 2] > 0.01).all()                                  # upper platform
        else:
            assert np.allclose(o[:, c, 2], 0.0)


# ── the ToF cone rays (the hard sensor model of run 11c; TofPatternCfg / TofHeightMap keep it) ──

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


# ── the registered cfgs ─────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_obs_contract_and_perception(task):
    cfg = load_env_cfg(task, play=False)
    assert any(getattr(s, "name", "") == tof.TOF_SENSOR_NAME for s in cfg.scene.sensors)
    actor = list(cfg.observations["actor"].terms)
    assert actor == ACTOR_HEAD + ["height_scan"]                      # 61-D family block, then the map scan
    scan = cfg.observations["actor"].terms["height_scan"]
    assert scan.func is tof_map_scan and scan.params["map_kwargs"] == su.TOF_MAP_KWARGS
    assert scan.params["sensor_name"] == tof.TOF_SENSOR_NAME
    assert "tof" not in actor                                         # the frame reaches the actor only via the map
    clean = cfg.observations["critic"].terms["height_scan_clean"]
    assert clean.func is microduck_mdp.terrain_height_scan and clean.params["noise"] == 0.0
    for k in ("x_range", "y_range", "res", "nominal", "clip"):
        assert scan.params[k] == clean.params[k] == su.SCAN_PARAMS[k], k
    assert cfg.episode_length_s == su.EPISODE_LENGTH_S


@pytest.mark.parametrize("task", TASKS)
def test_reward_signs_and_weights(task):
    """mjlab-style costs (≥ 0) carry a NEGATIVE weight, self-negating penalties and bonuses a positive one."""
    cfg = load_env_cfg(task, play=False)
    r = cfg.rewards
    D = mt.DOWN_TILES
    sensor = r["foot_edge_contact"].params["sensor_name"]
    # the climb
    hp = r["height_progress"]
    assert hp.func is microduck_mdp.trunk_height_progress and hp.weight == su.CLEAN_PROGRESS_WEIGHT > 0
    assert hp.params == {"clip": su.PROGRESS_CLIP, "skip_tile": D}
    fs = r["foot_on_step"]
    assert fs.func is microduck_mdp.foot_on_step_bonus and fs.weight == su.CLEAN_FOOT_ON_STEP_WEIGHT > 0
    assert tuple(fs.params["asset_cfg"].site_names) == ("left_foot", "right_foot")
    assert fs.params["sole"] is True and fs.params["min_rise"] == su.FOOT_ON_STEP_MIN_RISE
    assert fs.params["sensor_name"] == sensor
    assert r["trunk_impact"].func is microduck_mdp.trunk_vertical_accel_penalty and r["trunk_impact"].weight > 0  # self-negating
    edge = r["foot_edge_contact"]
    assert edge.func is microduck_mdp.foot_edge_contact_cost and edge.weight == su.EDGE_WEIGHT < 0
    assert edge.params["tol"] == su.EDGE_TOL
    assert edge.params["scale_tiles"] == D and edge.params["tile_scale"] == su.DOWN_EDGE_SCALE > 1
    assert edge.params["shrink"] == su.EDGE_SHRINK and edge.params["shrink_tiles"] == D
    assert r["fall"].weight == su.FALL_WEIGHT < 0
    fp = r["foot_placement"]
    assert fp.func is microduck_mdp.foot_placement_bonus and fp.weight == su.PLACEMENT_WEIGHT > 0
    assert fp.params["sensor_name"] == sensor and fp.params["down_tiles"] == D
    assert fp.params["margin"] == su.PLACEMENT_MARGIN and fp.params["min_stride"] == su.PLACEMENT_MIN_STRIDE
    assert "swing_apex" in r and r["swing_apex"].weight > 0 and r["torque_headroom"].weight < 0
    # the descent
    dp, fb = r["descent_progress"], r["foot_below_step"]
    assert dp.func is microduck_mdp.descent_progress and dp.weight == su.CLEAN_PROGRESS_WEIGHT > 0
    assert fb.func is microduck_mdp.foot_below_step_bonus and fb.weight == su.CLEAN_FOOT_ON_STEP_WEIGHT > 0
    assert dp.params["tiles"] == fb.params["tiles"] == D
    assert fb.params["min_rise"] == fs.params["min_rise"] and fb.params["sensor_name"] == sensor
    cs = r["clean_step_down"]
    assert cs.func is microduck_mdp.clean_step_down_bonus and cs.weight == su.CLEAN_STEP_DOWN_WEIGHT > 0
    assert cs.params["tiles"] == D and cs.params["min_rise"] == fs.params["min_rise"]
    assert cs.params["tol"] == edge.params["tol"] and cs.params["sensor_name"] == sensor
    assert cs.params["shrink"] == su.EDGE_SHRINK
    td = r["trunk_drop"]
    assert td.func is microduck_mdp.trunk_drop_penalty and td.weight == su.TRUNK_DROP_WEIGHT > 0   # self-negating
    assert td.params["tiles"] == D
    assert 0.1 < td.params["max_down_vel"] == su.TRUNK_DROP_MAX_VEL < 0.6        # between the treads and a drop
    assert td.params["zone_max_down_vel"] == su.EDGE_DROP_MAX_VEL < su.TRUNK_DROP_MAX_VEL
    # the down edge
    assert r["upright"].func is microduck_mdp.upright_edge_lean and r["upright"].params["max_back_deg"] == su.EDGE_MAX_BACK_DEG
    assert r["pose"].func is microduck_mdp.variable_posture_edge
    assert r["pose"].params["free_joints"] == r".*(hip_pitch|knee|ankle).*"
    assert r["track_vx_ema"].func is microduck_mdp.track_velocity_ema_edge
    assert r["track_linear_velocity"].func is microduck_mdp.track_linear_velocity_edge
    for k in ("upright", "pose", "track_vx_ema", "track_linear_velocity"):
        assert r[k].params["tiles"] == D and r[k].weight > 0, k
    assert r["track_vx_ema"].params["v_edge"] == r["track_linear_velocity"].params["v_edge"] == su.EDGE_V
    assert 0 < su.EDGE_V < su.VX_RANGE[0]                                       # below every command
    assert cfg.curriculum["terrain_levels"].params["max_edge_steps"] == su.MAX_EDGE_STEPS


def test_commands_point_into_the_step_and_head_down():
    cfg = load_env_cfg(UP, play=False)
    tw = cfg.commands["twist"]
    assert tw.ranges.lin_vel_x == su.VX_RANGE and tw.ranges.lin_vel_x[0] > 0
    assert tw.ranges.lin_vel_y == su.VY_RANGE and tw.ranges.ang_vel_z == su.WZ_RANGE
    hp = cfg.commands["head_pose"].ranges
    assert hp[0][0] < hp[0][1] < 0 and 0 < hp[1][0] < hp[1][1]          # neck down (−), head down (+)
    assert np.isclose(np.mean(hp[0]), su.HEAD_DOWN[0]) and np.isclose(np.mean(hp[1]), su.HEAD_DOWN[1])
    assert hp[0][1] - hp[0][0] == pytest.approx(2 * su.HEAD_JITTER)       # a range, not a constant: the slot stays alive
    head = cfg.rewards["head_pose_tracking"]
    assert head.params["std"] == su.HEAD_TRACK_STD and head.weight == su.HEAD_TRACK_WEIGHT     # a weak prior
    assert "head_pose_range" not in cfg.curriculum
    rb = cfg.events["reset_base"]
    assert rb.func is microduck_mdp.reset_root_on_terrain
    assert rb.params["yaw_range"] == (-su.SPAWN_YAW, su.SPAWN_YAW)
    assert rb.params["low_frac"] == su.SPAWN_LOW_FRAC > 0.5 and rb.params["min_dx"] == su.SPAWN_MIN_DX


@pytest.mark.parametrize("play", [False, True])
def test_terrain_and_tread_dr_in_training_only(play):
    tg = load_env_cfg(UP, play=play).scene.terrain.terrain_generator
    assert list(tg.sub_terrains) == MENU
    want = su.TREAD_DR_RANGE if not play else None                     # the play grid keeps 15 cm
    assert tg.sub_terrains["short_flight"].step_width_range == tg.sub_terrains["down_flight"].step_width_range == want
    assert tg.sub_terrains["single_step"].step_width_range is None


@pytest.mark.parametrize("play", [False, True])
def test_the_mid_step_spawn_follows_the_joint_reset(play):
    ev = list(load_env_cfg(UP_BL, play=play).events)
    if play:
        assert "reset_at_down_edge" not in ev                                      # batteries stay comparable
        return
    cfg = load_env_cfg(UP_BL, play=False)
    assert ev.index("reset_at_down_edge") == ev.index("reset_robot_joints") + 1
    e = cfg.events["reset_at_down_edge"]
    assert e.func is microduck_mdp.reset_at_down_edge
    assert e.params["tiles"] == mt.DOWN_TILES and e.params["prob"] == su.EDGE_SPAWN_PROB
    assert e.params["bank"] == su.EDGE_SPAWN_BANK


def test_robot_models_have_leg_collisions():
    assert load_env_cfg(UP).scene.entities["robot"].spec_fn is mc.get_walk_legcol_spec
    assert load_env_cfg(UP, play=True).scene.entities["robot"].spec_fn is mc.get_walk_legcol_spec
    assert load_env_cfg(UP_BL).scene.entities["robot"].spec_fn is mc.get_walk_backlash_legcol_spec
    assert load_env_cfg(UP_BL, play=True).scene.entities["robot"].spec_fn is mc.get_walk_backlash_legcol_spec


@pytest.mark.parametrize("task", TASKS)
def test_runner(task):
    rl = load_rl_cfg(task)
    assert rl.experiment_name == "stairs_up" and rl.run_name == "stairs_up"
    assert rl.max_iterations == 2000
    assert rl.algorithm.class_name == "PPO"                                       # plain PPO, no BC anchor


def test_leg_collision_spec_keeps_the_mass_and_names_the_new_geoms():
    plain, legs = mc.get_walk_backlash_spec().compile(), mc.get_walk_backlash_legcol_spec().compile()
    assert plain.nbody == legs.nbody and np.allclose(plain.body_mass, legs.body_mass)
    assert np.allclose(plain.body_inertia, legs.body_inertia)
    col = sorted(legs.geom(g).name for g in range(legs.ngeom) if legs.geom(g).name.endswith("_collision"))
    spec = mc.get_walk_backlash_legcol_spec()
    assert all(spec.geom(n) is not None for n in col)                             # mjlab's CollisionCfg looks them up
    assert col == sorted(f"{s}_{p}_collision" for s in ("left", "right") for p in ("foot", "shin", "thigh"))
    for s, thigh, shin in (("left", "upper_leg_left", "leg"), ("right", "upper_leg_right", "leg_2")):
        assert legs.geom_bodyid[legs.geom(f"{s}_thigh_collision").id] == legs.body(thigh).id
        assert legs.geom_bodyid[legs.geom(f"{s}_shin_collision").id] == legs.body(shin).id


# ── fake envs ────────────────────────────────────────────────────────────────

class _Asset:
    def __init__(self, n):
        self.data = type("D", (), {})()
        self.data.root_link_pos_w = torch.zeros(n, 3)
        self.data.site_pos_w = torch.zeros(n, 2, 3)
        self.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)
        self.data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)


class _Env:
    """Flat ground, every robot on one column (trunk_height_progress / the site-based foot_on_step)."""

    def __init__(self, n=3):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.asset = _Asset(n)
        self.found = torch.ones(n, 2)
        origins = torch.zeros(n, 3)
        contact = type("C", (), {"data": type("CD", (), {"found": self.found})()})()
        self.scene = type("S", (), {"env_origins": origins})()
        self.scene.terrain = type("T", (), {"env_origins": origins})()
        self.scene.__class__.__getitem__ = lambda s, k: self.asset if k == "robot" else contact


TILES = list(MENU)


def _stairs_down(edges: list[float], h: float, top_levels: int):
    """Exact field: level (top_levels − k)·h after edge k along x."""
    res = 0.01
    xs = torch.arange(-1.0, 3.0 + res / 2, res)
    z = torch.full((len(xs),), top_levels * h)
    for e in edges:
        z = torch.where(xs >= e, z - h, z)
    return microduck_mdp.TerrainHeightField(-1.0, -1.0, res, z[:, None].repeat(1, 201))


def _env(hf, types: list[str], origin_z: list[float], feet_xyz: torch.Tensor, found=None, quat=None):
    """Robots on the step_updown menu's columns ``types``, foot boxes at ``feet_xyz`` (B, 2, 3)."""
    n = len(types)
    origins = torch.tensor([[0.0, 0.0, z] for z in origin_z])
    terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={k: None for k in TILES})),
                              terrain_types=torch.tensor([TILES.index(t) for t in types]), env_origins=origins)
    robot = SimpleNamespace(data=SimpleNamespace(
        site_pos_w=feet_xyz.clone(),
        root_link_quat_w=(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1) if quat is None else quat),
        root_link_pos_w=origins + torch.tensor([0.0, 0.0, 0.12])))          # standing, upright
    contact = SimpleNamespace(data=SimpleNamespace(found=(torch.ones(n, 2) if found is None else found)))
    items = {"robot": robot, "feet": contact}
    scene = type("S", (), {"__getitem__": lambda self, k: items[k], "env_origins": origins, "terrain": terrain})()
    aabb = np.array([[0.0, 0.0, 0.0, 0.027, 0.02, 0.005]] * 2)                # foot boxes: centre, half sizes
    sim = SimpleNamespace(mj_model=SimpleNamespace(geom=lambda name: SimpleNamespace(id=0 if "left" in name else 1),
                                                   geom_aabb=aabb),
                          data=SimpleNamespace(geom_xpos=feet_xyz.clone(), geom_xmat=torch.eye(3).flatten().repeat(n, 2, 1)))
    return SimpleNamespace(num_envs=n, device="cpu", scene=scene, sim=sim, _terrain_hf=hf)


def _put(env, foot: int, x: float):
    env.sim.data.geom_xpos[:, foot, 0] = x


# ── the climb ────────────────────────────────────────────────────────────────

def test_trunk_height_progress_is_potential_based():
    env = _Env()
    term = microduck_mdp.trunk_height_progress(SimpleNamespace(params={}), env)
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
    term = microduck_mdp.foot_on_step_bonus(SimpleNamespace(params={}), env)
    site_cfg = type("SC", (), {"name": "robot", "site_ids": [0, 1]})()
    env.asset.data.site_pos_w[:, :, 2] = 0.02                             # spawned ON the step: never stood low
    assert torch.all(term(env, "feet", 0.012, 40.0, site_cfg) == 0)      # → nothing (no free bonus)
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
    term.reset(torch.tensor([0]))                                         # new episode: latch cleared
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [0.0, 0.0, 0.0]   # still on the step: no pay
    env.asset.data.site_pos_w[0, :, 2] = 0.0                                # both feet down to the ground
    term(env, "feet", 0.012, 40.0, site_cfg)
    env.asset.data.site_pos_w[0, :, 2] = 0.02                               # and up together
    assert term(env, "feet", 0.012, 40.0, site_cfg).tolist() == [2.0, 0.0, 0.0]


def test_foot_on_step_on_the_sole_ignores_flat_ground():
    hf = _stairs_down([], 0.0, 0)
    half_z = 0.005
    flat = torch.tensor([[[0.2, 0.05, half_z], [0.2, -0.05, half_z]]])
    env = _env(hf, ["flat"], [0.0], flat)
    env.scene["robot"].data.site_pos_w[:, :, 2] = 0.013                    # the foot sites stand 13 mm above the sole
    fs = SimpleNamespace(site_ids=[0, 1], name="robot")
    by_site = microduck_mdp.foot_on_step_bonus(SimpleNamespace(params={}), env)
    by_sole = microduck_mdp.foot_on_step_bonus(SimpleNamespace(params={"sole": True}), env)
    env.scene["robot"].data.site_pos_w[:, :, 2] = 0.004                    # a dip arms the site version
    by_site(env, "feet", asset_cfg=fs)
    by_sole(env, "feet", asset_cfg=fs, sole=True)
    env.scene["robot"].data.site_pos_w[:, :, 2] = 0.013
    assert by_site(env, "feet", asset_cfg=fs)[0] == 2.0                    # the old bug: paid on flat ground
    assert by_sole(env, "feet", asset_cfg=fs, sole=True)[0] == 0.0
    env.sim.data.geom_xpos[0, :, 2] = 0.015 + half_z                       # soles on a 1.5 cm step
    assert by_sole(env, "feet", asset_cfg=fs, sole=True)[0] == 2.0


def test_terrain_tiles_mask_accepts_a_name_or_several():
    env = _env(_stairs_down([], 0.0, 0), ["flat", "single_drop", "down_flight"], [0, 0, 0], torch.zeros(3, 2, 3))
    assert microduck_mdp.terrain_tiles_mask(env, "single_drop").tolist() == [False, True, False]
    assert microduck_mdp.terrain_tiles_mask(env, mt.DOWN_TILES).tolist() == [False, True, True]
    assert microduck_mdp.terrain_tiles_mask(env, ("nope",)).tolist() == [False, False, False]


# ── the descent ─────────────────────────────────────────────────────────────

def test_descent_progress_telescopes_on_down_tiles_only():
    hf = _stairs_down([0.5, 0.7, 0.9], 0.03, 3)
    feet = torch.tensor([[[0.2, 0.0, 0.09], [0.2, 0.0, 0.09]]]).repeat(2, 1, 1)
    env = _env(hf, ["down_flight", "short_flight"], [0.09, 0.0], feet)
    term = microduck_mdp.descent_progress(SimpleNamespace(params={}), env)
    total = torch.zeros(2)
    path = [0.2, 0.4, 0.55, 0.6, 0.75, 0.8, 0.95, 1.2, 0.95, 1.3]       # down, a step back up, down again
    for x in path:
        env.sim.data.geom_xpos[:, :, 0] = x
        total += term(env, tiles=mt.DOWN_TILES)
    assert total[0] == pytest.approx(0.09, abs=1e-5)                       # ground start − ground end = 3 risers
    assert total[1] == 0.0                                                  # not a downhill column
    term.reset(torch.tensor([0]))
    env.sim.data.geom_xpos[:, :, 0] = 0.2                                  # teleport back up: fresh, pays 0
    assert term(env, tiles=mt.DOWN_TILES)[0] == 0.0


def test_foot_below_step_pays_once_per_foot_after_standing_on_the_platform():
    hf = _stairs_down([0.5], 0.03, 1)
    half_z = 0.005                                                         # foot box half height: sole = centre − 5 mm
    on_top = torch.tensor([[[0.2, 0.05, 0.03 + half_z], [0.2, -0.05, 0.03 + half_z]]])   # soles on the platform (z 0.03)
    env = _env(hf, ["single_drop"], [0.03], on_top)
    term = microduck_mdp.foot_below_step_bonus(SimpleNamespace(params={}), env)
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 0.0                 # on the platform: arms it
    env.sim.data.geom_xpos[0, 0, 2] = half_z + 0.019                       # left sole 11 mm below: not yet
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 0.0
    env.sim.data.geom_xpos[0, 0, 2] = half_z                               # left foot down on the lower level ...
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 0.0                 # ... but the trunk still over the platform: a probe
    env.scene["robot"].data.root_link_pos_w[0, 0] = 0.6                    # trunk over the lower level
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 1.0
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 0.0                 # once
    env.sim.data.geom_xpos[0, 1, 2] = half_z
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 1.0
    # spawned at the bottom: never stood on the platform → nothing
    low = _env(hf, ["single_drop"], [0.03], torch.tensor([[[0.8, 0.05, half_z], [0.8, -0.05, half_z]]]))
    low.scene["robot"].data.root_link_pos_w[0, 0] = 0.8
    t2 = microduck_mdp.foot_below_step_bonus(SimpleNamespace(params={}), low)
    assert t2(low, "feet", tiles=mt.DOWN_TILES)[0] == 0.0
    # an uphill column never pays
    up = _env(hf, ["single_step"], [0.03], on_top)
    t3 = microduck_mdp.foot_below_step_bonus(SimpleNamespace(params={}), up)
    t3(up, "feet", tiles=mt.DOWN_TILES)
    up.sim.data.geom_xpos[0, :, 2] = half_z
    up.scene["robot"].data.root_link_pos_w[0, 0] = 0.6
    assert t3(up, "feet", tiles=mt.DOWN_TILES)[0] == 0.0


def test_clean_step_down_pays_new_levels_on_clean_touchdowns_only():
    hf = _stairs_down([0.5, 0.7, 0.9], 0.03, 3)                 # top 9 cm, treads at 6 / 3 / 0 cm
    feet = torch.tensor([[[0.2, 0.05, 0.0], [0.2, -0.05, 0.0]]]).repeat(2, 1, 1)
    env = _env(hf, ["down_flight", "short_flight"], [0.09, 0.0], feet)
    term = microduck_mdp.clean_step_down_bonus(SimpleNamespace(params={}), env)
    kw = dict(sensor_name="feet", tiles=mt.DOWN_TILES)
    assert term(env, **kw).tolist() == [0.0, 0.0]                # first contact on the platform: reference
    _put(env, 0, 0.6)                                            # left foot clean on the first tread below
    assert term(env, **kw)[0] == pytest.approx(3.0)              # 3 cm
    assert term(env, **kw)[0] == 0.0                              # once per level
    _put(env, 1, 0.5)                                            # right foot ACROSS the first edge
    assert term(env, **kw)[0] == 0.0
    _put(env, 1, 0.6)                                            # ... then clean on that tread: forfeited
    assert term(env, **kw)[0] == 0.0
    _put(env, 0, 0.8)                                            # left clean one more tread down
    _put(env, 1, 0.8)                                            # right too: its next riser pays again
    assert term(env, **kw)[0] == pytest.approx(6.0)
    _put(env, 0, 0.2)                                            # back up and down again: nothing new
    term(env, **kw)
    _put(env, 0, 0.8)
    assert term(env, **kw)[0] == 0.0
    assert term(env, **kw)[1] == 0.0                              # an uphill column never pays
    # in the air (no contact) over a lower tread: nothing, and the level is not consumed
    env.scene["feet"].data.found[0, 0] = 0.0
    _put(env, 0, 1.2)
    assert term(env, **kw)[0] == 0.0
    env.scene["feet"].data.found[0, 0] = 1.0
    assert term(env, **kw)[0] == pytest.approx(3.0)              # lands clean on the bottom
    term.reset(torch.tensor([0]))
    assert term(env, **kw)[0] == 0.0                              # new episode: the first contact is the reference


def test_clean_step_down_pays_both_feet():
    hf = _stairs_down([0.5, 0.7, 0.9], 0.03, 3)
    env = _env(hf, ["down_flight"], [0.09], torch.tensor([[[0.2, 0.05, 0.0], [0.2, -0.05, 0.0]]]))
    term = microduck_mdp.clean_step_down_bonus(SimpleNamespace(params={}), env)
    term(env, "feet", tiles=mt.DOWN_TILES)
    _put(env, 0, 0.6)
    _put(env, 1, 0.6)
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0].item() == pytest.approx(6.0)


def test_clean_step_down_not_paid_when_fallen():
    hf = _stairs_down([0.5], 0.03, 1)
    tilt = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])      # 90° roll
    env = _env(hf, ["single_drop"], [0.03], torch.tensor([[[0.2, 0.05, 0.0], [0.2, -0.05, 0.0]]]), quat=tilt)
    term = microduck_mdp.clean_step_down_bonus(SimpleNamespace(params={}), env)
    term(env, "feet", tiles=mt.DOWN_TILES)
    _put(env, 0, 0.8)
    assert term(env, "feet", tiles=mt.DOWN_TILES)[0] == 0.0


def test_edge_cost_scaled_on_down_tiles_only():
    hf = _stairs_down([0.5], 0.03, 1)
    across = torch.tensor([[[0.5, 0.05, 0.0], [0.2, -0.05, 0.0]]]).repeat(2, 1, 1)        # left foot on the edge
    env = _env(hf, ["single_drop", "single_step"], [0.03, 0.0], across)
    cost = microduck_mdp.foot_edge_contact_cost(SimpleNamespace(params={}), env)
    out = cost(env, "feet", scale_tiles=mt.DOWN_TILES, tile_scale=2.0)
    assert out.tolist() == [2.0, 1.0]
    assert env._foot_edge_steps.tolist() == [1.0, 1.0]           # the ladder counts contact steps, unscaled
    assert cost(env, "feet").tolist() == [1.0, 1.0]              # no scale


# ── trunk_drop_penalty ──────────────────────────────────────────────────────

def _drop_env(vz: list[float], types: list[str], steps: int = 50):
    hf = _stairs_down([0.5], 0.03, 1)
    n = len(types)
    env = _env(hf, types, [0.03 if t in mt.DOWN_TILES else 0.0 for t in types],
               torch.tensor([[[0.2, 0.05, 0.0], [0.2, -0.05, 0.0]]]).repeat(n, 1, 1))
    env.scene["robot"].data.root_link_lin_vel_w = torch.tensor([[0.2, 0.0, v] for v in vz])
    env.episode_length_buf = torch.full((n,), steps, dtype=torch.long)
    env.step_dt = 0.02
    return env


def test_trunk_drop_prices_only_the_speed_beyond_the_cap():
    env = _drop_env([0.0, 0.2, -0.2, -0.3, -0.8], ["down_flight"] * 5)
    out = microduck_mdp.trunk_drop_penalty(env, max_down_vel=0.3, tiles=mt.DOWN_TILES)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, -0.5])           # rising, slow, at the cap: free
    assert (out <= 0).all()


def test_trunk_drop_only_on_down_tiles_and_after_settling():
    env = _drop_env([-0.8] * 4, ["down_flight", "single_drop", "short_flight", "flat"])
    out = microduck_mdp.trunk_drop_penalty(env, max_down_vel=0.3, tiles=mt.DOWN_TILES)
    assert out.tolist() == pytest.approx([-0.5, -0.5, 0.0, 0.0])
    env.episode_length_buf = torch.tensor([9, 10, 50, 50])                      # 0.18 s / 0.20 s into the episode
    out = microduck_mdp.trunk_drop_penalty(env, max_down_vel=0.3, tiles=mt.DOWN_TILES, settle_s=0.2)
    assert out.tolist() == pytest.approx([0.0, -0.5, 0.0, 0.0])


def test_trunk_drop_episode_sum_is_the_fall_beyond_the_cap():
    """weight · dt · Σ = weight · metres fallen faster than the cap: a 3 cm free fall costs ≈ 1.1 cm."""
    env = _drop_env([0.0], ["single_drop"])
    g, cap, h, dt = 9.81, 0.3, 0.03, 0.0005
    t, z, total = 0.0, 0.0, 0.0
    while z < h:
        env.scene["robot"].data.root_link_lin_vel_w[0, 2] = -g * t
        total += float(microduck_mdp.trunk_drop_penalty(env, max_down_vel=cap, tiles=mt.DOWN_TILES)[0]) * dt
        t += dt
        z = 0.5 * g * t * t
    t1, t2 = cap / g, (2 * h / g) ** 0.5
    assert -total == pytest.approx(0.5 * g * (t2 ** 2 - t1 ** 2) - cap * (t2 - t1), rel=0.03)
    assert -total * 100 == pytest.approx(1.13, abs=0.05)                        # cm


def test_trunk_drop_zone_cap(monkeypatch):
    env = _env(_stairs_down([0.5], 0.03, 1), ["single_drop"] * 3, [0.03] * 3, torch.zeros(3, 2, 3))
    env.scene["robot"].data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, -0.25]] * 3)
    env.episode_length_buf = torch.full((3,), 50, dtype=torch.long)
    env.step_dt = 0.02
    monkeypatch.setattr(microduck_mdp, "down_edge_zone", lambda env: (torch.tensor([True, False, True]), torch.zeros(3)))
    out = microduck_mdp.trunk_drop_penalty(env, max_down_vel=0.3, tiles=mt.DOWN_TILES, zone_max_down_vel=0.15)
    assert out.tolist() == pytest.approx([-0.10, 0.0, -0.10])
    out = microduck_mdp.trunk_drop_penalty(env, max_down_vel=0.3, tiles=mt.DOWN_TILES)   # no zone cap
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_trunk_drop_zone_only_charges_the_drop_at_the_edge_alone(monkeypatch):
    env = _env(_stairs_down([0.5], 0.03, 1), ["single_drop", "single_drop", "flat"], [0.03] * 3, torch.zeros(3, 2, 3))
    env.scene["robot"].data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, -0.5]] * 3)
    env.episode_length_buf = torch.full((3,), 50, dtype=torch.long)
    env.step_dt = 0.02
    monkeypatch.setattr(microduck_mdp, "down_edge_zone", lambda env: (torch.tensor([True, False, True]), torch.zeros(3)))
    kw = dict(max_down_vel=0.3, tiles=mt.DOWN_TILES, zone_max_down_vel=0.15)
    edge = microduck_mdp.trunk_drop_penalty(env, **kw, zone_only=True)
    full = microduck_mdp.trunk_drop_penalty(env, **kw)
    assert edge.tolist() == pytest.approx([-0.35, 0.0, 0.0])                      # off the zone / off the tiles: 0
    assert full.tolist() == pytest.approx([-0.35, -0.2, 0.0])
    env.episode_length_buf = torch.full((3,), 5, dtype=torch.long)                # still settling from the spawn
    assert microduck_mdp.trunk_drop_penalty(env, **kw, zone_only=True).tolist() == [0.0, 0.0, 0.0]


# ── the down edge: the zone and the freed terms ─────────────────────────────

def _robot_at(x: float, quat=None):
    env = _env(_stairs_down([0.5], 0.03, 1), ["single_drop", "short_flight"], [0.03, 0.0],
               torch.zeros(2, 2, 3), quat=quat)
    env.scene["robot"].data.root_link_pos_w[:, 0] = x
    env.common_step_counter = 0
    return env


@pytest.mark.parametrize("x,inside", [(0.30, False), (0.40, True), (0.49, True), (0.55, True), (0.65, False)])
def test_down_edge_zone_spans_behind_and_ahead(x, inside):
    zone, drop = microduck_mdp.down_edge_zone(_robot_at(x))
    assert zone.tolist() == [inside, inside]                                       # every column: callers gate tiles
    assert drop[0].item() == pytest.approx(0.03 if inside else 0.0, abs=1e-6)


def test_down_edge_zone_is_cached_per_step():
    env = _robot_at(0.40)
    assert microduck_mdp.down_edge_zone(env)[0].all()
    env.scene["robot"].data.root_link_pos_w[:, 0] = 0.30                          # moved, same step: cached
    assert microduck_mdp.down_edge_zone(env)[0].all()
    env.common_step_counter = 1
    assert not microduck_mdp.down_edge_zone(env)[0].any()


def test_nose_up_is_negative_projected_gravity_x():
    th = math.radians(20)
    nose_up = torch.tensor([[math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0]])      # about y by −20°
    g = quat_apply_inverse(nose_up, torch.tensor([[0.0, 0.0, -1.0]]))
    assert g[0, 0].item() == pytest.approx(-math.sin(th), abs=1e-6)


def _gravity_env(pitch_deg: list[float], zone: list[bool], monkeypatch):
    n = len(pitch_deg)
    g = torch.tensor([[math.sin(math.radians(p)), 0.0, -math.cos(math.radians(p))] for p in pitch_deg])  # + = nose down
    robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=g))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=n, device="cpu")
    monkeypatch.setattr(microduck_mdp, "_edge_zone_on", lambda env, tiles: torch.tensor(zone))
    return env


def test_upright_frees_only_a_backward_lean_and_only_in_the_zone(monkeypatch):
    std = 0.2236                                                                   # the walk recipe's upright std
    env = _gravity_env([-25, -35, 10, -25], [True, True, True, False], monkeypatch)
    out = microduck_mdp.upright_edge_lean(env, std=std, tiles=mt.DOWN_TILES, max_back_deg=30.0)
    s = lambda d: math.sin(math.radians(d))  # noqa: E731
    assert out[0].item() == pytest.approx(1.0)                                     # 25° back at the edge: free
    assert out[1].item() == pytest.approx(math.exp(-(s(35) - s(30)) ** 2 / std ** 2))  # beyond 30°: the excess
    assert out[2].item() == pytest.approx(math.exp(-s(10) ** 2 / std ** 2))        # forward: charged as before
    assert out[3].item() == pytest.approx(math.exp(-s(25) ** 2 / std ** 2))        # off the zone: stock upright
    assert out[3].item() < 0.1                                                     # (what runs 12-14 paid: ~3 %)


def _vel_env(cmd_vx: float, v: list[float], zone: list[bool], monkeypatch):
    n = len(v)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=torch.tensor([[x, 0.0, 0.0] for x in v])))
    cmd = torch.tensor([[cmd_vx, 0.0, 0.0]] * n)
    env = SimpleNamespace(scene={"robot": robot}, num_envs=n, device="cpu", step_dt=0.02,
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    monkeypatch.setattr(microduck_mdp, "_edge_zone_on", lambda env, tiles: torch.tensor(zone))
    return env


def test_velocity_targets_are_capped_at_the_edge(monkeypatch):
    env = _vel_env(0.3, [0.08, 0.08, 0.0], [True, False, True], monkeypatch)
    lin = microduck_mdp.track_linear_velocity_edge(env, std=0.316, command_name="twist", tiles=mt.DOWN_TILES, v_edge=0.08)
    assert lin[0].item() == pytest.approx(1.0)                                     # crouch pace at the edge: full pay
    assert lin[1].item() == pytest.approx(math.exp(-0.22 ** 2 / 0.316 ** 2))
    assert lin[2].item() < lin[0].item()                                           # standing at the edge still costs
    term = microduck_mdp.track_velocity_ema_edge(SimpleNamespace(params={"tau_s": 0.5}), env)
    for _ in range(300):
        out = term(env, std=0.15, command_name="twist", tiles=mt.DOWN_TILES, v_edge=0.08)
    assert out[0].item() == pytest.approx(1.0, abs=1e-4)
    assert out[1].item() == pytest.approx(math.exp(-0.22 ** 2 / 0.15 ** 2), rel=1e-3)
    assert out[2].item() == pytest.approx(math.exp(-0.08 ** 2 / 0.15 ** 2), rel=1e-3)   # parking: 0.75, not free


def test_posture_leaves_out_the_pitch_chain_in_the_zone(monkeypatch):
    names = ["left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"]
    default = torch.zeros(1, 4)
    asset = SimpleNamespace(find_joints=lambda expr: (list(range(4)), names),
                            data=SimpleNamespace(default_joint_pos=default.repeat(2, 1),
                                                 joint_pos=torch.tensor([[0.1, 0.6, 1.2, 0.5]] * 2)))
    stds = {".*hip_roll.*": 0.12, ".*hip_pitch.*": 0.4, ".*knee.*": 0.4, ".*ankle.*": 0.25}
    ac = SimpleNamespace(name="robot", joint_names=(".*",), joint_ids=slice(None))
    params = {"asset_cfg": ac, "std_standing": stds, "std_walking": stds, "std_running": stds,
              "command_name": "twist", "walking_threshold": 0.01, "running_threshold": 0.3,
              "tiles": mt.DOWN_TILES, "free_joints": r".*(hip_pitch|knee|ankle).*"}
    cmd = torch.tensor([[0.2, 0.0, 0.0]] * 2)
    env = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=2,
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    monkeypatch.setattr(microduck_mdp, "_edge_zone_on", lambda env, tiles: torch.tensor([True, False]))
    term = microduck_mdp.variable_posture_edge(SimpleNamespace(params=params), env)
    out = term(env, **params)
    assert out[0].item() == pytest.approx(math.exp(-(0.1 / 0.12) ** 2))             # only the hip roll counts
    e = [(0.1 / 0.12) ** 2, (0.6 / 0.4) ** 2, (1.2 / 0.4) ** 2, (0.5 / 0.25) ** 2]
    assert out[1].item() == pytest.approx(math.exp(-sum(e) / 4))                   # the stock mean
    assert out[1].item() < 0.05 < out[0].item()                                     # the crouch: ~free vs ~all lost


# ── the exact height field, the critic's scan, the edge flags ───────────────

def _step_field(edge_x=1.0, h=0.03, res=0.01):
    """A 2 m × 1 m field: ground 0 before ``edge_x``, ``h`` after it."""
    nx, ny = int(2.0 / res) + 1, int(1.0 / res) + 1
    grid = torch.zeros(nx, ny)
    grid[int(round(edge_x / res)):, :] = h
    return microduck_mdp.TerrainHeightField(0.0, -0.5, res, grid)


def test_height_field_lookup_and_outside_default():
    hf = _step_field()
    xy = torch.tensor([[0.5, 0.0], [0.99, 0.2], [1.01, -0.2], [1.8, 0.0], [5.0, 0.0], [0.5, 3.0]])
    assert hf.heights(xy).tolist() == pytest.approx([0.0, 0.0, 0.03, 0.03, 0.0, 0.0])


def test_scan_sees_a_step_ahead():
    """A trunk 0.3 m before a 3 cm step at nominal height: the far scan rows read −0.03."""
    hf = _step_field(edge_x=1.0, h=0.03)
    env = SimpleNamespace(device="cpu", _terrain_hf=hf)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.7, 0.0, 0.115]]),
                                                 root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
    env.scene = {"robot": robot}
    out = microduck_mdp.terrain_height_scan(env, **su.SCAN_PARAMS).reshape(19, 7)
    xs, ys = microduck_mdp.scan_grid(su.SCAN_PARAMS["x_range"], su.SCAN_PARAMS["y_range"], su.SCAN_PARAMS["res"])
    assert (len(xs), len(ys)) == (19, 7)
    for ix, x in enumerate(xs):
        expected = -0.03 if 0.7 + x >= 1.0 - 1e-6 else 0.0   # the riser cell belongs to the top
        assert out[ix].tolist() == pytest.approx([expected] * 7, abs=1e-6), x


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


def _foot_corners(toe_x: float, shrink: float = 0.0):
    aabb = torch.tensor([[0.0, 0.0, 0.0, 0.027, 0.02, 0.005]])                       # x length, z thickness
    signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=torch.float32)
    local = microduck_mdp._shrunk_sole(aabb, signs, shrink) if shrink else aabb[:, None, :3] + signs[None] * aabb[:, None, 3:]
    centre = torch.tensor([toe_x - 0.027, 0.0, 0.035])
    return (centre + local)[None]                                                   # (1, 1, 8, 3)


def test_grid_slop_flags_a_flat_foot_3mm_behind_the_nose_and_shrink_does_not():
    hf = _stairs_down([0.50], 0.03, 1)                     # a cell centre sits on the edge: the worst case
    found = torch.ones(1, 1, dtype=torch.bool)
    flat = microduck_mdp.foot_edge_flags(hf, _foot_corners(0.497), found)
    assert flat.item()                                                              # charged although on one level
    assert not microduck_mdp.foot_edge_flags(hf, _foot_corners(0.497, shrink=0.008), found).item()
    assert microduck_mdp.foot_edge_flags(hf, _foot_corners(0.515, shrink=0.008), found).item()   # 15 mm over: still
    shrunk = microduck_mdp._shrunk_sole(torch.tensor([[0.0, 0, 0, 0.006, 0.0217, 0.0272]]),
                                        torch.ones(1, 3), 0.008)
    assert shrunk[0, 0].tolist() == pytest.approx([0.006, 0.0137, 0.0192])           # thinnest axis kept


# ── the terrain ladder ──────────────────────────────────────────────────────

def _ladder_env(n, sums, **extra):
    calls = {}
    terrain = SimpleNamespace(update_env_origins=lambda ids, up, down: calls.update(up=up.clone(), down=down.clone()),
                              terrain_levels=torch.zeros(n, dtype=torch.long))
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        termination_manager=SimpleNamespace(get_term=lambda k: torch.zeros(n, dtype=torch.bool) if k == "fell_over" else torch.ones(n, dtype=torch.bool)),
        reward_manager=SimpleNamespace(_episode_sums={"track_linear_velocity": sums},
                                       get_term_cfg=lambda k: SimpleNamespace(weight=2.0)),
        **extra,
    )
    return env, calls


def test_ladder_promotes_only_clean_episodes():
    env, calls = _ladder_env(3, torch.full((3,), 0.9 * 2.0 * 20), max_episode_length_s=20.0,
                             _foot_edge_steps=torch.tensor([0.0, 3.0, 12.0]))
    microduck_mdp.terrain_levels_survival(env, torch.arange(3), min_tracking=0.55, max_edge_steps=3)
    assert calls["up"].tolist() == [True, True, False]       # 12 edge steps: timed out, but dirty
    assert calls["down"].tolist() == [False, False, False]


def test_ladder_progress_rule_and_its_exemption():
    """Stairs up's ladder demotes a robot that tracked its command but did not get anywhere; an env
    attribute (``progress_exempt_attr``, run 11a's stop latch) can exempt robots from that rule."""
    n = 3
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.3, 0, 0.1], [0.3, 0, 0.1], [2.0, 0, 0.1]])))
    env, calls = _ladder_env(n, torch.full((n,), 0.9 * 2.0 * 8), max_episode_length_s=8.0,
                             command_manager=SimpleNamespace(get_command=lambda k: torch.tensor([[0.25, 0.0, 0.0]]).repeat(n, 1)),
                             _exempt=torch.tensor([False, True, False]))
    terrain = env.scene.terrain
    env.scene = type("S", (), {"__getitem__": lambda self, k: robot, "terrain": terrain, "env_origins": torch.zeros(n, 3)})()
    microduck_mdp.terrain_levels_survival(env, torch.arange(n), min_tracking=0.55, min_progress=0.3,
                                          demote_no_progress=True, progress_exempt_attr="_exempt")
    # 0: 0.3 m of the 0.6 m needed → demoted; 1: same distance but exempt → promoted; 2: walked → promoted
    assert calls["up"].tolist() == [False, True, True] and calls["down"].tolist() == [True, False, False]
    microduck_mdp.terrain_levels_survival(env, torch.arange(n), min_tracking=0.55, min_progress=0.3,
                                          demote_no_progress=True)
    assert calls["up"].tolist() == [False, False, True] and calls["down"].tolist() == [True, True, False]


# ── the mid-step spawn (reset_at_down_edge) and the map prefill ─────────────

def _bank(tmp_path):
    joints = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
    poses = [{"h": 0.025, "trunk_x": -0.01, "trunk_z": 0.09, "pitch_deg": -20.0, "q": [0.1 * (i + 1) for i in range(10)]},
             {"h": 0.04, "trunk_x": -0.02, "trunk_z": 0.08, "pitch_deg": -25.0, "q": [0.2] * 10}]
    p = tmp_path / "bank.json"
    p.write_text(json.dumps({"joints": joints, "mirror_sign": [-1.0] * 10, "stance_sole_y": 0.04,
                             "foot_half_len": 0.027, "poses": poses}))
    return str(p), joints


class _SpawnAsset:
    def __init__(self, names, n):
        self.joint_names = names
        self.data = SimpleNamespace(joint_pos=torch.zeros(n, len(names)))
        self.root = self.vel = self.joints = None

    def find_joints(self, keys, preserve_order=False):
        import re
        if isinstance(keys, str):
            ids = [i for i, n in enumerate(self.joint_names) if re.match(keys, n)]
        else:
            ids = [self.joint_names.index(k) for k in keys]
        return ids, [self.joint_names[i] for i in ids]

    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.root = (pose.clone(), env_ids.clone())

    def write_root_link_velocity_to_sim(self, vel, env_ids):
        self.vel = vel.clone()

    def write_joint_state_to_sim(self, pos, vel, env_ids):
        self.joints = (pos.clone(), vel.clone(), env_ids.clone())


def _spawn_env(tmp_path, n=64):
    path, joints = _bank(tmp_path)
    names = joints + ["passive_left_knee_backlash", "neck_pitch"]
    hf = _stairs_down([0.9, 1.1], 0.03, 2)                                        # two 3 cm risers ahead of x 0
    # the boxes the event reads its exact edges from: approach, tread, bottom
    hf.boxes = np.array([[-0.5, 0.9, -1, 1, 0.06], [0.9, 1.1, -1, 1, 0.03], [1.1, 3.0, -1, 1, 0.0]])
    env = _env(hf, ["down_flight"] * (n - 1) + ["short_flight"], [0.06] * n, torch.zeros(n, 2, 3))
    asset = _SpawnAsset(names, n)
    env.scene = type("S", (), {"__getitem__": lambda self, k: asset if k == "robot" else None,
                               "env_origins": torch.tensor([[0.0, 0.0, 0.06]] * n),
                               "terrain": env.scene.terrain})()
    cfg = SimpleNamespace(params={"bank": path})
    return env, asset, microduck_mdp.reset_at_down_edge(cfg, env), path


def test_spawn_puts_the_rear_toe_at_an_edge_in_a_bank_pose(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    env, asset, ev, path = _spawn_env(tmp_path)
    ids = torch.arange(env.num_envs)
    ev(env, ids, bank=path, prob=1.0, tiles=mt.DOWN_TILES, gap=(0.005, 0.005), yaw_noise=0.0, lateral=0.0)
    pose, got = asset.root
    assert 63 not in got.tolist() and len(got) == 63                                # the uphill column: untouched
    assert env._edge_spawned[:63].all() and not env._edge_spawned[63]
    assert env._tof_prefill[:63].all()
    jp, jv, _ = asset.joints
    assert (jv == 0).all() and (asset.vel == 0).all()
    for k in range(len(got)):
        x, y, z = pose[k, :3].tolist()
        q = jp[k]
        left = q[0].item() > 0                                                     # bank q > 0: left stance
        # the edge: 0.9 (drop 3 cm: only the 2.5 cm pose fits, the 4 cm one is too tall) or 1.1 (same)
        sole_x = x - (-0.01)
        assert sole_x + 0.027 + 0.005 == pytest.approx(0.9, abs=1e-5) or sole_x + 0.032 == pytest.approx(1.1, abs=1e-5)
        top = 0.06 if sole_x < 0.9 else 0.03
        assert z == pytest.approx(top + 0.09 + 0.002, abs=1e-5)                     # the 2.5 cm pose (4 cm > 3 cm drop)
        assert y == pytest.approx(-0.04 if left else 0.04, abs=1e-6)                 # the trunk beside the stance sole
        if left:
            assert q[:10].tolist() == pytest.approx([0.1 * (i + 1) for i in range(10)])
        else:                                                                      # mirrored: legs swapped, negated
            assert q[:5].tolist() == pytest.approx([-0.1 * (i + 6) for i in range(5)])
            assert q[5:10].tolist() == pytest.approx([-0.1 * (i + 1) for i in range(5)])
        assert q[10].item() == 0.0                                                 # backlash joint zeroed
    w, xq, yq, zq = pose[0, 3:].tolist()
    assert (w, yq) == pytest.approx((math.cos(math.radians(-10)), math.sin(math.radians(-10))))   # 20° nose up
    lefts = (jp[:, 0] > 0).float().mean().item()
    assert 0.2 < lefts < 0.8                                                        # both stance sides


def test_spawn_probability_and_tile_edges(tmp_path):
    env, asset, ev, path = _spawn_env(tmp_path, n=400)
    ev(env, torch.arange(400), bank=path, prob=0.3, tiles=mt.DOWN_TILES)
    assert 0.2 < env._edge_spawned.float().mean().item() < 0.4
    edges = ev._tile_edges(env._terrain_hf, 0.0, 0.0, 0.06)
    assert [e[:3] for e in edges] == [pytest.approx((0.9, 0.06, 0.03)), pytest.approx((1.1, 0.03, 0.03))]
    assert edges[1][3] == pytest.approx(0.2)                                        # the upper tread's length


def test_spawn_skips_a_pose_that_would_leave_the_front_foot_hovering(tmp_path):
    env, asset, ev, path = _spawn_env(tmp_path, n=32)
    ev(env, torch.arange(32), bank=path, prob=1.0, tiles=mt.DOWN_TILES, max_hover=0.004)   # 2.5 cm pose, 3 cm drop
    assert not env._edge_spawned.any() and asset.root is None


def test_bank_file_covers_the_training_risers():
    bank = microduck_mdp._load_step_down_bank(su.EDGE_SPAWN_BANK)
    hs = sorted({round(p["h"], 3) for p in bank["poses"]})
    lo, hi = mt.STEP_UP_HEIGHT_RANGE
    assert hs[0] <= lo + 1e-6 and hs[-1] >= hi - 0.005                              # every training riser has a pose ≤ it
    assert bank["mirror_sign"] == [-1.0] * len(bank["joints"])
    assert 0.03 < bank["stance_sole_y"] < 0.05
    for p in bank["poses"]:
        assert 0.06 <= p["trunk_z"] <= 0.13 and -35 <= p["pitch_deg"] <= 35
        assert len(p["q"]) == len(bank["joints"])


def test_prefill_writes_the_exact_ground_into_the_map():
    m = TofHeightMap(2, "cpu")
    pos = torch.tensor([[1.0, 2.0, 0.15], [1.0, 2.0, 0.15]])
    quat = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    m.reset(torch.arange(2), pos, quat)
    ground = lambda xy: torch.where(xy[..., 0] >= 1.1, torch.zeros_like(xy[..., 0]), torch.full_like(xy[..., 0], 0.03))  # noqa: E731
    m.prefill(torch.tensor([0]), ground)
    offs = torch.tensor([[0.0, 0.0], [0.2, 0.0], [-0.3, 0.1], [0.8, 0.0]])
    vals, known = m.scan(offs, pos, quat, nominal=0.0)
    assert known[0].tolist() == [True, True, True, False]                          # beyond the 0.6 m window: unknown
    assert not known[1].any()                                                      # env 1 not prefilled
    # trunk z − ground (nominal 0): 3 cm tread behind x 1.1, floor ahead; one noisy reading per cell
    assert vals[0, :3].tolist() == pytest.approx([0.15 - 0.03, 0.15 - 0.0, 0.15 - 0.03], abs=0.015)


# ── warm start across observation widths (mdp Patch 6; how run 10 grew 61 → 327) ──

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


# ── the real env (CPU, a handful of robots) ─────────────────────────────────

@pytest.fixture(scope="module")
def stairs_env():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg(UP, play=True)
    cfg.scene.num_envs = 6
    cfg.scene.terrain.terrain_generator.seed = 0
    # Driven by constant actions, not a policy: a sagging HOME would trip fell_over and reset (and
    # rightly clear) the maps mid-test. Only the time-out may end an episode here.
    for name in [k for k in cfg.terminations if k != "time_out"]:
        cfg.terminations.pop(name)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_height_field_matches_raycasts_on_the_real_terrain(stairs_env):
    hf = microduck_mdp.terrain_height_field(stairs_env)
    mjm = stairs_env.sim.mj_model
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    origins = stairs_env.scene.terrain.terrain_origins.reshape(-1, 3).cpu().numpy()
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


def test_critic_scan_is_the_exact_ground(stairs_env):
    obs, _ = stairs_env.reset()
    names = list(stairs_env.observation_manager._group_obs_term_names["critic"])
    assert names[-1] == "height_scan_clean"
    scan = obs["critic"][:, -133:]
    assert scan.abs().max() <= su.SCAN_PARAMS["clip"] + 1e-6
    # robots on the flat column read flat ground around them (trunk bob only: the critic scan is noise-free)
    tnames = list(stairs_env.scene.terrain.cfg.terrain_generator.sub_terrains)
    types = stairs_env.scene.terrain.terrain_types.cpu().numpy()
    flat = [i for i in range(stairs_env.num_envs) if tnames[types[i]] == "flat"]
    assert flat
    for i in flat:
        assert scan[i].abs().max() < 0.04


def test_tofmap_obs_is_values_then_mask_and_fills_in(stairs_env):
    obs, _ = stairs_env.reset()
    assert obs["actor"].shape[1] == 61 + 2 * 133 == 327
    assert stairs_env.observation_manager.group_obs_dim["actor"] == (327,)
    # Head pitched down as the policies hold it (neck −, head +): with a level head the bottom ToF
    # row lands 0.72 m ahead, beyond the scan (≤ 0.45 m), and the scan would rightly stay unknown.
    action = torch.zeros(stairs_env.num_envs, 14)
    action[:, 5], action[:, 6] = -4.0, 4.0
    with torch.no_grad():
        for _ in range(40):                               # 0.8 s: ≈ 12 ToF frames
            obs, *_ = stairs_env.step(action)
    vals, mask = obs["actor"][:, 61:194], obs["actor"][:, 194:]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.all(vals[mask == 0] == 0)                # unknown reads "level with me"
    assert mask.mean() > 0.05                             # the map has filled in ahead of the robots
    assert vals.abs().max() <= su.SCAN_PARAMS["clip"] + 1e-6


def test_tofmap_mirror_flips_values_and_mask_and_is_an_involution(stairs_env):
    obs, _ = stairs_env.reset()
    obs_perm, obs_sign, _, _ = symmetry._get_tensors(torch.device("cpu"))
    layout = symmetry._actor_layout(stairs_env)
    assert layout == [(61, 266, ("grid", 19, 7, 2))]
    a = obs["actor"]
    m = symmetry.mirror_actor_obs(a, obs_perm, obs_sign, layout)
    assert torch.equal(m[:, 61:].reshape(-1, 2, 19, 7), a[:, 61:].reshape(-1, 2, 19, 7).flip(-1))
    assert torch.allclose(symmetry.mirror_actor_obs(m, obs_perm, obs_sign, layout), a)


def test_foot_edge_cost_and_placement_run(stairs_env):
    stairs_env.reset()
    with torch.no_grad():
        for _ in range(5):
            stairs_env.step(torch.zeros(stairs_env.num_envs, 14))
    rm = stairs_env.reward_manager
    edge = rm.get_term_cfg("foot_edge_contact").func
    assert isinstance(edge, microduck_mdp.foot_edge_contact_cost)
    assert stairs_env._foot_edge_steps.shape == (stairs_env.num_envs,)
    assert torch.all(stairs_env._foot_edge_steps >= 0)
    edge.reset(torch.tensor([0]))
    assert stairs_env._foot_edge_steps[0] == 0
    term = rm.get_term_cfg("foot_placement")
    val = term.func(stairs_env, **term.params)
    assert val.shape == (stairs_env.num_envs,) and bool((val >= 0).all()) and bool((val <= 2).all())
