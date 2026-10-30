"""Cfg invariants for Walk v2 on the duck-scaled terrain grid (microduck_terrains.py +
microduck_walk_v2_terrain_env_cfg.py). CPU only; builds terrain specs, never steps an env."""
import math

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_terrains as mt
from mjlab_microduck.tasks import microduck_walk_v2_env_cfg as w2
from mjlab_microduck.tasks import microduck_walk_v2_terrain_env_cfg as w2t

TASKS = (
    "Mjlab-WalkV2-Terrain-MicroDuck",
    "Mjlab-WalkV2-Stairs-MicroDuck",
    "Mjlab-WalkV2-Stairs-Finetune-MicroDuck",
    "Mjlab-WalkV2-Terrain-Backlash-MicroDuck",
    "Mjlab-WalkV2-Stairs-Backlash-MicroDuck",
    "Mjlab-WalkV2-Stairs-Finetune-Backlash-MicroDuck",
    "Mjlab-WalkV2-StairsV2-Finetune-MicroDuck",
    "Mjlab-WalkV2-StairsV2-Finetune-Backlash-MicroDuck",
)
EXPERIMENT = {"Terrain": "walk_v2_terrain", "Stairs": "walk_v2_stairs", "StairsV2": "walk_v2_stairs"}


def _terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


# ── straight stairs geometry ─────────────────────────────────────────────────

@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_straight_stairs_boxes_are_contiguous_and_spawn_is_on_the_start_platform(difficulty):
    cfg = mt.BoxStraightStairsTerrainCfg()
    cfg.size = (mt.PATCH_SIZE, mt.PATCH_SIZE)
    out = cfg.function(difficulty, _terrain_spec(), np.random.default_rng(0))
    n = cfg.num_steps
    assert len(out.geometries) == 2 * n + 3          # start, n up, top, n down, end
    boxes = [g.geom for g in out.geometries]
    # x-contiguous, tiling the whole tile length, full tile width.
    x = 0.0
    for b in boxes:
        assert abs((b.pos[0] - b.size[0]) - x) < 1e-9
        x = b.pos[0] + b.size[0]
        assert abs(2 * b.size[1] - cfg.size[1]) < 1e-9
        assert abs((b.pos[2] - b.size[2]) + cfg.thickness) < 1e-9   # common bottom
    assert abs(x - cfg.size[0]) < 1e-9
    tops = [b.pos[2] + b.size[2] for b in boxes]
    h = cfg.step_height(difficulty)
    expected = [0.0] + [k * h for k in range(1, n + 1)] + [n * h] + [k * h for k in range(n - 1, -1, -1)] + [0.0]
    assert np.allclose(tops, expected)
    # Spawn: centre of the start platform at ground level.
    assert np.allclose(out.origin, [cfg.start_length / 2, cfg.size[1] / 2, 0.0])


def test_straight_stairs_are_within_the_ducks_reach():
    cfg = mt.BoxStraightStairsTerrainCfg()
    assert cfg.step_width >= 2 * mt.FOOT_LENGTH                 # a foot lands on one level
    assert cfg.step_height(0.0) == pytest.approx(0.005)
    assert cfg.step_height(1.0) == pytest.approx(0.030)         # hostile-terrain validated rise
    assert cfg.step_height(2.0) == pytest.approx(0.030)         # clamped
    cfg.size = (mt.PATCH_SIZE, mt.PATCH_SIZE)
    assert cfg.size[0] - cfg.total_length() >= 0.3              # an end platform remains


def test_straight_stairs_refuse_a_tile_that_is_too_short():
    cfg = mt.BoxStraightStairsTerrainCfg()
    cfg.size = (2.0, 2.0)
    with pytest.raises(AssertionError):
        cfg.function(0.5, _terrain_spec(), np.random.default_rng(0))


def test_rubble_scales_with_difficulty_and_spawns_above_the_bumps():
    cfg = mt.microduck_subterrains()["rubble"]
    cfg.size = (mt.PATCH_SIZE, mt.PATCH_SIZE)
    easy = cfg.function(0.0, _terrain_spec(), np.random.default_rng(0))
    hard = cfg.function(1.0, _terrain_spec(), np.random.default_rng(0))
    assert easy.origin[2] == pytest.approx(0.004 + 0.002)
    assert hard.origin[2] == pytest.approx(0.04 + 0.002)
    assert cfg.noise_range == (-0.02, 0.02)                     # restored after the call


# ── menus compile ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make", [mt.make_microduck_terrains_cfg, mt.make_microduck_stairs_terrains_cfg])
def test_menu_compiles_as_a_curriculum_grid(make):
    gen_cfg = make(num_rows=2, seed=0)
    assert gen_cfg.curriculum and gen_cfg.size == (mt.PATCH_SIZE, mt.PATCH_SIZE)
    assert "straight_stairs" in gen_cfg.sub_terrains
    assert sum(st.proportion for st in gen_cfg.sub_terrains.values()) == pytest.approx(1.0)
    gen = TerrainGenerator(gen_cfg)
    spec = mujoco.MjSpec()
    gen.compile(spec)
    model = spec.compile()
    assert model.ngeom > 0
    origins = np.asarray(gen.terrain_origins)
    assert origins.shape == (2, len(gen_cfg.sub_terrains), 3)
    # No spawn origin below the deepest pit (inverted pyramid, 3 cm × ≤ 12 steps) or above
    # the highest platform.
    assert origins[..., 2].min() > -0.5 and origins[..., 2].max() < 0.5


# ── env cfg ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_tasks_registered(task):
    assert load_env_cfg(task) is not None and load_env_cfg(task, play=True) is not None
    assert load_rl_cfg(task).experiment_name == EXPERIMENT[task.split("-")[2]]


def test_backlash_variant_uses_walk_model():
    cfg = load_env_cfg("Mjlab-WalkV2-Terrain-Backlash-MicroDuck")
    assert cfg.scene.entities["robot"].spec_fn is MICRODUCK_WALK_BACKLASH_ROBOT_CFG.spec_fn


def test_train_cfg_starts_on_row_zero_with_the_full_ladder():
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg()
    gen = cfg.scene.terrain.terrain_generator
    assert cfg.scene.terrain.terrain_type == "generator" and gen.curriculum
    assert gen.num_rows == mt.NUM_ROWS
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert "terrain_levels" in cfg.curriculum                   # mjlab's promotion rule kept


def test_play_cfg_keeps_the_curriculum_layout_for_preview():
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg(play=True)
    gen = cfg.scene.terrain.terrain_generator
    assert gen.curriculum and gen.num_rows == w2t.PLAY_ROWS
    assert cfg.scene.terrain.max_init_terrain_level is None


def test_stairs_menu_is_stairs():
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg(menu="stairs")
    names = set(cfg.scene.terrain.terrain_generator.sub_terrains)
    assert names == {"flat", "straight_stairs", "pyramid_stairs", "pyramid_stairs_inv"}
    with pytest.raises(ValueError):
        w2t.make_microduck_walk_v2_terrain_env_cfg(menu="lava")


def test_spawn_fits_the_start_platform():
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg()
    pr = cfg.events["reset_base"].params["pose_range"]
    stairs = cfg.scene.terrain.terrain_generator.sub_terrains["straight_stairs"]
    assert pr["x"] == (-w2t.SPAWN_XY, w2t.SPAWN_XY) and pr["y"] == (-w2t.SPAWN_XY, w2t.SPAWN_XY)
    assert w2t.SPAWN_XY + mt.FOOT_LENGTH < stairs.start_length / 2   # feet stay on the platform
    assert pr["z"] == (0.12, 0.13)                                    # recipe's absolute trunk z


def test_only_terrain_spawn_and_foot_targets_differ_from_walk_v2():
    base = w2.make_microduck_walk_v2_env_cfg(rough=True)
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg()
    assert set(cfg.rewards) == set(base.rewards)
    for name in base.rewards:
        assert cfg.rewards[name].func is base.rewards[name].func, name
        assert cfg.rewards[name].weight == base.rewards[name].weight, name
    assert cfg.rewards["foot_clearance"].params["target_height"] == w2t.FOOT_TARGET_HEIGHT
    assert cfg.rewards["foot_swing_height"].params["target_height"] == w2t.FOOT_TARGET_HEIGHT
    assert cfg.rewards["pose"].params["std_walking"][r".*hip_roll.*"] == w2t.HIP_ROLL_STD
    assert cfg.rewards["pose"].params["std_standing"] == base.rewards["pose"].params["std_standing"]
    # 61D actor contract untouched; rough-terrain solver settings inherited.
    assert cfg.observations["actor"].terms.keys() == base.observations["actor"].terms.keys()
    assert cfg.sim.nconmax == base.sim.nconmax == 200
    assert cfg.scene.spec_fn is base.scene.spec_fn
    assert set(cfg.curriculum) == set(base.curriculum)


# ── finetune (warm start) variant ────────────────────────────────────────────

def _stage_lists(cfg):
    return {(name, key): val for name, term in cfg.curriculum.items() for key, val in term.params.items()
            if isinstance(val, list) and val and all(isinstance(v, dict) and "step" in v for v in val)}


def test_finetune_pins_inherited_curricula_final_and_keeps_the_terrain_ladder():
    base = w2t.make_microduck_walk_v2_terrain_env_cfg(menu="stairs")
    ft = w2t.make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True)
    base_stages, ft_stages = _stage_lists(base), _stage_lists(ft)
    assert base_stages and set(ft_stages) == set(base_stages)
    for key, stages in ft_stages.items():
        assert len(stages) == 1 and stages[0]["step"] == 0
        assert {k: v for k, v in stages[0].items() if k != "step"} == {k: v for k, v in base_stages[key][-1].items() if k != "step"}
    # reward_weight terms already run at their final weight from step 0
    for term in ft.curriculum.values():
        if "weight_stages" in term.params and term.params.get("reward_name") in ft.rewards:
            assert ft.rewards[term.params["reward_name"]].weight == term.params["weight_stages"][0]["weight"]
    assert "vx_range" not in ft.curriculum and "vx_range" in base.curriculum
    assert ft.commands["twist"].ranges.lin_vel_x == (-w2.VX_FINAL_MAX, w2.VX_FINAL_MAX)
    assert "terrain_levels" in ft.curriculum and ft.scene.terrain.max_init_terrain_level == 0
    assert ft.observations["actor"].terms.keys() == base.observations["actor"].terms.keys()


def test_finetune_play_cfg_is_the_plain_play_cfg():
    assert _stage_lists(w2t.make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs", finetune=True)) == \
        _stage_lists(w2t.make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs"))


def test_finetune_rl_cfg():
    rl = load_rl_cfg("Mjlab-WalkV2-Stairs-Finetune-Backlash-MicroDuck")
    assert rl.experiment_name == "walk_v2_stairs" and rl.run_name == "walk_v2_stairs_ft"
    assert rl.max_iterations == w2t.FINETUNE_MAX_ITERATIONS


# ── terrain ladder starts where the cfg says ─────────────────────────────────

def test_terrain_ladder_guard_is_wired():
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg(menu="stairs")
    assert cfg.curriculum["terrain_levels"].func is microduck_mdp.terrain_levels_first_reset_guarded


def test_first_reset_does_not_promote_anyone():
    """Builds a small CPU env: before the guard, mjlab's rule promoted every robot to row 1 at the
    initial reset (robots start at the world origin, 'metres walked' from their tile)."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg(menu="stairs")
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        levels = env.scene.terrain.terrain_levels
        assert int(levels.max()) == 0, levels.tolist()
        assert env.curriculum_manager._curriculum_state["terrain_levels"]["max"] == 0
        # The real rule is live after the first step: a fallen robot is demoted, never promoted.
        with torch.no_grad():
            for _ in range(3):
                env.step(torch.zeros(8, 14))
        assert int(env.scene.terrain.terrain_levels.max()) == 0
    finally:
        env.close()


# ── v2: survival ladder, spawn anywhere, wide apron ───────────────────────────
from types import SimpleNamespace


def _fake_env(fell, timed_out, tracking_fraction, weight=2.0, T=20.0):
    n = len(fell)
    calls = {}

    def update_env_origins(env_ids, move_up, move_down):
        calls["up"], calls["down"] = move_up.clone(), move_down.clone()

    terrain = SimpleNamespace(update_env_origins=update_env_origins, terrain_levels=torch.zeros(n, dtype=torch.long))
    term = SimpleNamespace(get_term=lambda name: {"fell_over": torch.tensor(fell), "time_out": torch.tensor(timed_out)}[name])
    sums = torch.tensor(tracking_fraction) * weight * T
    rew = SimpleNamespace(_episode_sums={"track_linear_velocity": sums}, get_term_cfg=lambda n: SimpleNamespace(weight=weight))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), termination_manager=term, reward_manager=rew, max_episode_length_s=T)
    return env, calls


def _with_robot(env, positions, commands):
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor(positions)))
    n = len(positions)
    env.scene = type("Scene", (), {"terrain": env.scene.terrain, "env_origins": torch.zeros(n, 3), "__getitem__": lambda self, k: robot})()
    env.command_manager = SimpleNamespace(get_command=lambda name: torch.tensor(commands))
    return env


def test_survival_ladder_rule():
    #            fell   timeout  tracking
    fell =      [True,  False,   False, False]
    timed_out = [False, True,    True,  False]
    tracking =  [0.9,   0.8,     0.3,   0.9]   # 0 fell → down; 1 good → up; 2 stood still → stay; 3 other end → stay
    env, calls = _fake_env(fell, timed_out, tracking)
    out = microduck_mdp.terrain_levels_survival(env, torch.arange(4), min_tracking=0.55)
    assert calls["up"].tolist() == [False, True, False, False]
    assert calls["down"].tolist() == [True, False, False, False]
    assert out["promoted"].item() == pytest.approx(0.25) and out["demoted"].item() == pytest.approx(0.25)


def test_survival_ladder_progress_blocks_circling_and_demotes_no_progress():
    env, calls = _fake_env([False, False, False], [True, True, True], [0.9, 0.9, 0.9])
    # 0 walked 3 m (needs 0.3·0.3·20 = 1.8 m), 1 circled (0.2 m), 2 was told to stand
    env = _with_robot(env, [[3.0, 0.0, 0.1], [0.2, 0.0, 0.1], [0.1, 0.0, 0.1]],
                      [[0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    microduck_mdp.terrain_levels_survival(env, torch.arange(3), min_tracking=0.55, min_progress=0.3, demote_no_progress=True)
    assert calls["up"].tolist() == [True, False, True]
    assert calls["down"].tolist() == [False, True, False]


def test_survival_ladder_first_reset_moves_nobody():
    env, calls = _fake_env([False, False], [False, False], [0.0, 0.0])
    env = _with_robot(env, [[5.0, 0.0, 0.1], [5.0, 0.0, 0.1]], [[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
    microduck_mdp.terrain_levels_survival(env, torch.arange(2), min_tracking=0.55, min_progress=0.3, demote_no_progress=True)
    assert not calls["up"].any() and not calls["down"].any()


def test_v2_cfg():
    cfg = w2t.make_microduck_walk_v2_terrain_env_cfg(menu="stairs", finetune=True, v2=True)
    gen = cfg.scene.terrain.terrain_generator
    assert gen.seed == w2t.V2_TERRAIN_SEED and gen.border_width == w2t.V2_BORDER_WIDTH and gen.num_rows == mt.NUM_ROWS
    assert cfg.curriculum["terrain_levels"].func is microduck_mdp.terrain_levels_survival
    assert cfg.curriculum["terrain_levels"].params["demote_no_progress"] is True
    assert cfg.events["reset_base"].func is microduck_mdp.reset_root_on_terrain
    assert cfg.events["reset_base"].params["flat_tol"] == 0.02 and cfg.events["reset_base"].params["res"] == 0.04
    assert cfg.terminations["out_of_terrain_bounds"].params["margin"] == w2t.V2_BOUNDS_MARGIN
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert cfg.rewards["track_linear_velocity"].params["std"] == pytest.approx(math.sqrt(0.1))  # the 0.55 bar's curve
    # play cfg is the plain stairs play cfg (platform spawns, stock bounds)
    play = w2t.make_microduck_walk_v2_terrain_env_cfg(play=True, menu="stairs", finetune=True, v2=True)
    assert play.events["reset_base"].func is not microduck_mdp.reset_root_on_terrain
    assert "margin" not in play.terminations["out_of_terrain_bounds"].params


def test_height_lookup_on_the_stairs():
    """Compile a 2-row stairs grid + robot on CPU, ray-cast the lookup: flat is all usable, the
    straight stairs' start platform is at 0 and usable, and on the top row (2.4-3 cm steps) the
    step edges are excluded while tread middles stay usable."""
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML
    gcfg = mt.make_microduck_stairs_terrains_cfg(num_rows=2, seed=1)
    gcfg.difficulty_range = (0.0, 1.0)
    spec = mujoco.MjSpec()
    gen = TerrainGenerator(gcfg)
    gen.compile(spec)
    spec.attach(mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML)), prefix="", frame=spec.worldbody.add_frame())
    model = spec.compile()
    centers = np.asarray(gen.terrain_origins).reshape(-1, 3)[:, :2]
    heights, allowed, offsets = microduck_mdp.build_terrain_height_lookup(model, centers, span=1.7, res=0.04, flat_tol=0.02)
    names = list(gcfg.sub_terrains)
    n = len(offsets); c = n // 2
    origins = np.asarray(gen.terrain_origins)
    for row in range(2):
        for col, name in enumerate(names):
            p = row * len(names) + col
            assert abs(heights[p, c, c] - origins[row, col, 2]) < 0.035, (name, row, heights[p, c, c], origins[row, col, 2])
            if name == "flat":
                assert np.allclose(heights[p], 0.0, atol=1e-3) and allowed[p].all()
            if name == "straight_stairs":
                st = gcfg.sub_terrains[name]
                top = heights[p].max()
                assert top > 0.0 and top <= st.num_steps * st.step_height(1.0) + 1e-3
                if row == 1:
                    assert 0.3 < allowed[p].mean() < 0.95     # edges of 2.4-3 cm steps excluded
    assert 0.3 < allowed.mean() <= 1.0
