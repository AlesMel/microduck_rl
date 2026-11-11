"""Stairs down (run 18d, tasks/microduck_stairs_down_env_cfg.py; docs/stairs.md, history in
docs/archive/stairs_runlog.md §26-28): the cfg against stairs up's (same obs, its rewards + three,
down tiles only, the spawn chain), the spawn banks, the down-specific terms on a fake env, the
upright gate below a downhill origin, and the edge-contact classification of
scripts/eval/stairs_contacts.py.

Ported 2026-11-11 from the run-18 teacher tests and the old contacts-battery tests."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.robot import microduck_constants as mc
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_stairs_down_env_cfg as sd
from mjlab_microduck.tasks import microduck_stairs_up_env_cfg as su

from test_stairs_up_cfg import _env, _stairs_down

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "eval"))
import stairs_contacts as sdc  # noqa: E402

UP, UP_BL = "Mjlab-StairsUp-MicroDuck", "Mjlab-StairsUp-Backlash-MicroDuck"
DOWN, DOWN_BL = "Mjlab-StairsDown-MicroDuck", "Mjlab-StairsDown-Backlash-MicroDuck"
CHAIN = ("reset_edge_stand", "reset_edge_crouch", "reset_at_down_edge")
NEW_TERMS = {"trunk_level", "trunk_level_l1", "drop_free_step_down"}


# ── the cfg ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("up_id,down_id", [(UP, DOWN), (UP_BL, DOWN_BL)])
@pytest.mark.parametrize("play", [False, True])
def test_down_is_stairs_up_with_the_same_obs(up_id, down_id, play):
    up, cfg = load_env_cfg(up_id, play=play), load_env_cfg(down_id, play=play)
    for g in ("actor", "critic"):                                      # same 327-D actor: swaps with stairs up per step
        assert list(cfg.observations[g].terms) == list(up.observations[g].terms)
    assert cfg.observations["actor"].terms["height_scan"].params == up.observations["actor"].terms["height_scan"].params
    assert cfg.scene.entities["robot"].spec_fn is up.scene.entities["robot"].spec_fn
    assert cfg.episode_length_s == sd.EPISODE_LENGTH_S


def test_robot_models():
    assert load_env_cfg(DOWN).scene.entities["robot"].spec_fn is mc.get_walk_legcol_spec
    assert load_env_cfg(DOWN_BL).scene.entities["robot"].spec_fn is mc.get_walk_backlash_legcol_spec


@pytest.mark.parametrize("play", [False, True])
def test_terrain_is_down_tiles_only_every_row_from_the_start(play):
    up, cfg = load_env_cfg(UP_BL, play=play), load_env_cfg(DOWN_BL, play=play)
    tg = cfg.scene.terrain.terrain_generator
    assert list(tg.sub_terrains) == list(sd.TILES) and tg.num_cols == sd.NUM_COLS
    assert tg.size == (sd.TILE_X, up.scene.terrain.terrain_generator.size[1])
    assert all(t.down and tuple(t.step_height_range) == sd.RISERS and tuple(t.approach_range) == sd.APPROACH
               for t in tg.sub_terrains.values())
    assert tg.sub_terrains["single_drop"].num_steps == 1
    f = tg.sub_terrains["down_flight"]
    assert f.num_steps == sd.FLIGHT_STEPS == 20 and tuple(f.step_width_range) == su.TREAD_DR_RANGE
    assert tg.sub_terrains["single_drop"].proportion == sd.SINGLE_DROP_SHARE
    assert f.proportion == pytest.approx(1.0 - sd.SINGLE_DROP_SHARE)
    # the whole flight fits its tile with a bottom the robot cannot walk off in one episode
    bottom = sd.TILE_X - (sd.APPROACH[1] + (sd.FLIGHT_STEPS - 1) * su.TREAD_DR_RANGE[1])
    assert bottom >= max(1.0, sd.TWIST["lin_vel_x"][1] * sd.EPISODE_LENGTH_S)
    assert cfg.scene.terrain.max_init_terrain_level is None
    assert "terrain_levels" not in cfg.curriculum                                # no promotion curriculum
    assert set(cfg.curriculum) == set(up.curriculum) - {"terrain_levels"}


def test_commands_are_forward_with_a_heading():
    cfg = load_env_cfg(DOWN_BL, play=False)
    tw = cfg.commands["twist"]
    for k, v in sd.TWIST.items():
        assert tuple(getattr(tw.ranges, k)) == v, k
    assert min(tw.ranges.lin_vel_x) > 0                                          # always forward, toward the edge
    assert tw.heading_command and tw.rel_heading_envs == 1.0
    assert tuple(tw.ranges.heading) == sd.HEADING


@pytest.mark.parametrize("play", [False, True])
def test_rewards_are_stairs_ups_plus_three(play):
    up, cfg = load_env_cfg(UP_BL, play=play), load_env_cfg(DOWN_BL, play=play)
    assert set(cfg.rewards) == set(up.rewards) | NEW_TERMS
    for k, r in up.rewards.items():
        assert cfg.rewards[k].func is r.func and cfg.rewards[k].weight == r.weight and cfg.rewards[k].params == r.params, k
    names = list(cfg.rewards)
    assert names.index("trunk_level") < names.index("trunk_level_l1")            # the L1 reads the tracker's error
    t = cfg.rewards["trunk_level"]
    assert t.func is microduck_mdp.trunk_ground_tracking and t.weight == sd.LEVEL_WEIGHT > 0
    assert t.params == {"std": sd.LEVEL_STD, "rate": sd.LEVEL_RATE, "stand_h": sd.STAND_H,
                        "look": sd.LOOK, "tiles": sd.TILES}
    l1 = cfg.rewards["trunk_level_l1"]
    assert l1.func is microduck_mdp.trunk_level_l1 and l1.weight == sd.LEVEL_L1_WEIGHT > 0        # self-negating
    d = cfg.rewards["drop_free_step_down"]
    assert d.func is microduck_mdp.drop_free_level_bonus and d.weight == sd.DROP_FREE_WEIGHT > 0
    c = up.rewards["clean_step_down"].params
    assert d.params == {"sensor_name": c["sensor_name"], "tiles": sd.TILES, "tol": c["tol"],
                        "shrink": su.EDGE_SHRINK, "v_gate": sd.DROP_GATE_V}


@pytest.mark.parametrize("play", [False, True])
def test_spawn_chain(play):
    cfg = load_env_cfg(DOWN_BL, play=play)
    ev = list(cfg.events)
    i = ev.index("reset_robot_joints")
    assert tuple(ev[i + 1:i + 4]) == CHAIN                                       # after the joints, in order
    assert ev.index("reset_base") < i and ev.count("reset_at_down_edge") == 1
    p = {k: cfg.events[k].params for k in CHAIN}
    assert all(cfg.events[k].func is microduck_mdp.reset_at_down_edge for k in CHAIN)
    assert [p[k]["clear"] for k in CHAIN] == [True, False, False]                # only the first clears
    assert [p[k]["bank"] for k in CHAIN] == [sd.STAND_BANK, sd.CROUCH_BANK, su.EDGE_SPAWN_BANK]
    assert p["reset_edge_stand"]["prob"] == 1.0 - sd.ORIGIN_SHARE
    want = (0.0, 0.0) if play else (sd.CROUCH_PROB, sd.MIDSTEP_PROB)
    assert (p["reset_edge_crouch"]["prob"], p["reset_at_down_edge"]["prob"]) == want
    assert p["reset_edge_stand"]["gap"] == sd.STAND_GAP and p["reset_edge_crouch"]["gap"] == sd.CROUCH_GAP
    assert p["reset_edge_crouch"]["max_hover"] == sd.CROUCH_MATCH
    assert all(p[k]["tiles"] == sd.TILES for k in CHAIN)


@pytest.mark.parametrize("task", (DOWN, DOWN_BL))
def test_runner(task):
    rl, up = load_rl_cfg(task), load_rl_cfg(task.replace("StairsDown", "StairsUp"))
    assert rl.experiment_name == "stairs_down" and rl.run_name == "stairs_down"
    assert rl.max_iterations == 1000
    assert rl.algorithm == up.algorithm and rl.actor == up.actor                # same network: one switch, two weights


def test_spawn_banks():
    for name, lo, hi in ((sd.STAND_BANK, 0.12, 0.126), (sd.CROUCH_BANK, 0.06, 0.111)):
        b = json.load(open(Path(microduck_mdp.__file__).parent / "data" / name))
        assert b["joints"] == [f"{s}_{j}" for s in ("left", "right")
                               for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
        assert len(b["poses"]) >= 3
        for q in b["poses"]:
            assert lo <= q["trunk_z"] <= hi
            assert len(q["q"]) == 10
            assert q["h"] == pytest.approx(0.0 if name == sd.STAND_BANK else 0.12 - q["trunk_z"], abs=1e-4)
    crouch = json.load(open(Path(microduck_mdp.__file__).parent / "data" / sd.CROUCH_BANK))
    hs = sorted(q["h"] for q in crouch["poses"])
    lo, hi = sd.RISERS
    for r in (lo, (lo + hi) / 2, hi):                                            # every riser has a crouch in range
        assert any(r - sd.CROUCH_MATCH <= h <= r for h in hs), r


# ── the terms on a fake env: a 3-riser down flight ──────────────────────────

def _flight(n: int = 1):
    """A 3-riser down flight (3 cm, edges at x 0.5 / 0.7 / 0.9, platform z 0.09); feet on the platform."""
    hf = _stairs_down([0.5, 0.7, 0.9], 0.03, 3)
    feet = torch.tensor([[[0.2, 0.05, 0.095], [0.2, -0.05, 0.095]]]).repeat(n, 1, 1)
    env = _env(hf, ["down_flight"] * n, [0.09] * n, feet)
    env.scene.terrain.cfg.terrain_generator.sub_terrains["down_flight"] = SimpleNamespace(num_steps=3, down=True)
    env.step_dt = 0.02
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env.scene["robot"].data.root_link_lin_vel_w = torch.zeros(n, 3)
    return env


def _feet_at(env, x0: float, x1: float, level: float):
    env.sim.data.geom_xpos[:, 0, 0], env.sim.data.geom_xpos[:, 1, 0] = x0, x1
    env.sim.data.geom_xpos[:, :, 2] = level + 0.005


def _trunk_x(env, x):
    env.scene["robot"].data.root_link_pos_w[:, 0] = x


def test_goal_is_12cm_minus_a_riser_over_the_higher_foots_tread_near_an_edge():
    env = _flight()
    term = microduck_mdp.trunk_ground_tracking(SimpleNamespace(params={}), env)
    goal = lambda: term.goal(env, 0.12, 0.14)[0][0].item()  # noqa: E731
    assert microduck_mdp._down_tile_riser(env).item() == pytest.approx(0.03)
    _trunk_x(env, 0.2)
    assert goal() == pytest.approx(0.21)                     # no edge within 14 cm: stand on the platform
    _trunk_x(env, 0.4)
    assert goal() == pytest.approx(0.18)                     # edge ahead: 12 cm − riser over the platform
    _feet_at(env, 0.6, 0.45, 0.06)                           # leading foot down, trailing still up
    _trunk_x(env, 0.52)
    assert goal() == pytest.approx(0.18)                     # = 12 cm over the lower tread
    _feet_at(env, 0.6, 0.6, 0.06)
    _trunk_x(env, 0.6)
    assert goal() == pytest.approx(0.15)                     # both down, the next edge within 14 cm: crouch again
    _feet_at(env, 1.3, 1.3, 0.0)
    _trunk_x(env, 1.3)
    assert goal() == pytest.approx(0.12)                     # the bottom: stand
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.12
    assert term(env, rate=0.1, tiles=("down_flight",)).item() == pytest.approx(1.0)
    assert term(env, rate=0.1, tiles=("single_drop",)).item() == 0.0    # off the tiles: nothing


def test_the_target_slews_and_the_l1_reads_its_error():
    env = _flight()
    term = microduck_mdp.trunk_ground_tracking(SimpleNamespace(params={}), env)
    kw = dict(std=0.025, rate=0.1, stand_h=0.12, look=0.14, tiles=("down_flight",))
    _trunk_x(env, 1.3)                                       # over the bottom: the goal is 12 cm
    _feet_at(env, 1.3, 1.3, 0.0)
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.20
    first = term(env, **kw).item()                           # fresh: the target starts where the trunk is,
    assert term.target.item() == pytest.approx(0.20 - 0.1 * 0.02)   # then moves ≤ rate·dt toward the goal
    assert first == pytest.approx(float(np.exp(-(0.002 / 0.025) ** 2)), rel=1e-4)
    for _ in range(9):
        term(env, **kw)
    assert term.target.item() == pytest.approx(0.18, abs=1e-6)
    assert env._trunk_level_err.item() == pytest.approx(0.02, abs=1e-6)
    assert microduck_mdp.trunk_level_l1(env).item() == pytest.approx(-0.02, abs=1e-6)   # ≤ 0 → positive weight
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.12     # a fast drop to the goal: ahead of the slew
    fast = term(env, **kw).item()                            # scores like behind it (no jackpot)
    assert fast == pytest.approx(float(np.exp(-(0.058 / 0.025) ** 2)), rel=1e-3)
    for _ in range(100):
        term(env, **kw)
    assert term.target.item() == pytest.approx(0.12)         # the target never passes the goal
    term.reset(torch.tensor([0]))
    env.scene["robot"].data.root_link_pos_w[:, 2] = 0.16
    assert term(env, **kw).item() == pytest.approx(float(np.exp(-(0.002 / 0.025) ** 2)), rel=1e-4)


def test_drop_free_level_pays_every_riser_and_a_drop_forfeits_only_its_riser():
    env = _flight()
    term = microduck_mdp.drop_free_level_bonus(SimpleNamespace(params={}), env)
    kw = dict(sensor_name="feet", tiles=("down_flight",), v_gate=0.25)
    assert term(env, **kw).item() == 0.0                     # on the platform (start level)
    _feet_at(env, 0.6, 0.45, 0.06)                           # feet on two levels: not yet
    assert term(env, **kw).item() == 0.0
    _feet_at(env, 0.6, 0.6, 0.06)
    assert term(env, **kw).item() == 1.0                     # first riser, drop-free
    assert term(env, **kw).item() == 0.0                     # once per level
    env.episode_length_buf[:] = 20
    env.scene["robot"].data.root_link_lin_vel_w[:, 2] = -0.3
    term(env, **kw)                                          # a drop on the way to the second riser
    env.scene["robot"].data.root_link_lin_vel_w[:, 2] = 0.0
    _feet_at(env, 0.8, 0.8, 0.03)
    assert term(env, **kw).item() == 0.0                     # forfeited
    _feet_at(env, 1.2, 1.2, 0.0)
    assert term(env, **kw).item() == 1.0                     # the latch cleared at the second level
    _feet_at(env, 0.6, 0.6, 0.06)                            # back up and down again: no new low
    term(env, **kw)
    _feet_at(env, 1.2, 1.2, 0.0)
    assert term(env, **kw).item() == 0.0
    assert term.paid.item() == 2.0
    term.reset(torch.tensor([0]))
    env._edge_upper_z = torch.tensor([0.06])                 # spawned at the second edge (tread 6 cm)
    _feet_at(env, 0.6, 0.6, 0.06)
    assert term(env, **kw).item() == 0.0                     # its own tread is the start
    _feet_at(env, 0.8, 0.8, 0.03)
    assert term(env, **kw).item() == 1.0


def test_drop_free_level_ignores_the_spawn_settling():
    env = _flight()
    term = microduck_mdp.drop_free_level_bonus(SimpleNamespace(params={}), env)
    kw = dict(sensor_name="feet", tiles=("down_flight",), v_gate=0.25, settle_s=0.2)
    env.scene["robot"].data.root_link_lin_vel_w[:, 2] = -0.5    # within the first 0.2 s: ignored
    term(env, **kw)
    assert not term.dropped[0]
    env.scene["robot"].data.root_link_lin_vel_w[:, 2] = 0.0
    _feet_at(env, 0.6, 0.6, 0.06)
    assert term(env, **kw).item() == 1.0


def test_upright_gate_ignores_height_below_a_downhill_origin():
    """§28: the env origin of a down tile is its TOP platform; an upright robot 20 cm below it is
    walking down the flight, not fallen. A positive height gate (the recovery terms) still applies."""
    up, tilted = [1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]          # 0° and 90° tilt
    asset = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.40], [0.0, 0.0, 0.40], [0.0, 0.0, 0.62]]),
        root_link_quat_w=torch.tensor([up, tilted, up])))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.60]] * 3))))
    assert microduck_mdp._fallen_mask(env, asset, 0.0, 40.0).tolist() == [0.0, 1.0, 0.0]
    assert microduck_mdp._fallen_mask(env, asset, 0.05, 40.0).tolist() == [1.0, 1.0, 1.0]


# ── scripts/eval/stairs_contacts.py: classifying descent edge contacts (no sim) ──

EDGE, TOP, LOW, HALF = 0.5, 0.03, 0.0, 0.027        # one 3 cm drop at x = 0.5, a 54 mm foot


def _traj(T: int = 40) -> dict:
    R = {k: np.zeros((T, 1, 2)) for k in ("found", "edgeflag", "heel", "toe", "cx", "sole", "pitch", "fz")}
    R["found"] = R["found"].astype(bool)
    R["edgeflag"] = R["edgeflag"].astype(bool)

    def put(f, ks, cx, sole, contact=True, flag=False, pitch=0.0):
        for k in ks:
            R["cx"][k, 0, f], R["heel"][k, 0, f], R["toe"][k, 0, f] = cx, cx - HALF, cx + HALF
            R["sole"][k, 0, f], R["found"][k, 0, f], R["edgeflag"][k, 0, f] = sole, contact, flag
            R["pitch"][k, 0, f], R["fz"][k, 0, f] = pitch, 5.0 if contact else 0.0
    # left (0), LEADING: stands on top, swings, lands ACROSS the edge (heel 1 cm short), settles below
    put(0, range(0, 10), 0.45, TOP)
    put(0, range(10, 15), 0.48, TOP + 0.02, contact=False)
    put(0, range(15, 19), 0.517, TOP - 0.01, flag=True, pitch=-10.0)
    put(0, range(19, 40), 0.537, LOW)
    # right (1), TRAILING: stands clean on top (toe 1 cm behind the edge), then slides / rolls over
    # the nose while loaded, swings, lands wholly below
    put(1, range(0, 21), 0.463, TOP)
    put(1, range(21, 25), 0.478, TOP - 0.005, flag=True, pitch=-15.0)
    put(1, range(25, 29), 0.52, TOP + 0.01, contact=False)
    put(1, range(29, 40), 0.557, LOW)
    return R


def test_touchdowns_merge_contact_flicker():
    c = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert sdc.touchdowns(c, swing=2) == [(0, 4), (8, 9)]     # a 1-step gap is flicker, 3 steps a swing


def test_classify_landing_vs_stance_and_roles():
    out = sdc.classify(_traj(), 0, np.array([EDGE]), np.array([LOW]), TOP, end=40)
    ev = {e["foot"]: e for e in out["events"]}
    assert len(out["events"]) == 2
    lead, trail = ev[0], ev[1]
    assert (lead["kind"], lead["role"]) == ("landing", "leading")
    assert (trail["kind"], trail["role"]) == ("stance", "trailing")
    assert trail["td_toe_mm"] == pytest.approx(-10.0)                # it had touched down wholly behind
    assert trail["slide_mm"] == pytest.approx(15.0)                  # then moved 15 mm while loaded
    assert trail["dpitch_deg"] == pytest.approx(-15.0)               # pitching toe down over the nose
    assert lead["reach_mm"] == pytest.approx(10.0) and lead["steps"] == 4
    cr = {c["foot"]: c for c in out["crossings"]}
    assert (cr[0]["role"], cr[1]["role"]) == ("leading", "trailing")
    assert cr[0]["left_behind_mm"] == pytest.approx(23.0)            # lifted off 23 mm behind the edge
    assert cr[0]["landed_beyond_mm"] == pytest.approx(-10.0)         # ... and came down short of clear
    assert cr[1]["left_behind_mm"] == pytest.approx(-5.0)            # left from across the nose
    assert cr[1]["landed_beyond_mm"] == pytest.approx(30.0)
    assert len(out["pitch"]["edge"]) == 8                           # 4 + 4 loaded edge-contact steps
    assert all(p == 0.0 for p in out["pitch"]["flat"])


def test_summarize_shares():
    per = [{"robot": 0, "riser_cm": 3.0, "outcome": "top", "descended": 1, "nan": False,
            **sdc.classify(_traj(), 0, np.array([EDGE]), np.array([LOW]), TOP, end=40)}]
    s = sdc.summarize(per, "synthetic")["per_height"][0]
    assert s["events_per_riser"] == 2.0
    assert s["shares_pct"]["landing_leading"] == 50.0 and s["shares_pct"]["stance_trailing"] == 50.0
    assert s["stance_td_clean_pct"] == 100.0
    assert s["crossings_leading"]["clean_pct"] == 0.0 and s["crossings_trailing"]["clean_pct"] == 0.0


def test_impact_peaks_around_a_touchdown():
    T = 20
    R = {"fz": np.zeros((T, 1, 2)), "vz": np.zeros((T, 1))}
    R["vz"][:10, 0] = -0.2                       # falling at 0.2 m/s, stopped at step 10
    R["fz"][10:13, 0, 0] = [12.0, 30.0, 9.0]     # the landing force spike on the left foot
    out = sdc._impact(R, 0, 0, 10, T, 0.02)
    assert out["impact_fz_n"] == 30.0
    assert out["impact_az"] == pytest.approx(10.0)   # 0.2 m/s lost in one 0.02 s step
    assert out["vz_td"] == pytest.approx(-0.2)
    flat = sdc._impact({"fz": R["fz"]}, 0, 0, 10, T, 0.02)   # an old record without the trunk v_z
    assert flat["impact_fz_n"] == 30.0 and np.isnan(flat["impact_az"])


def test_classify_records_crossing_impacts():
    R = _traj()
    R["vz"] = np.zeros((40, 1))
    out = sdc.classify(R, 0, np.array([EDGE]), np.array([LOW]), TOP, end=40)
    for c in out["crossings"]:
        assert {"impact_fz_n", "impact_az", "vz_td"} <= set(c)
    assert out["flat_td"] == []                   # both second stances are crossings
