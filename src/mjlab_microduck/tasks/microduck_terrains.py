"""Terrain menus for the microduck, IsaacLab-style, scaled to a 25 cm robot.

IsaacLab's ``ROUGH_TERRAINS_CFG`` is a curriculum grid: one column per terrain type
(flat, random rough, slopes up/down, stairs up/down, boxes), difficulty growing along the
rows, robots promoted a row when they cover enough ground in an episode. mjlab has the
same machinery (``TerrainGeneratorCfg(curriculum=True)`` + the ``terrain_levels``
curriculum the velocity recipe already registers); what differs here is the SCALE. The
microduck's foot is 5.4 cm long and 4.1 cm wide (sole mesh of robot_walk.xml, STAND
keyframe), its trunk stands 12 cm up, and the walking policies lift a foot 2-3.5 cm, so
every obstacle below is 5-10× smaller than IsaacLab's humanoid/quadruped numbers.

Physical scaling used (why each number):
  - ``FOOT_LENGTH`` 0.054 m   → stair treads ``STAIR_TREAD`` 0.15 m (≈ 2.8 feet: a foot
    landing anywhere on a tread sits on ONE level, never bridging the nose).
  - step heights 0.5 → 3 cm by difficulty. 3 cm is what the 2026-08 hostile-terrain runs
    trained with a 3.5 cm swing target (branch hostile-terrain), i.e. a step the leg can
    clear with margin; the base rough mix caps at 1.5 cm.
  - rubble ±2 mm → ±2 cm, grid cells 0 → 1.5 cm, slopes 1.7° → 8.5°.
  - patches are 4 m: 4× fewer boxes than the 8 m base patches, and the promotion rule
    (walk half a patch = 2 m in 20 s) is reachable at the 0.1-0.5 m/s commands.

Two menus:
  * :func:`make_microduck_terrains_cfg` — the full IsaacLab-like mix (+ straight stairs).
  * :func:`make_microduck_stairs_terrains_cfg` — stairs only (straight + pyramid + a flat
    column so the gait is not forgotten), for a step-focused curriculum.

Preview either, with the robot standing on it, via ``scripts/view_terrain_env.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

FOOT_LENGTH = 0.054          # m, sole x-extent on robot_walk.xml (STAND keyframe)
FOOT_WIDTH = 0.041           # m
STAIR_TREAD = 0.15           # m, ≈ 2.8 foot lengths
STEP_HEIGHT_RANGE = (0.005, 0.030)   # m, difficulty 0 → 1
PATCH_SIZE = 4.0             # m, one sub-terrain tile
NUM_ROWS = 10                # difficulty rows (row r ≈ difficulty r / NUM_ROWS)
BORDER_WIDTH = 2.0           # m, flat apron around the grid


def _lerp(lo: float, hi: float, difficulty: float) -> float:
    d = float(np.clip(difficulty, 0.0, 1.0))
    return lo + d * (hi - lo)


@dataclass(kw_only=True)
class BoxStraightStairsTerrainCfg(SubTerrainCfg):
    """Small straight stairs along +x: start platform → ``num_steps`` up → top platform →
    ``num_steps`` down → end platform. Full tile width in y.

    This is the "small steps" terrain: unlike the pyramid, a robot commanded along ±x meets
    a real flight of stairs (and descends on the far side); commanded along ±y it walks a
    flat shelf. The robot spawns at the centre of the start platform, on z = 0, so the
    reset's absolute trunk height (0.12 m + env origin) is exactly as on flat ground.

    Box tops: start 0, step k = k·h, top platform n·h, then back down to 0. Each box
    extends from z = -``thickness`` up to its top so neighbours share a vertical face —
    no gaps, no overlap-induced contact noise.
    """

    step_height_range: tuple[float, float] = STEP_HEIGHT_RANGE
    """Step rise, metres, interpolated by difficulty."""
    step_width: float = STAIR_TREAD
    """Tread depth, metres."""
    num_steps: int = 4
    """Steps up (and down again)."""
    start_length: float = 1.0
    """Spawn platform, metres along x."""
    top_length: float = 0.6
    """Landing on top, metres along x."""
    thickness: float = 0.5
    """Box depth below z = 0, metres."""

    def total_length(self) -> float:
        return self.start_length + 2 * self.num_steps * self.step_width + self.top_length

    def step_height(self, difficulty: float) -> float:
        return _lerp(self.step_height_range[0], self.step_height_range[1], difficulty)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # Deterministic geometry.
        assert self.step_width >= 2 * FOOT_LENGTH, (
            f"tread {self.step_width} m must hold a {FOOT_LENGTH} m foot on one level"
        )
        end_length = self.size[0] - self.total_length()
        assert end_length >= 0.3, (
            f"stairs ({self.total_length():.2f} m) leave < 0.3 m end platform in a "
            f"{self.size[0]} m tile"
        )
        body = spec.body("terrain")
        h = self.step_height(difficulty)
        width = self.size[1]
        n = self.num_steps

        # (x0, x1, top_z) left to right.
        segments: list[tuple[float, float, float]] = []
        x = 0.0
        segments.append((x, x + self.start_length, 0.0))
        x += self.start_length
        for k in range(1, n + 1):
            segments.append((x, x + self.step_width, k * h))
            x += self.step_width
        segments.append((x, x + self.top_length, n * h))
        x += self.top_length
        for k in range(n - 1, -1, -1):
            segments.append((x, x + self.step_width, k * h))
            x += self.step_width
        segments.append((x, self.size[0], 0.0))

        geometries: list[TerrainGeometry] = []
        for x0, x1, top in segments:
            half_h = (top + self.thickness) / 2.0
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2.0, width / 2.0, half_h),
                pos=((x0 + x1) / 2.0, width / 2.0, top - half_h),
            )
            # Lighter the higher the tread, so steps read at a glance in the viewer.
            shade = 0.45 + 0.5 * (top / max(n * h, 1e-6)) if n * h > 0 else 0.5
            geometries.append(TerrainGeometry(geom=geom, color=(shade, 0.75 * shade, 0.4, 1.0)))

        origin = np.array([self.start_length / 2.0, width / 2.0, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class HfRubbleTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    """Continuous bumps (gravel / grass proxy) whose amplitude scales with difficulty.

    mjlab's ``HfRandomUniformTerrainCfg`` ignores difficulty and puts the spawn origin at
    the MEAN height. Here ``noise_range`` is the amplitude at difficulty 1 and
    ``noise_range_easy`` at difficulty 0, and the origin is lifted to the tallest bump so
    a reset never starts with a foot inside the ground. (Ported from the 2026-08
    hostile-terrain branch.)
    """

    noise_range_easy: tuple[float, float] = (-0.002, 0.002)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        lo = _lerp(self.noise_range_easy[0], self.noise_range[0], difficulty)
        hi = _lerp(self.noise_range_easy[1], self.noise_range[1], difficulty)
        saved = self.noise_range
        self.noise_range = (lo, hi)
        try:
            out = super().function(difficulty, spec, rng)
        finally:
            self.noise_range = saved
        out.origin[2] = max(float(out.origin[2]), hi - lo) + 0.002
        return out


def microduck_subterrains(scale: float = 1.0) -> dict[str, SubTerrainCfg]:
    """The IsaacLab-like menu. ``scale`` multiplies every obstacle height (1.0 = the
    numbers in the module docstring). Proportions = share of robots per column."""
    s = scale
    return {
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.10),
        # IsaacLab "random_rough": a heightfield of bumps every 8 cm, 4 cm cells.
        "rubble": HfRubbleTerrainCfg(
            proportion=0.15,
            noise_range=(-0.02 * s, 0.02 * s),
            noise_range_easy=(-0.002 * s, 0.002 * s),
            noise_step=0.001,
            downsampled_scale=0.08,
            horizontal_scale=0.04,
            vertical_scale=0.001,
            border_width=0.2,
        ),
        # IsaacLab "boxes": random cell heights. 0.3 m cells on a 4 m tile = 13 × 13
        # (0.3 × 13 = 3.9: must not divide the tile evenly, see the base rough cfg).
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.3,
            grid_height_range=(0.0, 0.015 * s),
            platform_width=0.6,
            border_width=0.2,
        ),
        # IsaacLab "pyramid_stairs" / "pyramid_stairs_inv": 15 cm treads, 0.5 → 3 cm rise.
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(STEP_HEIGHT_RANGE[0] * s, STEP_HEIGHT_RANGE[1] * s),
            step_width=STAIR_TREAD,
            platform_width=0.6,
            border_width=0.2,
        ),
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(STEP_HEIGHT_RANGE[0] * s, STEP_HEIGHT_RANGE[1] * s),
            step_width=STAIR_TREAD,
            platform_width=0.6,
            border_width=0.2,
        ),
        # The small steps: a straight flight up and down along x.
        "straight_stairs": BoxStraightStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(STEP_HEIGHT_RANGE[0] * s, STEP_HEIGHT_RANGE[1] * s),
        ),
        # IsaacLab "hf_pyramid_slope": rise/run 0.03 → 0.15 (1.7° → 8.5°), 1 mm quantisation.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.03 * s, 0.15 * s),
            platform_width=0.6,
            horizontal_scale=0.1,
            vertical_scale=0.001,
        ),
    }


def microduck_stairs_subterrains(scale: float = 1.0) -> dict[str, SubTerrainCfg]:
    """Stairs-focused menu: straight small steps, pyramid stairs both ways, one flat column."""
    s = scale
    heights = (STEP_HEIGHT_RANGE[0] * s, STEP_HEIGHT_RANGE[1] * s)
    return {
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.10),
        "straight_stairs": BoxStraightStairsTerrainCfg(proportion=0.45, step_height_range=heights),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=heights, step_width=STAIR_TREAD,
            platform_width=0.6, border_width=0.2,
        ),
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=heights, step_width=STAIR_TREAD,
            platform_width=0.6, border_width=0.2,
        ),
    }


def _generator(subs: dict[str, SubTerrainCfg], num_rows: int, seed: int | None,
               border_width: float = BORDER_WIDTH) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=seed,
        size=(PATCH_SIZE, PATCH_SIZE),
        border_width=border_width,
        num_rows=num_rows,
        num_cols=len(subs),          # curriculum mode: one column per type anyway
        curriculum=True,
        color_scheme="height",
        sub_terrains=subs,
        add_lights=False,
    )


def make_microduck_terrains_cfg(
    num_rows: int = NUM_ROWS, scale: float = 1.0, seed: int | None = None, border_width: float = BORDER_WIDTH
) -> TerrainGeneratorCfg:
    """IsaacLab-like curriculum grid for the microduck (see module docstring)."""
    return _generator(microduck_subterrains(scale), num_rows, seed, border_width)


def make_microduck_stairs_terrains_cfg(
    num_rows: int = NUM_ROWS, scale: float = 1.0, seed: int | None = None, border_width: float = BORDER_WIDTH
) -> TerrainGeneratorCfg:
    """Stairs-only curriculum grid for the microduck."""
    return _generator(microduck_stairs_subterrains(scale), num_rows, seed, border_width)
