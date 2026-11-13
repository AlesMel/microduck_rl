"""The head ToF sensor (VL53L8CX) in simulation — the perception input of the step-up skill.

Hardware facts (``microduck`` repo: ``tofd``, ``kinematics/src/tof.rs``): 8×8 zones, 45° square
field of view, 15 Hz, ranges 0.12–3.54 m, per zone either a range, "nothing in range" or
"could not measure". Wire order is row-major with **row 0 = top, column 0 = the sensor's
left**; zone centres are spread over the FoV with a half-zone inset (8 centres, not 9
fenceposts). Sensor frame: +x forward, +y left, +z up.

In the MJCF the sensor is the site ``tof`` on body ``jaw_soft``; checked in sim (head at HOME,
``scratch tof_probe2``): the site's **x axis points forward, y left, z up** — the sensor frame
itself — so the beams are cast as-is by mjlab's ``RayCastSensor`` with full (``"base"``)
alignment and move with the head. (The site's quat looks like a z-forward rotation in the XML,
but the parent body's own frame undoes it; trust the world-frame check, not the XML.)

Observation contract (``docs/archive/stairs_runlog.md``): 64 values in wire order,
``min(range, OBS_CLIP_M) / OBS_CLIP_M``; "nothing in range", "could not measure" and ranges
below the 0.12 m floor all read 1.0. The runtime must append the live frame with the same rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.sensor import ObjRef, RayCastSensorCfg

ROWS, COLS = 8, 8
N_ZONES = ROWS * COLS
FOV_DEG = 45.0
RANGE_MIN_M = 0.12          # the sensor's own floor; closer returns are "could not measure"
RANGE_MAX_M = 3.54
OBS_CLIP_M = 1.0            # everything a step-up policy acts on is within a metre
TOF_HZ = 15.0
TOF_SITE = "tof"
TOF_SENSOR_NAME = "tof"


def tof_beams_sensor_frame() -> np.ndarray:
    """Unit beam directions, (64, 3), row-major like the wire, in the sensor frame
    (+x forward, +y left, +z up). A direct port of ``Reprojector::new`` in ``tof.rs``."""
    half = math.radians(FOV_DEG / 2.0 - FOV_DEG / COLS / 2.0)
    step = 2.0 * half / (COLS - 1)
    beams = np.zeros((N_ZONES, 3))
    for i in range(N_ZONES):
        elevation = half - (i // COLS) * step        # row 0 = top
        azimuth = half - (i % COLS) * step           # column 0 = left (+y)
        beams[i] = (math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation))
    return beams


def tof_beams_site_frame() -> np.ndarray:
    """Beams in the MJCF ``tof`` site frame — identical to the sensor frame (see module doc)."""
    return tof_beams_sensor_frame()


# Each zone integrates a cone of FOV/8 = 5.6° — the centre ray is only its middle. Sub-rays at
# these fractions of the half-zone angle (elevation, azimuth) sample the cone's corners.
CONE_OFFSETS = ((0.6, 0.6), (0.6, -0.6), (-0.6, 0.6), (-0.6, -0.6))


def tof_cone_beams_sensor_frame(cone_rays: int) -> np.ndarray:
    """``cone_rays`` (≤ 4) extra beams per zone inside its cone, (64 × cone_rays, 3), ZONE-MAJOR
    (ray 64 + z·cone_rays + k belongs to zone z). With the 64 centre beams first, every consumer
    that reads ``[:, :64]`` still gets the wire frame."""
    assert 0 < cone_rays <= len(CONE_OFFSETS)
    half = math.radians(FOV_DEG / 2.0 - FOV_DEG / COLS / 2.0)
    step = 2.0 * half / (COLS - 1)
    half_zone = math.radians(FOV_DEG / COLS / 2.0)
    beams = np.zeros((N_ZONES * cone_rays, 3))
    for i in range(N_ZONES):
        elevation = half - (i // COLS) * step
        azimuth = half - (i % COLS) * step
        for k, (de, da) in enumerate(CONE_OFFSETS[:cone_rays]):
            e, a = elevation + de * half_zone, azimuth + da * half_zone
            beams[i * cone_rays + k] = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    return beams


@dataclass
class TofPatternCfg:
    """mjlab ray pattern for the VL53L8CX, duck-typed like ``PinholeCameraPatternCfg``: 64
    diverging rays from the site origin, wire order, site frame (= sensor frame); with
    ``cone_rays`` > 0, that many extra rays per zone inside its cone follow (run 11c's edge blur:
    the real sensor integrates the cone and mixes targets at an edge)."""

    cone_rays: int = 0

    def generate_rays(self, mj_model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        beams = tof_beams_site_frame()
        if self.cone_rays > 0:
            beams = np.concatenate([beams, tof_cone_beams_sensor_frame(self.cone_rays)], axis=0)
        directions = torch.as_tensor(beams, dtype=torch.float32, device=device)
        offsets = torch.zeros((directions.shape[0], 3), dtype=torch.float32, device=device)
        return offsets, directions


def make_tof_sensor_cfg(name: str = TOF_SENSOR_NAME, debug_vis: bool = False, cone_rays: int = 0) -> RayCastSensorCfg:
    """The scene sensor. Geom group 0 = the terrain (same as ``foot_height_scan``); the parent
    body (``jaw_soft``) is excluded so the beak does not shadow the grid."""
    return RayCastSensorCfg(
        name=name,
        frame=ObjRef(type="site", name=TOF_SITE, entity="robot"),
        pattern=TofPatternCfg(cone_rays=cone_rays),
        ray_alignment="base",
        max_distance=RANGE_MAX_M,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=debug_vis,
    )


def encode_ranges(distances: torch.Tensor, clip_m: float = OBS_CLIP_M) -> torch.Tensor:
    """Raycast distances (−1 = no hit) → the observation encoding, in [0, 1]."""
    invalid = (distances < 0) | (distances < RANGE_MIN_M) | (distances > RANGE_MAX_M)
    return torch.where(invalid, torch.full_like(distances, clip_m), distances.clamp(max=clip_m)) / clip_m


class tof_depth:
    """Observation term: the ToF frame as the policy would receive it on the robot.

    ``clean=True`` (critic) returns the current encoded raycast. Otherwise the term models the
    sensor's timing and noise per env:

    - a new frame every ``1 / hz`` seconds of policy time (50 Hz / 15 Hz ≈ 3.3 steps, so the
      hold alternates 3 and 4 steps), with a random phase per env, held in between;
    - ``latency_frames`` frames of latency (the frame the policy sees is that old);
    - per-zone range noise, σ = ``noise_rel`` × range + ``noise_abs`` (metres), on the metres
      before clipping, so far returns are noisier than near ones;
    - per-zone dropouts with probability ``dropout_prob`` → "could not measure" (1.0);
    - a per-env range scale factor ±``scale_range`` (calibration DR), zero-centred.

    At reset the frame buffer is blank (1.0 everywhere) and the first real frame arrives within
    one period — on the robot the first frame after a policy swap is just as stale.
    """

    def __init__(self, cfg, env):
        p = cfg.params
        self.sensor_name = p.get("sensor_name", TOF_SENSOR_NAME)
        self.clip_m = float(p.get("clip_m", OBS_CLIP_M))
        self.clean = bool(p.get("clean", False))
        self.hz = float(p.get("hz", TOF_HZ))
        self.latency = int(p.get("latency_frames", 1))
        self.noise_rel = float(p.get("noise_rel", 0.01))
        self.noise_abs = float(p.get("noise_abs", 0.004))
        self.dropout = float(p.get("dropout_prob", 0.05))
        self.scale_range = float(p.get("scale_range", 0.02))
        self.blank = bool(p.get("blank", False))   # ablation: the policy sees "nothing in range" forever
        n, dev = env.num_envs, env.device
        self.step_dt = float(env.step_dt)
        depth = self.latency + 1
        self.frames = torch.ones((n, depth, N_ZONES), device=dev)   # [env, age, zone]; age 0 = newest
        self.phase = torch.rand(n, device=dev)                      # fraction of a period elapsed
        self.scale = 1.0 + (torch.rand(n, device=dev) * 2 - 1) * self.scale_range

    # mjlab calls reset(env_ids) on class terms.
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.frames[env_ids] = 1.0
        n = self.frames[env_ids].shape[0]
        self.phase[env_ids] = torch.rand(n, device=self.frames.device)
        self.scale[env_ids] = 1.0 + (torch.rand(n, device=self.frames.device) * 2 - 1) * self.scale_range

    def _raw_m(self, env) -> torch.Tensor:
        d = env.scene.sensors[self.sensor_name].data.distances[:, :N_ZONES]
        return d

    def __call__(self, env, **_) -> torch.Tensor:
        d = self._raw_m(env)
        if self.blank:
            return torch.ones_like(d)
        if self.clean:
            return encode_ranges(d, self.clip_m)
        # Advance the sensor clock; envs crossing a period boundary measure a new frame.
        self.phase += self.step_dt * self.hz
        new = self.phase >= 1.0
        if bool(new.any()):
            self.phase[new] -= 1.0
            valid = (d >= RANGE_MIN_M) & (d <= RANGE_MAX_M)
            m = d * self.scale[:, None]
            m = m + torch.randn_like(m) * (self.noise_rel * m.abs() + self.noise_abs)
            m = torch.where(valid, m, torch.full_like(m, -1.0))
            if self.dropout > 0:
                drop = torch.rand_like(m) < self.dropout
                m = torch.where(drop, torch.full_like(m, -1.0), m)
            frame = encode_ranges(m, self.clip_m)
            # shift the buffer for the envs that measured: age k ← age k-1
            self.frames[new, 1:] = self.frames[new, :-1].clone()
            self.frames[new, 0] = frame[new]
        return self.frames[:, self.latency]
