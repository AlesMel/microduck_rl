# Step-up skill with ToF perception — design

Status: design + first implementation (2026-11-02). Background: five blind stairs runs
(`microduck_walk_v2_terrain_env_cfg.py` docstring) put the ceiling of the blind walk at
≈ 2 cm risers; on 2.4–3.0 cm the knee sits at the XL330's 0.64 N·m current cap during every
step-up, and pricing torque (runs 4–5) only lowered the swing. Quasi-statically the lift needs
0.24–0.39 N·m, so the budget exists for a *deliberate* step — which needs to see the step.

## 1. What the robot can see

The head carries a **VL53L8CX** time-of-flight sensor (`tofd` in the `microduck` repo):
8×8 zones, 45° square field of view, 15 Hz, ranges 0.12–3.54 m, per-zone "nothing in range"
and "could not measure" marks. Wire order is row-major, **row 0 = top, column 0 = sensor's
left**; zone centres are spread over the FoV with a half-zone inset (`kinematics/src/tof.rs`,
`Reprojector::new`). Sensor frame: +x forward, +y left, +z up.

In the MJCF the sensor is the site `tof` on body `jaw_soft`. Checked in the world (head at
HOME): the site's **x axis points forward, y left, z up** — the sensor frame itself; the XML
quat looks like a z-forward rotation but the parent body's frame undoes it, so trust the
world-frame check, not the XML. mjlab's `PinholeCameraPatternCfg` assumes −z forward and
fencepost spacing, so the sim uses its own `TofPatternCfg` that reproduces `tof.rs` exactly.
Measured with the head at HOME on a flat tile: bottom row ranges the floor at 0.76 m (sensor
0.24 m up, beam 19.7° below the axis), rows 0–4 are above the horizon; on the stairs tile the
risers read as shorter ranges in rows 5–6.

The head is actuated (neck_pitch −90…+60°, head_pitch ±90°; a NEGATIVE neck_pitch delta or a
POSITIVE head_pitch delta pitches the sensor down, 0.4 rad ≈ 23°), so the policy can aim it:
at 0.2 m sensor height and 0° pitch the bottom row reaches the floor ≈ 0.48 m ahead; pitched
down 25–30° the grid covers the floor from ≈ 0.15 m to ≈ 1.2 m ahead — the whole approach to
a step.

## 2. Observation contract (perceptive policies)

```
[0:61]    the family's 61-D actor obs, bit-identical (48 proprio + 13 command block)
[61:125]  tof: 64 zones in wire order, value = min(range, 1.0 m) / 1.0 m
          no return ("nothing in range") → 1.0; could-not-measure ("x") → 1.0
```

The first 61 stay the family contract so a perceptive policy is the blind policy plus an
appended frame. The runtime side (repo `microduck`) needs: a policy kind whose manifest says
`obs_len: 125` (today `POLICY_OBS_LEN = 61` refuses anything else), and the control loop
appending the newest `tofd` frame in wire order with the same clip/fill rule — the 15 Hz
reader-thread-plus-slot already exists for the theremin (`robotd/src/theremin.rs`).

Why 1.0 m: everything a step-up policy acts on is within a metre; the clip keeps the input
bounded and makes far returns and misses identical, which is what they are for this skill.

## 3. Sensor model in sim (`tasks/tof.py`)

- `RayCastSensorCfg` on site `tof`, 64 rays from `TofPatternCfg`, `ray_alignment="base"`,
  `max_distance` 3.54, geom group 0 (the terrain), parent body excluded.
- Observation term `tof_depth` (class, keeps state per env):
  - **15 Hz**: a new frame every 3–4 policy steps (50 Hz / 15 Hz = 3.33), phase random per env,
    held in between; **latency** one frame (the frame the policy sees was measured 67 ms ago);
  - **range noise** σ = 1 % of range + 4 mm per zone; **dropouts** p = 0.05 per zone → 1.0;
  - **mounting misalignment** ±3° pitch/yaw per env, zero-centred (same philosophy as the IMU
    DR: tolerance to magnitude, not compensation of a systematic bias);
  - ranges < 0.12 m read as could-not-measure (1.0) as on the hardware.
- Critic gets the clean frame (`tof_depth_clean`) in addition — privileged, no noise/hold.

## 4. The step-up task (`Mjlab-StepUp-MicroDuck`)

Built on the stairs-mode recipe (slow walk, apex reward, run 4's gait), not from scratch:

- **Terrain** `single_step`: flat approach 0.3–0.9 m, one riser 1.5–4.0 cm (difficulty), a
  0.6 m top platform, then flat; small flights (2–3 risers) as a second sub-terrain so the
  policy also learns to continue. Spawn yaw ±20° so the step is not always square-on.
- **Commands**: twist vx ∈ [0.1, 0.3] (go forward), head and body slots zero-padded with the
  usual tiny ranges (kept alive, never deleted).
- **Rewards** (on top of the inherited gait stack):
  - potential-based progress of trunk height toward the top level: pays Δz clipped at the riser
    height — rising pays, standing on top pays zero, falling back pays negative;
  - foot-on-tread: contact gated, foot centre ≥ 2 cm past the riser edge, once per foot per
    episode (no per-step jackpot);
  - |a_z| impact penalty on the step-up and torque headroom at run 4's weight (cheap);
  - head posture: weak Gaussian on neck_pitch around −0.45 rad so the floor ahead is in view
    from step 0 (the policy may deviate; it is shaping, not a gate).
- **Terminations**: fall, out of tile; time-out 8 s.
- **Reverse-curriculum spawns**: 30 % of episodes start with the lead foot already on the step
  and 10 % standing on top, so the last mile gets on-policy data.
- **Head mass**: tilting the head forward moves the CoM forward; the settle test (hold the
  pitched-down HOME 3 s from noisy inits, check tilt) runs before training.

## 5. Warm start across observation widths

Run 4's actor is 61→…→14; the step-up actor is 125→…→14. `MICRODUCK_WARM_START_EXPAND_OBS=1`
loads a 61-D checkpoint into the 125-D network: first-layer columns for inputs 0–60 copied,
columns 61–124 zero-initialised, normalizer statistics for the new inputs fresh. Iteration 0
is then exactly the blind stairs policy; the ToF columns grow from zero as they earn reward.

## 6. Verification before any long run

1. Render the sim ToF grid for a robot standing before the stairs at several head pitches and
   check it against the geometry by hand (bottom row = floor at the expected distance, the
   riser as a band of shorter ranges). Same picture from a real duck on the lab stairs via
   `robotctl monitor` → `t` is the sim2real check of the sensor model.
2. Settle test of the pitched-down HOME.
3. Smoke test (64 envs, 5 iterations): 125-D obs, every reward term computes, ONNX exports
   with the 125-D normalizer.

## 7. Out of scope here, needed for deployment

- Runtime: `obs_len 125` policy kind + ToF append in the control loop (`microduck` repo).
- `publish`: `OBS_LEN` gate per policy kind (manifest schema change on both sides).

## 8. Result of the first run (2026-11-02)

Run 6 (wandb 5fiepb9r, warm-started from the blind stairs policy) uses the sensor: on the
play grid, sighted 50/86 robots end on top of their step (3.7 cm single step 7/13, 11 cm
three-riser flight 4/12) with 11 falls; the same network with the ToF frame blanked ends on
top 15/86 with 61 falls, i.e. back at the blind walk's ≈ 2 cm ceiling. No wandb aggregate
showed it — the survival ladder plateaued at 5.9 — the sighted/blind eval did
(`scripts/eval/step_up_eval.py --blind`). Details in the env cfg docstring.

## 9. Runs 7 and 7b (2026-11-03)

Doubling the training (run 7) changed nothing: 49/86 on top, as run 6. Loosening the head
tracking (run 7b, `Mjlab-StepUp-Tof-FreeHead-MicroDuck`: weight 2.0 → 0.5, std 0.25 → 0.5)
gave 64/86 on top with 14 falls — 3.1 cm single steps 13/15, 3.7 cm 8/13, 11 cm flights 6/12.
Blind baselines stay at 5–18/86 for every run. Run 7b's model_5999 is the policy to publish
once the runtime side is merged. Details in the env cfg docstring.
