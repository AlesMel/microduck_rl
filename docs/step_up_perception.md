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

## 10. Why it stumbles (2026-11-03, `scripts/eval/step_up_failures.py`, run 7b model_5999)

"64/86 on top" hid the real behaviour. Per robot, with the feet's collision boxes, the contact
force (global frame) and the ToF hit points recorded at 50 Hz, 96 robots, 12 s:

| vx (m/s) | on top / fell / stalled | clean climbs (no toe catch, no edge landing) | top without a HARD catch (\|Fx\| > 0.5 Fz) | falls preceded by a toe catch |
| --- | --- | --- | --- | --- |
| 0.25 | 68 / 14 / 4 | 1 / 86 | 15 / 68 | 9 / 14 |
| 0.15 | 46 / 19 / 21 | 1 / 86 | 7 / 46 | 12 / 19 |
| 0.10 | 13 / 7 / 66 | 0 / 86 | 0 / 13 | 6 / 7 |

- **The policy climbs by kicking the riser.** A climb has a median of 3 toe catches (1 hard); the
  riser is the step detector. Falls happen just past the edge (trunk 7 cm over it, median).
- **It cannot see the step when it matters.** The visible ground starts ≈ 20 cm ahead of the
  toes (beam 23° below level, bottom row 43°, the sensor 0.24 m up, 0.12 m minimum range), so the
  riser leaves the frame 1.3 s / 24 cm before the first foot reaches it. The actor is a
  feed-forward MLP: for the last ~3 steps it has no record of where the step is. Contact is the
  only information left, and it uses it.
- **It does not slow down** (0.23 → 0.21 m/s over the last 0.6 m): `track_vx_ema` pays constant
  speed through the step, and nothing prices a toe catch or a fall (no termination penalty;
  a fall only forfeits future reward, and the survival ladder keeps robots where ~half fail).
- **Slower is worse with this gait**: the swing apex scales with speed (stairs FEASIBILITY), so at
  0.1 m/s the feet do not clear the riser and the robots stall in front of it.

Consequences for the design: the missing piece is MEMORY of the terrain between seeing it and
stepping on it (recurrent actor, or a body-frame height map built from ToF + odometry — the
runtime's `kinematics::tof::Reprojector` already gives points in the trunk frame), then an
objective that prices stumbles and falls, and a success metric that counts clean climbs only.

## 11. Run 8: the oracle bound (2026-11-03, wandb 8vey00uh)

Requirement (the user's): every physically feasible step crossed cleanly, as a person would —
no toe catch, no edge landing; a step too high is refused (later: portable mini-stairs).

Run 8 (`Mjlab-StepUp-Oracle-Clean-Backlash-MicroDuck`, warm start from run 4's blind 61-D
policy, 2048 envs × 4000 it) asks one question: **with perfect terrain memory, can this robot
climb cleanly, and up to what height?** Its actor reads the 61-D contract plus an exact height
scan of the ground around and ahead of the feet (19 × 7 points, 3 cm, trunk yaw frame, ±5 mm
noise) — the layout a ToF + odometry height map would have on the robot. Objective changes:

- `foot_edge_contact` (−10 per foot-second): a foot in contact while the ground under its
  footprint is not one level (exact height field + the feet's collision boxes);
- `fall` (−10 once) and the step-up stack resized to dt units — the runs 6-7b weights paid
  ≈ 0.07 for a whole climb because mjlab scales every term by dt;
- the survival ladder promotes only episodes with ≤ 3 foot-steps of edge contact.

Reading the result: if the clean-climb rate (step_up_failures.py) is high up to some height,
perception/memory is the bottleneck below it and the ToF height map is worth building; the
height where it breaks down is the robot's envelope (stop above it). If it is low everywhere,
the gait/actuators are the limit and no sensor will fix it.

## 12. Run 8 result: perfect terrain memory removes the falls, not all the bumps (2026-11-03)

`step_up_failures.py`, 384 robots, 12 s, same seed; run 7b re-measured on the same battery.
"clean" = ended on top with no toe catch and no edge landing; "no hard" = no catch with
|Fx| > 0.5 Fz and no edge landing.

| step | riser | 7b ToF 0.25 clean / no hard / fell | 8 oracle 0.25 clean / no hard / fell | 8 oracle 0.15 clean / no hard / fell |
| --- | --- | --- | --- | --- |
| single | 2.0 cm | 15 % / 24 % / 2 of 41 | 71 % / 85 % / 0 | 37 % / 73 % / 1 |
| single | 2.5 cm | 6 % / 21 % / 3 of 53 | 91 % / 96 % / 0 | 75 % / 85 % / 1 |
| single | 3.1 cm | 5 % / 16 % / 4 of 43 | 88 % / 93 % / 0 | 74 % / 77 % / 0 |
| single | 3.9 cm | 0 % / 4 % / 25 of 55 | 76 % / 84 % / 1 | 75 % / 80 % / 0 |
| flight ×3 | 2.0 cm | 0 % / 0 % / 2 of 53 | 21 % / 55 % / 0 | 28 % / 55 % / 0 |
| flight ×3 | 2.6 cm | 0 % / 3 % / 7 of 29 | 34 % / 66 % / 0 | 28 % / 69 % / 0 |
| flight ×3 | 3.3 cm | 0 % / 0 % / 7 of 27 | 41 % / 63 % / 0 | 22 % / 33 % / 0 |
| flight ×3 | 3.6 cm | 0 % / 2 % / 18 of 44 | 43 % / 59 % / 0 | 30 % / 57 % / 1 |
| all | | 11 / 345 clean, 68 falls | 208 / 345 clean, 1 fall, 10 stalls | 170 / 345 clean, 3 falls |

- Falls 68 → 1. Single steps up to 3.9 cm are 76-91 % clean: the "torque ceiling" of the
  blind stairs runs was mostly a strategy/information problem — with the terrain known, the
  same actuators climb 3.9 cm without kicking three times in four.
- Flights (three risers, 15 cm treads) stay at 21-43 % clean: the feet have to fit 10 cm of
  usable tread per step, which needs stride adaptation the policy has not learned yet.
- Slower is still not cleaner (0.15 m/s: 170 vs 208).
- CONFOUND: run 8 changed the information (exact scan) AND the objective (edge cost, fall
  penalty, dt-sized step stack, clean ladder) AND the warm start (run 4, not 7b). The next run
  separates them: the ToF policy with the same clean objective. If it approaches the oracle, the
  height map is not needed; if not, the oracle's margin is what the map is worth.

## 13. Run 9: the objective alone does nothing — the gap is terrain memory (2026-11-03)

Run 9 (`Mjlab-StepUp-Tof-Clean-Backlash-MicroDuck`, wandb e48wxpyn) = run 8 with the head ToF
(free head) instead of the exact height scan; same objective, warm start (run 4), budget. Same
384-robot battery, vx 0.25:

| step | riser | 7b ToF | 9 ToF + clean objective | 8 exact scan + clean objective |
| --- | --- | --- | --- | --- |
| single | 2.0 cm | 15 % clean, 5 % fell | 10 % clean, 0 % fell | 71 % clean, 0 % fell |
| single | 2.5 cm | 6 %, 6 % | 13 %, 2 % | 91 %, 0 % |
| single | 3.1 cm | 5 %, 9 % | 9 %, 9 % | 88 %, 0 % |
| single | 3.9 cm | 0 %, 45 % | 2 %, 55 % | 76 %, 2 % |
| flight ×3 | 2.0-3.6 cm | 0 %, 4-41 % | 0-2 %, 6-64 % | 21-43 %, 0 % |
| all | | 11 / 345 clean, 68 falls | 17 / 345 clean, 79 falls | 208 / 345 clean, 1 fall |

Pricing edge contact and falls changed nothing for the ToF policy: it still kicks (980 toe
catches on 270 robots) because it still cannot see the riser for the last 21 cm / 1.1 s (head
pitched 30° down, visible ground from 17 cm ahead of the toes). The whole of run 8's gain is
the terrain information. Decision: the step-up skill needs MEMORY of the terrain between seeing
it and stepping on it. Two ways, cheapest first:

1. a recurrent actor on the ToF (training-side only; the runtime runs recurrent policies) — if
   it approaches run 8, nothing else is needed;
2. a body-frame height map built from the ToF and odometry, in the oracle's exact layout
   (19 × 7, 3 cm), with sim-side drift/noise/unseen-cell DR, and the same map built in `robotd`
   from `kinematics::tof::Reprojector` + leg odometry.

## 14. A height map from the ToF only: how close to the oracle? (2026-11-03)

Run 8's actor read the simulator's exact terrain — an oracle that cannot go on the robot. (The
CRITIC and the REWARDS may keep exact geometry: they are training-only.) `tasks/tof_map.py`
builds the same 19 × 7 scan from what the duck has: ToF frames at 15 Hz with noise and dropouts,
the sensor pose from kinematics, and the trunk pose from odometry with drift (±3 % scale, ±2°/m
yaw, 1 cm/√m random walk, ±1 cm/m in z). A 2 cm grid per env, anchored at the episode start,
keeps the running mean of the heights measured per cell; hits are also spread one cell to each
side (the zone columns are 3-5 cm apart at 0.3-0.5 m, which leaves lateral stripes otherwise).
`scripts/eval/tof_map_quality.py`, 256 robots driven by the run 9 policy (head down), compared
with the oracle scan at the same pose while approaching / climbing the first riser:

| map | foot-zone cells known | \|err\| median / p95 | edge error median / p90 | imminent edge within 1.5 / 3 cm |
| --- | --- | --- | --- | --- |
| ideal ToF, perfect odometry | 74 % | 0.0 / 0.8 cm | 0 / 3 cm | 56 / 95 % |
| ToF noise + dropouts | 74 % | 0.1 / 1.1 cm | 3 / 6 cm | 40 / 87 % |
| + odometry drift (the robot) | 74 % | 0.2 / 1.1 cm | 3 / 6 cm | 39 / 81 % |

Fusion matters: overwriting each cell with its latest hit gave edge errors p90 12 cm (single
noisy cells read as a riser); averaging halves it. Lateral spreading raised coverage 66 → 74 %
and imminent-edge-within-a-cell 75 → 81 %. Remaining limits: a quarter of the foot zone is never
seen (the policy must be told which cells are unknown), and the edge is placed to one 3 cm scan
cell, not better. The sim ToF casts each zone's centre ray; the real VL53L8CX integrates the
cone and mixes targets at an edge, so the real edge will be blurrier — to check against real
frames from `robotctl monitor` → `t` in front of a real step.
