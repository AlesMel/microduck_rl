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

## 15. Run 10: the map the robot can build matches the oracle (2026-11-03, wandb w4z8f80u)

Run 10 (`Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck`, 2048 envs × 4000 it, 1 h 45 min) is run 8
with the actor's exact scan replaced by the §14 map (`tasks/tof_map.py`: 15 Hz noisy ToF frames
+ drifting odometry → 2 cm grid, running mean, lateral splat) read in the oracle's 19 × 7 layout
plus a known-cell mask (61 + 266-D actor); the critic keeps the exact scan. Warm start from run 8's
model_3999 (`MICRODUCK_WARM_START_EXPAND_OBS=1`: the scan columns copied, the mask columns zero),
so iteration 0 was the oracle policy reading a noisier, partly unknown version of its input.
Training: terrain level 2.5 → 5.4 by iteration 1000, flat after (run 8 ended at 5.5); reward
45 vs the oracle's 51, falls per iteration 0.9 vs 0.6. Same 384-robot battery, vx 0.25, 12 s,
same seed (`failures_10.json`):

| step | riser | 9 ToF only clean / fell | 8 oracle clean / no hard / fell | 10 ToF map clean / no hard / fell / stalled |
| --- | --- | --- | --- | --- |
| single | 2.0 cm | 10 % / 0 | 71 % / 85 % / 0 | 83 % / 88 % / 0 / 0 |
| single | 2.5 cm | 13 % / 1 | 91 % / 96 % / 0 | 79 % / 87 % / 0 / 0 |
| single | 3.1 cm | 9 % / 4 | 88 % / 93 % / 0 | 86 % / 93 % / 0 / 0 |
| single | 3.9 cm | 2 % / 30 | 76 % / 84 % / 1 | 65 % / 71 % / 2 / 4 |
| flight ×3 | 2.0 cm | 2 % / 3 | 21 % / 55 % / 0 | 34 % / 57 % / 1 / 0 |
| flight ×3 | 2.6 cm | 0 % / 5 | 34 % / 66 % / 0 | 62 % / 72 % / 1 / 0 |
| flight ×3 | 3.3 cm | 0 % / 8 | 41 % / 63 % / 0 | 67 % / 70 % / 1 / 0 |
| flight ×3 | 3.6 cm | 0 % / 28 | 43 % / 59 % / 0 | 57 % / 68 % / 1 / 8 |
| all | | 17 / 345 clean, 79 falls | 208 / 345 clean, 1 fall, 10 stalls | 228 / 345 clean, 6 falls, 12 stalls |

- **The map closes the gap.** Same sensor as run 9 (riser in view for 315 / 345 robots, then out of
  the frame for 1.24 s / 37 cm before the foot lands on it — unchanged), 17 → 228 clean climbs,
  79 → 6 falls, toe catches 980 → 132. Everything run 8 had and run 9 lacked was memory of the
  terrain, and a map built from the real sensor's noise, dropouts and a drifting odometry holds
  it well enough.
- **Single steps up to 3.1 cm: 79-86 % clean, no falls.** 3.9 cm: 65 % clean, 2 falls and 4
  stalls in 55 — the top of the envelope, where "refuse" starts to appear by itself.
- **Flights are better than the oracle's** (57-67 % vs 34-43 % at 2.6-3.6 cm). Caveat: run 10 is
  run 8 plus another 4000 iterations, so part of this is training time, not the map; the oracle
  was not trained on for comparison. The remaining flight failures are stalls at 3.6 cm (8) and one
  fall per row.
- The policy does not slow down for the step (0.25 → 0.26 m/s) and holds the sensor 26° below
  level; it reads the map, not the riser.
- Video: `showcase_10.mp4` (beams, the map as the policy reads it with never-seen cells hollow,
  foot halos), `compare_9_10.mp4` (same step, ToF alone vs ToF + map), `compare_8_10.mp4` (oracle
  vs the buildable map), all on urkui-3.

Checkpoint: `logs/rsl_rl/step_up/2026-11-03_20-30-50_step_up_tofmap_clean/model_3999.pt`
(urkui-3). This is the first step-up policy whose every input exists on the robot.

### Next

1. **Runtime side** (`microduck` repo): build the same map in `robotd` — `kinematics::tof::Reprojector`
   hit points + leg odometry into a 2 cm grid anchored at policy start, running mean, lateral splat,
   the 19 × 7 query in the trunk's estimated yaw frame, `[values, known]` encoding with unknown = 0
   — and a policy kind with `obs_len 327`. `publish` needs the matching perception kind (`tof_map`,
   layout in the manifest) read off the ONNX width. Export run 10 and rehearse the full obs
   contract in `scripts/infer_policy.py` first.
2. **Sensor-model check against real frames**: the sim casts one ray per zone; the VL53L8CX
   integrates a cone and mixes targets at an edge, so the real edge is blurrier. Record `robotctl
   monitor → t` in front of a real 3 cm step and compare the map it produces with §14's error table
   before trusting the 65-86 % numbers. Widen the sim's edge blur / drift DR if the real map is worse.
3. **The envelope and the refusal**: the requirement is "too high → stop". 3.9 cm already shows
   stalls; measure where falls begin (add a 4.5 and 5 cm row to the battery) and decide whether to
   train an explicit stop (a "no step" posture command, or a hard cap on the map's visible rise
   that the reward turns into stopping) or to leave the refusal to the policy.
4. **Flights**: stride adaptation on 15 cm treads is the remaining skill gap (57-67 %). Longer
   training alone is unlikely to be enough (run 7 lesson); candidates are tread-length DR and a
   foot-placement term on the map's level cells.
5. Train the oracle for the same extra 4000 iterations if the "map beats oracle" question matters
   for the design; otherwise it does not change the plan.

## 16. Run 11 (2026-11-03 night): the envelope, a worse sensor, the flights

Three runs on run 10's checkpoint (`model_3999`, same 327-D obs, `MICRODUCK_WARM_START=1`), one
after the other on the 12 GB box (`scripts/run11/chain.sh`, 2048 envs × 4000 it each, a battery
after each; `scripts/run11/smoke.sh` first). Order by value: 11a → 11c → 11b. Each is run 10 plus
ONE lever, so the batteries read against `failures_10.json` directly.

**11a `Mjlab-StepUp-TofMap-Envelope-*` — "too high → stop".** The requirement (§11) has two
halves and run 10 only trained the first. The grid gets a fourth column, `high_step`: one riser of
4.5-6.5 cm (`HIGH_STEP_HEIGHT_RANGE`; the 4.0-4.5 band is left out so the map, p95 error 1.1 cm,
can tell the columns apart). The objective (all privileged, from the exact height field):
- `track_linear_velocity_stop_gated`: within `STOP_DIST` 0.30 m of a riser taller than
  `MAX_FEASIBLE_RISE` 4.2 cm the velocity TARGET is zero — the policy still sees its 0.25 m/s
  command (the runtime will keep sending it); standing still there pays the full tracking reward.
- `stop_before_high_step` (cost, −2/s × |v|/0.25 while stopping is due): the explicit brake, since
  the tracking Gaussian (std² 0.1) only loses 46 % between 0 and 0.25 m/s.
- `height_progress` and `foot_on_step` pay nothing on the `high_step` column (`skip_tile`), so a
  5 cm climb earns nothing and risks the edge and fall costs.
- the ladder's progress rule exempts a robot that reached the stop zone (`_stop_seen`), so the
  refusal promotes instead of being demoted as "no progress".
Battery: `high_step` tiles report `stopped` (survived, not on top, < 0.05 m/s over the last 0.5 s,
before the riser) with the clean count and the stop distance; the feasible rows must stay at run
10's clean rates — the thing to check is that stopping did not leak into the 3.1-3.9 cm rows.

**11c `Mjlab-StepUp-TofMap-Hard-*` — the map under a worse sensor.** The sim casts one ray per
zone; the VL53L8CX integrates a 5.6° cone and mixes the targets at an edge, and the real odometry
is a guess. The sensor gets 4 extra rays per zone inside its cone (`tof.TofPatternCfg(cone_rays)`,
after the 64 centres so every `[:, :64]` reader is unchanged) and each frame reads every zone from
ONE of its five rays at random (`TofHeightMap.pick_cone`) — at an edge the return lands on either
level, on the floor it is the centre's hit. Noise, dropouts and all four drift terms are doubled
(`TOF_MAP_KWARGS_HARD`). The critic keeps the exact scan. Batteries: on the base grid (comparable
to run 10) and under the hard sensor; the chain also runs run 10's checkpoint under the hard sensor
first, so "what the sensor model alone costs" is on the table before 11c finishes.

**11b `Mjlab-StepUp-TofMap-Flights-*` — stride adaptation.** Run 10's flights (57-67 % clean) were
trained on 15 cm treads only. Training tiles draw the flight's tread per tile from 11-24 cm
(`STEP_UP_TREAD_RANGE`; ≥ 2 foot lengths); the play grid keeps 15 cm so the battery stays
comparable. `foot_placement_bonus` (+1 per touchdown, `PLACEMENT_WEIGHT` 50·dt) pays a touchdown
that is on raised ground, one level under the footprint grown by 2 cm, a stride of ≥ 4 cm from
that foot's previous touchdown, and still inside the flight (exact ground 0.5 m ahead is higher) —
on the top platform it pays nothing, so it cannot be farmed by walking on.

### Results (2026-11-04, `chain11_summary.txt`; battery 384 robots, 12 s, 0.25 m/s, same seed as run 10)

| step | riser | 10 clean | 11a Envelope | 11c Hard | 11c under the hard sensor | 11b Flights |
| --- | --- | --- | --- | --- | --- | --- |
| single | 2.0 cm | 83 % | 91 % | 88 % | 63 % | 95 % |
| single | 2.5 cm | 79 % | 83 % | 98 % | 89 % | 98 % |
| single | 3.1 cm | 86 % | 88 % | 91 % | 91 % | 100 % |
| single | 3.9 cm | 65 % | 89 % | 80 % | 80 % | 82 % |
| flight | 2.0 cm | 34 % | 60 % | 55 % | 40 % | 57 % |
| flight | 2.6 cm | 62 % | 77 % | 79 % | 72 % | 66 % |
| flight | 3.3 cm | 67 % | 52 % | 70 % | 63 % | 81 % |
| flight | 3.6 cm | 57 % | 74 % | 55 % | 52 % | 75 % |
| **high** | **4.5 cm** | – | **85 %** (0 stopped) | – | – | – |
| **high** | **5.3 cm** | – | **91 %** (0 stopped) | – | – | – |
| **high** | **5.8 cm** | – | **85 %** (0 stopped) | – | – | – |
| **high** | **6.5 cm** | – | **96 %** (0 stopped) | – | – | – |
| all | | 228 / 345, 6 falls | 279 / 345, 3 falls | 266 / 345, 0 falls | 238 / 345, 6 falls | 283 / 345, 3 falls |

Control: run 10 itself under the hard sensor = 198 / 345 clean, 10 falls (the sensor model alone
costs ≈ 30 clean climbs). wandb: 11a `v1zgwli4`, 11c `ommolfg5`, 11b `o34sb08x`; checkpoints
`model_3999` in `logs/rsl_rl/step_up/2026-11-03_23-43-02_step_up_tofmap_envelope`,
`2026-11-04_01-41-38_step_up_tofmap_hard`, `2026-11-04_03-41-40_step_up_tofmap_flights` (urkui-3).
Videos: `showcase_11{a,c,b}.mp4`, `compare_10_11b.mp4`, `compare_10_11c_hard.mp4`.

- **11a: the stop was never learned — the duck climbed the "too high" column instead.** On
  4.5-6.5 cm it ended on top 74 / 77 times, 85-96 % clean, ONE fall, zero stops (the showcase hero
  climbs a 6.2 cm riser with no edge contact). `stop_before_high_step` sat at −0.05 per step for
  the whole run against a mean reward of 44: 20 % of the robots × 1.2 s in the zone × 2/s is
  ≈ 0.5 per episode, and once on top the velocity reward is back in full — PPO paid the toll and
  walked on. The lesson is not about the brake but about the envelope: run 10's 3.9 cm "ceiling"
  was the top of its training range, not the robot's limit. Before any refusal is trained again the
  physical limit has to be measured (extend the grid upward until clean rates collapse); the
  refusal objective then needs reward mass comparable to the climb (a terminal cost on crossing
  the riser, or a termination), not a per-step nudge.
- **11c: training under the worse sensor costs nothing and buys robustness.** On the nominal
  sensor 266 / 345 clean with no fall at all; under the hard sensor 238 / 345, 6 falls, vs run 10's
  198 / 10 under the same sensor. The 2.0 cm single step under the hard sensor (63 %) is the one
  row where the blurred edge still hurts: a 2 cm riser is within the map's noise.
- **11b: the flights moved most where the tread matters.** 3.3 / 3.6 cm flights 81 / 75 % clean
  (run 10: 67 / 57 %); the 3.6 cm stalls persist (7 / 44). Single steps up to 3.1 cm are at
  95-100 %. Best overall at 283 / 345.
- **Confound:** every run 11 checkpoint has 4000 more iterations than run 10, and the three are
  close on their shared rows (279 / 266 / 283 total), so most of the single-step gain is training
  time, not the lever. The lever shows only where the runs differ: 11c under the hard sensor, 11b
  on the 3.3-3.6 cm flights, 11a on the high column.

### Where clean climbing ends (2026-11-04, `scripts/run11/limit.sh`, `failures_limit_*.json`)

Single steps beyond the training range, play grid with `--riser-range`, 384 robots, 12 s, 0.25 m/s,
the base task's nominal sensor. 11a was trained up to 6.5 cm, 11b up to 4 cm:

| riser | 11a clean / fell | 11b clean / fell |
| --- | --- | --- |
| 5.1 cm | 90 % / 5 % | 98 % / 2 % |
| 5.7 cm | 91 % / 0 % | 94 % / 4 % |
| 6.4 cm | 91 % / 2 % | 88 % / 5 % |
| 7.4 cm | 85 % / 5 % | 53 % / 35 % |
| 7.7 cm | 88 % / 2 % | 68 % / 27 % |
| 8.4 cm | 68 % / 25 % | 58 % / 34 % |
| 9.2 cm | 30 % / 67 % | 40 % / 53 % |
| 10.3 cm | 5 % / 95 % | 0 % / 100 % |

The limit is a cliff between 8 and 9 cm (trunk at 11.5 cm, 25 cm robot): up to 7.7 cm the policy
trained on 6.5 cm risers climbs 85-91 % clean with ≤ 5 % falls, at 8.4 cm one in four falls, at
9.2 cm two in three, 10.3 cm is a wall (toe catches on 35 / 55 robots). Training range matters past
7 cm: 11b (4 cm max) already falls 35 % at 7.4 cm where 11a falls 5 %. So the refusal belongs at
≈ 8 cm, twice the 4.2 cm assumed for run 11a, and the climb column should go to 7.5 cm.

### Next (after run 11)

1. ~~Find the real envelope~~ — done above: climb ≤ 7.5 cm, refuse ≥ 8.5 cm, 7.5-8.5 cm is the
   grey band to leave out of the grid.
2. **Retrain the refusal above that height** with a cost that matches the climb: terminate (or pay
   the fall penalty) when the trunk crosses an infeasible riser, keep the zero target in the zone.
3. **Merge the levers**: 11b's tread DR + placement and 11c's hard sensor model in one run (they
   touch different parts of the cfg), warm-started from 11b.
4. Runtime side unchanged from §15 (build the map in `robotd`, export, rehearse).

