> **Archived 2026-11-11, superseded by [`docs/stairs.md`](../stairs.md)** (the current state: policies, checkpoints, the switch).
> Kept unchanged as the run diary (runs 6 → 18d, §1–§29); code comments cite its sections as `docs/archive/stairs_runlog.md §N`.
> Task ids here are the pre-cleanup ones; the old code is in git history and in `logs/backup/pre_cleanup_2026-11-11.*`.

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
after each; a 64-env × 5-it smoke of each task first). Order by value: 11a → 11c → 11b. Each is run 10 plus
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


## 17. How many steps in a row (2026-11-08, `scripts/eval/step_up_staircase.py`)

One long straight staircase per riser height: a 0.7 m run-up, 20 risers, a 1.2 m top platform.
1200 robots (200 per height), spawn yaw 0, a constant 0.25 m/s through the observation with no
heading correction, and 40-50 s per run.

The ToF map buffer is extended to cover the whole tile, because training's map ends 3 m from the
spawn. That is a limit of the sim buffer, not of the skill. The odometry drift DR stays as in
training.

Per robot the test records:
- the risers climbed (the highest tread a foot stood on fully);
- the **clean run**: risers climbed before the first toe catch, edge landing or fall, with the
  catch and edge definitions of `step_up_failures.py`;
- where the robot fell.

Robots that drift more than 0.9 m sideways into the neighbouring column (7-23 %, late in the climb)
are not climbing failures. They are censored: the numbers below are Kaplan-Meier over risers. That
is also why they differ from the script's raw "share with a clean run ≥ k" when many robots drift
(24 cm treads). JSONs are in `logs/staircase/` (gpu box and Mac); `step_up_staircase.py --from-json logs/staircase/*.json` reprints every table below.

**Each cell is median clean run / share of robots with all 20 risers clean / share with no fall through 20.**

| riser | 11b, tread 15 cm | 11b, tread 20 cm | 11b, tread 24 cm | 11a, tread 15 cm |
| --- | --- | --- | --- | --- |
| 2 cm | 6 / 8 % / 97 % | 10 / 29 % / 100 % | 7 / 18 % / 97 % | 4 / 2 % / 99 % |
| 3 cm | 10 / 29 % / 98 % | > 20 / 60 % / 97 % | 20 / 50 % / 98 % | 5 / 4 % / 92 % |
| 4 cm | 8 / 20 % / 88 % | > 20 / 56 % / 97 % | > 20 / 53 % / 95 % | 5 / 4 % / 70 % |
| 5 cm | 6 / 5 % / 46 % | 13 / 38 % / 73 % | 19 / 46 % / 82 % | 4 / 1 % / 15 % |
| 6 cm | 4 / 0 % / 4 % | 5 / 7 % / 22 % | 8 / 14 % / 37 % | 3 / 0 % / 1 % |
| 7 cm | 3 / 0 % / 0 % | 3 / 0 % / 0 % | 3 / 0 % / 0 % | 3 / 0 % / 0 % |

**Hazard of the first unclean riser, % per riser, by riser band 1-3 / 4-6 / 7-10 / 11-15 / 16-20 (11b):**

| riser | tread 15 cm | tread 20 cm |
| --- | --- | --- |
| 3 cm | 8.5 / 5.7 / 6.2 / 5.0 / 5.3 | 2.6 / 2.4 / 2.7 / 3.1 / 1.9 |
| 4 cm | 5.5 / 12.2 / 8.5 / 5.3 / 8.2 | 3.3 / 3.4 / 2.7 / 2.4 / 2.8 |
| 5 cm | 7.1 / 20.1 / 15.3 / 14.4 / 12.0 | 4.4 / 5.5 / 5.8 / 5.0 / 3.2 |
| 6 cm | 14.6 / 37.7 / 36.9 / – / – | 11.8 / 16.0 / 11.7 / 13.3 / 9.4 |

- **The answer depends on the tread more than on the count.** On 20 cm treads at 3-4 cm, the duck
  climbs all 20 risers cleanly in 56-60 % of runs and falls in ≤ 3 %. The hazard is flat at about
  2.5-3 % per riser from the first riser to the twentieth, so an unclean step comes about once
  every 35 risers. There is no accumulation: no gait drift and no map degradation 4-5 m from the
  start.
- **On 15 cm treads the same heights give a clean run of 8-10 risers** (20-29 % all clean). The
  15 cm tread is the short end of 11b's tread DR (11-24 cm), and the feet have to fit about 10 cm
  of usable tread (§12).
- **5-6 cm: the first three risers go like the trained flights, then the hazard jumps 2.5-3×**
  (15 cm treads: 7 → 20 % at 5 cm, 15 → 38 % at 6 cm), and the falls follow (median at riser 5-9).
  11b's training flights had at most 3 risers, of at most about 4 cm. On wider treads the jump
  mostly disappears (5 cm at 24 cm treads: about 4 % per riser, flat). So it is the steep,
  continuous flight it never saw, not the count alone.
  - The scan clip is not the cause: the 0.45 m scan holds 3 risers ≤ 15 cm, well within its
    ±0.2 m clip.
- **Single-step height ≠ flight height.** A single 6.4 cm step is 88 % clean for 11b (§16), but a
  6 cm staircase is a wall: 4-37 % reach the top without falling, depending on the tread. 7 cm
  stairs: nobody.
- **2 cm is worse than 3 cm** at every tread (hazard 7-14 % vs 2.5-8.5 %). Mostly these are toe
  catches; a 2 cm riser sits in the map's noise (§16, 11c).
- **11a (the envelope run) is the wrong policy for stairs.** At 2-4 cm it has a median clean run
  of 4-5 and many edge landings (336-359 at 2-3 cm). It lacks 11b's tread DR and placement.
- **Heading.** With a pure vx command 7-23 % of robots wander more than 0.9 m sideways over 3-4.6 m
  of stairs. Any real use (the playground orchestrator) has to hold heading on the stairs.

Video: `scripts/render_staircase.py` (record + render on the gpu box), one typical robot per
staircase picked from 24 — 4 cm / 20 cm clean, 3 cm / 15 cm with a toe catch, 5 cm / 24 cm clean,
5 cm / 15 cm and 6 cm / 15 cm falls — `logs/staircase/video/stairs_11b.mp4`.

**Next for stairs:** train flights of 5-20 risers up to about 5 cm, with the tread DR kept
(11-24 cm) and warm-started from 11b. The bar is a flat hazard at 15 cm treads like the one at
20 cm treads today.

## 18. Up the stairs, turn on top, back down (2026-11-08, `scripts/eval/step_up_roundtrip.py`)

The staircase of §17 (20 risers, tread 20 cm, a 2 m top platform, a 3 m wide tile) with a scripted
round trip:

1. **Climb.** 11b at 0.25 m/s, with a heading hold to 0 (wz ≤ 0.2, its trained range).
2. **Turn.** 0.35 m past the last riser edge the robot stands for 1 s and then turns to heading π
   (done within ±0.08 rad).
3. **Descend.** Heading hold to π at 0.2 m/s, until 0.35 m before the first riser.

Three variants, named by who turns and who goes down. The climb is always 11b.

| variant | turn | down |
| --- | --- | --- |
| `stepup` | 11b, arcing at vx 0.1, wz 0.2 (it never learned to turn in place: zero-command and turn-in-place were off in training) | 11b |
| `orch` | VelStand v8, in place, wz ≤ 0.6 | 11b |
| `velstand` | VelStand v8 | VelStand v8, blind |

The setup: 100 robots per riser height and variant (1200 in all), 75 s per run, and the step-up
head-down command. Rows are in `logs/staircase/roundtrip_11b_t20.json` (gpu box and Mac).

**Turn on top.**
- VelStand turns in place in a median of 3.3–3.5 s (p90 4.0–4.3 s), with no falls and no stalls
  at any height.
- The 11b arc takes 12–14 s (p90 17–22 s). Only 69–78 % of the robots on top finish it at
  2–4 cm (50 % at 5 cm), and up to 4 % fall while turning.

**Down** (counts are of the robots that started the descent):

| riser | `stepup`: ≥ 1 riser down / walked off along the top | `orch`: ≥ 1 riser down / walked off | `velstand`: back at the bottom / fell / median risers before the end |
| --- | --- | --- | --- |
| 2 cm | 10 of 75 / 74 | 6 of 95 / 91 | 31 of 100 / 66 / 11 |
| 3 cm | 9 of 69 / 64 | 6 of 97 / 93 | 0 of 95 / 95 / 3 |
| 4 cm | 5 of 67 / 62 | 5 of 88 / 84 | 0 of 89 / 88 / 2 |
| 5 cm | 1 of 32 / 31 | 1 of 67 / 66 | 0 of 64 / 64 / 2 |

- **11b will not go down.** Not one robot reaches the bottom in either 11b variant.
  - 87–99 % never step off the top riser. They turn away from the drop and walk off along the
    top platform: heading drifts from π to about −1.8 rad while they walk sideways, against a
    heading hold capped at the trained wz ±0.2.
  - This is consistent with 11b's training, which had no step-downs; its ToF map has never seen
    an edge that falls away.
- **VelStand goes down blind and falls.**
  - At 2 cm, 31 % reach the bottom (median 31.5 s), all with edge landings: about one per riser,
    and none clean over 20.
  - From 3 cm every robot falls, after a median of 2–3 risers.
- **The turn itself is solved** by switching to VelStand on top, which is what the playground
  orchestrator already does (`orch`).
- Going down needs a skill of its own: a step-down terrain with the same ToF map.

**Two checks on §17 made while building this test:**
- **Head command.** Every battery before 2026-11-08 sent the head slot as zero, while 11b was
  trained with its head-down command (−0.22, 0.22). Re-running §17 at tread 15 and 20 cm with
  head down gives the same numbers within noise. For example, at tread 20 cm, all 20 risers clean
  at 3 / 4 / 5 cm is 62 / 62 / 40 % with the head down vs 60 / 56 / 38 % with it zero; no fall is
  98 / 90 / 75 % vs 97 / 97 / 73 %. Files: `11b_t15_headdown.json`, `11b_t20_headdown.json`.
- **Truncation.** `out_of_terrain_bounds` is a truncation: mjlab resets the robot without a
  `terminated` flag. The first round-trip recorder did not log it, and a reset robot showed up as
  "back at the bottom" 1.4 s later. `FootRecorder` and the round trip now count truncation as
  leaving the tile. Re-running 11b at tread 20 cm with the fix (`11b_t20_trunccheck.json`) gives
  the same §17 numbers: all 20 clean at 3 / 4 / 5 cm is 59 / 62 / 40 % vs 60 / 56 / 38 %.

Video: `scripts/render_roundtrip.py` (record + render on the gpu box) →
`logs/staircase/video/roundtrip_11b.mp4`. It shows three typical robots, each picked as the one
nearest the median of risers descended among 96 recorded:
- 3 cm, `orch`: walks off along the top;
- 3 cm, `velstand`: falls after 3 risers down;
- 2 cm, `velstand`: back at the bottom with 28 edge landings.

The camera is fixed at the foot of the stairs, so the riser faces stay visible on the way down.

**Next:** a step-down skill. The plan is step-up 11b's recipe with the terrain mirrored (flights
going down, single drops of 1.5–7 cm, the same tread DR and ToF map), warm-started from 11b. After
that, one policy for both directions or an orchestrator switch on the map's sign of the height
change.

## 19. Run 12: teach the climbing policy to go down too (2026-11-08, `Mjlab-StepUp-TofMap-UpDown-*`)

§18 showed that 11b never goes down a staircase. It never met a riser going down in training.

The downhill battery confirms it before training (`step_up_staircase.py --down`: spawn on the top
platform, 20 risers going down, tread 20 cm, 2–5 cm, 800 robots):
- 0 robots descend even one riser.
- 11b stops 8–14 cm before the first edge and drifts sideways off the tile.

**Run 12 = 11b + the same steps mirrored downhill.** It is warm-started from 11b's `model_3999`
(`MICRODUCK_WARM_START=1`, same 327-D obs), with 4096 envs × 4000 iterations on one 5090.

**Terrain.** `BoxSingleStepTerrainCfg(down=True)` builds the same tile with the levels reversed.
The spawn origin is on the upper platform, and the risers go down along +x.

| column | share | what |
| --- | --- | --- |
| `flat` | 10 % | flat ground |
| `single_step` | 25 % | one riser up |
| `short_flight` | 25 % | 3 risers up |
| `single_drop` | 15 % | one riser down |
| `down_flight` | 25 % | 3 risers down |

Riser heights are 1.5–4 cm by row on every column. Treads are 11–24 cm on both flights.

**Rewards.** All of 11b's terms and weights are kept. On the downhill columns:
- The climb's trunk-z potential (`height_progress`) is off. On a downhill tile it would charge the
  descent itself.
- Its mirrors are on, sized like the climb:
  - `descent_progress`: a potential on the exact ground under the two feet, 1 per cm descended.
    It uses the ground under the feet, not trunk z, so a fall is not paid as progress.
  - `foot_below_step`: 2 per foot, once, when the sole stands one riser below the platform and the
    trunk is over the lower ground.
  - `foot_placement` mirrored: inside a downhill flight, it pays touchdowns centred on a tread.
- Edge contact, falls, impacts and the clean-promotion ladder are unchanged; they do not depend on
  direction.

**Spawns.** 80 % of resets put the robot at the origin's level: below the step uphill, above the
drop downhill (`low_frac` now tests |h − origin z|). Spawns also stay on their own tile
(`min_dx` −0.45 m). Behind x = 0 lies the previous row's tile, which meets this one at a cliff or
a wall up to 3 risers high.

**Found while checking the cfg** (`scripts/eval/step_updown_check.py`, 2026-11-08):
- **Foot site vs sole.** The foot sites stand 13 mm above the sole (p5–p95 8–18 mm). The climb
  bonus `foot_on_step` measured the site against `min_rise` 12 mm, so in runs 6–11b it paid
  4.0 per episode on flat ground too: an almost constant bonus, not a step signal.
  - Run 12 measures the sole for both step bonuses (`sole=True`). With the sole, 11b earns 0.08 on
    flat ground and 3.4 on the up columns.
- **Probing.** At iteration 0, 11b probes a drop: it puts a foot over the edge and pulls it back.
  A foot-only bonus paid that without any descent (0.78 per episode on `single_drop` against 0.006
  of descent progress). Hence the trunk gate on `foot_below_step`.

**Iteration 0 (11b on the run 12 training env, its own commands, 1024 robots, 8 s):**

| | flat | single_step | short_flight | single_drop | down_flight |
| --- | --- | --- | --- | --- | --- |
| fell | 0 % | 2 % | 3 % | 10 % | 12 % |
| ≥ 1 riser down | | | | 25 % | 32 % |
| all risers down | | | | 25 % | 18 % |
| foot_edge_contact / episode | −0.04 | −0.64 | −1.22 | −1.74 | −1.53 |

Smoke test (64 envs × 5 it): NaN-free, 327-D obs, every term computes, every penalty ≤ 0, and the
ONNX export works.

### Results (2026-11-09, `model_3999`; run dir `logs/rsl_rl/step_up/2026-11-08_22-00-57_step_up_tofmap_updown` on the gpu box)

Training ran 4000 iterations at 4096 envs. At the end:
- every penalty is ≤ 0;
- mean reward 54.2, episode length 394 of 400 steps;
- terrain level mean 5.6.

The JSONs are in `logs/staircase/` (gpu box and Mac). `step_up_staircase.py --from-json` reprints
the staircase tables, including the edge-contact breakdown of the `--down` runs.

**On the training tiles** (`step_updown_check.py`, 1024 robots, 8 s, its own commands; 11b
iteration 0 → run 12):

| | flat | single_step | short_flight | single_drop | down_flight |
| --- | --- | --- | --- | --- | --- |
| fell | 0 → 0 % | 2 → 0.8 % | 3 → 5.5 % | 10 → 3.2 % | 12 → 3.1 % |
| ≥ 1 riser down | | | | 25 → 88 % | 32 → 96 % |
| all risers down | | | | 25 → 88 % | 18 → 93 % |
| foot_edge_contact / episode | −0.04 → −0.15 | −0.64 → −0.74 | −1.22 → −1.67 | −1.74 → −1.28 | −1.53 → −3.02 |

**Down a 20-riser staircase** (`step_up_staircase.py --down`: spawn on the top platform, 0.25 m/s,
no heading hold, head down; 200 robots per cell, 40–45 s). Before run 12, 0 of 800 robots
descended even one riser.

Each cell is the share that reached the bottom / fell / left the tile sideways.

| riser | tread 20 cm | tread 15 cm |
| --- | --- | --- |
| 2 cm | 83 / 0 / 17 % | 94 / 0 / 7 % |
| 3 cm | 85 / 1 / 15 % | 91 / 7 / 3 % |
| 4 cm | 75 / 11 / 14 % | 68 / 28 / 4 % |
| 5 cm | 47 / 37 / 17 % | 31 / 68 / 2 % |
| 6 cm | 9 / 81 / 11 % | 3 / 97 / 1 % |

- **It goes down, but not cleanly.**
  - There are 0.7–0.8 edge contacts per riser on 20 cm treads and 1.1–1.2 on 15 cm treads.
  - The median clean run is 1–2 risers.
  - No robot descends 10 risers without an edge contact.
- **These are real straddles, not cell-size artefacts of the height field.** The contacts are
  binned by how far the foot reaches across the edge on its shorter side:

  | reach across the edge | ≤ 5 mm | 5–15 mm | 15–27 mm | > 27 mm (half the 54 mm foot) |
  | --- | --- | --- | --- | --- |
  | share of contacts | 9–24 % | 34–39 % | 30–38 % | 11–18 % |

  Counting only straddles of ≥ 5 mm on both sides changes the clean-run table very little.
- **The foot rolls over the nose.** A re-run at tread 20 cm (2 / 3 / 4 / 4.5 cm, 100 robots each,
  `down_12_t20_sole.json`) records where the sole is during each straddle:
  - in 83–94 % of straddles the sole is between the two levels: the foot pivots over the nose of
    the tread, toe down, instead of stepping clear of it;
  - only 5–16 % are flat on the upper tread with the toe over the drop.

  The flat case is the only one the round trip's edge-landing definition counts (sole on the
  tread). That is why the round trip below reports just 0.05–0.12 edge landings per riser on the
  way down.
- **Falls start where training ended.** Training risers were 1.5–4 cm.
  - At 4 cm, 11–28 % fall (the higher figure on 15 cm treads).
  - At 5–6 cm most robots fall.
- **Leaving the tile sideways (2–17 %)** is the pure-vx command with no heading hold, as in §17.
- **Speed is not a factor.** At 0.2 m/s instead of 0.25 the outcomes are the same:
  - reached the bottom: 90 / 91 / 82 / 47 / 6 % at 2–6 cm;
  - fell: 0 / 2 / 7 / 41 / 81 %.

**Up the same staircases** (head down; 11b's head-down rows from §18). Each cell is median
clean run / all 20 risers clean / no fall through 20.

| riser | 11b, tread 20 cm | run 12, tread 20 cm | 11b, tread 15 cm | run 12, tread 15 cm |
| --- | --- | --- | --- | --- |
| 2 cm | 9 / 23 % / 99 % | 9 / 18 % / 98 % | 5 / 14 % / 96 % | 6 / 9 % / 98 % |
| 3 cm | > 20 / 62 % / 98 % | 14 / 40 % / 99 % | 12 / 31 % / 98 % | 8 / 21 % / 95 % |
| 4 cm | > 20 / 62 % / 90 % | 14 / 36 % / 90 % | 9 / 25 % / 88 % | 8 / 13 % / 86 % |
| 5 cm | 16 / 40 % / 75 % | 6 / 11 % / 42 % | 7 / 6 % / 49 % | 4 / 0 % / 19 % |
| 6 cm | 5 / 7 % / 27 % | 3 / 0 % / 3 % | 4 / 1 % / 8 % | 2 / 0 % / 0 % |

- **The climb got worse on long staircases.**
  - At 3–4 cm on 20 cm treads, the hazard of the first unclean riser doubled, from 2–3 % to
    4–6 % per riser.
  - At 5 cm, the share with no fall dropped from 75 % to 42 % (20 cm treads) and from 49 % to
    19 % (15 cm treads).
- **On the training tiles the climb barely moved.** On short_flight, falls went 3 → 5.5 % and
  edge contact −1.22 → −1.67.
- **So the loss is outside training**: past 3 risers and above 4 cm, where 11b had extrapolated.
- **Two causes are possible, neither tested:**
  - run 12 has 50 % uphill tiles against 90 % for 11b;
  - `foot_on_step` now measures the sole, which removed 11b's almost-free 4 per episode.

**Round trip** (§18 setup, variant `orch`: the step-up policy climbs, VelStand turns, the step-up
policy descends; 100 robots per height). Each cell is reached the bottom / fell on the way down,
out of all 100 robots. At 4–5 cm, 15–69 robots never reach the top (most fall on the way up).

| riser | §18: 11b climbs and descends | run 12 climbs and descends | run 12, map cleared at the turn | run 12, no odometry drift |
| --- | --- | --- | --- | --- |
| 2 cm | 0 / 3 (94 walk off along the top) | 87 / 13 | 98 / 0 | 99 / 0 |
| 3 cm | 0 / 3 | 71 / 28 | 98 / 0 | 99 / 1 |
| 4 cm | 0 / 4 | 47 / 36 | 79 / 5 | 81 / 9 |
| 5 cm | 0 / 1 | 5 / 26 | 10 / 17 | 16 / 13 |

- **The falls on the way down came from the map, not the policy.**
  - The ToF map keeps a running mean per cell of everything seen since the episode start.
  - Going down, the robot re-reads the cells it mapped on the way up. Those were written under a
    different odometry error: z drift and yaw drift grow with every metre walked, in either
    direction.
  - Clearing the map when the turn starts brings falls to 0 % at 2–3 cm, the same as perfect
    odometry (`--map-reset-at-turn`, `--no-drift`).
  - Training never re-reads a cell: episodes are 8 s and straight.
- **For the runtime:** the robot's map has to forget. Clear it at a turn-around or skill switch,
  or age cells out.

**11b up, run 12 down** (`--checkpoint-down`, map cleared at the turn,
`roundtrip_11bup_12down_mapreset.json`). This compares the paired policies with run 12 alone,
same setup. Each cell is reached the bottom / climbed all 20 risers clean, out of 100 robots.

| riser | run 12 up and down | 11b up, run 12 down |
| --- | --- | --- |
| 2 cm | 98 / 25 | 95 / 31 |
| 3 cm | 98 / 55 | 95 / 68 |
| 4 cm | 79 / 27 | 73 / 57 |
| 5 cm | 10 / 5 | 25 / 35 |

- At 2–3 cm, both reach the bottom in 95–98 % of runs.
- The pair climbs more cleanly at every height, and twice as many pairs reach the top at 5 cm
  (62 vs 31).
- Until a single policy does both as well as this, the orchestrator can switch policies on the
  sign of the riser in the map.

Video: `logs/staircase/video/roundtrip_12.mp4` (97 s; `scripts/render_roundtrip.py`, recorded
with `--map-reset-at-turn`). It shows run 12 plus a VelStand turn, three typical robots picked
from 40 per height:
- 2 cm, back at the bottom with 23 foot contacts across an edge;
- 3 cm, back at the bottom with 21 contacts, 1 of them flat on the nose;
- 4 cm, falls after 10 risers down.

Then the 3 cm descent again from a low side camera, without the ToF overlay. On the way down a
foot's halo turns red at any contact across an edge.

**Next for run 12:**
1. **A clean descent.** The policy pays its edge cost (−3.0 per episode on `down_flight`) and
   still rolls each foot over the nose. Two options:
   - charge the downhill straddle harder;
   - pay `descent_progress` only for steps that touched no edge, i.e. a clean-gated potential.
2. **Keep the climb.** Either restore 11b's uphill share, or add flights longer than 3 risers in
   both directions. Re-check the climb against 11b on the §17 staircases at 15 and 20 cm treads.
3. **Runtime map.** Forget cells on a turn-around (see above) before any real stair descent.

## 20. Run 13: a clean descent that keeps 11b's climb (2026-11-09, `Mjlab-StepUp-TofMap-UpDownClean-*`)

Run 12 goes down 20 risers, but it rolls a foot over almost every nose, and its climb lost ground
against 11b on long staircases (§19). Run 13 targets both, warm-started from run 12's `model_3999`
(`MICRODUCK_WARM_START=1`, same 327-D obs, same terrain and spawns), with 4096 envs × 3000
iterations on one 5090. The factory flag is `clean_down=True` (it requires `updown=True`).

**1. Charge the downhill straddle harder.** On the downhill columns, `foot_edge_contact` is
multiplied by 2 (`DOWN_EDGE_SCALE`; params `scale_tiles` / `tile_scale`). The counter behind the
clean-promotion ladder still counts contact steps unscaled, so promotion is unchanged.

**2. Pay only the risers a foot steps down cleanly.** The new `clean_step_down` term
(`mdp.clean_step_down_bonus`, weight 25) works per foot, on the downhill columns only:
- The foot's first contact in the episode sets its reference level.
- A later touchdown on a level at least `min_rise` (12 mm) below that reference pays the drop:
  0.5 per cm per foot.
- It pays only if the touchdown is clean: no `foot_edge_flags` (the sole's corners and centre read
  heights within 5 mm of each other) and the trunk upright (tilt < 40°).
- Any contact on a lower level lowers the reference, clean or not. A foot that lands across an
  edge therefore forfeits that riser's pay for good: crossing it again later pays nothing.

On a 3-riser flight at 3 cm, a perfectly clean descent earns 9 cm × 2 feet × 0.5 = 9 per episode
on top of run 12's terms. Run 12's `descent_progress` (1 per cm, edge-blind) stays, so stepping
down is still paid when it is not clean; the clean way is just worth about twice as much.

**3. Keep the climb: a behaviour-cloning anchor toward 11b** (`tasks/climb_anchor.py`,
`PpoWithClimbAnchor`):
- After every PPO update, a separate pass (own Adam, lr 3e-4, 5 epochs × 4 mini-batches, MSE on
  the deterministic action) regresses the actor onto 11b's actions.
- The pass uses only the rollout frames whose scan shows no drop ahead. Uphill, flat and the
  approach to a drop stay 11b's; every frame whose scan shows a drop belongs to PPO alone.
- The split is a pure function of the actor obs (`downhill_ahead_mask`): a frame counts as
  downhill when the mean of any scan row ahead (x > 3 cm, at least 3 of 7 cells known) reads more
  than 8 mm below the mean under the robot. The lowest training riser is 15 mm.
- **Row means, not single cells.** The first version took the maximum over the ~100 cells ahead.
  Map noise then flagged 29 % of flat frames as downhill, so only 71 % of flat frames were
  anchored. With row means, 93 % are.

**Iteration 0** (run 12 on the run 13 training env, `step_updown_check.py --anchor`, 1024 robots,
8 s, its own commands):

| | flat | single_step | short_flight | single_drop | down_flight |
| --- | --- | --- | --- | --- | --- |
| fell | 0 % | 3.1 % | 1.6 % | 2.6 % | 2.0 % |
| all risers down | | | | 94 % | 92 % |
| frames anchored to 11b | 93 % | 93 % | 93 % | 78 % | 65 % |
| run 12 vs 11b action MSE (rad²) | 0.014 | 0.017 | 0.022 | 0.016 | 0.017 |
| `clean_step_down` / episode | 0 | 0 | 0 | 1.28 | 3.86 |
| `foot_edge_contact` (× 2 downhill) / episode | −0.06 | −0.90 | −1.42 | −2.27 | −5.51 |

- `clean_step_down` at iteration 0 is 1.28 of a possible 1.5–4 on `single_drop` (one riser of
  1.5–4 cm) and 3.86 of 4.5–12 on `down_flight`. The term starts visibly non-zero and has room to
  grow.
- 11b scored against itself gives an MSE of exactly 0, which checks the anchor's loading path.
  On the same env, 11b earns 0.07 / 0.19 of `clean_step_down` (it rarely descends).
- Smoke test (64 envs × 5 iterations): NaN-free, the anchor loads, the BC loss falls 0.066 →
  0.020, every penalty ≤ 0, and the ONNX export works.
- At the start of training, 76 % of the rollout frames are anchored, and the BC loss sits at
  0.005–0.010.

### The anchor broke the descent (checks during training, 2026-11-09)

`step_updown_check.py` on intermediate checkpoints (1024 robots, 8 s, own commands; logs in
`logs/clean13/check_*.log` on the gpu box). Three runs, all warm-started from run 12:
- **13 (strong anchor):** the design above, every frame without a drop ahead, 20 Adam steps per
  iteration at lr 3e-4.
- **13n (no anchor):** the same task with `MICRODUCK_CLIMB_ANCHOR=off`: only the two reward changes.
- **13b (uphill anchor):** `MICRODUCK_CLIMB_ANCHOR_CFG=mode=uphill,epochs=2,learning_rate=1e-4`.
  It anchors only frames whose scan shows a step UP ahead (15 % of the rollout frames), at about
  7× less BC strength.

`down_flight` (3 risers down). Each cell is fell / all risers down / `clean_step_down` / edge cost
(× 2), per episode:

| checkpoint | 13, strong anchor | 13n, no anchor | 13b, uphill anchor |
| --- | --- | --- | --- |
| 0 (= run 12) | 2.4 % / 89 % / 3.86 / −5.5 | same | same |
| 250 | 25 % / 68 % / 2.17 / −10.5 | 3.9 % / 95 % / 3.91 / −5.1 | 3.5 % / 91 % / 3.52 / −7.7 |
| 500 | | 3.1 % / 93 % / 3.80 / −4.5 | 7.5 % / 89 % / 3.22 / −7.3 |
| 1000 | 20 % / 55 % / 1.43 / −9.4 | 4.7 % / 95 % / 4.32 / −4.7 | 5.5 % / 88 % / 3.35 / −6.8 |

Edge cost per episode on the uphill columns, `single_step` / `short_flight`:

| checkpoint | 13, strong anchor | 13n, no anchor | 13b, uphill anchor |
| --- | --- | --- | --- |
| 0 (= run 12) | −0.90 / −1.42 | same | same |
| 250 | −0.91 / −2.12 | −0.54 / −1.39 | −0.88 / −1.22 |
| 1000 | −0.52 / −1.76 | −0.67 / −1.25 | −0.42 / −1.16 |

- **The strong anchor turned the policy back into 11b.**
  - Its action MSE to 11b on anchored frames fell from 0.017 to 0.001 by iteration 1000.
  - The descent fell apart with it: 11b's own iteration 0 on this env is 13 % falls and 18 % all
    risers down on `down_flight`.
  - The frames whose scan shows a drop (35–40 % on `down_flight`) were left to PPO. That was not
    enough: the network is shared, and 20 BC steps per iteration on 75 % of the data outweighed
    PPO's advantage gradient.
  - Stopped at iteration 1773.
- **Without the anchor the two reward changes clean the descent up a little,** and the descent
  stays as safe as run 12's.
- **The uphill anchor keeps the uphill tiles cleanest,** but costs about a third more edge contact
  on the descent than no anchor.
- The climb that regressed in run 12 is the 20-riser staircase. The training tiles do not show
  it, so both runs go through the §17–§19 staircase battery.

### 13b at iteration 2000: the best climber so far, a worse descender (2026-11-09)

Run dir: `logs/rsl_rl/step_up/2026-11-09_02-51-10_step_up_tofmap_cleandown_upanchor/model_2000.pt`
(gpu box). JSONs: `logs/staircase/13u_*`, `down_13u_*`, `roundtrip_13u_t20_mapreset.json`,
`roundtrip_11bup_13udown_mapreset.json`. Same batteries and seeds as §19.

**Up 20 risers, head down.** Each cell is median clean run / all 20 risers clean / no fall
through 20 (11b and run 12 from §19):

| riser | 11b, t20 | run 12, t20 | 13b, t20 | 11b, t15 | run 12, t15 | 13b, t15 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 cm | 9 / 23 % / 99 % | 9 / 18 % / 98 % | 13 / 38 % / 100 % | 5 / 14 % / 96 % | 6 / 9 % / 98 % | 11 / 30 % / 99 % |
| 3 cm | > 20 / 62 % / 98 % | 14 / 40 % / 99 % | > 20 / 72 % / 98 % | 12 / 31 % / 98 % | 8 / 21 % / 95 % | 15 / 39 % / 95 % |
| 4 cm | > 20 / 62 % / 90 % | 14 / 36 % / 90 % | > 20 / 66 % / 91 % | 9 / 25 % / 88 % | 8 / 13 % / 86 % | 14 / 35 % / 90 % |
| 5 cm | 16 / 40 % / 75 % | 6 / 11 % / 42 % | > 20 / 51 % / 84 % | 7 / 6 % / 49 % | 4 / 0 % / 19 % | 8 / 20 % / 62 % |
| 6 cm | 5 / 7 % / 27 % | 3 / 0 % / 3 % | 8 / 20 % / 37 % | 4 / 1 % / 8 % | 2 / 0 % / 0 % | 5 / 2 % / 17 % |

- **13b climbs better than 11b at every height on both treads.** At 5 cm on 20 cm treads, 51 % of
  the robots climb all 20 risers clean (11b 40 %, run 12 11 %), and 84 % do not fall (75 %, 42 %).
- The run 12 regression is gone. The uphill anchor gives back 11b's climb, and PPO keeps
  improving on top of it.

**Down 20 risers.** Each cell is reached the bottom / fell / left the tile sideways, then edge
contacts per riser:

| riser | run 12, t20 | 13n@1000, t20 | 13b, t20 | run 12, t15 | 13b, t15 |
| --- | --- | --- | --- | --- | --- |
| 2 cm | 83 / 0 / 17 %, 0.83 | 86 / 0.5 / 14 %, 0.52 | 91 / 0.5 / 9 %, 0.59 | 94 / 0 / 7 %, 1.18 | 93 / 0.5 / 7 %, 1.13 |
| 3 cm | 85 / 1 / 15 %, 0.75 | 84 / 4 / 12 %, 0.43 | 81 / 6 / 13 %, 0.71 | 91 / 7 / 3 %, 1.16 | 79 / 19 / 3 %, 1.19 |
| 4 cm | 75 / 11 / 14 %, 0.67 | 69 / 16 / 15 %, 0.43 | 51 / 38 / 12 %, 0.84 | 68 / 28 / 4 %, 1.08 | 24 / 68 / 9 %, 1.26 |
| 5 cm | 47 / 37 / 17 %, 0.78 | 27 / 61 / 13 %, 0.63 | 14 / 75 / 12 %, 1.07 | 31 / 68 / 2 %, 1.14 | 5 / 91 / 5 %, 1.29 |
| 6 cm | 9 / 81 / 11 %, 1.10 | 5 / 86 / 9 %, 0.94 | 2 / 92 / 7 % | 3 / 97 / 1 %, 1.31 | 0 / 98 / 3 % |

- **13b descends worse than run 12 from 3 cm up.** At 4 cm on 20 cm treads, 38 % fall (run 12
  11 %). On 15 cm treads, 68 % fall (run 12 28 %).
- **No variant descends cleanly.** In 81–94 % of the straddles, the sole is still tipped between
  the two levels: the foot rolls over the nose of the tread. The median clean run is 2–3 risers.
- **13n@1000 has the fewest edge contacts per riser:** 0.43–0.52 at 2–4 cm, against run 12's
  0.67–0.83, i.e. 35–43 % fewer. 13b has fewer than run 12 only at 2 cm, and more from 4 cm up. Its falls at 4–5 cm are between run 12's and 13b's. 13n continues to 3000
  iterations; its full battery follows.

**Round trip** (orch, map cleared at the turn, 100 robots per height). Each cell is reached the
top / reached the bottom / fell on the way down:

| riser | run 12 alone | 13b alone | 11b up, 13b down |
| --- | --- | --- | --- |
| 2 cm | 98 / 98 / 0 | 98 / 97 / 0 | 98 / 98 / 0 |
| 3 cm | 99 / 98 / 0 | 99 / 90 / 8 | 99 / 87 / 9 |
| 4 cm | 85 / 79 / 5 | 88 / 55 / 32 | 89 / 45 / 40 |
| 5 cm | 31 / 10 / 17 | 78 / 13 / 60 | 58 / 3 / 50 |

- 13b reaches the top of 5 cm stairs in 78 % of runs. 11b manages 58–62 % (here and in §19's
  pair), run 12 31 %.
- 13b's descent costs the round trip from 3 cm up.
- The natural pairing is 13b up, with run 12 or 13n down. It is not measured yet.

### 13n at iteration 2999 (no anchor, 2026-11-09)

Run dir `logs/rsl_rl/step_up/2026-11-09_03-22-46_step_up_tofmap_cleandown_noanchor_r/model_2999.pt`
(resumed from 13n `model_1000`). JSONs: `logs/staircase/13n_*`, `down_13n_*`,
`roundtrip_13n_t20_mapreset.json`, `roundtrip_11bup_13ndown_mapreset.json`.

**Down 20 risers.** Each cell is reached the bottom / fell / left the tile, then edge contacts per
riser, then the median clean run:

| riser | run 12, t20 | 13n, t20 | run 12, t15 | 13n, t15 |
| --- | --- | --- | --- | --- |
| 2 cm | 83 / 0 / 17 %, 0.83, 1 | 88 / 0 / 12 %, 0.32, 4 | 94 / 0 / 7 %, 1.18 | 97 / 1.5 / 1.5 %, 0.81 |
| 3 cm | 85 / 1 / 15 %, 0.75, 2 | 87 / 5.5 / 7.5 %, 0.25, 5 | 91 / 7 / 3 %, 1.16 | 89 / 10.5 / 1 %, 0.77 |
| 4 cm | 75 / 11 / 14 %, 0.67, 2 | 54 / 28 / 19 %, 0.34, 4 | 68 / 28 / 4 %, 1.08 | 59 / 40 / 2 %, 0.68 |
| 5 cm | 47 / 37 / 17 %, 0.78 | 20 / 69 / 12 %, 0.53, 4 | 31 / 68 / 2 %, 1.14 | 16 / 84 / 0.5 %, 0.72 |

- On 20 cm treads at 2–4 cm, 13n has 49–67 % fewer edge contacts per riser than run 12, and its
  median clean run is 4–5 risers instead of 1–2.
- It falls more from 3 cm up: at 4 cm 28 % against run 12's 11 %.

**Up 20 risers, t20** (median clean run / all 20 clean / no fall):
- 13n: 2 cm 11 / 31 % / 99 %, 3 cm 13 / 38 % / 97 %, 4 cm 13 / 33 % / 90 %, 5 cm 8 / 14 % / 63 %.
- That is about run 12's climb, slightly better at 5 cm, and below 11b and 13b.

**Round trip** (map cleared at the turn). Each cell is reached the top / reached the bottom / fell
on the way down:

| riser | 13n alone | 11b up, 13n down |
| --- | --- | --- |
| 2 cm | 98 / 98 / 0 | 95 / 95 / 0 |
| 3 cm | 97 / 87 / 9 | 96 / 92 / 1 |
| 4 cm | 87 / 51 / 28 | 90 / 46 / 40 |
| 5 cm | 48 / 10 / 35 | 65 / 10 / 52 |

### What the descent's edge contacts are (2026-11-09, `scripts/eval/step_down_contacts.py`)

Before deciding how to train the descent clean, the contacts were measured on the 20-riser
staircase: tread 20 cm, 2 / 3 / 4 cm, 200 robots per height, 0.25 m/s, head down. JSONs:
`logs/staircase/contacts_{12,12_hard,13u,13n}_t20.json`.

The script sorts each edge contact two ways:
- **landing** (the foot touches down already across the edge) or **stance** (it stood clean on
  the tread, then went over the nose while loaded);
- **leading** (the other foot still above the edge) or **trailing** (the other foot already below
  it).

For every crossing of an edge by one foot, it measures how far behind the edge the toe left
(lift-off) and how far beyond it the heel landed. Both must be ≥ 0 for a clean step-off.

**The share of contacts in each class.** Each cell is leading landing / trailing landing /
leading stance / trailing stance:

| riser | run 12 | run 12, hard terrain | 13b | 13n |
| --- | --- | --- | --- | --- |
| 2 cm | 63 / 4 / 30 / 2 % | 65 / 4 / 29 / 2 % | 61 / 4 / 33 / 2 % | 63 / 4 / 30 / 2 % |
| 3 cm | 68 / 5 / 26 / 2 % | 66 / 5 / 28 / 2 % | 57 / 11 / 29 / 3 % | 59 / 16 / 20 / 4 % |
| 4 cm | 71 / 5 / 22 / 2 % | 67 / 7 / 23 / 2 % | 54 / 15 / 26 / 5 % | 60 / 18 / 19 / 3 % |

**Crossings by the leading foot, the step off the edge.** Each cell is clean % / lift-off behind
the edge p10–p50 / landing beyond it p10–p50 / stride p50 (mm):

| riser | run 12 | 13b | 13n |
| --- | --- | --- | --- |
| 2 cm | 72 % / 38–68 / −20–27 / 156 | 81 % / 28–72 / −13–45 / 181 | 91 % / 43–69 / 2–52 / 183 |
| 3 cm | 78 % / 27–66 / −17–43 / 169 | 80 % / 17–69 / −13–52 / 185 | 93 % / 41–70 / 8–62 / 193 |
| 4 cm | 77 % / 15–64 / −16–50 / 174 | 79 % / 15–72 / −14–58 / 187 | 93 % / 31–64 / 15–72 / 198 |

**The leading foot landing on the nose is the edge contact.**
- 54–71 % of all contacts are the leading foot touching down across the edge.
- Another 19–33 % are the leading foot that stood on the tread and slid over it (median 7–12 mm).
- The trailing foot rolling off the nose behind the robot accounts for only 2–5 %.
- The trailing foot landing across the edge is 4–7 % in run 12, but 11–18 % for 13b and 13n at
  3–4 cm.

**These contacts are not rolls.**
- The loaded foot's pitch in an edge contact is p50 −1 to −5°, p10 −7 to −22°.
- On flat ground the same walks pitch their loaded soles by p10–p90 −6 to +6° (run 12), −8 to
  +7° (13n) and −10 to +9° (13b).
- A "keep the foot flat" term would barely tell the two apart. The tipped-between-levels soles
  of §19 are feet resting with part of the sole on the nose, mostly a few degrees off flat.

**The soft terrain is not the reason.** With the terrain contacts back at MuJoCo's defaults
(`--hard-terrain`, solref 0.02), run 12 behaves the same:
- edge contacts 0.84 / 0.78 / 0.69 per riser against 0.81 / 0.75 / 0.70;
- falls 0.5 / 2 / 12 % against 0 / 3 / 14 %;
- 0 NaN terminations.

**No policy walks up to the edge before stepping off.**
- The leading foot lifts off a median 63–72 mm behind the edge in every policy, unchanged by
  training. A duck would close that gap and then step down normally.
- For run 12, a clean step-off is close to chance. With a 156–174 mm stride, a lift-off spread
  over one stride clears the 54 mm foot in about (stride − 54) / stride = 65–69 % of crossings;
  72–78 % are clean.
- 13n learned to land beyond the edge (p10 +2 to +15 mm, 91–93 % clean) by never taking a short
  step across it. Among robots that reached the bottom, its stride p10 is 151 mm against run 12's
  98–101 mm. It lengthened the stride and kept the lift-off distance.

**Falls.**
- In every policy, the faller's last step-off started further back (lift-off p50 68–85 mm) and
  landed short: landing p50 −8 to +40 mm, against +43 to +73 mm for robots that reached the bottom.
- 13n has fewer nose landings but survives them worse. Of its unclean step-offs, 3.3 % (3 cm)
  and 11 % (4 cm) end in a fall, against 0.5 % and 1.5 % for run 12: about 7× more.
- Run 12 practises landing on the nose every few steps; 13n stopped practising it.
- At 4 cm, 2 of 3 of 13n's falls follow a clean last step-off. That part is not explained yet.

**Ideas this measurement rules out:**
- a foot-flat term;
- un-softening the terrain;
- a trailing-foot (push-off) fix.

**What it points to:** the lift-off distance in front of the edge, i.e. the steps before the
drop. That is how a duck descends: shorten the steps up to the edge, then step down normally.

### Does a slower command make the step-off cleaner? No (2026-11-09)

The same contact test at 0.10 and 0.15 m/s; the training range is 0.1–0.3. JSONs:
`logs/staircase/contacts_*_t20_v010.json` and `contacts_*_t20_v015.json`. The table comes from
`scripts/eval/step_down_speed_compare.py`.

The "chance" column is the clean rate with no planning at all: a lift-off spread uniformly over
one stride, computed from each crossing's own stride as max(0, s − 54) / s.

Leading step-offs at 3 cm. Each cell is clean % (chance %) / lift-off behind the edge p50 / stride
p50 (mm), then edge contacts per riser:

| vx | run 12 | 13b | 13n |
| --- | --- | --- | --- |
| 0.10 | 71 % (58 %) / 57 / 138, 1.23 | 78 % (65 %) / 78 / 169, 1.11 | 90 % (69 %) / 78 / 185, 0.41 |
| 0.15 | 75 % (61 %) / 66 / 154, 1.04 | 79 % (65 %) / 76 / 178, 0.89 | 93 % (70 %) / 73 / 188, 0.30 |
| 0.25 | 78 % (63 %) / 66 / 169, 0.75 | 80 % (65 %) / 69 / 185, 0.76 | 93 % (71 %) / 70 / 193, 0.28 |

- **Slower is not cleaner, for any policy.** Edge contacts per riser rise as the command slows,
  or stay the same (13n at 4 cm: 0.37 / 0.38 / 0.33).
  - Run 12 at 2 cm: 0.81 → 1.43 contacts per riser. Its step-offs there are clean exactly at
    chance: 43 % against 43 %.
- **None of them approaches the edge.** The lift-off stays a median 42–83 mm behind the edge at
  every speed. 13n even stops further back at 0.10 m/s (78–83 mm at 2–3 cm).
  - The 42 mm is run 12 at 0.10 m/s on 2 cm. It is that close only because its stride shrank to
    100 mm, and those step-offs are clean exactly at chance.
- **What the policies do plan is the step-off itself.** The stride across the edge stays long
  at low speed:
  - 13n: 176–192 mm at 0.10 m/s;
  - run 12 at 4 cm: 160 mm, against 100 mm at 2 cm.

  13n's +20 points over chance come from that one long step across the edge, at every speed. A
  duck instead shortens its steps up to the edge and then steps down normally.
- Falls barely depend on speed. Run 12 at 4 cm falls 14–16 %. 13n at 4 cm falls 24 % at 0.10 m/s
  and 38 % at 0.15 and 0.25.

**So relaxing the speed tracking near drops would not, by itself, bring out the duck's descent.**
The policies have learned to solve the edge in one step (13n: a long step over it) rather than
over the approach.

### The step-off is a drop, not a step (2026-11-09)

`step_down_contacts.py` now also records, around every touchdown:
- the peak foot normal force over the next 0.1 s;
- the peak trunk |a_z| (finite difference of v_z, as the training's `trunk_impact`) from 0.04 s
  before to 0.1 s after;
- the trunk v_z just before the touchdown.

The runs use 20 risers, tread 20 cm, 0.25 m/s, 200 robots per height (`logs/staircase/impact_*_t20.json`).
"Flat tread" means touchdowns on a tread at least 3 cm from any edge.

Leading step-offs (the foot that goes down first) against flat-tread touchdowns. Each cell is foot
force p50 (p90) / trunk |a_z| p50 / trunk v_z before, p50:

| riser | | run 12 | 13b | 13n |
| --- | --- | --- | --- | --- |
| 2 cm | flat tread | 19 (29) N / 13 / −0.20 | 17 (26) N / 11 / −0.16 | 15 (23) N / 9 / −0.13 |
| | leading step-off | 31 (50) N / 28 / −0.61 | 31 (56) N / 30 / −0.68 | 37 (62) N / 36 / −0.79 |
| 3 cm | flat tread | 20 (31) N / 14 / −0.22 | 16 (26) N / 11 / −0.17 | 15 (24) N / 9 / −0.15 |
| | leading step-off | 36 (63) N / 36 / −0.80 | 34 (63) N / 35 / −0.81 | 42 (71) N / 40 / −0.93 |
| | trailing step-off | 28 (46) N / 23 / −0.37 | 24 (40) N / 18 / −0.29 | 24 (40) N / 19 / −0.26 |
| 4 cm | flat tread | 19 (31) N / 14 / −0.24 | 17 (27) N / 12 / −0.18 | 15 (24) N / 11 / −0.16 |
| | leading step-off | 40 (68) N / 39 / −0.91 | 37 (66) N / 38 / −0.89 | 42 (75) N / 43 / −1.02 |

- **The trunk free-falls the riser height onto the leading foot.**
  - Free fall from 2 / 3 / 4 cm ends at 0.63 / 0.77 / 0.89 m/s.
  - Before a leading step-off the trunk comes down at 0.61–0.91 m/s (run 12) and 0.79–1.02 m/s
    (13n), 3–6 times its v_z on a flat tread.
  - The stance leg on the upper tread does not lower the body; the robot drops.
- **The leading foot lands with 1.6–2.8× the flat-ground force:** p50 31–42 N (4–5× the robot's
  7.8 N weight), p90 50–75 N (6–10×). Trunk |a_z| is 2–4× the flat value.
- **The trailing foot steps down gently.** It comes down at 0.26–0.37 m/s at 3 cm, close to the
  flat walk; the body is already low by then. The robot can lower itself; it just does not do it
  for the first foot.
- **Longer step-offs land harder.** At every height and in every policy, leading strides ≥ 180 mm,
  against < 150 mm:
  - force p50 +30–55 %;
  - |a_z| +30–60 %;
  - v_z −0.73 to −1.05 m/s against −0.42 to −0.74.

  13n takes the most long step-offs (55–70 % of its leading crossings are ≥ 180 mm) and lands
  hardest.
- **Training charges almost nothing for this.**
  - `trunk_impact` (−|a_z|, weight 0.02) sums to −0.56 per episode on `down_flight` against −0.45
    on flat ground (run 12 at iteration 0).
  - A whole 3-riser descent of free-fall step-offs costs about 0.1 more than walking on flat,
    against ~10 of descent pay.

**The duck way, in these terms:** the stance leg lowers the body while the front foot reaches
down, so the trunk never drops. The trailing foot already does that. The leading step-off, the
long step and the nose landing all come from the same shortcut: falling down the step instead of
stepping down it. Nothing in training prices the fall.

## 21. Run 14: lower the body, do not drop it (2026-11-09, `Mjlab-StepUp-TofMap-UpDownSoft-*`)

§20 ended on one shortcut behind the nose landings, the long step-offs and the hard leading
touchdowns. The robot falls down the step instead of stepping down it, and nothing in training
prices that fall. Run 14 prices it and leaves everything else to the policy. There is no
foot-placement target and no "approach the edge" term: where the feet go is for the policy to
find.

**The term (`trunk_drop_penalty`, mdp.py):**
- It is the trunk's downward speed beyond `TRUNK_DROP_MAX_VEL` = 0.3 m/s.
- It applies only on the downhill columns (`single_drop`, `down_flight`), after the first 0.2 s of
  an episode, while the spawn settles.
- It is self-negating, so it takes a positive weight. Summed over an episode with dt, it is the
  metres the trunk fell faster than the cap: weight W costs W/100 per cm of such fall.
- The cap sits above the trailing foot (0.21–0.42 m/s at touchdown) and above flat treads
  (0.13–0.24 m/s), and below a leading drop (0.61–1.02 m/s). A 3 cm free fall exceeds it for
  about 1.1 cm of the fall.
- Lowering the body with the stance leg while the front foot reaches down pays nothing. Falling
  over the edge pays.

**Everything else is run 13** (`soft_down` needs `clean_down`). Same 327-D obs and terrain menu,
the edge cost ×2 on the downhill columns, `clean_step_down`.

**Runner.** The climb anchor now points at 13b itself (`ckpts/stepup_13b_model_2000.pt`), the best
climber in §20. It anchors uphill frames only, with 13b's light BC (2 epochs, lr 1e-4).
`MICRODUCK_CLIMB_ANCHOR=off` now switches the anchor off even when the cfg names a checkpoint.

**Two runs**, both warm-started (`MICRODUCK_WARM_START=1`):
- **14b:** from 13b `model_2000`, with the anchor (GPU 0).
- **14n:** from 13n `model_2999`, without the anchor (GPU 1). 13n takes the longest step-offs and
  lands the hardest.

### Sizing at iteration 0 (2026-11-09, `step_updown_check.py`, 2048 envs, 8 s, own commands)

The table gives the cm per episode the trunk fell faster than 0.3 m/s, on every column (the term
ungated), and the descent pay on `down_flight` (`descent_progress` + `foot_below_step` +
`clean_step_down`).

| policy | flat | single_step | short_flight | single_drop | down_flight | descent pay, down_flight |
| --- | --- | --- | --- | --- | --- | --- |
| 11b | 0.27 | 0.51 | 0.79 | 0.68 | 1.32 | 2.2 (descends 37 %) |
| run 12 | 0.37 | 0.58 | 0.83 | 1.61 | 4.72 | 14.1 |
| 13b `model_2000` | 0.44 | 0.61 | 0.83 | 1.78 | 4.52 | 13.7 |
| 13n `model_2999` | 0.35 | 0.46 | 0.72 | 2.25 | 6.08 | 15.3 |

- **The drop is the descent's.** On a down flight the trunk falls 4.5–6.1 cm per episode beyond
  the cap. Walking the uphill and flat columns gives 0.3–0.8 cm.
  - 13n, the long-stride descender, drops the most.
  - The in-env term at weight 100 reads −4.52 on 13b's `down_flight`, matching the ungated 4.52 cm:
    the gating and the settle window are right.
- **Choice of cap** (13b, `down_flight` vs flat):

  | cap | down_flight | flat |
  | --- | --- | --- |
  | 0.2 m/s | 7.9 cm | 2.0 cm |
  | 0.3 m/s | 4.5 cm | 0.44 cm |
  | 0.4 m/s | 2.8 cm | 0.11 cm |

  A 0.2 cap would tax the ordinary gait; at 0.4 a 2–3 cm drop barely shows. 0.3 is kept.
- **Weight: 150** (1.5 per cm). At iteration 0 that costs 13b −6.8 per `down_flight` episode
  (single_drop −2.7) and 13n −9.1 (−3.4). That is about half the descent pay, next to the
  downhill edge cost (13b −5.3).
  - Descending stays worth it: stopping at the top loses the descent pay and the speed tracking
    (track_vx_ema ≈ 27 per episode).
  - Stepping down without a drop pays it all.

**Smoke test** (64 envs × 5 iterations, both runs). Both build and step NaN-free.
- 14b loads the anchor (`mode=uphill`, 13b, 2 epochs, lr 1e-4).
- 14n prints `[climb_anchor] OFF`.
- `trunk_drop` reads ≤ 0, and the ONNX export goes through.

**Training** (2026-11-09, gpu box, tmux `md_run14b` / `md_run14n`, 4096 envs, 2000 iterations,
about 2.3 s per iteration):
- **14b:** `logs/rsl_rl/step_up/2026-11-09_13-12-44_step_up_tofmap_updown_soft_13b`
- **14n:** `logs/rsl_rl/step_up/2026-11-09_13-12-44_step_up_tofmap_updown_soft_13n`

`logs/soft14/chain.sh` checks each run at 250 / 500 / 1000 / 1500 with `step_updown_check.py`. On
the last checkpoint it runs the §20 set: contacts and impacts, the 20- and 15-riser stairs up and
down, and the round trip.

### Checks at iterations 250, 500 and 1000 (2026-11-09, 1024 envs, 8 s)

The table gives cm per episode the trunk fell faster than 0.3 m/s (ungated), then the
`down_flight` column.

| | flat (ungated) | single_drop | down_flight | ≥1 riser down | fell | edge cost | descent pay |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13b (it 0, 2048 envs) | 0.44 | 1.78 | 4.52 | 95 % | 4.1 % | −5.3 | 13.7 |
| 14b @ 250 | 0.26 | 1.26 | 3.71 | 91 % | 3.5 % | −5.9 | 12.8 |
| 14b @ 500 | 0.31 | 1.39 | 3.58 | 84 % | 4.3 % | −4.9 | 12.4 |
| 14b @ 1000 | 0.19 | 1.23 | 3.56 | 94 % | 5.9 % | −5.6 | 13.8 |
| 13n (it 0, 2048 envs) | 0.35 | 2.25 | 6.08 | 96 % | 3.5 % | −3.2 | 15.3 |
| 14n @ 250 | 0.12 | 1.24 | 3.20 | 91 % | 3.5 % | −3.5 | 14.3 |
| 14n @ 500 | 0.16 | 1.13 | 3.18 | 93 % | 4.3 % | −3.9 | 14.5 |
| 14n @ 1000 | 0.09 | 1.02 | 2.90 | 97 % | 4.7 % | −3.5 | 15.4 |

- **The drop is coming down.** On `down_flight`:
  - 14n falls 48 % less beyond the cap than 13n, almost all of it within 250 iterations.
  - 14b falls 21 % less than 13b.
- **The descent holds.** Speed tracking on `down_flight` is unchanged (14b 27.3, 14n 26.8,
  against 26.7 / 26.6 at iteration 0), so the robots are not just walking slower. Falls are
  unchanged within noise.
  - **14b to watch:** its share of robots that get at least one riser down slipped from 95 % to
    84 % at 500 (n = 255). It was back at 94 % at 1000, so it was noise, not stopping at the top.
- **The climb holds.** On `short_flight`, height progress is 6.8 / 6.7 (13b at iteration 0:
  7.0). 14b's anchor MSE stays at 0.002–0.007.
- **At 1000 the two runs part.** 14b has stalled at about −21 %. 14n keeps going: 2.90 cm,
  −52 % against 13n. Its `clean_step_down` (5.1) is the highest of any run so far.
- The edge cost has not moved yet (14b −4.9 to −5.6, 14n −3.5 to −3.9). Whether a softer drop also lands the
  front foot off the nose is the §20 contact test on the last checkpoint.

### Final checkpoints, `model_1999` (2026-11-09)

**Training check** (`step_updown_check.py`, 2048 envs, `down_flight`; drop = cm per episode
beyond 0.3 m/s):

| | drop, down_flight | drop, single_drop | ≥1 riser down | fell | edge cost | clean_step_down |
| --- | --- | --- | --- | --- | --- | --- |
| 13b (it 0) | 4.52 | 1.78 | 95 % | 4.1 % | −5.3 | 3.9 |
| 14b | 3.18 (−30 %) | 1.23 | 92 % | 4.7 % | −5.3 | 3.8 |
| 13n (it 0) | 6.08 | 2.25 | 96 % | 3.5 % | −3.2 | 4.9 |
| 14n | 2.11 (−65 %) | 0.78 | 93 % | 4.9 % | −3.1 | 5.0 |

**The step-off** (`step_down_contacts.py`, 20 risers, tread 20 cm, 0.25 m/s, 200 robots per
height). Leading step-off: trunk v_z before touchdown p50 / foot force p50 / trunk |a_z| p50.
Free fall from 2 / 3 / 4 cm ends at 0.63 / 0.77 / 0.89 m/s.

| riser | 13b | 14b | 13n | 14n |
| --- | --- | --- | --- | --- |
| 2 cm | −0.68 / 31 N / 30 | −0.47 / 26 N / 23 | −0.79 / 37 N / 36 | −0.47 / 28 N / 24 |
| 3 cm | −0.81 / 34 N / 35 | −0.64 / 32 N / 30 | −0.93 / 42 N / 40 | −0.60 / 35 N / 30 |
| 4 cm | −0.89 / 37 N / 38 | −0.76 / 37 N / 35 | −1.02 / 42 N / 43 | −0.68 / 39 N / 34 |

- **The trunk no longer free-falls.**
  - 14n comes down at 0.47 / 0.60 / 0.68 m/s: 33–40 % slower than 13n and 22–26 % under free
    fall.
  - Long step-offs (≥ 180 mm) are now almost as soft as short ones: −0.49 / −0.61 / −0.70 against
    −0.41 / −0.48 / −0.58 (13n: −0.83 / −0.96 / −1.05).
  - The trailing foot (0.18–0.21 m/s) and flat treads (0.12–0.13) are softer too.
- **The foot still does not approach the edge.**
  - Lift-off is 65–72 mm behind it at p50 in all four policies, as in §20.
  - 14n shortened the leading step-off (stride p50 170 / 180 / 188 mm against 183 / 193 / 198)
    and lands nearer the edge (34 / 45 / 54 mm beyond it, against 53 / 63 / 71).
  - The leading foot landing on the nose is still the main edge contact (62–66 % of events).
- **Contacts, down** (edge contacts per riser 2 / 3 / 4 cm; reached the bottom of 200):
  - 13n 0.31 / 0.25 / 0.36, 199 / 187 / 113.
  - 14n 0.30 / 0.27 / 0.25, 198 / 173 / 130.
  - 13b 0.60 / 0.70 / 0.87, 198 / 180 / 106.
  - 14b 0.99 / 0.76 / 0.71, 158 / 135 / 75; it falls 11 / 48 / 102 against 13b's 0 / 19 / 89.
    **14b descends worse than the 13b it started from.**

**14n on the 20-riser stairs** (1000 robots down, 1200 up; tread 20 cm; per riser height
2 / 3 / 4 / 5 / 6 cm, 7 cm added on the way up):
- **Down, edge contacts per riser:** 0.29 / 0.20 / 0.24 / 0.37 / 0.60 (13n 0.32 / 0.25 / 0.34 /
  0.53 / 0.70), 9–30 % fewer.
- **Down, ≥ 10 clean risers in a row:** 11 / 28 / 28 / 14 / 2 % (13n 12 / 13 / 11 / 3 / 2 %), more
  than double from 3 to 5 cm.
- **Down, falls** (of 200): 1 / 13 / 59 / 132 / 189 (13n 0 / 11 / 55 / 138 / 186), unchanged.
  - At 3 cm, 58 robots walked off the side of the tile (13n 15), which is why fewer reached the
    bottom (129 against 174).
- **Up, all 20 risers clean:** 29 / 38 / 43 / 23 / 5 / 0 % (13n 31 / 38 / 33 / 14 / 4 / 1 %).
  - No fall: 99 / 99 / 95 / 73 / 21 / 1 % (13n 99 / 97 / 90 / 63 / 18 / 0 %).
  - It climbs a little better than 13n, but still well below 13b (all 20 clean 38 / 72 / 66 /
    51 / 20 / 0 %).

**14b drifts sideways on the way down.** On the 20-riser stairs going down, 151 / 126 / 83 of 200
robots leave the 2 m wide staircase sideways at 2 / 3 / 4 cm (13b 18 / 26 / 23, 14n 20 / 58 / 26).
They do it after 8–12 risers (p50). Only 46 / 47 / 42 reach the bottom (13b 181 / 162 / 101).
The 15 cm treads show the same: 97 / 71 / 45 leave sideways.

**Narrow treads** (15 cm, 20 risers, 1000 robots down / 1200 up):
- **14n down:**
  - Edge contacts per riser 0.48 / 0.44 / 0.50 / 0.66 / 0.69 at 2–6 cm (13n 0.81 / 0.77 / 0.68 /
    0.72 / 0.81), 8–43 % fewer.
  - It falls more at 3–4 cm: 30 / 111 of 200 (13n 21 / 79).
  - It reaches the bottom 191 / 158 / 81 / 18 / 2 times (13n 194 / 177 / 117 / 32 / 1).
- **14n up, all 20 clean:** 5 / 20 / 18 / 4 / 0 % (13n 10 / 9 / 4 / 1 / 0, 13b 30 / 39 / 35 / 20 /
  2).
- **14b up, all 20 clean:** 32 / 33 / 27 / 11 / 2 %, a little below 13b. On 20 cm treads it is
  33 / 55 / 53 / 41 / 13 % against 13b's 38 / 72 / 66 / 51 / 20 %. The anchor did not keep 13b's
  climb.

**Round trip** (`step_up_roundtrip.py --variants orch --map-reset-at-turn`, 100 robots per height:
up with the policy, VelStand turns, down with the policy at 0.2 m/s):

| riser | 13n top / bottom / fell going down | 14n | 13b | 14b |
| --- | --- | --- | --- | --- |
| 2 cm | 98 / 98 / 0 | 100 / 95 / 1 | 98 / 97 / 0 | 96 / 82 / 0 |
| 3 cm | 97 / 87 / 9 | 98 / 81 / 10 | 99 / 90 / 8 | 99 / 77 / 16 |
| 4 cm | 87 / 51 / 28 | 94 / 53 / 39 | 88 / 55 / 32 | 92 / 34 / 53 |
| 5 cm | 48 / 10 / 35 | 62 / 6 / 54 | 78 / 13 / 60 | 76 / 7 / 64 |

- 14n reaches the top more often than 13n (94 against 87 at 4 cm, 62 against 48 at 5 cm).
- 14n brings as many robots to the bottom at 2–4 cm.
- Of those that started down at 4–5 cm, more of 14n's fall.

### What run 14 shows

- **Pricing the drop works.** In 14n the trunk no longer free-falls off a step (v_z 33–40 % lower,
  22–26 % under free fall) and the leading foot lands softer. There are fewer edge contacts going
  down (−9 to −30 % on 20 cm treads, −8 to −43 % on 15 cm), and runs of ≥ 10 clean risers
  doubled. Speed and the climb did not suffer; the climb improved a little.
- **But the step still does not start at the edge.** Lift-off stays 65–72 mm behind it in every
  policy. Nothing the drop price changes pulls the stance foot forward. The leading foot still
  lands on the nose (62–66 % of edge events), and falls going down did not go down. On narrow
  treads 14n falls more at 3–4 cm.
- **14b (anchored to 13b) is a dead end.** It descends worse than 13b, drifts off the side of the
  stairs, and lost part of 13b's climb.
- **A suspect for why the foot stays back:** the edge cost charges any loaded foot whose
  footprint spans two levels. That includes a foot standing flat on the upper tread with its toe
  over the nose, which is how a person or a duck stands before stepping down. Such straddles are
  only 2–6 % of 14n's edge events: the policy keeps its stance foot away from the edge, and the
  doubled downhill edge cost (§20) pays it to. Not measured yet: how much of the downhill edge cost
  is this case, and whether pricing only landings across the edge and tipped feet lets the stance
  foot come forward.

## 22. Static test: can the joints step down without a drop? (2026-11-09, `scripts/eval/step_down_static.py`)

**Question:** is a "normal" step down, with no free fall, physically possible for this robot?

**Method:** the same moment in both directions. The rear leg carries all the weight, CoM over its
foot, and the front foot has just been placed flat on the next tread.
- Down: the rear foot is on the upper tread, its toe 3 mm behind the edge.
- Up: the rear foot is on the lower tread, its toe 3 mm before the riser.
- The front foot lies wholly past the edge (foot-centre distance 60.4–120 mm).

The scan covers trunk height 4–13 cm and trunk pitch −30..+30°. A least-squares solve gives each
pose. Static single-support torques come from the floating-base equation. A pose is rejected if
any mesh (convex hull) sinks more than 1 mm into the step. The test is sagittal: the duck cannot
balance on one flat foot laterally, having no ankle roll.

| riser | 2 cm | 3 cm | 4 cm | 5 cm | 6 cm | 7 cm |
| --- | --- | --- | --- | --- | --- | --- |
| down: highest trunk that works (cm, stand 12) | 10.0 | 9.0 | 8.0 | 7.0 | none | none |
| down: poses dropped, front thigh/knee in the step | 16 | 13 | 11 | 15 | 1 | 0 |
| up: highest trunk that works (cm) | 12.8 | 12.8 | 12.8 | 12.5 | none | none |
| up: poses dropped, front shin in the step nose | 0 | 1 | 14 | 32 | 60 | 73 |
| worst rear-leg pitch torque of any pose (N·m) | 0.29 | 0.29 | 0.26 | 0.17 | – | – |

**What it shows:**
- **Torque is not the limit.** Every pose that exists needs at most 0.29 N·m, under half the
  firmware current limit (0.64) and under a third of the sim's forcerange (0.96).
- **Reach is the limit, and it is much tighter going down.** The rear leg must crouch and the trunk
  lean back 20–25°:
  - trunk at 10 / 9 / 8 / 7 cm for 2 / 3 / 4 / 5 cm risers;
  - at 6–7 cm the front leg cannot get flat onto the lower tread at all.

  Going up, the rear leg stays tall to 5 cm.
- **The duck's knee points backward, toward the riser.** Going down, 11–16 poses per height put the
  front thigh into the step nose.
- **So a drop-free step down exists only up to 5 cm, and only from a deep crouch.** Above that the
  last centimetres must be a controlled drop. Even at 3–4 cm the policy would have to lower the trunk
  by 3–4 cm before every step.
  - Run 14n instead lifts off 65–72 mm behind the edge and drops 0.47–0.68 m/s (§21).
- **The up result is not the policy's climb.** The step-up policy climbs 7.7 cm in sim with momentum
  and the CoM moving forward, not from this static pose. The 6–7 cm "shin in the nose" rejections
  only say that a static, CoM-over-the-rear-foot placement is blocked.
- **Not modelled:** dynamics, backlash, self-collision.
- **Training physics:** runs 1–14 train on `walk_backlash` (`FULL_COLLISION`), where only the
  feet collide with the terrain: shins, thighs and trunk pass through the steps. A descending knee
  or a crouched shin in the step nose costs nothing in training. Run 15 adds shin and thigh
  collisions (§23).

Files: `logs/staircase/step_static.{json,log}` on the gpu box; video
`logs/staircase/video/static/step_static.mp4` (per height down | up, then a results card).

## 23. Run 15: let the duck crouch at a down edge (2026-11-09, `Mjlab-StepUp-TofMap-UpDownEdge-*`)

**Question:** §22 says a drop-free step down exists up to 5 cm, but only from a deep crouch, leaning
back 20–25°. Runs 12–14 never crouch: they walk to the nose and drop. Does the policy step down
softly if the walking terms stop charging for the crouch, and if it starts episodes inside one?

**Changes vs run 14 (`edge_down=True`, warm start from 14n `model_1999`, 2000 iterations, plain PPO):**
- **Down-edge zone:** on the downhill tiles only (`single_drop`, `down_flight`), from 8 cm behind to
  14 cm past an edge that drops ≥ 1 cm, read from the exact height field along the heading
  (`mdp.down_edge_zone`). Inside it:
  - `upright` does not charge a backward lean up to 30°; forward lean and lean beyond 30° cost as before;
  - `pose` leaves out the hip-pitch / knee / ankle chain (the crouch);
  - both velocity terms track min(command, 0.08 m/s), so crouching slowly pays fully and standing
    still at the edge still costs (track_vx_ema 0.75 of full);
  - `trunk_drop` caps the trunk's fall at 0.15 m/s instead of 0.3.
- **Edge cost without grid slop:** `foot_edge_contact` and `clean_step_down` test a sole shrunk by 8 mm
  on its long sides on the downhill tiles. The 1 cm height grid placed an edge up to 5 mm off, so a flat
  foot 3 mm behind the nose was charged as an edge contact; a 15 mm overhang is still charged.
- **Leg collisions:** runs 1–14 trained with only the feet colliding with the terrain. Run 15 adds the
  shins (their exported collision mesh) and the thighs (convex hull of the visual mesh, ≤ 48 vertices,
  massless) to both robot models (`MICRODUCK_WALK*_LEGCOL_ROBOT_CFG`).
- **Reverse-curriculum spawns** (`mdp.reset_at_down_edge`, 30 % of downhill resets): rear toe 3–8 mm
  behind a random edge of the tile, in a §22 pose, either leg as the rear one, yaw ±0.05 rad, lateral
  ±5 cm, the ToF map prefilled from the exact ground (±0.6 × ±0.3 m) as if the duck had walked there.
  - The bank: 390 poses from the §22 scan at 1.5–4.5 cm (99 at 1.5 cm … 16 at 4.5 cm), no mesh in the
    step, trunk 6.5–10.5 cm, pitch −30..0° (`scripts/make_step_down_bank.py`,
    `tasks/data/step_down_poses.json`).
  - The pose's riser is ≤ the edge's and at most 6 mm lower, so the front foot starts on the lower tread
    or ≤ 6 mm above it.

**Spawn check before training** (`scripts/eval/edge_spawn_check.py`, 409 spawns on the downhill tiles):

| | rear sole gap | front sole gap | CoM − rear foot centre | fell within 2 s |
| --- | --- | --- | --- | --- |
| first version (any bank riser ≤ the edge) | 1.4 mm | 8.2 mm (p95 23) | 0 ± 5 mm | 14n 60 %, hold the pose 95 % |
| ≤ 6 mm below the edge (used) | 1.4 mm | 4.1 mm (p95 6.7) | 0 ± 5 mm | 14n 52 %, hold the pose 99 % |

- The gaps include the spawn's 2 mm lift: the first step is 20 ms of free fall (trunk −0.19 m/s), no
  pop-out from interpenetration; 36–47 of 409 spawns have a shin or thigh touching the terrain.
- **The §22 pose is not self-stable on the simulated servos.** Holding its joint targets, the hip pitch
  sags 3.6° (p95 7°) in 0.2 s, plus the 2° backlash, the trunk tips further back (−20° → −26°) and
  falls. A policy has to balance actively in the crouch.

**14n on the leg-collision model** (baseline; the same batteries as §21, now on the Edge task):
- Down staircase, tread 20 cm, reached the bottom: 177 / 138 / 123 / 55 / 2 of 200 at 2–6 cm
  (feet-only model, §21: 179 / 129 / 115 / 55 / 3). Up, head down: 181 / 194 / 185 / 143 / 50 / 0
  (§21: 185 / 195 / 188 / 136 / 38 / 1). Leg collisions change nothing measurable for 14n.
- Step-off leading foot: trunk v_z before touchdown −0.47 / −0.60 / −0.69 m/s at 2 / 3 / 4 cm.

**Results, run 15 `model_1999` vs 14n** (both on the leg-collision model; 14n here = `model_1999` of §21):

| | 14n | 15 |
| --- | --- | --- |
| step-off leading foot, trunk v_z before touchdown, 2 / 3 / 4 cm (m/s) | −0.47 / −0.61 / −0.69 | −0.34 / −0.47 / −0.54 |
| same, peak trunk \|a_z\| p50 (m/s²) | 24 / 30 / 34 | 17 / 24 / 27 |
| leading foot lifts off behind the edge, p50 (mm) | 71 / 71 / 71 | 53 / 59 / 59 |
| step-off stride p50 (mm) | 169 / 181 / 188 | 146 / 161 / 168 |
| **trunk above the upper tread at that lift-off, p50 (mm; flat tread 128)** | **124 / 124 / 122** | **124 / 124 / 124** |
| training env, no spawns (`UpDownSoft`): falling faster than 0.3 m/s, cm per episode, single_drop / down_flight | 0.78 / 2.11 | 0.46 / 1.14 |
| same, `foot_edge_contact` per episode on down_flight (unshrunk) | −3.1 | −4.3 |
| down staircase (tread 20 cm), reached the bottom of 200, 2 / 3 / 4 / 5 / 6 cm | 177 / 138 / 123 / 55 / 2 | 167 / 181 / 149 / 61 / 5 |
| same, edge contacts per riser | 0.29 / 0.24 / 0.22 / 0.41 / 0.58 | 0.52 / 0.29 / 0.32 / 0.49 / 0.71 |
| same, edge contacts reaching ≤ 5 mm across | 20 / 27 / 24 / 16 / 12 % | 14 / 37 / 58 / 40 / 28 % |
| up staircase (head down), reached the top of 200, 2 … 7 cm | 181 / 194 / 185 / 143 / 50 / 0 | 185 / 192 / 187 / 154 / 27 / 0 |
| same, clean risers p50 | 13 / 16 / 16 / 11 / 5 / 2 | 19 / >20 / >20 / 19 / 5 / 2 |
| round trip (VelStand turns), reached the bottom of 100, 2 / 3 / 4 / 5 cm | 98 / 86 / 59 / 8 | 99 / 93 / 57 / 26 |
| from a §22 spawn: fell within 2 s | 52 % | 3 % |

**What it shows:**
- **The duck does not crouch.** Its trunk is 124 mm above the upper tread when the leading foot leaves,
  the same as 14n and 4 mm under a flat-tread step. It lifts off 12–18 mm nearer the edge, with
  shorter steps, and lands 22–27 % slower, but it still drops: −0.34 to −0.54 m/s against ≤ 0.15 m/s
  for a drop-free step.
- **The spawns taught recovery, not the approach.** Started inside the crouch, run 15 falls 3 % (14n 52 %),
  but only 27 % of those robots are 1 cm lower after 2 s; the rest stand or step back up.
- **Side effects:**
  - The climb got cleaner (clean risers p50 up 6–8 at 2–5 cm), but 6 cm climbs fell from 50 to 27.
  - The descent reaches the bottom more often at 3–5 cm.
  - Edge contacts per riser rose at 2 cm. Many new ones overhang ≤ 5 mm, which the 8 mm shrink lets
    through in training.
- **Why no crouch, likely:** reward mass.
  - On the down tiles `trunk_drop` costs 1.6–4.4 per episode, against about 100 for the positive terms, so
    dropping stays cheaper than learning an unstable crouch.
  - Holding the §22 pose needs active balance: the servos sag 3.6° at the hip.

Files (gpu box): `logs/edge15/` (train, checks at 250–1999, spawn checks), `logs/staircase/*15*`,
`*14n_legcol*`; checkpoint `ckpts/stepup_15_model_1999.pt`
(run dir `logs/rsl_rl/step_up/2026-11-09_22-52-19_step_up_tofmap_updown_edge_15`).

## 24. Run 16: price the drop at the edge harder, or pay a soft step (2026-11-10, `Mjlab-StepUp-TofMap-UpDownEdge{Drop,DropHi,Gentle}-*`)

**Question:** §23 says the drop at an edge is cheap: `trunk_drop` costs 1.6–4.4 per downhill episode,
against about 100 of positive terms. Does the duck crouch if the drop costs more?

**Variants** (all warm-started from run 15 `model_1999`, plain PPO, 2000 iterations, on run 15's env):
- **16a / 16b, `edge_drop`:** `trunk_drop_edge` charges the same fall faster than 0.15 m/s, inside the
  down-edge zone only. Its weight is ramped by iteration (¼ → ½ → ¾ → full at 0 / 250 / 750 / 1250),
  so the zone ends at ×5 (16a, +600) or ×10 (16b, +1350) of `trunk_drop`'s 150.
- **16c, `soft_step`:** no extra penalty. `soft_step_down` pays the leading foot's clean step onto a new
  lower level (cm, weight 250), × exp(−(peak excess / 0.3 m/s)²). The excess is the trunk's fall beyond
  0.15 m/s, and its peak is a running max that decays 0.85 per step. Not descending earns nothing; a hard
  step earns a little.

**What happened in training** (training env with spawns, `down_flight`, all risers down / fell):

| iteration | 15 (final) | 16a | 16b | 16c |
| --- | --- | --- | --- | --- |
| 250 | 88 % / 5 % | 85 % / 4 % | 61 % / 8 % | 90 % / 17 % |
| 500 | | 69 % / 7 % | 38 % / 14 % (stopped) | 91 % / 12 % |
| 1000 | | 45 % / 11 % | | 90 % / 14 % |
| 1999 | | 35 % / 11 % | | 90 % / 11 % |

**Final checkpoints** (same batteries as §23, leg-collision model):

| | 15 | 16a | 16c |
| --- | --- | --- | --- |
| leading step-off: trunk v_z before touchdown, 2 / 3 / 4 cm (m/s) | −0.34 / −0.47 / −0.54 | (24–41 step-offs) | **−0.13 / −0.19 / −0.30** |
| same, peak trunk \|a_z\| p50 (m/s²) | 17 / 24 / 27 | | 14 / 17 / 21 |
| trunk at the leading foot's lift-off, p50 (mm; flat 127–128) | 124 / 124 / 124 | 124 / 123 / 123 | 125 / 124 / 123 |
| leading foot lifts off behind the edge, p50 (mm) | 53 / 59 / 59 | | 100 / 100 / 98 |
| leading stride across the edge, p50 (mm) | 146 / 161 / 168 | | 189 / 193 / 198 |
| down staircase (20 risers, tread 20 cm), bottom of 200, 2 / 3 / 4 / 5 / 6 cm | 167 / 181 / 149 / 61 / 5 | 0 / 0 / 0 / 0 / 0 (185–200 walk off sideways) | 137 / 79 / 14 / 0 / 0 |
| same, edge contacts per riser, 2 / 3 / 4 cm | 0.52 / 0.29 / 0.32 | | 0.43 / 0.56 / 0.67 |
| up staircase (head down), top of 200, 2 … 6 cm | 185 / 192 / 187 / 154 / 27 | 194 / 190 / 157 / 78 / 16 | 187 / 195 / 177 / 127 / 11 |
| same, clean risers p50, 2 / 3 / 4 / 5 cm | 19 / >20 / >20 / 19 | 18 / >20 / 13 / 6 | 9 / 9 / 9 / 7 |
| round trip, bottom of 100, 2 / 3 / 4 / 5 cm | 99 / 93 / 57 / 26 | 0 / 0 / 0 / 0 | 82 / 44 / 4 / 0 |
| from a §22 spawn: fell within 2 s / 1 cm lower after 2 s | 3 % / 27 % | 7 % / 5 % | 4 % / 25 % |

**What it shows:**
- **A bigger drop price buys refusal, not a crouch.**
  - 16a and 16b stopped descending as their weight rose. At the end 16a walks off the side of every
    staircase rather than step down: 0 of 1000 reach the bottom.
  - A policy that cannot yet crouch can lower the cost only by not descending (AGENTS.md: an attempt tax
    before the skill exists makes "do nothing" win).
- **Paying the soft step made it soft, still without a crouch.**
  - 16c's leading foot lands at a third of 15's trunk speed at 2–3 cm, about half at 4 cm.
  - The trunk is no lower at lift-off. Instead, 16c leaves 100 mm before the edge with a 190 mm stride:
    the front leg reaches down to the lower tread while the rear leg is still on the upper one.
  - That gait fits one riser, not a staircase. It falls on 20 cm treads from 3 cm up (bottom 79 / 14 /
    0 of 200 at 3 / 4 / 5 cm), touches more edges, and climbs worse: clean risers p50 9 against 19.
- **No variant crouches.** The trunk stays 123–125 mm over the upper tread at lift-off in 14n, 15, 16a
  and 16c alike. Run 15 remains the best descent policy (`ckpts/stepup_15_model_1999.pt`).

Process note: the 16a eval chain stopped after its 1500 check. Its script was overwritten while bash was
still reading it (bash reads a running script lazily), so the final batteries were rerun by hand. Write
a new file for a changed script on the gpu box; do not edit one that is running.

Files (gpu box): `logs/drop16/` (train, chain and eval scripts, checks), `logs/staircase/*16a*`, `*16c*`;
checkpoints `ckpts/stepup_16a_model_1999.pt`, `ckpts/stepup_16c_model_1999.pt` (run dirs
`logs/rsl_rl/step_up/2026-11-10_08-20-42_..._edgedrop_16a`, `2026-11-10_08-47-18_..._edgegentle_16c`).

## 25. Run 17: pay the crouch (2026-11-10, `Mjlab-StepUp-TofMap-UpDownEdgeCrouch{,Soft}-*`)

**Question:** in runs 14n, 15, 16a and 16c the trunk stood 123–125 mm over the upper tread when the
leading foot lifted off (§24). Runs 15–16 freed the crouch, priced the drop and paid a soft step, but
nothing paid the crouch itself. Does the duck crouch if the step down pays by how low the trunk was?

**Changes vs run 15 (`crouch_step`, warm start from run 15 `model_1999`, plain PPO, 2000 iterations):**
- `crouch_step_down` (`mdp.crouch_step_down_bonus`) pays the leading foot's clean step onto a new lower
  level (cm), as 16c did, × a crouch factor exp(−(max(0, h − h*) / 3 cm)²):
  - h: the trunk over that foot's support at its last contact before the step (its lift-off), the
    quantity `step_down_contacts.py` reports;
  - h* = 12 cm − the drop the foot came down: §22's highest working trunk (10 / 9 / 8 / 7 cm at
    2 / 3 / 4 / 5 cm);
  - a lower trunk than h* earns no more. Paid once per step, at the touchdown, so standing crouched at
    the edge earns nothing.
- **17a:** that alone, weight 150. **17b:** × 16c's softness factor too (the drop-free step), weight 200.
- A §22 spawn's front foot has no lift-off this episode and earns nothing. The spawn's completion is
  paid as in run 15 (`clean_step_down`, `descent_progress`).

**Sizing at iteration 0** (`step_updown_check.py`, run 15 `model_1999`, 2048 envs, 8 s; weighted per
episode, single_drop / down_flight):

| | crouch_step_down | clean_step_down | trunk_drop | all risers down |
| --- | --- | --- | --- | --- |
| 17a task (weight 150 for this check) | 1.06 / 4.26 | 1.63 / 5.29 | −1.65 / −4.78 | 91 / 91 % |
| 17b task (weight 300 for this check) | 1.61 / 4.53 | 1.69 / 4.90 | −1.71 / −4.51 | 88 / 89 % |

- On down_flight, run 15 earns 1.42 cm·crouch per episode (17a) and 0.76 cm·crouch·softness (17b), out of
  about 4.8 cm of leading-foot steps: mean crouch factor about 0.3, crouch × softness about 0.16.
- Final weights 150 (17a) and 200 (17b): 4.3 and 3.0 per down_flight episode now, 14 and 19 at a full
  crouch, against about 95 of positive terms (16c's 250 paid 24 fully soft, and its climb lost).

**What happened in training** (training env with spawns, `down_flight`, all risers down / fell; the
leading step-off on the 20-riser down battery, 300 envs at 500 and 1000, 600 at the end):

| iteration | 17a: down / fell | 17a: trunk at lift-off p50 / p10, 2 / 3 / 4 cm (mm) | 17b: down / fell | 17b: trunk at lift-off p50 / p10 (mm) | 17b: step-off v_z (m/s) |
| --- | --- | --- | --- | --- | --- |
| 250 | 91 % / 6 % | | 91 % / 4 % | | |
| 500 | 91 % / 4 % | 122 / 122 / 121 · 118 / 116 / 113 | 89 % / 5 % | 122 / 122 / 122 · 118 / 116 / 114 | −0.32 / −0.40 / −0.50 |
| 1000 | 94 % / 2 % | 122 / 121 / 120 · 118 / 115 / 113 | 92 % / 4 % | 122 / 121 / 120 · 116 / 113 / 110 | −0.23 / −0.31 / −0.41 |
| 1999 | 89 % / 5 % | 122 / 121 / 121 · 118 / 116 / 114 | 91 % / 6 % | 123 / 122 / 122 · 118 / 117 / 114 | −0.32 / −0.38 / −0.47 |

**Final checkpoints, `model_1999`** (same batteries as §23-24, leg-collision model):

| | 15 | 17a | 17b |
| --- | --- | --- | --- |
| trunk at the leading foot's lift-off, p50 (mm), 2 / 3 / 4 cm (target ≤ 110 / 100 / 90) | 124 / 124 / 124 | 122 / 121 / 121 | 123 / 122 / 122 |
| same, p10 (mm) | 122 / 121 / 119 | 118 / 116 / 114 | 118 / 117 / 114 |
| leading step-off: trunk v_z before touchdown (m/s) | −0.34 / −0.47 / −0.54 | −0.40 / −0.47 / −0.55 | −0.32 / −0.38 / −0.47 |
| same, peak trunk \|a_z\| p50 (m/s²) | 17 / 24 / 27 | 24 / 27 / 29 | 21 / 24 / 26 |
| leading foot lifts off behind the edge, p50 (mm) | 53 / 59 / 59 | 65 / 64 / 66 | 66 / 64 / 63 |
| leading stride across the edge, p50 (mm) | 146 / 161 / 168 | 165 / 172 / 176 | 165 / 167 / 168 |
| down staircase (20 risers, tread 20 cm), bottom of 200, 2 / 3 / 4 / 5 / 6 cm | 167 / 181 / 149 / 61 / 5 | 157 / 147 / 98 / 26 / 5 | 161 / 161 / 89 / 12 / 0 |
| same, edge contacts per riser, 2 / 3 / 4 cm | 0.52 / 0.29 / 0.32 | 0.28 / 0.29 / 0.39 | 0.33 / 0.30 / 0.49 |
| up staircase (head down), top of 200, 2 … 6 cm | 185 / 192 / 187 / 154 / 27 | 189 / 191 / 174 / 93 / 10 | 180 / 186 / 174 / 112 / 25 |
| same, clean risers p50, 2 / 3 / 4 / 5 cm | 19 / >20 / >20 / 19 | >20 / >20 / 19 / 9 | 18 / 20 / 11 / 8 |
| round trip, bottom of 100, 2 / 3 / 4 / 5 cm | 99 / 93 / 57 / 26 | 97 / 84 / 49 / 5 | 93 / 84 / 39 / 4 |
| from a §22 spawn: fell within 2 s / 1 cm lower after 2 s | 3 % / 27 % | 2 % / 27 % | 4 % / 23 % |

**What it shows:**
- **Paying the crouch did not make one.** The trunk at the leading lift-off went down 2-3 mm (p50) and
  4-5 mm (p10) in 2000 iterations; the target needs 24-44 mm. The pay for a crouch is visible (crouch
  factor 0.3 → 1 is worth +10 per down_flight episode in 17a), and the policy moved toward it, about
  1 mm per 250 iterations, then stopped.
- **The steps got longer instead.** Both variants lift off 63-66 mm before the edge (15: 53-59) and
  stride 165-176 mm across it (15: 146-168), the direction 16c went to its extreme.
- **17b is softer than 15, 17a is not.** 17b lands 0.02-0.09 m/s slower; at iteration 1000 it was
  softer still (−0.23 / −0.31 / −0.41, 300 envs) and lost it by 1999.
- **Both are worse on the staircases.** Down: 98 / 89 of 200 at 4 cm against 149, 26 / 12 at 5 cm against
  61. The climb lost at 5-6 cm (93 / 112 of 200 against 154). Run 15 remains the best descent policy.
- **Likely why:** the crouch is a narrow, unstable region (§23: the §22 pose sags 3.6° at the hip and
  must be balanced actively), and a bonus paid at the touchdown gives credit for one quantity at one
  instant. Shorter-horizon alternatives (a longer reach, a slower fall) earn part of the same pay with
  the gait the policy already has.

Files (gpu box): `logs/crouch17/` (size, smoke, train, chain, checks at 250-1999, impact at 500 and
1000, spawn checks), `logs/staircase/*17a*`, `*17b*`; checkpoints `ckpts/stepup_17a_model_1999.pt`,
`ckpts/stepup_17b_model_1999.pt` (run dirs `logs/rsl_rl/step_up/2026-11-10_15-28-16_..._edgecrouch_17{a,b}`).

## 26. Run 18a: a step-down teacher (2026-11-10, `Mjlab-StepDown-Teacher-*`)

**Question:** runs 15–17 never crouched at a down edge (trunk 121–124 mm at the leading lift-off, target
≤ 110 / 100 / 90). The tasks here that did learn a crouch (ground_pick, sitstand, roller crouch) make
the trunk height their MAIN term, and VelStand's commanded −2 cm was never learned ("the axis has no
teacher"). Does a specialist trained that way, on one step down, crouch? If it does, 18b distils it
into the step-up policy (climb_anchor on the frames whose map shows a drop ahead); if not, no 18b.

**Task** (`tasks/microduck_step_down_teacher_env_cfg.py`): run 15's env (same 327-D obs and network,
warm start from run 15 `model_1999`, `MICRODUCK_WARM_START=1`, no climb anchor), 1000 iterations, with:
- **terrain:** only `single_drop`, riser 1.5–4.5 cm by row (every row from the start, no promotion);
  episodes 4 s;
- **spawns** (`mdp.reset_at_down_edge`, three chained banks, a later one overrides):
  - every robot standing on the upper tread, toe 2–12 cm before the edge (`edge_stand_poses.json`,
    trunk 12.0–12.5 cm);
  - 20 % of them crouched in double support at the edge, toe 3–30 mm back, at the crouch their riser
    needs (`edge_crouch_poses.json`: 94 poses, trunk 6.5–11 cm, pitch −20…0°, built with §22's solver
    by `scripts/make_edge_crouch_bank.py`; the pose's h = 12 cm − trunk is 0–1 cm below the riser);
  - 15 % in run 15's §22 mid-step pose;
- **`trunk_level`** (main term, weight 10): exp(−(err / 2.5 cm)²), err = trunk world z − a SLEWED
  target that starts at the trunk's height at the episode's first step and moves at 5 cm/s toward
  the lower level + 12 cm, i.e. 12 cm − riser over the upper tread (§22's highest working crouch).
  The trunk goes down before the edge and crosses it level. Ahead of the slew scores like behind
  (no jackpot). **`trunk_level_l1`** (weight 100): −|err| (m);
- **`drop_free_step_down`** (weight 750, × dt = 15 once): both feet clean on the spawn edge's lower
  level and upright, paid only if the trunk never fell faster than 0.25 m/s after 0.2 s (run 15's
  step-offs: 0.34–0.54 m/s). A drop forfeits it: a hard gate, not a nudge;
- forward commands only (0.1–0.25 m/s, no heading targets).

The proposal's "slewed feet goal" was not built: the completion is the one-shot gate plus run 15's
`clean_step_down` and `descent_progress`.

**Pre-check: do the spawns hold on the servos?** (`edge_spawn_check.py --event … --hold --seconds 3`,
1024 envs, open loop: each robot holds its spawn joint targets)

| | crouch, backlash | crouch, no backlash | stand, backlash | stand, no backlash | crouch, run 15 policy |
| --- | --- | --- | --- | --- | --- |
| fell within 3 s | 73 % | 63 % | 95 % | 89 % | 3 % |
| servo sag at 0.2 s, knee p50 / p95 | 3.2 / 5.5° | 3.9 / 6.0° | 3.0 / 4.3° | 3.6 / 4.7° | |
| trunk pitch after 0.2 s, p5 / p50 / p95 | −20 / −4 / +10° | −19 / −6 / +9° | +1 / +3 / +7° | 0 / +3 / +6° | −5 / +1 / +6° |
| alive: trunk end − spawn, p50 | −9 mm | −8 mm | −6 mm | −5 mm | 0 mm |

- Open loop, even the walking stand falls (89–95 % in 3 s): with no feedback the 54 mm feet do not
  hold any of these poses, so an open-loop hold says nothing about the crouch alone. The crouch falls
  LESS often than the stand.
- Closed loop, run 15 (never trained crouched) keeps 97 % of the crouch spawns up; it stands up out of them.
- The spawns sit 1.3 mm over the tread, no leg geom touches it, CoM within ±5 mm of the feet.

**Iteration 0** (`scripts/eval/step_down_teacher_eval.py`, run 15 `model_1999`, 1024 envs, first
4 s episode each; per spawn type and riser bin 1.5–2.5 / 2.5–3.5 / 3.5–4.5 cm):

| | stand | crouch | mid-step |
| --- | --- | --- | --- |
| drop-free success | 8 / 0 / 0.5 % | 13 / 4 / 0.5 % | 33 / 14 / 3 % |
| both feet down (with or without a drop) | 100 / 98 / 95 % | 98 / 99 / 93 % | 98 / 97 / 94 % |
| fell | 0 / 2 / 5 % | 2 / 2 / 7 % | 2 / 4 / 6 % |
| trunk at the leading lift-off p50 (mm; target ≤ 100 / 90 / 80) | 122 / 119 / 119 | 112 / 107 / 102 | 97 / 78 / 86 |
| trunk peak fall speed in that swing p50 (m/s) | 0.37 / 0.48 / 0.52 | 0.35 / 0.45 / 0.54 | 0.19 / 0.19 / 0.19 |

Reward per episode (weighted, all spawns): `trunk_level` 34.8 of 40, `trunk_level_l1` −2.7,
`drop_free_step_down` 0.7, `clean_step_down` 2.5, `descent_progress` 2.8, `trunk_drop` −2.2, all other
penalties ≤ 0 (`step_updown_check.py`). Run 15 already earns most of `trunk_level`: after the step it
stands at the goal height. What separates a crouched, drop-free step from run 15's is about +15
(the gate) +4–5 (`trunk_level` during the approach) +2.5 (L1) +2 (`trunk_drop`) ≈ 24 per episode,
of about 100 positive. Run 17's crouch pay was worth about 10, spread over a whole staircase. Weights
kept as designed.

**Training** (`step_down_teacher_eval.py` at each checkpoint, 1024 envs, 2048 at the end; stand spawn
unless noted; riser bins 1.5–2.5 / 2.5–3.5 / 3.5–4.5 cm):

| iteration | drop-free success | trunk at the leading lift-off p50 (mm) | swing fall speed p50 (m/s) | crouch spawn: lift-off p50 (mm) / success |
| --- | --- | --- | --- | --- |
| 0 (run 15) | 8 / 0 / 0.5 % | 122 / 119 / 119 | 0.37 / 0.48 / 0.52 | 112 / 107 / 102 · 13 / 4 / 0.5 % |
| 250 | 91 / 83 / 59 % | 110 / 109 / 110 | 0.14 / 0.17 / 0.22 | 102 / 97 / 92 · 93 / 91 / 87 % |
| 500 | 99.5 / 97 / 90 % | 105 / 105 / 104 | 0.12 / 0.15 / 0.17 | 96 / 91 / 85 · 98 / 98 / 99 % |
| 750 | 99 / 98 / 92 % | 106 / 106 / 106 | 0.11 / 0.14 / 0.16 | 95 / 89 / 83 · 100 / 100 / 98 % |
| 999 | 99.4 / 99.5 / 96 % | 107 / 106 / 106 | 0.11 / 0.13 / 0.16 | 98 / 93 / 87 · 99.5 / 99 / 98 % |

Final, `model_999` (`ckpts/stepdown_teacher_18a_last.pt`):
- falls ≤ 0.3 % (stand, crouch), 0.3–2 % (mid-step); mid-step success 97 / 94 / 60 %;
- lowest trunk before the leading touchdown (stand) 98 / 93 / 87 mm: it keeps lowering during the swing;
- reward per episode: `trunk_level` 38.6 of 40, `drop_free_step_down` 14.6 of 15, `trunk_drop` −0.2
  (iteration 0: −2.2), every penalty ≤ 0, trunk fall > 0.25 m/s 0.06 cm per episode.

**On run 15's 20-riser down staircase** (`step_down_contacts.py`, UpDownEdge task, 300 envs, 45 s — a
terrain the teacher never saw):

| | run 15 (§25) | 18a |
| --- | --- | --- |
| bottom reached, 2 / 3 / 4 cm | 167 / 181 / 149 of 200 | 57 / 6 / 0 of 100 |
| fell | | 34 / 93 / 100 % |
| trunk at the leading lift-off p50 / p10 (mm) | 124 / 124 / 124 · 122 / 121 / 119 | 116 / 115 / 113 · 109 / 106 / 102 |
| leading step-off: trunk v_z before touchdown (m/s) | −0.34 / −0.47 / −0.54 | −0.19 / −0.28 / −0.35 |

**What it shows:**
- **The teacher learned a drop-free single step down** (96–99.5 % from a stand, fall speed in the
  swing 0.11–0.16 m/s instead of 0.37–0.52) within 500 iterations, where runs 15–17 never did.
- **It crouches, but not by the riser.** From a stand the trunk is 106 mm at lift-off for every riser
  (−15 mm vs run 15; target 100 / 90 / 80) and goes on down during the swing. From the crouch spawns it
  stays at the target (98 / 93 / 87 mm). The single crouch depth is enough for the gate at ≤ 4.5 cm.
- **It does not walk stairs.** Trained only on one riser and 4 s episodes, it falls on 34–100 % of
  the staircases — its step-off there is still lower and softer than run 15's.

## 27. Run 18c: the step-down teacher on flights (2026-11-10, `Mjlab-StepDown-TeacherStairs-*`)

**Question:** 18a steps down one riser drop-free but falls on 34–100 % of the 20-riser staircase it
never saw. Does it learn the stairs if it trains on flights with the same two terms made per riser?

**Changes vs 18a** (`make_microduck_step_down_stairs_teacher_env_cfg`; RESUMES 18a `model_999` —
same obs, curricula continue — for 1000 more iterations, 1000–1999):
- **terrain:** `single_drop` (1 of 4 columns) + `down_flight` (3 of 4): 6 risers, tread 11–24 cm (run
  15's tread DR), riser 1.5–4.5 cm by row, the first edge 0.3–0.8 m after the origin. Episodes 6 s
  (≤ 1.5 m at 0.25 m/s: a robot spawned at the last edge stays on the tile);
- **spawns:** 75 % stand at a random edge of the tile (on a flight, any riser), 25 % on the platform
  (walk to the first edge); the crouch / mid-step overrides as in 18a. The stand gap is clamped so
  the whole sole stays on its tread (`reset_at_down_edge`: gap ≤ tread − foot − 5 mm);
- **`trunk_level`** → `mdp.trunk_ground_tracking`: goal = 12 cm over the tread under the HIGHER foot,
  minus the tile's riser while a down edge is within 14 cm ahead of the trunk. So 12 cm − riser over
  the tread it is about to leave, 12 cm over the lower tread once a foot is down, and down one riser
  at a time on a flight. Slew 10 cm/s (stairs at 0.25 m/s descend ≤ 0.10 m/s);
- **`drop_free_step_down`** → `mdp.drop_free_level_bonus`: 15 per riser, each time both feet stand
  clean on a new lowest level, if the trunk did not fall faster than 0.25 m/s since the previous one
  (a drop forfeits that riser only).

**Pre-check:** the stand spawns on flights sit 1.3 mm over the tread (p5–p95), no leg geom touches.

**Iteration 0** (18a `model_999` on the 18c env, `step_down_teacher_eval.py`, 1024 envs, first 6 s
episode; risers 1.5–2.5 / 2.5–3.5 / 3.5–4.5 cm):

| | flight, stand at an edge | flight, from the platform | single drop, stand | single drop, from the platform |
| --- | --- | --- | --- | --- |
| fell | 10 / 39 / 47 % | 20 / 49 / 64 % | 0 / 0 / 0 % | 0 / 13 / 10 % |
| risers down per episode | 2.1 / 0.5 / 0.3 | 2.5 / 0.7 / 0.5 | 1 / 1 / 1 | 0.8 / 0.7 / 0.7 |
| of them drop-free | 49 / 22 / 31 % | 23 / 11 / 11 % | 100 / 100 / 96 % | 44 / 42 / 31 % |
| trunk at the first leading lift-off p50 (mm) | 105 / 101 / 101 | 112 / 108 / 108 | 107 / 107 / 106 | 113 / 112 / 110 |

- 18a falls on flights from 3 cm on, and even a single drop goes wrong after a walk from the platform
  (18a only ever started 2–12 cm from its edge).
- Reward per episode (`step_updown_check.py`, single_drop / down_flight): `trunk_level` 58.3 / 42.6
  of 60, `drop_free_step_down` 12.5 / 7.7, `trunk_drop` −0.4 / −6.0, `foot_edge_contact` −0.7 / −6.0,
  `fall` −0.1 / −3.4; every penalty ≤ 0. Weights kept from 18a.

**Training** (`step_down_teacher_eval.py`, 1024 envs, 2048 at the end; stand at an edge, risers
1.5–2.5 / 2.5–3.5 / 3.5–4.5 cm; the 20-riser stairs: `step_down_contacts.py`, UpDownEdge task, 100
envs per riser at 1500, 200 at the end):

| iteration | flight: fell | flight: risers drop-free | flight: trunk at the first lift-off p50 (mm) | single drop: drop-free | 20-riser stairs: bottom, 2 / 3 / 4 cm |
| --- | --- | --- | --- | --- | --- |
| 999 (18a) | 10 / 39 / 47 % | 49 / 22 / 31 % | 105 / 101 / 101 | 100 / 100 / 96 % | 57 / 6 / 0 % |
| 1250 | 1 / 11 / 14 % | 73 / 75 / 61 % | 103 / 98 / 97 | 98 / 97 / 88 % | |
| 1500 | 2 / 5 / 12 % | 68 / 68 / 63 % | 103 / 100 / 99 | 98 / 92 / 82 % | 89 / 86 / 62 % |
| 1750 | 5 / 6 / 11 % | 73 / 76 / 74 % | 103 / 99 / 98 | 99 / 96 / 83 % | |
| 1998 | 2 / 4 / 8 % | 70 / 75 / 68 % | 101 / 98 / 96 | 97 / 93 / 86 % | 90 / 79 / 58 % |

A robot on a flight descends only 2.4 / 1.0 / 0.6 risers per 6 s episode (it goes slowly).

**Final, `model_1998`** (`ckpts/stepdown_teacher_18c_last.pt`) against run 15 on the §25 batteries (UpDownEdge
task, leg-collision model, 20 risers, tread 20 cm, 0.25 m/s, 45 s):

| | run 15 | 18c |
| --- | --- | --- |
| down staircase: bottom of 200, 2 / 3 / 4 / 5 / 6 cm | 167 / 181 / 149 / 61 / 5 | 146 / 148 / 110 / 57 / 7 |
| same: fell | 2 / 13 / 44 / 131 / 176 | 7 / 24 / 74 / 124 / 176 |
| same: left the stairs sideways | 31 / 6 / 7 / 8 / 19 | 47 / 28 / 16 / 19 / 17 |
| same: first-unclean hazard per riser (edge contact or fall), risers 1–3 / 4–6 / 7–10, 3 cm (%) | 22 / 18 / 20 | 16 / 33 / 37 |
| same: FALL hazard per riser, risers 1–3 / 4–6 / 7–10 / 11–15 / 16–20, 4 cm (%; `step_up_staircase.py` fall-hazard row, §28) | 0.8 / 2.1 / 0.8 / 1.9 / 0.7 | 0.2 / 1.4 / 1.9 / 3.9 / 3.3 |
| trunk at the leading lift-off p50 (mm), 2 / 3 / 4 cm (target ≤ 100 / 90 / 80) | 124 / 124 / 124 | 105 / 101 / 98 |
| leading step-off: trunk v_z before touchdown (m/s) | −0.34 / −0.47 / −0.54 | −0.09 / −0.14 / −0.19 |
| edge contacts per riser, 2 / 3 / 4 cm | 0.56 / 0.31 / 0.34 | 0.78 / 0.70 / 0.67 |
| up staircase, top of 200 at 2 … 7 cm | 185 / 192 / 187 / 154 / 27 (… 6 cm) | 0 at every height (stalls, walks off) |

**What it shows:**
- **The descent is crouched and soft.** The trunk is 98–105 mm at the leading lift-off (run 15: 124)
  and lands at 0.09–0.19 m/s (run 15: 0.34–0.54), on stairs it never saw at that length. Still not
  riser-scaled: 7 mm between 2 and 4 cm against the 20 mm the target asks.
- **It is less robust on long stairs than run 15.** More falls at 3–4 cm (24 / 74 against 13 / 44 of
  200), more sideways exits, and its hazard of a first unclean riser grows after the third riser
  (16 → 33 → 37 % at 3 cm) where run 15's stays flat. Its true fall hazard per riser grows along
  the flight at 4 cm (0.2 → 3.9 %), where run 15's stays at 1–2 %. (This bullet first read the
  unclean hazard as falls; corrected in §28.) 6 s episodes carry it down 1–2.5 risers; nothing
  trained the 20th, and the upright-gate bug of §28 zeroed its stair terms below the 12 cm line.
  At 5–6 cm both fall alike.
- **It stepped on more edges** (0.65–0.78 per riser against 0.31–0.56).
- **It no longer climbs** (0 of 200 at any height): a downhill specialist, as intended for a teacher
  that only teaches on frames with a drop ahead.

Files (gpu box): `logs/teacher18c/` (size, smoke, train, chain, evals at 1250–1998, impact at 1500 and the
end, down / up staircases, round trip); checkpoints `ckpts/stepdown_teacher_18c_last.pt` (run dir
`logs/rsl_rl/step_up/2026-11-10_21-09-45_step_down_teacher_18c`), 18a: `ckpts/stepdown_teacher_18a_last.pt`,
`logs/teacher18/`.

## 28. Run 18d: the stairs teacher on 20-riser flights, and the upright gate that switched off below 12 cm (2026-11-10, `Mjlab-StepDown-TeacherLongStairs-*`)

**Question:** 18c's falls per riser grow after the third riser, and its 6 s episodes rarely go
further. If the teacher trains on flights as long as the test staircase, does the hazard stay flat?

**Changes vs 18c** (`make_microduck_step_down_long_stairs_teacher_env_cfg`; RESUMES 18c
`model_1998` for 1000 iterations, 1999–2998):
- **terrain:** tiles 10 m long (x), the flight 20 risers (tread 11–24 cm, riser 1.5–4.5 cm by row);
  `single_drop` stays 1 of 4 columns;
- **episodes 15 s**; every robot gets a heading target (±0.15 rad, ω_z ±0.2 rad/s), so a long
  descent can correct its drift instead of walking off the side;
- rewards, weights and spawns as 18c (75 % stand at a random edge, so the spawns spread over the
  whole flight: x from the origin p50 1.7 m, p95 3.9 m).

### The bug: on a down tile every "upright" gate read the robot as fallen below 12 cm

A probe of 18c on the 18d env (stand spawns, 20-riser flights, 512 envs) showed the trunk going down
9 risers in 14 s (p50) while `drop_free_level_bonus` counted 0 new levels (p50) — it paid almost
nothing. The cause is in `mdp._fallen_mask`, the helper behind every upright gate. With
`gate_z_below = 0.0` (all the "upright" callers) it still marked a robot fallen when its trunk was
below the ENV ORIGIN. On flat and uphill tiles that never happens. On a down tile the origin is the
top platform, so a robot walking down the flight is "fallen" once it is ~12 cm below it: after
3 risers at 4 cm, 4 at 3 cm, 8 at 1.5 cm.

Terms switched off from there on (each multiplies by the upright mask):
- `clean_step_down`, `drop_free_step_down`;
- `track_vx_ema`, `track_angular_velocity`;
- `air_time`, `swing_apex`, `low_swing` (a cost).

It hit 18c hard: 6 risers up to 4.5 cm, so its stair terms went silent from the 3rd–4th riser (at
3–4.5 cm) or the 8th (at 1.5 cm) on. The step-up runs 12–17 were barely touched. Their
`down_flight` has 3 risers of ≤ 4 cm, so only the bottom tread of a 3.5–4 cm flight (and a crouch
there) fell below the line. Single drops never reached it.

**Fix (2026-11-10):** `gate_z_below <= 0` is tilt only. The height test stays for the recovery terms
that pass a positive height. Regression test
`test_upright_gate_ignores_height_below_a_downhill_origin`.

On the same probe after the fix, 18c reaches new levels on p50 8 risers per 15 s episode, and is
paid on p50 2 of them (the rest are forfeited to a trunk drop over 0.25 m/s).

**Iteration 0, after the fix** (18c `model_1998` on the 18d env, `step_down_teacher_eval.py`, 1024
envs, first 15 s episode; risers 1.5–2.5 / 2.5–3.5 / 3.5–4.5 cm):

| | flight, stand at an edge | flight, from the platform | single drop, stand |
| --- | --- | --- | --- |
| fell | 10 / 19 / 34 % | 20 / 33 / 52 % | 0 / 0 / 2 % |
| risers down per episode | 9.4 / 8.8 / 7.1 | 10.6 / 11.0 / 6.8 | 1 / 1 / 1 |
| of them drop-free | 49 / 33 / 29 % | 44 / 24 / 25 % | 93 / 96 / 82 % |
| trunk at the first leading lift-off p50 (mm) | 101 / 97 / 95 | 106 / 100 / 96 | 103 / 100 / 98 |

- Measured before the fix with the same eval, "risers down" read 0.7 / 0.3 / 0.2 per episode: the
  count stopped at the 12 cm line, not the robot.
- Reward per 15 s episode on flights (`step_updown_check.py`, before → after the fix):
  - `drop_free_step_down` 8.4 → 49.7;
  - `track_vx_ema` 3.5 → 31.9;
  - `clean_step_down` → 22.8 (stand spawns, flights).
- Every penalty is ≤ 0. `fall` is −2.6 and `foot_edge_contact` −13.8 on flights.

**Training** (`step_down_teacher_eval.py` on the 18d env, 1024 envs, 2048 at the end, first 15 s
episode, stand at an edge on a 20-riser flight, risers 1.5–2.5 / 2.5–3.5 / 3.5–4.5 cm; 20-riser
stairs: `step_up_staircase.py --down`, UpDownEdge task, 100 robots per riser at 2500):

| iteration | flight: fell | flight: risers down per episode | of them drop-free | trunk at the first lift-off p50 (mm) | single drop: success | 20-riser stairs: bottom, 2 / 3 / 4 / 5 / 6 cm |
| --- | --- | --- | --- | --- | --- | --- |
| 1998 (18c, fixed gate) | 10 / 19 / 34 % | 9.4 / 8.8 / 7.1 | 49 / 33 / 29 % | 101 / 97 / 95 | 93 / 96 / 82 % | 73 / 74 / 55 / 29 / 4 % (§27) |
| 2250 | 6 / 20 / 40 % | 9.6 / 8.6 / 7.7 | 49 / 33 / 29 % | 102 / 100 / 99 | 94 / 89 / 86 % | |
| 2500 | 10 / 16 / 37 % | 9.4 / 9.5 / 6.7 | 44 / 32 / 27 % | 102 / 98 / 97 | 95 / 96 / 80 % | 71 / 71 / 53 / 32 / 12 % |
| 2750 | 8 / 19 / 48 % | 9.3 / 9.1 / 6.9 | 46 / 32 / 28 % | 103 / 101 / 100 | 94 / 96 / 81 % | |
| 2997 | 10 / 20 / 32 % | 9.4 / 9.0 / 8.0 | 52 / 36 / 28 % | 103 / 100 / 99 | 97 / 92 / 83 % | below |

The teacher's own battery did not move in 1000 iterations. Its falls on a 15 s flight stay at
10 / 20 / 32–48 %, a drop-free riser stays 1 in 2 / 3 / 3.5, and the crouch is 3–4 mm shallower.
The training log agrees: `Episode_Reward/drop_free_step_down` was flat at 0.8–1.0 (per second) from
iteration 2050 on. Crouch and mid-step spawns at 2997 land drop-free on 52 / 36 / 32 % and
47 / 32 / 28 % of flight risers.

**Final, `model_2997`** (`ckpts/stepdown_teacher_18d_last.pt`) against run 15 and 18c on the §25
batteries: UpDownEdge task, leg-collision model, 20 risers, tread 20 cm, 0.25 m/s, 45 s, no heading
correction. The down staircase is `step_up_staircase.py`, 1000 robots; the rest is
`step_down_contacts.py`, 600 robots.

| | run 15 | 18c | 18d |
| --- | --- | --- | --- |
| down staircase: bottom of 200, 2 / 3 / 4 / 5 / 6 cm | 167 / 181 / 149 / 61 / 5 | 146 / 148 / 110 / 57 / 7 | **176** / 160 / 141 / **93** / **27** |
| same: fell | 2 / 13 / 44 / 131 / 176 | 7 / 24 / 74 / 124 / 176 | 5 / 20 / **37** / **95** / **150** |
| same: left the stairs sideways | 31 / 6 / 7 / 8 / 19 | 47 / 28 / 16 / 19 / 17 | 19 / 20 / 22 / 12 / 23 |
| fall hazard per riser at 4 cm, risers 1–3 / 4–6 / 7–10 / 11–15 / 16–20 (%) | 0.8 / 2.1 / 0.8 / 1.9 / 0.7 | 0.2 / 1.4 / 1.9 / 3.9 / 3.3 | 0.3 / 0.8 / 0.7 / 1.7 / 1.3 |
| same at 5 cm (%) | 3.3 / 5.8 / 6.7 / 6.1 / 4.9 | 1.2 / 2.7 / 6.0 / 5.1 / 8.3 | 0.5 / 1.9 / 4.7 / 4.2 / 3.6 |
| contacts battery: bottom / fell of 200, 2 / 3 / 4 cm | 194 / 187 / 141 · 4 / 13 / 56 | 180 / 157 / 115 · 18 / 41 / 82 | 188 / 175 / 145 · 9 / 22 / 55 |
| edge contacts per riser, 2 / 3 / 4 cm | 0.56 / 0.31 / 0.34 | 0.78 / 0.70 / 0.67 | 0.58 / 0.59 / 0.59 |
| trunk at the leading lift-off p50 (mm), 2 / 3 / 4 cm (target ≤ 100 / 90 / 80) | 124 / 124 / 124 | 105 / 101 / 98 | 108 / 106 / 104 |
| leading step-off: trunk v_z before touchdown (m/s) | −0.34 / −0.47 / −0.54 | −0.09 / −0.14 / −0.19 | −0.12 / −0.17 / −0.24 |
| leading step-off: peak foot force p50 (N) | 23 / 28 / 34 | 21 / 22 / 23 | 20 / 22 / 23 |

**What it shows:**
- **The long staircase got better.** 18d changed the gate and the horizon at once, so this run cannot separate them.
  - At 4 cm 18d reaches the bottom as often as run 15 (141 against 149) and falls less (37 against
    44). At 5–6 cm it is clearly better: 93 / 27 against 61 / 5.
  - At 2 cm it is the best so far (176). At 3 cm it is still behind run 15: 160 against 181, with
    20 falls against 13.
  - Its fall hazard no longer grows along the flight at 2–4 cm. At 4 cm it is 0.3–1.7 % per riser,
    against 18c's 0.2 → 3.9 %.
- **It keeps the soft landing.**
  - The leading step-off lands at 0.12–0.24 m/s against run 15's 0.34–0.54, with a third less peak
    foot force at 4 cm.
  - The crouch is 3–6 mm shallower than 18c's (104–108 mm at lift-off) and still not riser-scaled.
- **Edge contacts are 0.58 per riser.** That is fewer than 18c (0.67–0.78) but more than run 15 at
  3–4 cm (0.31 / 0.34). 9 of 10 of them are a foot tipped across the nose.
- **The teacher's own battery did not move in 1000 iterations** (15 s flights at 11–24 cm treads,
  heading targets). The gain shows on the fixed 20 cm-tread staircase only. At iteration 2500 that
  staircase read 71 / 71 / 53 / 32 / 12 % (100 robots per riser), so part of the gap to the final
  number may be sample noise.
- 18d is a downhill specialist like 18c: it was not trained or tested on up stairs.

Files (gpu box): `logs/teacher18d/`
- `size.log`, `base.log` (iteration 0 before and after the fix), `probe*.py`;
- `train.log`, `chain.log`;
- evals at 2250–2997 (`eval_18d_*.json`), `check_18d_last.log`, `impact_18d_last.*`;
- `down_18d_2500.*`, `down_18d_t20.*`.

Checkpoint `ckpts/stepdown_teacher_18d_last.pt` (run dir
`logs/rsl_rl/step_up/2026-11-10_22-29-09_step_down_teacher_18d`). The fall-hazard row is now
printed by `step_up_staircase.py` (also with `--from-json`).

## 29. Run 18b: distil the stairs teacher into the step-up policy (2026-11-11, `Mjlab-StepUp-TofMap-UpDownDistill-*`)

**Question:** 18d descends 20 risers as safely as run 15 (better at 5–6 cm) and lands at half its
speed, but cannot climb. Can the one step-up policy get 18d's descent and keep run 15's climb?

**Task** (`make_microduck_step_up_distill_env_cfg` in `tasks/microduck_step_down_teacher_env_cfg.py`):
- **Start:** RESUMES run 15 `model_1999` (curricula continue) for 1000 iterations, 1999–2998.
- **Env:** run 15's env: up, flat and down tiles, 3-riser flights, 8 s episodes, the fixed
  upright gate of §28.
- **Teacher terms:** on the down tiles only, 18d's `trunk_level` (`mdp.trunk_ground_tracking`),
  `trunk_level_l1` and `drop_free_step_down` (`mdp.drop_free_level_bonus`), with the teacher's
  weights and params. PPO is paid what the teacher was paid, so it does not pull against the BC.
- **Two BC anchors after every PPO update** (`climb_anchor.py`, new `extra_anchors` and mode
  `downhill`):
  - frames whose ToF map shows a drop ahead → 18d (`ckpts/stepdown_teacher_18d_last.pt`;
    5 epochs, lr 3e-4, as run 13);
  - frames that show a step up → run 15 itself (2 epochs, lr 1e-4, as run 14), so its climb stays;
  - flat frames are PPO's alone. The masks are disjoint (`tests/test_step_up_distill.py`).

**Iteration 0** (`step_updown_check.py` on the 18b env, 2048 envs, 8 s, own commands; down tiles
single_drop / down_flight):

| | run 15 (the student at 0) | 18d (the teacher) |
| --- | --- | --- |
| down tiles: fell | 2.6 / 5.1 % | 1.0 / 3.3 % |
| all risers down | 93 / 87 % | 93 / 89 % |
| `drop_free_step_down` per episode | 1.7 / 1.7 | 7.5 / 15.3 |
| `trunk_drop` | −1.7 / −4.6 | −0.7 / −1.9 |
| `trunk_level` (of 80) | 70 / 65 | 78 / 76 |
| up tiles (single_step / short_flight): fell | 1.4 / 1.6 % | 21 / 21 % |
| `height_progress` | 2.2 / 6.7 | −0.9 / −1.1 |
| frames with a drop ahead (downhill BC) | 19 / 32 % of the down tiles' frames | |

- Every penalty is ≤ 0 for run 15.
- In the smoke test, the first BC pass saw a downhill MSE of 0.20 rad² between run 15 and 18d, and
  0.0006 for the climb anchor (the student is the anchor).

**18b, the first try, lost the climb.** The climb anchor was as in run 14: uphill frames only,
2 epochs, lr 1e-4. The 20-riser staircases (`step_up_staircase.py`, 100 robots per riser, % of the
robots at 2 / 3 / 4 / 5 / 6 cm):

| | run 15 | 18d | 18b @ 2250 | 18b @ 2500 |
| --- | --- | --- | --- | --- |
| down: bottom | 84 / 90 / 74 / 30 / 2 | 88 / 80 / 70.5 / 46.5 / 13.5 | 66 / 66 / 47 / 18 / 3 | 82 / 81 / 47 / 20 / 4 |
| down: fell | 1 / 6 / 22 / 66 / 88 | 2.5 / 10 / 18.5 / 47.5 / 75 | 9 / 25 / 42 / 72 / 88 | 4 / 14 / 39 / 75 / 92 |
| up: top | 92.5 / 96 / 93.5 / 77 / 13.5 | 0 (18c) | | 82 / 75 / 25 / 2 / 0 |

- The downhill BC MSE fell from 0.15 to 0.025 rad² by iteration 2250: the student was taking on the
  teacher's actions.
- The climb anchor's MSE rose from 0.0001 to about 0.05 rad² and stayed there. The teacher's
  5 epochs at lr 3e-4 outweigh the climb's 2 at 1e-4, about 7×, on one shared network.
- By iteration 2500 both directions were worse than run 15. It was stopped at iteration 2645
  (checkpoints 2250 and 2500 kept, `logs/distill18b/`).

**18b2** (the registered recipe now): the run 15 anchor on EVERY frame without a drop ahead (mode
`not_downhill`: up, flat, the approach), as strong as the teacher's (5 epochs, lr 3e-4). Every
frame is regressed onto one of the two experts, chosen by the map. PPO with the teacher terms
on top. Same start (run 15 `model_1999`), 1000 iterations; the up staircase is checked at every
milestone. 18b itself is reproduced with
`MICRODUCK_CLIMB_ANCHOR_CFG="mode=uphill,epochs=2,learning_rate=0.0001"`.

**18b2 did not work either.** The 20-riser staircases, 100 robots per riser, % at 2 / 3 / 4 / 5 / 6 cm:

| iteration | down: bottom | down: fell | up: top | up: fell |
| --- | --- | --- | --- | --- |
| 1999 (run 15) | 84 / 90 / 74 / 30 / 2 | 1 / 6 / 22 / 66 / 88 | 92.5 / 96 / 93.5 / 77 / 13.5 | |
| 2250 | 72 / 68 / 32 / 3 / 2 | 5 / 17 / 58 / 96 / 87 | 81 / 91 / 76 / 31 / 2 | 1 / 2 / 22 / 65 / 94 |
| 2500 | 51 / 43 / 19 / 7 / 0 | 8 / 33 / 70 / 90 / 94 | 50 / 90 / 78 / 41 / 0 | 4 / 2 / 14 / 52 / 93 |
| 2750 | 75 / 63 / 28 / 5 / 0 | 6 / 30 / 62 / 93 / 99 | 76 / 87 / 73 / 18 / 0 | 3 / 7 / 22 / 77 / 96 |

- The climb anchor's MSE settled at about 0.02 rad² (18b: 0.05). The teacher's MSE stayed about
  0.05 rad² (18b: 0.025): the two anchors split the network between them.
- Neither direction matched its expert. Both got worse than run 15 within 250 iterations and
  stayed there.

### The hard switch: 18d while the map shows a drop ahead, run 15 otherwise (2026-11-11)

No training. The two networks are used as they are, switched every step by the same scan test
that split the BC frames (`climb_anchor.downhill_ahead_mask`).
- Eval spec: `--checkpoint switch:<down.pt>+<rest.pt>` in every eval that loads through
  `scripts/eval/_common.py:load_policy`.
- Batteries: §25 (UpDownEdge task, 20 risers, tread 20 cm, 0.25 m/s, 45 s), 1000 robots on each
  staircase, 600 on the contacts battery.

| | run 15 | 18d | switch |
| --- | --- | --- | --- |
| down staircase: bottom of 200, 2 / 3 / 4 / 5 / 6 cm | 167 / 181 / 149 / 61 / 5 | 176 / 160 / 141 / 93 / 27 | 171 / 161 / 149 / 80 / 20 |
| same: fell | 2 / 13 / 44 / 131 / 176 | 5 / 20 / 37 / 95 / 150 | 11 / 24 / 39 / 112 / 166 |
| same: fall hazard per riser at 4 cm, risers 1–3 / 4–6 / 7–10 / 11–15 / 16–20 (%) | 0.8 / 2.1 / 0.8 / 1.9 / 0.7 | 0.3 / 0.8 / 0.7 / 1.7 / 1.3 | 0.2 / 1.0 / 1.2 / 1.5 / 1.3 |
| up staircase: top of 200, 2 / 3 / 4 / 5 / 6 cm | 185 / 192 / 187 / 154 / 27 | 0 (18c) | 181 / 193 / 188 / 149 / 25 |
| contacts battery: bottom / fell of 200, 2 / 3 / 4 cm | 194 / 187 / 141 · 4 / 13 / 56 | 188 / 175 / 145 · 9 / 22 / 55 | 186 / 173 / 126 · 6 / 24 / 71 |
| edge contacts per riser, 2 / 3 / 4 cm | 0.56 / 0.31 / 0.34 | 0.58 / 0.59 / 0.59 | 0.57 / 0.55 / 0.56 |
| trunk at the leading lift-off p50 (mm) | 124 / 124 / 124 | 108 / 106 / 104 | 109 / 106 / 104 |
| leading step-off: trunk v_z (m/s) | −0.34 / −0.47 / −0.54 | −0.12 / −0.17 / −0.24 | −0.12 / −0.18 / −0.24 |

**What it shows:**
- The switch climbs like run 15 and descends like 18d, with 18d's soft landing.
- It falls a little more than 18d on the way down: 11 / 24 against 5 / 20 at 2–3 cm on the
  staircase, and 71 against 55 at 4 cm on the contacts battery. Part of that may be the switch
  itself (back to run 15 on the bottom platform, or a flicker of the mask on a tread). Part may be
  sample noise: the two 4 cm batteries disagree in sign.
- Distilling the two into one network (18b, 18b2) lost both skills. Switching between them keeps
  both.

Files (gpu box):
- `logs/distill18b/`: 18b, with `size.log` (iteration 0 of run 15 and 18d on the 18b env), smoke,
  train, and the staircases at 2250 / 2500. Run dir
  `logs/rsl_rl/step_up/2026-11-11_00-47-31_step_up_distill_18b`.
- `logs/distill18b2/`: 18b2 train, the staircases at 2250 / 2500 / 2750, and the switch
  (`switch.sh`, `down_switch_t20.*`, `up_switch_t20.*`, `impact_switch.*`). The 18b2 final chain was
  stopped once 2750 had settled it. Last checkpoint `ckpts/stepup_18b2_last.pt` (model_2998).

## 30. Head fine-tunes of 18d: head calm, head smooth, head roll; the lucky checkpoint (2026-11-11 → 13, `Mjlab-StairsDown-Head{Calm,Smooth,Roll}-*`)

**Question:** 18d descends with its head wobbling. Head yaw swings 27° rms, the yaw servo turns at
346 °/s rms, and head roll sits on its stop (|q| > 0.42 rad) 86 % of the descent. The head carries
the ToF sensor, so the map knows only 34 % of the cells ahead. Can a fine-tune calm the head and
keep the descent?

**Where the wobble comes from** (`scripts/eval/head_wobble.py`, new; 18d, 3 cm unless noted):
- The head is almost free in the reward: `head_pose_tracking` 0.5 at std 0.5, and `head_pose_bias`
  is a 1 s EMA that lets a symmetric swing cancel.
- Zero-shot ablations, no retraining (`--head-lowpass-hz` and holds, seed 1, 3 / 4 / 5 cm, 200 robots each):

| 18d with | bottom of 200 | fell | yaw servo ω rms (°/s) | map ahead known |
| --- | --- | --- | --- | --- |
| nothing | 145 / 131 / 75 | 16 / 50 / 106 | 302 / 324 / 338 | 0.34 |
| roll held at 0 | 151 / 127 / 87 | 12 / 49 / 94 | 291 / 310 / 332 | 0.40 |
| yaw and roll held at 0 | 134 / 119 / 58 | 21 / 58 / 124 | 111 / 120 / 134 | 0.69 |
| yaw low-pass 2 Hz | 136 / 120 / 69 | 26 / 58 / 111 | 190 / 202 / 210 | 0.40 |
| yaw low-pass 1 Hz | 135 / 121 / 73 | 29 / 64 / 105 | 146 / 152 / 158 | 0.45 |
| yaw low-pass 0.5 Hz | 136 / 110 / 62 | 38 / 73 / 112 | 128 / 129 / 133 | 0.51 |

- **Roll:** holding it costs nothing.
- **Yaw:** any filtering costs about 10 of 200 at 3–4 cm, about the same from 2 Hz to a full hold.
  That looks like the policy being off its trained loop more than needing a fast yaw. Even with yaw
  held, the yaw rate reads 111 °/s from the trunk's motion, so judge the yaw swing (output rms), not
  the rate.
- **Pitch** is used: it is the counterweight of the step-off. None of the fine-tunes price it.

**Three fine-tunes of 18d**, each a warm start (`MICRODUCK_WARM_START=1`) for 500 iterations. Every
inherited curriculum sits at its final stage from step 0, so only the new ramps start over. The head
calm and head smooth task ids were retired once head roll was swapped in (2026-11-13). Their cfgs and
tests are in `logs/backup/head_calm_smooth_2026-11-13.tar.gz` (gitignored, Mac).
- **Head calm** (retired `Mjlab-StairsDown-HeadCalm-*`):
  - `head_calm_l1` (`mdp.head_joint_error_l1`, self-negating) on head yaw and roll, weight
    1 → 2 → 4 at iterations 0 / 50 / 100
  - `head_roll_stop` (`mdp.joint_pos_limit_proximity`, a cost) on head roll within 0.10 rad of the
    limits, weight −5 → −10 at 50
- **Head calm, resumed** 499 → 998 at the same weights.
- **Head smooth** (retired `Mjlab-StairsDown-HeadSmooth-*`):
  - `head_yaw_vel` (`joint_vel_l2`) on the yaw servo, −0.0025 → −0.01
  - `head_roll_l1` with weight 1
  - `head_roll_stop` with weight −5
- **Control:** the same warm start with no head terms (`--agent.experiment_name stairs_down_ctrl`).

**At first all three looked like they cost the descent**, judged against 18d `model_2997`. That
checkpoint scored 169 / 165 / 144 / 95 / 33 on the down battery (§29 settings, seed 0, 200 robots per
height, 2–6 cm):

| checkpoint | bottom of 200 | fell | left sideways | yaw ω rms (°/s) | roll on its stop | map ahead |
| --- | --- | --- | --- | --- | --- | --- |
| 18d 2997 | 169 / 165 / 144 / 95 / 33 | 5 / 19 / 43 / 93 / 146 | 26 / 16 / 13 / 12 / 21 | 346 | 86 % | 0.34 |
| head calm 499 | 162 / 147 / 130 / 89 / 20 | 8 / 24 / 43 / 92 / 153 | 30 / 29 / 27 / 19 / 27 | 202 | 38 % | 0.44 |
| head calm 998 (resumed) | 136 / 130 / 83 / 49 / 21 | 21 / 41 / 88 / 130 / 149 | | 192 | 12 % | 0.37 |
| head smooth 499 | 134 / 134 / 115 / 75 / 35 | 10 / 27 / 53 / 89 / 140 | 56 / 39 / 32 / 36 / 25 | 261 | 46 % | 0.31 |
| control 499 | 130 / 145 / 101 / 59 / 20 | 12 / 33 / 70 / 108 / 162 | 58 / 22 / 29 / 33 / 18 | 374 | 87 % | 0.30 |

**The control lost as much as the head fine-tunes, so the head terms were not the cause.**
- **Same cfg:** 18d's `params/env.yaml` and the control's are identical. The agent cfgs differ only
  in iterations and names.
- **Training did not regress.** Over its last 50 iterations the control matches 18d's last 50
  (iterations 2947–2997) on every term: mean reward 85.8 against 84.2, `fell_over` 4.9 against 4.6,
  `descent_progress` 1.08 against 1.03.
- **The battery varies far more between checkpoints of one run than between seeds:**

| | bottom of 200, 2 / 3 / 4 / 5 / 6 cm |
| --- | --- |
| 18d 2997, seeds 0 / 1 / 2 | 169 / 165 / 144 / 95 / 33 · 161 / 164 / 153 / 81 / 30 · 172 / 167 / 138 / 93 / 43 |
| control model_0 (18d + 1 update) | 171 / 161 / 131 / 83 / 29 |
| 18d 2250 | 155 / 146 / 111 / 51 / 12 |
| 18d 2500 | 144 / 138 / 114 / 60 / 33 |
| 18d 2750 | 132 / 129 / 84 / 48 / 20 |

18d's last checkpoint is the best of its own run on this battery. Its siblings 250 iterations apart
average about 144 / 138 / 103 / 53 / 22. §28 already showed the spread: at iteration 2500, 18d read
71 / 71 / 53 / 32 / 12 % at 100 robots per height. The fine-tunes had fallen back to the run's
typical level, not below it.

**Why the battery is so checkpoint-sensitive:** it runs 45 s on fixed 20 cm treads with no heading
command. Training uses 15 s episodes with heading targets. So straightness over a long flight (robots
leaving sideways: 12–60 between siblings) and falls late in a flight are barely pinned by the reward.

**Re-judged against the spread.** Each checkpoint got two descent samples: A, the battery (seed 0,
2–6 cm), and B, `head_wobble.py` (seed 1, 3 / 4 / 5 cm, 600 robots, 45 s).

| run, mean of 4 checkpoints | A: bottom of 200 | B: bottom of 200 |
| --- | --- | --- |
| 18d (2250 / 2500 / 2750 / 2997) | 150 / 145 / 113 / 64 / 25 | 131 / 100 / 52 |
| head calm (250 / 499 / 750 / 998) | 146 / 135 / 104 / 65 / 23 | 119 / 100 / 53 |

- **Descent:** head calm is level with 18d's run at 4–6 cm and about 10 robots lower at 3 cm.
  `model_499` is its best checkpoint, equal to 2997 on B (139 / 131 / 75 against 145 / 131 / 75).
- **Head:** every 18d checkpoint and the control wobble alike (yaw ω 340–374 °/s, roll on its stop
  83–87 %), so the calming is the penalty's doing, not checkpoint noise.
- **Not caused by head calm:** "head looks further down" and "trunk roll up" were features of the
  2997 outlier. 18d's siblings read −10 to −13° and 5.4–6.1°.
- **Caused by head calm:** head pitch rms rose 2–4°.

**Head roll** (`make_microduck_stairs_down_head_roll_env_cfg`, experiment `stairs_down_head_roll`,
save every 50). A warm start from head calm `model_499` with roll priced harder and yaw lighter:
- `head_roll_l1` on roll, weight 4 → 6 → 8 at iterations 0 / 50 / 100
- `head_yaw_l1` on yaw, weight 2 (head calm: 4 on yaw and roll)
- `head_roll_stop` with margin 0.12, weight −10 → −20 → −30
- pitch free

Over its last 50 iterations against head calm's: `drop_free` 1.30 (1.11), `descent_progress` 1.12
(1.17), `fell_over` 3.93 (3.72), every penalty ≤ 0. Roll on its stop at 3 cm, by checkpoint:
50 → 40 %, 200 → 19 %, 350 → 10 %, 400 → 5.4 %, 450 → 3.7 %, 499 → 0.5 %.

| checkpoint | A: bottom of 200 | A: fell | B: bottom | yaw ω (°/s) | yaw out rms | roll out rms | roll on its stop, 3 / 4 / 5 cm | map ahead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18d 2997 | 169 / 165 / 144 / 95 / 33 | 5 / 19 / 43 / 93 / 146 | 145 / 131 / 75 | 346 | 27.2° | 13.0° | 86 / 87 / 87 % | 0.34 |
| head calm 499 | 162 / 147 / 130 / 89 / 20 | 8 / 24 / 43 / 92 / 153 | 139 / 131 / 75 | 202 | 15.4° | 12.0° | 38 / 41 / 44 % | 0.44 |
| head roll 300 | 167 / 159 / 147 / 98 / 40 | 4 / 13 / 41 / 84 / 143 | 149 / 145 / 78 | 239 | 17.9° | 10.3° | 16 % at 3 cm | 0.56 |
| head roll 450 | 158 / 150 / 117 / 84 / 32 | 5 / 17 / 35 / 85 / 125 | 127 / 120 / 64 | 219 | 16.7° | 7.3° | 3.7 / 4.4 / 4.9 % | 0.54 |
| head roll 499 | 139 / 148 / 125 / 86 / 37 | 9 / 21 / 48 / 94 / 132 | 120 / 115 / 76 | 223 | 17.3° | 5.5° | 0.5 / 1.0 / 1.5 % | 0.52 |

- **Roll:** below 5 % at every height from 450 on.
- **Descent:** 450 and 499 sit at or above 18d's run mean at 3–6 cm. 499's falls are at 2997's
  level; its 2 cm shortfall is robots leaving sideways (52 of 200, against 26).
- **Yaw:** a little worse than head calm, still a third below 18d.
- **Gaze:** the head no longer looks further down (mean world pitch −4 to +2°; head calm −11°, 18d
  −5°). Head pitch is 19–20° rms (18d 16.6°).
- **Pick:** `model_499`, whose roll is essentially never on its stop. Runner-up 450.

**Swap checks** (2026-11-13):
- **ONNX:** `scripts/export.py Mjlab-StairsDown-HeadRoll-Backlash-MicroDuck --checkpoint-file
  <run>/model_499.pt` gives obs [1, 327] and actions [1, 14]. Its metadata carries
  `checkpoint_sha256` = the .pt's sha256.
- **The rehearsal runs in sim.** `infer_policy.py` cannot drive a 327-D policy, and robotd has no
  327-D kind yet. `scripts/eval/_common.load_policy` now loads `*.onnx`: onnxruntime on the CPU, one
  robot at a time, the exact exported file. Either half of a `switch:` can be one.
- **Parity:** over 64 robots × 1500 steps on the 3 cm down staircase, the largest
  |a_pt − a_onnx| is 3.3e-6 (mean |a| 0.50).
- **Switch battery** (1000 robots, §29 settings, the 18d switch re-run alongside):

| | switch with 18d | switch with head roll 499 (.pt) | same, ONNX down half |
| --- | --- | --- | --- |
| down: bottom of 200 | 173 / 162 / 131 / 82 / 26 | 156 / 162 / 141 / 100 / 37 | 155 / 160 / 128 / 90 / 27 |
| down: fell | 8 / 23 / 57 / 109 / 158 | 6 / 23 / 46 / 90 / 152 | 9 / 23 / 59 / 94 / 163 |
| down: left sideways | 19 / 15 / 12 / 9 / 16 | 38 / 15 / 13 / 10 / 11 | 36 / 17 / 13 / 16 / 10 |
| up: top of 200 | 180 / 195 / 188 / 149 / 25 | 185 / 192 / 186 / 158 / 30 | |
| up: fell | 0 / 0 / 6 / 44 / 157 | 0 / 1 / 7 / 32 / 157 | |

- **Contacts battery** (the down policy alone, 600 robots, 2 / 3 / 4 cm), 18d → head roll 499:

| | 18d | head roll 499 |
| --- | --- | --- |
| bottom of 200 | 188 / 167 / 145 | 172 / 152 / 136 |
| fell | 8 / 33 / 55 | 17 / 37 / 52 |
| edge contacts per riser | 0.61 / 0.59 / 0.58 | 0.38 / 0.34 / 0.34 |
| clean leading crossings | 80 / 79 / 78 % | 90 / 91 / 90 % |
| trunk at the leading lift-off p50 (mm) | 108 / 106 / 104 | 106 / 103 / 101 |
| leading step-off: trunk v_z before touchdown (m/s) | −0.12 / −0.17 / −0.23 | −0.08 / −0.13 / −0.18 |
| peak foot force p50 (N) | 20.2 / 22.1 / 23.3 | 19.7 / 21.4 / 22.6 |

**What it shows:**
- **In the switch,** head roll 499 descends as well as 18d from 3 cm up, better at 4–6 cm on the .pt
  run, and 17 fewer at 2 cm, where robots leave sideways. Climbing is unchanged.
- **On its own,** it touches 40 % fewer edges and lands its step-offs softer.
- **What to watch:** more falls at 2 cm on the contacts battery (17 against 8). That is measured
  against the lucky 18d checkpoint; no other 18d checkpoint was run on that battery.
- **Swapped in 2026-11-13:** `ckpts/stairs/stairs_down.pt` = head roll `model_499`,
  `stairs_down.onnx` its export, and 18d kept as `stairs_down_18d.pt`.

**The general lesson:** judge a stairs fine-tune by several checkpoints of each run (save every 50)
and compare spreads. A single final checkpoint can be the run's best or its worst.

Files (gpu box, `~/microduck_rl_playground`):
- Run dirs under `logs/rsl_rl/`:
  - head calm: `stairs_down_head_calm/2026-11-11_13-00-04_*`, resumed in `2026-11-11_21-48-41_*`
  - head smooth: `stairs_down_head_smooth/2026-11-12_09-15-27_*`
  - control: `stairs_down_ctrl/2026-11-12_10-09-56_*`
  - head roll: `stairs_down_head_roll/2026-11-12_23-02-39_*`
- Evals under `mk/`:
  - `head_wobble/`: the zero-shot ablations
  - `head_calm/eval/`, `head_smooth/`, `down_ctrl/eval/`
  - `down_var/`: 18d's siblings and seeds
  - `rejudge/`
  - `head_roll/eval/`: every checkpoint
  - `head_roll/swap/`: ONNX, parity and the switch and contacts batteries
