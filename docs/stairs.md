# Stairs — up, down, and the switch between them

Status 2026-11-11. Two 327-D perceptive policies (one for going up, one for going down) and a
hand-written switch between them. Everything here is simulation; nothing has run on the robot.

Every number cites its source as §N: section N of the run diary
[`docs/archive/stairs_runlog.md`](archive/stairs_runlog.md) (runs 6 → 18d). Read the diary for
the full tables and for runs not covered here. Unless a line says otherwise, a number comes from
the **20-riser staircase battery**: one straight staircase per riser height, 20 risers, tread 20 cm,
0.25 m/s forward command, head-down command, no heading correction, 45 s, 200 robots per height,
on the leg-collision robot model (§23–§29).

## Summary

- **Up.** The stairs-up policy reaches the top of 20 risers in 185 / 192 / 187 / 154 / 27 of 200
  runs at 2 / 3 / 4 / 5 / 6 cm risers, and 0 at 7 cm (§23). Its median clean run (risers before
  the first toe catch or edge landing) is 19 / >20 / >20 / 19 / 5 (§23).
- **Down.** The stairs-down policy reaches the bottom in 176 / 160 / 141 / 93 / 27 of 200 at
  2–6 cm and falls in 5 / 20 / 37 / 95 / 150 (§28). It lowers the trunk before the edge (104–108 mm
  at the leading foot's lift-off, against 124 mm for the up policy) and its leading foot lands at
  0.12–0.24 m/s, against 0.34–0.54 m/s for the up policy going down (§28).
- **Switch.** Using the down policy while the map shows a drop ahead, and the up policy otherwise,
  keeps both skills. Up: 181 / 193 / 188 / 149 / 25 of 200 reach the top. Down: 171 / 161 / 149 / 80 / 20
  reach the bottom, and 11 / 24 / 39 / 112 / 166 fall (§29).
- **No build-up along the flight.** On the way down at 4 cm the switch's fall hazard per riser is
  0.2–1.5 % in every band from riser 1 to riser 20 (§29).
- **Not clean.** Going down, the switch makes 0.55–0.57 edge contacts per riser (§29). Nine in ten
  of the down policy's contacts are a foot tipped across the nose of a tread (§28).
- **Envelope.** Up to 4 cm both directions work. At 5 cm the switch gets 75 % of the robots to the
  top and 40 % to the bottom; at 6 cm, 13 % and 10 % (§29). Statically, a step down without a free
  fall exists only up to 5 cm risers, and only from a deep crouch (trunk 10 / 9 / 8 / 7 cm for
  2 / 3 / 4 / 5 cm). Torque is never the limit: at most 0.29 N·m against the 0.64 N·m cap (§22).
- **Turning on top** is not part of either policy (trained with no turn in place). VelStand turns
  in place in a median 3.3–3.5 s with no falls (§18). The ToF map must be cleared at the turn (§19).

Caveats:
- **Simulation only, on softened terrain** (contact solref 0.04 s; MuJoCo's default is 0.02). The
  final policies were never run on stiff terrain; the one stiff check, run 12's descent, behaved the
  same (§20).
- **No hardware test.** The sim ToF casts one ray per zone; the real sensor integrates a cone and
  blurs edges. Real frames have never been compared with the sim map (§14, §15).
- **Cannot be deployed yet.** The runtime refuses any policy with `obs_len != 61` and does not build
  the height map (§2, §15). `publish` has no 327-D policy kind (§7).
- **The switch is a hand rule** (a fixed scan threshold, evaluated every step), not a learned
  arbiter. Its round trip (up, turn, down) has not been measured.
- **Both policies come from long warm-started chains.** Retraining either one from scratch with the
  final cfg was never tested.

## Perception (shared by both policies)

- **Sensor:** the head VL53L8CX (8×8 zones, 45° field of view, 15 Hz, 0.12–3.54 m, §1), modelled in
  `src/mjlab_microduck/tasks/tof.py` with one frame of latency, range noise 1 % + 4 mm, 5 % dropouts
  and ±3° mounting misalignment (§3).
- **Height map** (`src/mjlab_microduck/tasks/tof_map.py`): a 2 cm grid fixed in the odometry frame of
  the episode start. Each cell keeps the running mean of the ToF hit heights, spread one cell
  sideways. The odometry drifts (±3 % scale, ±2°/m yaw, 1 cm/√m random walk, ±1 cm/m in z). Against
  the exact terrain while approaching a riser, 74 % of the foot-zone cells are known and the height
  error is 0.2 cm median, 1.1 cm p95 (§14).
- **What the actor reads:** a 19 × 7 scan of the map at 3 cm spacing, x −9…+45 cm and y ±9 cm in
  the trunk's estimated yaw frame, clipped to ±0.2 m. It is followed by a known-cell mask; unknown
  cells read 0. Obs = the family's 61-D contract, bit-identical, + 133 + 133 = **327-D**. The critic
  gets the exact height scan (§15).
- **Why a map:** the riser leaves the ToF frame 1.24 s / 37 cm before the foot lands on it. With the
  raw frame the duck climbed by kicking the riser: 17 / 345 clean climbs and 79 falls. With the map:
  228 / 345 and 6 falls (§15).
- **Network:** actor and critic MLPs 512-256-128, ELU, obs normalization (baked into the ONNX by
  `scripts/export.py`). The head_pose slot carries a head-down command in training and in the evals
  (`--head down`); a zero head command gives the same numbers within noise (§18).

## Stairs up

| | |
| --- | --- |
| task | `Mjlab-StairsUp-Backlash-MicroDuck` (trained as `Mjlab-StepUp-TofMap-UpDownEdge-Backlash-MicroDuck`) |
| cfg | `src/mjlab_microduck/tasks/microduck_stairs_up_env_cfg.py`, `make_microduck_stairs_up_env_cfg` |
| checkpoint | `ckpts/stairs/stairs_up.pt` = run 15 `model_1999` (§23) |

**What the env trains** (the design was built up over runs 8–15, see the sections cited):
- **Terrain menu** (§19): flat 10 %, one riser up 25 %, a 3-riser flight up 25 %, one riser down 15 %,
  and a 3-riser flight down 25 %. Risers are 1.5–4 cm by terrain row; treads 11–24 cm (§16).
  Episodes last 8 s, and the robot model has colliding shins and thighs (§23).
- **Commands:** forward 0.1–0.3 m/s, plus the head-down command (§18, §20).
- **Edge pricing on clean feet** (§11): `foot_edge_contact` charges a loaded foot whose footprint
  spans two levels. There is also a one-off fall penalty, and the terrain ladder promotes only clean
  episodes. The cost is ×2 on the down tiles (§20). There it tests a sole shrunk by 8 mm, so a flat
  foot just behind the nose is not charged (§23).
- **Potential-based height progress:** `height_progress` pays Δ trunk z, clipped at the riser.
  Rising pays and standing on top pays zero (§4). The downhill mirror `descent_progress` uses the
  ground under the feet, so a fall is not paid as progress (§19).
- **Placement on treads:** `foot_placement_bonus` pays a touchdown on one level with a stride of
  ≥ 4 cm, inside a flight, both directions (§16, §19). Step bonuses measure the sole: the foot site
  sits 13 mm higher and had paid 4 per episode on flat ground (§19).
- **Downhill terms:** `foot_below_step` (trunk-gated, against probing a foot over the edge) and
  `clean_step_down`, which pays 0.5 per cm per foot only for touchdowns with no edge contact (§19, §20).
- **Trunk-drop price:** `trunk_drop_penalty` (weight 150) charges the trunk's downward speed beyond
  0.3 m/s on the down tiles. In a zone from 8 cm before to 14 cm past a down edge, the cap is
  0.15 m/s, speed tracking is capped at 0.08 m/s, and backward lean and a crouched leg chain are not
  charged (§21, §23).
- **Edge spawns:** 30 % of downhill resets start mid-step at an edge, in a §22 crouch pose (§23).

**How it was reached:** a chain of warm starts, each from the previous run's final checkpoint. Run 4
(blind stairs, 61-D) → run 8 (oracle height scan, §11) → run 10 (ToF map, §15) → 11b (tread DR,
placement, §16) → run 12 (down tiles, §19) → 13n (clean step-down pay, §20) → 14n (trunk-drop price,
§21) → run 15 (edge zone, leg collisions, edge spawns, §23). That is about 23 000 PPO iterations
after run 4. Retraining from scratch with this final cfg was never tested.

**Measured** (§23, run 15 against its parent 14n on the same model):
- **Up:** top of 200, 185 / 192 / 187 / 154 / 27 / 0 at 2…7 cm. Median clean run 19 / >20 / >20 / 19 / 5 / 2,
  against 14n's 13 / 16 / 16 / 11 / 5 / 2.
- **Down (it can descend, but drops):** bottom of 200, 167 / 181 / 149 / 61 / 5; fell
  2 / 13 / 44 / 131 / 176 (§28). The trunk is 124 mm over the upper tread at the leading lift-off
  (no crouch), and the leading foot lands at 0.34–0.54 m/s.
- **Round trip** (§21 setup: it climbs, VelStand turns, it descends, map cleared at the turn): bottom
  of 100, 99 / 93 / 57 / 26 at 2–5 cm.
- **Single risers** were not re-measured for this policy. Its ancestor 11b climbed a single 6.4 cm
  step 88 % clean and fell 35 % at 7.4 cm; for the run 11 policies the cliff is between 8 and 9 cm (§16).

## Stairs down

| | |
| --- | --- |
| task | `Mjlab-StairsDown-Backlash-MicroDuck` (trained as `Mjlab-StepDown-TeacherLongStairs-Backlash-MicroDuck`) |
| cfg | `src/mjlab_microduck/tasks/microduck_stairs_down_env_cfg.py`, `make_microduck_stairs_down_env_cfg` |
| checkpoint | `ckpts/stairs/stairs_down.pt` = run 18d `model_2997` (§28) |

**The idea:** the walking policy never crouched when only rewarded for it (see "What did not work").
The tasks in this repo that did learn a crouch (ground pick, sitstand, roller crouch) make the
trunk height their main term (§26). So does this one:
- **`trunk_level`** (main term, weight 10) and **`trunk_level_l1`** (weight 100): the trunk tracks a
  slewed target of 12 cm − riser over the tread it is about to leave, while a down edge is within
  14 cm ahead. Once a foot is down, the target is 12 cm over the lower tread, and so on down one riser
  at a time. The slew is 10 cm/s, and being ahead of it scores like being behind, so arriving early
  is no jackpot (§26, §27). 12 cm − riser is §22's highest trunk from which a drop-free step exists.
- **Drop-free completion gate:** `drop_free_level_bonus` pays 15 per riser when both feet stand clean
  on a new lowest level. It is forfeited if the trunk fell faster than 0.25 m/s since the previous
  level, so it acts as a hard gate, not a nudge (§26, §27).
- **Spawn banks:** 75 % stand at a random edge of the flight (toe 2–12 cm before it), 25 % start on
  the platform. Of these, 20 % are overridden by a crouch at the edge (94 poses from §22's solver)
  and 15 % by run 15's mid-step pose (§26, §27).
- **Terrain:** 20-riser flights on 10 m tiles (3 of 4 columns) and single drops (1 of 4). Risers
  are 1.5–4.5 cm by row, treads 11–24 cm, episodes 15 s. Every robot gets a heading target (±0.15
  rad) so that a long descent can correct its drift (§28).
- Run 15's downhill terms stay on (`clean_step_down`, `descent_progress`, the edge cost and
  `trunk_drop`) (§26).

**Lineage:** run 15 → 18a → 18c → 18d, 1000 iterations each. 18a (§26) was a warm start from run 15
(`MICRODUCK_WARM_START=1`) on one riser with 4 s episodes. It learned a drop-free single step down,
which runs 15–17 never did: from a stand, 90–99.5 % at iteration 500 and 96–99.5 % at 1000. 18c (§27)
resumed it on 6-riser flights with per-riser terms; 18d (§28) resumed 18c on 20-riser flights with
15 s episodes and heading targets.

**It depended on a bug fix.** `mdp._fallen_mask`, behind every "upright" gate, also marked a robot
fallen when its trunk was below the env origin. On a down tile the origin is the top platform, so
about 12 cm down (3 risers at 4 cm) every upright-gated term went to zero: `clean_step_down`,
`drop_free_step_down`, `track_vx_ema`, `track_angular_velocity`, `air_time`, `swing_apex` and
`low_swing`.
- **Fix:** a non-positive `gate_z_below` now tests tilt only; regression test
  `test_upright_gate_ignores_height_below_a_downhill_origin`.
- **Effect:** with 18c's checkpoint on the 18d env (15 s flights), the fix raised
  `drop_free_step_down` from 8.4 to 49.7 per episode and `track_vx_ema` from 3.5 to 31.9.
- **Who it hit:** 18c trained with it: its stair terms went silent from the 3rd–4th riser. Runs 12–17
  (the up policy included) and 18a were barely touched, because their down tiles have at most 3
  risers of ≤ 4 cm, or a single drop (§28).

**Measured** (§28, against run 15):
- **Down:** bottom of 200, 176 / 160 / 141 / 93 / 27 at 2–6 cm (run 15: 167 / 181 / 149 / 61 / 5).
  Fell 5 / 20 / 37 / 95 / 150 (run 15: 2 / 13 / 44 / 131 / 176). Left the stairs sideways
  19 / 20 / 22 / 12 / 23.
- **Fall hazard per riser** at 4 cm, by riser band 1–3 / 4–6 / 7–10 / 11–15 / 16–20: 0.3 / 0.8 / 0.7 /
  1.7 / 1.3 %, flat along the flight.
- **Leading step-off:** trunk at lift-off 108 / 106 / 104 mm at 2 / 3 / 4 cm (run 15: 124). The
  target is ≤ 100 / 90 / 80, so the crouch does not scale with the riser. Landing speed 0.12–0.24 m/s
  (run 15: 0.34–0.54), peak foot force p50 20–23 N (run 15: 23–34 N).
- **Edge contacts:** 0.58–0.59 per riser (run 15: 0.31–0.56).
- **Caveat:** 18d changed the gate and the horizon at once, so the run cannot separate the two. Its own
  training battery did not move in 1000 iterations (§28).
- **It cannot climb.** Its parent 18c reaches the top of 0 of 200 staircases at any height (§27);
  18d itself was never tested going up.

## Combining them: the switch

`src/mjlab_microduck/tasks/stairs_switch.py` runs both networks and picks one per control step. The
down policy runs while the height scan shows a drop ahead; the up policy runs otherwise.
- **The rule** (§20, §29): a frame is "drop ahead" when the mean of any scan row ahead of the feet
  (x > 3 cm, at least 3 of 7 cells known) is more than 8 mm below the mean under the robot. The
  lowest training riser is 15 mm.
- **Why row means:** taking the deepest single cell flagged 29 % of flat frames as downhill, from map
  noise (§20).
- **No training is involved.** Evals load it with
  `--checkpoint switch:ckpts/stairs/stairs_down.pt+ckpts/stairs/stairs_up.pt` (down first), through
  `scripts/eval/_common.py:load_policy`.

| §29, 20-riser battery | up policy (run 15) | down policy (18d) | switch |
| --- | --- | --- | --- |
| up: top of 200, 2 / 3 / 4 / 5 / 6 cm | 185 / 192 / 187 / 154 / 27 | 0 (18c, §27) | 181 / 193 / 188 / 149 / 25 |
| down: bottom of 200 | 167 / 181 / 149 / 61 / 5 | 176 / 160 / 141 / 93 / 27 | 171 / 161 / 149 / 80 / 20 |
| down: fell of 200 | 2 / 13 / 44 / 131 / 176 | 5 / 20 / 37 / 95 / 150 | 11 / 24 / 39 / 112 / 166 |
| down: fall hazard per riser at 4 cm, risers 1–3 … 16–20 (%) | 0.8 / 2.1 / 0.8 / 1.9 / 0.7 | 0.3 / 0.8 / 0.7 / 1.7 / 1.3 | 0.2 / 1.0 / 1.2 / 1.5 / 1.3 |
| edge contacts per riser down, 2 / 3 / 4 cm (contacts battery, 600 robots) | 0.56 / 0.31 / 0.34 | 0.58 / 0.59 / 0.59 | 0.57 / 0.55 / 0.56 |
| trunk at the leading lift-off p50 (mm) | 124 / 124 / 124 | 108 / 106 / 104 | 109 / 106 / 104 |
| leading step-off: trunk v_z before touchdown (m/s) | −0.34 / −0.47 / −0.54 | −0.12 / −0.17 / −0.24 | −0.12 / −0.18 / −0.24 |

**The switch falls a little more than 18d going down** (11 / 24 against 5 / 20 at 2–3 cm). Possible
causes, untested: the hand-back to the up policy on the bottom platform, or a mask flicker on a
tread. Part may be noise: at 4 cm the staircase battery has the switch reaching the bottom more
often than 18d (149 against 141), the contacts battery less often (126 against 145) (§29).

**Why not one distilled policy** (§29). 18b resumed run 15 with 18d's terms on the down tiles. After
every PPO update, two BC anchors regressed the network: onto 18d on frames with a drop ahead (5
epochs, lr 3e-4), and onto run 15 on frames with a step up (2 epochs, lr 1e-4).
- **18b lost the climb** (5 cm top 77 → 2 %) and the descent got worse too (4 cm bottom 74 → 47 %).
- **18b2 balanced the anchors.** Run 15 anchored every frame without a drop, at the teacher's
  strength. Both directions were worse than run 15 within 250 iterations and stayed there (4 cm down:
  bottom 74 → 19–32 %; 5 cm up: top 77 → 18–41 %).
- **Why:** the two anchors split one network between them (BC errors settled at about 0.02 and
  0.05 rad²).

Switching between the two experts keeps both skills.

## What did not work

| approach | result | read |
| --- | --- | --- |
| Blind stairs (runs 1–5, no perception) | ceiling ≈ 2 cm risers; on 2.4–3 cm the knee sits at the 0.64 N·m current cap | archive preamble, `microduck_walk_v2_terrain_env_cfg.py` docstring |
| ToF frame without memory (runs 6–9) | the duck climbs by kicking the riser: 17 / 345 clean, 79 falls | §10, §13 |
| A reward to refuse risers too high (11a) | climbed 74 of the 77 risers it was told to refuse; refusal lives in the playground orchestrator | §16 |
| BC anchor to 11b to keep the climb while learning to descend (run 13) | turned the policy back into 11b; down-flight falls 2 → 20–25 % | §20 |
| Slower command for a cleaner step-off | not cleaner at 0.10 / 0.15 m/s for any policy | §20 |
| Crouch by reward in the walking policy (runs 15–17: free the crouch, price the drop ×5–10, pay a soft step, pay the crouch) | trunk 121–125 mm at lift-off in every variant; a higher drop price buys refusal (16a: 0 of 1000 reach the bottom) | §23–§25 |
| Distilling down + up into one network (18b, 18b2) | both skills lost | §29 |

## How to reproduce and use

**Checkpoints.** Both are gitignored (`*.pt`). Copies are under `ckpts/stairs/` on the Mac and under
`~/microduck_rl_playground/ckpts/stairs/` on the GPU box. Each is 7.4 MB: a full rsl_rl checkpoint
including the optimizer. Future runs log under the experiment names `stairs_up` / `stairs_down`.

| file | = | sha256 | GPU-box run dir |
| --- | --- | --- | --- |
| `stairs_up.pt` | run 15 `model_1999` (box copy `ckpts/stepup_15_model_1999.pt`) | `a6481bedcdfaf13902457969c5bc755057af18a99e84894d0a83fafe9016690a` | `logs/rsl_rl/step_up/2026-11-09_22-52-19_step_up_tofmap_updown_edge_15/` |
| `stairs_down.pt` | run 18d `model_2997` (box copy `ckpts/stepdown_teacher_18d_last.pt`) | `e2c03de940e06e049f9e850a71aa28e8214e9a1a695c65ea8c45ec0e91890cce` | `logs/rsl_rl/step_up/2026-11-10_22-29-09_step_down_teacher_18d/` |

They are not on the Hub. Sharing them is opt-in, with `uv run share-training`
(`docs/training_checkpoints.md`).

**Evals.** All three policies are evaluated on the up task, which builds the battery's staircase
terrain. The §25–§29 numbers come from these settings:

```bash
CK=ckpts/stairs/stairs_up.pt     # or ckpts/stairs/stairs_down.pt, or
                                 # switch:ckpts/stairs/stairs_down.pt+ckpts/stairs/stairs_up.pt
# 20-riser staircase, 2-6 cm, 200 robots per height (add --heights 2 3 4 5 6 7 for 7 cm)
uv run python scripts/eval/stairs_battery.py --task Mjlab-StairsUp-Backlash-MicroDuck \
    --checkpoint $CK --device cuda --num-envs 1000 --tread 0.20 --rows 2 --seconds 45 \
    --map-x 7.0 --head down --json up.json            # add --down for the descent
uv run python scripts/eval/stairs_battery.py --from-json up.json down.json   # reprint, fall-hazard rows included
# Edge contacts, step-off lift-off, landing speed and force (600 robots at 2 / 3 / 4 cm; defaults = the battery)
uv run python scripts/eval/stairs_contacts.py --task Mjlab-StairsUp-Backlash-MicroDuck \
    --checkpoint $CK --device cuda --json contacts.json   # --hard-terrain: solref 0.02
```

- **Round trip** (climb, a VelStand turn, descend; §18, §19): its script `step_up_roundtrip.py` was
  retired in the cleanup and is kept only in `logs/backup/pre_cleanup_2026-11-11.tar.gz` (pre-cleanup
  task ids).

**Fine-tuning.** Run the 64-env, 5-iteration smoke test first (AGENTS.md).

```bash
uv run train Mjlab-StairsUp-Backlash-MicroDuck --env.scene.num-envs 4096 \
    --init-checkpoint ckpts/stairs/stairs_up.pt
```

`--init-checkpoint` resumes: the iteration and step counter come back, so step-based curricula
continue at their final stage. That is right for the same task. Starting the other task from a
checkpoint is a warm start: prefix `MICRODUCK_WARM_START=1`, as 18a did from run 15. Both tasks share
the 327-D obs, so their normalizers are compatible. A 61-D checkpoint needs
`MICRODUCK_WARM_START_EXPAND_OBS=1` (§5).

## Open problems

- **No clean descent inside one policy.** The switch is required, and distillation lost both
  skills (§29).
- **The crouch** was never learned by reward in the walking policy (§23–§25); the down policy's is
  fixed at 104–108 mm, not riser-scaled (§28). Going down still touches 0.55–0.59 edges per riser,
  mostly a foot tipped over the nose (§28, §29).
- **Stiff terrain:** the final policies were run on softened contacts only (solref 0.04).
- **Hardware and runtime:** a 327-D policy kind, the height map built in `robotd` from the ToF and
  leg odometry, a `publish` perception kind (§7, §15), and a check of the sim ToF model against real
  frames in front of a real step (§14, §15).
- **Stale ToF map at turns.** The map is a running mean since the episode start, and on the way
  back the robot re-reads cells written under a different odometry error. In run 12's round trip,
  clearing the map at the turn cut falls going down from 28 % to 0 % at 3 cm (§19). The runtime map
  has to forget: clear it at a turn-around or a skill switch, or age cells out.
- **Heading:** the batteries run with no heading hold, and 6–12 % of the down policy's robots leave
  the 2 m wide staircase sideways (§28). Real use needs a heading hold.
- **Choice of climber:** 13b (anchored to 11b) climbed long stairs better than its siblings (§20). It
  was never compared with run 15 on the leg-collision model or in the switch.
