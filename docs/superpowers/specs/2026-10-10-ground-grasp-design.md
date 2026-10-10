# Ground grasp — design (2026-10-10)

**Branch** `ground-grasp` (off `origin/develop` ceca017, local only) · **Tasks** `Mjlab-GroundGrasp-*-MicroDuck` ·
**Brief** `/Users/remi/microduck/notes/ground-grasp/BRIEF.md` · **Notes / evals** `/Users/remi/microduck/notes/ground-grasp/`

## 1. Goal (Rémi, 2026-10-10)

The ground-pick move is beautiful but rarely grasps: the beak does not get low enough, it is not parallel to the
ground, and the final approach is not vertical (a frontal or sideways approach pushes the object away). Train a
better grasp move, ideally starting from today's move, and make the training environment actually grasp small home
objects (Lego, batteries, dice, small toys, socks). Assume the beak has enough friction and torque; the hard part is
collision and the movement. Report whether the heavy head makes it dynamically impossible.

## 2. What exists today (measured, not assumed)

**The deployed move** (`microduck/policies/alpha_ground_pick.onnx`, robotd slot `ground_pick`, A button): robotd feeds
`twist = [cos 2πφ, sin 2πφ, 0]`, φ from 0 to **0.7** over 0.7 × 4 s = 2.8 s, head/body slots zero, then hands back to
the stand net. Low-pass head 0.5 / legs 0.7 on every net. Evaluated in CPU MuJoCo on the exact training model (BAM
servos, actuator delay, robotd timing; `notes/ground-grasp/tools/gg_eval.py`):

| | value |
|---|---|
| beak pitch at the lowest point | **86°** (pointing straight down) |
| mouth tip height at the lowest point | **2.7 cm** |
| lowest head point | 1.8 cm (the forehead dome: it is lower than the lips when the beak points down) |
| final approach | 2.0 cm horizontal for the last 3.0 cm of descent (ratio 0.67) |
| lowest point reached at | φ = 0.35 (1.4 s) |
| robotd low-pass on/off | no difference |

**The beak (CAD).** The "beak" is the whole underside of the head. The lower jaw is a 3.5 × 8.8 cm plate on the front
half of the head's underside with two side arms going back to a hinge at the back of the head (mouth servo on one side,
bearing on the other, same axis): direct drive, robotd `MOUTH_CLOSED = -5°` (squeeze), `MOUTH_OPEN = +30°`. Soft pads
on the front 3.3 cm of both jaws. Lip gap at full opening ≈ 3.5 cm; mouth cavity ≈ 3.5 cm deep. The opening faces
forward, so with the beak level an object can only enter from the front; with the beak pitched down the wedge opens
downward and a vertical approach puts the two lips on both sides of the object.

**The training model** has no jaw joint (merged into the head) and MuJoCo collides the CAD meshes by convex hull: the
jaw hull and the bottom-shell hull fill the mouth cavity. Nothing could ever enter the mouth in the old env.

**Old env quirks found on the way** (not fixed there, noted for Rémi): `mouth_payload_force` is a weight-0 reward used
as a per-step hook, but mjlab's reward manager skips weight-0 terms, so the payload force has never been applied;
`RISE_END = 0.80` > robotd's 0.7 cutoff (the rise is handed to the stand net unfinished).

## 3. Static feasibility (before any training)

`notes/ground-grasp/tools/pose_study.py`: symmetric two-foot crouch (hip pitch, knee, ankle, neck pitch, head pitch),
soles flat, mouth open 26°, lowest beak point at 3 mm. For each target beak pitch: centre-of-mass margin inside the
feet, holding torques (gravity, both feet sharing the weight), self-collisions. Results in §3.1 (filled from the run).

### 3.1 Results

(see `notes/ground-grasp/poses/study_theta045.txt`)

## 4. Environment

### 4.1 Model

`get_ground_grasp_spec()` = develop's all-collisions robot (`name_servo_collision_geoms`) + the trunk/hip hull-phantom
excludes of flamingo v2 + `add_mouth()`:

- body `jaw` on hinge **`passive_mouth`** at the real pivot (head frame u = 0.0037, f = 0.0182), range −5°…31.5°.
  Named `passive_*` so every 14-servo selector, the BAM actuator regex, the obs terms and the ONNX export skip it
  (AGENTS.md convention; `_servo_joint_ids` already lists "the jaw linkage" as a passive joint).
- its own actuator: `BuiltinPositionActuatorCfg(stiffness 0.55 N·m/rad (XL330 at firmware kP 200), effort 0.5 N·m)`.
- the four underside hulls (bottom shell, upper pad, jaw, lower pad) replaced by boxes: `head_rear_collision`,
  `upper_beak_collision`, `jaw_plate_collision` (the latter on the jaw). Named `*_collision` so FULL_COLLISION keeps
  them; their contacts with the object get friction from the object's condim/friction (max rule).
- sites `lip_up`, `lip_low`, `jaw_tip_bottom`, `pad_up`, `pad_low` (grasp point = midpoint of the two pads).

### 4.2 Mouth (not a policy output — the 14-action contract stays)

A zero-width action term (`MouthScheduleAction`, action_dim 0) writes the jaw target every physics step from the phase
command: closed (−5°) → open (`MOUTH_OPEN_RAD`) over φ ∈ [0.02, 0.10] → closing ramp over φ ∈ [φ_close, φ_close+0.06]
→ closed (squeezing whatever is between the pads). Per-episode DR: open angle ±10 %, close phase ±0.01. On the real
robot this is a mouth table played with the pick (padd's `pick` cue already does "ground pick + beak open"), so the
timing in seconds is part of the deliverable.

### 4.3 Objects

A free-body object entity with one primitive geom, placed every reset at a nominal point in the robot's yaw frame
`OBJ_NOMINAL_XY` (from §3: where the beak reaches while balanced) plus uniform noise (±1 cm → ±2 cm by curriculum;
the policy is BLIND to the object, like the ball kick: the operator places the object, robustness comes from the
placement DR). Size DR with `dr.geom_size` (box half-extents 0.7–1.3 cm, height 0.6–1.2 cm: Lego/dice scale), mass
5–25 g, friction 1.2–1.6, condim 4 (soft TPU pads: Rémi says assume enough friction). Random yaw. Variant later:
lying cylinder (AA battery), capsule.

### 4.4 Timeline (period 4 s, episode = one cycle, φ starts at 0 from a noisy stand)

| φ | s | what |
|---|---|---|
| 0.00–0.05 | 0–0.2 | standing, mouth starts opening |
| 0.05–0.30 | 0.2–1.2 | approach: beak over the object, then down |
| 0.30–0.36 | 1.2–1.44 | mouth closes (φ_close = 0.30) |
| 0.36–0.62 | 1.44–2.5 | lift and stand up holding the object |
| 0.62–1.00 | 2.5–4.0 | stand holding (robotd hands over to the stand net at 0.7) |

### 4.5 Rewards

Main task (positive):
- `grasp_held` — per step for φ ≥ 0.40: object touching BOTH jaw pads, not touching the floor, lifted ≥ 1 cm. Plus a
  lift term ∝ clamp(Δz / 4 cm, 0, 1) under the same gates.
- `approach_track` — φ ∈ [0.05, 0.36]: Gaussian (σ 3 cm) of the distance between the grasp point and a SLEWED target
  that sits above the object from the start (xy = object) and descends linearly from the standing height to the
  object centre by φ = 0.29. Being ahead of the ramp pays nothing (AGENTS: no jackpots), and since the target xy never
  leaves the object, a vertical final approach is the argmax.

Physical consequences priced directly (negative):
- `object_pushed` — before the mouth closes: horizontal displacement of the object from its spawn point (L1, capped).
  This is exactly "a frontal or sideways approach pushes the object away".
- `head_floor_impact` — contact force of the head boxes/hulls with the floor above 1 N (the jaw plate may touch the
  floor lightly: threshold 3 N).
Kept from ground pick (stability / sim2real): feet_grounded, feet_flat, upright (low), return pose legs/neck and
return_upright gated on φ ≥ 0.45, action_rate (curriculum), neck action rate, joint torques, self collisions, fall
termination (tilt > 70°).

Optional shaping (variant knob): `beak_pitch_pref` — Gaussian on beak pitch around a target during the grasp window,
to push "more parallel" if the task alone converges to a straight-down beak.

### 4.6 Observations

Actor: the unchanged 61-D contract (blind to object and mouth). Critic (asymmetric): + object position/velocity in the
base frame, object size, mouth angle, contact flags.

## 5. Training plan

- **Start points**: (A) from scratch; (B) from today's move — `GroundGraspFromPickRunner` copies the actor (and its
  obs normalizer) from `alpha_ground_pick.onnx` into the new runner, fresh critic (the critic obs differ), init
  std 0.3 so the move is not drowned in exploration noise; curricula restart at 0.
- Smoke: local CPU (16 envs × 2 it), then HF t4-small 64 × 5 (≈ $0.05).
- Runs: 4096 envs × 1500 it on rtx-pro-6000 ≈ 35 min ≈ $1.6 each. One job at a time.
- **Budget: $100 hard cap for the whole experiment.** Ledger: `notes/ground-grasp/LEDGER.md` (flavor × minutes per job).

## 6. Success metric

CPU eval with the robotd sequence (stand → pick φ 0…0.7 → stand net 2 s), objects at nominal ± offsets grid:
**grasp success** = object off the floor between the pads 1.5 s after the hand-over, no fall. Also beak pitch and lip
height at the grasp, approach ratio, object push before closing, holding torques. Videos (side close-up) of the best.

## 7. Plan B

- If no balanced pose reaches the floor at a flatter pitch: keep the down-facing wedge (pitch ≈ 70–80°) and make it
  lower (the dome-first problem) — a small head roll or a slightly open mouth before contact.
- If the blind grasp is too position-sensitive: put the object offset in the (unused) head-command slots; zeros at
  deployment = the nominal spot, vision can fill them later.
- If the stand net drops the object at the hand-over: train the hold to φ = 1 and ask robotd to run the pick to 1.0
  for this policy (`GROUND_PICK_END_PHASE` is a constant today).
