# Gait evaluation scripts

Headless rollouts of a checkpoint (`model_N.pt`) in a task's play cfg, no pushes, driving the
policy through its observation (twist slots set, head/body slots zeroed). CPU works for
small batches; use `--device cuda:0` on the training box. Any 61D family checkpoint runs in any
walk task (v8, VelStandFast stages, Walk v2).

`--robot-chain` (calib_vx, battery walk, gait_asym) replaces the training action path
(target = HOME + 1.0 · action) with the real robot's defaults: scale 0.9 × 7.4 / V with each
env's sampled battery voltage, then a target low-pass of 0.7 on the legs and 0.5 on the head;
the policy still sees its raw last action, as on the robot. See `_common.RobotChain`.

The default task is Walk v2 (`Mjlab-WalkV2-*`).

## calib_vx.py — speed response and drift

```bash
uv run scripts/eval/calib_vx.py MODEL.pt out.json [--fed 0.1,0.2,0.3,0.4,0.5] [--per 16] [--robot-chain]
```

Per fed forward command: upright robots, achieved mean body vx (± across robots), open-loop
yaw drift (signed mean and mean |·|). JSON: `rows[]`.

## gait_asym.py — where drift comes from

```bash
uv run scripts/eval/gait_asym.py MODEL.pt out.json [--vx 0.4] [--n 64] [--robot-chain]
```

Open loop at one forward command: per robot drift; trunk yaw accumulated in each support
phase; per foot (right mirrored) stance/swing time, step length, width, toe-out, pivot, push
impulse and joint means; L−R and its correlation with drift and with the randomized trunk CoM
and hip-yaw encoder bias. JSON: `summary`, `rows[]`.

## battery.py — falls and get-up

```bash
uv run scripts/eval/battery.py A.pt B.pt --mode walk [--family WalkV2] [--terrain rough] [--cmds 0,0.2,0.3,0.4] [--robot-chain]
uv run scripts/eval/battery.py A.pt --mode getup [--getup-task Mjlab-VelStand-Flat-Backlash-MicroDuck]
```

walk: per command and checkpoint, mean vx of robots that never fell, % ever fell (tilt > 40°),
% upright at the end. getup: per spawn family (7), % recovered and held, % stuck, median
time to recover, median angular speed while fallen; needs a velstand-family task (it uses
`random_prone_init`). Prints a table.

## reward_acct.py — score a gait under a task's reward

```bash
uv run scripts/eval/reward_acct.py MODEL.pt out.json [--true 0.4] [--fed 0.4,0.5,0.6] [--per 8] [--mirror]
uv run scripts/eval/reward_acct.py --zero-action out.json --true 0.1
```

Feeds the policy `--fed` but scores every reward term at `--true` (the counterfactual: what
would this gait earn for this command). Weighted per-step value of each term, averaged over
upright robots after settling, with train-cfg weights (reward_weight curricula at their final
stage), plus achieved vx, swing time, stride frequency, and whether every penalty is ≤ 0.
`--zero-action` scores standing still; `--mirror` scores the policy's left-right mirror image.
This is the Walk v2 reward audit (Phase 2 of the design spec).

## render_ckpt.py — a video of one robot

```bash
uv run scripts/eval/render_ckpt.py MODEL.pt out.mp4 [--task Mjlab-WalkV2-Rough-Backlash-MicroDuck] [--label TEXT] [--robot-chain]
```

50 fps, camera following the trunk, through a fixed schedule: stand, walk 0.1-0.5 m/s,
stop, backward 0.3, sideways 0.2, turn 1.0 rad/s, stand. The overlay shows the command and
the measured forward speed; a per-segment summary is printed at the end.

## Stairs (`docs/stairs.md`; history in `docs/archive/stairs_runlog.md`)

The stairs line keeps three policies: stairs up (`Mjlab-StairsUp[-Backlash]-MicroDuck`, run 15),
stairs down (`Mjlab-StairsDown[-Backlash]-MicroDuck`, run 18d) and the switch of the two
(`tasks/stairs_switch.py`: the down policy while the ToF map shows a drop ahead, the up policy
otherwise). The default task of these scripts is `Mjlab-StairsUp-Backlash-MicroDuck`; all take
`--checkpoint` / `--task` / `--device`, and `--checkpoint switch:<down.pt>+<up.pt>` runs the switch
(`_common.load_policy`).

- **stairs_battery.py** — how many steps in a row: one long straight staircase per riser height
  (`--heights` cm, `--steps 20`, `--tread`), robots walked at a constant `--vx`; per robot the
  risers climbed, the clean run (risers before the first toe catch / edge landing, same
  definitions as `step_up_failures.py`), and where it fell; Kaplan-Meier summaries per height.
  `--down` turns the staircase around (spawn on top, walk down; clean = no edge contact, the
  training cost's definition). The ToF map is extended to cover the tile (`--map-x`). The runlog's
  batteries: 20 risers, `--tread 0.20 --seconds 45 --head down`. §17-19, §28-29.
- **stairs_contacts.py** — what the descent's edge contacts are (the `--down` staircase).
  - Each contact is classified as landing vs stance (the foot touched down across the edge, or
    stood clean and then went over the nose) and as leading vs trailing foot.
  - Per edge crossing by one foot: lift-off distance behind the edge, landing distance beyond it,
    stride, and the touchdown impact (peak foot force, trunk |a_z|, v_z).
  - The loaded foot's pitch on flat ground, near an edge, and in an edge contact.
  - `--hard-terrain` undoes the training's softened terrain contacts; `--vx` runs it at other speeds.

  §20 and on.
- **step_up_failures.py** — the clean-climb test on the play grid: per riser height, WHY a robot
  stumbles (toe catch, edge landing, blind gap, slowdown, fall context); `--riser-range LO HI`
  puts other single-step heights on the grid (where clean climbing ends). JSON: `rows[]` per robot.
- **step_down_static.py** — static test: can the rear leg hold all the weight while the front foot is
  placed flat on the next tread, step down vs step up (reach, torques, mesh hits on the step; `--video`).
  `--keep-poses` stores every valid pose; `scripts/make_step_down_bank.py` and
  `scripts/make_edge_crouch_bank.py` turn that into the spawn banks in `tasks/data/`
  (`step_down_poses.json`, `edge_stand_poses.json`, `edge_crouch_poses.json`). §22, §26.
- **edge_spawn_check.py** — the down-edge spawns before training on them: sole gaps and CoM at
  spawn, first-step trunk speed, leg-geom contacts, falls within `--seconds` under a checkpoint,
  zero actions, or `--hold` (hold the spawn pose; prints the servo sag). `--event` picks the spawn
  (`reset_at_down_edge`, or stairs down's `reset_edge_stand` / `reset_edge_crouch` with
  `--task Mjlab-StairsDown-Backlash-MicroDuck`). §23, §26.
- **stairs_feasibility.py** — quasi-static torque budget of a step-up: what the knee needs per
  riser height vs the XL330 cap, from a blind walk-v2 stairs checkpoint (`Mjlab-WalkV2-Stairs-*`).
  The "why perception" number.

Related, one level up: `scripts/view_terrain_env.py` (interactive viewer / headless video of a
policy on the terrain grid, and the `build_env` helper the batteries use),
`scripts/render_showcase.py` (showcase and side-by-side comparison videos with the policy's
perception drawn in; `render_marketing.py` reuses its renderer), and
`scripts/render_staircase.py` (the `stairs_battery.py` test filmed: one typical hero per
staircase — clean climb, climb with slips, fall — with the battery's own catch / edge-landing
counts in the HUD).
