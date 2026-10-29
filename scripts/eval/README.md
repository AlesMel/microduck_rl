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
