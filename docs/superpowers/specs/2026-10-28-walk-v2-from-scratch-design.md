# Walk v2: a fast, straight gait trained from scratch — design

Status: RUN 1 DONE 2026-10-29 (wandb `mg26yora`, pick `model_5000`). All gates run; every
success criterion met except open-loop drift, which sits on the 0.05 rad/s line and changes
sign between checkpoints. Results in the docstring of
`src/mjlab_microduck/tasks/microduck_walk_v2_env_cfg.py`.

## Why from scratch

VelStandFast reached about 0.35 m/s at a 0.4 command, but over nine resumed fine-tunes of v8
(stages 1-7c, `microduck_velstand_fast_env_cfg.py` on `ales/velstand-fast`), each trading one defect for another:

1. **Stage 1 never resumed a real training state.** v8 was rebuilt from the Hub ONNX (no
   wandb access): action std hard-set to 0.2, critic and optimizer from a 5-iteration smoke
   run. PPO, which only searches near the current policy (KL ≈ 0.01 per update), then
   fine-tuned a low-noise policy against an untrained value function. Stage 2 showed the
   result: the reward was fixed (faster scored higher) and the 0.4 behaviour did not move.
2. **The speed came from behaviour cloning.** Stage 3 cloned a frozen stage-2 policy at
   coefficient 1.0. It turns right by itself at speed, and that bias was cloned in.
3. **Symmetry came too late.** The mirror loss was off while v8's gait formed; added at
   stage 7 it averaged two lopsided gaits into a slower one.

The slow gait's root cause is known and its fix validated by counterfactual reward accounting:
v8's air_time window paid a 2 Hz gait 1.50/step, more than speed tracking (0.94 at 0.4).

## Deployment facts that set the targets

- **No heading-hold loop on the robot.** robotd sends the stick's wz straight to the policy.
  The stage 6/7c "with heading loop" numbers do not describe the robot. Straightness is judged
  open loop.
- **Action pipeline mismatch.** robotd's defaults are action scale 0.9 × 7.4/V (voltage
  adaptation), target low-pass 0.7 legs / 0.5 head. Every env in this repo trains scale 1.0,
  unfiltered, with battery voltage simulated by BAM (`vin_range` 6.5-8.2 V). Decision: the
  robot matches training — deploy with `action_scale = 1.0`, `legs_lowpass = 1.0`,
  `head_lowpass = 1.0`, `voltage_adapt = false`. Measured first in sim (Phase 0).
- **Physical ceiling.** Leg-model study: ≈ 0.5 m/s grounded run at 7.4 V (0.47 at 6.5 V, 0.58
  at 8.2 V). v8 uses ≤ 40 % of servo speed at 0.4; stage 3 reached 0.39. 0.4 is feasible,
  0.5 is the edge.

## Principle

Change only what the diagnosis points at: the proven velocity recipe, plus only the
VelStandFast reward fixes that accounting validated, no behaviour cloning, the mirror loss from
iteration 0. No new speculative terms in run 1 (no energy term, no gait clock). Every
expensive step has a cheap go/no-go gate before it.

## Scope

Walk only. Fall recovery comes afterwards as VelStand v2 (warm start from walk v2, walk v2 as
the walk anchor, stand-expert BC on fallen frames, the route v1 took). Until then walk v2 does
not replace velstand on the robot, whose walk slot also stands and gets up.

## Observations

**Actor: the 61D family layout, unchanged** — hot-swappable, no runtime work.

**Critic (not deployed):** the velocity recipe's privileged terms plus EMA vx and EMA yaw rate
(the quantities the EMA rewards score), battery voltage and BAM friction scale, trunk CoM
offset, foot friction, heading error.

Held in reserve: a gait clock (plan B, needs a robot-side command encoding) and an LSTM actor
(the runtime already accepts the rsl_rl explicit-state export, `model_api: 2`).

## Reward

| term | vs velocity recipe | source |
| --- | --- | --- |
| `air_time` → `feet_air_time_speed_window` | window [0.125, 0.30] s at ≤ 0.1 m/s → [0.09, 0.15] s at ≥ 0.4, w 3 | VSF stage 1 |
| `track_vx_ema` | new: 0.5 s EMA of body vx, std 0.15, w 4 | VSF stage 2 |
| `track_angular_velocity` | scored on the 0.5 s EMA of yaw rate, std √0.5, w 2 | VSF stage 2 |
| `pose` | running tier at 0.3 (recipe's 1.5 was unreachable) | VSF stage 2 |
| `low_swing` | new: one-sided cost for apex < 10 mm, w −60, ramped by vx 0.15→0.30 | VSF stage 4 |
| everything else | unchanged | velocity recipe |

Locked by `tests/test_walk_v2_cfg.py`: walking at half the commanded speed (0.25-0.5 m/s)
forfeits more tracking pay than the air-time term can pay at all in a walk (w × (1 − lo/hi));
standing still at 0.1 forfeits > 1.5/step.

## Commands

- vx from ±0.3, widened by 0.05 up to ±0.5 only when the mean signed shortfall at the edge is
  < 0.08 m/s (`vx_range_performance_curriculum`, every 50 iterations, never narrows).
- vy ±0.3, wz ±1.0, turn-in-place 15 %, standing-env curriculum: as the recipe.
- 30 % heading envs (stiffness 2.0, targets ±0.5 rad around the current heading).

## PPO

The recipe's runner (init std 1.0, entropy 0.01, adaptive LR, KL 0.01), mirror loss 0.5 from
iteration 0, no BC. 6000 iterations at 2048 envs on urkui-3 (≈ 5 h).

## Measurement built in

`walk_v2_gait_stats` logs `Curriculum/gait_stats/*`: speed ratio and stride frequency per
|vx command| bucket (0.1-0.5), signed yaw bias, L/R air-time asymmetry; `Curriculum/vx_range/*`
logs the current vx max and edge shortfall. (These ride the curriculum channel because mjlab
replaces `extras["log"]` on every step that resets an env.) Final checks use `scripts/eval/`.

## Gates

1. **Phase 0, parity A/B (sim):** v8 and 7c `model_7625` under the training pipeline vs the
   robot's, 32 robots, 0.2/0.3/0.4 m/s.
2. **Phase 2, reward audit:** existing gaits scored under the new reward with
   `reward_acct.py` — at a true 0.4 reward rises with achieved vx (0.18 → 0.39, ≥ +1.0/step
   from 0.25 to 0.39); v8 scores below stage 3; at 0.1 stepping beats standing; mirrored gait
   within 2 %; penalties ≤ 0.
3. **Phase 3, pilot:** smoke test, then 400 iterations — std ≥ 0.4, air_time > 0, track_vx_ema
   rising, penalties ≤ 0, stats populated.
4. **Phase 4, checkpoints 2000/4000:** stop if stride frequency is flat across buckets or vx
   max is stuck at 0.3 (→ plan B).

## Success criteria

Rough-Backlash, 32-64 robots, 20 s, open loop: vx ≥ 0.35 at a 0.4 command; a 0.1 command
walks; stride frequency rises with the command; |drift| ≤ 0.05 rad/s at 0.4; L/R asymmetry
≤ 10 %; falls ≤ 2 % flat, ≤ 4 % rough.
