#!/usr/bin/env bash
# Run 11 smoke tests on the GPU box: each task builds, steps NaN-free and trains 5 iterations at
# 64 envs; the last one also exercises the warm start from run 10 (same 327-D obs, Patch 5).
#   tmux new -d -s smoke11 "bash ~/microduck_rl_walk_terrain/scripts/run11/smoke.sh"
set -u
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
LOG=smoke11.log
: > "$LOG"
RUN10=2026-11-03_20-30-50_step_up_tofmap_clean
ok=0; fail=0
for task in Mjlab-StepUp-TofMap-Envelope-Backlash-MicroDuck Mjlab-StepUp-TofMap-Hard-Backlash-MicroDuck Mjlab-StepUp-TofMap-Flights-Backlash-MicroDuck; do
  echo "=== $(date +%H:%M:%S) smoke $task" | tee -a "$LOG"
  if uv run train "$task" --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard >> "$LOG" 2>&1; then
    echo "--- OK $task" | tee -a "$LOG"; ok=$((ok+1))
  else
    echo "--- FAIL $task (see $LOG)" | tee -a "$LOG"; fail=$((fail+1))
  fi
done
echo "=== $(date +%H:%M:%S) smoke warm start from run 10 (Envelope)" | tee -a "$LOG"
if MICRODUCK_WARM_START=1 uv run train Mjlab-StepUp-TofMap-Envelope-Backlash-MicroDuck \
     --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard \
     --agent.resume True --agent.load-run "$RUN10" --agent.load-checkpoint model_3999.pt >> "$LOG" 2>&1; then
  echo "--- OK warm start" | tee -a "$LOG"; ok=$((ok+1))
else
  echo "--- FAIL warm start" | tee -a "$LOG"; fail=$((fail+1))
fi
echo "=== SMOKE DONE ok=$ok fail=$fail" | tee -a "$LOG"
