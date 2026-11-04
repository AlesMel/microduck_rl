#!/usr/bin/env bash
# Run 11 overnight chain on the GPU box (12 GB → one 2048-env run at a time), in this order:
#   11a Envelope  (too-high column + stop objective)       → battery on the envelope grid
#   11c Hard      (map under the harder sensor model)      → battery on the base grid + under the hard sensor
#   11b Flights   (tread DR + foot placement)              → battery on the base grid
# All warm-started from run 10's model_3999 (same 327-D obs; MICRODUCK_WARM_START=1 restarts the
# ladder). Each phase is logged to chain11.log and summarised in chain11_summary.txt; a failed
# phase does not stop the chain.
#   tmux new -d -s chain11 "bash ~/microduck_rl_walk_terrain/scripts/run11/chain.sh"
set -u
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
LOG=chain11.log; SUM=chain11_summary.txt
RUN10=2026-11-03_20-30-50_step_up_tofmap_clean
NUM_ENVS=${NUM_ENVS:-2048}; ITERS=${ITERS:-4000}
BASE_TASK=Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck
HARD_TASK=Mjlab-StepUp-TofMap-Hard-Backlash-MicroDuck
: > "$LOG"; : > "$SUM"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" "$SUM"; }

battery() {   # battery <ckpt> <task> <tag>
  local ckpt=$1 task=$2 tag=$3
  log "battery $tag: $task ← $ckpt"
  uv run python scripts/eval/step_up_failures.py --checkpoint "$ckpt" --device cuda --num-envs 384 \
      --seconds 12 --vx 0.25 --task "$task" --json "failures_$tag.json" > "battery_$tag.log" 2>&1 \
    && { grep -E "^\s+(single_step|short_flight|high_step|ALL|CLEAN)" "battery_$tag.log" | tee -a "$SUM"; } \
    || log "battery $tag FAILED (battery_$tag.log)"
}

train_one() {   # train_one <task> <run_name> <tag>
  local task=$1 run_name=$2 tag=$3
  log "train $tag: $task ($NUM_ENVS envs × $ITERS it, warm start $RUN10/model_3999)"
  local before; before=$(ls -d logs/rsl_rl/step_up/*_"$run_name" 2>/dev/null | tail -1)
  MICRODUCK_WARM_START=1 uv run train "$task" --env.scene.num-envs "$NUM_ENVS" --agent.max_iterations "$ITERS" \
      --agent.resume True --agent.load-run "$RUN10" --agent.load-checkpoint model_3999.pt > "train_$tag.log" 2>&1
  local rc=$?
  local run; run=$(ls -d logs/rsl_rl/step_up/*_"$run_name" 2>/dev/null | tail -1)
  local wb; wb=$(grep -o -E "wandb.ai/[^ ]+/runs/[A-Za-z0-9]+" "train_$tag.log" | head -1)
  if [ $rc -ne 0 ] || [ -z "$run" ] || [ "$run" = "$before" ]; then
    log "train $tag FAILED rc=$rc (train_$tag.log)"; return 1
  fi
  log "train $tag done: $run  $wb"
  CKPT=$(ls "$run"/model_*.pt | sort -V | tail -1)
  log "train $tag checkpoint: $CKPT"
}

# control: run 10 itself under the harder sensor (how much the sensor model alone costs)
battery "logs/rsl_rl/step_up/$RUN10/model_3999.pt" "$HARD_TASK" 10_hardsensor

# 11a
if train_one Mjlab-StepUp-TofMap-Envelope-Backlash-MicroDuck step_up_tofmap_envelope 11a; then
  battery "$CKPT" Mjlab-StepUp-TofMap-Envelope-Backlash-MicroDuck 11a
fi
# 11c
if train_one "$HARD_TASK" step_up_tofmap_hard 11c; then
  battery "$CKPT" "$BASE_TASK" 11c
  battery "$CKPT" "$HARD_TASK" 11c_hardsensor
fi
# 11b
if train_one Mjlab-StepUp-TofMap-Flights-Backlash-MicroDuck step_up_tofmap_flights 11b; then
  battery "$CKPT" "$BASE_TASK" 11b
fi
log "CHAIN DONE"
