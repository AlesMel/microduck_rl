#!/usr/bin/env bash
# Where does clean climbing end? Run 11a (trained up to 6.5 cm) and 11b (up to 4 cm) on single
# steps 4.5-7.5 cm and 7-10.5 cm, 384 robots, 12 s, 0.25 m/s. Output: limit11.log + failures_limit_*.json
set -u
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
S=logs/rsl_rl/step_up
declare -A CK=([11a]=$S/2026-11-03_23-43-02_step_up_tofmap_envelope/model_3999.pt [11b]=$S/2026-11-04_03-41-40_step_up_tofmap_flights/model_3999.pt)
: > limit11.log
for tag in 11a 11b; do
  for span in "0.045 0.075" "0.07 0.105"; do
    name="${tag}_$(echo $span | tr ' .' '_p')"
    echo "=== $tag  risers $span" | tee -a limit11.log
    uv run python scripts/eval/step_up_failures.py --checkpoint "${CK[$tag]}" --device cuda --num-envs 384 --seconds 12 --vx 0.25 \
      --task Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck --riser-range $span --json "failures_limit_$name.json" 2>&1 \
      | grep -E "^\s+(single_step|ALL|CLEAN)" | tee -a limit11.log
  done
done
echo "=== LIMIT_DONE" | tee -a limit11.log
