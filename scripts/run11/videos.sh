#!/usr/bin/env bash
# Run 11 videos on the GPU box: record 10 s rollouts of the three checkpoints (64 robots, 0.25 m/s),
# render one showcase per run and two side-by-side comparisons against run 10.
#   tmux new -d -s vid11 "bash ~/microduck_rl_walk_terrain/scripts/run11/videos.sh"
set -o pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
LOG=vid11.log; : > "$LOG"
S=logs/rsl_rl/step_up
CK10=$S/2026-11-03_20-30-50_step_up_tofmap_clean/model_3999.pt
CK11A=$S/2026-11-03_23-43-02_step_up_tofmap_envelope/model_3999.pt
CK11C=$S/2026-11-04_01-41-38_step_up_tofmap_hard/model_3999.pt
CK11B=$S/2026-11-04_03-41-40_step_up_tofmap_flights/model_3999.pt
BASE=Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck
ENV=Mjlab-StepUp-TofMap-Envelope-Backlash-MicroDuck
HARD=Mjlab-StepUp-TofMap-Hard-Backlash-MicroDuck
R="uv run python scripts/render_showcase.py"
step() { echo "[$(date '+%T')] $*" | tee -a "$LOG"; }
step "record 11a (envelope grid)"; $R record --task $ENV --checkpoint $CK11A --device cuda --num-envs 64 --seconds 10 --out cmp_11a.npz >> $LOG 2>&1 || exit 2
step "record 11c (base grid)";     $R record --task $BASE --checkpoint $CK11C --device cuda --num-envs 64 --seconds 10 --out cmp_11c.npz >> $LOG 2>&1 || exit 2
step "record 11c (hard sensor)";   $R record --task $HARD --checkpoint $CK11C --device cuda --num-envs 64 --seconds 10 --out cmp_11c_hard.npz >> $LOG 2>&1 || exit 2
step "record 10 (hard sensor)";    $R record --task $HARD --checkpoint $CK10  --device cuda --num-envs 64 --seconds 10 --out cmp_10_hard.npz >> $LOG 2>&1 || exit 2
step "record 11b (base grid)";     $R record --task $BASE --checkpoint $CK11B --device cuda --num-envs 64 --seconds 10 --out cmp_11b.npz >> $LOG 2>&1 || exit 2
step "render showcase_11a"; $R render --states cmp_11a.npz --video showcase_11a.mp4 >> $LOG 2>&1 || exit 3
step "render showcase_11c"; $R render --states cmp_11c.npz --video showcase_11c.mp4 >> $LOG 2>&1 || exit 3
step "render showcase_11b"; $R render --states cmp_11b.npz --video showcase_11b.mp4 >> $LOG 2>&1 || exit 3
step "render compare_10_11b"
$R compare --left cmp_10.npz --right cmp_11b.npz --segments "short_flight:3,short_flight:2,single_step:3" \
  --left-label "run 10" --right-label "run 11b: random tread depth + foot placement" \
  --title "Three-riser flights: run 10 vs run 11b" \
  --summary "Run 11b: flights trained on 11-24 cm treads, +1 per well-placed foot" "384 robots, 12 s, 0.25 m/s, same seed" \
            "clean climbs 283 / 345 vs 228 / 345" "falls 3 vs 6" "flights 3.3 / 3.6 cm: 81 / 75 % clean vs 67 / 57 %" \
  --video compare_10_11b.mp4 >> $LOG 2>&1 || exit 4
step "render compare_10_11c_hard"
$R compare --left cmp_10_hard.npz --right cmp_11c_hard.npz --segments "single_step:3,short_flight:3,single_step:0" \
  --left-label "run 10 under the harder sensor" --right-label "run 11c: trained under the harder sensor" \
  --title "Worse ToF (cone blur, 2x noise, 2x drift): run 10 vs run 11c" \
  --summary "Both policies read a map built from a worse sensor model" "384 robots, 12 s, 0.25 m/s, same seed" \
            "clean climbs 238 / 345 vs 198 / 345" "falls 6 vs 10" "on the nominal sensor run 11c: 266 / 345 clean, 0 falls" \
  --video compare_10_11c_hard.mp4 >> $LOG 2>&1 || exit 4
step "VID11_OK"
