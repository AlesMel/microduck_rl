#!/usr/bin/env bash
# Recordings for scripts/marketing/step_up_board.json (render with scripts/render_marketing.py):
# the original shipped walking policy (Hub pollen-robotics/microduck-policies: alpha_walking.onnx,
# blind, 61-D) and step-up run 11a (ToF-built height map) on the SAME single-step spans, grid and
# seed, so robot i has the same tile, row and start in both. 50 states/s for slow motion.
#
#   R11A=<path to run 11a model_3999.pt> scripts/marketing/step_up_record.sh
#   cp scripts/marketing/step_up_board.json mk/board.json
#   MUJOCO_GL=egl uv run python scripts/render_marketing.py pair --left mk/alpha_walking_hi.npz --right mk/now_hi.npz
#   MUJOCO_GL=egl uv run python scripts/render_marketing.py cut --board mk/board.json --video mk/microduck_steps.mp4
#
# Run 11a = wandb v1zgwli4, logs/rsl_rl/step_up/2026-11-03_23-43-02_step_up_tofmap_envelope/model_3999.pt.
set -eu
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl
: "${R11A:?set R11A to run 11a's model_3999.pt}"
MAP=Mjlab-StepUp-TofMap-Clean-Backlash-MicroDuck
mkdir -p mk/pollen
uv run python -c "from huggingface_hub import hf_hub_download as d; d('pollen-robotics/microduck-policies', 'alpha_walking.onnx', local_dir='mk/pollen')"
rec() { uv run python scripts/render_showcase.py record --device cuda --fps 50 --seconds 12 --num-envs 128 --task $MAP "$@"; }
for span in "0.02 0.05:lo" "0.055 0.08:hi"; do
  rr=${span%%:*}; tag=${span##*:}
  rec --checkpoint mk/pollen/alpha_walking.onnx --riser-range $rr --out mk/alpha_walking_$tag.npz
  rec --checkpoint "$R11A" --riser-range $rr --out mk/now_$tag.npz
done
