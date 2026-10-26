"""Export a Soccer-1v1 checkpoint to ONNX for `soccer_referee.py`.

    uv run scripts/soccer_export.py logs/rsl_rl/soccer_1v1/<run>/model_3000.pt [--out soccer.onnx]

Not `scripts/export.py`: that one stamps 61D-family metadata (joint names, gains, action scale)
read from a `robot` entity and a `joint_pos` action, and this policy has neither — its input is
26D match state and its output a twist. Same core, though: the runner's `export_policy_to_onnx`
emits `actor(normalizer(obs))`, so the normalizer is baked in as the house rule requires.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import onnx

import mjlab_microduck.tasks  # noqa: F401  (registers the task)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_microduck.soccer.frozen_walker import TWIST_LOWER, TWIST_UPPER
from mjlab_microduck.soccer.mdp import OBS_DIM

TASK = "Mjlab-Soccer-1v1-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Soccer-1v1 checkpoint to ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <run dir>/soccer_<iteration>.onnx")
    args = parser.parse_args()

    env_cfg = load_env_cfg(TASK, play=True)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cpu"))
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), log_dir=None, device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location="cpu")

    out = args.out or args.checkpoint.with_name(f"soccer_{args.checkpoint.stem.removeprefix('model_')}.onnx")
    runner.export_policy_to_onnx(str(out.parent.resolve()), out.name)

    model = onnx.load(str(out))
    for key, value in {
        "task": TASK,
        "obs_dim": str(OBS_DIM),
        "action": "[-1, 1]^4: vx, vy, yaw rate (a * upper for a >= 0, a * -lower below 0), kick trigger (> 0)",
        "twist_lower": ",".join(str(v) for v in TWIST_LOWER),
        "twist_upper": ",".join(str(v) for v in TWIST_UPPER),
        "checkpoint": str(args.checkpoint),
    }.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
