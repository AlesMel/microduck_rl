"""Render a Soccer-1v1 checkpoint as a smooth match video — 50 fps rather than training's 10.

    uv run scripts/soccer_video.py logs/rsl_rl/soccer_1v1/<run>/model_1500.pt [--seconds 30] [--out match.mp4]

Training's videos come from mjlab's recorder, which grabs one frame per env step: one per match
decision, 10 a second, so a duck that takes five walker steps between decisions jumps between
frames. Here a frame is grabbed at every walker tick instead (50 Hz, inside the decimation loop),
which is every step the duck's legs actually take.

Both sides play the checkpoint's deterministic policy — the learner against itself — so this
shows what the policy does, not what training's exploration noise does to it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mediapy as media

import mjlab_microduck.tasks  # noqa: F401  (registers the task)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_microduck.soccer.mdp import match_stats

TASK = "Mjlab-Soccer-1v1-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Soccer-1v1 checkpoint at 50 fps")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <checkpoint>_50fps.mp4 beside it")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_env_cfg(TASK, play=True)
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array"))
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    raw = env.unwrapped
    term = raw.action_manager.get_term("blue_walk")
    frames = []
    apply = term.apply_actions

    def apply_and_capture() -> None:
        # The walker ticks on substep 0, 4, 8, ... of each decision; that is when the legs move.
        tick = term._substep % term._period == 0
        apply()
        if tick:
            frames.append(raw.render())

    term.apply_actions = apply_and_capture
    fps = round(1.0 / (term._period * raw.physics_dt))

    obs = env.get_observations()
    for _ in range(round(args.seconds / raw.step_dt)):
        obs, *_ = env.step(policy(obs))

    out = args.out or args.checkpoint.with_name(f"{args.checkpoint.stem}_{fps}fps.mp4")
    media.write_video(str(out), frames, fps=fps)
    blue, red, draws = match_stats(raw).take()
    print(f"wrote {out}: {len(frames)} frames at {fps} fps; goals blue {blue:.0f} red {red:.0f}, "
          f"kicks {float(term.kicks_fired):.0f}")


if __name__ == "__main__":
    main()
