"""Play kickoff points between any two of {statue, oracle, checkpoint} and report how they went.

    uv run scripts/soccer_benchmark.py --blue oracle --red statue
    uv run scripts/soccer_benchmark.py --blue logs/rsl_rl/soccer_1v1/<run>/model_2000.pt --red oracle

The yardstick for the soccer work: goals per point for each side, how many kicks fired, how many
of those the routine cancelled, and how many sent the ball at goal. Every number is from kickoff
points only (the play cfg), each env playing points back to back for `--seconds`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the task)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_microduck.soccer.mdp import (
    BLUE, OPP_ORACLE, OPP_SNAPSHOT, OPP_STATUE, RED, ball_to_goal_speed, match_stats, opponent_kind,
)
from mjlab_microduck.soccer.oracle import Oracle

TASK = "Mjlab-Soccer-1v1-MicroDuck"
GOALWARD = 0.3  # m/s toward the attacked goal, 0.6 s after a kick fired


def _checkpoint_policy(env, path: str, device: str):
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), log_dir=None, device=device)
    runner.load(path, load_cfg={"actor": True}, map_location=device)
    return runner.alg.get_policy().as_jit().to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer 1v1 kickoff benchmark")
    parser.add_argument("--blue", default="oracle", help="statue | oracle | path to a checkpoint")
    parser.add_argument("--red", default="statue", help="statue | oracle | path to a checkpoint")
    parser.add_argument("--red-skill", type=float, default=1.0, help="an oracle red's skill, (0, 1]")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=90.0, help="of play per env, points back to back")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = args.envs
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device))
    raw = env.unwrapped
    blue_term, red_term = raw.action_manager.get_term("blue_walk"), raw.action_manager.get_term("red_walk")

    if args.blue == "statue":
        blue = lambda obs: torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=args.device).expand(args.envs, 4)
    elif args.blue == "oracle":
        oracle = Oracle(raw, BLUE, RED, 1)
        blue = lambda obs: oracle()
    else:
        net = _checkpoint_policy(env, args.blue, args.device)
        blue = lambda obs: net(obs["actor"])
    # Who red is decides kickoff possession (blue always kicks off against a statue), so say.
    opponent_kind(raw)[:] = {"statue": OPP_STATUE, "oracle": OPP_ORACLE}.get(args.red, OPP_SNAPSHOT)
    if args.red == "statue":
        red_term.opponent_policy = None
    elif args.red == "oracle":
        red_term.opponent_policy = Oracle(raw, RED, BLUE, -1, skill=args.red_skill)
    else:
        red_term.opponent_policy = _checkpoint_policy(env, args.red, args.device)

    env.reset()  # place the first points for the kind of red just set
    for term in (blue_term, red_term):
        term.kicks_fired.zero_()
        term.kicks_cancelled.zero_()
    match_stats(raw).take()
    pending = torch.full((args.envs,), -1, dtype=torch.long, device=args.device)
    goalward = kicks_seen = 0
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(round(args.seconds / raw.step_dt)):
            obs, *_ = env.step(blue(obs))
            pending = torch.where(blue_term.fired_now, torch.full_like(pending, 6), pending - 1)
            due = pending == 0
            if due.any():
                speed = ball_to_goal_speed(raw, team=1, max_speed=5.0)[due]
                goalward += int((speed > GOALWARD).sum())
                kicks_seen += int(due.sum())

    by_possession = match_stats(raw).take_scenarios()
    blue_goals, red_goals, draws = match_stats(raw).take()
    points = blue_goals + red_goals + draws
    per = lambda x: x / points if points else float("nan")
    print(f"blue={args.blue} vs red={args.red}: {points:.0f} points | blue {blue_goals:.0f} goals ({per(blue_goals):.2f}/point), "
          f"red {red_goals:.0f} ({per(red_goals):.2f}/point), draws {draws:.0f}")
    print(f"  blue kicks fired {float(blue_term.kicks_fired):.0f}, cancelled {float(blue_term.kicks_cancelled):.0f}, "
          f"goalward {goalward}/{kicks_seen} ({goalward / max(1, kicks_seen):.0%}); red kicks fired {float(red_term.kicks_fired):.0f}")
    if "kickoff_kicker_goals" in by_possession:
        print(f"  by possession: the side kicking off scored {by_possession['kickoff_kicker_goals']:.2f}/point, "
              f"the other side {by_possession['kickoff_defender_goals']:.2f}/point")


if __name__ == "__main__":
    main()
