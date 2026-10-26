"""Does the kick work in play? Walk at the ball with the trigger held and let the routine do the rest.

    uv run scripts/soccer_kick_check.py [approach action, default 1.0]

32 ducks, each aimed at the red goal with the ball 0.7 m out and 0.35 m ahead of its kicking spot.
The stop-then-kick routine (`soccer/actions.py`) must turn >= 80% of these into goals. Kicked on
contact instead, they scored 0-12%. Rerun after any change to the walker, the kicks, the FIRE band
or the routine's numbers.
"""
import sys, torch
import mjlab_microduck.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_soccer_env_cfg import make_microduck_soccer_env_cfg
from mjlab_microduck.soccer import mdp as M
from mjlab_microduck.soccer.scenarios import yaw_quat, _behind_ball

speed = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
n = 32
cfg = make_microduck_soccer_env_cfg(play=True); cfg.scene.num_envs = n
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "nan_state"}
env = ManagerBasedRlEnv(cfg, device="cpu"); env.reset()
g = torch.Generator().manual_seed(0)
ball = torch.stack((0.7 + 0.0 * torch.rand(n, generator=g), (torch.rand(n, generator=g) - 0.5) * 0.4), -1)
yaw = torch.atan2(-ball[:, 1], 1.5 - ball[:, 0])
foot = torch.where(torch.rand(n, generator=g) < 0.5, 1.0, -1.0)
start = _behind_ball(ball, yaw, foot) - torch.stack((torch.cos(yaw), torch.sin(yaw)), -1) * 0.35
z6 = torch.zeros(n, 6); st = lambda xy, z, q: torch.cat((xy, torch.full((n, 1), z), q, z6), -1)
env.scene["robot_blue"].write_root_state_to_sim(st(start, M.TRUNK_Z, yaw_quat(yaw)))
env.scene["robot_red"].write_root_state_to_sim(st(torch.tensor([[-1.3, 0.8]]).expand(n, 2), M.TRUNK_Z, yaw_quat(torch.zeros(n))))
env.scene["ball"].write_root_state_to_sim(st(ball, M.BALL_RADIUS, torch.tensor([1., 0, 0, 0]).expand(n, 4)))
a = torch.zeros(n, 4); a[:, 3] = -1
for _ in range(8): env.step(a)
term = env.action_manager.get_term("blue_walk")
done = torch.zeros(n, dtype=torch.bool); scored = torch.zeros(n, dtype=torch.bool)
for i in range(80):
    a = torch.zeros(n, 4); a[:, 0] = speed; a[:, 3] = 1.0      # walk at it, trigger held
    a[done] = torch.tensor([0.0, 0.0, 0.0, -1.0])                # after one kick, stand
    env.step(a); done |= term.fired_now
    b, _ = M.goal_flags(M._local(env, "ball")); scored |= b
rate = scored.float().mean().item()
print(f"walk-in at action {speed}: kicks fired {float(term.kicks_fired):.0f}/{n}, cancelled {float(term.kicks_cancelled):.0f}, scored {rate:.0%}")
sys.exit(0 if rate >= 0.8 else 1)
