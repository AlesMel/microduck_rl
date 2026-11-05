"""50 fps video of one robot running a checkpoint through a fixed command schedule (CPU sim).

One frame per 50 Hz policy step, camera following the trunk. The command is written into the
twist obs slots (head/body slots zeroed), as in the other eval scripts. The overlay shows the
command and the measured forward speed (1 s average); a per-segment summary (mean vx / wz
over the last 2 s of each segment) is printed at the end.

usage: uv run scripts/eval/render_ckpt.py MODEL.pt OUT.mp4 [--task ...] [--label TEXT]
       [--robot-chain]
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from _common import DEFAULT_WALK_TASK, RobotChain, load_policy, make_play_env, step  # noqa: E402

# (seconds, vx, vy, wz, label)
SCHEDULE = [
    (1.5, 0.0, 0.0, 0.0, "stand"),
    (3.0, 0.1, 0.0, 0.0, "walk 0.1"),
    (3.0, 0.2, 0.0, 0.0, "walk 0.2"),
    (3.0, 0.3, 0.0, 0.0, "walk 0.3"),
    (4.0, 0.4, 0.0, 0.0, "walk 0.4"),
    (3.0, 0.5, 0.0, 0.0, "walk 0.5"),
    (2.0, 0.0, 0.0, 0.0, "stop"),
    (3.0, -0.3, 0.0, 0.0, "backward 0.3"),
    (3.0, 0.0, 0.2, 0.0, "sideways 0.2"),
    (3.0, 0.0, 0.0, 1.0, "turn 1.0 rad/s"),
    (1.5, 0.0, 0.0, 0.0, "stand"),
]

p = argparse.ArgumentParser()
p.add_argument("ckpt"); p.add_argument("out")
p.add_argument("--task", default=DEFAULT_WALK_TASK)
p.add_argument("--label", default=None)
p.add_argument("--robot-chain", action="store_true")
a = p.parse_args()
label = a.label or a.ckpt.rsplit("/", 1)[-1]


def camera(cfg):
    cfg.viewer.distance = 0.65
    cfg.viewer.elevation = -12.0
    cfg.viewer.azimuth = 135.0
    cfg.viewer.width, cfg.viewer.height = 960, 720


policy = load_policy(a.ckpt)
env, raw = make_play_env(a.task, 1, "cpu", cfg_hook=camera)
# make_play_env builds without a render mode; rebuild the env with one.
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
cfg = raw.cfg
env.close()
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu", render_mode="rgb_array"))
raw = env.unwrapped
robot = raw.scene["robot"]
chain = RobotChain(raw, 1, "cpu") if a.robot_chain else None
fps = round(1.0 / raw.step_dt)

try:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
except OSError:
    font = ImageFont.load_default()

obs = env.get_observations()["actor"]
writer = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
vx_hist, summary = [], []
for dur, vx_cmd, vy_cmd, wz_cmd, seg in SCHEDULE:
    seg_vx, seg_wz = [], []
    for _ in range(int(round(dur * fps))):
        obs = step(env, policy, obs, vx_cmd, vy_cmd, wz_cmd, chain=chain)
        vx = float(robot.data.root_link_lin_vel_b[0, 0])
        wz = float(robot.data.root_link_ang_vel_b[0, 2])
        vx_hist.append(vx); seg_vx.append(vx); seg_wz.append(wz)
        frame = Image.fromarray(raw.render())
        d = ImageDraw.Draw(frame)
        d.rectangle((0, 0, frame.width, 64), fill=(0, 0, 0))
        d.text((12, 6), f"{label}   [{seg}]{'  robot chain' if chain else ''}", fill=(255, 255, 255), font=font)
        d.text((12, 34), f"cmd vx {vx_cmd:+.2f} vy {vy_cmd:+.2f} m/s  wz {wz_cmd:+.1f}   measured vx "
                         f"{np.mean(vx_hist[-fps:]):+.2f} m/s (1 s avg)", fill=(255, 220, 120), font=font)
        writer.append_data(np.asarray(frame))
    tail = int(2 * fps)
    summary.append((seg, vx_cmd, wz_cmd, float(np.mean(seg_vx[-tail:])), float(np.mean(seg_wz[-tail:]))))
writer.close()
up = float(-robot.data.projected_gravity_b[0, 2])
print(f"VIDEO {a.out} fps={fps} frames={len(vx_hist)} upright_cos={up:.2f}")
for seg, vxc, wzc, vx, wz in summary:
    print(f"  {seg:16s} cmd vx {vxc:+.2f} wz {wzc:+.1f} -> vx {vx:+.3f} m/s, wz {wz:+.2f} rad/s (last 2 s)")
