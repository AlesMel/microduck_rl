"""Referee a duck-sim football match: two real robotd daemons, each steered by a match policy.

    DUCK_SIM_DUCKS=2 DUCK_SIM_SCENE=soccer scripts/duck-sim        # in the microduck repo
    uv run scripts/soccer_referee.py --blue soccer_3000.onnx [--red soccer_1500.onnx] [--points 5]

Every 0.1 s it reads the world from `duck-body`'s referee port (`body_server.py --referee-port`:
where both ducks and the ball are — what an overhead camera would give a real pitch), builds each
side's 26D observation with the very `side_obs` training used, runs the exported policy, and sends
each duck `robot.move` over its own robotd socket. robotd's walker does the rest, exactly as it
would for a gamepad. Goals are decided by the training referee's `goal_flags`; after one, play
restarts from the kickoff spots.

`robot.move` expires after 500 ms (the deadman), so each twist is resent every 50 ms; a referee
that dies leaves two ducks that stop on their own.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab_microduck.soccer.frozen_walker import action_to_twist
from mjlab_microduck.soccer.mdp import (
    KICK_LEFT, KICK_NONE, ROUTINE, fire_side, goal_flags, kick_side, routine_steer, side_obs, to_frame,
)

DECISION_DT = 0.1
RESEND_DT = 0.05


class Line:
    """One newline-framed JSON connection (the referee port, or a robotd socket)."""

    def __init__(self, sock: socket.socket):
        sock.settimeout(5.0)
        self.sock = sock
        self.file = sock.makefile("rw")

    def send(self, message: dict) -> None:
        self.file.write(json.dumps(message) + "\n")
        self.file.flush()

    def ask(self, message: dict) -> dict:
        self.send(message)
        answer = json.loads(self.file.readline())
        if "error" in answer and answer["error"]:
            raise RuntimeError(answer["error"])
        return answer


class Duck:
    """A robotd socket: requests for enable, notifications for the twist."""

    def __init__(self, path: Path):
        s = socket.socket(socket.AF_UNIX)
        s.connect(str(path))
        self.line = Line(s)
        self.path = path
        self._id = 0
        # The stop-then-kick routine, as `WalkerCommandAction` runs it in training: walk in
        # steering the ball to a foot, stop at `stop_x`, settle, then `robot.do` or cancel.
        self.mode = "walk"  # walk | approach | settle | kick
        self.foot = KICK_NONE
        self.until = 0.0  # end of the settle or the kick; the approach's timeout

    def kick_fraction(self, now: float) -> float:
        """The observation's routine slot: 0 walking, 1 closing in, then 1 -> 0 over settle + kick."""
        if self.mode == "approach":
            return 1.0
        if self.mode == "settle":
            return (self.until - now + ROUTINE["kick_s"]) / (ROUTINE["settle_s"] + ROUTINE["kick_s"])
        if self.mode == "kick":
            return max(0.0, self.until - now) / (ROUTINE["settle_s"] + ROUTINE["kick_s"])
        return 0.0

    def call(self, method: str, params: dict) -> dict:
        self._id += 1
        return self.line.ask({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})

    def move(self, twist) -> None:
        vx, vy, vyaw = (float(v) for v in twist)
        self.line.send({"jsonrpc": "2.0", "method": "robot.move", "params": {"vx": vx, "vy": vy, "vyaw": vyaw}})


class Policy:
    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def decide(self, obs: torch.Tensor) -> tuple[np.ndarray, bool]:
        """(twist, kick trigger), the same split `WalkerCommandAction` makes in training."""
        action = self.session.run(None, {self.input: obs.numpy().astype(np.float32)})[0][0]
        return action_to_twist(action[:3]), bool(action[3] > 0)


def observe(world: dict, team: int, time_left: float, own_kick: float) -> torch.Tensor:
    """`side_obs` from the referee port's numbers; duck a is blue, duck b is red."""
    own, opp = (world["ducks"][0], world["ducks"][1]) if team > 0 else (world["ducks"][1], world["ducks"][0])
    t = lambda v: torch.tensor([v], dtype=torch.float32)
    return side_obs(
        t(own["pos"]), t(own["quat"]), t(own["linvel"]), t(own["angvel"][2]), t(own["gravity_z"]),
        t(opp["pos"]), t(opp["quat"]), t(opp["linvel"]), t(opp["gravity_z"]),
        t(world["ball"]["pos"]), t(world["ball"]["vel"]), t(time_left), t(own_kick), team,
    )


def ball_rel(world: dict, index: int) -> torch.Tensor:
    own = world["ducks"][index]
    t = lambda v: torch.tensor([v], dtype=torch.float32)
    return to_frame(t(own["pos"]), t(own["quat"]), t(world["ball"]["pos"]))


def routine_step(duck: Duck, world: dict, index: int, now: float) -> np.ndarray | None:
    """Advance a duck's kick routine; the twist it needs, or None once it is back to walking."""
    rel = ball_rel(world, index)
    if duck.mode == "approach":
        if int(kick_side(rel)[0]) == KICK_NONE or now >= duck.until:
            duck.mode = "walk"
            return None
        if rel[0, 0] <= ROUTINE["stop_x"]:
            duck.mode, duck.until = "settle", now + ROUTINE["settle_s"]
            return np.zeros(3)
        steer = float(routine_steer(rel, torch.tensor([duck.foot]))[0])
        return np.array([ROUTINE["approach_cmd"], 0.0, steer])
    if duck.mode == "settle":
        if now < duck.until:
            return np.zeros(3)
        foot = int(fire_side(rel)[0])
        speed = float(np.linalg.norm(world["ducks"][index]["linvel"][:2]))
        if foot != KICK_NONE and speed < ROUTINE["still_speed"]:
            skill = "kick_left" if foot == KICK_LEFT else "kick_right"
            answer = duck.call("robot.do", {"skill": skill}).get("result") or {}
            if answer.get("accepted"):
                duck.mode, duck.until = "kick", now + ROUTINE["kick_s"]
                print(f"==   {duck.path.stem} {skill}")
                return np.zeros(3)
        duck.mode = "walk"
        return None
    if duck.mode == "kick":
        if now < duck.until:
            return np.zeros(3)
        duck.mode = "walk"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Referee a duck-sim football match")
    parser.add_argument("--blue", type=Path, required=True, help="exported match policy for duck a")
    parser.add_argument("--red", type=Path, default=None, help="for duck b (default: the same as blue)")
    parser.add_argument("--referee", default="127.0.0.1:7800", help="duck-body's referee port")
    parser.add_argument("--state", type=Path, default=Path(os.environ.get("DUCK_SIM_STATE", "~/.cache/duck-sim")).expanduser())
    parser.add_argument("--match-seconds", type=float, default=45.0, help="per point, as in training")
    parser.add_argument("--points", type=int, default=0, help="stop after this many points (0: forever)")
    args = parser.parse_args()

    host, port = args.referee.rsplit(":", 1)
    referee = Line(socket.create_connection((host, int(port))))
    ducks = [Duck(args.state / f"duck-{c}.sock") for c in "ab"]
    policies = [Policy(args.blue), Policy(args.red or args.blue)]
    for duck in ducks:
        print(f"== {duck.path.name}: enable -> {duck.call('robot.enable', {'on': True}).get('result')}")

    score = [0, 0]
    points = 0
    referee.ask({"op": "kickoff"})
    started = time.monotonic()
    twists = [np.zeros(3), np.zeros(3)]
    next_decision = next_send = time.monotonic()
    print("== kickoff")
    try:
        while not args.points or points < args.points:
            now = time.monotonic()
            if now >= next_decision:
                next_decision += DECISION_DT
                world = referee.ask({"op": "world"})
                elapsed = now - started
                time_left = max(0.0, 1.0 - elapsed / args.match_seconds)
                twists = []
                for i, team in enumerate((1, -1)):
                    duck = ducks[i]
                    twist, trigger = policies[i].decide(observe(world, team, time_left, duck.kick_fraction(now)))
                    foot = int(kick_side(ball_rel(world, i))[0])
                    if trigger and duck.mode == "walk" and foot != KICK_NONE:
                        duck.mode, duck.foot, duck.until = "approach", foot, now + ROUTINE["timeout_s"]
                    # In the routine the twist is the routine's, not the policy's.
                    routine = routine_step(duck, world, i, now) if duck.mode != "walk" else None
                    twists.append(twist if routine is None else routine)
                blue_scored, red_scored = (bool(f) for f in goal_flags(torch.tensor([world["ball"]["pos"]])))
                if blue_scored or red_scored or time_left == 0.0:
                    points += 1
                    if blue_scored:
                        score[0] += 1
                    if red_scored:
                        score[1] += 1
                    what = "blue scores" if blue_scored else "red scores" if red_scored else "time — a draw"
                    print(f"== {elapsed:5.1f}s  {what}   blue {score[0]} : {score[1]} red")
                    for duck in ducks:
                        duck.move((0.0, 0.0, 0.0))
                    time.sleep(1.0)
                    referee.ask({"op": "kickoff"})
                    time.sleep(1.0)  # let the walker settle before the whistle
                    started = next_decision = next_send = time.monotonic()
                    continue
            if now >= next_send:
                next_send += RESEND_DT
                # The routine needs the stop timed to the centimetre, so it runs at the resend rate
                # on a fresh world, not at the policy's 10 Hz.
                if any(d.mode != "walk" for d in ducks):
                    world = referee.ask({"op": "world"})
                    for i, duck in enumerate(ducks):
                        if duck.mode != "walk":
                            routine = routine_step(duck, world, i, now)
                            twists[i] = np.zeros(3) if routine is None else routine
                for duck, twist in zip(ducks, twists):
                    duck.move(twist)
            time.sleep(max(0.0, min(next_decision, next_send) - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        for duck in ducks:
            try:
                duck.move((0.0, 0.0, 0.0))
            except OSError:
                pass
        print(f"== final: blue {score[0]} : {score[1]} red over {points} point(s)")


if __name__ == "__main__":
    main()
