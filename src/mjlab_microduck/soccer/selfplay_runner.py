"""Who the learner plays, and when it stops being taught: teacher phase first, then self-play.

**Teacher phase.** Red is a statue in `P_STATUE` of the envs (scoring practice) and the scripted
`Oracle` in the rest (a competent striker and defender). `PpoWithTeacher` imitates the oracle
with weight `bc_coef`. Once the learner holds its own against the oracle — its kickoff win rate
minus loss rate against it, smoothed, reaches 0 — the weight fades to zero over `FADE_ITERS`.
Once it's gone and the learner is ahead by `SELFPLAY_MARGIN`, play switches over.

**Self-play phase** — Unity ML-Agents style. Red is a frozen snapshot of the learner in
`P_SNAPSHOT` of the envs and the oracle in the rest, so beating a competent scripted player stays
part of the job and isn't forgotten. Every `SAVE_EVERY` iterations the actor joins a pool of the
last `WINDOW` snapshots. Every `SWAP_EVERY` iterations the snapshot in use is replaced: the newest
with probability `PLAY_LATEST`, otherwise one drawn uniformly. Elo (`SelfPlay/elo`) is computed
from kickoff points against snapshots only.

Who plays red in which env is redrawn every `RESAMPLE_EVERY` iterations, and kickoff outcomes are
logged per opponent kind (`Opponent/*`). rsl_rl has no iteration hook, so all of this rides on
`alg.update`, which runs exactly once per iteration after the rollout.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy

import torch

from mjlab.rl import MjlabOnPolicyRunner

from mjlab_microduck.soccer.mdp import (
    BLUE, OPP_NAMES, OPP_ORACLE, OPP_SNAPSHOT, OPP_STATUE, RED, match_stats, opponent_kind,
)
from mjlab_microduck.soccer.oracle import Oracle

TEACHER, SELFPLAY = "teacher", "selfplay"
P_STATUE = 0.3  # teacher phase: share of envs where red stands still
P_SNAPSHOT = 0.7  # self-play phase: share of envs against snapshots (the rest: the oracle)
RESAMPLE_EVERY = 25
MARGIN_SMOOTHING = 0.05  # EMA step of (win - loss) against the oracle, per iteration
FADE_ITERS = 500
SELFPLAY_MARGIN = 0.10
SAVE_EVERY = 50
SWAP_EVERY = 25
WINDOW = 10
PLAY_LATEST = 0.5
ELO_K = 16.0
ELO_START = 1200.0
OPPONENT_TERM = "red_walk"
LEARNER_TERM = "blue_walk"
POOL_FILE = "selfplay_pool.pt"


class MixedOpponent:
    """Red's driver: per env a statue, the oracle or the current snapshot, by `opponent_kind`."""

    def __init__(self, env):
        self.env = env
        self.oracle = Oracle(env, RED, BLUE, -1)
        self.snapshot: torch.nn.Module | None = None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        kind = opponent_kind(self.env)
        out = torch.zeros(self.env.num_envs, 4, device=self.env.device)
        out[:, 3] = -1.0  # a statue never triggers a kick
        oracle = kind == OPP_ORACLE
        if bool(oracle.any()):
            out = torch.where(oracle.unsqueeze(-1), self.oracle(), out)
        snap = kind == OPP_SNAPSHOT
        if self.snapshot is not None and bool(snap.any()):
            out = torch.where(snap.unsqueeze(-1), self.snapshot(obs), out)
        return out


class SelfPlayRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device)
        self.mode = TEACHER
        self.bc_coef = 1.0
        self.margin = -1.0  # smoothed (win - loss) against the oracle
        self.fade_started: int | None = None
        self.elo = ELO_START
        self.opponent_elo = ELO_START
        self.pool: list[tuple[torch.nn.Module, float]] = []
        self._updates = 0
        self.opponent = MixedOpponent(self.env.unwrapped)
        self._opponent_term.opponent_policy = self.opponent
        self._resample_opponents()
        self._push_coef()
        update = self.alg.update

        def update_then_selfplay():
            losses = update()
            self._after_update()
            return losses

        self.alg.update = update_then_selfplay

    @property
    def _opponent_term(self):
        return self.env.unwrapped.action_manager.get_term(OPPONENT_TERM)

    def _snapshot(self) -> torch.nn.Module:
        # `as_jit` is a deep copy of the normalizer and the MLP with a deterministic head:
        # tensor in, action out — exactly what the opponent term calls.
        return self.alg.get_policy().as_jit().to(self.device).eval().requires_grad_(False)

    def _push_coef(self) -> None:
        if hasattr(self.alg, "bc_coef"):
            self.alg.bc_coef = self.bc_coef

    def _resample_opponents(self) -> None:
        kind = opponent_kind(self.env.unwrapped)
        r = torch.rand(kind.shape[0], device=kind.device)
        if self.mode == TEACHER:
            kind[:] = torch.where(r < P_STATUE, OPP_STATUE, OPP_ORACLE)
        else:
            kind[:] = torch.where(r < P_SNAPSHOT, OPP_SNAPSHOT, OPP_ORACLE)

    def _after_update(self) -> None:
        self._updates += 1
        it = self.current_learning_iteration + 1
        unwrapped = self.env.unwrapped
        learner = unwrapped.action_manager.get_term(LEARNER_TERM)
        self._log(it, kicks_per_env=float(learner.kicks_fired) / self.env.num_envs,
                  kicks_cancelled_per_env=float(learner.kicks_cancelled) / self.env.num_envs)
        learner.kicks_fired.zero_()
        learner.kicks_cancelled.zero_()
        stats = match_stats(unwrapped)
        # Per practice situation and per opponent kind, before `take` clears the counts.
        self._log(it, prefix="Scenario", **stats.take_scenarios())
        vs = stats.take_vs()
        stats.take()
        for name, (w, l, d) in vs.items():
            n = w + l + d
            if n > 0:
                self._log(it, prefix="Opponent", **{f"{name}_win": w / n, f"{name}_loss": l / n, f"{name}_draw": d / n})

        w, l, d = vs[OPP_NAMES[OPP_ORACLE]]
        if w + l + d > 0:
            self.margin += MARGIN_SMOOTHING * ((w - l) / (w + l + d) - self.margin)
        if self.mode == TEACHER:
            self._teacher_step(it)
        else:
            self._selfplay_step(it, vs[OPP_NAMES[OPP_SNAPSHOT]])
        self._log(it, prefix="Teacher", coef=self.bc_coef, margin_vs_oracle=self.margin,
                  selfplay=float(self.mode == SELFPLAY))
        if self._updates % RESAMPLE_EVERY == 0:
            self._resample_opponents()

    def _teacher_step(self, it: int) -> None:
        if self.fade_started is None and self.margin >= 0.0:
            self.fade_started = self._updates
        if self.fade_started is not None:
            self.bc_coef = max(0.0, 1.0 - (self._updates - self.fade_started) / FADE_ITERS)
            self._push_coef()
        if self.bc_coef == 0.0 and self.margin >= SELFPLAY_MARGIN:
            self.mode = SELFPLAY
            self.pool.append((self._snapshot(), self.elo))
            self.opponent.snapshot = self.pool[-1][0]
            self._resample_opponents()
            print(f"[selfplay] iteration {it}: learner beats the oracle (margin {self.margin:.2f}); switching to self-play")

    def _selfplay_step(self, it: int, vs_snapshot: tuple[float, float, float]) -> None:
        w, l, d = vs_snapshot
        points = w + l + d
        if points > 0:
            score = (w + 0.5 * d) / points
            expected = 1.0 / (1.0 + 10 ** ((self.opponent_elo - self.elo) / 400.0))
            self.elo += ELO_K * (score - expected)
            self._log(it, elo=self.elo, opponent_elo=self.opponent_elo, win_rate=w / points, loss_rate=l / points,
                      draw_rate=d / points, goals_per_point=(w + l) / points, pool=len(self.pool))
        if self._updates % SAVE_EVERY == 0:
            self.pool.append((self._snapshot(), self.elo))
            del self.pool[:-WINDOW]
        if self._updates % SWAP_EVERY == 0 and self.pool:
            policy, elo = self.pool[-1] if random.random() < PLAY_LATEST else random.choice(self.pool)
            self.opponent.snapshot, self.opponent_elo = policy, elo

    def _log(self, it: int, prefix: str = "SelfPlay", **scalars: float) -> None:
        writer = self.logger.writer
        if writer is None:
            return
        for name, value in scalars.items():
            writer.add_scalar(f"{prefix}/{name}", value, it)

    # ── persistence: a resumed run keeps its phase, its opponents and its rating ──

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        state = {
            "mode": self.mode, "bc_coef": self.bc_coef, "margin": self.margin, "elo": self.elo,
            "fade_started": self.fade_started, "updates": self._updates,
            "snapshots": [(deepcopy(p.state_dict()), elo) for p, elo in self.pool],
        }
        torch.save(state, os.path.join(os.path.dirname(path), POOL_FILE))

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        state_path = os.path.join(os.path.dirname(path), POOL_FILE)
        if os.path.exists(state_path):
            saved = torch.load(state_path, map_location=self.device, weights_only=False)
            self.mode = saved.get("mode", SELFPLAY)
            self.bc_coef, self.margin = saved.get("bc_coef", 0.0), saved.get("margin", 0.0)
            self.elo, self.fade_started = saved["elo"], saved.get("fade_started")
            self._updates = saved.get("updates", 0)
            self.pool = []
            for state, elo in saved["snapshots"]:
                policy = self._snapshot()
                policy.load_state_dict(state)
                self.pool.append((policy, elo))
            self._push_coef()
        # Whatever is loaded plays itself: `play` and the 50 fps renders show a match. A resumed
        # training run redraws its opponents at the next `RESAMPLE_EVERY`.
        self.opponent.snapshot = self._snapshot()
        opponent_kind(self.env.unwrapped)[:] = OPP_SNAPSHOT
        return infos
