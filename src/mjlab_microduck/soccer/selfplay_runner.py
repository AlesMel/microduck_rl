"""Self-play the way Unity ML-Agents does it: the learner against frozen copies of its past self.

Every `SAVE_EVERY` iterations the current actor is frozen into a pool of the last `WINDOW`
snapshots. Every `SWAP_EVERY` iterations red's driver is replaced: the newest snapshot with
probability `PLAY_LATEST`, otherwise one drawn uniformly from the pool. Playing old selves as well
as the newest one is what stops the cycle Unity's docs warn about — a policy that only ever beats
its latest self forgets how to beat the ones before it.

Progress is an Elo rating (`SelfPlay/elo`): each snapshot keeps the learner's rating at the moment
it was frozen, and the learner's moves by K x (score - expected) against the current opponent's
rating every iteration, from that iteration's win/draw/loss counts. A flat Elo with a falling
`SelfPlay/goals_per_point` is the usual sign the two sides have learned to defend; a rising one is
the learner getting better than its past.

rsl_rl has no iteration hook, so the bookkeeping rides on `alg.update`, which runs exactly once per
iteration after the rollout.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy

import torch

from mjlab.rl import MjlabOnPolicyRunner

from mjlab_microduck.soccer.mdp import match_stats

SAVE_EVERY = 50
SWAP_EVERY = 25
WINDOW = 10
PLAY_LATEST = 0.5
ELO_K = 16.0
ELO_START = 1200.0
OPPONENT_TERM = "red_walk"
LEARNER_TERM = "blue_walk"
POOL_FILE = "selfplay_pool.pt"


class SelfPlayRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device)
        self.elo = ELO_START
        self.opponent_elo = ELO_START  # the statue red starts as
        self.pool: list[tuple[torch.nn.Module, float]] = []
        self._updates = 0
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
        # tensor in, twist out — exactly what the opponent term calls.
        return self.alg.get_policy().as_jit().to(self.device).eval().requires_grad_(False)

    def _set_opponent(self, policy: torch.nn.Module | None, elo: float) -> None:
        self._opponent_term.opponent_policy = policy
        self.opponent_elo = elo

    def _after_update(self) -> None:
        self._updates += 1
        it = self.current_learning_iteration + 1
        learner = self.env.unwrapped.action_manager.get_term(LEARNER_TERM)
        self._log(it, kicks_per_env=float(learner.kicks_fired) / self.env.num_envs)
        learner.kicks_fired.zero_()
        blue, red, draws = match_stats(self.env.unwrapped).take()
        points = blue + red + draws
        if points > 0:
            score = (blue + 0.5 * draws) / points
            expected = 1.0 / (1.0 + 10 ** ((self.opponent_elo - self.elo) / 400.0))
            self.elo += ELO_K * (score - expected)
            self._log(it, elo=self.elo, opponent_elo=self.opponent_elo, win_rate=blue / points,
                      loss_rate=red / points, draw_rate=draws / points, goals_per_point=(blue + red) / points,
                      pool=len(self.pool))

        if self._updates % SAVE_EVERY == 0:
            self.pool.append((self._snapshot(), self.elo))
            del self.pool[:-WINDOW]
        if self._updates % SWAP_EVERY == 0 and self.pool:
            policy, elo = self.pool[-1] if random.random() < PLAY_LATEST else random.choice(self.pool)
            self._set_opponent(policy, elo)

    def _log(self, it: int, **scalars: float) -> None:
        writer = self.logger.writer
        if writer is None:
            return
        for name, value in scalars.items():
            writer.add_scalar(f"SelfPlay/{name}", value, it)

    # ── persistence: a resumed run keeps its opponents and its rating ──

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        pool = {
            "elo": self.elo,
            "snapshots": [(deepcopy(p.state_dict()), elo) for p, elo in self.pool],
        }
        torch.save(pool, os.path.join(os.path.dirname(path), POOL_FILE))

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        # Whatever is loaded plays itself until a pool says otherwise — which is also what makes
        # `play` show a match rather than a duck scoring past a statue.
        self._set_opponent(self._snapshot(), self.elo)
        pool_path = os.path.join(os.path.dirname(path), POOL_FILE)
        if os.path.exists(pool_path):
            saved = torch.load(pool_path, map_location=self.device, weights_only=False)
            self.elo = saved["elo"]
            self.pool = []
            for state, elo in saved["snapshots"]:
                policy = self._snapshot()
                policy.load_state_dict(state)
                self.pool.append((policy, elo))
        return infos
