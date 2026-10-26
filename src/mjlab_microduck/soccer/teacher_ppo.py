"""PPO + behaviour cloning from the scripted striker, on the states the student itself visits.

Five runs of pure RL never found the multi-second chain a goal needs (go round the ball, stop,
face the goal, trigger the kick); every one settled on whatever dense term was cheapest. The
scripted `Oracle` plays that chain and scores in 68% of points against a statue. So the match
policy first learns to do what the oracle does, DAgger-style: after every PPO update, a BC pass
regresses the student's deterministic action onto the oracle's action *for the states the student
reached*. The oracle's action rides along in the rollout as the `teacher` observation group, which
neither actor nor critic reads, but rsl_rl's storage keeps.

The same mechanics as `tasks/distill.py`'s `PpoWithExpertBc` (post-update pass, own Adam, minimum
mini-batch), minus the gating that one needs: here every frame has a teacher label. The weight
`bc_coef` is set by `SelfPlayRunner` from competence — full until the student holds its own against
the oracle, then faded out, leaving the goals reward to take it further.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from rsl_rl.algorithms import PPO

from mjlab.rl import RslRlPpoAlgorithmCfg


def default_teacher_cfg() -> dict:
    return {
        "group": "teacher",  # observation group carrying the oracle's action
        "learning_rate": 3.0e-4,
        "epochs": 2,
        "mini_batches": 4,
        "min_mini_batch": 4096,
        # Twist dims and the kick trigger. The trigger is the rarest and most decisive output.
        "action_weights": (1.0, 1.0, 1.0, 2.0),
    }


@dataclass(kw_only=True)
class PpoWithTeacherCfg(RslRlPpoAlgorithmCfg):
    teacher_cfg: dict | None = field(default_factory=default_teacher_cfg)
    class_name: str = "mjlab_microduck.soccer.teacher_ppo.PpoWithTeacher"


class PpoWithTeacher(PPO):
    def __init__(self, actor, critic, storage, *args, teacher_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.teacher_cfg = {**default_teacher_cfg(), **(teacher_cfg or {})}
        self.bc_coef = 1.0  # set by the runner
        self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.teacher_cfg["learning_rate"])
        self._weights = torch.tensor(self.teacher_cfg["action_weights"], device=self.device)

    def _bc_update(self) -> dict[str, float]:
        cfg = self.teacher_cfg
        if self.bc_coef <= 0.0:
            return {"teacher_bc": 0.0, "teacher_coef": 0.0}
        obs_td = self.storage.observations.flatten(0, 1)
        target = obs_td[cfg["group"]].clamp(-1.0, 1.0)
        n = target.shape[0]
        mb = max(cfg["min_mini_batch"], n // cfg["mini_batches"])
        total, steps = 0.0, 0
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                sel = perm[s : s + mb]
                # The student's actions are clipped to [-1, 1] when used; pull the clipped value.
                pred = self.actor(obs_td[sel]).clamp(-1.5, 1.5)
                loss = self.bc_coef * ((pred - target[sel]).pow(2) * self._weights).mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                total += loss.item() / self.bc_coef
                steps += 1
        return {"teacher_bc": total / max(steps, 1), "teacher_coef": self.bc_coef}

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict.update(self._bc_update())
        return loss_dict
