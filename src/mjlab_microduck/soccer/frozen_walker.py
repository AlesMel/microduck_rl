"""The shipped walking policy, frozen, as a batched torch module.

Every policy in the official set is the same graph: `(obs - mean) / std`, then Gemm/ELU layers
(61 -> 512 -> 256 -> 128 -> 14). Reading that graph's initializers into `nn.Linear` layers gives
the exact network robotd runs, on the env's device, at 4096 envs per call — onnxruntime would
need a host round trip per substep. `tests/test_frozen_walker.py` pins the two to each other.

Which file: `velstand.onnx` from the Hub set. It walks on a twist, stands at zero command and gets
back up from a fall (set v8), which is what a duck needs in a game where it gets bumped.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

POLICY_REPO = "pollen-robotics/microduck-policies"
# The set robotd seeds a laptop with (`scripts/seed-policies.sh` in the microduck repo). Pinned,
# because the match policy is trained against one walker and a newer gait is a different game.
POLICY_VERSION = "v8"
WALKER_FILE = "velstand.onnx"

OBS_DIM = 61
ACTION_DIM = 14


def walker_path(filename: str = WALKER_FILE, revision: str = POLICY_VERSION) -> Path:
    """The walker on disk, fetched from the Hub once and cached by huggingface_hub."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(POLICY_REPO, filename, revision=revision))


class FrozenMlp(nn.Module):
    """`(obs - mean) / std` followed by the policy MLP. No gradients, ever."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, mlp: nn.Sequential):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.mlp = mlp
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / self.std)

    @classmethod
    def from_onnx(cls, path: str | Path, device: str | torch.device = "cpu") -> FrozenMlp:
        """Rebuild the network from the exporter's graph.

        Walks the nodes in order rather than assuming names, but refuses anything that is not the
        exporter's shape: a policy with an LSTM or a different normalizer is a different contract,
        and loading half of it would run a wrong network without complaint.
        """
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        init = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in model.graph.initializer}
        mean = std = None
        layers: list[nn.Module] = []
        for node in model.graph.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "Sub" and mean is None:
                mean = init[node.input[1]].float().reshape(1, -1)
            elif node.op_type == "Div" and std is None:
                std = init[node.input[1]].float().reshape(1, -1)
            elif node.op_type == "Gemm":
                if attrs.get("transB", 0) != 1 or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                    raise ValueError(f"{path}: Gemm {node.name} is not the exporter's (transB=1, alpha=beta=1)")
                weight, bias = init[node.input[1]].float(), init[node.input[2]].float()
                linear = nn.Linear(weight.shape[1], weight.shape[0])
                linear.weight.data.copy_(weight)
                linear.bias.data.copy_(bias)
                layers.append(linear)
            elif node.op_type == "Elu":
                layers.append(nn.ELU(alpha=float(attrs.get("alpha", 1.0))))
            else:
                raise ValueError(f"{path}: unexpected {node.op_type} node — not a plain exported MLP")
        if mean is None or std is None or not layers:
            raise ValueError(f"{path}: no obs normalizer or no layers found")
        net = cls(mean, std, nn.Sequential(*layers))
        if net.mean.shape[1] != OBS_DIM or layers[-1].out_features != ACTION_DIM:
            raise ValueError(f"{path}: {net.mean.shape[1]} -> {layers[-1].out_features}, expected {OBS_DIM} -> {ACTION_DIM}")
        return net.to(device)
