"""The torch walker must be the ONNX walker robotd runs, to float precision."""

import numpy as np
import pytest
import torch

from mjlab_microduck.soccer.frozen_walker import ACTION_DIM, OBS_DIM, FrozenMlp, walker_path


@pytest.fixture(scope="module")
def onnx_file():
    try:
        return walker_path()
    except Exception as error:  # offline: nothing to compare against
        pytest.skip(f"cannot fetch the walker from the Hub: {error}")


def test_matches_onnxruntime(onnx_file):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    net = FrozenMlp.from_onnx(onnx_file)
    rng = np.random.default_rng(0)
    for _ in range(8):
        obs = rng.normal(size=(1, OBS_DIM)).astype(np.float32)
        expected = session.run(None, {session.get_inputs()[0].name: obs})[0]
        actual = net(torch.from_numpy(obs)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_batched_and_frozen(onnx_file):
    net = FrozenMlp.from_onnx(onnx_file)
    out = net(torch.zeros(4096, OBS_DIM))
    assert out.shape == (4096, ACTION_DIM)
    assert not any(p.requires_grad for p in net.parameters())
