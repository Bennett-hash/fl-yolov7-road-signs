"""The corrected averaging averages; the 2023 function returns the first client."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fedyolo.averaging import average_checkpoints, average_state_dicts
from fedyolo.original import average_weights as original_average_weights


def toy_model(seed: int) -> nn.Module:
    """A small network with the layer types YOLOv7 has: convolutions and BatchNorm."""
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.SiLU(), nn.Conv2d(4, 2, 1))
    # Give BatchNorm different running statistics on each client, as training would.
    m.train()
    m(torch.randn(8, 3, 6, 6) * (seed + 1))
    m.eval()
    return m.half()


def toy_checkpoint(seed: int) -> dict:
    """The shape of a best.pt after YOLOv7's strip_optimizer."""
    return {
        "epoch": -1,
        "best_fitness": np.array([0.1 * seed]),
        "training_results": None,
        "model": toy_model(seed),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "wandb_id": None,
    }


@pytest.fixture
def checkpoints(tmp_path):
    paths = []
    for seed in (1, 2, 3):
        p = tmp_path / f"client_{seed}.pt"
        torch.save(toy_checkpoint(seed), p)
        paths.append(p)
    return paths


def states(paths):
    return [torch.load(p, weights_only=False)["model"].state_dict() for p in paths]


def true_mean(sds, key):
    return sum(sd[key].double() for sd in sds) / len(sds)


def test_fixed_averaging_is_the_mean_of_every_float_tensor(checkpoints):
    clients = states(checkpoints)
    out = average_checkpoints(checkpoints)["model"].state_dict()
    floats = [k for k in out if torch.is_floating_point(out[k])]
    assert any("running_mean" in k for k in floats), "BatchNorm statistics must be included"
    for key in floats:
        assert out[key].dtype == torch.float16
        assert torch.allclose(out[key].double(), true_mean(clients, key), atol=2e-3), key


def test_fixed_averaging_keeps_counters_from_the_first_client(checkpoints):
    out = average_checkpoints(checkpoints)["model"].state_dict()
    key = next(k for k in out if k.endswith("num_batches_tracked"))
    assert torch.equal(out[key], states(checkpoints)[0][key])


def test_fixed_averaging_leaves_a_checkpoint_train_py_can_start_from(checkpoints):
    out = average_checkpoints(checkpoints)
    assert out["epoch"] == -1  # train.py then starts at epoch 0
    assert out["optimizer"] is None
    assert set(out) == set(toy_checkpoint(0))


def test_fixed_averaging_does_not_change_the_clients(checkpoints):
    before = states(checkpoints)
    average_checkpoints(checkpoints)
    for a, b in zip(before, states(checkpoints), strict=True):
        for key in a:
            assert torch.equal(a[key], b[key])


def test_averaging_one_client_returns_it(checkpoints):
    out = average_checkpoints(checkpoints[:1])["model"].state_dict()
    for key, value in states(checkpoints[:1])[0].items():
        assert torch.equal(out[key], value)


def test_the_2023_function_returns_the_first_client_unchanged(checkpoints, monkeypatch):
    """The error behind experiments 2 to 5: nothing is averaged."""
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    clients = states(checkpoints)
    out = original_average_weights([str(p) for p in checkpoints])["model"].state_dict()
    for key in out:
        assert torch.equal(out[key], clients[0][key]), key
    conv = "0.weight"
    assert not torch.allclose(out[conv].double(), true_mean(clients, conv), atol=1e-3)


def test_state_dicts_must_match():
    a = nn.Linear(2, 2).state_dict()
    b = nn.Linear(2, 3).state_dict()
    with pytest.raises(ValueError, match="shapes differ"):
        average_state_dicts([a, b])
    with pytest.raises(ValueError, match="same keys"):
        average_state_dicts([a, nn.Conv2d(1, 1, 1).state_dict() | {"extra": torch.zeros(1)}])
    with pytest.raises(ValueError, match="nothing"):
        average_state_dicts([])


class WithAnchors(nn.Module):
    """A layer with buffers named as in YOLOv7's Detect layer (model.77.anchors, model.77.anchor_grid)."""

    def __init__(self, seed: int, anchors: torch.Tensor):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 2, 1)
        self.register_buffer("anchors", anchors.clone())
        self.register_buffer("anchor_grid", anchors.clone().view(1, -1, 1, 1, 2))


def save_with_anchors(tmp_path, anchors_per_client):
    paths = []
    for seed, anchors in enumerate(anchors_per_client, 1):
        p = tmp_path / f"anchors_{seed}.pt"
        torch.save(toy_checkpoint(seed) | {"model": WithAnchors(seed, anchors).half()}, p)
        paths.append(p)
    return paths


def test_identical_anchors_are_kept(tmp_path):
    anchors = torch.tensor([[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]])
    paths = save_with_anchors(tmp_path, [anchors, anchors, anchors])
    out = average_checkpoints(paths)["model"].state_dict()
    first = states(paths)[0]
    assert torch.equal(out["anchors"], first["anchors"])
    assert torch.equal(out["anchor_grid"], first["anchor_grid"])


def test_different_anchors_are_refused(tmp_path):
    anchors = torch.tensor([[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]])
    paths = save_with_anchors(tmp_path, [anchors, anchors, anchors * 1.5])
    with pytest.raises(ValueError, match=r"anchors: the clients' anchors differ \(YOLOv7's autoanchor changed them\)"):
        average_checkpoints(paths)


@pytest.mark.parametrize("key", ["optimizer", "ema"])
def test_a_checkpoint_that_is_not_stripped_is_refused(tmp_path, checkpoints, key):
    unstripped = tmp_path / "last.pt"
    torch.save(toy_checkpoint(4) | {key: {"state": {}}}, unstripped)
    with pytest.raises(ValueError, match="pass the stripped best.pt"):
        average_checkpoints([*checkpoints, unstripped])


def test_epoch_is_always_reset(tmp_path):
    p = tmp_path / "epoch5.pt"
    torch.save(toy_checkpoint(1) | {"epoch": 5}, p)
    assert average_checkpoints([p, p])["epoch"] == -1
