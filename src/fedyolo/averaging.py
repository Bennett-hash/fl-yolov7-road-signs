"""Averaging YOLOv7 checkpoints across clients.

The paper (Section 2.2) aggregates the clients' weights with "a custom
function that calculates the arithmetic mean of all subset weights". This is
that function, written so that it does what the paper says. The version the
2023 notebooks used is in :mod:`fedyolo.original`.

Every client trains on the same number of images, so the plain mean is also
the sample-weighted mean of federated averaging (McMahan et al., 2017).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from os import PathLike

import torch


def average_state_dicts(state_dicts: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Return the element-wise mean of several state dicts with the same keys.

    Every floating-point tensor is averaged: the weights, and BatchNorm's
    running mean and variance. The mean is taken in float64 and cast back to
    each tensor's own type, so half-precision checkpoints stay half precision.
    Integer tensors, such as BatchNorm's ``num_batches_tracked``, are counters
    rather than parameters; they are taken from the first state dict.
    """
    if not state_dicts:
        raise ValueError("nothing to average")
    keys = list(state_dicts[0])
    for sd in state_dicts[1:]:
        if list(sd) != keys:
            raise ValueError("the state dicts do not have the same keys")
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        first = state_dicts[0][key]
        for sd in state_dicts[1:]:
            if sd[key].shape != first.shape:
                raise ValueError(f"{key}: shapes differ ({tuple(first.shape)} and {tuple(sd[key].shape)})")
        # The anchors (YOLOv7's "anchors" and "anchor_grid") are part of the
        # model's definition, not learned weights. They are the same for every
        # client unless YOLOv7's autoanchor replaced them, and a mean of
        # different anchors would fit none of the clients.
        if "anchor" in key and any(not torch.equal(sd[key], first) for sd in state_dicts[1:]):
            raise ValueError(f"{key}: the clients' anchors differ (YOLOv7's autoanchor changed them)")
        if torch.is_floating_point(first):
            total = torch.zeros(first.shape, dtype=torch.float64)
            for sd in state_dicts:
                total += sd[key].detach().to("cpu", torch.float64)
            out[key] = (total / len(state_dicts)).to(first.dtype)
        else:
            out[key] = first.detach().clone()
    return out


def load_checkpoint(path: str | PathLike[str]) -> dict:
    """Load a YOLOv7 checkpoint, which pickles the model object itself.

    ``weights_only=False`` is needed for that, and is safe only for files you
    made yourself.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def average_checkpoints(paths: Sequence[str | PathLike[str]]) -> dict:
    """Average the models in several YOLOv7 checkpoints.

    Returns a checkpoint in the same form, ready for ``train.py --weights``.
    The model is a copy of the first client's with the averaged state loaded
    into it. The other entries are combined as the 2023 notebooks combined
    them: the values that are not ``None`` are added up and divided by the
    number of clients (all of them, including any whose value is ``None``),
    and an entry no client has a value for stays ``None``. ``epoch`` is set
    to -1, so that ``train.py`` starts its schedule afresh.

    The inputs must be stripped checkpoints, as ``train.py`` leaves its
    ``best.pt``: one with an ``optimizer`` or ``ema`` (a ``last.pt`` of a
    run that has not finished) is refused.
    """
    paths = list(paths)
    ckpts = [load_checkpoint(p) for p in paths]
    if not ckpts:
        raise ValueError("nothing to average")
    for path, c in zip(paths, ckpts, strict=True):
        if c.get("optimizer") is not None or c.get("ema") is not None:
            raise ValueError(f"{path} still has its optimizer or EMA model; pass the stripped best.pt")

    model = copy.deepcopy(ckpts[0]["model"])
    model.load_state_dict(average_state_dicts([c["model"].state_dict() for c in ckpts]))

    out: dict = {}
    for key in ckpts[0]:
        if key == "model":
            out[key] = model
            continue
        values = [c[key] for c in ckpts if c.get(key) is not None]
        out[key] = sum(values) / len(ckpts) if values else None
    out["epoch"] = -1
    return out
