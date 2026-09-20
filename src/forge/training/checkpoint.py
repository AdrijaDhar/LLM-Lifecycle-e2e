"""Save / restore a training run: model weights, optimizer state, step counter.

A checkpoint is a directory:
    <dir>/model.safetensors    flattened model parameters
    <dir>/optim.safetensors    flattened optimizer state (Adam moments, etc.)
    <dir>/state.json           step, config, rng, loss history pointer
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


def save(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    extra: dict | None = None,
) -> None:
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)

    mx.save_safetensors(str(d / "model.safetensors"), dict(tree_flatten(model.parameters())))
    mx.save_safetensors(str(d / "optim.safetensors"), dict(tree_flatten(optimizer.state)))
    (d / "state.json").write_text(json.dumps({"step": step, **(extra or {})}, indent=2))


def load(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> dict:
    d = Path(ckpt_dir)

    weights = list(mx.load(str(d / "model.safetensors")).items())
    model.update(tree_unflatten(weights))

    if optimizer is not None and (d / "optim.safetensors").exists():
        state = list(mx.load(str(d / "optim.safetensors")).items())
        optimizer.state = tree_unflatten(state)

    return json.loads((d / "state.json").read_text())
