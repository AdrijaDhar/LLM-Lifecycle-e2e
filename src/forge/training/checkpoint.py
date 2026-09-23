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


def load_resized(ckpt_dir: str | Path, model: nn.Module) -> dict:
    """Like `load`, but tolerates a vocab-size change (e.g. new chat special
    tokens added after pretraining grew the embedding table).

    `model` must already be constructed at the NEW (larger) vocab size, so its
    freshly-initialised `tok_emb`/`lm_head` rows are the right shape and already
    randomly initialised. For any saved weight whose shape doesn't match, we
    keep the old rows as-is and keep the model's fresh random init for the rest
    - i.e. old token ids keep their trained embedding, new ones start random.
    """
    d = Path(ckpt_dir)
    saved = dict(mx.load(str(d / "model.safetensors")))
    current = dict(tree_flatten(model.parameters()))

    for name, old in saved.items():
        new = current.get(name)
        if new is None or old.shape == new.shape:
            continue
        if old.ndim != 1 or new.ndim != 1:
            if old.shape[1:] != new.shape[1:] or old.shape[0] > new.shape[0]:
                raise ValueError(f"cannot resize {name}: {old.shape} -> {new.shape}")
        n_old = old.shape[0]
        saved[name] = mx.concatenate([old, new[n_old:]], axis=0)
        print(f"checkpoint.load_resized: {name} {tuple(old.shape)} -> {tuple(saved[name].shape)}")

    model.update(tree_unflatten(list(saved.items())))
    return json.loads((d / "state.json").read_text())
