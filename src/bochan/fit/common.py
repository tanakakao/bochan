from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def get_train_inputs_tensor(model: Any) -> Tensor:
    """Return the first training input tensor from a BoTorch / GPyTorch model."""
    if not hasattr(model, "train_inputs"):
        raise AttributeError("Could not find `train_inputs` on model.")

    train_inputs = model.train_inputs
    if isinstance(train_inputs, tuple):
        if len(train_inputs) == 0:
            raise AttributeError("`model.train_inputs` is an empty tuple.")
        return train_inputs[0]
    return train_inputs


def get_train_targets_tensor(model: Any) -> Tensor:
    """Return training targets from a BoTorch / GPyTorch model."""
    if hasattr(model, "train_targets"):
        return model.train_targets
    if hasattr(model, "train_Y"):
        return model.train_Y
    raise AttributeError("Could not find training targets from model.")


def get_fit_train_X(model: Any) -> Tensor:
    """
    Return the X tensor that should be passed to the model during fitting.

    Priority:
        1. raw_train_X:
            PCA / random-projection wrappers should receive raw X.
        2. train_X:
            Some custom wrappers keep the public training X here.
        3. train_inputs:
            Standard BoTorch / GPyTorch models.
    """
    if hasattr(model, "raw_train_X") and getattr(model, "raw_train_X") is not None:
        return model.raw_train_X

    if hasattr(model, "train_X") and getattr(model, "train_X") is not None:
        return model.train_X

    return get_train_inputs_tensor(model)


def get_fit_train_Y(model: Any) -> Tensor:
    """
    Return the Y tensor that should be used during fitting.

    Priority:
        1. train_Y:
            BoTorch-style wrappers often keep the original target here.
        2. train_targets:
            GPyTorch approximate models.
    """
    if hasattr(model, "train_Y") and getattr(model, "train_Y") is not None:
        return model.train_Y

    return get_train_targets_tensor(model)


def squeeze_single_output_target(y: Tensor) -> Tensor:
    """Convert [..., 1] target tensors to [...] while preserving multi-output targets."""
    if y.ndim > 1 and y.shape[-1] == 1:
        return y.squeeze(-1)
    return y


def view_single_output_target(y: Tensor) -> Tensor:
    """
    Convert single-output targets to one-dimensional targets.

    Multi-output targets are returned as-is.
    """
    if y.ndim > 1 and y.shape[-1] > 1:
        return y
    return y.view(-1)


def resolve_batch_size(train_X: Tensor, batch_size: Optional[int]) -> int:
    """Return a valid batch size, defaulting to full-batch over the n dimension."""
    if batch_size is None:
        return int(train_X.shape[-2])
    return int(batch_size)


def build_tensor_dataloader(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Build a TensorDataset/DataLoader pair using the model's training tensors."""
    resolved_batch_size = resolve_batch_size(train_X, batch_size)
    dataset = TensorDataset(train_X, train_Y)
    return DataLoader(dataset, batch_size=resolved_batch_size, shuffle=shuffle)


def move_batch_like(
    xb: Tensor,
    yb: Tensor,
    *,
    train_X: Tensor,
    train_Y: Tensor,
) -> tuple[Tensor, Tensor]:
    """Move mini-batch tensors to the same device/dtype as their source tensors."""
    xb = xb.to(device=train_X.device, dtype=train_X.dtype)
    yb = yb.to(device=train_Y.device, dtype=train_Y.dtype)
    return xb, yb


def set_mll_train_mode(mll: Any) -> None:
    """Set mll, model, and likelihood to train mode when available."""
    mll.train()
    if hasattr(mll, "model"):
        mll.model.train()
    if hasattr(mll, "likelihood"):
        mll.likelihood.train()


def set_mll_eval_mode(mll: Any) -> None:
    """Set mll, model, and likelihood to eval mode when available."""
    mll.eval()
    if hasattr(mll, "model"):
        mll.model.eval()
    if hasattr(mll, "likelihood"):
        mll.likelihood.eval()


def set_model_and_likelihood_train_mode(model: Any, likelihood: Any | None = None) -> None:
    """Set a fitting model and its likelihood to train mode."""
    model.train()
    if likelihood is not None:
        likelihood.train()
    elif hasattr(model, "likelihood"):
        model.likelihood.train()


def set_model_and_likelihood_eval_mode(model: Any, likelihood: Any | None = None) -> None:
    """Set a fitting model and its likelihood to eval mode."""
    model.eval()
    if likelihood is not None:
        likelihood.eval()
    elif hasattr(model, "likelihood"):
        model.likelihood.eval()


def get_likelihood_from_mll_or_model(mll: Any, model: Any | None = None) -> Any | None:
    """Resolve likelihood from mll first, then model."""
    if hasattr(mll, "likelihood"):
        return mll.likelihood
    if model is not None and hasattr(model, "likelihood"):
        return model.likelihood
    return None


def maybe_clip_grad_norm(parameters, clip_grad_norm: Optional[float]) -> None:
    """Clip gradients when clip_grad_norm is not None."""
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
