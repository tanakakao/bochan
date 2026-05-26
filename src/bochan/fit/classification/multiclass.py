from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from gpytorch.mlls import MarginalLogLikelihood


@dataclass
class ClassificationFitResult:
    """分類 GP 用 fit 結果。"""
    model: Any
    mll: MarginalLogLikelihood
    losses: list[float]


def _get_train_inputs_from_model(model: Any) -> tuple[Tensor, ...]:
    if hasattr(model, "fit_train_inputs"):
        x = model.fit_train_inputs
    elif hasattr(model, "transformed_train_inputs"):
        x = model.transformed_train_inputs
    elif hasattr(model, "train_inputs"):
        x = model.train_inputs
    else:
        raise AttributeError("Could not find train_inputs / transformed_train_inputs.")
    if isinstance(x, Tensor):
        return (x,)
    return tuple(x)


def _get_train_targets_from_model(model: Any) -> Tensor:
    if hasattr(model, "fit_train_targets"):
        return model.fit_train_targets
    if hasattr(model, "train_targets"):
        return model.train_targets
    if hasattr(model, "train_Y"):
        return model.train_Y
    raise AttributeError("Could not find train_targets.")


def fit_classification_mll(
    mll: MarginalLogLikelihood,
    *,
    train_inputs: Optional[Sequence[Tensor] | Tensor] = None,
    train_targets: Optional[Tensor] = None,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
) -> list[float]:
    """VariationalELBO / DeepApproximateMLL 用の分類 GP fit helper。"""
    base_mll = getattr(mll, "base_mll", mll)
    model = base_mll.model
    likelihood = base_mll.likelihood

    if train_inputs is None:
        train_inputs = _get_train_inputs_from_model(model)
    elif isinstance(train_inputs, Tensor):
        train_inputs = (train_inputs,)
    else:
        train_inputs = tuple(train_inputs)

    if train_targets is None:
        train_targets = _get_train_targets_from_model(model)
    if train_targets.ndim > 1 and train_targets.shape[-1] == 1:
        train_targets = train_targets.squeeze(-1)
    train_targets = train_targets.long()

    if batch_size is None:
        batch_size = train_inputs[0].shape[-2]

    dataset = TensorDataset(*train_inputs, train_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    mll.train()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(mll.parameters(), lr=float(lr))
    losses: list[float] = []

    for epoch in range(int(num_epochs)):
        total = 0.0
        n_batch = 0
        for batch in loader:
            *xb, yb = batch
            optimizer.zero_grad(set_to_none=True)
            output = model(*tuple(xb))
            loss = -mll(output, yb).mean()
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=float(clip_grad_norm))
            optimizer.step()
            total += float(loss.detach().cpu())
            n_batch += 1
        mean_loss = total / max(n_batch, 1)
        losses.append(mean_loss)
        if verbose:
            print(f"[{epoch + 1:04d}/{int(num_epochs):04d}] loss={mean_loss:.6f}")

    mll.eval()
    model.eval()
    likelihood.eval()
    return losses


def fit_classification_gp(
    model: Any,
    *,
    mll: Optional[MarginalLogLikelihood] = None,
    **kwargs: Any,
) -> ClassificationFitResult:
    """`model.make_mll()` を使って分類 GP を fit する。"""
    if mll is None:
        if not hasattr(model, "make_mll"):
            raise AttributeError(f"{model.__class__.__name__} does not have make_mll().")
        mll = model.make_mll()
    losses = fit_classification_mll(mll, **kwargs)
    return ClassificationFitResult(model=model, mll=mll, losses=losses)


fit_multiclass_gp = fit_classification_gp
fit_multiclass_mll = fit_classification_mll


__all__ = [
    "ClassificationFitResult",
    "fit_classification_gp",
    "fit_classification_mll",
    "fit_multiclass_gp",
    "fit_multiclass_mll",
]
