from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import MarginalLogLikelihood


@dataclass
class FitResult:
    """非ガウス GP 用 fit 結果。"""
    model: Any
    mll: MarginalLogLikelihood
    losses: list[float]


def _unwrap_mll(mll: MarginalLogLikelihood) -> MarginalLogLikelihood:
    return getattr(mll, "base_mll", mll)


def _get_train_inputs_from_model(model: Any) -> tuple[Tensor, ...]:
    if hasattr(model, "transformed_train_inputs"):
        train_inputs = model.transformed_train_inputs
    elif hasattr(model, "fit_train_inputs"):
        train_inputs = model.fit_train_inputs
    elif hasattr(model, "train_inputs"):
        train_inputs = model.train_inputs
    else:
        raise AttributeError("Could not find training inputs.")
    if isinstance(train_inputs, Tensor):
        return (train_inputs,)
    if isinstance(train_inputs, list):
        return tuple(train_inputs)
    return tuple(train_inputs)


def _get_train_targets_from_model(model: Any) -> Tensor:
    if hasattr(model, "fit_train_targets"):
        return model.fit_train_targets
    if hasattr(model, "train_targets"):
        return model.train_targets
    if hasattr(model, "train_Y"):
        return model.train_Y
    raise AttributeError("Could not find training targets.")


def _prepare_targets_for_mll(y: Tensor) -> Tensor:
    if y.ndim > 1 and y.shape[-1] == 1:
        return y.squeeze(-1)
    return y


def fit_non_gaussian_mll(
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
    """非ガウス GP 用の汎用 MLL fit helper。"""
    base_mll = _unwrap_mll(mll)
    gp_model = base_mll.model
    likelihood = base_mll.likelihood
    if train_inputs is None:
        train_inputs = _get_train_inputs_from_model(gp_model)
    elif isinstance(train_inputs, Tensor):
        train_inputs = (train_inputs,)
    else:
        train_inputs = tuple(train_inputs)
    if train_targets is None:
        train_targets = _get_train_targets_from_model(gp_model)
    train_targets = _prepare_targets_for_mll(train_targets)
    x0 = train_inputs[0]
    if batch_size is None:
        batch_size = x0.shape[-2]
    loader = DataLoader(TensorDataset(*train_inputs, train_targets), batch_size=batch_size, shuffle=shuffle)
    mll.train(); base_mll.train(); gp_model.train(); likelihood.train()
    optimizer = torch.optim.Adam(mll.parameters(), lr=float(lr))
    losses: list[float] = []
    for epoch in range(int(num_epochs)):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            *xb_list, yb = batch
            optimizer.zero_grad(set_to_none=True)
            output = gp_model(*tuple(xb_list))
            loss = -mll(output, yb).mean()
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=float(clip_grad_norm))
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)
        if verbose:
            print(f"[{epoch + 1:04d}/{int(num_epochs):04d}] loss={mean_loss:.6f}")
    mll.eval(); base_mll.eval(); gp_model.eval(); likelihood.eval()
    return losses


def fit_non_gaussian_gp(
    model: Any,
    *,
    mll: Optional[MarginalLogLikelihood] = None,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
) -> FitResult:
    """make_mll() を持つ非ガウス GP model を fit する。"""
    if mll is None:
        if not hasattr(model, "make_mll"):
            raise AttributeError(f"{model.__class__.__name__} does not have make_mll().")
        mll = model.make_mll()
    losses = fit_non_gaussian_mll(
        mll,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        clip_grad_norm=clip_grad_norm,
        verbose=verbose,
    )
    return FitResult(model=model, mll=mll, losses=losses)


def fit_gpytorch_mll_like_botorch(
    mll: MarginalLogLikelihood,
    *,
    use_botorch_fit: bool = False,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
):
    """BoTorch 風に MLL を渡して fit する entry point。"""
    if use_botorch_fit:
        return fit_gpytorch_mll(mll)
    return fit_non_gaussian_mll(
        mll,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        clip_grad_norm=clip_grad_norm,
        verbose=verbose,
    )


fit_negative_binomial_gp = fit_non_gaussian_gp
fit_negative_binomial_mll = fit_non_gaussian_mll

__all__ = [
    "FitResult",
    "fit_non_gaussian_gp",
    "fit_non_gaussian_mll",
    "fit_gpytorch_mll_like_botorch",
    "fit_negative_binomial_gp",
    "fit_negative_binomial_mll",
]
