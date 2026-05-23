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
    """
    非ガウス GP 用 fit 結果。

    Attributes:
        model:
            fit 済みモデル。
        mll:
            学習に使った MLL。
        losses:
            epoch ごとの平均 loss。
    """

    model: Any
    mll: MarginalLogLikelihood
    losses: list[float]


def _unwrap_mll(mll: MarginalLogLikelihood) -> MarginalLogLikelihood:
    """
    DeepApproximateMLL の場合は内側の base_mll を返す。
    """
    return getattr(mll, "base_mll", mll)


def _get_train_inputs_from_model(model: Any) -> tuple[Tensor, ...]:
    """
    モデルから training input を取得する。

    優先順位:
        1. transformed_train_inputs
        2. fit_train_inputs
        3. train_inputs
    """
    if hasattr(model, "transformed_train_inputs"):
        train_inputs = model.transformed_train_inputs
    elif hasattr(model, "fit_train_inputs"):
        train_inputs = model.fit_train_inputs
    elif hasattr(model, "train_inputs"):
        train_inputs = model.train_inputs
    else:
        raise AttributeError(
            "Could not find training inputs. Expected transformed_train_inputs, "
            "fit_train_inputs, or train_inputs."
        )

    if isinstance(train_inputs, Tensor):
        return (train_inputs,)

    if isinstance(train_inputs, list):
        return tuple(train_inputs)

    return tuple(train_inputs)


def _get_train_targets_from_model(model: Any) -> Tensor:
    """
    モデルから training target を取得する。

    優先順位:
        1. fit_train_targets
        2. train_targets
        3. train_Y
    """
    if hasattr(model, "fit_train_targets"):
        return model.fit_train_targets
    if hasattr(model, "train_targets"):
        return model.train_targets
    if hasattr(model, "train_Y"):
        return model.train_Y

    raise AttributeError(
        "Could not find training targets. Expected fit_train_targets, "
        "train_targets, or train_Y."
    )


def _prepare_targets_for_mll(y: Tensor) -> Tensor:
    """MLL に渡す target を必要に応じて 1D に整形する。"""
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
    """
    非ガウス GP 用の汎用 MLL fit helper。

    Args:
        mll:
            `model.make_mll()` で作成した MLL。
            通常は `VariationalELBO` または `DeepApproximateMLL`。
        train_inputs:
            明示的に渡す training inputs。
            省略時は `mll.model` または `mll.base_mll.model` から取得する。
        train_targets:
            明示的に渡す training targets。
            省略時は model から取得する。
        lr:
            Adam optimizer の学習率。
        num_epochs:
            epoch 数。
        batch_size:
            mini-batch size。`None` の場合は full-batch。
        shuffle:
            DataLoader で shuffle するかどうか。
        clip_grad_norm:
            勾配 norm clipping を使う場合の上限。
        verbose:
            True の場合、epoch ごとの loss を表示する。

    Returns:
        list[float]:
            epoch ごとの平均 loss。

    Notes:
        BoTorch の `fit_gpytorch_mll` は ExactGP では自然に使えますが、
        非ガウス SVGP / DeepGP では明示的な training loop の方が挙動を制御しやすいです。
        この helper は `fit_gpytorch_mll` 風に、MLL を渡して fit できるようにするためのものです。
    """
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

    dataset = TensorDataset(*train_inputs, train_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    mll.train()
    base_mll.train()
    gp_model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(mll.parameters(), lr=float(lr))
    losses: list[float] = []

    for epoch in range(int(num_epochs)):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            *xb_list, yb = batch
            xb_tuple = tuple(xb_list)

            optimizer.zero_grad(set_to_none=True)

            output = gp_model(*xb_tuple)
            loss = -mll(output, yb)

            # DeepApproximateMLL / VariationalELBO の loss が batch_shape を持つ場合に備える。
            loss = loss.mean()

            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    mll.parameters(),
                    max_norm=float(clip_grad_norm),
                )

            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)

        if verbose:
            print(f"[{epoch + 1:04d}/{int(num_epochs):04d}] loss={mean_loss:.6f}")

    mll.eval()
    base_mll.eval()
    gp_model.eval()
    likelihood.eval()

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
    """
    非ガウス GP model を fit する。

    Args:
        model:
            `make_mll()` を持つモデル。
            例: `PoissonGPModel`, `GammaGPModel`, `PoissonDeepGPModel`,
            `GammaDeepGPModel` など。
        mll:
            明示的に使う MLL。省略時は `model.make_mll()` を呼ぶ。
        lr:
            Adam optimizer の学習率。
        num_epochs:
            epoch 数。
        batch_size:
            mini-batch size。
        shuffle:
            DataLoader で shuffle するかどうか。
        clip_grad_norm:
            勾配 clipping の上限。
        verbose:
            True の場合、学習ログを表示する。

    Returns:
        FitResult:
            model, mll, losses を含む結果。
    """
    if mll is None:
        if not hasattr(model, "make_mll"):
            raise AttributeError(
                f"{model.__class__.__name__} does not have make_mll(). "
                "Pass mll explicitly or implement make_mll()."
            )
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

    return FitResult(
        model=model,
        mll=mll,
        losses=losses,
    )


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
    """
    BoTorch 風に MLL を渡して fit する entry point。

    Args:
        mll:
            学習する MLL。
        use_botorch_fit:
            True の場合は `botorch.fit.fit_gpytorch_mll(mll)` を使う。
            ExactGP 系では True が自然です。
            非ガウス SVGP / DeepGP では False を推奨します。
        lr:
            Adam optimizer の学習率。
        num_epochs:
            epoch 数。
        batch_size:
            mini-batch size。
        shuffle:
            DataLoader で shuffle するかどうか。
        clip_grad_norm:
            勾配 clipping の上限。
        verbose:
            True の場合、学習ログを表示する。

    Returns:
        use_botorch_fit=True:
            `fit_gpytorch_mll(mll)` の戻り値。
        use_botorch_fit=False:
            list[float] の loss history。
    """
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


# Poisson / Gamma で名前を分けたい場合の alias。
fit_poisson_gp = fit_non_gaussian_gp
fit_gamma_gp = fit_non_gaussian_gp
fit_poisson_mll = fit_non_gaussian_mll
fit_gamma_mll = fit_non_gaussian_mll


__all__ = [
    "FitResult",
    "fit_non_gaussian_gp",
    "fit_non_gaussian_mll",
    "fit_gpytorch_mll_like_botorch",
    "fit_poisson_gp",
    "fit_gamma_gp",
    "fit_poisson_mll",
    "fit_gamma_mll",
]
