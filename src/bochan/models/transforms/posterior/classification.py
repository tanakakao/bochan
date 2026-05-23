from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from torch import Tensor

class MeanVariancePosterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance

def aggregate_perturbed_posterior_chunked(
    model,
    X: torch.Tensor,
    *,
    n_w: int,
    chunk_size: int = 512,
    variance_mode: str = "total",
    observation_noise: bool = False,
    posterior_transform=None,
    **posterior_kwargs,
):
    means = []
    variances = []

    for X_chunk in X.split(chunk_size, dim=-2):
        post = aggregate_perturbed_posterior(
            model=model,
            X=X_chunk,
            n_w=n_w,
            variance_mode=variance_mode,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **posterior_kwargs,
        )
        means.append(post.mean)
        variances.append(post.variance)

    return MeanVariancePosterior(
        mean=torch.cat(means, dim=-2),
        variance=torch.cat(variances, dim=-2),
    )

@dataclass
class AggregatedPerturbedPosterior:
    """
    入力摂動 posterior を元の X ごとに集約した結果。

    Attributes:
        posterior:
            model.posterior(X) の元の posterior。
        mean:
            摂動方向 n_w で平均した posterior mean。
            shape: [..., q, m]
        variance:
            摂動方向 n_w で集約した posterior variance。
            shape: [..., q, m]
        mean_per_w:
            摂動ごとの posterior mean。
            shape: [..., q, n_w, m]
        variance_per_w:
            摂動ごとの posterior variance。
            shape: [..., q, n_w, m]
    """

    posterior: Any
    mean: Tensor
    variance: Optional[Tensor]
    mean_per_w: Tensor
    variance_per_w: Optional[Tensor]


def aggregate_perturbed_posterior(
    model: Any,
    X: Tensor,
    n_w: int,
    *,
    layout: Literal["point_major", "perturbation_major"] = "point_major",
    variance_mode: Literal[
        "total",
        "mean_posterior",
        "input_sensitivity",
        "none",
    ] = "total",
    observation_noise: bool = False,
    posterior_transform: Optional[Any] = None,
    strict: bool = False,
    **posterior_kwargs: Any,
) -> AggregatedPerturbedPosterior:
    """
    入力摂動 transform により q * n_w に展開された posterior を、
    元の q 点ごとの予測に戻す。

    想定する posterior.mean の基本 shape は [..., q * n_w, m]。
    これを [..., q, n_w, m] に reshape し、n_w 方向に集約する。

    Args:
        model:
            BoTorch model または posterior(X) を持つモデル。
        X:
            予測したい入力。
            shape: [..., q, d] または [q, d]
        n_w:
            各点あたりの摂動数。
        layout:
            摂動後の点の並び方。

            "point_major":
                [x0_w0, x0_w1, ..., x0_w{n_w-1},
                 x1_w0, x1_w1, ..., x1_w{n_w-1}, ...]

            "perturbation_major":
                [x0_w0, x1_w0, ..., x{q-1}_w0,
                 x0_w1, x1_w1, ..., x{q-1}_w1, ...]

        variance_mode:
            variance の集約方法。

            "total":
                全分散の法則。
                E_w[Var[Y|w]] + Var_w[E[Y|w]]
                ロバスト予測として一番自然。

            "mean_posterior":
                E_w[Var[Y|w]]
                各摂動 posterior variance の平均のみ。

            "input_sensitivity":
                Var_w[E[Y|w]]
                入力摂動による平均予測のばらつきのみ。

            "none":
                variance を返さない。

        observation_noise:
            model.posterior に渡す observation_noise。

        posterior_transform:
            model.posterior に渡す posterior_transform。

        strict:
            True の場合、posterior の q 次元が q * n_w でないとエラー。
            False の場合、すでに展開されていない posterior はそのまま返す。

        **posterior_kwargs:
            model.posterior に追加で渡す引数。

    Returns:
        AggregatedPerturbedPosterior
    """

    if X.ndim < 2:
        raise ValueError(
            f"X must have shape [..., q, d]. Got X.shape={tuple(X.shape)}."
        )

    if n_w <= 0:
        raise ValueError(f"n_w must be positive. Got n_w={n_w}.")

    q = X.shape[-2]

    posterior = model.posterior(
        X,
        observation_noise=observation_noise,
        posterior_transform=posterior_transform,
        **posterior_kwargs,
    )

    mean = posterior.mean

    # BoTorch posterior は通常 [..., q, m]。
    # ただし custom posterior で [..., q] の場合に備えて最後に output 次元を足す。
    squeezed_output = False
    if mean.ndim >= 1 and mean.shape[-1] == q * n_w:
        mean = mean.unsqueeze(-1)
        squeezed_output = True
    elif mean.ndim >= 1 and mean.shape[-1] == q:
        mean = mean.unsqueeze(-1)
        squeezed_output = True

    if mean.ndim < 2:
        raise ValueError(
            f"posterior.mean must have shape [..., q, m]. "
            f"Got mean.shape={tuple(mean.shape)}."
        )

    expanded_q = mean.shape[-2]
    expected_q = q * n_w

    # 摂動展開されていない場合
    if expanded_q == q:
        if strict:
            raise ValueError(
                f"posterior appears not to be expanded by n_w. "
                f"X q={q}, posterior q={expanded_q}, n_w={n_w}."
            )

        variance = None
        if variance_mode != "none" and hasattr(posterior, "variance"):
            variance = posterior.variance
            if squeezed_output and variance.ndim >= 1 and variance.shape[-1] == q:
                variance = variance.unsqueeze(-1)

        return AggregatedPerturbedPosterior(
            posterior=posterior,
            mean=mean.squeeze(-1) if squeezed_output else mean,
            variance=variance.squeeze(-1) if squeezed_output and variance is not None else variance,
            mean_per_w=mean.unsqueeze(-2).squeeze(-1) if squeezed_output else mean.unsqueeze(-2),
            variance_per_w=(
                variance.unsqueeze(-2).squeeze(-1)
                if squeezed_output and variance is not None
                else variance.unsqueeze(-2)
                if variance is not None
                else None
            ),
        )

    if expanded_q != expected_q:
        raise ValueError(
            f"posterior q dimension is inconsistent with X and n_w. "
            f"X q={q}, n_w={n_w}, expected posterior q={expected_q}, "
            f"but got posterior q={expanded_q}. "
            f"posterior.mean.shape={tuple(mean.shape)}"
        )

    # mean: [..., q * n_w, m]
    leading_shape = mean.shape[:-2]
    m = mean.shape[-1]

    if layout == "point_major":
        # [..., q * n_w, m] -> [..., q, n_w, m]
        mean_per_w = mean.reshape(*leading_shape, q, n_w, m)

    elif layout == "perturbation_major":
        # [..., n_w, q, m] -> [..., q, n_w, m]
        mean_per_w = mean.reshape(*leading_shape, n_w, q, m).transpose(-3, -2)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    mean_agg = mean_per_w.mean(dim=-2)

    variance_per_w = None
    variance_agg = None

    if variance_mode != "none" and hasattr(posterior, "variance"):
        variance = posterior.variance

        if squeezed_output and variance.ndim >= 1 and variance.shape[-1] == expected_q:
            variance = variance.unsqueeze(-1)

        if layout == "point_major":
            variance_per_w = variance.reshape(*leading_shape, q, n_w, m)
        else:
            variance_per_w = variance.reshape(*leading_shape, n_w, q, m).transpose(-3, -2)

        if variance_mode == "total":
            # Var[Y] = E_w[Var[Y|w]] + Var_w[E[Y|w]]
            variance_agg = (
                variance_per_w.mean(dim=-2)
                + mean_per_w.var(dim=-2, unbiased=False)
            )

        elif variance_mode == "mean_posterior":
            variance_agg = variance_per_w.mean(dim=-2)

        elif variance_mode == "input_sensitivity":
            variance_agg = mean_per_w.var(dim=-2, unbiased=False)

        else:
            raise ValueError(f"Unknown variance_mode: {variance_mode}")

    if squeezed_output:
        mean_agg = mean_agg.squeeze(-1)
        mean_per_w = mean_per_w.squeeze(-1)

        if variance_agg is not None:
            variance_agg = variance_agg.squeeze(-1)
        if variance_per_w is not None:
            variance_per_w = variance_per_w.squeeze(-1)

    return AggregatedPerturbedPosterior(
        posterior=posterior,
        mean=mean_agg,
        variance=variance_agg,
        mean_per_w=mean_per_w,
        variance_per_w=variance_per_w,
    )