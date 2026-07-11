from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor


class MeanVariancePosterior:
    def __init__(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor | None,
        *,
        q_dim: int = -2,
    ) -> None:
        self.mean = mean
        self.variance = variance
        self.q_dim = int(q_dim)


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
    posts = []

    for X_chunk in X.split(chunk_size, dim=-2):
        posts.append(
            aggregate_perturbed_posterior(
                model=model,
                X=X_chunk,
                n_w=n_w,
                variance_mode=variance_mode,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **posterior_kwargs,
            )
        )

    if len(posts) == 0:
        raise ValueError("X must contain at least one point.")

    q_dim = posts[0].q_dim
    if any(post.q_dim != q_dim for post in posts[1:]):
        raise RuntimeError("Inconsistent posterior q dimensions across chunks.")

    means = [post.mean for post in posts]
    variances = [post.variance for post in posts]

    if all(variance is None for variance in variances):
        variance_out = None
    elif any(variance is None for variance in variances):
        raise RuntimeError("Posterior variance availability changed across chunks.")
    else:
        variance_out = torch.cat(variances, dim=q_dim)  # type: ignore[arg-type]

    return MeanVariancePosterior(
        mean=torch.cat(means, dim=q_dim),
        variance=variance_out,
        q_dim=q_dim,
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
            基本 shape は ``batch_shape x q x event_shape``。
        variance:
            摂動方向 n_w で集約した posterior variance。
        mean_per_w:
            摂動ごとの posterior mean。
            基本 shape は ``batch_shape x q x n_w x event_shape``。
        variance_per_w:
            摂動ごとの posterior variance。
        q_dim:
            ``mean`` 上で q 軸が置かれている非負の次元番号。
    """

    posterior: Any
    mean: Tensor
    variance: Tensor | None
    mean_per_w: Tensor
    variance_per_w: Tensor | None
    q_dim: int


def _infer_posterior_q_dim(
    posterior: Any,
    moment: Tensor,
    *,
    q: int,
    n_w: int,
) -> int:
    """Infer the q-like axis from posterior event metadata and tensor shape.

    Standard regression / binary / ordinal posteriors usually expose
    ``[..., q_like, m]``. Multi-output multiclass probability posteriors expose
    ``[..., q_like, m, C]``. The q-like axis is therefore not always ``-2``.
    """

    expected_q = int(q) * int(n_w)
    valid_sizes = {int(q), expected_q}

    try:
        event_shape = torch.Size(posterior.event_shape)
    except Exception:
        event_shape = torch.Size()

    if len(event_shape) > 0 and len(event_shape) <= moment.ndim:
        q_dim = moment.ndim - len(event_shape)
        if 0 <= q_dim < moment.ndim and int(moment.shape[q_dim]) in valid_sizes:
            return q_dim

    expected_candidates = [
        dim for dim, size in enumerate(moment.shape) if int(size) == expected_q
    ]
    if len(expected_candidates) == 1:
        return expected_candidates[0]

    q_candidates = [dim for dim, size in enumerate(moment.shape) if int(size) == int(q)]
    if len(q_candidates) == 1:
        return q_candidates[0]

    # Backward-supported fallbacks for the historical layouts.
    for q_dim in (moment.ndim - 2, moment.ndim - 3, moment.ndim - 1):
        if 0 <= q_dim < moment.ndim and int(moment.shape[q_dim]) in valid_sizes:
            return q_dim

    raise ValueError(
        "Could not infer the posterior q dimension. "
        f"X q={q}, n_w={n_w}, expected expanded q={expected_q}, "
        f"posterior.mean.shape={tuple(moment.shape)}, "
        f"posterior.event_shape={tuple(event_shape)}."
    )


def _canonicalize_perturbed_moment_shape(
    moment: Tensor | None,
    *,
    posterior: Any,
    q: int,
    n_w: int,
) -> tuple[Tensor | None, int | None]:
    """Canonicalize a posterior moment and return its q-axis index.

    Some single-output multiclass posteriors expose ``[..., q_like, 1, C]``.
    The singleton output dimension is removed so the public result remains
    ``[..., q_like, C]``. Multi-output multiclass tensors such as
    ``[..., q_like, m, C]`` are preserved.
    """

    if moment is None:
        return None, None

    q_dim = _infer_posterior_q_dim(
        posterior,
        moment,
        q=q,
        n_w=n_w,
    )

    if (
        moment.ndim >= q_dim + 3
        and int(moment.shape[q_dim + 1]) == 1
        and int(moment.shape[q_dim]) in (int(q), int(q) * int(n_w))
    ):
        moment = moment.squeeze(q_dim + 1)

    return moment, q_dim


def _canonicalize_variance(
    posterior: Any,
    *,
    q: int,
    n_w: int,
    q_dim: int,
    squeezed_output: bool,
) -> Tensor | None:
    if not hasattr(posterior, "variance"):
        return None

    variance, variance_q_dim = _canonicalize_perturbed_moment_shape(
        posterior.variance,
        posterior=posterior,
        q=q,
        n_w=n_w,
    )
    if variance is None or variance_q_dim is None:
        return None
    if variance_q_dim != q_dim:
        variance = variance.movedim(variance_q_dim, q_dim)
    if squeezed_output and q_dim == variance.ndim - 1:
        variance = variance.unsqueeze(-1)
    return variance


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
    posterior_transform: Any | None = None,
    strict: bool = False,
    **posterior_kwargs: Any,
) -> AggregatedPerturbedPosterior:
    """
    入力摂動 transform により q * n_w に展開された posterior を、
    元の q 点ごとの予測に戻す。

    回帰・binary・ordinal の基本 shape ``[..., q * n_w, m]`` に加えて、
    multiclass の ``[..., q * n_w, m, C]`` も扱う。posterior の
    ``event_shape`` を優先して q 軸を特定するため、出力軸やクラス軸を
    q 軸と誤認しない。

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
                E_w[Var[Y|w]] + Var_w[E[Y|w]]
            "mean_posterior":
                E_w[Var[Y|w]]
            "input_sensitivity":
                Var_w[E[Y|w]]
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

    q = int(X.shape[-2])
    expected_q = q * int(n_w)

    posterior = model.posterior(
        X,
        observation_noise=observation_noise,
        posterior_transform=posterior_transform,
        **posterior_kwargs,
    )

    mean, q_dim = _canonicalize_perturbed_moment_shape(
        posterior.mean,
        posterior=posterior,
        q=q,
        n_w=n_w,
    )
    if mean is None or q_dim is None:
        raise ValueError("posterior.mean is required for perturbation aggregation.")

    # A scalar-output posterior may expose ``[..., q_like]`` without an explicit
    # output dimension. Add one temporarily so the generic event-shape code can
    # aggregate it together with multi-output and multiclass posteriors.
    squeezed_output = False
    if q_dim == mean.ndim - 1:
        mean = mean.unsqueeze(-1)
        squeezed_output = True

    if mean.ndim < 2:
        raise ValueError(
            "posterior.mean must contain a q dimension and at least one event "
            f"dimension. Got mean.shape={tuple(mean.shape)}."
        )

    expanded_q = int(mean.shape[q_dim])

    # 摂動展開されていない場合
    if expanded_q == q:
        if strict:
            raise ValueError(
                "posterior appears not to be expanded by n_w. "
                f"X q={q}, posterior q={expanded_q}, n_w={n_w}."
            )

        variance = None
        if variance_mode != "none":
            variance = _canonicalize_variance(
                posterior,
                q=q,
                n_w=n_w,
                q_dim=q_dim,
                squeezed_output=squeezed_output,
            )

        mean_out = mean.squeeze(-1) if squeezed_output else mean
        variance_out = (
            variance.squeeze(-1)
            if squeezed_output and variance is not None
            else variance
        )
        mean_per_w = mean.unsqueeze(q_dim + 1)
        variance_per_w = (
            variance.unsqueeze(q_dim + 1) if variance is not None else None
        )
        if squeezed_output:
            mean_per_w = mean_per_w.squeeze(-1)
            if variance_per_w is not None:
                variance_per_w = variance_per_w.squeeze(-1)

        return AggregatedPerturbedPosterior(
            posterior=posterior,
            mean=mean_out,
            variance=variance_out,
            mean_per_w=mean_per_w,
            variance_per_w=variance_per_w,
            q_dim=q_dim,
        )

    if expanded_q != expected_q:
        raise ValueError(
            "posterior q dimension is inconsistent with X and n_w. "
            f"X q={q}, n_w={n_w}, expected posterior q={expected_q}, "
            f"but got posterior q={expanded_q}. "
            f"posterior.mean.shape={tuple(mean.shape)}, q_dim={q_dim}"
        )

    leading_shape = tuple(mean.shape[:q_dim])
    event_shape = tuple(mean.shape[q_dim + 1 :])
    perturbation_dim = q_dim + 1

    if layout == "point_major":
        mean_per_w = mean.reshape(
            *leading_shape,
            q,
            n_w,
            *event_shape,
        )
    elif layout == "perturbation_major":
        mean_per_w = mean.reshape(
            *leading_shape,
            n_w,
            q,
            *event_shape,
        ).transpose(q_dim, q_dim + 1)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    mean_agg = mean_per_w.mean(dim=perturbation_dim)

    variance_per_w = None
    variance_agg = None

    if variance_mode != "none":
        variance = _canonicalize_variance(
            posterior,
            q=q,
            n_w=n_w,
            q_dim=q_dim,
            squeezed_output=squeezed_output,
        )
        if variance is not None:
            if tuple(variance.shape) != tuple(mean.shape):
                raise ValueError(
                    "posterior variance shape must match posterior mean shape after "
                    f"canonicalization. mean={tuple(mean.shape)}, "
                    f"variance={tuple(variance.shape)}."
                )

            if layout == "point_major":
                variance_per_w = variance.reshape(
                    *leading_shape,
                    q,
                    n_w,
                    *event_shape,
                )
            else:
                variance_per_w = variance.reshape(
                    *leading_shape,
                    n_w,
                    q,
                    *event_shape,
                ).transpose(q_dim, q_dim + 1)

            if variance_mode == "total":
                variance_agg = (
                    variance_per_w.mean(dim=perturbation_dim)
                    + mean_per_w.var(dim=perturbation_dim, unbiased=False)
                )
            elif variance_mode == "mean_posterior":
                variance_agg = variance_per_w.mean(dim=perturbation_dim)
            elif variance_mode == "input_sensitivity":
                variance_agg = mean_per_w.var(
                    dim=perturbation_dim,
                    unbiased=False,
                )
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
        q_dim=q_dim,
    )
