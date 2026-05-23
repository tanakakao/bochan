from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor
from botorch.models.model import Model


def _align_like(t: Tensor, ref: Tensor) -> Tensor:
    """
    t を ref と同じ shape に揃える。

    主に noise sigma を mean_like に合わせるための helper。

    対応例:
        t:   (*batch, q_like)
        ref: (*batch, q_like, 1)

        t:   (*batch, q_like, 1)
        ref: (*batch, q_like, 1)

        t:   (*batch, q)
        ref: (*batch, q*n_w, 1)
        -> n_w が推定できる場合は repeat_interleave する。
    """
    if t.shape == ref.shape:
        return t

    # ref が最後に singleton output dim を持つ場合
    if ref.ndim >= 1 and ref.shape[-1] == 1:
        ref_no_last = ref.squeeze(-1)

        if t.shape == ref_no_last.shape:
            return t.unsqueeze(-1)

        if t.ndim >= 1 and t.shape[-1] == 1 and t.squeeze(-1).shape == ref_no_last.shape:
            return t

        # q -> q*n_w に拡張する保険
        # e.g. t=[B,q], ref=[B,q*n_w,1]
        if (
            t.ndim == ref_no_last.ndim
            and t.shape[:-1] == ref_no_last.shape[:-1]
            and ref_no_last.shape[-1] % t.shape[-1] == 0
        ):
            n_w = ref_no_last.shape[-1] // t.shape[-1]
            return t.repeat_interleave(n_w, dim=-1).unsqueeze(-1)

        # e.g. t=[B,q,1], ref=[B,q*n_w,1]
        if (
            t.ndim == ref.ndim
            and t.shape[-1] == 1
            and t.shape[:-2] == ref.shape[:-2]
            and ref.shape[-2] % t.shape[-2] == 0
        ):
            n_w = ref.shape[-2] // t.shape[-2]
            return t.repeat_interleave(n_w, dim=-2)

    if t.ndim >= 2 and t.transpose(-1, -2).shape == ref.shape:
        return t.transpose(-1, -2)

    if t.numel() == ref.numel():
        return t.reshape_as(ref)

    try:
        return t.expand_as(ref)
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot align tensor shape {tuple(t.shape)} to ref shape {tuple(ref.shape)}."
        ) from e


def _flatten_X(X: Tensor) -> tuple[Tensor, torch.Size]:
    prefix = X.shape[:-1]
    Xf = X.reshape(-1, X.shape[-1])
    return Xf, prefix


def _prod(shape: Sequence[int] | torch.Size) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _normalize_class_probs_shape(
    probs: Tensor,
    X: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """
    class_probs の shape を (*batch, q_like, K) に正規化する。

    X:
        shape = (*batch, q, d)

    q_like:
        通常は q。
        InputPerturbation ありでは q*n_w。

    重要:
        ここでは q_like を q に戻さない。
        objective 側で q*n_w -> q に集約する。
    """
    if probs.ndim < 2:
        raise RuntimeError(
            f"class probabilities must have at least 2 dims [..., K], "
            f"got shape={tuple(probs.shape)}."
        )

    # Some implementations return [..., q_like, 1, K].
    if probs.ndim >= 3 and probs.shape[-2] == 1:
        probs = probs.squeeze(-2)

    K = int(probs.shape[-1])
    batch_shape = tuple(X.shape[:-2])
    target_ndim = len(batch_shape) + 2  # batch + q_like + K

    out = probs

    # SAAS / MC / model batch などの余分な leading dims を平均で落とす。
    # X batch shape は保護する。
    while out.ndim > target_ndim:
        prefix = tuple(out.shape[:-2])

        reduce_dim = 0
        if len(batch_shape) > 0 and len(prefix) >= len(batch_shape):
            match_start = None
            max_start = len(prefix) - len(batch_shape)
            for s in range(max_start + 1):
                if tuple(prefix[s : s + len(batch_shape)]) == batch_shape:
                    match_start = s
                    break

            if match_start is not None:
                protected = set(range(match_start, match_start + len(batch_shape)))
                extra_dims = [i for i in range(len(prefix)) if i not in protected]
                if not extra_dims:
                    break
                reduce_dim = extra_dims[0]

        out = out.mean(dim=reduce_dim)

    # すでに (*batch, q_like, K)
    if out.ndim == target_ndim and tuple(out.shape[: len(batch_shape)]) == batch_shape:
        out = out.clamp_min(eps)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)

    # batch なし: [q_like, K]
    if len(batch_shape) == 0 and out.ndim == 2:
        out = out.clamp_min(eps)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)

    # 最後の保険:
    #   probs.numel = prod(batch_shape) * q_like * K
    # とみなして reshape する。
    batch_prod = _prod(batch_shape)
    total_without_K = out.numel() // K

    if total_without_K % batch_prod != 0:
        raise RuntimeError(
            "Cannot reshape class probabilities to (*batch, q_like, K). "
            f"probs.shape={tuple(probs.shape)}, reduced.shape={tuple(out.shape)}, "
            f"X.shape={tuple(X.shape)}, K={K}."
        )

    q_like = total_without_K // batch_prod
    out = out.reshape(*batch_shape, q_like, K)
    out = out.clamp_min(eps)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x.pow(2)) / math.sqrt(2.0 * math.pi)


def get_noise_sigma(
    model: Model,
    X: Tensor,
    *,
    mean_like: Tensor,
    default_sigma: float = 0.0,
) -> Tensor:
    """
    hetero model であれば predict_noise_std を優先する。
    """
    if hasattr(model, "predict_noise_std"):
        sigma = model.predict_noise_std(X)
        return _align_like(sigma, mean_like)

    if hasattr(model, "predict_noise_var"):
        sigma = model.predict_noise_var(X).clamp_min(1e-12).sqrt()
        return _align_like(sigma, mean_like)

    if hasattr(model, "predict_noise_logvar"):
        sigma = model.predict_noise_logvar(X).exp().clamp_min(1e-12).sqrt()
        return _align_like(sigma, mean_like)

    sigma_val = torch.as_tensor(
        default_sigma, device=mean_like.device, dtype=mean_like.dtype
    )
    return sigma_val * torch.ones_like(mean_like)


def get_class_probs(model: Model, X: Tensor) -> Tensor:
    """
    ordinal class probabilities を取得する。

    X:
        shape = (*batch, q, d) または (..., d)

    Returns:
        Tensor:
            shape = (*batch, q_like, K)

    Notes:
        InputPerturbation ありでは q_like = q*n_w になり得る。
        ここでは q_like を q に戻さない。
        acquisition 側の objective で q*n_w -> q に戻す。
    """
    # t_batch_mode_transform 後の acquisition では通常 X=[B,q,d]。
    # まずはこの shape のまま model.class_probs(X) を呼び、
    # model 側の input_transform / InputPerturbation の batch 構造を保持する。
    if hasattr(model, "class_probs") and callable(getattr(model, "class_probs")):
        try:
            probs = model.class_probs(X)
            return _normalize_class_probs_shape(probs, X)
        except Exception:
            # 一部 model.class_probs が 2D 入力しか受けない場合の fallback。
            pass

    # fallback: flatten して呼ぶ。ただし reshape では q*n_w を保持する。
    Xf, _ = _flatten_X(X)

    if hasattr(model, "class_probs") and callable(getattr(model, "class_probs")):
        probs = model.class_probs(Xf)
        return _normalize_class_probs_shape(probs, X)

    # posterior + ordinal likelihood fallback
    posterior = model.posterior(X)

    ordinal_likelihood = getattr(model, "ordinal_likelihood", None)
    if ordinal_likelihood is None:
        ordinal_likelihood = getattr(model, "likelihood", None)

    if ordinal_likelihood is None:
        raise AttributeError(
            "Could not obtain class probabilities. Expected model.class_probs(X), "
            "model.ordinal_likelihood, or model.likelihood."
        )

    if hasattr(ordinal_likelihood, "marginal_class_probs"):
        probs = ordinal_likelihood.marginal_class_probs(posterior.distribution)
    elif hasattr(ordinal_likelihood, "class_probs_from_posterior"):
        probs = ordinal_likelihood.class_probs_from_posterior(posterior)
    else:
        raise AttributeError(
            "ordinal_likelihood must expose marginal_class_probs(distribution) "
            "or class_probs_from_posterior(posterior)."
        )

    return _normalize_class_probs_shape(probs, X)


def _get_utility_values(
    probs: Tensor,
    utility_values: Optional[Sequence[float] | Tensor] = None,
) -> Tensor:
    K = probs.shape[-1]
    if utility_values is None:
        u = torch.arange(K, device=probs.device, dtype=probs.dtype)
    else:
        u = torch.as_tensor(utility_values, device=probs.device, dtype=probs.dtype)
        if u.numel() != K:
            raise ValueError(
                f"utility_values must have length {K}, got {u.numel()}."
            )

    view_shape = (1,) * (probs.ndim - 1) + (K,)
    return u.view(*view_shape)


def ordinal_stats(
    probs: Tensor,
    *,
    utility_values: Optional[Sequence[float] | Tensor] = None,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    u = _get_utility_values(probs, utility_values=utility_values)

    mean_u = (probs * u).sum(dim=-1)
    second_u = (probs * (u ** 2)).sum(dim=-1)
    var_u = (second_u - mean_u.pow(2)).clamp_min(0.0)

    probs_clamped = probs.clamp_min(eps)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)

    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    if probs.shape[-1] >= 2:
        margin = sorted_probs[..., 0] - sorted_probs[..., 1]
    else:
        margin = sorted_probs[..., 0]
    margin_uncertainty = 1.0 - margin

    return {
        "mean_u": mean_u,
        "var_u": var_u,
        "entropy": entropy,
        "margin_uncertainty": margin_uncertainty,
    }


def get_hetero_ordinal_summary(
    model: Model,
    X: Tensor,
    *,
    utility_values: Optional[Sequence[float] | Tensor] = None,
    noise_penalty: float = 0.0,
    variance_scale: float = 1.0,
    tau: float = 1e-6,
    default_sigma: float = 0.0,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    probs = get_class_probs(model, X)
    stats = ordinal_stats(
        probs,
        utility_values=utility_values,
        eps=eps,
    )

    sigma = get_noise_sigma(
        model,
        X,
        mean_like=stats["mean_u"].unsqueeze(-1),
        default_sigma=default_sigma,
    ).squeeze(-1)

    robust_mean = stats["mean_u"] - float(noise_penalty) * sigma
    total_std = (
        stats["var_u"] + float(variance_scale) * sigma.pow(2) + float(tau) ** 2
    ).clamp_min(eps).sqrt()

    return {
        "probs": probs,
        "mean_u": stats["mean_u"],
        "var_u": stats["var_u"],
        "entropy": stats["entropy"],
        "margin_uncertainty": stats["margin_uncertainty"],
        "sigma": sigma,
        "robust_mean": robust_mean,
        "total_std": total_std,
    }


def reduce_q(score: Tensor, reduce: str = "max") -> Tensor:
    """
    (..., q) -> (...) に縮約
    """
    if score.ndim <= 1:
        return score

    if reduce == "max":
        return score.max(dim=-1).values
    if reduce == "mean":
        return score.mean(dim=-1)
    if reduce == "sum":
        return score.sum(dim=-1)
    raise ValueError(f"Unknown reduce='{reduce}'")


def compute_hetero_ordinal_best_f(
    model: Model,
    train_X: Tensor,
    *,
    utility_values: Optional[Sequence[float] | Tensor] = None,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    eps: float = 1e-12,
) -> Tensor:
    with torch.no_grad():
        summary = get_hetero_ordinal_summary(
            model,
            train_X,
            utility_values=utility_values,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            eps=eps,
        )
        return summary["robust_mean"].max()


def _get_submodels(model: Model) -> list[Model]:
    models = getattr(model, "models", None)
    if models is None:
        raise ValueError(
            "Expected a ModelList-like object with `.models` for multi-output acquisition."
        )
    return list(models)


def _expand_scalar_or_list(value, n: int, name: str) -> list:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return [value.item()] * n
        if value.ndim == 1 and value.numel() == n:
            return value.detach().cpu().tolist()
        raise ValueError(f"{name} tensor must be scalar or length {n}.")

    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} must have length {n}, got {len(value)}.")
        return list(value)

    return [value] * n


def _expand_utility_values_list(
    utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]],
    n: int,
) -> list[Optional[Sequence[float] | Tensor]]:
    if utility_values_list is None:
        return [None] * n
    if len(utility_values_list) != n:
        raise ValueError(
            f"utility_values_list must have length {n}, got {len(utility_values_list)}."
        )
    return list(utility_values_list)


def _param_tensor(values, *, ref: Tensor) -> Tensor:
    return torch.as_tensor(values, device=ref.device, dtype=ref.dtype)


def stack_multi_summaries(
    model: Model,
    X: Tensor,
    *,
    utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]] = None,
    noise_penalties: float | Sequence[float] | Tensor = 0.0,
    variance_scales: float | Sequence[float] | Tensor = 1.0,
    taus: float | Sequence[float] | Tensor = 1e-6,
    default_sigmas: float | Sequence[float] | Tensor = 0.0,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    submodels = _get_submodels(model)
    m = len(submodels)

    utility_values_list = _expand_utility_values_list(utility_values_list, m)
    noise_penalties = _expand_scalar_or_list(noise_penalties, m, "noise_penalties")
    variance_scales = _expand_scalar_or_list(variance_scales, m, "variance_scales")
    taus = _expand_scalar_or_list(taus, m, "taus")
    default_sigmas = _expand_scalar_or_list(default_sigmas, m, "default_sigmas")

    summaries = []
    for i, submodel in enumerate(submodels):
        s = get_hetero_ordinal_summary(
            submodel,
            X,
            utility_values=utility_values_list[i],
            noise_penalty=float(noise_penalties[i]),
            variance_scale=float(variance_scales[i]),
            tau=float(taus[i]),
            default_sigma=float(default_sigmas[i]),
            eps=eps,
        )
        summaries.append(s)

    return {
        "mean_u": torch.stack([s["mean_u"] for s in summaries], dim=-1),
        "var_u": torch.stack([s["var_u"] for s in summaries], dim=-1),
        "entropy": torch.stack([s["entropy"] for s in summaries], dim=-1),
        "margin_uncertainty": torch.stack(
            [s["margin_uncertainty"] for s in summaries], dim=-1
        ),
        "sigma": torch.stack([s["sigma"] for s in summaries], dim=-1),
        "robust_mean": torch.stack([s["robust_mean"] for s in summaries], dim=-1),
        "total_std": torch.stack([s["total_std"] for s in summaries], dim=-1),
    }


def aggregate_objectives(
    values: Tensor,
    *,
    method: str = "mean",
    weights: Optional[Tensor] = None,
) -> Tensor:
    if method == "product":
        return values.prod(dim=-1)
    if method == "min":
        return values.min(dim=-1).values
    if method == "max":
        return values.max(dim=-1).values
    if method == "sum":
        return values.sum(dim=-1)
    if method == "mean":
        if weights is None:
            return values.mean(dim=-1)
        w = weights / weights.sum().clamp_min(1e-12)
        return (values * w).sum(dim=-1)
    if method == "weighted_sum":
        if weights is None:
            raise ValueError("weights must be provided when method='weighted_sum'.")
        return (values * weights).sum(dim=-1)

    raise ValueError(f"Unknown aggregate method: {method}")


def make_weight_tensor(
    weights: Optional[Sequence[float] | Tensor],
    *,
    ref: Tensor,
    m: int,
) -> Optional[Tensor]:
    if weights is None:
        return None
    w = _expand_scalar_or_list(weights, m, "objective_weights")
    w = _param_tensor(w, ref=ref)
    view_shape = (1,) * (ref.ndim - 1) + (m,)
    return w.view(*view_shape)


def compute_multiobjective_best_f(
    model: Model,
    train_X: Tensor,
    *,
    utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]] = None,
    noise_penalties: float | Sequence[float] | Tensor = 0.0,
    default_sigmas: float | Sequence[float] | Tensor = 0.0,
    eps: float = 1e-12,
) -> Tensor:
    submodels = _get_submodels(model)
    m = len(submodels)
    utility_values_list = _expand_utility_values_list(utility_values_list, m)
    noise_penalties = _expand_scalar_or_list(noise_penalties, m, "noise_penalties")
    default_sigmas = _expand_scalar_or_list(default_sigmas, m, "default_sigmas")

    out = []
    for i, submodel in enumerate(submodels):
        best_f = compute_hetero_ordinal_best_f(
            submodel,
            train_X=train_X,
            utility_values=utility_values_list[i],
            noise_penalty=float(noise_penalties[i]),
            default_sigma=float(default_sigmas[i]),
            eps=eps,
        )
        out.append(best_f)
    return torch.stack(out)