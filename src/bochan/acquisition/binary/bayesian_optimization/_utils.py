
from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model


ReductionType = Literal["mean", "sum"]
MultiOutputMode = Literal["mean", "sum", "max", "min", "weighted_mean", "all_positive"]
PoFMode = Literal["mc_sigmoid", "latent_cdf"]

def _as_t_batch_X(X: Tensor) -> Tensor:
    """X を batch_shape x q x d 形式にそろえる。"""
    return X if X.dim() > 2 else X.unsqueeze(0)


def _prod_shape(shape: torch.Size | tuple[int, ...]) -> int:
    return int(math.prod(tuple(shape))) if len(shape) > 0 else 1


def reduce_flattened_input_perturbation(
    values: Tensor,
    X: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    InputPerturbation により q * n_w に展開された値を q に戻す。

    Args:
        values:
            posterior.mean, posterior samples, noise sigma など。
            例:
                batch x (q * n_w)
                sample_shape x batch x (q * n_w)
                batch x (q * n_w) x 1
                sample_shape x batch x (q * n_w) x 1
        X:
            raw candidate tensor.
            shape: batch_shape x q x d
        reduction:
            現時点では "mean" を推奨。

    Returns:
        values with shape:
            ... x batch_shape x q
    """
    if values.shape[-1:] == torch.Size([1]):
        values = values.squeeze(-1)

    Xb = _as_t_batch_X(X)
    expected_prefix = Xb.shape[:-1]  # batch_shape x q
    expected_numel = _prod_shape(expected_prefix)
    nd = len(expected_prefix)

    # すでに batch_shape x q の場合
    if values.shape[-nd:] == expected_prefix:
        return values

    # すでに batch_shape x q x n_w の場合
    if values.dim() >= nd + 1 and values.shape[-(nd + 1):-1] == expected_prefix:
        if reduction == "mean":
            return values.mean(dim=-1)
        if reduction == "sum":
            return values.sum(dim=-1)
        raise ValueError(f"Unsupported reduction: {reduction}")

    # 末尾 nd 次元を flatten 扱いして、q * n_w になっているか確認
    trailing_shape = values.shape[-nd:]
    trailing_numel = _prod_shape(trailing_shape)

    if trailing_numel == expected_numel:
        return values.reshape(*values.shape[:-nd], *expected_prefix)

    if trailing_numel % expected_numel != 0:
        raise RuntimeError(
            "Cannot reduce flattened InputPerturbation dimension: "
            f"X.shape={tuple(X.shape)}, values.shape={tuple(values.shape)}, "
            f"expected_prefix={tuple(expected_prefix)}"
        )

    n_w = trailing_numel // expected_numel

    values = values.reshape(
        *values.shape[:-nd],
        *expected_prefix,
        n_w,
    )

    if reduction == "mean":
        return values.mean(dim=-1)
    if reduction == "sum":
        return values.sum(dim=-1)

    raise ValueError(f"Unsupported reduction: {reduction}")

class _StackedPosterior:
    """ModelList / model.models 由来の posterior を multi-output 風に束ねる簡易 posterior。"""

    def __init__(self, posteriors: list) -> None:
        if len(posteriors) == 0:
            raise ValueError("At least one posterior is required.")
        self.posteriors = posteriors
        self._mean = torch.cat([self.ensure_last_output_dim(p.mean) for p in posteriors], dim=-1)

        vars_ = []
        for p in posteriors:
            if hasattr(p, "variance"):
                v = p.variance
            else:
                dist = getattr(p, "distribution", None)
                if dist is None or not hasattr(dist, "variance"):
                    raise AttributeError("Could not extract variance from posterior.")
                v = dist.variance
            vars_.append(self.ensure_last_output_dim(v))
        self._variance = torch.cat(vars_, dim=-1)

    @staticmethod
    def ensure_last_output_dim(x: Tensor) -> Tensor:
        if x.ndim == 0:
            return x.view(1, 1)
        if x.ndim >= 1 and x.shape[-1] == 1:
            return x
        return x.unsqueeze(-1)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        samples = []
        for p in self.posteriors:
            s = p.rsample(sample_shape)
            samples.append(self.ensure_last_output_dim(s))
        return torch.cat(samples, dim=-1)


def ensure_q_batch(X: Tensor) -> Tensor:
    return X.unsqueeze(-2) if X.ndim == 2 else X


def _try_apply_input_transform_for_shape(
    input_transform,
    X: Tensor,
) -> Optional[Tensor]:
    """Safely apply an input transform for shape inference.

    Some mixed-input wrappers keep ``train_X`` in raw space while their internal
    ``input_transform`` is defined on an encoded / internal feature space.  In
    that case, directly calling ``input_transform(raw_X)`` raises a BoTorch
    dimension error such as ``Received 3, expected 4``.

    This helper only uses the transform when it can be applied safely.  If the
    transform is incompatible with the provided raw X, it returns ``None`` and
    the caller can fall back to raw X as the shape reference.

    Args:
        input_transform: BoTorch input transform or ``None``.
        X: Candidate tensor.

    Returns:
        Optional[Tensor]: Transformed X if successful, otherwise ``None``.
    """
    if input_transform is None:
        return None

    try:
        Xt = input_transform(X)
    except Exception:
        return None

    if isinstance(Xt, tuple):
        Xt = Xt[0]
    if not torch.is_tensor(Xt):
        return None

    return ensure_q_batch(Xt)


def shape_X_for_model(model: Model, X: Tensor) -> Tensor:
    """Return a shape reference for posterior outputs.

    The function prefers the model's transformed input shape when the transform
    can be safely applied.  For mixed-input models, however, raw input may have
    fewer columns than the internal encoded input expected by ``input_transform``.
    In that case, this function falls back to raw ``X`` instead of raising.

    Args:
        model: BoTorch-compatible model.
        X: Raw candidate tensor.

    Returns:
        Tensor: Shape reference.  This is either transformed X or raw X.
    """
    X = ensure_q_batch(X)

    Xt = _try_apply_input_transform_for_shape(
        getattr(model, "input_transform", None),
        X,
    )
    if Xt is not None:
        return Xt

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        Xt = _try_apply_input_transform_for_shape(
            getattr(models[0], "input_transform", None),
            X,
        )
        if Xt is not None:
            return Xt

    return X


def get_single_model_posterior(model, X: Tensor, *, samples_are_probs: bool):
    if samples_are_probs:
        fn = getattr(model, "probability_posterior", None)
        if callable(fn):
            return fn(X)
        return model.posterior(X)

    for name in ("latent_posterior", "posterior_latent", "posterior_f"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(X)
    return model.posterior(X)


def get_model_posterior(model: Model, X: Tensor, *, samples_are_probs: bool):
    X = ensure_q_batch(X)

    if samples_are_probs:
        fn = getattr(model, "probability_posterior", None)
        if callable(fn):
            return fn(X)
    else:
        for name in ("latent_posterior", "posterior_latent", "posterior_f"):
            fn = getattr(model, name, None)
            if callable(fn):
                return fn(X)

    if hasattr(model, "models"):
        return _StackedPosterior([
            get_single_model_posterior(sm, X, samples_are_probs=samples_are_probs)
            for sm in model.models
        ])

    return model.posterior(X)


def normalize_mean_shape(mean: Tensor, X: Tensor) -> Tensor:
    X = ensure_q_batch(X)
    expected_prefix = X.shape[:-1]
    n_points = math.prod(expected_prefix)

    if mean.shape == expected_prefix:
        return mean.unsqueeze(-1)
    if mean.ndim == X.ndim and mean.shape[:-1] == expected_prefix:
        return mean
    if mean.numel() % n_points == 0:
        m = mean.numel() // n_points
        return mean.reshape(*expected_prefix, m)

    raise RuntimeError(
        "Unsupported posterior.mean shape for multi-output binary classification: "
        f"X.shape={tuple(X.shape)}, posterior.mean.shape={tuple(mean.shape)}"
    )


def reshape_samples(samples: Tensor, X: Tensor, sample_shape: torch.Size | None = None) -> Tensor:
    X = ensure_q_batch(X)
    expected_prefix = X.shape[:-1]
    if sample_shape is None:
        S = samples.shape[0]
        sample_shape = torch.Size([S])

    sample_numel = math.prod(sample_shape)
    n_points = math.prod(expected_prefix)
    if samples.numel() % (sample_numel * n_points) != 0:
        raise RuntimeError(
            f"Unexpected sample shape: {tuple(samples.shape)} for X={tuple(X.shape)}"
        )
    m = samples.numel() // (sample_numel * n_points)
    return samples.reshape(*sample_shape, *expected_prefix, m)


def to_probability(
    x: Tensor,
    *,
    apply_sigmoid_if_needed: bool,
    eps: float,
    name: str,
) -> Tensor:
    xmin = x.min().item()
    xmax = x.max().item()
    if 0.0 <= xmin and xmax <= 1.0:
        return x.clamp(eps, 1.0 - eps)
    if apply_sigmoid_if_needed:
        return torch.sigmoid(x).clamp(eps, 1.0 - eps)
    raise RuntimeError(
        f"{name} is not in [0,1] (min={xmin:.4g}, max={xmax:.4g}). "
        "Enable sigmoid conversion or return probability posterior."
    )


def binary_entropy(p: Tensor, eps: float = 1e-6) -> Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def reduce_q(score: Tensor, reduction: ReductionType) -> Tensor:
    if reduction == "mean":
        return score.mean(dim=-1)
    if reduction == "sum":
        return score.sum(dim=-1)
    raise ValueError(f"Unknown reduction: {reduction}")


def aggregate_outputs(
    score_per_output: Tensor,
    *,
    output_mode: MultiOutputMode,
    output_weights: Optional[Tensor] = None,
    probs_for_all_positive: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    if output_mode == "mean":
        return score_per_output.mean(dim=-1)
    if output_mode == "sum":
        return score_per_output.sum(dim=-1)
    if output_mode == "max":
        return score_per_output.max(dim=-1).values
    if output_mode == "min":
        return score_per_output.min(dim=-1).values
    if output_mode == "weighted_mean":
        if output_weights is None:
            raise ValueError("output_weights is required for weighted_mean.")
        w = output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype)
        w = w / w.sum().clamp_min(eps)
        return (score_per_output * w.view(*([1] * (score_per_output.ndim - 1)), -1)).sum(dim=-1)
    if output_mode == "all_positive":
        if probs_for_all_positive is None:
            raise ValueError("probs_for_all_positive is required for all_positive.")
        return probs_for_all_positive.log().sum(dim=-1).exp().clamp(eps, 1.0 - eps)

    raise ValueError(f"Unknown output_mode: {output_mode}")


def is_classification_score_objective(objective) -> bool:
    if isinstance(objective, MCMultiOutputObjective):
        return False
    cls_name = objective.__class__.__name__
    module_name = objective.__class__.__module__
    return (
        cls_name in {"ClassificationScoreObjective", "MultiOutputClassificationScoreObjective"}
        or ("classification" in module_name and hasattr(objective, "n_w") and hasattr(objective, "risk_type"))
    )


def apply_pointwise_score_objective(
    owner,
    score: Tensor,
    *,
    raw_X: Tensor,
    expanded_X: Tensor,
    name: str,
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    if is_classification_score_objective(objective):
        try:
            out = objective(score, X=raw_X)
        except TypeError:
            out = objective(score)
    else:
        score_in = score
        if isinstance(objective, MCMultiOutputObjective) and score_in.ndim == expanded_X.ndim - 1:
            score_in = score_in.unsqueeze(-1)
        try:
            out = objective(score_in, X=raw_X)
        except TypeError:
            out = objective(score_in)

    if not torch.is_tensor(out):
        raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    if out.ndim == raw_X.ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out
# ============================================================
# Single-output binary helpers
# ============================================================

def normalize_binary_mean_shape(
    mean: Tensor,
    X: Tensor,
    perturbation_reduction: str = "mean",
) -> Tensor:
    """
    binary classification posterior.mean を X の batch_shape x q にそろえる。

    InputPerturbation がある場合:
        X.shape = batch_shape x q x d
        mean.shape = batch_shape x (q * n_w)

    のようになるため、q * n_w を q x n_w に戻して n_w 方向を集約する。
    """
    return reduce_flattened_input_perturbation(
        values=mean,
        X=X,
        reduction=perturbation_reduction,
    )


def reshape_binary_samples(
    samples: Tensor,
    X: Tensor,
    perturbation_reduction: str = "mean",
) -> Tensor:
    """
    binary classification posterior samples を sample_shape x batch_shape x q にそろえる。

    InputPerturbation がある場合:
        samples.shape = sample_shape x batch_shape x (q * n_w)

    を

        sample_shape x batch_shape x q

    に戻す。
    """
    return reduce_flattened_input_perturbation(
        values=samples,
        X=X,
        reduction=perturbation_reduction,
    )


def apply_score_objective(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    *,
    attr_name: str = "objective",
    name: str = "BinaryClassificationScore",
) -> Tensor:
    """
    Apply a score objective to a single-output binary acquisition score.

    This helper is intended for acquisition scores, not posterior samples.
    ``ClassificationScoreObjective`` can receive ``(*batch, q_like)`` directly.
    BoTorch MC objectives receive ``(*batch, q_like, 1)``.

    Args:
        owner: Object that owns the objective attribute.
        score: Pointwise or aggregated acquisition score.
        X: Raw candidate tensor.
        attr_name: Attribute name that stores the objective.
        name: Name used in error messages.

    Returns:
        Tensor: Objective-transformed score.
    """
    objective = getattr(owner, attr_name, None)
    if objective is None:
        return score

    if is_classification_score_objective(objective):
        try:
            out = objective(score, X=X)
        except TypeError:
            out = objective(score)
    else:
        score_in = score
        if X is not None and score_in.ndim == X.ndim - 1:
            score_in = score_in.unsqueeze(-1)
        try:
            out = objective(score_in, X=X)
        except RuntimeError as err:
            if hasattr(objective, "_verify_output_shape"):
                old_verify = objective._verify_output_shape
                try:
                    objective._verify_output_shape = False
                    out = objective(score_in, X=X)
                finally:
                    objective._verify_output_shape = old_verify
            else:
                raise err
        except TypeError:
            out = objective(score_in)

    if not torch.is_tensor(out):
        raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

    if X is not None and out.ndim == X.ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)

    return out


# ============================================================
# best_f helpers for binary Bayesian optimization
# ============================================================

def _aggregate_expanded_binary_values(
    values: Tensor,
    *,
    train_X: Tensor,
    shape_X: Tensor,
    risk_type: Optional[Literal["var", "cvar"]] = None,
    alpha: float = 0.5,
    maximize: bool = True,
) -> Tensor:
    """
    Aggregate values expanded by InputPerturbation back to one value per
    original training point.

    Args:
        values: Pointwise values, typically ``(n, n_w)`` or ``(n * n_w,)``.
        train_X: Original training inputs with shape ``(n, d)``.
        shape_X: Shape reference after model input transform.
        risk_type: ``None`` for mean, ``"var"`` for tail VaR, ``"cvar"``
            for tail mean.
        alpha: Tail fraction for VaR / CVaR.
        maximize: Whether larger values are better.

    Returns:
        Tensor: Values with shape ``(n,)``.
    """
    n = train_X.shape[-2]

    y = values
    if y.ndim >= 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    if y.ndim == 0:
        return y.reshape(1)

    if y.shape == torch.Size([n]):
        return y

    # Typical InputPerturbation case:
    # train_X: (n, d), shape_X: (n, n_w, d), values: (n, n_w)
    if y.ndim == 2 and y.shape[0] == n:
        n_w = y.shape[1]
        if risk_type is None:
            return y.mean(dim=-1)

        descending = not maximize
        sorted_y = torch.sort(y, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(n_w * float(alpha))))
        tail = sorted_y[..., :k]
        if risk_type == "var":
            return tail[..., -1]
        if risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {risk_type}")

    # Flattened expanded case: values: (n * n_w,)
    if y.numel() % n == 0:
        n_w = y.numel() // n
        y_w = y.reshape(n, n_w)
        if risk_type is None:
            return y_w.mean(dim=-1)

        descending = not maximize
        sorted_y = torch.sort(y_w, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(n_w * float(alpha))))
        tail = sorted_y[..., :k]
        if risk_type == "var":
            return tail[..., -1]
        if risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {risk_type}")

    # Fallback for already batched shape, e.g. (1, n)
    if y.numel() == n:
        return y.reshape(n)

    raise RuntimeError(
        "Could not aggregate expanded binary values to original train_X. "
        f"values.shape={tuple(values.shape)}, train_X.shape={tuple(train_X.shape)}, "
        f"shape_X.shape={tuple(shape_X.shape)}."
    )


def get_binary_noise_posterior(model: Model, X: Tensor):
    """
    Best-effort retrieval of a heteroscedastic noise posterior.

    Args:
        model: Binary classification model.
        X: Candidate tensor.

    Returns:
        Posterior-like object for the noise model.

    Raises:
        AttributeError: If no noise posterior accessor is found.
    """
    if hasattr(model, "posterior_noise") and callable(getattr(model, "posterior_noise")):
        return model.posterior_noise(X)

    if hasattr(model, "noise_posterior") and callable(getattr(model, "noise_posterior")):
        return model.noise_posterior(X)

    noise_model = getattr(model, "noise_model", None)
    if noise_model is None:
        inner_model = getattr(model, "model", None)
        if inner_model is not None:
            noise_model = getattr(inner_model, "noise_model", None)

    if noise_model is None:
        raise AttributeError(
            "Noise posterior was not found. Expected one of:\n"
            "  - model.posterior_noise(X)\n"
            "  - model.noise_posterior(X)\n"
            "  - model.noise_model.posterior(X)\n"
            "  - model.model.noise_model.posterior(X)"
        )

    noise_in = X
    if getattr(model, "noise_model_uses_transformed_inputs", True):
        it = getattr(model, "input_transform", None)
        if it is not None:
            noise_in = it(X)

    return noise_model.posterior(noise_in)


def get_binary_noise_std(
    model: Model,
    X: Tensor,
    *,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-6,
    shape_X: Optional[Tensor] = None,
) -> Tensor:
    """
    Return heteroscedastic noise standard deviation for binary classification.

    Args:
        model: Binary classification model.
        X: Raw candidate tensor.
        default_sigma: Fallback value used when a noise posterior is unavailable.
        noise_is_log_var: If True, noise posterior mean is interpreted as log variance.
            If False, it is interpreted as variance.
        eps: Numerical lower bound.
        shape_X: Optional transformed / expanded shape reference.

    Returns:
        Tensor: Noise standard deviation with shape ``(*batch, q_like)``.
    """
    X = ensure_q_batch(X)
    if shape_X is None:
        shape_X = shape_X_for_model(model, X)

    try:
        noise_post = get_binary_noise_posterior(model, X)
        noise_mean = normalize_binary_mean_shape(noise_post.mean, shape_X)

        if noise_is_log_var:
            noise_var = torch.exp(noise_mean.clamp(min=math.log(eps), max=30.0))
        else:
            noise_var = noise_mean.clamp_min(eps)

        return noise_var.sqrt().clamp_min(eps)

    except Exception:
        ref = torch.zeros(shape_X.shape[:-1], device=shape_X.device, dtype=shape_X.dtype)
        return torch.full_like(ref, float(default_sigma))


def compute_binary_best_f(
    model: Model,
    train_X: Tensor,
    *,
    apply_sigmoid_if_needed: bool = False,
    risk_type: Optional[Literal["var", "cvar"]] = None,
    alpha: float = 0.5,
    eps: float = 1e-6,
    best_f_margin: float = 1e-4,
    best_f_quantile: Optional[float] = None,
) -> Tensor:
    """
    Compute ``best_f`` for ``qBinaryExpectedImprovement`` / binary qPI.

    The returned value is computed from the model-predicted positive-class
    probability on the existing training inputs. Do not use ``train_Y.max()``
    for binary classification, because it is usually 1 and can make EI almost
    zero everywhere.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}.")

    if best_f_quantile is not None and not (0.0 < best_f_quantile <= 1.0):
        raise ValueError(
            f"best_f_quantile must satisfy 0 < best_f_quantile <= 1, "
            f"got {best_f_quantile}."
        )

    model.eval()

    with torch.no_grad():
        Xq = ensure_q_batch(train_X)

        # Used to normalize posterior mean shape and to handle expanded
        # InputPerturbation shapes consistently.
        shape_X = shape_X_for_model(model, Xq)

        posterior = get_model_posterior(model, Xq, samples_are_probs=True)
        mean = normalize_binary_mean_shape(posterior.mean, shape_X)

        prob = to_probability(
            mean,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            name="posterior.mean",
        )

        prob_train = _aggregate_expanded_binary_values(
            prob,
            train_X=train_X,
            shape_X=shape_X,
            risk_type=risk_type,
            alpha=alpha,
            maximize=True,
        )

        prob_flat = prob_train.reshape(-1)

        if best_f_quantile is None:
            best_f = prob_flat.max()
        else:
            best_f = torch.quantile(prob_flat, best_f_quantile)

        # Important:
        # Do not allow best_f to become exactly the upper probability cap.
        # Otherwise EI can become zero everywhere.
        upper = 1.0 - best_f_margin
        best_f = best_f.clamp(min=eps, max=upper)

        return best_f.detach()


def compute_hetero_binary_classification_best_f(
    model: Model,
    train_X: Tensor,
    *,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    apply_sigmoid_if_needed: bool = False,
    risk_type: Optional[Literal["var", "cvar"]] = None,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute robust ``best_f`` for heteroscedastic binary BO acquisitions.

    The score is computed on existing inputs as

    ``p(y=1 | x) - noise_penalty * sigma_noise(x)``

    and then maximized. This is appropriate for
    ``qHeteroBinaryExpectedImprovement`` and
    ``qHeteroBinaryProbabilityOfImprovement``.

    Args:
        model: Heteroscedastic binary classification model.
        train_X: Existing input points with shape ``(n, d)``.
        noise_penalty: Penalty coefficient for input-dependent noise.
        default_sigma: Fallback noise standard deviation when no noise posterior
            is available.
        noise_is_log_var: If True, noise posterior mean is interpreted as log variance.
        apply_sigmoid_if_needed: If True, posterior mean outside ``[0, 1]`` is
            converted to probability by sigmoid.
        risk_type: Optional robust aggregation over InputPerturbation samples.
            ``None`` uses mean, ``"var"`` uses lower-tail VaR, and ``"cvar"``
            uses lower-tail CVaR.
        alpha: Tail fraction for VaR / CVaR.
        eps: Probability clipping value.

    Returns:
        Tensor: Scalar robust ``best_f``.
    """
    model.eval()

    with torch.no_grad():
        Xq = ensure_q_batch(train_X)
        # shape_X_for_model is intentionally safe for mixed-input models.
        # It falls back to raw X when model.input_transform expects an
        # encoded / internal feature dimension.
        shape_X = shape_X_for_model(model, Xq)

        posterior = get_model_posterior(model, Xq, samples_are_probs=True)
        mean = normalize_binary_mean_shape(posterior.mean, shape_X)

        prob = to_probability(
            mean,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            name="posterior.mean",
        )

        sigma_noise = get_binary_noise_std(
            model,
            Xq,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            eps=eps,
            shape_X=shape_X,
        )

        if sigma_noise.shape != prob.shape:
            if sigma_noise.numel() == prob.numel():
                sigma_noise = sigma_noise.reshape_as(prob)
            else:
                raise RuntimeError(
                    "Noise shape mismatch in compute_hetero_binary_classification_best_f: "
                    f"prob.shape={tuple(prob.shape)}, sigma_noise.shape={tuple(sigma_noise.shape)}"
                )

        robust_prob = (prob - float(noise_penalty) * sigma_noise).clamp(eps, 1.0 - eps)

        robust_train = _aggregate_expanded_binary_values(
            robust_prob,
            train_X=train_X,
            shape_X=shape_X,
            risk_type=risk_type,
            alpha=alpha,
            maximize=True,
        )

        return robust_train.max().detach()
