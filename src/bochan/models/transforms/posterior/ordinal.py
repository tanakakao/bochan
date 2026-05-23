from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence
from contextlib import nullcontext

import torch
from torch import Tensor


RiskType = Optional[Literal["var", "cvar"]]


@dataclass
class AggregatedPerturbedOrdinalUtility:
    """
    InputPerturbation 付き ordinal model の expected utility 集約結果。

    Attributes:
        posterior:
            model.posterior(X) の元の posterior。

        utility:
            摂動方向で集約した expected utility。
            shape: [..., q]

        utility_per_w:
            摂動ごとの expected utility。
            shape: [..., q, n_w]
    """

    posterior: Any
    utility: Tensor
    utility_per_w: Tensor

@dataclass
class AggregatedPerturbedOrdinalUtilityChunked:
    """
    chunk 分割して計算した InputPerturbation 付き ordinal expected utility。

    Attributes:
        utility:
            摂動方向で集約した expected utility。
            shape: [..., q]

        utility_per_w:
            摂動ごとの expected utility。
            shape: [..., q, n_w]
            return_per_w=False の場合は None。

        chunk_outputs:
            各 chunk の元結果。
            return_chunk_outputs=True の場合のみ保持。
    """

    utility: Tensor
    utility_per_w: Optional[Tensor]
    chunk_outputs: Optional[list[Any]] = None

def _aggregate_risk_axis_scalar(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
) -> Tensor:
    """
    values_w:
        shape = [..., q, n_w]
    """

    if risk_type is None:
        return values_w.mean(dim=-1)

    sorted_values = torch.sort(values_w, dim=-1, descending=False).values

    k = max(1, int(math.ceil(n_w * alpha)))
    tail = sorted_values[..., :k]

    if risk_type == "var":
        return tail[..., k - 1]

    if risk_type == "cvar":
        return tail.mean(dim=-1)

    raise ValueError(f"Unknown risk_type: {risk_type}")


def aggregate_perturbed_ordinal_expected_utility(
    model: Any,
    X: Tensor,
    utilities: Sequence[float] | Tensor,
    n_w: int,
    *,
    layout: Literal["point_major", "perturbation_major"] = "point_major",
    risk_type: RiskType = None,
    alpha: float = 0.5,
    maximize: bool = True,
    observation_noise: bool = False,
    posterior_transform: Optional[Any] = None,
    strict: bool = False,
    **posterior_kwargs: Any,
) -> AggregatedPerturbedOrdinalUtility:
    """
    InputPerturbation により q * n_w に展開された ordinal expected utility を、
    元の q 点ごとの値に戻す。

    重要:
        latent f を平均してから expected utility に変換するのではなく、
        各摂動点ごとに expected utility を計算してから集約する。

    Args:
        model:
            OrdinalGPModel / OrdinalMixedGPModel を想定。
            expected_utility(X, utilities) を持つモデル。

        X:
            元の入力点。
            shape: [..., q, d] または [q, d]

        utilities:
            ordinal class ごとの utility。
            例: [0.0, 1.0, 2.0]

        n_w:
            各点あたりの摂動数。

        layout:
            InputPerturbation 後の並び順。

            "point_major":
                [x0_w0, x0_w1, ..., x1_w0, x1_w1, ...]

            "perturbation_major":
                [x0_w0, x1_w0, ..., x0_w1, x1_w1, ...]

        risk_type:
            None:
                摂動方向の平均。

            "var":
                下側 alpha tail の VaR。

            "cvar":
                下側 alpha tail の平均。

        alpha:
            VaR / CVaR の tail 比率。

        maximize:
            False の場合、-utility を返す。

        strict:
            True の場合、posterior が q * n_w に展開されていないとエラー。

    Returns:
        AggregatedPerturbedOrdinalUtility
    """

    if X.ndim < 2:
        raise ValueError(
            f"X must have shape [..., q, d]. Got X.shape={tuple(X.shape)}."
        )

    if n_w <= 0:
        raise ValueError(f"n_w must be positive. Got n_w={n_w}.")

    if risk_type not in (None, "var", "cvar"):
        raise ValueError(f"Unknown risk_type: {risk_type}")

    if risk_type is not None and not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    q = X.shape[-2]

    utility_tensor = torch.as_tensor(
        utilities,
        device=X.device,
        dtype=X.dtype,
    )

    posterior = model.posterior(
        X,
        observation_noise=observation_noise,
        posterior_transform=posterior_transform,
        **posterior_kwargs,
    )

    # モデル側の expected_utility を使う。
    # ここで InputPerturbation により q * n_w 分の utility が返る。
    with torch.no_grad():
        # no_grad を外したい場合は、この with を削除してもよい。
        # 予測・可視化用途なら no_grad 推奨。
        pass

    utility = model.expected_utility(X, utility_tensor)

    if not maximize:
        utility = -utility

    # custom posterior / single-output の shape 対策
    if utility.ndim >= 2 and utility.shape[-1] == 1:
        utility = utility.squeeze(-1)

    expanded_q = utility.shape[-1]
    expected_q = q * n_w

    # InputPerturbation なし、またはすでに q に戻っている場合
    if expanded_q == q:
        if strict:
            raise ValueError(
                f"utility appears not to be expanded by n_w. "
                f"X q={q}, utility q={expanded_q}, n_w={n_w}."
            )

        return AggregatedPerturbedOrdinalUtility(
            posterior=posterior,
            utility=utility,
            utility_per_w=utility.unsqueeze(-1),
        )

    if expanded_q != expected_q:
        raise ValueError(
            f"utility q dimension is inconsistent with X and n_w. "
            f"X q={q}, n_w={n_w}, expected utility q={expected_q}, "
            f"but got utility q={expanded_q}. "
            f"utility.shape={tuple(utility.shape)}"
        )

    leading_shape = utility.shape[:-1]

    if layout == "point_major":
        utility_per_w = utility.reshape(*leading_shape, q, n_w)

    elif layout == "perturbation_major":
        utility_per_w = utility.reshape(*leading_shape, n_w, q).transpose(-2, -1)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    utility_agg = _aggregate_risk_axis_scalar(
        utility_per_w,
        n_w=n_w,
        risk_type=risk_type,
        alpha=alpha,
    )

    return AggregatedPerturbedOrdinalUtility(
        posterior=posterior,
        utility=utility_agg,
        utility_per_w=utility_per_w,
    )

def aggregate_perturbed_ordinal_expected_utility_chunked(
    model: Any,
    X: Tensor,
    utilities: Sequence[float] | Tensor,
    n_w: int,
    *,
    chunk_size: int = 512,
    layout: Literal["point_major", "perturbation_major"] = "point_major",
    risk_type: RiskType = None,
    alpha: float = 0.5,
    maximize: bool = True,
    observation_noise: bool = False,
    posterior_transform: Optional[Any] = None,
    strict: bool = False,
    return_per_w: bool = True,
    return_chunk_outputs: bool = False,
    use_no_grad: bool = True,
    detach: bool = True,
    **posterior_kwargs: Any,
) -> AggregatedPerturbedOrdinalUtilityChunked:
    """
    InputPerturbation 付き ordinal expected utility を chunk ごとに計算する。

    通常版:
        aggregate_perturbed_ordinal_expected_utility(...)

    の chunked wrapper。

    Args:
        model:
            OrdinalGPModel / OrdinalMixedGPModel を想定。

        X:
            入力点。
            shape: [..., q, d] または [q, d]

        utilities:
            ordinal class ごとの utility。
            例: [0.0, 1.0, 2.0]

        n_w:
            各点あたりの摂動数。

        chunk_size:
            q 方向に分割するサイズ。

        layout:
            InputPerturbation 後の並び順。
            "point_major" または "perturbation_major"。

        risk_type:
            None:
                摂動方向の平均。

            "var":
                下側 alpha tail の VaR。

            "cvar":
                下側 alpha tail の平均。

        alpha:
            VaR / CVaR の tail 比率。

        maximize:
            False の場合は -utility を返す。

        observation_noise:
            model.posterior に渡す observation_noise。

        posterior_transform:
            model.posterior に渡す posterior_transform。

        strict:
            True の場合、InputPerturbation 展開されていないとエラー。

        return_per_w:
            True の場合、utility_per_w も返す。
            メモリ削減したい場合は False 推奨。

        return_chunk_outputs:
            True の場合、各 chunk の元 result も保持する。

        use_no_grad:
            True の場合、torch.no_grad() で予測する。
            可視化・評価用途では True 推奨。
            acquisition 内で使う場合は False。

        detach:
            True の場合、各 chunk 結果を detach してから保存する。
            可視化・評価用途では True 推奨。

        **posterior_kwargs:
            model.posterior に渡す追加引数。

    Returns:
        AggregatedPerturbedOrdinalUtilityChunked
    """

    if X.ndim < 2:
        raise ValueError(
            f"X must have shape [..., q, d]. Got X.shape={tuple(X.shape)}."
        )

    if n_w <= 0:
        raise ValueError(f"n_w must be positive. Got n_w={n_w}.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got chunk_size={chunk_size}.")

    if risk_type not in (None, "var", "cvar"):
        raise ValueError(f"Unknown risk_type: {risk_type}")

    if risk_type is not None and not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    utilities_list: list[Tensor] = []
    utilities_per_w_list: list[Tensor] = []
    chunk_outputs: list[Any] = []

    context = torch.no_grad() if use_no_grad else nullcontext()

    with context:
        for X_chunk in X.split(chunk_size, dim=-2):
            out = aggregate_perturbed_ordinal_expected_utility(
                model=model,
                X=X_chunk,
                utilities=utilities,
                n_w=n_w,
                layout=layout,
                risk_type=risk_type,
                alpha=alpha,
                maximize=maximize,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                strict=strict,
                **posterior_kwargs,
            )

            utility = out.utility
            utility_per_w = out.utility_per_w

            if detach:
                utility = utility.detach()
                if utility_per_w is not None:
                    utility_per_w = utility_per_w.detach()

            utilities_list.append(utility)

            if return_per_w:
                if utility_per_w is None:
                    raise RuntimeError(
                        "return_per_w=True but chunk output utility_per_w is None."
                    )
                utilities_per_w_list.append(utility_per_w)

            if return_chunk_outputs:
                chunk_outputs.append(out)

    # utility shape:
    #   [q] or [..., q]
    # q 方向は最後の次元
    utility_all = torch.cat(utilities_list, dim=-1)

    # utility_per_w shape:
    #   [q, n_w] or [..., q, n_w]
    # q 方向は -2
    if return_per_w:
        utility_per_w_all = torch.cat(utilities_per_w_list, dim=-2)
    else:
        utility_per_w_all = None

    return AggregatedPerturbedOrdinalUtilityChunked(
        utility=utility_all,
        utility_per_w=utility_per_w_all,
        chunk_outputs=chunk_outputs if return_chunk_outputs else None,
    )

def aggregate_perturbed_ordinal_class_probs(
    model: Any,
    X: Tensor,
    n_w: int,
    *,
    layout: Literal["point_major", "perturbation_major"] = "point_major",
    strict: bool = False,
) -> Tensor:
    """
    InputPerturbation 付き ordinal model の class probability を
    元の X ごとに集約する。

    Args:
        model:
            OrdinalGPModel / OrdinalMixedGPModel

        X:
            shape: [q, d] または [..., q, d]

        n_w:
            各点あたりの摂動数。

        layout:
            InputPerturbation 後の点の並び順。

            "point_major":
                [x0_w0, x0_w1, ..., x1_w0, x1_w1, ...]

            "perturbation_major":
                [x0_w0, x1_w0, ..., x0_w1, x1_w1, ...]

        strict:
            True の場合、q * n_w に展開されていなければエラー。

    Returns:
        probs_agg:
            shape: [q, K] または [..., q, K]
    """

    if X.ndim < 2:
        raise ValueError(
            f"X must have shape [..., q, d]. Got X.shape={tuple(X.shape)}."
        )

    if n_w <= 0:
        raise ValueError(f"n_w must be positive. Got n_w={n_w}.")

    q = X.shape[-2]

    with torch.no_grad():
        probs = model.class_probs(X)

    # probs: [..., q_like, K]
    if probs.ndim < 2:
        raise ValueError(
            f"class_probs must have shape [..., q_like, K]. "
            f"Got probs.shape={tuple(probs.shape)}."
        )

    q_like = probs.shape[-2]
    num_classes = probs.shape[-1]

    # InputPerturbation なし、またはすでに q に戻っている場合
    if q_like == q:
        if strict:
            raise ValueError(
                f"class_probs appears not to be expanded by n_w. "
                f"X q={q}, probs q={q_like}, n_w={n_w}."
            )
        return probs

    expected_q = q * n_w

    if q_like != expected_q:
        raise ValueError(
            f"class_probs q dimension is inconsistent with X and n_w. "
            f"X q={q}, n_w={n_w}, expected q_like={expected_q}, "
            f"but got q_like={q_like}. probs.shape={tuple(probs.shape)}"
        )

    leading_shape = probs.shape[:-2]

    if layout == "point_major":
        # [..., q * n_w, K] -> [..., q, n_w, K]
        probs_per_w = probs.reshape(*leading_shape, q, n_w, num_classes)

    elif layout == "perturbation_major":
        # [..., n_w, q, K] -> [..., q, n_w, K]
        probs_per_w = probs.reshape(*leading_shape, n_w, q, num_classes).transpose(-3, -2)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    # 摂動方向で確率を平均
    probs_agg = probs_per_w.mean(dim=-2)

    # 数値誤差対策
    probs_agg = probs_agg.clamp_min(0.0)
    probs_agg = probs_agg / probs_agg.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return probs_agg