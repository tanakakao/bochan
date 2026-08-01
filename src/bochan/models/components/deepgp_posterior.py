from __future__ import annotations

from math import prod, sqrt

import torch
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import PsdSumLinearOperator, RootLinearOperator
from torch import Tensor


def moment_match_deepgp_distribution(
    dist: MultivariateNormal,
    X: Tensor,
) -> MultivariateNormal:
    """DeepGP の内部サンプル次元をモーメントマッチングで集約する。

    BoTorch の単一出力モデルは、入力 ``batch_shape x q x d`` に対して
    posterior の event shape が ``q x 1`` となることを期待する。一方、
    GPyTorch の DeepGP は likelihood sample 次元を先頭に追加するため、
    その次元を公開 posterior に残さないように集約する。

    各 DeepGP 成分を ``N(mu_s, Sigma_s)`` としたとき、

    ``mu = E_s[mu_s]``
    ``Sigma = E_s[Sigma_s] + Cov_s(mu_s)``

    を用いて単一の MultivariateNormal に近似する。

    Args:
        dist: DeepGP の最終層が返す潜在分布。
        X: 変換後の評価入力。shape は ``batch_shape x q x d``。

    Returns:
        内部サンプル次元を除去した潜在 MultivariateNormal。
    """
    mean = dist.mean
    expected_mean_shape = torch.Size(X.shape[:-1])
    extra_ndim = mean.ndim - len(expected_mean_shape)

    if extra_ndim < 0:
        raise RuntimeError(
            "DeepGP posterior mean has fewer dimensions than expected. "
            f"mean.shape={tuple(mean.shape)}, X.shape={tuple(X.shape)}."
        )

    if extra_ndim == 0:
        return dist

    if torch.Size(mean.shape[extra_ndim:]) != expected_mean_shape:
        raise RuntimeError(
            "DeepGP posterior mean cannot be aligned with X. "
            f"mean.shape={tuple(mean.shape)}, X.shape={tuple(X.shape)}, "
            f"extra_ndim={extra_ndim}."
        )

    q = int(X.shape[-2])
    expected_covar_shape = torch.Size((*X.shape[:-2], q, q))
    lazy_covar = dist.lazy_covariance_matrix
    if torch.Size(lazy_covar.shape[extra_ndim:]) != expected_covar_shape:
        raise RuntimeError(
            "DeepGP posterior covariance cannot be aligned with X. "
            f"covariance.shape={tuple(lazy_covar.shape)}, X.shape={tuple(X.shape)}, "
            f"extra_ndim={extra_ndim}."
        )

    component_shape = mean.shape[:extra_ndim]
    n_components = prod(int(size) for size in component_shape)
    component_mean = mean.reshape(n_components, *expected_mean_shape)
    matched_mean = component_mean.mean(dim=0)

    centered_mean = component_mean - matched_mean.unsqueeze(0)
    between_component_root = centered_mean.movedim(0, -1) / sqrt(n_components)
    between_component_covar = RootLinearOperator(between_component_root)

    within_component_covar = lazy_covar
    for _ in range(extra_ndim):
        within_component_covar = within_component_covar.sum(dim=0)
    within_component_covar = within_component_covar * (1.0 / n_components)

    matched_covar = PsdSumLinearOperator(
        within_component_covar,
        between_component_covar,
    )
    return MultivariateNormal(matched_mean, matched_covar)


__all__ = ["moment_match_deepgp_distribution"]
