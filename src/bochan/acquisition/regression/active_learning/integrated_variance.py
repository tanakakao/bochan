"""Integrated posterior variance acquisitions with model capability fallback."""

from __future__ import annotations

import warnings
from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from .single_output import qRegressionIntegratedPosteriorVarianceProxy
from .single_output import (
    qRegressionNegIntegratedPosteriorVariance as _BoTorchNegIntegratedPosteriorVariance,
)


class qRegressionNegIntegratedPosteriorVariance(AcquisitionFunction):
    """モデル能力に応じて真の NIPV または proxy を使用する獲得関数。

    BoTorch の ``qNegIntegratedPosteriorVariance`` は内部で
    ``model.fantasize()`` を使用する。そのため、``fantasize()`` を
    実装しないモデルではそのまま利用できない。

    本クラスは、``fantasize()`` 対応モデルでは従来の BoTorch 実装へ
    委譲し、非対応モデルでは ``mc_points`` を参照点とする
    ``qRegressionIntegratedPosteriorVarianceProxy`` へフォールバックする。
    フォールバックは近似であり、候補観測後の fantasy posterior を
    厳密に構成するものではない。

    Args:
        model:
            学習済み回帰モデル。
        mc_points:
            積分点。proxy 利用時は参照点として使用する。2 次元を超える
            leading batch 次元は平坦化する。
        sampler:
            BoTorch の fantasy sampler。proxy 利用時は使用しない。
        objective:
            オプションの objective。proxy 利用時は score objective として
            既存 proxy に渡す。
        posterior_transform:
            BoTorch 用 posterior transform。proxy 利用時は使用しない。
        X_pending:
            評価待ち点。
        fallback_to_proxy:
            ``fantasize()`` 非対応モデルで proxy にフォールバックするか。
            False の場合は明示的な ``NotImplementedError`` を送出する。
        proxy_kernel_lengthscale:
            proxy の参照点重み付けに使用する lengthscale。
        proxy_normalize_weights:
            proxy の参照点重みを正規化するか。
        **kwargs:
            委譲先獲得関数へ渡す追加引数。
    """

    def __init__(
        self,
        model,
        mc_points: Tensor,
        *,
        sampler: Any | None = None,
        objective: Any | None = None,
        posterior_transform: Any | None = None,
        X_pending: Tensor | None = None,
        fallback_to_proxy: bool = True,
        proxy_kernel_lengthscale: float = 0.2,
        proxy_normalize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)

        if mc_points.ndim < 2:
            raise ValueError(
                "mc_points must have shape [..., n_mc, d]. "
                f"Got {tuple(mc_points.shape)}."
            )

        supports_fantasize = callable(getattr(model, "fantasize", None))
        self._uses_proxy = not supports_fantasize

        if supports_fantasize:
            self.acqf = _BoTorchNegIntegratedPosteriorVariance(
                model=model,
                mc_points=mc_points,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
                X_pending=X_pending,
                **kwargs,
            )
            return

        if not fallback_to_proxy:
            raise NotImplementedError(
                f"{model.__class__.__name__} does not implement fantasize(), which is "
                "required by BoTorch qNegIntegratedPosteriorVariance. Use "
                "qRegressionIntegratedPosteriorVarianceProxy or set "
                "fallback_to_proxy=True."
            )

        ignored = []
        if sampler is not None:
            ignored.append("sampler")
        if posterior_transform is not None:
            ignored.append("posterior_transform")
        ignored_suffix = ""
        if ignored:
            ignored_suffix = f" The following arguments are ignored: {', '.join(ignored)}."

        warnings.warn(
            f"{model.__class__.__name__} does not implement fantasize(); "
            "qRegressionNegIntegratedPosteriorVariance is using "
            "qRegressionIntegratedPosteriorVarianceProxy instead. This is an "
            f"approximation, not fantasy-based NIPV.{ignored_suffix}",
            RuntimeWarning,
            stacklevel=2,
        )

        proxy_kwargs = dict(kwargs)
        proxy_kwargs.setdefault("kernel_lengthscale", float(proxy_kernel_lengthscale))
        proxy_kwargs.setdefault("normalize_weights", bool(proxy_normalize_weights))
        if objective is not None:
            proxy_kwargs.setdefault("objective", objective)
        if X_pending is not None:
            proxy_kwargs.setdefault("X_pending", X_pending)

        X_ref = mc_points.reshape(-1, mc_points.shape[-1])
        self.acqf = qRegressionIntegratedPosteriorVarianceProxy(
            model=model,
            X_ref=X_ref,
            **proxy_kwargs,
        )

    @property
    def uses_proxy(self) -> bool:
        """proxy 実装を使用している場合 True を返す。"""
        return self._uses_proxy

    @property
    def X_pending(self) -> Tensor | None:
        """Return pending points from the delegated acquisition."""
        return getattr(self.acqf, "X_pending", None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """評価待ち点を委譲先へ設定する。"""
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        else:
            self.acqf.X_pending = X_pending

    def forward(self, X: Tensor) -> Tensor:
        """候補点の獲得関数値を返す。"""
        return self.acqf(X)


__all__ = ["qRegressionNegIntegratedPosteriorVariance"]
