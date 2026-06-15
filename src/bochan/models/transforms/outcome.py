from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.transformed import TransformedPosterior

ScaleMethod = Literal["median", "mean", "max", "quantile"]


class PositiveScaleOutcomeTransform(OutcomeTransform):
    """正値 target 用の単純なスケール変換。

    Gamma / Poisson / Negative Binomial など、目的変数が非負または正値である
    モデルでは、通常の ``Standardize`` のように平均を引くと target が負になり、
    likelihood の仮定と合わなくなることがあります。この transform は
    ``Y / scale`` のみを行い、符号を変えずに数値スケールだけを調整します。

    Args:
        scale:
            スケールの決め方、または固定スケール値。
            ``"median"``, ``"mean"``, ``"max"``, ``"quantile"`` を指定した場合、
            学習時の ``Y`` から出力次元ごとに scale を推定します。
            float または Tensor を渡した場合は固定スケールとして扱います。
        quantile:
            ``scale="quantile"`` のときに使う分位点。
        min_scale:
            scale の下限。ゼロ除算を防ぐために使います。
        validate_positive:
            True の場合、``Y <= 0`` を検出して ValueError を出します。
            Gamma 回帰で厳密に正値を要求したい場合に使います。

    Notes:
        - 変換後: ``Y_tf = Y / scale``
        - 逆変換後: ``Y = Y_tf * scale``
        - 分散は ``scale ** 2`` で変換します。
    """

    _is_linear = True

    def __init__(
        self,
        scale: ScaleMethod | float | Tensor = "median",
        quantile: float = 0.8,
        min_scale: float = 1e-8,
        validate_positive: bool = False,
    ) -> None:
        super().__init__()
        self.quantile = float(quantile)
        self.min_scale = float(min_scale)
        self.validate_positive = bool(validate_positive)

        if isinstance(scale, str):
            if scale not in {"median", "mean", "max", "quantile"}:
                raise ValueError(
                    "scale must be one of {'median', 'mean', 'max', 'quantile'} "
                    "or a positive float / Tensor."
                )
            self.scale_method: ScaleMethod | None = scale  # type: ignore[assignment]
            self.fixed_scale = False
            init_scale = torch.ones(1)
            is_fitted = False
        else:
            init_scale = torch.as_tensor(scale).detach().clone().reshape(-1)
            if (init_scale <= 0).any():
                raise ValueError("Fixed scale must be positive.")
            self.scale_method = None
            self.fixed_scale = True
            is_fitted = True

        self.register_buffer("scale", init_scale.clamp_min(self.min_scale))
        self.register_buffer("_is_fitted", torch.tensor(is_fitted, dtype=torch.bool))

        if not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must satisfy 0 < quantile <= 1.")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive.")

    def _check_y(self, Y: Tensor) -> None:
        if not torch.isfinite(Y).all():
            raise ValueError("Y contains NaN or Inf values.")
        if self.validate_positive and (Y <= 0).any():
            raise ValueError("PositiveScaleOutcomeTransform requires Y > 0 when validate_positive=True.")

    def _compute_scale(self, Y: Tensor) -> Tensor:
        Y_flat = Y.reshape(-1, 1) if Y.ndim <= 1 else Y.reshape(-1, Y.shape[-1])
        abs_Y = Y_flat.abs()

        if self.scale_method == "median":
            scale = abs_Y.median(dim=0).values
        elif self.scale_method == "mean":
            scale = abs_Y.mean(dim=0)
        elif self.scale_method == "max":
            scale = abs_Y.max(dim=0).values
        elif self.scale_method == "quantile":
            scale = torch.quantile(abs_Y, q=self.quantile, dim=0)
        else:
            raise RuntimeError("scale_method is not set for a learned-scale transform.")

        return scale.clamp_min(self.min_scale)

    def _maybe_update_scale(self, Y: Tensor) -> None:
        if self.fixed_scale:
            self.scale = self.scale.to(device=Y.device, dtype=Y.dtype)
            return
        if self.training or not bool(self._is_fitted.item()):
            self.scale = self._compute_scale(Y).to(device=Y.device, dtype=Y.dtype)
            self._is_fitted.fill_(True)

    def _require_fitted(self) -> None:
        if not bool(self._is_fitted.item()):
            raise RuntimeError(
                "PositiveScaleOutcomeTransform has not been fitted. "
                "Call it with train_Y in training mode before untransforming."
            )

    def forward(
        self,
        Y: Tensor,
        Yvar: Tensor | None = None,
        X: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Y を scale で割り、必要なら Yvar も scale**2 で割る。"""
        del X
        self._check_y(Y)
        self._maybe_update_scale(Y)
        scale = self.scale.to(device=Y.device, dtype=Y.dtype)
        Y_tf = Y / scale
        Yvar_tf = None if Yvar is None else Yvar / scale.pow(2)
        return Y_tf, Yvar_tf

    def transform(
        self,
        Y: Tensor,
        Yvar: Tensor | None = None,
        X: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """``forward`` の明示的な別名。"""
        return self.forward(Y=Y, Yvar=Yvar, X=X)

    def untransform(
        self,
        Y: Tensor,
        Yvar: Tensor | None = None,
        X: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """変換済み Y を元スケールに戻す。"""
        del X
        self._require_fitted()
        scale = self.scale.to(device=Y.device, dtype=Y.dtype)
        Y_utf = Y * scale
        Yvar_utf = None if Yvar is None else Yvar * scale.pow(2)
        return Y_utf, Yvar_utf

    def untransform_posterior(
        self,
        posterior: Posterior,
        X: Tensor | None = None,
    ) -> Posterior:
        """posterior を元スケールに戻す。"""
        del X
        self._require_fitted()
        scale = self.scale.to(device=posterior.device, dtype=posterior.dtype)
        scale_sq = scale.pow(2)

        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda samples: samples * scale,
            mean_transform=lambda mean, variance: mean * scale,
            variance_transform=lambda mean, variance: variance * scale_sq,
        )

    def subset_output(self, idcs: list[int] | Tensor) -> "PositiveScaleOutcomeTransform":
        """指定された出力次元だけを持つ transform を返す。"""
        new = type(self)(
            scale=1.0,
            quantile=self.quantile,
            min_scale=self.min_scale,
            validate_positive=self.validate_positive,
        )
        new.scale_method = self.scale_method
        new.fixed_scale = self.fixed_scale

        if self.scale.numel() == 1:
            new.scale = self.scale.detach().clone()
        else:
            new.scale = self.scale[idcs].detach().clone()
        new._is_fitted.fill_(bool(self._is_fitted.item()))
        return new


__all__ = ["PositiveScaleOutcomeTransform", "ScaleMethod"]
