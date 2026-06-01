"""High-level Bayesian optimization engine.

`BayesianOptimizer` はモデル生成・学習・予測・獲得関数生成・候補点最適化を
1つのクラスから扱うための薄い高レベル API です。

内部処理は `factory.py` の関数に委譲しており、研究用途では関数単位、
アプリ用途ではクラス単位で使い分けられるようにしています。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .configs import (
    AcquisitionConfig,
    CandidateResult,
    DataContext,
    FitConfig,
    ModelBundle,
    ModelConfig,
    OptimizeConfig,
    PredictionResult,
)
from .factory import build_acquisition, build_model, fit_model, optimize_candidates


class BayesianOptimizer:
    """Bayesian Optimization の高レベル API。

    Examples:
        >>> bo = BayesianOptimizer(model_config, fit_config, bounds=bounds)
        >>> bo.fit(train_X, train_Y)
        >>> mean, var = bo.predict(test_X, return_type="mean_variance")
        >>> candidates, acq_value = bo.candidate(acq_config, opt_config)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        *,
        bounds: Any | None = None,
        model_registry: Mapping[Any, Any] | None = None,
        data_context: DataContext | None = None,
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.bounds = bounds
        self.model_registry = model_registry

        self.data_context = data_context

        self.bundle: ModelBundle | None = None
        self.model: Any | None = None
        self.mll: Any | None = None

        self.train_X: Any | None = None
        self.train_Y: Any | None = None

        self.history: list[CandidateResult] = []

    def fit(
        self,
        train_X: Any,
        train_Y: Any,
        *,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        """モデルを生成し、必要なら学習する。"""
        if model_config is not None:
            self.model_config = model_config

        if fit_config is not None:
            self.fit_config = fit_config

        self.train_X = train_X
        self.train_Y = train_Y

        self.bundle = build_model(
            train_X=train_X,
            train_Y=train_Y,
            config=self.model_config,
            model_registry=self.model_registry,
        )
        self.bundle = fit_model(self.bundle, self.fit_config)

        self.model = self.bundle.model
        self.mll = self.bundle.mll

        return self

    def refit(self, *, fit_config: FitConfig | None = None) -> "BayesianOptimizer":
        """保持している train_X / train_Y で再学習する。"""
        if self.train_X is None or self.train_Y is None:
            raise RuntimeError("No training data found. Call fit() first.")

        return self.fit(
            self.train_X,
            self.train_Y,
            fit_config=fit_config or self.fit_config,
        )

    def predict(
        self,
        X: Any,
        *,
        return_type: str = "posterior",
        return_result: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """予測を行う。

        Args:
            X: 予測したい入力。
            return_type: posterior / mean / variance / mean_variance。
            return_result: True の場合は PredictionResult を返す。
            posterior_kwargs: model.posterior に渡す追加 kwargs。
        """
        self._check_fitted()
        posterior_kwargs = posterior_kwargs or {}

        posterior = self.model.posterior(X, **posterior_kwargs)
        mean = getattr(posterior, "mean", None)
        variance = getattr(posterior, "variance", None)

        if return_result:
            return PredictionResult(
                posterior=posterior,
                mean=mean,
                variance=variance,
            )

        if return_type == "posterior":
            return posterior

        if return_type == "mean":
            return mean

        if return_type == "variance":
            return variance

        if return_type == "mean_variance":
            return mean, variance

        raise ValueError(
            "Unknown return_type. Expected one of "
            "'posterior', 'mean', 'variance', 'mean_variance'."
        )

    def acquisition(
        self,
        acq_config: AcquisitionConfig,
        *,
        data_context: DataContext | None = None,
    ) -> Any:
        """獲得関数を生成する。"""
        self._check_fitted()
        context = self._resolve_data_context(data_context)
        return build_acquisition(
            bundle=self.bundle,
            config=acq_config,
            data_context=context,
        )

    def candidate(
        self,
        acq_config: AcquisitionConfig,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> CandidateResult | tuple[Any, Any]:
        """獲得関数を作成し、候補点を最適化する。

        デフォルトでは既存コードと相性がよいように `(candidates, acq_value)` を返します。
        `return_result=True` の場合は acqf や config を含む CandidateResult を返します。
        """
        self._check_fitted()
        context = self._resolve_data_context(data_context)

        opt_bounds = bounds if bounds is not None else context.bounds
        if opt_bounds is None:
            opt_bounds = self.bounds

        acqf = build_acquisition(
            bundle=self.bundle,
            config=acq_config,
            data_context=context,
        )

        candidates, acq_value = optimize_candidates(
            acqf=acqf,
            bounds=opt_bounds,
            config=opt_config,
        )

        result = CandidateResult(
            candidates=candidates,
            acq_value=acq_value,
            acqf=acqf,
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=context,
        )
        self.history.append(result)

        if return_result:
            return result

        return candidates, acq_value

    def ask(
        self,
        acq_config: AcquisitionConfig,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> CandidateResult | tuple[Any, Any]:
        """candidate() の alias。ask-and-tell 形式で使う場合に便利。"""
        return self.candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
            return_result=return_result,
        )

    def tell(
        self,
        new_X: Any,
        new_Y: Any,
        *,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        """新しい観測データを追加し、必要なら再学習する。"""
        self.update_data(new_X, new_Y)

        if refit:
            self.refit(fit_config=fit_config or self.fit_config)

        return self

    def update_data(self, new_X: Any, new_Y: Any) -> "BayesianOptimizer":
        """保持している訓練データに新しい観測を追加する。"""
        if self.train_X is None or self.train_Y is None:
            self.train_X = new_X
            self.train_Y = new_Y
            return self

        self.train_X = _concat_rows(self.train_X, new_X)
        self.train_Y = _concat_rows(self.train_Y, new_Y)
        return self

    def compare_acquisitions(
        self,
        acq_configs: Sequence[AcquisitionConfig],
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
    ) -> dict[str, CandidateResult]:
        """同じ学習済みモデルに対して複数の獲得関数を比較する。"""
        results: dict[str, CandidateResult] = {}

        for acq_config in acq_configs:
            result = self.candidate(
                acq_config=acq_config,
                opt_config=opt_config,
                data_context=data_context,
                bounds=bounds,
                return_result=True,
            )
            results[acq_config.name] = result

        return results

    def set_bounds(self, bounds: Any) -> "BayesianOptimizer":
        """探索範囲を更新する。"""
        self.bounds = bounds
        if self.data_context is not None:
            self.data_context.bounds = bounds
        return self

    def _resolve_data_context(
        self,
        data_context: DataContext | None = None,
    ) -> DataContext:
        if data_context is not None:
            return data_context

        if self.data_context is not None:
            if self.data_context.bounds is None:
                self.data_context.bounds = self.bounds
            if self.data_context.X_baseline is None:
                self.data_context.X_baseline = self.train_X
            return self.data_context

        return DataContext(
            bounds=self.bounds,
            X_baseline=self.train_X,
        )

    def _check_fitted(self) -> None:
        if self.bundle is None or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def _concat_rows(x: Any, y: Any) -> Any:
    """torch.Tensor / numpy.ndarray / pandas object の行方向結合を簡易的に扱う。"""
    try:
        import torch

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.cat([x, y], dim=-2)
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.concatenate([x, y], axis=-2)
    except Exception:
        pass

    if hasattr(x, "append"):
        # pandas.DataFrame.append は廃止済みのため、存在しても使わない。
        pass

    try:
        import pandas as pd

        if isinstance(x, (pd.DataFrame, pd.Series)) and isinstance(y, type(x)):
            return pd.concat([x, y], axis=0)
    except Exception:
        pass

    raise TypeError(
        "Unsupported data type for update_data(). "
        "Pass torch.Tensor, numpy.ndarray, pandas objects, "
        "or update train_X/train_Y manually before refit()."
    )
