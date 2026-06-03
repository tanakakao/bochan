"""High-level Bayesian optimization engine.

`BayesianOptimizer` はモデル生成・学習・予測・獲得関数生成・候補点最適化を
1つのクラスから扱うための薄い高レベル API です。

内部処理は `factory.py` の関数に委譲しており、研究用途では関数単位、
アプリ用途ではクラス単位で使い分けられるようにしています。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .acquisition_registry import resolve_acqf_cls
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
    """Bayesian Optimization の高レベル API。"""

    def __init__(
        self,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        *,
        bounds: Any | None = None,
        model_registry: Mapping[Any, Any] | None = None,
        acquisition_registry: Mapping[str, Any] | None = None,
        data_context: DataContext | None = None,
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.bounds = bounds
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry

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

        if self.bounds is None:
            self.bounds = _infer_bounds_from_train_X(train_X)
        if self.data_context is not None and self.data_context.bounds is None:
            self.data_context.bounds = self.bounds

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
        return self.fit(self.train_X, self.train_Y, fit_config=fit_config or self.fit_config)

    def predict(
        self,
        X: Any,
        *,
        return_type: str = "posterior",
        return_result: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """予測を行う。"""
        self._check_fitted()
        posterior_kwargs = posterior_kwargs or {}
        posterior = self.model.posterior(X, **posterior_kwargs)
        mean = getattr(posterior, "mean", None)
        variance = getattr(posterior, "variance", None)

        if return_result:
            return PredictionResult(posterior=posterior, mean=mean, variance=variance)
        if return_type == "posterior":
            return posterior
        if return_type == "mean":
            return mean
        if return_type == "variance":
            return variance
        if return_type == "mean_variance":
            return mean, variance
        raise ValueError("Unknown return_type. Expected 'posterior', 'mean', 'variance', or 'mean_variance'.")

    def _resolve_acquisition_config(self, acq_config: AcquisitionConfig) -> AcquisitionConfig:
        if acq_config.acqf_cls is not None or acq_config.acqf_factory is not None:
            return acq_config
        self._check_fitted()
        acqf_cls = resolve_acqf_cls(
            acq_config.name,
            self.acquisition_registry,
            task_type=self.bundle.task_type,
            model_type=self.bundle.model_type,
            multi_output=bool(self.bundle.metadata.get("multi_output", False)),
        )
        return replace(acq_config, acqf_cls=acqf_cls)

    def acquisition(
        self,
        acq_config: AcquisitionConfig,
        *,
        data_context: DataContext | None = None,
    ) -> Any:
        """獲得関数を生成する。"""
        self._check_fitted()
        context = self._resolve_data_context(data_context)
        acq_config = self._resolve_acquisition_config(acq_config)
        return build_acquisition(bundle=self.bundle, config=acq_config, data_context=context)

    def candidate(
        self,
        acq_config: AcquisitionConfig,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> CandidateResult | tuple[Any, Any]:
        """獲得関数を作成し、候補点を最適化する。"""
        self._check_fitted()
        context = self._resolve_data_context(data_context)

        opt_bounds = bounds if bounds is not None else context.bounds
        if opt_bounds is None:
            opt_bounds = self.bounds
        if opt_bounds is None and self.train_X is not None:
            opt_bounds = _infer_bounds_from_train_X(self.train_X)
            self.bounds = opt_bounds
            context.bounds = opt_bounds

        acq_config = self._resolve_acquisition_config(acq_config)
        acqf = build_acquisition(bundle=self.bundle, config=acq_config, data_context=context)
        candidates, acq_value = optimize_candidates(acqf=acqf, bounds=opt_bounds, config=opt_config)

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

    def _resolve_data_context(self, data_context: DataContext | None = None) -> DataContext:
        if data_context is not None:
            if data_context.bounds is None:
                data_context.bounds = self.bounds
            if data_context.X_baseline is None:
                data_context.X_baseline = self.train_X
            return data_context
        if self.data_context is not None:
            if self.data_context.bounds is None:
                self.data_context.bounds = self.bounds
            if self.data_context.X_baseline is None:
                self.data_context.X_baseline = self.train_X
            return self.data_context
        return DataContext(bounds=self.bounds, X_baseline=self.train_X)

    def _check_fitted(self) -> None:
        if self.bundle is None or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def _infer_bounds_from_train_X(train_X: Any) -> Any:
    """Infer BoTorch-style bounds from training inputs.

    For a 2D tensor with shape ``n x d``, this returns ``2 x d``.
    For batched inputs with shape ``batch_shape x n x d``, this returns
    ``batch_shape x 2 x d``.
    """
    if train_X is None:
        return None

    try:
        import torch

        if isinstance(train_X, torch.Tensor):
            if train_X.ndim < 2:
                raise ValueError("train_X must have shape n x d or batch_shape x n x d to infer bounds.")
            return torch.stack(
                [
                    train_X.min(dim=-2).values,
                    train_X.max(dim=-2).values,
                ],
                dim=-2,
            )
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(train_X, np.ndarray):
            if train_X.ndim < 2:
                raise ValueError("train_X must have shape n x d or batch_shape x n x d to infer bounds.")
            return np.stack(
                [
                    np.min(train_X, axis=-2),
                    np.max(train_X, axis=-2),
                ],
                axis=-2,
            )
    except ImportError:
        pass

    raise TypeError(
        "bounds is None and automatic bounds inference failed. "
        "Pass bounds to BayesianOptimizer(...), candidate(...), or DataContext(bounds=...)."
    )


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

    try:
        import pandas as pd
        if isinstance(x, (pd.DataFrame, pd.Series)) and isinstance(y, type(x)):
            return pd.concat([x, y], axis=0)
    except Exception:
        pass

    raise TypeError(
        "Unsupported data type for update_data(). Pass torch.Tensor, numpy.ndarray, pandas objects, "
        "or update train_X/train_Y manually before refit()."
    )
