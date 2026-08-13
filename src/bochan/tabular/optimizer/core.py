"""Canonical DataFrame / numpy facade for :class:`bochan.api.BayesianOptimizer`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bochan.api import CrossValidationConfig, ExperimentFailureConfig, FitConfig, ModelConfig

from ..config import UNSET, ColumnKey, TabularDataConfig
from ..data import dataframe_to_tensors, tensor_to_dataframe
from .configuration import initialize_optimizer
from .fitting import fit_optimizer, sync_visualization_metadata, to_dataset
from .prediction import (
    DATAFRAME_RETURN_TYPES,
    LABEL_RETURN_TYPES,
    classification_prediction_dataframe,
    prediction_frame,
)


class TabularBayesianOptimizer:
    """Single public tabular optimizer coordinating explicit domain components."""

    def __init__(
        self,
        model_config: ModelConfig | Mapping[str, Any] | None = None,
        fit_config: FitConfig | Mapping[str, Any] | None = None,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]] | None = None,
        composition_total_constraints: Sequence[Any] | None = None,
        composition_element_constraints: Sequence[Any] | None = None,
        composition_constraint_rerank: bool = True,
        composition_constraint_rerank_factor: int = 4,
        composition_constraint_max_supports: int = 256,
        data_config: TabularDataConfig | None = None,
        data: Any | None = None,
        cross_validation: bool = False,
        cv_config: CrossValidationConfig | Mapping[str, Any] | None = None,
        failure_config: ExperimentFailureConfig | None = None,
        target_missing_strategy: str | None = None,
        experiment_status_col: ColumnKey | None = None,
        alpha: float | None = None,
        beta: float | None | Any = UNSET,
        normalize: bool | Any = UNSET,
        perturbation: bool | Any = UNSET,
        n_w: int | Any = UNSET,
        std: float | Any = UNSET,
        **kwargs: Any,
    ) -> None:
        initialize_optimizer(
            self,
            model_config,
            fit_config,
            composition_sites=composition_sites,
            composition_total_constraints=composition_total_constraints,
            composition_element_constraints=composition_element_constraints,
            composition_constraint_rerank=composition_constraint_rerank,
            composition_constraint_rerank_factor=composition_constraint_rerank_factor,
            composition_constraint_max_supports=composition_constraint_max_supports,
            data_config=data_config,
            data=data,
            cross_validation=cross_validation,
            cv_config=cv_config,
            failure_config=failure_config,
            target_missing_strategy=target_missing_strategy,
            experiment_status_col=experiment_status_col,
            alpha=alpha,
            beta=beta,
            normalize=normalize,
            perturbation=perturbation,
            n_w=n_w,
            std=std,
            kwargs=dict(kwargs),
        )

    @property
    def composition_enabled(self) -> bool:
        return self.composition.enabled

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        read_csv_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TabularBayesianOptimizer":
        import pandas as pd

        return cls(data=pd.read_csv(path, **(read_csv_kwargs or {})), **kwargs)

    def _to_dataset(
        self,
        data: Any,
        y: Any | None = None,
        *,
        data_config: TabularDataConfig | None = None,
        feature_names: Any = None,
        target_names: Any = None,
    ) -> Any:
        return to_dataset(
            self,
            data,
            y,
            data_config=data_config,
            feature_names=feature_names,
            target_names=target_names,
        )

    def _check_fitted(self) -> None:
        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        data_config: TabularDataConfig | None = None,
        alpha: float | None | Any = UNSET,
        beta: float | None | Any = UNSET,
        normalize: bool | Any = UNSET,
        perturbation: bool | Any = UNSET,
        n_w: int | Any = UNSET,
        std: float | Any = UNSET,
        target_missing_strategy: str | None = None,
        experiment_status_col: ColumnKey | None = None,
        failure_config: ExperimentFailureConfig | None = None,
        cross_validation: bool | None = None,
        cv_config: CrossValidationConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TabularBayesianOptimizer":
        return fit_optimizer(
            self,
            data,
            y,
            data_config=data_config,
            alpha=alpha,
            beta=beta,
            normalize=normalize,
            perturbation=perturbation,
            n_w=n_w,
            std=std,
            target_missing_strategy=target_missing_strategy,
            experiment_status_col=experiment_status_col,
            failure_config=failure_config,
            cross_validation=cross_validation,
            cv_config=cv_config,
            kwargs=dict(kwargs),
        )

    def transform_compositions(self, data: Any) -> Any:
        if not self.composition.enabled:
            return data
        if not self.composition.transformers:
            raise RuntimeError("Call fit() before transform_compositions().")
        return self.composition.prepare_frame(data, fit_transformers=False)

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        restored = self.composition.inverse(
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        return self.candidates.repair_compositions(restored, repair=repair)

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        *,
        data_context: Any | None = None,
        bounds: Any = None,
        return_dataframe: bool = True,
        return_result: bool = False,
        return_composition: bool = True,
        keep_composition_coordinates: bool = False,
        composition_constraint_rerank: bool | None = None,
        composition_constraint_rerank_factor: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.candidates.generate(
            self,
            acq_config,
            opt_config,
            data_context=data_context,
            bounds=bounds,
            return_dataframe=return_dataframe,
            return_result=return_result,
            return_composition=return_composition,
            keep_composition_coordinates=keep_composition_coordinates,
            composition_constraint_rerank=composition_constraint_rerank,
            composition_constraint_rerank_factor=composition_constraint_rerank_factor,
            values=dict(kwargs),
        )

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        return self.candidate(*args, **kwargs)

    def _prediction_input(self, data: Any) -> tuple[Any, Any | None]:
        self._check_fitted()
        try:
            import pandas as pd
        except ImportError:
            pd = None
        model_data = self.transform_compositions(data) if self.composition.enabled else data
        if pd is not None and isinstance(model_data, pd.DataFrame):
            from dataclasses import replace

            config = replace(
                self.data_config,
                target_cols=None,
                input_cols=self.dataset.feature_names,
            )
            return dataframe_to_tensors(model_data, config).X, model_data.index
        return model_data, None

    def predict(
        self,
        data: Any,
        *,
        return_type: str = "dataframe",
        include_input: bool = False,
        return_dataframe_input: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
        include_prediction_labels: bool = True,
        binary_threshold: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        X, index = self._prediction_input(data)
        normalized = str(return_type).lower()
        labels_only = normalized in LABEL_RETURN_TYPES
        dataframe_return = normalized in DATAFRAME_RETURN_TYPES
        returned_input = data if return_dataframe_input else None
        prediction_df = None

        if not labels_only:
            if not dataframe_return:
                result = self.bo.predict(
                    X,
                    return_type=return_type,
                    posterior_kwargs=posterior_kwargs,
                    **kwargs,
                )
                return (result, returned_input) if return_dataframe_input else result
            prediction = self.bo.predict(
                X,
                return_result=True,
                posterior_kwargs=posterior_kwargs,
                **kwargs,
            )
            prediction_df = self.prediction_to_dataframe(prediction)
            if index is not None:
                prediction_df.index = index
            if not include_prediction_labels:
                output = prediction_df
                if include_input:
                    output = self._prediction_input_to_dataframe(data, X).join(output)
                return (output, returned_input) if return_dataframe_input else output

        labels_df = classification_prediction_dataframe(
            self,
            X,
            posterior_kwargs=posterior_kwargs,
            binary_threshold=binary_threshold,
        )
        if labels_only and labels_df.shape[1] == 0:
            raise ValueError("The fitted optimizer has no classification outputs.")
        if index is not None:
            labels_df.index = index
        if prediction_df is None:
            output = labels_df
        elif labels_df.shape[1] == 0:
            output = prediction_df
        else:
            attrs = dict(prediction_df.attrs)
            output = prediction_df.join(labels_df)
            output.attrs.update(attrs)
        if include_input:
            output = self._prediction_input_to_dataframe(data, X).join(output)
        return (output, returned_input) if return_dataframe_input else output

    def prediction_to_dataframe(self, prediction: Any):
        import pandas as pd

        mean = getattr(prediction, "mean", None)
        variance = getattr(prediction, "variance", None)
        task_type = getattr(prediction, "task_type", None)
        if mean is None and hasattr(prediction, "posterior"):
            mean = getattr(prediction.posterior, "mean", None)
            variance = getattr(prediction.posterior, "variance", None)
        names = self.dataset.target_names if self.dataset is not None else []
        frame = pd.concat(
            [
                prediction_frame(mean, kind="mean", target_names=names, task_type=task_type),
                prediction_frame(variance, kind="variance", target_names=names, task_type=task_type),
            ],
            axis=1,
        )
        frame.attrs["task_type"] = task_type
        frame.attrs["prediction_space"] = getattr(prediction, "prediction_space", None)
        frame.attrs["variance_kind"] = getattr(prediction, "variance_kind", None)
        return frame

    def _prediction_input_to_dataframe(self, original_data: Any, X: Any):
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(original_data, pd.DataFrame):
            return original_data.copy()
        return tensor_to_dataframe(
            X,
            self.dataset.feature_names if self.dataset is not None else [],
            inverse_category_maps=(self.dataset.inverse_category_maps if self.dataset is not None else None),
            decode_categories=self.data_config.return_original_categories,
        )

    def feature_importance(self, *args: Any, **kwargs: Any) -> Any:
        return self.diagnostics.feature_importance(self, *args, **kwargs)

    def feature_importance_dataframe(self, *args: Any, **kwargs: Any) -> Any:
        return self.diagnostics.dataframe(self, *args, **kwargs)

    def visualization_training_dataframe(self, *, feature_cols: Any = None, target_cols: Any = None) -> Any:
        from bochan.visualization import training_dataframe
        self._check_fitted()
        return training_dataframe(
            self.bo,
            feature_cols=list(self.dataset.feature_names if feature_cols is None else feature_cols),
            target_cols=list(self.dataset.target_names if target_cols is None else target_cols),
        )

    def visualization_candidates_dataframe(self, *, candidate_result: Any = None, feature_cols: Any = None, target_cols: Any = None, include_prediction: bool = True) -> Any:
        from bochan.visualization import candidates_dataframe
        self._check_fitted()
        return candidates_dataframe(
            self.bo,
            candidate_result=candidate_result,
            feature_cols=list(self.dataset.feature_names if feature_cols is None else feature_cols),
            target_cols=list(self.dataset.target_names if target_cols is None else target_cols),
            include_prediction=include_prediction,
        )

    def update_data(self, new_data: Any, new_y: Any | None = None) -> "TabularBayesianOptimizer":
        self._check_fitted()
        model_data = self.transform_compositions(new_data) if self.composition.enabled else new_data
        dataset = self._to_dataset(
            model_data,
            new_y,
            data_config=self.data_config,
            feature_names=self.dataset.feature_names,
        )
        if dataset.Y is None:
            raise ValueError("Target values are required for update_data().")
        self.bo.update_data(dataset.X, dataset.Y)
        return self

    def tell(self, new_data: Any, new_y: Any | None = None, *, refit: bool = True, fit_config: FitConfig | None = None) -> "TabularBayesianOptimizer":
        self.update_data(new_data, new_y)
        if refit:
            self.bo.refit(fit_config=fit_config or self.fit_config)
            sync_visualization_metadata(self)
        return self

    def candidates_to_dataframe(self, candidates: Any):
        self._check_fitted()
        return tensor_to_dataframe(
            candidates,
            self.dataset.feature_names,
            inverse_category_maps=self.dataset.inverse_category_maps,
            decode_categories=self.data_config.return_original_categories,
        )

    @property
    def train_X(self) -> Any | None:
        return None if self.dataset is None else self.dataset.X

    @property
    def train_Y(self) -> Any | None:
        return None if self.dataset is None else self.dataset.Y


__all__ = ["TabularBayesianOptimizer"]
