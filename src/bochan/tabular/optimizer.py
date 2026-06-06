'''Pandas / numpy friendly wrapper around :class:`bochan.api.BayesianOptimizer`.'''

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    DataContext,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)

from .config import ColumnKey, TabularDataConfig
from .converter import (
    TabularDataset,
    dataframe_to_tensors,
    numpy_to_tensors,
    resolve_dtype,
    resolve_optimize_config_columns,
    tensor_to_dataframe,
)


class TabularBayesianOptimizer:
    '''BayesianOptimizer wrapper for DataFrame / numpy / CSV workflows.'''

    def __init__(
        self,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        *,
        data_config: TabularDataConfig | None = None,
        data: Any | None = None,
        **bo_kwargs: Any,
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.data_config = data_config or TabularDataConfig()
        self.data = data

        self.bo_kwargs = dict(bo_kwargs)
        self.bo = BayesianOptimizer(
            model_config=model_config,
            fit_config=fit_config,
            **bo_kwargs,
        )

        self.dataset: TabularDataset | None = None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        data_config: TabularDataConfig | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
        **bo_kwargs: Any,
    ) -> "TabularBayesianOptimizer":
        '''Create an optimizer with data loaded from a CSV file.'''

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for TabularBayesianOptimizer.from_csv().") from exc

        data = pd.read_csv(path, **(read_csv_kwargs or {}))
        return cls(
            model_config=model_config,
            fit_config=fit_config,
            data_config=data_config,
            data=data,
            **bo_kwargs,
        )

    def _to_dataset(
        self,
        data: Any,
        y: Any | None = None,
        *,
        feature_names: Sequence[ColumnKey] | None = None,
        target_names: Sequence[ColumnKey] | None = None,
    ) -> TabularDataset:
        try:
            import pandas as pd
        except ImportError:
            pd = None

        if pd is not None and isinstance(data, pd.DataFrame):
            return dataframe_to_tensors(data, self.data_config)

        return numpy_to_tensors(
            data,
            y,
            self.data_config,
            feature_names=feature_names,
            target_names=target_names,
        )

    def _model_config_with_tabular_cat_dims(self, dataset: TabularDataset) -> ModelConfig:
        if self.model_config.cat_dims is not None:
            return self.model_config
        if not dataset.cat_dims:
            return self.model_config
        return replace(self.model_config, cat_dims=dataset.cat_dims)

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        feature_names: Sequence[ColumnKey] | None = None,
        target_names: Sequence[ColumnKey] | None = None,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> "TabularBayesianOptimizer":
        '''Fit from a pandas DataFrame or numpy-like arrays.'''

        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data was supplied. Pass data to fit(...) or use from_csv(...).")

        if model_config is not None:
            self.model_config = model_config
        if fit_config is not None:
            self.fit_config = fit_config

        dataset = self._to_dataset(
            data,
            y,
            feature_names=feature_names,
            target_names=target_names,
        )
        if dataset.Y is None:
            raise ValueError("Target values are required for fit(). Set target_cols or pass y.")

        self.dataset = dataset
        resolved_model_config = self._model_config_with_tabular_cat_dims(dataset)

        self.bo.fit(
            dataset.X,
            dataset.Y,
            model_config=resolved_model_config,
            fit_config=self.fit_config,
        )

        if dataset.bounds is not None:
            self.bo.set_bounds(dataset.bounds)

        return self

    def candidate(
        self,
        acq_config: AcquisitionConfig,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_dataframe: bool = True,
        return_result: bool = False,
    ) -> Any:
        '''Generate candidates and optionally return them as a DataFrame.'''

        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")

        dtype = resolve_dtype(self.data_config.dtype)
        opt_config = resolve_optimize_config_columns(
            opt_config,
            self.dataset.feature_names,
            dtype=dtype,
            device=self.data_config.device,
        )

        call_bounds = bounds
        if call_bounds is None:
            call_bounds = self.dataset.bounds

        result = self.bo.candidate(
            acq_config,
            opt_config,
            data_context=data_context,
            bounds=call_bounds,
            return_result=return_result,
        )

        if return_result:
            return result

        candidates, acq_value = result
        if not return_dataframe:
            return candidates, acq_value

        candidates_df = self.candidates_to_dataframe(candidates)
        return candidates_df, acq_value

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        '''Alias for candidate().'''

        return self.candidate(*args, **kwargs)

    def predict(self, data: Any, *, return_dataframe_input: bool = False, **kwargs: Any) -> Any:
        '''Predict from tabular input or raw tensor input.'''

        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")

        try:
            import pandas as pd
        except ImportError:
            pd = None

        X = data
        if pd is not None and isinstance(data, pd.DataFrame):
            tmp_config = replace(
                self.data_config,
                target_cols=None,
                input_cols=self.dataset.feature_names,
            )
            X = dataframe_to_tensors(data, tmp_config).X

        prediction = self.bo.predict(X, **kwargs)
        if return_dataframe_input:
            return prediction, data
        return prediction

    def update_data(self, new_data: Any, new_y: Any | None = None) -> "TabularBayesianOptimizer":
        '''Append new observations to the underlying tensor optimizer state.'''

        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
        new_dataset = self._to_dataset(new_data, new_y, feature_names=self.dataset.feature_names)
        if new_dataset.Y is None:
            raise ValueError("Target values are required for update_data().")
        self.bo.update_data(new_dataset.X, new_dataset.Y)
        return self

    def tell(
        self,
        new_data: Any,
        new_y: Any | None = None,
        *,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> "TabularBayesianOptimizer":
        '''Append observations and optionally refit the model.'''

        self.update_data(new_data, new_y)
        if refit:
            self.bo.refit(fit_config=fit_config or self.fit_config)
        return self

    def candidates_to_dataframe(self, candidates: Any):
        '''Convert candidate tensor output to a pandas DataFrame.'''

        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
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
