'''Pandas / numpy friendly wrapper around :class:`bochan.api.BayesianOptimizer`.

The public API accepts tabular options as direct keyword arguments.  Internally
these values are normalized to ``TabularDataConfig`` so the lower-level
conversion helpers can remain small and testable.
'''

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _make_tabular_data_config(
    data_config: TabularDataConfig | None = None,
    *,
    input_cols: Sequence[ColumnKey] | None = None,
    target_cols: Sequence[ColumnKey] | ColumnKey | None = None,
    categorical_cols: Sequence[ColumnKey] | None = None,
    bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
    dtype: Any | None = None,
    device: Any | None = None,
    dropna: bool | None = None,
    encode_categories: bool | None = None,
    category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
    return_original_categories: bool | None = None,
) -> TabularDataConfig:
    '''Merge direct keyword arguments with an optional config object.'''

    base = data_config or TabularDataConfig()
    return replace(
        base,
        input_cols=base.input_cols if input_cols is None else input_cols,
        target_cols=base.target_cols if target_cols is None else target_cols,
        categorical_cols=base.categorical_cols if categorical_cols is None else categorical_cols,
        bounds=base.bounds if bounds is None else bounds,
        dtype=base.dtype if dtype is None else dtype,
        device=base.device if device is None else device,
        dropna=base.dropna if dropna is None else bool(dropna),
        encode_categories=base.encode_categories if encode_categories is None else bool(encode_categories),
        category_maps=base.category_maps if category_maps is None else category_maps,
        return_original_categories=(
            base.return_original_categories
            if return_original_categories is None
            else bool(return_original_categories)
        ),
    )


class TabularBayesianOptimizer:
    '''BayesianOptimizer wrapper for DataFrame / numpy / CSV workflows.

    Args:
        model_config: Core bochan model configuration.
        fit_config: Optional fitting configuration.
        input_cols: Feature columns to use from a DataFrame.  If omitted,
            all non-target columns are used.
        target_cols: Target column or columns.  Required when fitting from a
            DataFrame unless supplied later in ``fit(...)``.
        categorical_cols: Categorical feature columns.  These are converted to
            ``ModelConfig.cat_dims`` when ``model_config.cat_dims`` is omitted.
        bounds: Optional column-name mapping or ``2 x d`` bounds array.
        dtype: Torch dtype or dtype name.  Defaults to ``torch.double``.
        device: Optional torch device.
        dropna: Whether to drop rows with missing input / target values.
        encode_categories: Whether to encode string categorical columns.
        category_maps: Optional explicit category encoders.
        return_original_categories: Decode encoded categories in returned
            candidate DataFrames when possible.
        data_config: Optional low-level config object kept for backward
            compatibility.  Direct keyword arguments take precedence.
        data: Optional DataFrame / array to store for later ``fit()``.
        **bo_kwargs: Forwarded to ``BayesianOptimizer``.
    '''

    def __init__(
        self,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        *,
        input_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | ColumnKey | None = None,
        categorical_cols: Sequence[ColumnKey] | None = None,
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
        dropna: bool | None = None,
        encode_categories: bool | None = None,
        category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        return_original_categories: bool | None = None,
        data_config: TabularDataConfig | None = None,
        data: Any | None = None,
        **bo_kwargs: Any,
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.data_config = _make_tabular_data_config(
            data_config,
            input_cols=input_cols,
            target_cols=target_cols,
            categorical_cols=categorical_cols,
            bounds=bounds,
            dtype=dtype,
            device=device,
            dropna=dropna,
            encode_categories=encode_categories,
            category_maps=category_maps,
            return_original_categories=return_original_categories,
        )
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
        input_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | ColumnKey | None = None,
        categorical_cols: Sequence[ColumnKey] | None = None,
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
        dropna: bool | None = None,
        encode_categories: bool | None = None,
        category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        return_original_categories: bool | None = None,
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
            input_cols=input_cols,
            target_cols=target_cols,
            categorical_cols=categorical_cols,
            bounds=bounds,
            dtype=dtype,
            device=device,
            dropna=dropna,
            encode_categories=encode_categories,
            category_maps=category_maps,
            return_original_categories=return_original_categories,
            data_config=data_config,
            data=data,
            **bo_kwargs,
        )

    def _to_dataset(
        self,
        data: Any,
        y: Any | None = None,
        *,
        data_config: TabularDataConfig | None = None,
        feature_names: Sequence[ColumnKey] | None = None,
        target_names: Sequence[ColumnKey] | None = None,
    ) -> TabularDataset:
        config = data_config or self.data_config
        try:
            import pandas as pd
        except ImportError:
            pd = None

        if pd is not None and isinstance(data, pd.DataFrame):
            return dataframe_to_tensors(data, config)

        return numpy_to_tensors(
            data,
            y,
            config,
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
        input_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | ColumnKey | None = None,
        categorical_cols: Sequence[ColumnKey] | None = None,
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
        dropna: bool | None = None,
        encode_categories: bool | None = None,
        category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        return_original_categories: bool | None = None,
        data_config: TabularDataConfig | None = None,
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

        resolved_data_config = _make_tabular_data_config(
            data_config or self.data_config,
            input_cols=input_cols,
            target_cols=target_cols,
            categorical_cols=categorical_cols,
            bounds=bounds,
            dtype=dtype,
            device=device,
            dropna=dropna,
            encode_categories=encode_categories,
            category_maps=category_maps,
            return_original_categories=return_original_categories,
        )
        self.data_config = resolved_data_config

        dataset = self._to_dataset(
            data,
            y,
            data_config=resolved_data_config,
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
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
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
