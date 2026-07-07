'''Pandas / numpy friendly wrapper around :class:`bochan.api.BayesianOptimizer`.

The public API accepts model / fit / acquisition / optimization / repair
options as direct keyword arguments. Internally those values are normalized to
existing bochan config dataclasses, so the tensor-based core API remains
unchanged.
'''

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    CandidateRepairConfig,
    DataContext,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
)

from .builders import (
    UNSET,
    make_acquisition_config,
    make_fit_config,
    make_model_config,
    make_optimize_config,
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
    target_categorical_cols: Sequence[ColumnKey] | None = None,
    bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
    dtype: Any | None = None,
    device: Any | None = None,
    dropna: bool | None = None,
    missing_strategy: str | None = None,
    continuous_impute_strategy: str | None = None,
    categorical_impute_strategy: str | None = None,
    impute_targets: bool | None = None,
    impute_random_state: int | None = None,
    impute_max_iter: int | None = None,
    multiple_impute_sample_posterior: bool | None = None,
    encode_categories: bool | None = None,
    category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
    target_category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
    return_original_categories: bool | None = None,
) -> TabularDataConfig:
    '''Merge direct tabular keyword arguments with an optional config object.'''

    base = data_config or TabularDataConfig()
    return replace(
        base,
        input_cols=base.input_cols if input_cols is None else input_cols,
        target_cols=base.target_cols if target_cols is None else target_cols,
        categorical_cols=base.categorical_cols if categorical_cols is None else categorical_cols,
        target_categorical_cols=(
            base.target_categorical_cols if target_categorical_cols is None else target_categorical_cols
        ),
        bounds=base.bounds if bounds is None else bounds,
        dtype=base.dtype if dtype is None else dtype,
        device=base.device if device is None else device,
        dropna=base.dropna if dropna is None else bool(dropna),
        missing_strategy=base.missing_strategy if missing_strategy is None else missing_strategy,
        continuous_impute_strategy=(
            base.continuous_impute_strategy
            if continuous_impute_strategy is None
            else continuous_impute_strategy
        ),
        categorical_impute_strategy=(
            base.categorical_impute_strategy
            if categorical_impute_strategy is None
            else categorical_impute_strategy
        ),
        impute_targets=base.impute_targets if impute_targets is None else bool(impute_targets),
        impute_random_state=(
            base.impute_random_state if impute_random_state is None else impute_random_state
        ),
        impute_max_iter=base.impute_max_iter if impute_max_iter is None else int(impute_max_iter),
        multiple_impute_sample_posterior=(
            base.multiple_impute_sample_posterior
            if multiple_impute_sample_posterior is None
            else bool(multiple_impute_sample_posterior)
        ),
        encode_categories=base.encode_categories if encode_categories is None else bool(encode_categories),
        category_maps=base.category_maps if category_maps is None else category_maps,
        target_category_maps=(
            base.target_category_maps if target_category_maps is None else target_category_maps
        ),
        return_original_categories=(
            base.return_original_categories
            if return_original_categories is None
            else bool(return_original_categories)
        ),
    )


def _as_prediction_array(value: Any):
    '''Convert tensor-like prediction values to a numpy array without exposing torch.'''

    if value is None:
        return None
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _target_names_or_default(target_names: Sequence[ColumnKey] | None) -> list[str]:
    names = [str(name) for name in (target_names or [])]
    return names or ["prediction"]


def _prediction_column_names(
    *,
    kind: str,
    tail_shape: tuple[int, ...],
    target_names: Sequence[ColumnKey] | None,
    task_type: str | None,
) -> list[str]:
    '''Build flat DataFrame column names for prediction mean / variance arrays.'''

    import numpy as np

    names = _target_names_or_default(target_names)
    task = str(task_type or "").lower()

    if not tail_shape:
        return [f"{names[0]}_{kind}"]

    if len(tail_shape) == 1:
        width = tail_shape[0]
        if width == len(names):
            return [f"{name}_{kind}" for name in names]
        if task in {"multiclass", "ordinal"} and len(names) == 1:
            return [f"{names[0]}_class_{i}_{kind}" for i in range(width)]
        if len(names) == 1 and width == 1:
            return [f"{names[0]}_{kind}"]
        return [f"output_{i}_{kind}" for i in range(width)]

    if len(tail_shape) == 2 and task in {"multiclass", "ordinal"}:
        n_outputs, n_classes = tail_shape
        columns: list[str] = []
        for output_idx in range(n_outputs):
            base = names[output_idx] if output_idx < len(names) else f"output_{output_idx}"
            columns.extend(f"{base}_class_{class_idx}_{kind}" for class_idx in range(n_classes))
        return columns

    columns = []
    for index in np.ndindex(tail_shape):
        suffix = "_".join(str(i) for i in index)
        columns.append(f"output_{suffix}_{kind}")
    return columns


def _prediction_array_to_frame(
    value: Any,
    *,
    kind: str,
    target_names: Sequence[ColumnKey] | None,
    task_type: str | None,
):
    '''Convert one prediction array, e.g. mean or variance, to a DataFrame.'''

    import pandas as pd

    array = _as_prediction_array(value)
    if array is None:
        return pd.DataFrame()
    n_rows = array.shape[0]
    tail_shape = tuple(int(dim) for dim in array.shape[1:])
    flat = array.reshape(n_rows, -1)
    columns = _prediction_column_names(
        kind=kind,
        tail_shape=tail_shape,
        target_names=target_names,
        task_type=task_type,
    )
    if len(columns) != flat.shape[1]:
        columns = [f"{kind}_{i}" for i in range(flat.shape[1])]
    return pd.DataFrame(flat, columns=columns)


class TabularBayesianOptimizer:
    '''BayesianOptimizer wrapper for DataFrame / numpy / CSV workflows.'''

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
        *,
        model_cls: type | Any | None = UNSET,
        model_factory: Any = UNSET,
        task_type: str | Any = UNSET,
        model_type: str | Any = UNSET,
        input_type: str | None | Any = UNSET,
        cat_dims: Sequence[int] | None | Any = UNSET,
        input_transform: Any = UNSET,
        input_transform_config: InputTransformConfig | None | Any = UNSET,
        outcome_transform: bool | Any = UNSET,
        model_kwargs: dict[str, Any] | Any = UNSET,
        multi_output_config: MultiOutputConfig | None | Any = UNSET,
        pass_train_data: bool | Any = UNSET,
        pass_cat_dims: bool | None | Any = UNSET,
        pass_input_transform: bool | Any = UNSET,
        pass_outcome_transform: bool | Any = UNSET,
        train_x_name: str | Any = UNSET,
        train_y_name: str | Any = UNSET,
        fit_method: str | Any = UNSET,
        num_epochs: int | None | Any = UNSET,
        lr: float | None | Any = UNSET,
        batch_size: int | None | Any = UNSET,
        shuffle: bool | Any = UNSET,
        verbose: bool | Any = UNSET,
        clip_grad_norm: float | None | Any = UNSET,
        maxiter: int | None | Any = UNSET,
        fit_optimizer_kwargs: dict[str, Any] | Any = UNSET,
        fit_kwargs: dict[str, Any] | Any = UNSET,
        mll_kwargs: dict[str, Any] | Any = UNSET,
        skip_fit: bool | Any = UNSET,
        fit_func: Any = UNSET,
        mll_factory: Any = UNSET,
        mll_cls: Any = UNSET,
        use_model_make_mll: bool | Any = UNSET,
        input_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | ColumnKey | None = None,
        categorical_cols: Sequence[ColumnKey] | None = None,
        target_categorical_cols: Sequence[ColumnKey] | None = None,
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
        dropna: bool | None = None,
        missing_strategy: str | None = None,
        continuous_impute_strategy: str | None = None,
        categorical_impute_strategy: str | None = None,
        impute_targets: bool | None = None,
        impute_random_state: int | None = None,
        impute_max_iter: int | None = None,
        multiple_impute_sample_posterior: bool | None = None,
        encode_categories: bool | None = None,
        category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        target_category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        return_original_categories: bool | None = None,
        data_config: TabularDataConfig | None = None,
        data: Any | None = None,
        **bo_kwargs: Any,
    ) -> None:
        self.model_config = make_model_config(
            model_config,
            model_cls=model_cls,
            model_factory=model_factory,
            task_type=task_type,
            model_type=model_type,
            input_type=input_type,
            cat_dims=cat_dims,
            input_transform=input_transform,
            input_transform_config=input_transform_config,
            outcome_transform=outcome_transform,
            model_kwargs=model_kwargs,
            multi_output_config=multi_output_config,
            pass_train_data=pass_train_data,
            pass_cat_dims=pass_cat_dims,
            pass_input_transform=pass_input_transform,
            pass_outcome_transform=pass_outcome_transform,
            train_x_name=train_x_name,
            train_y_name=train_y_name,
        )
        self.fit_config = make_fit_config(
            fit_config,
            fit_method=fit_method,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=verbose,
            clip_grad_norm=clip_grad_norm,
            maxiter=maxiter,
            fit_optimizer_kwargs=fit_optimizer_kwargs,
            fit_kwargs=fit_kwargs,
            mll_kwargs=mll_kwargs,
            skip_fit=skip_fit,
            fit_func=fit_func,
            mll_factory=mll_factory,
            mll_cls=mll_cls,
            use_model_make_mll=use_model_make_mll,
        )
        self.data_config = _make_tabular_data_config(
            data_config,
            input_cols=input_cols,
            target_cols=target_cols,
            categorical_cols=categorical_cols,
            target_categorical_cols=target_categorical_cols,
            bounds=bounds,
            dtype=dtype,
            device=device,
            dropna=dropna,
            missing_strategy=missing_strategy,
            continuous_impute_strategy=continuous_impute_strategy,
            categorical_impute_strategy=categorical_impute_strategy,
            impute_targets=impute_targets,
            impute_random_state=impute_random_state,
            impute_max_iter=impute_max_iter,
            multiple_impute_sample_posterior=multiple_impute_sample_posterior,
            encode_categories=encode_categories,
            category_maps=category_maps,
            target_category_maps=target_category_maps,
            return_original_categories=return_original_categories,
        )
        self.data = data
        self.bo_kwargs = dict(bo_kwargs)
        self.bo = BayesianOptimizer(
            model_config=self.model_config,
            fit_config=self.fit_config,
            **bo_kwargs,
        )
        self.dataset: TabularDataset | None = None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        read_csv_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TabularBayesianOptimizer":
        '''Create an optimizer with data loaded from a CSV file.'''

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for TabularBayesianOptimizer.from_csv().") from exc
        data = pd.read_csv(path, **(read_csv_kwargs or {}))
        return cls(data=data, **kwargs)

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

    def _check_tabular_fitted(self) -> None:
        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")

    def _visualization_feature_cols(
        self,
        feature_cols: Sequence[ColumnKey] | None = None,
    ) -> list[ColumnKey]:
        self._check_tabular_fitted()
        assert self.dataset is not None
        return list(self.dataset.feature_names if feature_cols is None else feature_cols)

    def _visualization_target_cols(
        self,
        target_cols: Sequence[ColumnKey] | None = None,
    ) -> list[ColumnKey]:
        self._check_tabular_fitted()
        assert self.dataset is not None
        return list(self.dataset.target_names if target_cols is None else target_cols)

    def _resolve_plot_target(
        self,
        target: ColumnKey | None,
        target_cols: Sequence[ColumnKey],
        *,
        name: str = "target",
    ) -> ColumnKey:
        if target is not None:
            return target
        if len(target_cols) == 1:
            return target_cols[0]
        raise ValueError(f"{name} must be specified when multiple target columns are available: {list(target_cols)!r}.")

    def _sync_visualization_metadata(self) -> None:
        '''Attach tabular column names and labels to the underlying optimizer bundle.'''

        if self.dataset is None or self.bo.bundle is None:
            return
        metadata = dict(getattr(self.bo.bundle, "metadata", {}) or {})
        metadata["feature_cols"] = list(self.dataset.feature_names)
        metadata["target_cols"] = list(self.dataset.target_names)

        if self.dataset.category_maps:
            labels = dict(metadata.get("labels") or {})
            labels.update(self.dataset.category_maps)
            metadata["labels"] = labels

        if self.dataset.inverse_target_category_maps:
            class_label_map: dict[int, list[Any]] = {}
            for output_idx, target_name in enumerate(self.dataset.target_names):
                inverse = self.dataset.inverse_target_category_maps.get(target_name)
                if inverse is None:
                    inverse = self.dataset.inverse_target_category_maps.get(str(target_name))
                if inverse:
                    class_label_map[output_idx] = [inverse[key] for key in sorted(inverse)]
            if len(class_label_map) == 1:
                metadata["class_labels"] = next(iter(class_label_map.values()))
            elif class_label_map:
                metadata["class_labels"] = class_label_map

        self.bo.bundle.metadata = metadata

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        input_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | ColumnKey | None = None,
        categorical_cols: Sequence[ColumnKey] | None = None,
        target_categorical_cols: Sequence[ColumnKey] | None = None,
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
        dropna: bool | None = None,
        missing_strategy: str | None = None,
        continuous_impute_strategy: str | None = None,
        categorical_impute_strategy: str | None = None,
        impute_targets: bool | None = None,
        impute_random_state: int | None = None,
        impute_max_iter: int | None = None,
        multiple_impute_sample_posterior: bool | None = None,
        encode_categories: bool | None = None,
        category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        target_category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None,
        return_original_categories: bool | None = None,
        data_config: TabularDataConfig | None = None,
        feature_names: Sequence[ColumnKey] | None = None,
        target_names: Sequence[ColumnKey] | None = None,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
        task_type: str | Any = UNSET,
        model_type: str | Any = UNSET,
        cat_dims: Sequence[int] | None | Any = UNSET,
        model_kwargs: dict[str, Any] | Any = UNSET,
        input_transform_config: InputTransformConfig | None | Any = UNSET,
        outcome_transform: bool | Any = UNSET,
        fit_method: str | Any = UNSET,
        num_epochs: int | None | Any = UNSET,
        lr: float | None | Any = UNSET,
        batch_size: int | None | Any = UNSET,
        maxiter: int | None | Any = UNSET,
        skip_fit: bool | Any = UNSET,
    ) -> "TabularBayesianOptimizer":
        '''Fit from a pandas DataFrame or numpy-like arrays.'''

        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data was supplied. Pass data to fit(...) or use from_csv(...).")

        self.model_config = make_model_config(
            model_config or self.model_config,
            task_type=task_type,
            model_type=model_type,
            cat_dims=cat_dims,
            model_kwargs=model_kwargs,
            input_transform_config=input_transform_config,
            outcome_transform=outcome_transform,
        )
        self.fit_config = make_fit_config(
            fit_config or self.fit_config,
            fit_method=fit_method,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            maxiter=maxiter,
            skip_fit=skip_fit,
        )
        resolved_data_config = _make_tabular_data_config(
            data_config or self.data_config,
            input_cols=input_cols,
            target_cols=target_cols,
            categorical_cols=categorical_cols,
            target_categorical_cols=target_categorical_cols,
            bounds=bounds,
            dtype=dtype,
            device=device,
            dropna=dropna,
            missing_strategy=missing_strategy,
            continuous_impute_strategy=continuous_impute_strategy,
            categorical_impute_strategy=categorical_impute_strategy,
            impute_targets=impute_targets,
            impute_random_state=impute_random_state,
            impute_max_iter=impute_max_iter,
            multiple_impute_sample_posterior=multiple_impute_sample_posterior,
            encode_categories=encode_categories,
            category_maps=category_maps,
            target_category_maps=target_category_maps,
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
        self._sync_visualization_metadata()
        return self

    def candidate(
        self,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        *,
        acq_name: str | Any = UNSET,
        name: str | Any = UNSET,
        acqf_cls: type | Any = UNSET,
        acqf_factory: Any = UNSET,
        objective: Any = UNSET,
        objective_config: ObjectiveConfig | None | Any = UNSET,
        objective_factory: Any = UNSET,
        objective_kwargs: dict[str, Any] | Any = UNSET,
        sampler: Any = UNSET,
        acqf_kwargs: dict[str, Any] | Any = UNSET,
        context_fields: tuple[str, ...] | Any = UNSET,
        filter_kwargs_by_signature: bool | Any = UNSET,
        objective_mode: str | Any = UNSET,
        objective_output: Any = UNSET,
        objective_outputs: Sequence[Any] | Any = UNSET,
        objective_specs: Sequence[Any] | Any = UNSET,
        objective_directions: Sequence[Any] | Any = UNSET,
        objective_weights: Sequence[float] | Any = UNSET,
        objective_direction: Any = UNSET,
        objective_weight: float | Any = UNSET,
        objective_n_w: int | None | Any = UNSET,
        objective_risk_type: str | None | Any = UNSET,
        objective_alpha: float | Any = UNSET,
        objective_utility_values: Sequence[float] | Any = UNSET,
        q: int | Any = UNSET,
        num_restarts: int | Any = UNSET,
        raw_samples: int | Any = UNSET,
        sequential: bool | Any = UNSET,
        optimizer: Any = UNSET,
        optimizer_kwargs: dict[str, Any] | Any = UNSET,
        post_processing_func: Any = UNSET,
        fixed_features: dict[Any, float] | Any = UNSET,
        fixed_features_list: list[dict[Any, float]] | Any = UNSET,
        inequality_constraints: Any = UNSET,
        equality_constraints: Any = UNSET,
        return_best_only: bool | Any = UNSET,
        repair_config: CandidateRepairConfig | None | Any = UNSET,
        repair_bounds: Any | Mapping[ColumnKey, Sequence[float]] | Any = UNSET,
        numeric_indices: Sequence[ColumnKey] | Any = UNSET,
        steps: Any = UNSET,
        comp_idx: Sequence[ColumnKey] | Any = UNSET,
        k: int | Any = UNSET,
        repair_equality_constraints: Any = UNSET,
        repair_inequality_constraints: Any = UNSET,
        repair_inequality_sense: str | Any = UNSET,
        repair_fixed_features: dict[Any, float] | Any = UNSET,
        final_sum_constraint: Any = UNSET,
        diversify: bool | Any = UNSET,
        diversify_kwargs: dict[str, Any] | Any = UNSET,
        score: str | Any = UNSET,
        support_selection: str | Any = UNSET,
        sample_tau: float | Any = UNSET,
        sample_eps: float | Any = UNSET,
        generator: Any = UNSET,
        max_iters: int | Any = UNSET,
        num_alternations: int | Any = UNSET,
        final_priority: str | Any = UNSET,
        support_eps: float | Any = UNSET,
        data_context: DataContext | None = None,
        bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None,
        return_dataframe: bool = True,
        return_result: bool = False,
    ) -> Any:
        '''Generate candidates and optionally return them as a DataFrame.'''

        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
        acq_config = make_acquisition_config(
            acq_config,
            acq_name=acq_name,
            name=name,
            acqf_cls=acqf_cls,
            acqf_factory=acqf_factory,
            objective=objective,
            objective_config=objective_config,
            objective_factory=objective_factory,
            objective_kwargs=objective_kwargs,
            sampler=sampler,
            acqf_kwargs=acqf_kwargs,
            context_fields=context_fields,
            filter_kwargs_by_signature=filter_kwargs_by_signature,
            objective_mode=objective_mode,
            objective_output=objective_output,
            objective_outputs=objective_outputs,
            objective_specs=objective_specs,
            objective_directions=objective_directions,
            objective_weights=objective_weights,
            objective_direction=objective_direction,
            objective_weight=objective_weight,
            objective_n_w=objective_n_w,
            objective_risk_type=objective_risk_type,
            objective_alpha=objective_alpha,
            objective_utility_values=objective_utility_values,
        )
        opt_config = make_optimize_config(
            opt_config,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=sequential,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            post_processing_func=post_processing_func,
            fixed_features=fixed_features,
            fixed_features_list=fixed_features_list,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            return_best_only=return_best_only,
            repair_config=None if repair_config is UNSET else repair_config,
            repair_bounds=repair_bounds,
            numeric_indices=numeric_indices,
            steps=steps,
            comp_idx=comp_idx,
            k=k,
            repair_equality_constraints=repair_equality_constraints,
            repair_inequality_constraints=repair_inequality_constraints,
            repair_inequality_sense=repair_inequality_sense,
            repair_fixed_features=repair_fixed_features,
            final_sum_constraint=final_sum_constraint,
            diversify=diversify,
            diversify_kwargs=diversify_kwargs,
            score=score,
            support_selection=support_selection,
            sample_tau=sample_tau,
            sample_eps=sample_eps,
            generator=generator,
            max_iters=max_iters,
            num_alternations=num_alternations,
            final_priority=final_priority,
            support_eps=support_eps,
        )
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

    def predict(
        self,
        data: Any,
        *,
        return_type: str = "dataframe",
        include_input: bool = False,
        return_dataframe_input: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        '''Predict from tabular input and return mean / variance as a DataFrame by default.'''

        if self.dataset is None:
            raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
        try:
            import pandas as pd
        except ImportError:
            pd = None
        X = data
        original_index = None
        if pd is not None and isinstance(data, pd.DataFrame):
            original_index = data.index
            tmp_config = replace(
                self.data_config,
                target_cols=None,
                input_cols=self.dataset.feature_names,
            )
            X = dataframe_to_tensors(data, tmp_config).X

        normalized_return_type = str(return_type).lower()
        dataframe_return_types = {"dataframe", "df", "mean_variance_dataframe", "mean_variance_df"}
        if normalized_return_type not in dataframe_return_types:
            prediction = self.bo.predict(
                X,
                return_type=return_type,
                posterior_kwargs=posterior_kwargs,
                **kwargs,
            )
            if return_dataframe_input:
                return prediction, data
            return prediction

        result = self.bo.predict(
            X,
            return_result=True,
            posterior_kwargs=posterior_kwargs,
            **kwargs,
        )
        prediction_df = self.prediction_to_dataframe(result)
        if original_index is not None:
            prediction_df.index = original_index
        if include_input:
            input_df = self._prediction_input_to_dataframe(data, X)
            if original_index is not None:
                input_df.index = original_index
            prediction_df = input_df.join(prediction_df)
        if return_dataframe_input:
            return prediction_df, data
        return prediction_df

    def prediction_to_dataframe(self, prediction: Any):
        '''Convert a PredictionResult or posterior-like object to a pandas DataFrame.'''

        import pandas as pd

        mean = getattr(prediction, "mean", None)
        variance = getattr(prediction, "variance", None)
        task_type = getattr(prediction, "task_type", None)
        if mean is None and hasattr(prediction, "posterior"):
            posterior = getattr(prediction, "posterior")
            mean = getattr(posterior, "mean", None)
            variance = getattr(posterior, "variance", None)

        target_names = self.dataset.target_names if self.dataset is not None else []
        mean_df = _prediction_array_to_frame(
            mean,
            kind="mean",
            target_names=target_names,
            task_type=task_type,
        )
        variance_df = _prediction_array_to_frame(
            variance,
            kind="variance",
            target_names=target_names,
            task_type=task_type,
        )
        prediction_df = pd.concat([mean_df, variance_df], axis=1)
        prediction_df.attrs["task_type"] = task_type
        prediction_df.attrs["prediction_space"] = getattr(prediction, "prediction_space", None)
        prediction_df.attrs["variance_kind"] = getattr(prediction, "variance_kind", None)
        return prediction_df

    def _prediction_input_to_dataframe(self, original_data: Any, X: Any):
        '''Return input data as a DataFrame without exposing tensor internals.'''

        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(original_data, pd.DataFrame):
            return original_data.copy()
        return tensor_to_dataframe(
            X,
            self.dataset.feature_names if self.dataset is not None else [],
            inverse_category_maps=self.dataset.inverse_category_maps if self.dataset is not None else None,
            decode_categories=self.data_config.return_original_categories,
        )

    def visualization_training_dataframe(
        self,
        *,
        feature_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | None = None,
    ) -> Any:
        '''Return training X / Y DataFrames using bochan.visualization helpers.'''

        from bochan.visualization import training_dataframe

        return training_dataframe(
            self.bo,
            feature_cols=self._visualization_feature_cols(feature_cols),
            target_cols=self._visualization_target_cols(target_cols),
        )

    def visualization_candidates_dataframe(
        self,
        *,
        candidate_result: Any | None = None,
        feature_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | None = None,
        include_prediction: bool = True,
    ) -> Any:
        '''Return the latest candidate batch as a visualization-friendly DataFrame.'''

        from bochan.visualization import candidates_dataframe

        return candidates_dataframe(
            self.bo,
            candidate_result=candidate_result,
            feature_cols=self._visualization_feature_cols(feature_cols),
            target_cols=self._visualization_target_cols(target_cols),
            include_prediction=include_prediction,
        )

    def plot_yy(
        self,
        target: ColumnKey | None = None,
        *,
        feature_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | None = None,
        candidate_result: Any | None = None,
        cycle: str | Sequence[Any] | Any | None = None,
        **kwargs: Any,
    ) -> Any:
        '''Create a YY plot or multiclass correct-label probability plot.'''

        from bochan.visualization import show_yyplot_from_optimizer

        resolved_target_cols = self._visualization_target_cols(target_cols)
        resolved_target = self._resolve_plot_target(target, resolved_target_cols)
        return show_yyplot_from_optimizer(
            self.bo,
            resolved_target,
            feature_cols=self._visualization_feature_cols(feature_cols),
            target_cols=resolved_target_cols,
            candidate_result=candidate_result,
            cycle=cycle,
            **kwargs,
        )

    def plot_1d(
        self,
        feature: ColumnKey,
        target: ColumnKey | None = None,
        *,
        feature_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | None = None,
        value_dict: dict[str, Any] | None = None,
        candidate_result: Any | None = None,
        n: int = 50,
        cycle: str | Sequence[Any] | Any | None = None,
        **kwargs: Any,
    ) -> Any:
        '''Create a 1D prediction curve from the fitted tabular optimizer.'''

        from bochan.visualization import show_1dplot_from_optimizer

        resolved_target_cols = self._visualization_target_cols(target_cols)
        resolved_target = self._resolve_plot_target(target, resolved_target_cols)
        return show_1dplot_from_optimizer(
            self.bo,
            feature,
            resolved_target,
            feature_cols=self._visualization_feature_cols(feature_cols),
            target_cols=resolved_target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            n=n,
            cycle=cycle,
            **kwargs,
        )

    def plot_2d(
        self,
        feature_col1: ColumnKey,
        feature_col2: ColumnKey,
        target: ColumnKey | None = None,
        *,
        feature_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | None = None,
        value_dict: dict[str, Any] | None = None,
        candidate_result: Any | None = None,
        n: int = 25,
        show_type: str = "acqf",
        cycle: str | Sequence[Any] | Any | None = None,
        **kwargs: Any,
    ) -> Any:
        '''Create a 2D acquisition or prediction heatmap from the fitted optimizer.'''

        from bochan.visualization import show_scatter_with_acqf_from_optimizer

        resolved_target_cols = self._visualization_target_cols(target_cols)
        resolved_target = self._resolve_plot_target(target, resolved_target_cols)
        return show_scatter_with_acqf_from_optimizer(
            self.bo,
            feature_col1,
            feature_col2,
            resolved_target,
            feature_cols=self._visualization_feature_cols(feature_cols),
            target_cols=resolved_target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            n=n,
            show_type=show_type,
            cycle=cycle,
            **kwargs,
        )

    def plot_heatmap(self, *args: Any, **kwargs: Any) -> Any:
        '''Alias for plot_2d().'''

        return self.plot_2d(*args, **kwargs)

    def plot_scatter(self, *args: Any, **kwargs: Any) -> Any:
        '''Alias for plot_2d().'''

        return self.plot_2d(*args, **kwargs)

    def plot_tri(
        self,
        feature_col1: ColumnKey,
        feature_col2: ColumnKey,
        feature_col3: ColumnKey,
        target: ColumnKey | None = None,
        *,
        feature_cols: Sequence[ColumnKey] | None = None,
        target_cols: Sequence[ColumnKey] | None = None,
        value_dict: dict[str, Any] | None = None,
        candidate_result: Any | None = None,
        sum_value: float | None = None,
        n: int = 50,
        show_type: str = "acqf",
        cycle: str | Sequence[Any] | Any | None = None,
        ncontours: int = 25,
        **kwargs: Any,
    ) -> Any:
        '''Create a ternary acquisition or prediction plot from the fitted optimizer.'''

        from bochan.visualization import show_triscatter_with_acqf_from_optimizer

        resolved_target_cols = self._visualization_target_cols(target_cols)
        resolved_target = self._resolve_plot_target(target, resolved_target_cols)
        return show_triscatter_with_acqf_from_optimizer(
            self.bo,
            feature_col1,
            feature_col2,
            feature_col3,
            resolved_target,
            feature_cols=self._visualization_feature_cols(feature_cols),
            target_cols=resolved_target_cols,
            value_dict=value_dict,
            candidate_result=candidate_result,
            sum_value=sum_value,
            n=n,
            show_type=show_type,
            cycle=cycle,
            ncontours=ncontours,
            **kwargs,
        )

    def plot_ternary(self, *args: Any, **kwargs: Any) -> Any:
        '''Alias for plot_tri().'''

        return self.plot_tri(*args, **kwargs)

    def plot_pareto(
        self,
        target1: ColumnKey | None = None,
        target2: ColumnKey | None = None,
        *,
        target_cols: Sequence[ColumnKey] | None = None,
        candidate_result: Any | None = None,
        df_cand: Any | None = None,
        cycle: str | Sequence[Any] | Any | None = None,
    ) -> Any:
        '''Create a two-objective scatter plot with optional candidate predictions.'''

        from bochan.visualization import show_pareto_plot

        resolved_target_cols = self._visualization_target_cols(target_cols)
        if target1 is None or target2 is None:
            if len(resolved_target_cols) < 2:
                raise ValueError("target1 and target2 are required when fewer than two target columns are available.")
            target1 = resolved_target_cols[0] if target1 is None else target1
            target2 = resolved_target_cols[1] if target2 is None else target2
        _, y_df = self.visualization_training_dataframe(target_cols=resolved_target_cols)
        if df_cand is None:
            df_cand = self.visualization_candidates_dataframe(
                candidate_result=candidate_result,
                target_cols=resolved_target_cols,
                include_prediction=True,
            )
        return show_pareto_plot(y_df, target1, target2, df_cand=df_cand, cycle=cycle)

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
            self._sync_visualization_metadata()
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
