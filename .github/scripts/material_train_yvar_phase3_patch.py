from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:160]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


# Canonical tabular config and dataset.
replace_once(
    "src/bochan/tabular/config/data.py",
    "    target_cols: Sequence[ColumnKey] | ColumnKey | None = None\n    categorical_cols: Sequence[ColumnKey] = field(default_factory=list)\n",
    "    target_cols: Sequence[ColumnKey] | ColumnKey | None = None\n    target_variance_cols: Sequence[ColumnKey] | ColumnKey | None = None\n    categorical_cols: Sequence[ColumnKey] = field(default_factory=list)\n",
)
replace_once(
    "src/bochan/tabular/data/dataset.py",
    "    target_names: list[ColumnKey]\n    cat_dims: list[int]\n    bounds: Any | None = None\n",
    "    target_names: list[ColumnKey]\n    cat_dims: list[int]\n    Yvar: Any | None = None\n    bounds: Any | None = None\n",
)

# DataFrame variance-column extraction. Variance columns are metadata, never features.
replace_once(
    "src/bochan/tabular/data/conversion.py",
    '''    target_cols = _as_list(config.target_cols)
    input_cols = (
        [column for column in data.columns if column not in target_cols]
        if config.input_cols is None
        else _as_list(config.input_cols)
    )
    if not input_cols:
        raise ValueError("input_cols could not be inferred. Pass TabularDataConfig.input_cols.")

    selected_cols = list(dict.fromkeys(input_cols + target_cols))
''',
    '''    target_cols = _as_list(config.target_cols)
    target_variance_cols = _as_list(config.target_variance_cols)
    if target_variance_cols and not target_cols:
        raise ValueError("target_variance_cols requires target_cols.")
    if target_variance_cols and len(target_variance_cols) != len(target_cols):
        raise ValueError(
            "target_variance_cols must contain exactly one variance column per target column."
        )
    overlap = sorted(set(target_cols).intersection(target_variance_cols), key=str)
    if overlap:
        raise ValueError(
            "target_variance_cols must be distinct from target_cols; "
            f"overlap={overlap!r}."
        )
    excluded = set(target_cols) | set(target_variance_cols)
    input_cols = (
        [column for column in data.columns if column not in excluded]
        if config.input_cols is None
        else _as_list(config.input_cols)
    )
    variance_inputs = sorted(set(input_cols).intersection(target_variance_cols), key=str)
    if variance_inputs:
        raise ValueError(
            "target_variance_cols must not be included in input_cols; "
            f"overlap={variance_inputs!r}."
        )
    if not input_cols:
        raise ValueError("input_cols could not be inferred. Pass TabularDataConfig.input_cols.")

    selected_cols = list(dict.fromkeys(input_cols + target_cols + target_variance_cols))
''',
)
replace_once(
    "src/bochan/tabular/data/conversion.py",
    '''    X_df = work.loc[:, input_cols].copy()
    Y_df = work.loc[:, target_cols].copy() if target_cols else None

    category_maps, inverse_maps = _encode_dataframe_category_columns(
''',
    '''    X_df = work.loc[:, input_cols].copy()
    Y_df = work.loc[:, target_cols].copy() if target_cols else None
    Yvar_df = (
        work.loc[:, target_variance_cols].copy()
        if target_variance_cols
        else None
    )

    category_maps, inverse_maps = _encode_dataframe_category_columns(
''',
)
replace_once(
    "src/bochan/tabular/data/conversion.py",
    '''    Y = None
    if Y_df is not None:
        Y = _to_tensor(Y_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

    feature_names = list(input_cols)
    return TabularDataset(
        X=X,
        Y=Y,
''',
    '''    Y = None
    if Y_df is not None:
        Y = _to_tensor(Y_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

    Yvar = None
    if Yvar_df is not None:
        non_numeric = [
            column
            for column in target_variance_cols
            if not pd.api.types.is_numeric_dtype(Yvar_df.loc[:, column])
        ]
        if non_numeric:
            raise TypeError(
                "target_variance_cols must be numeric variance columns; "
                f"non_numeric={non_numeric!r}."
            )
        variance_values = Yvar_df.to_numpy(dtype=float)
        np = _numpy()
        if not np.isfinite(variance_values).all():
            raise ValueError("target variance values must be finite.")
        if (variance_values <= 0.0).any():
            raise ValueError("target variance values must be strictly positive.")
        Yvar = _to_tensor(variance_values, dtype=dtype, device=config.device)
        if Yvar.ndim == 1:
            Yvar = Yvar.reshape(-1, 1)
        if Y is None or Yvar.shape != Y.shape:
            raise ValueError(
                "Target variance shape must match target shape after tabular conversion."
            )

    feature_names = list(input_cols)
    return TabularDataset(
        X=X,
        Y=Y,
        Yvar=Yvar,
''',
)

# Generic model builder: forward scalar/wide Yvar and slice independent outputs.
p = "src/bochan/api/modeling/build.py"
replace_once(
    p,
    "def _build_model_kwargs(train_X: Any, train_Y: Any, config: ModelConfig, cat_dims: list[int]) -> dict[str, Any]:\n",
    '''def _build_model_kwargs(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any | None,
    config: ModelConfig,
    cat_dims: list[int],
) -> dict[str, Any]:
''',
)
replace_once(
    p,
    '''    if config.pass_train_data:
        kwargs[config.train_x_name] = train_X
        kwargs[config.train_y_name] = train_Y

    pass_cat_dims = config.pass_cat_dims
''',
    '''    if config.pass_train_data:
        kwargs[config.train_x_name] = train_X
        kwargs[config.train_y_name] = train_Y
        if train_Yvar is not None:
            if "train_Yvar" in config.model_kwargs:
                raise ValueError(
                    "train_Yvar was supplied both as training data and model_kwargs."
                )
            kwargs["train_Yvar"] = train_Yvar

    pass_cat_dims = config.pass_cat_dims
''',
)
replace_once(
    p,
    '''def _build_single_model(
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
''',
    '''def _build_single_model(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any | None,
    config: ModelConfig,
''',
)
replace_once(
    p,
    "    kwargs = _build_model_kwargs(train_X, train_Y, config, cat_dims)\n",
    "    kwargs = _build_model_kwargs(train_X, train_Y, train_Yvar, config, cat_dims)\n",
)
replace_once(
    p,
    "def build_multi_output_model(train_X: Any, train_Y: Any, config: ModelConfig, *, model_registry: Mapping[Any, Any] | None = None) -> ModelBundle:\n",
    '''def build_multi_output_model(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any | None,
    config: ModelConfig,
    *,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
''',
)
replace_once(
    p,
    '''    for i, output_config in enumerate(output_configs):
        output_train_Y = _slice_output_y(train_Y, i, dim=mo_config.train_y_slice_dim)
        sub_bundles.append(_build_single_model(train_X=train_X, train_Y=output_train_Y, config=output_config, model_registry=model_registry))
''',
    '''    for i, output_config in enumerate(output_configs):
        output_train_Y = _slice_output_y(train_Y, i, dim=mo_config.train_y_slice_dim)
        output_train_Yvar = (
            None
            if train_Yvar is None
            else _slice_output_y(train_Yvar, i, dim=mo_config.train_y_slice_dim)
        )
        sub_bundles.append(
            _build_single_model(
                train_X=train_X,
                train_Y=output_train_Y,
                train_Yvar=output_train_Yvar,
                config=output_config,
                model_registry=model_registry,
            )
        )
''',
)
replace_once(
    p,
    '''def build_model(train_X: Any, train_Y: Any, config: ModelConfig, *, model_registry: Mapping[Any, Any] | None = None) -> ModelBundle:
    if config.multi_output_config is not None:
        return build_multi_output_model(train_X, train_Y, config, model_registry=model_registry)
    return _build_single_model(train_X, train_Y, config, model_registry=model_registry)
''',
    '''def _validate_train_yvar_shape(train_Y: Any, train_Yvar: Any | None) -> None:
    if train_Yvar is None:
        return
    if not hasattr(train_Y, "shape") or not hasattr(train_Yvar, "shape"):
        raise TypeError("train_Y and train_Yvar must expose shape attributes.")
    y_shape = tuple(train_Y.shape)
    yvar_shape = tuple(train_Yvar.shape)
    if len(y_shape) == 1:
        y_shape = (y_shape[0], 1)
    if len(yvar_shape) == 1:
        yvar_shape = (yvar_shape[0], 1)
    if y_shape != yvar_shape:
        raise ValueError(
            "train_Yvar must match train_Y shape; "
            f"got train_Y={tuple(train_Y.shape)!r}, train_Yvar={tuple(train_Yvar.shape)!r}."
        )


def build_model(
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    *,
    train_Yvar: Any | None = None,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
    _validate_train_yvar_shape(train_Y, train_Yvar)
    if config.multi_output_config is not None:
        return build_multi_output_model(
            train_X,
            train_Y,
            train_Yvar,
            config,
            model_registry=model_registry,
        )
    return _build_single_model(
        train_X,
        train_Y,
        train_Yvar,
        config,
        model_registry=model_registry,
    )
''',
)

# High-level BayesianOptimizer lifecycle.
p = "src/bochan/api/optimizer/core.py"
replace_once(
    p,
    "        self.train_X: Any | None = None\n        self.train_Y: Any | None = None\n",
    "        self.train_X: Any | None = None\n        self.train_Y: Any | None = None\n        self.train_Yvar: Any | None = None\n",
)
replace_once(
    p,
    '''    def fit(
        self,
        train_X: Any,
        train_Y: Any,
        *,
''',
    '''    def fit(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any | None = None,
        *,
''',
)
replace_once(
    p,
    "        self.train_X = train_X\n        self.train_Y = train_Y\n",
    "        self.train_X = train_X\n        self.train_Y = train_Y\n        self.train_Yvar = train_Yvar\n",
)
replace_once(
    p,
    '''        self.bundle = build_model(
            train_X=train_X,
            train_Y=train_Y,
            config=self.model_config,
''',
    '''        self.bundle = build_model(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            config=self.model_config,
''',
)
replace_once(
    p,
    "        return self.fit(self.train_X, self.train_Y, fit_config=fit_config or self.fit_config)\n",
    '''        return self.fit(
            self.train_X,
            self.train_Y,
            self.train_Yvar,
            fit_config=fit_config or self.fit_config,
        )
''',
)
replace_once(
    p,
    '''    def cross_validate(
        self,
        train_X: Any,
        train_Y: Any,
        *,
''',
    '''    def cross_validate(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any | None = None,
        *,
''',
)
replace_once(
    p,
    '''            train_X,
            train_Y,
            model_config=model_config,
''',
    '''            train_X,
            train_Y,
            train_Yvar,
            model_config=model_config,
''',
)
replace_once(
    p,
    '''    def tell(
        self,
        new_X: Any,
        new_Y: Any,
        *,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        """新しい観測データを追加し、必要なら再学習する。"""
        self.update_data(new_X, new_Y)
''',
    '''    def tell(
        self,
        new_X: Any,
        new_Y: Any,
        new_Yvar: Any | None = None,
        *,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        """新しい観測データを追加し、必要なら再学習する。"""
        self.update_data(new_X, new_Y, new_Yvar)
''',
)
replace_once(
    p,
    '''    def update_data(self, new_X: Any, new_Y: Any) -> "BayesianOptimizer":
        """保持している訓練データに新しい観測を追加する。"""
        if self.train_X is None or self.train_Y is None:
            self.train_X = new_X
            self.train_Y = new_Y
            return self
        self.train_X = _concat_rows(self.train_X, new_X)
        self.train_Y = _concat_rows(self.train_Y, new_Y)
        return self
''',
    '''    def update_data(
        self,
        new_X: Any,
        new_Y: Any,
        new_Yvar: Any | None = None,
    ) -> "BayesianOptimizer":
        """保持している訓練データに新しい観測を追加する。"""
        if self.train_X is None or self.train_Y is None:
            self.train_X = new_X
            self.train_Y = new_Y
            self.train_Yvar = new_Yvar
            return self
        if self.train_Yvar is not None and new_Yvar is None:
            raise ValueError(
                "new_Yvar is required because the fitted optimizer uses known observation variance."
            )
        if self.train_Yvar is None and new_Yvar is not None:
            raise ValueError(
                "new_Yvar cannot be added to an optimizer whose existing observations have no train_Yvar."
            )
        self.train_X = _concat_rows(self.train_X, new_X)
        self.train_Y = _concat_rows(self.train_Y, new_Y)
        if self.train_Yvar is not None:
            self.train_Yvar = _concat_rows(self.train_Yvar, new_Yvar)
        return self
''',
)

# CV fold models must receive the matching variance rows.
p = "src/bochan/api/evaluation/cross_validation.py"
replace_once(
    p,
    '''def cross_validate_optimizer(
    optimizer: Any,
    train_X: Any,
    train_Y: Any,
    *,
''',
    '''def cross_validate_optimizer(
    optimizer: Any,
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any | None = None,
    *,
''',
)
replace_once(
    p,
    '''    X, Y = torch.as_tensor(train_X), _as_2d(train_Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of rows.")
''',
    '''    X, Y = torch.as_tensor(train_X), _as_2d(train_Y)
    Yvar = None if train_Yvar is None else _as_2d(train_Yvar)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of rows.")
    if Yvar is not None and Yvar.shape != Y.shape:
        raise ValueError(
            "train_Yvar must match train_Y shape for cross-validation; "
            f"got train_Y={tuple(Y.shape)!r}, train_Yvar={tuple(Yvar.shape)!r}."
        )
''',
)
replace_once(
    p,
    "        fold_optimizer.fit(X[train_idx], Y[train_idx])\n",
    '''        fold_optimizer.fit(
            X[train_idx],
            Y[train_idx],
            None if Yvar is None else Yvar[train_idx],
        )
''',
)

# Tabular facade: reject ambiguous alpha/likelihood and pass Dataset.Yvar.
p = "src/bochan/tabular/optimizer/fitting.py"
replace_once(
    p,
    "from ..data import dataframe_to_tensors, numpy_to_tensors\n",
    "from ..data import dataframe_to_tensors, numpy_to_tensors\nfrom ..data.columns import _as_list\n",
)
replace_once(
    p,
    '''    if config.cat_dims is None and dataset.cat_dims:
        config = replace(config, cat_dims=dataset.cat_dims)
    return apply_alpha_to_model_config(
''',
    '''    if config.cat_dims is None and dataset.cat_dims:
        config = replace(config, cat_dims=dataset.cat_dims)
    if dataset.Yvar is not None:
        model_kwargs = dict(config.model_kwargs or {})
        if owner.alpha is not None or "_tabular_noise_alpha" in model_kwargs:
            raise ValueError(
                "target_variance_cols cannot be combined with alpha; known per-row "
                "observation variance already defines the Gaussian noise."
            )
        if "likelihood" in model_kwargs:
            raise ValueError(
                "target_variance_cols cannot be combined with model_kwargs['likelihood']; "
                "remove the explicit likelihood to use fixed known observation variance."
            )
    return apply_alpha_to_model_config(
''',
)
replace_once(
    p,
    '''    if owner.observation.uses_observation_conversion(resolved) and run_cv:
        raise ValueError(
            "Cross-validation requires an observation-aware validation protocol."
        )
    dataset = to_dataset(owner, fit_data, y, data_config=resolved)
''',
    '''    if owner.observation.uses_observation_conversion(resolved) and run_cv:
        raise ValueError(
            "Cross-validation requires an observation-aware validation protocol."
        )
    if (
        owner.observation.uses_observation_conversion(resolved)
        and resolved.target_variance_cols is not None
    ):
        raise ValueError(
            "target_variance_cols is not yet supported with "
            "target_missing_strategy='keep' or experiment_status_col."
        )
    dataset = to_dataset(owner, fit_data, y, data_config=resolved)
''',
)
replace_once(
    p,
    '''        owner.cross_validation_result_ = owner.bo.cross_validate(
            dataset.X,
            dataset.Y,
            model_config=model_config,
''',
    '''        owner.cross_validation_result_ = owner.bo.cross_validate(
            dataset.X,
            dataset.Y,
            dataset.Yvar,
            model_config=model_config,
''',
)
replace_once(
    p,
    '''    owner.bo.fit(
        dataset.X,
        dataset.Y,
        model_config=model_config,
''',
    '''    owner.bo.fit(
        dataset.X,
        dataset.Y,
        dataset.Yvar,
        model_config=model_config,
''',
)
replace_once(
    p,
    '''    metadata["feature_cols"] = list(owner.dataset.feature_names)
    metadata["target_cols"] = list(owner.dataset.target_names)
''',
    '''    metadata["feature_cols"] = list(owner.dataset.feature_names)
    metadata["target_cols"] = list(owner.dataset.target_names)
    metadata["known_observation_variance"] = owner.dataset.Yvar is not None
    if owner.data_config.target_variance_cols is not None:
        metadata["target_variance_cols"] = list(
            _as_list(owner.data_config.target_variance_cols)
        )
''',
)

# Direct FastAPI tensor schemas.
p = "src/bochan/serving/fastapi/schemas/requests.py"
replace_once(
    p,
    '''class FitModelRequest(APIRequest):
    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    train_X: Any
    train_Y: Any
    bounds: Any | None = None
''',
    '''class FitModelRequest(APIRequest):
    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    train_X: Any
    train_Y: Any
    train_Yvar: Any | None = None
    bounds: Any | None = None
''',
)
replace_once(
    p,
    '''class AutoCandidateRequest(APIRequest):
    """Plan model settings, fit a model, and generate candidates in one request."""

    goal: str
    train_X: Any
    train_Y: Any
    bounds: Any | None = None
''',
    '''class AutoCandidateRequest(APIRequest):
    """Plan model settings, fit a model, and generate candidates in one request."""

    goal: str
    train_X: Any
    train_Y: Any
    train_Yvar: Any | None = None
    bounds: Any | None = None
''',
)
replace_once(
    p,
    '''class TellRequest(APIRequest):
    """Append observations to an existing optimizer and optionally refit."""

    new_X: Any
    new_Y: Any
    refit: bool = True
''',
    '''class TellRequest(APIRequest):
    """Append observations to an existing optimizer and optionally refit."""

    new_X: Any
    new_Y: Any
    new_Yvar: Any | None = None
    refit: bool = True
''',
)
replace_once(
    p,
    '''    bo_model_config: ModelConfigSchema = Field(default_factory=ModelConfigSchema, alias="model_config")
    train_X: Any
    train_Y: Any
    bounds: Any
''',
    '''    bo_model_config: ModelConfigSchema = Field(default_factory=ModelConfigSchema, alias="model_config")
    train_X: Any
    train_Y: Any
    train_Yvar: Any | None = None
    bounds: Any
''',
)

# Direct FastAPI router forwarding.
p = "src/bochan/serving/fastapi/routers/models.py"
replace_once(
    p,
    '''        train_Y = to_target_tensor(
            request.train_Y,
            options,
            metadata=category_metadata,
        )
        fit_config = to_fit_config(request.fit_config)
''',
    '''        train_Y = to_target_tensor(
            request.train_Y,
            options,
            metadata=category_metadata,
        )
        train_Yvar = (
            to_tensor(request.train_Yvar, options)
            if request.train_Yvar is not None
            else None
        )
        fit_config = to_fit_config(request.fit_config)
''',
)
replace_once(
    p,
    "        optimizer.fit(train_X, train_Y)\n",
    "        optimizer.fit(train_X, train_Y, train_Yvar)\n",
)
replace_once(
    p,
    '''        train_Y = (
            to_target_tensor(
                request.train_Y,
                options,
                metadata=category_metadata,
            )
            if explicit_model_config is not None
            else to_tensor(request.train_Y, options)
        )
        plan = _plan_from_request(request, train_X, train_Y, bounds)
''',
    '''        train_Y = (
            to_target_tensor(
                request.train_Y,
                options,
                metadata=category_metadata,
            )
            if explicit_model_config is not None
            else to_tensor(request.train_Y, options)
        )
        train_Yvar = (
            to_tensor(request.train_Yvar, options)
            if request.train_Yvar is not None
            else None
        )
        plan = _plan_from_request(request, train_X, train_Y, bounds)
''',
)
replace_once(
    p,
    "        optimizer.fit(train_X, train_Y)\n",
    "        optimizer.fit(train_X, train_Y, train_Yvar)\n",
)
replace_once(
    p,
    '''        new_Y = to_target_tensor(
            request.new_Y,
            options,
            optimizer=optimizer,
        )
        fit_config = (
''',
    '''        new_Y = to_target_tensor(
            request.new_Y,
            options,
            optimizer=optimizer,
        )
        new_Yvar = (
            to_tensor(request.new_Yvar, options)
            if request.new_Yvar is not None
            else None
        )
        fit_config = (
''',
)
replace_once(
    p,
    '''        optimizer.tell(
            new_X,
            new_Y,
            refit=request.refit,
''',
    '''        optimizer.tell(
            new_X,
            new_Y,
            new_Yvar,
            refit=request.refit,
''',
)

p = "src/bochan/serving/fastapi/routers/suggestions.py"
replace_once(
    p,
    '''        train_Y = to_target_tensor(
            request.train_Y,
            options,
            metadata=category_metadata,
        )
        fit_config = to_fit_config(request.fit_config)
''',
    '''        train_Y = to_target_tensor(
            request.train_Y,
            options,
            metadata=category_metadata,
        )
        train_Yvar = (
            to_tensor(request.train_Yvar, options)
            if request.train_Yvar is not None
            else None
        )
        fit_config = to_fit_config(request.fit_config)
''',
)
replace_once(
    p,
    "        optimizer.fit(train_X, train_Y)\n",
    "        optimizer.fit(train_X, train_Y, train_Yvar)\n",
)

# Tabular FastAPI schema and services.
p = "src/bochan/serving/fastapi/schemas/tabular.py"
replace_once(
    p,
    "    target_cols: list[str] | str\n    categorical_cols: list[str] = Field(default_factory=list)\n",
    "    target_cols: list[str] | str\n    target_variance_cols: list[str] | str | None = None\n    categorical_cols: list[str] = Field(default_factory=list)\n",
)
replace_once(
    p,
    '''        if status is not None:
            if status in self.input_cols:
                raise ValueError("experiment_status_col must not be included in input_cols.")
            if status in targets:
                raise ValueError("experiment_status_col must not be included in target_cols.")
''',
    '''        variance_cols = (
            []
            if self.target_variance_cols is None
            else list(self.target_variance_cols)
            if isinstance(self.target_variance_cols, list)
            else [self.target_variance_cols]
        )
        if variance_cols:
            if len(variance_cols) != len(targets):
                raise ValueError(
                    "target_variance_cols must contain exactly one variance column per target column."
                )
            overlap = sorted(set(variance_cols).intersection(targets))
            if overlap:
                raise ValueError(
                    "target_variance_cols must be distinct from target_cols; "
                    f"overlap={overlap!r}."
                )
            input_overlap = sorted(set(variance_cols).intersection(self.input_cols))
            if input_overlap:
                raise ValueError(
                    "target_variance_cols must not be included in input_cols; "
                    f"overlap={input_overlap!r}."
                )
        if status is not None:
            if status in self.input_cols:
                raise ValueError("experiment_status_col must not be included in input_cols.")
            if status in targets:
                raise ValueError("experiment_status_col must not be included in target_cols.")
            if status in variance_cols:
                raise ValueError(
                    "experiment_status_col must not be included in target_variance_cols."
                )
''',
)

p = "src/bochan/serving/fastapi/services/tabular.py"
replace_once(
    p,
    "        target_cols=request.target_cols,\n        categorical_cols=request.categorical_cols,\n",
    "        target_cols=request.target_cols,\n        target_variance_cols=request.target_variance_cols,\n        categorical_cols=request.categorical_cols,\n",
)
for p in [
    "src/bochan/serving/fastapi/services/mace_tabular.py",
    "src/bochan/serving/fastapi/services/chgnet_tabular.py",
    "src/bochan/serving/fastapi/services/m3gnet_tabular.py",
    "src/bochan/serving/fastapi/services/alignn_tabular.py",
]:
    replace_once(
        p,
        "        target_cols=request.target_cols,\n        categorical_cols=request.categorical_cols,\n",
        "        target_cols=request.target_cols,\n        target_variance_cols=request.target_variance_cols,\n        categorical_cols=request.categorical_cols,\n",
    )

Path("tests/test_material_train_yvar_phase3.py").write_text(
    '''from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest
import torch

from bochan.api import BayesianOptimizer, ModelConfig, MultiOutputConfig
from bochan.api.modeling.build import build_model
from bochan.serving.fastapi.schemas.requests import FitModelRequest, TellRequest
from bochan.serving.fastapi.schemas.tabular import TabularFitModelRequest
from bochan.tabular import TabularDataConfig
from bochan.tabular.data import dataframe_to_tensors


class CaptureModel:
    def __init__(self, train_X: Any, train_Y: Any, train_Yvar: Any | None = None, **_: Any) -> None:
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar


class CaptureWrapper:
    def __init__(self, submodels: list[Any]) -> None:
        self.models = submodels


def _capture_wrapper(*, submodels: list[Any], **_: Any) -> CaptureWrapper:
    return CaptureWrapper(submodels)


def _capture_config() -> ModelConfig:
    return ModelConfig(
        task_type="regression",
        model_type="capture",
        model_factory=CaptureModel,
    )


def test_dataframe_target_variance_columns_are_not_features() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, 1.5, 2.0],
            "y2": [2.0, 2.5, 3.0],
            "y1_var": [0.01, 0.02, 0.03],
            "y2_var": [0.04, 0.05, 0.06],
        }
    )
    dataset = dataframe_to_tensors(
        frame,
        TabularDataConfig(
            target_cols=["y1", "y2"],
            target_variance_cols=["y1_var", "y2_var"],
        ),
    )

    assert dataset.feature_names == ["x"]
    assert dataset.Yvar is not None
    assert dataset.Yvar.shape == dataset.Y.shape == torch.Size([3, 2])
    torch.testing.assert_close(
        dataset.Yvar,
        torch.tensor(
            [[0.01, 0.04], [0.02, 0.05], [0.03, 0.06]],
            dtype=torch.double,
        ),
    )


@pytest.mark.parametrize(
    ("variance_values", "match"),
    [
        ([0.01, 0.0, 0.03], "strictly positive"),
        ([0.01, float("inf"), 0.03], "finite"),
    ],
)
def test_dataframe_target_variance_values_are_validated(
    variance_values: list[float],
    match: str,
) -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "y_var": variance_values}
    )
    with pytest.raises(ValueError, match=match):
        dataframe_to_tensors(
            frame,
            TabularDataConfig(
                input_cols=["x"],
                target_cols="y",
                target_variance_cols="y_var",
            ),
        )


def test_dataframe_target_variance_column_contract_is_validated() -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0], "y1": [1.0, 2.0], "y2": [2.0, 3.0], "v": [0.01, 0.02]}
    )
    with pytest.raises(ValueError, match="exactly one variance column"):
        dataframe_to_tensors(
            frame,
            TabularDataConfig(
                input_cols=["x"],
                target_cols=["y1", "y2"],
                target_variance_cols=["v"],
            ),
        )
    with pytest.raises(ValueError, match="must not be included in input_cols"):
        dataframe_to_tensors(
            frame,
            TabularDataConfig(
                input_cols=["x", "v"],
                target_cols="y1",
                target_variance_cols="v",
            ),
        )


def test_build_model_forwards_scalar_and_wide_train_yvar() -> None:
    X = torch.rand(4, 2, dtype=torch.double)
    Y = torch.rand(4, 2, dtype=torch.double)
    Yvar = torch.full_like(Y, 0.01)
    model = build_model(X, Y, _capture_config(), train_Yvar=Yvar).model

    assert model.train_Yvar is Yvar
    assert model.train_Y.shape == torch.Size([4, 2])


def test_independent_multi_output_slices_train_yvar_per_output() -> None:
    X = torch.rand(4, 2, dtype=torch.double)
    Y = torch.rand(4, 2, dtype=torch.double)
    Yvar = torch.tensor(
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]],
        dtype=torch.double,
    )
    output = _capture_config()
    config = replace(
        output,
        task_type="multi_objective",
        multi_output_config=MultiOutputConfig(
            output_configs=[output, output],
            wrapper_factory=_capture_wrapper,
        ),
    )
    wrapper = build_model(X, Y, config, train_Yvar=Yvar).model

    assert len(wrapper.models) == 2
    torch.testing.assert_close(wrapper.models[0].train_Yvar, Yvar[:, :1])
    torch.testing.assert_close(wrapper.models[1].train_Yvar, Yvar[:, 1:2])


def test_bayesian_optimizer_update_requires_consistent_known_noise() -> None:
    optimizer = BayesianOptimizer(_capture_config())
    optimizer.train_X = torch.zeros(2, 1)
    optimizer.train_Y = torch.zeros(2, 1)
    optimizer.train_Yvar = torch.full((2, 1), 0.01)

    with pytest.raises(ValueError, match="new_Yvar is required"):
        optimizer.update_data(torch.ones(1, 1), torch.ones(1, 1))

    optimizer.update_data(
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.full((1, 1), 0.02),
    )
    assert optimizer.train_Yvar.shape == torch.Size([3, 1])
    torch.testing.assert_close(optimizer.train_Yvar[-1], torch.tensor([0.02]))


def test_fastapi_tensor_schemas_accept_known_variance() -> None:
    request = FitModelRequest.model_validate(
        {
            "model_config": {"model_type": "base", "task_type": "regression"},
            "train_X": [[0.0], [1.0]],
            "train_Y": [[1.0], [2.0]],
            "train_Yvar": [[0.01], [0.02]],
        }
    )
    assert request.train_Yvar == [[0.01], [0.02]]
    tell = TellRequest(new_X=[[2.0]], new_Y=[[3.0]], new_Yvar=[[0.03]])
    assert tell.new_Yvar == [[0.03]]


def test_tabular_fastapi_schema_validates_variance_columns() -> None:
    payload = {
        "data": [
            {"x": 0.0, "y1": 1.0, "y2": 2.0, "v1": 0.01, "v2": 0.02},
            {"x": 1.0, "y1": 1.5, "y2": 2.5, "v1": 0.02, "v2": 0.03},
        ],
        "model_config": {"model_type": "deepkernel", "task_type": "multi_objective"},
        "input_cols": ["x"],
        "target_cols": ["y1", "y2"],
        "target_variance_cols": ["v1", "v2"],
    }
    request = TabularFitModelRequest.model_validate(payload)
    assert request.target_variance_cols == ["v1", "v2"]

    bad = dict(payload)
    bad["target_variance_cols"] = ["v1"]
    with pytest.raises(ValueError, match="exactly one variance column"):
        TabularFitModelRequest.model_validate(bad)
''',
    encoding="utf-8",
)

Path("docs/material_train_yvar_phase3.md").write_text(
    '''# Material `train_Yvar` Phase 3

Phase 3 exposes the fixed known-observation-variance support from Phases 1 and 2 through bochan's high-level tabular and FastAPI workflows.

## Tabular column contract

`TabularDataConfig.target_variance_cols` contains one **variance** column for each `target_cols` entry, in the same order. Values are variances, not standard deviations. They must be numeric, finite, and strictly positive.

Variance columns are metadata: when `input_cols` is inferred they are excluded from `X`, and explicitly including a variance column in `input_cols` is rejected.

```python
optimizer = TabularBayesianOptimizer(
    model_config={"model_type": "mace_multitask", "task_type": "multi_objective"},
    input_cols=["structure_id", "temperature"],
    target_cols=["strength", "ductility"],
    target_variance_cols=["strength_var", "ductility_var"],
    structure_col="structure_id",
    structure_catalog=structures,
    bounds={"temperature": [800.0, 1200.0]},
)
optimizer.fit(frame)
```

The resulting `TabularDataset.Yvar` is forwarded to `BayesianOptimizer.fit`. Correlated multitask models receive the full `[n, m]` tensor. Independent multi-output models receive the corresponding `[n, 1]` slice for each output.

## Direct tensor API

The core optimizer accepts known observation variance directly:

```python
optimizer.fit(train_X, train_Y, train_Yvar)
```

`refit()` preserves the stored variance. `tell(new_X, new_Y, new_Yvar)` appends it. Once an optimizer was fitted with known variance, every later observed row must also provide `new_Yvar`; partial known-noise histories are rejected.

## FastAPI

The direct tensor endpoints accept `train_Yvar`, and `/models/{model_id}/tell` accepts `new_Yvar`. Tabular fit requests accept `target_variance_cols` and material-specific tabular schemas inherit the same field.

## Noise-policy rules

Known per-row variance and a global `alpha` / explicit tabular likelihood are different noise contracts. Phase 3 rejects `target_variance_cols` together with `alpha` or `model_kwargs.likelihood` instead of silently choosing one.

When no variance is supplied, existing learned-noise behavior is unchanged.

## Current boundary

Observation-aware tabular conversion (`target_missing_strategy="keep"` or `experiment_status_col`) is intentionally rejected together with `target_variance_cols` in Phase 3. That path needs a separate contract for missing/pending rows whose objective variance is not yet observed.
''',
    encoding="utf-8",
)

Path(".github/workflows/material-train-yvar-phase3-smoke.yml").write_text(
    '''name: Material train Yvar Phase 3 smoke

on:
  pull_request:
    paths:
      - "src/bochan/api/evaluation/cross_validation.py"
      - "src/bochan/api/modeling/build.py"
      - "src/bochan/api/optimizer/core.py"
      - "src/bochan/tabular/**"
      - "src/bochan/serving/fastapi/**"
      - "tests/test_material_train_yvar_phase3.py"
      - ".github/workflows/material-train-yvar-phase3-smoke.yml"
  push:
    branches: [main]
    paths:
      - "src/bochan/api/evaluation/cross_validation.py"
      - "src/bochan/api/modeling/build.py"
      - "src/bochan/api/optimizer/core.py"
      - "src/bochan/tabular/**"
      - "src/bochan/serving/fastapi/**"
      - "tests/test_material_train_yvar_phase3.py"
      - ".github/workflows/material-train-yvar-phase3-smoke.yml"

jobs:
  high-level-known-noise:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install high-level test surface
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,tabular,api]"
          python -m pip install ruff
      - name: Phase 3 known-variance tests
        run: python -m pytest tests/test_material_train_yvar_phase3.py -q
      - name: Ruff
        run: |
          ruff check \
            src/bochan/api/evaluation/cross_validation.py \
            src/bochan/api/modeling/build.py \
            src/bochan/api/optimizer/core.py \
            src/bochan/tabular/config/data.py \
            src/bochan/tabular/data/conversion.py \
            src/bochan/tabular/data/dataset.py \
            src/bochan/tabular/optimizer/fitting.py \
            src/bochan/serving/fastapi/routers/models.py \
            src/bochan/serving/fastapi/routers/suggestions.py \
            src/bochan/serving/fastapi/schemas/requests.py \
            src/bochan/serving/fastapi/schemas/tabular.py \
            src/bochan/serving/fastapi/services/tabular.py \
            src/bochan/serving/fastapi/services/mace_tabular.py \
            src/bochan/serving/fastapi/services/chgnet_tabular.py \
            src/bochan/serving/fastapi/services/m3gnet_tabular.py \
            src/bochan/serving/fastapi/services/alignn_tabular.py \
            tests/test_material_train_yvar_phase3.py
''',
    encoding="utf-8",
)
