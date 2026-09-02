from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:160]!r}")
    write(path, text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# ObservationData: carry cell-aligned known observation variance.
# ---------------------------------------------------------------------------
p = "src/bochan/api/observation/state.py"
replace_once(
    p,
    '''    observed_mask: Any | None = None
    failed_mask: Any | None = None
    pending_mask: Any | None = None
''',
    '''    observed_mask: Any | None = None
    failed_mask: Any | None = None
    pending_mask: Any | None = None
    Yvar: Any | None = None
''',
)
replace_once(
    p,
    '''        canonical_y = torch.full_like(Y, float("nan"))
        canonical_y = torch.where(observed, Y, canonical_y)

        self.X = X
        self.Y = canonical_y
        self.observed_mask = observed
''',
    '''        canonical_y = torch.full_like(Y, float("nan"))
        canonical_y = torch.where(observed, Y, canonical_y)

        canonical_yvar = None
        if self.Yvar is not None:
            Yvar = torch.as_tensor(self.Yvar, device=X.device)
            if Yvar.ndim == 1:
                Yvar = Yvar.unsqueeze(-1)
            if tuple(Yvar.shape) != tuple(Y.shape):
                raise ValueError(
                    "Yvar must have the same shape as Y. "
                    f"Y={tuple(Y.shape)}, Yvar={tuple(Yvar.shape)}."
                )
            if not torch.is_floating_point(Yvar):
                Yvar = Yvar.to(dtype=Y.dtype)
            else:
                Yvar = Yvar.to(dtype=Y.dtype)
            invalid_observed = observed & (~torch.isfinite(Yvar) | (Yvar <= 0.0))
            if bool(invalid_observed.any()):
                raise ValueError(
                    "Every observed target cell requires a finite, strictly positive Yvar."
                )
            canonical_yvar = torch.full_like(Yvar, float("nan"))
            canonical_yvar = torch.where(observed, Yvar, canonical_yvar)

        self.X = X
        self.Y = canonical_y
        self.Yvar = canonical_yvar
        self.observed_mask = observed
''',
)
replace_once(
    p,
    '''        *,
        status: Any,
        observed_mask: Any | None = None,
    ) -> ObservationData:
''',
    '''        *,
        status: Any,
        observed_mask: Any | None = None,
        Yvar: Any | None = None,
    ) -> ObservationData:
''',
)
replace_once(
    p,
    '''            observed_mask=observed_mask,
            failed_mask=[value == "failed" for value in statuses],
            pending_mask=[value == "pending" for value in statuses],
        )
''',
    '''            observed_mask=observed_mask,
            failed_mask=[value == "failed" for value in statuses],
            pending_mask=[value == "pending" for value in statuses],
            Yvar=Yvar,
        )
''',
)
replace_once(
    p,
    '''    def objective_training_data(self) -> tuple[Any, Any]:
        mask = self.objective_row_mask
        if not bool(mask.any()):
            raise ValueError("No successful experiment contains an observed objective value.")
        return self.X[mask], self.Y[mask]

    def output_training_data(self, output_index: int) -> tuple[Any, Any]:
''',
    '''    def objective_training_data(self) -> tuple[Any, Any]:
        mask = self.objective_row_mask
        if not bool(mask.any()):
            raise ValueError("No successful experiment contains an observed objective value.")
        return self.X[mask], self.Y[mask]

    def objective_training_data_with_variance(self) -> tuple[Any, Any, Any | None]:
        """Return successful objective rows with cell-aligned known variance."""
        mask = self.objective_row_mask
        if not bool(mask.any()):
            raise ValueError("No successful experiment contains an observed objective value.")
        Yvar = None if self.Yvar is None else self.Yvar[mask]
        return self.X[mask], self.Y[mask], Yvar

    def output_training_data(self, output_index: int) -> tuple[Any, Any]:
''',
)
replace_once(
    p,
    '''        if not bool(mask.any()):
            raise ValueError(f"Output {index} has no successful observed values.")
        return self.X[mask], self.Y[mask, index : index + 1]

    def success_training_data(self) -> tuple[Any, Any]:
''',
    '''        if not bool(mask.any()):
            raise ValueError(f"Output {index} has no successful observed values.")
        return self.X[mask], self.Y[mask, index : index + 1]

    def output_training_data_with_variance(
        self,
        output_index: int,
    ) -> tuple[Any, Any, Any | None]:
        """Return one observed output with its known observation variance."""
        index = int(output_index)
        if index < 0 or index >= int(self.Y.shape[-1]):
            raise IndexError(
                f"output_index={index} is outside [0, {int(self.Y.shape[-1]) - 1}]."
            )
        mask = self.success_mask & self.observed_mask[:, index]
        if not bool(mask.any()):
            raise ValueError(f"Output {index} has no successful observed values.")
        Yvar = None
        if self.Yvar is not None:
            Yvar = self.Yvar[mask, index : index + 1]
        return self.X[mask], self.Y[mask, index : index + 1], Yvar

    def success_training_data(self) -> tuple[Any, Any]:
''',
)
replace_once(
    p,
    '''        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        return ObservationData(
            X=torch.cat([self.X, other.X.to(self.X)], dim=0),
            Y=torch.cat([self.Y, other.Y.to(self.Y)], dim=0),
''',
    '''        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        if (self.Yvar is None) != (other.Yvar is None):
            raise ValueError(
                "ObservationData with known Yvar cannot be mixed with observations without Yvar."
            )
        Yvar = None
        if self.Yvar is not None:
            Yvar = torch.cat([self.Yvar, other.Yvar.to(self.Yvar)], dim=0)
        return ObservationData(
            X=torch.cat([self.X, other.X.to(self.X)], dim=0),
            Y=torch.cat([self.Y, other.Y.to(self.Y)], dim=0),
            Yvar=Yvar,
''',
)
replace_once(
    p,
    '''        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        if not bool(self.pending_mask.any()) or not bool(other.completed_mask.any()):
            return self.append(other)

        resolved_x = self.X.clone()
        resolved_y = self.Y.clone()
''',
    '''        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        if (self.Yvar is None) != (other.Yvar is None):
            raise ValueError(
                "ObservationData with known Yvar cannot be mixed with observations without Yvar."
            )
        if not bool(self.pending_mask.any()) or not bool(other.completed_mask.any()):
            return self.append(other)

        resolved_x = self.X.clone()
        resolved_y = self.Y.clone()
        resolved_yvar = None if self.Yvar is None else self.Yvar.clone()
''',
)
replace_once(
    p,
    '''            resolved_y[existing_index] = other.Y[incoming_index].to(resolved_y)
            resolved_observed[existing_index] = other.observed_mask[incoming_index].to(
''',
    '''            resolved_y[existing_index] = other.Y[incoming_index].to(resolved_y)
            if resolved_yvar is not None:
                resolved_yvar[existing_index] = other.Yvar[incoming_index].to(resolved_yvar)
            resolved_observed[existing_index] = other.observed_mask[incoming_index].to(
''',
)
replace_once(
    p,
    '''        resolved = ObservationData(
            X=resolved_x,
            Y=resolved_y,
            observed_mask=resolved_observed,
''',
    '''        resolved = ObservationData(
            X=resolved_x,
            Y=resolved_y,
            Yvar=resolved_yvar,
            observed_mask=resolved_observed,
''',
)
replace_once(
    p,
    '''            ObservationData(
                X=other.X[remaining],
                Y=other.Y[remaining],
                observed_mask=other.observed_mask[remaining],
''',
    '''            ObservationData(
                X=other.X[remaining],
                Y=other.Y[remaining],
                Yvar=None if other.Yvar is None else other.Yvar[remaining],
                observed_mask=other.observed_mask[remaining],
''',
)
replace_once(
    p,
    '''            "n_pending": int(self.pending_mask.sum().item()),
            "observed_per_output": [
''',
    '''            "n_pending": int(self.pending_mask.sum().item()),
            "known_observation_variance": self.Yvar is not None,
            "observed_per_output": [
''',
)


# ---------------------------------------------------------------------------
# Observation-aware builder: slice Yvar with exactly the same observed cells.
# ---------------------------------------------------------------------------
p = "src/bochan/api/observation/service.py"
replace_once(
    p,
    '''    train_Y: Any,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
''',
    '''    train_Y: Any,
    train_Yvar: Any | None,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
''',
)
replace_once(
    p,
    '''    observed_mask = torch.isfinite(torch.as_tensor(train_Y))
    sub_bundles: list[ModelBundle] = []
''',
    '''    observed_mask = torch.isfinite(torch.as_tensor(train_Y))
    Yvar = None if train_Yvar is None else torch.as_tensor(train_Yvar, device=train_Y.device)
    if Yvar is not None and tuple(Yvar.shape) != tuple(train_Y.shape):
        raise ValueError("train_Yvar must match train_Y shape for partial observations.")
    sub_bundles: list[ModelBundle] = []
''',
)
replace_once(
    p,
    '''        output_Y = train_Y[mask, index : index + 1]
        sub_bundles.append(
            _build_single_model(
                train_X=output_X,
                train_Y=output_Y,
                config=output_config,
''',
    '''        output_Y = train_Y[mask, index : index + 1]
        output_Yvar = None if Yvar is None else Yvar[mask, index : index + 1]
        sub_bundles.append(
            _build_single_model(
                train_X=output_X,
                train_Y=output_Y,
                train_Yvar=output_Yvar,
                config=output_config,
''',
)
replace_once(
    p,
    '''    train_Y: Any,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
    """Build objective models without imputing missing target values."""
''',
    '''    train_Y: Any,
    train_Yvar: Any | None = None,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
    """Build objective models without imputing missing target values."""
''',
)
replace_once(
    p,
    '''    Y = torch.as_tensor(train_Y)
    has_missing = bool(torch.isnan(Y).any())
''',
    '''    Y = torch.as_tensor(train_Y)
    Yvar = None if train_Yvar is None else torch.as_tensor(train_Yvar, device=Y.device)
    if Yvar is not None and tuple(Yvar.shape) != tuple(Y.shape):
        raise ValueError(
            "train_Yvar must match train_Y shape for observation-aware model building."
        )
    has_missing = bool(torch.isnan(Y).any())
''',
)
replace_once(
    p,
    '''            train_Y=train_Y,
            config=config,
            model_registry=model_registry,
''',
    '''            train_Y=train_Y,
            train_Yvar=Yvar,
            config=config,
            model_registry=model_registry,
''',
)
# The same call pattern occurs for the wide missing-target branch.
replace_once(
    p,
    '''            train_Y=train_Y,
            config=config,
            model_registry=model_registry,
''',
    '''            train_Y=train_Y,
            train_Yvar=Yvar,
            config=config,
            model_registry=model_registry,
''',
)
replace_once(
    p,
    '''            train_X=train_X,
            train_Y=Y,
            config=config,
            model_registry=model_registry,
''',
    '''            train_X=train_X,
            train_Y=Y,
            train_Yvar=Yvar,
            config=config,
            model_registry=model_registry,
''',
)
replace_once(
    p,
    '''        train_X=train_X[finite],
        train_Y=Y[finite],
        config=config,
''',
    '''        train_X=train_X[finite],
        train_Y=Y[finite],
        train_Yvar=None if Yvar is None else Yvar[finite],
        config=config,
''',
)


# ---------------------------------------------------------------------------
# Wide correlated multitask GP: map wide Yvar to the same long observations.
# ---------------------------------------------------------------------------
p = "src/bochan/models/multitask/wide.py"
replace_once(
    p,
    '''    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)
        kwargs.pop("num_tasks", None)
        kwargs.pop("task_feature", None)
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            task_feature=-1,
            all_tasks=list(range(num_tasks)),
            **kwargs,
        )
        self.num_tasks = num_tasks
        self.train_X_wide = torch.as_tensor(train_X)
        self.train_Y_wide = torch.as_tensor(train_Y)
''',
    '''    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_Y_tensor = torch.as_tensor(train_Y)
        X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y_tensor)
        train_Yvar = kwargs.pop("train_Yvar", None)
        train_Yvar_wide = None
        if train_Yvar is not None:
            train_Yvar_wide = torch.as_tensor(
                train_Yvar,
                dtype=Y_long.dtype,
                device=Y_long.device,
            )
            if tuple(train_Yvar_wide.shape) != tuple(train_Y_tensor.shape):
                raise ValueError(
                    "train_Yvar must match wide train_Y shape; "
                    f"train_Y={tuple(train_Y_tensor.shape)}, "
                    f"train_Yvar={tuple(train_Yvar_wide.shape)}."
                )
            observed = ~torch.isnan(train_Y_tensor)
            observed_yvar = train_Yvar_wide[observed]
            if not bool(torch.isfinite(observed_yvar).all()) or bool(
                (observed_yvar <= 0.0).any()
            ):
                raise ValueError(
                    "Observed wide multitask targets require finite, strictly positive train_Yvar."
                )
            kwargs["train_Yvar"] = observed_yvar.unsqueeze(-1)
            train_Yvar_wide = torch.where(
                observed,
                train_Yvar_wide,
                torch.full_like(train_Yvar_wide, float("nan")),
            )
        kwargs.pop("num_tasks", None)
        kwargs.pop("task_feature", None)
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            task_feature=-1,
            all_tasks=list(range(num_tasks)),
            **kwargs,
        )
        self.num_tasks = num_tasks
        self.train_X_wide = torch.as_tensor(train_X)
        self.train_Y_wide = train_Y_tensor
        self.train_Yvar_wide = train_Yvar_wide
''',
)


# ---------------------------------------------------------------------------
# Public optimizer: route Yvar through explicit observation state.
# ---------------------------------------------------------------------------
p = "src/bochan/api/optimizer/__init__.py"
text = read(p)
start = text.index("        if train_Yvar is not None:\n")
end = text.index("        if observation_data is None:\n", start)
replacement = '''        if observation_data is not None and train_Yvar is not None:
            raise ValueError(
                "Pass known observation variance inside ObservationData.Yvar when "
                "observation_data is supplied."
            )

        direct_known_noise = False
        if train_Yvar is not None and observation_data is None:
            import torch

            y_tensor = torch.as_tensor(train_Y)
            yvar_tensor = torch.as_tensor(train_Yvar)
            has_observation_state = any(
                value is not None
                for value in (observed_mask, failed_mask, pending_mask)
            ) or failure_config is not None
            direct_known_noise = (
                not has_observation_state
                and bool(torch.isfinite(y_tensor).all())
                and bool(torch.isfinite(yvar_tensor).all())
            )

        if direct_known_noise:
            if train_X is None or train_Y is None:
                raise ValueError("Provide both train_X and train_Y with train_Yvar.")

            base_model_config = model_config or self.model_config
            base_fit_config = fit_config or self.fit_config
            base_model_config, base_fit_config, llm_plan = resolve_llm_selected_model_config(
                base_model_config,
                train_X,
                train_Y,
                bounds=self.bounds,
                fit_config=base_fit_config,
            )
            resolved_model_config = resolve_multi_output_model_config(
                base_model_config,
                train_Y,
            )

            self.observations = None
            self.failure_config = None
            self.failure_bundle = None
            self.failure_model = None
            self.model_config = self._merge_llm_settings_into_model_config(
                resolved_model_config
            )
            self.fit_config = base_fit_config
            _CoreBayesianOptimizer.fit(
                self,
                train_X,
                train_Y,
                train_Yvar,
                model_config=self.model_config,
                fit_config=self.fit_config,
            )

            if llm_plan is not None:
                self.llm_plan = llm_plan
                self.bundle.metadata["llm_plan"] = llm_plan
                self.bundle.metadata["llm_selected_model_config"] = resolved_model_config

            self._llm_refit_required = False
            return self

'''
write(p, text[:start] + replacement + text[end:])
replace_once(
    p,
    '''            observation_data = ObservationData(
                X=train_X,
                Y=train_Y,
                observed_mask=observed_mask,
''',
    '''            observation_data = ObservationData(
                X=train_X,
                Y=train_Y,
                Yvar=train_Yvar,
                observed_mask=observed_mask,
''',
)
replace_once(
    p,
    '''        objective_X, objective_Y = observation_data.objective_training_data()
''',
    '''        objective_X, objective_Y, objective_Yvar = (
            observation_data.objective_training_data_with_variance()
        )
''',
)
replace_once(
    p,
    '''        self.train_X = objective_X
        self.train_Y = objective_Y
        self.train_Yvar = None
''',
    '''        self.train_X = objective_X
        self.train_Y = objective_Y
        self.train_Yvar = objective_Yvar
''',
)
replace_once(
    p,
    '''            train_X=objective_X,
            train_Y=objective_Y,
            config=self.model_config,
''',
    '''            train_X=objective_X,
            train_Y=objective_Y,
            train_Yvar=objective_Yvar,
            config=self.model_config,
''',
)
replace_once(
    p,
    '''        self.observations = self.observations.resolve_pending(observations)
        self.train_X, self.train_Y = self.observations.objective_training_data()
''',
    '''        self.observations = self.observations.resolve_pending(observations)
        self.train_X, self.train_Y, self.train_Yvar = (
            self.observations.objective_training_data_with_variance()
        )
''',
)
replace_once(
    p,
    '''    ) -> BayesianOptimizer:
        if self.train_Yvar is not None or new_Yvar is not None:
''',
    '''    ) -> BayesianOptimizer:
        if self.observations is not None:
            import torch

            X_tensor = torch.as_tensor(X_new)
            n_rows = int(X_tensor.shape[0]) if X_tensor.ndim > 1 else 1
            statuses = [status] * n_rows if isinstance(status, str) else list(status)
            observation_yvar = new_Yvar
            if self.observations.Yvar is not None and observation_yvar is None:
                Y_tensor = torch.as_tensor(Y_new)
                if Y_tensor.ndim == 1:
                    Y_tensor = Y_tensor.unsqueeze(-1)
                if not torch.is_floating_point(Y_tensor):
                    Y_tensor = Y_tensor.to(dtype=torch.get_default_dtype())
                observation_yvar = torch.full_like(Y_tensor, float("nan"))
            observations = ObservationData.from_status(
                X_new,
                Y_new,
                status=statuses,
                observed_mask=observed_mask,
                Yvar=observation_yvar,
            )
            return self.tell_observations(
                observations,
                refit=refit,
                fit_config=fit_config,
            )

        if self.train_Yvar is not None or new_Yvar is not None:
''',
)
replace_once(
    p,
    '''    ) -> BayesianOptimizer:
        if self.train_Yvar is not None or new_Yvar is not None:
            if not append:
''',
    '''    ) -> BayesianOptimizer:
        if self.observations is not None:
            if not append:
                return self.fit(
                    X_new,
                    Y_new,
                    new_Yvar,
                    model_config=self.model_config,
                    fit_config=self.fit_config,
                    failure_config=self.failure_config,
                )
            return self.tell(
                X_new,
                Y_new,
                new_Yvar,
                status="success",
                refit=False,
            )
        if self.train_Yvar is not None or new_Yvar is not None:
            if not append:
''',
)


# ---------------------------------------------------------------------------
# Observation-aware DataFrame conversion: target_variance_cols may be sparse.
# ---------------------------------------------------------------------------
p = "src/bochan/tabular/observation/data.py"
replace_once(
    p,
    '''            X=self.X,
            Y=self.Y,
            observed_mask=self.observed_mask,
''',
    '''            X=self.X,
            Y=self.Y,
            Yvar=self.Yvar,
            observed_mask=self.observed_mask,
''',
)
replace_once(
    p,
    '''    target_cols = _as_list(config.target_cols)
    status_col = config.experiment_status_col
    excluded = set(target_cols)
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
    status_col = config.experiment_status_col
    excluded = set(target_cols) | set(target_variance_cols)
''',
)
replace_once(
    p,
    '''    if not input_cols:
        raise ValueError("input_cols could not be inferred. Pass TabularDataConfig.input_cols.")

    selected = list(
''',
    '''    variance_inputs = sorted(set(input_cols).intersection(target_variance_cols), key=str)
    if variance_inputs:
        raise ValueError(
            "target_variance_cols must not be included in input_cols; "
            f"overlap={variance_inputs!r}."
        )
    if not input_cols:
        raise ValueError("input_cols could not be inferred. Pass TabularDataConfig.input_cols.")

    selected = list(
''',
)
replace_once(
    p,
    '''            input_cols
            + target_cols
            + ([status_col] if status_col is not None else [])
''',
    '''            input_cols
            + target_cols
            + target_variance_cols
            + ([status_col] if status_col is not None else [])
''',
)
replace_once(
    p,
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
    p,
    '''    Y = None
    observed_mask = None
    import torch
''',
    '''    Y = None
    Yvar = None
    observed_mask = None
    import torch
''',
)
replace_once(
    p,
    '''            Y = torch.where(
                observed_mask,
                Y,
                torch.full_like(Y, float("nan")),
            )

    feature_names = list(input_cols)
''',
    '''            Y = torch.where(
                observed_mask,
                Y,
                torch.full_like(Y, float("nan")),
            )

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
        Yvar = _to_tensor(
            Yvar_df.to_numpy(dtype=float),
            dtype=dtype,
            device=config.device,
        )
        if Yvar.ndim == 1:
            Yvar = Yvar.reshape(-1, 1)
        if Y is None or tuple(Yvar.shape) != tuple(Y.shape):
            raise ValueError("Target variance shape must match target shape.")
        invalid_observed = observed_mask & (~torch.isfinite(Yvar) | (Yvar <= 0.0))
        if bool(invalid_observed.any()):
            raise ValueError(
                "Every observed target cell requires a finite, strictly positive target variance."
            )
        Yvar = torch.where(
            observed_mask,
            Yvar,
            torch.full_like(Yvar, float("nan")),
        )

    feature_names = list(input_cols)
''',
)
replace_once(
    p,
    '''        X=X,
        Y=Y,
        feature_names=feature_names,
''',
    '''        X=X,
        Y=Y,
        Yvar=Yvar,
        feature_names=feature_names,
''',
)
replace_once(
    p,
    '''    if config.experiment_status_col is not None:
        raise ValueError("experiment_status_col requires DataFrame input.")
    dataset = numpy_to_tensors(
''',
    '''    if config.experiment_status_col is not None:
        raise ValueError("experiment_status_col requires DataFrame input.")
    if config.target_variance_cols is not None:
        raise ValueError(
            "target_variance_cols requires DataFrame input in observation-aware mode."
        )
    dataset = numpy_to_tensors(
''',
)


# ---------------------------------------------------------------------------
# Tabular fitting: observation-aware fit now owns both state and Yvar.
# ---------------------------------------------------------------------------
p = "src/bochan/tabular/optimizer/fitting.py"
replace_once(
    p,
    '''    if (
        owner.observation.uses_observation_conversion(resolved)
        and resolved.target_variance_cols is not None
    ):
        raise ValueError(
            "target_variance_cols is not yet supported with "
            "target_missing_strategy='keep' or experiment_status_col."
        )
''',
    '''''',
)
replace_once(
    p,
    '''    owner.bo.fit(
        dataset.X,
        dataset.Y,
        dataset.Yvar,
        model_config=model_config,
        fit_config=owner.fit_config,
    )
    if dataset.bounds is not None:
        owner.bo.set_bounds(dataset.bounds)
    owner.observation.attach(
        owner.bo,
        dataset,
        failure_config=owner.observation.resolve_failure_config(failure_config),
    )
''',
    '''    resolved_failure_config = owner.observation.resolve_failure_config(failure_config)
    if owner.observation.uses_observation_conversion(resolved):
        owner.bo.fit(
            observation_data=dataset.observation_data(),
            failure_config=resolved_failure_config,
            model_config=model_config,
            fit_config=owner.fit_config,
        )
        owner.observation.failure_config = resolved_failure_config
    else:
        owner.bo.fit(
            dataset.X,
            dataset.Y,
            dataset.Yvar,
            model_config=model_config,
            fit_config=owner.fit_config,
        )
        owner.observation.attach(
            owner.bo,
            dataset,
            failure_config=resolved_failure_config,
        )
    if dataset.bounds is not None:
        owner.bo.set_bounds(dataset.bounds)
''',
)


# ---------------------------------------------------------------------------
# Focused Phase 4 tests.
# ---------------------------------------------------------------------------
Path("tests/test_material_train_yvar_phase4.py").write_text(
    '''from __future__ import annotations

import pandas as pd
import pytest
import torch

import bochan.api.optimizer as optimizer_module
from bochan.api import BayesianOptimizer, ModelConfig, MultiOutputConfig, ObservationData, OutputConfig
from bochan.api.observation.service import build_objective_bundle
from bochan.models.multitask.wide import WideMultiTaskGP
from bochan.tabular import TabularDataConfig
from bochan.tabular.observation.data import dataframe_to_observation_tensors


class _CaptureModel:
    def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.kwargs = kwargs


def _single_config() -> ModelConfig:
    return ModelConfig(
        task_type="regression",
        model_type="capture",
        model_factory=_CaptureModel,
    )


def _multi_config() -> ModelConfig:
    return ModelConfig(
        task_type="multi_objective",
        model_type="capture",
        model_factory=_CaptureModel,
        multi_output_config=MultiOutputConfig(
            output_configs=[
                OutputConfig(task_type="regression", model_type="capture"),
                OutputConfig(task_type="regression", model_type="capture"),
            ],
            output_names=["a", "b"],
            use_hybrid=False,
        ),
    )


def test_observation_data_canonicalizes_unobserved_yvar() -> None:
    obs = ObservationData(
        X=torch.tensor([[0.0], [1.0], [2.0]]),
        Y=torch.tensor([[1.0, float("nan")], [2.0, 3.0], [9.0, 9.0]]),
        Yvar=torch.tensor([[0.1, 999.0], [0.2, 0.3], [999.0, 999.0]]),
        failed_mask=[False, False, False],
        pending_mask=[False, False, True],
    )
    assert obs.Yvar is not None
    assert torch.isnan(obs.Yvar[0, 1])
    assert torch.isnan(obs.Yvar[2]).all()
    torch.testing.assert_close(obs.Yvar[1], torch.tensor([0.2, 0.3]))
    assert obs.report()["known_observation_variance"] is True


def test_observation_data_requires_yvar_for_observed_cell() -> None:
    with pytest.raises(ValueError, match="strictly positive Yvar"):
        ObservationData(
            X=torch.tensor([[0.0], [1.0]]),
            Y=torch.tensor([[1.0], [2.0]]),
            Yvar=torch.tensor([[0.1], [float("nan")]]),
        )


def test_observation_append_rejects_known_noise_mode_mixing() -> None:
    known = ObservationData(
        X=torch.tensor([[0.0]]),
        Y=torch.tensor([[1.0]]),
        Yvar=torch.tensor([[0.1]]),
    )
    unknown = ObservationData(X=torch.tensor([[1.0]]), Y=torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="cannot be mixed"):
        known.append(unknown)


def test_resolve_pending_replaces_yvar() -> None:
    pending = ObservationData.from_status(
        torch.tensor([[1.0]]),
        torch.tensor([[float("nan")]]),
        Yvar=torch.tensor([[float("nan")]]),
        status=["pending"],
    )
    completed = ObservationData.from_status(
        torch.tensor([[1.0]]),
        torch.tensor([[4.0]]),
        Yvar=torch.tensor([[0.25]]),
        status=["success"],
    )
    resolved = pending.resolve_pending(completed)
    assert not bool(resolved.pending_mask.any())
    torch.testing.assert_close(resolved.Y, torch.tensor([[4.0]]))
    torch.testing.assert_close(resolved.Yvar, torch.tensor([[0.25]]))


def test_partial_split_builder_slices_yvar_per_output() -> None:
    X = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    Y = torch.tensor(
        [[1.0, float("nan")], [2.0, 4.0], [float("nan"), 5.0]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.1, float("nan")], [0.2, 0.3], [float("nan"), 0.4]],
        dtype=torch.double,
    )
    bundle = build_objective_bundle(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar,
        config=_multi_config(),
    )
    sub_bundles = bundle.metadata["sub_bundles"]
    torch.testing.assert_close(sub_bundles[0].model.train_Yvar, torch.tensor([[0.1], [0.2]], dtype=torch.double))
    torch.testing.assert_close(sub_bundles[1].model.train_Yvar, torch.tensor([[0.3], [0.4]], dtype=torch.double))


def test_observation_dataframe_allows_sparse_variance_and_excludes_columns() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, 2.0, None],
            "y2": [None, 4.0, None],
            "v1": [0.1, 0.2, None],
            "v2": [None, 0.3, None],
            "status": ["success", "success", "pending"],
        }
    )
    config = TabularDataConfig(
        target_cols=["y1", "y2"],
        target_variance_cols=["v1", "v2"],
        experiment_status_col="status",
        target_missing_strategy="keep",
    )
    dataset = dataframe_to_observation_tensors(frame, config)
    assert dataset.feature_names == ["x"]
    assert dataset.Yvar is not None
    torch.testing.assert_close(dataset.Yvar[1], torch.tensor([0.2, 0.3], dtype=torch.double))
    assert torch.isnan(dataset.Yvar[0, 1])
    assert torch.isnan(dataset.Yvar[2]).all()


def test_observation_dataframe_rejects_missing_variance_for_observed_target() -> None:
    frame = pd.DataFrame(
        {"x": [0.0], "y": [1.0], "v": [None], "status": ["success"]}
    )
    config = TabularDataConfig(
        target_cols=["y"],
        target_variance_cols=["v"],
        experiment_status_col="status",
        target_missing_strategy="keep",
    )
    with pytest.raises(ValueError, match="strictly positive target variance"):
        dataframe_to_observation_tensors(frame, config)


def test_public_optimizer_observation_fit_routes_yvar(monkeypatch) -> None:
    monkeypatch.setattr(optimizer_module, "fit_model", lambda bundle, config: bundle)
    optimizer = BayesianOptimizer(model_config=_single_config())
    optimizer.fit(
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([[1.0], [2.0], [float("nan")]]),
        torch.tensor([[0.1], [0.2], [float("nan")]]),
        pending_mask=[False, False, True],
    )
    assert optimizer.observations is not None
    assert optimizer.train_Yvar is not None
    torch.testing.assert_close(optimizer.model.train_Yvar, torch.tensor([[0.1], [0.2]]))


def test_public_optimizer_pending_then_tell_with_yvar(monkeypatch) -> None:
    monkeypatch.setattr(optimizer_module, "fit_model", lambda bundle, config: bundle)
    optimizer = BayesianOptimizer(model_config=_single_config())
    optimizer.fit(
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[1.0], [float("nan")]]),
        torch.tensor([[0.1], [float("nan")]]),
        pending_mask=[False, True],
    )
    optimizer.tell(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[0.2]]),
        status="success",
        refit=False,
    )
    assert optimizer.observations is not None
    assert not bool(optimizer.observations.pending_mask.any())
    torch.testing.assert_close(optimizer.train_Yvar, torch.tensor([[0.1], [0.2]]))


def test_wide_multitask_gp_maps_wide_yvar_to_long_noise() -> None:
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [[1.0, float("nan")], [2.0, 4.0], [float("nan"), 5.0]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.1, float("nan")], [0.2, 0.3], [float("nan"), 0.4]],
        dtype=torch.double,
    )
    model = WideMultiTaskGP(X, Y, train_Yvar=Yvar)
    assert model.train_Yvar_wide is not None
    torch.testing.assert_close(model.train_Yvar_wide, Yvar, equal_nan=True)
    noise = model.likelihood.noise.detach().reshape(-1)
    torch.testing.assert_close(noise, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=noise.dtype))
''',
    encoding="utf-8",
)

Path("docs/material_train_yvar_phase4.md").write_text(
    '''# Material known-observation variance Phase 4

Phase 4 integrates per-observation variance with the experiment observation-state workflow introduced for missing, failed, and pending experiments.

## Contract

- `ObservationData.Yvar` is optional and has the same wide `[n, m]` shape as `Y`.
- Every observed objective cell requires a finite, strictly positive variance.
- Variance for unobserved, failed, or pending cells is canonicalized to `NaN`.
- Known-variance and unknown-variance histories cannot be mixed implicitly during append / pending resolution.
- `target_variance_cols` may be used together with `target_missing_strategy="keep"` and `experiment_status_col` for DataFrame workflows.
- Variance columns are metadata/targets, never model input features.

## Partial multi-output behavior

Independent multi-output models slice `Yvar` with the exact per-output observation mask. `WideMultiTaskGP` converts wide variance to the same long task-feature rows used for observed targets, enabling correlated multitask regression with partial targets and known noise.

## Optimizer lifecycle

The public optimizer preserves Phase 3's direct fully-observed known-noise path. When explicit observation state, partial targets, pending/failed rows, or a failure model is present, it uses `ObservationData` and carries variance through fit/refit/tell/pending resolution.

Cross-validation for observation-aware status workflows remains intentionally unsupported because it requires a status-aware validation protocol rather than ordinary row splitting.
''',
    encoding="utf-8",
)
