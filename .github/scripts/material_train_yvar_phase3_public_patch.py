from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Keep internal model-builder helpers backward compatible with existing callers.
p = "src/bochan/api/modeling/build.py"
replace_once(
    p,
    '''def _build_single_model(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any | None,
    config: ModelConfig,
    *,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
''',
    '''def _build_single_model(
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    *,
    train_Yvar: Any | None = None,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
''',
)
replace_once(
    p,
    '''def build_multi_output_model(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any | None,
    config: ModelConfig,
    *,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
''',
    '''def build_multi_output_model(
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    *,
    train_Yvar: Any | None = None,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
''',
)
replace_once(
    p,
    '''        return build_multi_output_model(
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
    '''        return build_multi_output_model(
            train_X,
            train_Y,
            config,
            train_Yvar=train_Yvar,
            model_registry=model_registry,
        )
    return _build_single_model(
        train_X,
        train_Y,
        config,
        train_Yvar=train_Yvar,
        model_registry=model_registry,
    )
''',
)

# Public BayesianOptimizer overrides core lifecycle methods, so known variance
# must be routed explicitly here as well. Phase 3 keeps this path separate from
# partial / failed / pending ObservationData semantics.
p = "src/bochan/api/optimizer/__init__.py"
replace_once(
    p,
    '''    def fit(
        self,
        train_X: Any | None = None,
        train_Y: Any | None = None,
        *,
        observation_data: ObservationData | None = None,
''',
    '''    def fit(
        self,
        train_X: Any | None = None,
        train_Y: Any | None = None,
        train_Yvar: Any | None = None,
        *,
        observation_data: ObservationData | None = None,
''',
)
replace_once(
    p,
    '''        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        if observation_data is None:
''',
    '''        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        if train_Yvar is not None:
            if observation_data is not None:
                raise ValueError(
                    "train_Yvar cannot be combined with observation_data in Phase 3. "
                    "Use fully observed train_X/train_Y rows for known observation variance."
                )
            if any(
                value is not None
                for value in (observed_mask, failed_mask, pending_mask)
            ):
                raise ValueError(
                    "train_Yvar cannot be combined with observed/failed/pending masks in Phase 3."
                )
            if failure_config is not None:
                raise ValueError(
                    "train_Yvar cannot be combined with failure_config in Phase 3."
                )
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

        if observation_data is None:
''',
)
replace_once(
    p,
    '''        self.train_X = objective_X
        self.train_Y = objective_Y

        if self.bounds is None:
''',
    '''        self.train_X = objective_X
        self.train_Y = objective_Y
        self.train_Yvar = None

        if self.bounds is None:
''',
)
replace_once(
    p,
    '''    def tell(
        self,
        X_new: Any,
        Y_new: Any,
        *,
        status: Any = "success",
''',
    '''    def tell(
        self,
        X_new: Any,
        Y_new: Any,
        new_Yvar: Any | None = None,
        *,
        status: Any = "success",
''',
)
replace_once(
    p,
    '''    ) -> BayesianOptimizer:
        import torch

        X_tensor = torch.as_tensor(X_new)
''',
    '''    ) -> BayesianOptimizer:
        if self.train_Yvar is not None or new_Yvar is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            if any(str(value).lower() != "success" for value in statuses):
                raise ValueError(
                    "Known observation variance currently supports only successful observations."
                )
            if observed_mask is not None:
                raise ValueError(
                    "new_Yvar cannot be combined with observed_mask in Phase 3."
                )
            _CoreBayesianOptimizer.update_data(self, X_new, Y_new, new_Yvar)
            if refit:
                self.refit(fit_config=fit_config)
            return self

        import torch

        X_tensor = torch.as_tensor(X_new)
''',
)
replace_once(
    p,
    '''    def update_data(
        self,
        X_new: Any,
        Y_new: Any,
        *,
        append: bool = True,
    ) -> BayesianOptimizer:
        if not append:
''',
    '''    def update_data(
        self,
        X_new: Any,
        Y_new: Any,
        new_Yvar: Any | None = None,
        *,
        append: bool = True,
    ) -> BayesianOptimizer:
        if self.train_Yvar is not None or new_Yvar is not None:
            if not append:
                return self.fit(
                    X_new,
                    Y_new,
                    new_Yvar,
                    model_config=self.model_config,
                    fit_config=self.fit_config,
                    failure_config=None,
                )
            return _CoreBayesianOptimizer.update_data(
                self,
                X_new,
                Y_new,
                new_Yvar,
            )
        if not append:
''',
)

# Exercise the public tell path as well as update_data.
test_path = Path("tests/test_material_train_yvar_phase3.py")
test_text = test_path.read_text(encoding="utf-8")
needle = '''def test_fastapi_tensor_schemas_accept_known_variance() -> None:
'''
addition = '''def test_public_optimizer_tell_appends_known_variance() -> None:
    optimizer = BayesianOptimizer(_capture_config())
    optimizer.train_X = torch.zeros(2, 1)
    optimizer.train_Y = torch.zeros(2, 1)
    optimizer.train_Yvar = torch.full((2, 1), 0.01)
    optimizer.observations = None

    optimizer.tell(
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.full((1, 1), 0.02),
        refit=False,
    )

    assert optimizer.train_Yvar.shape == torch.Size([3, 1])
    torch.testing.assert_close(optimizer.train_Yvar[-1], torch.tensor([0.02]))


def test_fastapi_tensor_schemas_accept_known_variance() -> None:
'''
if needle not in test_text:
    raise RuntimeError("Phase 3 FastAPI test insertion point not found")
test_path.write_text(test_text.replace(needle, addition, 1), encoding="utf-8")
