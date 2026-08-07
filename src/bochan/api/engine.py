"""High-level Bayesian optimization engine.

`BayesianOptimizer` はモデル生成・学習・予測・獲得関数生成・候補点最適化を
1つのクラスから扱うための薄い高レベル API です。

内部処理は `factory.py` の関数に委譲しており、研究用途では関数単位、
アプリ用途ではクラス単位で使い分けられるようにしています。
"""

from __future__ import annotations

import inspect
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


def _compact_name(value: Any) -> str:
    return "".join(ch for ch in str(value).replace("-", "_").lower() if ch.isalnum() or ch == "_")


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
        llm_settings: Any | None = None,
    ) -> None:
        self.fit_config = fit_config
        self.bounds = bounds
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry

        self.data_context = data_context
        self.llm_settings = self._coerce_llm_settings(llm_settings)
        self.model_config = self._merge_llm_settings_into_model_config(model_config)

        self.bundle: ModelBundle | None = None
        self.model: Any | None = None
        self.mll: Any | None = None

        self.train_X: Any | None = None
        self.train_Y: Any | None = None

        self.history: list[CandidateResult] = []

    def configure_llm(
        self,
        *,
        goal: Any | None = None,
        llm_config: Any | None = None,
        llm_context: Any | None = None,
        **settings_kwargs: Any,
    ) -> "BayesianOptimizer":
        """LLM planner / candidate generator の共通設定を登録する。

        ``ModelConfig(model_type="llm_selected")`` と
        ``OptimizeConfig(optimizer="llm_candidate_set")`` の両方がこの設定を参照します。
        個別の ``model_kwargs`` / ``optimizer_kwargs`` に同名キーがある場合は、
        個別設定が優先されます。
        """

        from bochan.llm import LLMSettings

        self.llm_settings = LLMSettings(
            goal=goal,
            llm_config=llm_config,
            llm_context=llm_context,
            **settings_kwargs,
        )
        self.model_config = self._merge_llm_settings_into_model_config(self.model_config)
        return self

    @staticmethod
    def _coerce_llm_settings(value: Any | None) -> Any | None:
        if value is None:
            return None
        from bochan.llm.configs import coerce_llm_settings

        return coerce_llm_settings(value)

    @staticmethod
    def _is_llm_selected_model_config(model_config: ModelConfig) -> bool:
        return _compact_name(model_config.model_type) in {
            "llm",
            "llm_selected",
            "llmselected",
            "llm_model_select",
            "llmmodelselect",
            "llm_model_selected",
            "llmmodelselected",
            "llm_planned",
            "llmplanned",
            "llm_planner",
            "llmplanner",
        }

    def _merge_llm_settings_into_model_config(self, model_config: ModelConfig) -> ModelConfig:
        if self.llm_settings is None or not self._is_llm_selected_model_config(model_config):
            return model_config
        default_kwargs = self.llm_settings.model_kwargs()
        if not default_kwargs:
            return model_config
        merged_kwargs = {**default_kwargs, **dict(model_config.model_kwargs or {})}
        return replace(model_config, model_kwargs=merged_kwargs)

    def _merge_llm_settings_into_opt_config(self, opt_config: OptimizeConfig) -> OptimizeConfig:
        if self.llm_settings is None:
            return opt_config
        optimizer = opt_config.optimizer
        if callable(optimizer) and not isinstance(optimizer, str):
            return opt_config
        if _optimizer_name(str(optimizer)) not in {
            "llm",
            "llm_candidate",
            "llm_candidate_set",
            "optimize_acqf_llm",
            "optimize_acqf_llm_candidate_set",
        }:
            return opt_config

        default_kwargs = self.llm_settings.optimizer_kwargs()
        if not default_kwargs:
            return opt_config
        merged_kwargs = {**default_kwargs, **dict(opt_config.optimizer_kwargs or {})}
        return replace(opt_config, optimizer_kwargs=merged_kwargs)

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
            self.model_config = self._merge_llm_settings_into_model_config(model_config)
        if fit_config is not None:
            self.fit_config = fit_config

        self.train_X = train_X
        self.train_Y = train_Y

        if self.bounds is None:
            self.bounds = _infer_bounds_from_train_X(train_X)
        if self.data_context is not None:
            self._resolve_data_context(self.data_context)

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

    def cross_validate(
        self,
        train_X: Any,
        train_Y: Any,
        *,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
        cv_config: Any | None = None,
    ) -> Any:
        """Evaluate fresh fold models without changing this optimizer's state.

        Args:
            train_X: Complete input data.
            train_Y: Complete target data.
            model_config: Optional model configuration override.
            fit_config: Optional fitting configuration override.
            cv_config: Cross-validation settings.

        Returns:
            A ``CrossValidationResult`` grouped by output.
        """
        from .cross_validation import cross_validate_optimizer

        return cross_validate_optimizer(
            self,
            train_X,
            train_Y,
            model_config=model_config,
            fit_config=fit_config,
            cv_config=cv_config,
        )

    def feature_importance(
        self,
        X: Any | None = None,
        y: Any | None = None,
        *,
        config: Any | None = None,
        feature_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
    ) -> Any:
        """Inspect a fitted model using raw-space permutation importance.

        Args:
            X: Raw evaluation inputs. Training inputs are used when omitted.
            y: Evaluation targets. Training targets are used when omitted.
            config: A :class:`bochan.inspection.FeatureImportanceConfig`.
            feature_names: Optional raw input column names.
            output_names: Optional output names.

        Returns:
            A serializable ``FeatureImportanceResult``.
        """
        self._check_fitted()
        from bochan.inspection import compute_feature_importance

        use_training = X is None and y is None
        if (X is None) != (y is None):
            raise ValueError("X and y must either both be provided or both be omitted.")
        X = self.train_X if X is None else X
        y = self.train_Y if y is None else y
        task: str | Sequence[str] = str(self.bundle.task_type)
        multi = getattr(self.model_config, "multi_output_config", None)
        if multi is not None:
            specs = getattr(multi, "outputs", None) or getattr(multi, "output_configs", None)
            if specs:
                task = [str(getattr(spec, "task_type", self.bundle.task_type)) for spec in specs]
                output_names = output_names or [str(getattr(spec, "name", f"output_{i}")) for i, spec in enumerate(specs)]
        return compute_feature_importance(
            model=self.model,
            predictor=self,
            X=X,
            y=y,
            task_type=task,
            feature_names=feature_names,
            output_names=output_names,
            cat_dims=self.bundle.cat_dims,
            config=config,
            training_data=use_training,
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

        binary classification では、利用可能なら ``probability_posterior``
        を優先する。mean はクラス1確率、variance は通常 Bernoulli 観測分散。
        """
        self._check_fitted()
        posterior_kwargs = posterior_kwargs or {}
        task_type = str(self.bundle.task_type)

        probability_posterior = getattr(self.model, "probability_posterior", None)
        if task_type == "binary" and callable(probability_posterior):
            posterior = probability_posterior(X, **posterior_kwargs)
        else:
            posterior = self.model.posterior(X, **posterior_kwargs)

        mean = getattr(posterior, "mean", None)
        variance = getattr(posterior, "variance", None)

        if task_type == "binary":
            prediction_space = "probability"
            observation_noise = posterior_kwargs.get("observation_noise", False)
            has_observation_noise = observation_noise is not False and observation_noise is not None
            variance_kind = "bernoulli_observation_plus_noise" if has_observation_noise else "bernoulli_observation"
        else:
            prediction_space = "outcome"
            variance_kind = "posterior"

        if return_result:
            return PredictionResult(
                posterior=posterior,
                mean=mean,
                variance=variance,
                task_type=task_type,
                prediction_space=prediction_space,
                variance_kind=variance_kind,
            )
        if return_type == "posterior":
            return posterior
        if return_type == "mean":
            return mean
        if return_type == "variance":
            return variance
        if return_type == "mean_variance":
            return mean, variance
        raise ValueError("Unknown return_type. Expected 'posterior', 'mean', 'variance', or 'mean_variance'.")

    def _acquisition_routing_context(self) -> tuple[str, str, bool]:
        """Resolve task/model/output shape used only for acquisition lookup.

        A one-output ``HybridMultiOutputModel`` is still useful as a Web/API
        compatibility wrapper, but it must not force acquisition lookup into
        the multi-output family.  In that case the sole submodel defines the
        task and model family while the acquisition is resolved as
        single-output.
        """
        self._check_fitted()
        bundle = self.bundle
        task_type = str(bundle.task_type)
        model_type = str(bundle.model_type)
        multi_output = bool(bundle.metadata.get("multi_output", False))

        if task_type != "hybrid":
            return task_type, model_type, multi_output

        sub_bundles = list(bundle.metadata.get("sub_bundles") or [])
        if len(sub_bundles) == 1:
            sub_bundle = sub_bundles[0]
            return (
                str(sub_bundle.task_type),
                str(sub_bundle.model_type),
                False,
            )

        specs = list(getattr(bundle.model, "specs", None) or [])
        if len(specs) == 1:
            return str(specs[0].task_type), model_type, False

        return task_type, model_type, multi_output

    def _resolve_acquisition_config(self, acq_config: AcquisitionConfig) -> AcquisitionConfig:
        if acq_config.acqf_cls is not None or acq_config.acqf_factory is not None:
            return acq_config
        task_type, model_type, multi_output = self._acquisition_routing_context()
        acqf_cls = resolve_acqf_cls(
            acq_config.name,
            self.acquisition_registry,
            task_type=task_type,
            model_type=model_type,
            multi_output=multi_output,
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
        acq_config = _resolve_objective_config_n_w_from_input_transform(
            acq_config=acq_config,
            bundle=self.bundle,
        )
        acq_config = _filter_context_fields_for_acqf(acq_config)
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

        opt_config = self._merge_llm_settings_into_opt_config(opt_config)
        cat_dims = self.bundle.cat_dims if self.bundle is not None else []
        opt_config = _resolve_optimizer_from_cat_dims(
            opt_config=opt_config,
            cat_dims=cat_dims,
        )
        opt_config = _resolve_mixed_fixed_features_from_train_X(
            opt_config=opt_config,
            train_X=self.train_X,
            cat_dims=cat_dims,
        )
        opt_config = _resolve_mixed_optimizer_callable(opt_config)

        acq_config = self._resolve_acquisition_config(acq_config)
        acq_config = _resolve_objective_config_n_w_from_input_transform(
            acq_config=acq_config,
            bundle=self.bundle,
        )
        acq_config = _filter_context_fields_for_acqf(acq_config)
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
        """DataContext を解決し、未指定の安全な既定値を学習データから補完する。

        明示指定された値は上書きしません。補完するのは、獲得関数に渡しても
        意味が安定している文脈情報だけです。
        """
        if data_context is not None:
            context = data_context
        elif self.data_context is not None:
            context = self.data_context
        else:
            context = DataContext()

        if context.bounds is None:
            context.bounds = self.bounds
        if context.bounds is None and self.train_X is not None:
            context.bounds = _infer_bounds_from_train_X(self.train_X)
            self.bounds = context.bounds

        if context.X_baseline is None:
            context.X_baseline = self.train_X
        if context.Y_baseline is None:
            context.Y_baseline = self.train_Y
        if context.mc_points is None:
            context.mc_points = self.train_X

        return context

    def _check_fitted(self) -> None:
        if self.bundle is None or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def _filter_context_fields_for_acqf(config: AcquisitionConfig) -> AcquisitionConfig:
    """Keep only context fields explicitly accepted by the acquisition class.

    Some acquisition classes accept ``**kwargs`` and forward them to BoTorch /
    GPyTorch base classes. Passing automatic context fields such as
    ``X_baseline`` to those classes can fail with errors like
    ``MCAcquisitionFunction.__init__() got an unexpected keyword argument``.

    Explicit ``acqf_kwargs`` are preserved. This helper filters only the
    automatically injected fields from ``DataContext``.
    """
    if config.acqf_cls is None or not config.filter_kwargs_by_signature:
        return config
    try:
        signature = inspect.signature(config.acqf_cls)
    except (TypeError, ValueError):
        return config
    explicit_params = set(signature.parameters)
    filtered_fields = tuple(field for field in config.context_fields if field in explicit_params)
    if filtered_fields == config.context_fields:
        return config
    return replace(config, context_fields=filtered_fields)


def _input_transform_n_w_from_model_config(model_config: ModelConfig | None) -> int | None:
    """ModelConfig.input_transform_config から perturbation 用 n_w を取り出す。"""
    if model_config is None:
        return None
    transform_config = getattr(model_config, "input_transform_config", None)
    if transform_config is None:
        return None
    if not bool(getattr(transform_config, "perturbation", False)):
        return None
    n_w = getattr(transform_config, "n_w", None)
    return None if n_w is None else int(n_w)


def _safe_output_index(output: Any | None) -> int | None:
    if output is None or isinstance(output, str):
        return None
    try:
        return int(output)
    except (TypeError, ValueError):
        return None


def _input_transform_n_w_from_bundle(bundle: ModelBundle | None, output: Any | None = None) -> int | None:
    """ObjectiveConfig.n_w の未指定時に bundle の input_transform 設定から n_w を推定する。"""
    if bundle is None:
        return None

    n_w = _input_transform_n_w_from_model_config(bundle.model_config)
    if n_w is not None:
        return n_w

    sub_bundles = list(bundle.metadata.get("sub_bundles", []) or [])
    if not sub_bundles:
        return None

    output_index = _safe_output_index(output)
    if output_index is not None and 0 <= output_index < len(sub_bundles):
        return _input_transform_n_w_from_model_config(sub_bundles[output_index].model_config)

    inferred_values = [
        value
        for value in (_input_transform_n_w_from_model_config(sub_bundle.model_config) for sub_bundle in sub_bundles)
        if value is not None
    ]
    if inferred_values and len(set(inferred_values)) == 1:
        return inferred_values[0]
    return None


def _resolve_objective_config_n_w_from_input_transform(
    *,
    acq_config: AcquisitionConfig,
    bundle: ModelBundle | None,
) -> AcquisitionConfig:
    """ObjectiveConfig.n_w 未指定なら InputTransformConfig.n_w で補完する。"""
    objective_config = acq_config.objective_config
    if objective_config is None or objective_config.n_w is not None:
        return acq_config
    if "n_w" in objective_config.objective_kwargs:
        return acq_config

    inferred_n_w = _input_transform_n_w_from_bundle(bundle, output=objective_config.output)
    if inferred_n_w is None:
        return acq_config

    return replace(
        acq_config,
        objective_config=replace(objective_config, n_w=inferred_n_w),
    )


def _optimizer_name(optimizer: str) -> str:
    return optimizer.replace("-", "_").lower()


def _resolve_optimizer_from_cat_dims(
    *,
    opt_config: OptimizeConfig,
    cat_dims: Sequence[int] | None,
) -> OptimizeConfig:
    """カテゴリ列がある場合に canonical optimizer 名を mixed 実装へ解決する。

    利用者は通常 ``optimize_acqf`` / ``evo`` / ``torch`` の3系統を指定すればよく、
    ``cat_dims`` がある場合は内部で対応する mixed optimizer を使います。
    明示的に mixed 名が指定されている場合は互換性のためそのまま保持します。
    """
    if not cat_dims:
        return opt_config

    optimizer = opt_config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return opt_config

    name = _optimizer_name(str(optimizer))
    mixed_by_name = {
        "optimize_acqf": "optimize_acqf_mixed",
        "evo": "evo_mixed",
        "optimize_acqf_evo": "optimize_acqf_evo_mixed",
        "torch": "torch_mixed",
        "optimize_acqf_torch": "optimize_acqf_torch_mixed",
    }
    mixed_name = mixed_by_name.get(name)
    if mixed_name is None:
        return opt_config
    return replace(opt_config, optimizer=mixed_name)


def _optimize_acqf_evo_mixed_filtered(
    acq_function: Any,
    bounds: Any,
    *,
    q: int = 1,
    method: str = "ga",
    categorical_features: dict[int, Sequence[float]] | None = None,
    fixed_features: dict[int, float] | None = None,
    fixed_features_list: list[dict[int, float]] | None = None,
    candidate_transform: Any = None,
    enumerate_categorical_features: bool = True,
    use_categorical_rounding_transform: bool | None = None,
    inequality_constraints: Any = None,
    equality_constraints: Any = None,
    inequality_sense: str = "le",
    post_processing_func: Any = None,
    batch_initial_conditions: Any = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: dict[str, Any] | None = None,
    X_pending: Any = None,
    apply_post_processing_during_eval: bool = True,
    repair_final_candidate: bool = True,
) -> tuple[Any, Any]:
    """Signature-filtered wrapper for evo mixed optimization.

    ``OptimizeConfig`` always has BoTorch-style ``num_restarts`` / ``raw_samples``
    fields.  The evo backend does not accept those arguments, so this explicit
    signature lets the factory drop them before dispatching.
    """
    from bochan.optim import optimize_acqf_evo_mixed

    return optimize_acqf_evo_mixed(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        method=method,
        categorical_features=categorical_features,
        fixed_features=fixed_features,
        fixed_features_list=fixed_features_list,
        candidate_transform=candidate_transform,
        enumerate_categorical_features=enumerate_categorical_features,
        use_categorical_rounding_transform=use_categorical_rounding_transform,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        inequality_sense=inequality_sense,
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        sequential=sequential,
        options=options,
        X_pending=X_pending,
        apply_post_processing_during_eval=apply_post_processing_during_eval,
        repair_final_candidate=repair_final_candidate,
    )


def _resolve_mixed_optimizer_callable(opt_config: OptimizeConfig) -> OptimizeConfig:
    """mixed evo では明示 signature の callable に解決して余計な kwargs を落とす。"""
    optimizer = opt_config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return opt_config

    name = _optimizer_name(str(optimizer))
    if name not in {"evo_mixed", "optimize_acqf_evo_mixed"}:
        return opt_config

    optimizer_kwargs = dict(opt_config.optimizer_kwargs)
    if opt_config.fixed_features_list is not None:
        optimizer_kwargs.setdefault("fixed_features_list", opt_config.fixed_features_list)

    return replace(
        opt_config,
        optimizer=_optimize_acqf_evo_mixed_filtered,
        optimizer_kwargs=optimizer_kwargs,
    )


def _uses_mixed_fixed_features(optimizer: Any) -> bool:
    if callable(optimizer) and not isinstance(optimizer, str):
        return optimizer is _optimize_acqf_evo_mixed_filtered
    return _optimizer_name(str(optimizer)) in {
        "optimize_acqf_mixed",
        "evo_mixed",
        "optimize_acqf_evo_mixed",
        "torch_mixed",
        "optimize_acqf_torch_mixed",
    }


def _fixed_features_list_from_category_rows(
    rows: Any,
    cat_dims: Sequence[int],
) -> list[dict[int, float]]:
    fixed_features_list: list[dict[int, float]] = []
    for row in rows:
        fixed_features_list.append({int(dim): float(value) for dim, value in zip(cat_dims, row)})
    return fixed_features_list


def _infer_fixed_features_list_from_train_X(
    train_X: Any,
    cat_dims: Sequence[int] | None,
) -> list[dict[int, float]] | None:
    """train_X のカテゴリ列から mixed optimizer 用 fixed_features_list を推定する。

    観測済みのカテゴリ組み合わせだけを列挙します。これにより、複数カテゴリ列が
    ある場合でも、未観測かつ無効な組み合わせをデフォルトで探索しにくくします。
    """
    cat_dims = list(cat_dims or [])
    if not cat_dims or train_X is None:
        return None

    try:
        import torch

        if isinstance(train_X, torch.Tensor):
            if train_X.ndim < 2:
                raise ValueError("train_X must have shape n x d or batch_shape x n x d.")
            X_cat = train_X[..., cat_dims]
            if X_cat.ndim > 2:
                X_cat = X_cat.reshape(-1, len(cat_dims))
            unique_rows = torch.unique(X_cat, dim=0)
            if unique_rows.numel() == 0:
                return None
            return _fixed_features_list_from_category_rows(unique_rows.detach().cpu().tolist(), cat_dims)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(train_X, np.ndarray):
            if train_X.ndim < 2:
                raise ValueError("train_X must have shape n x d or batch_shape x n x d.")
            X_cat = train_X[..., cat_dims]
            if X_cat.ndim > 2:
                X_cat = X_cat.reshape(-1, len(cat_dims))
            unique_rows = np.unique(X_cat, axis=0)
            if unique_rows.size == 0:
                return None
            return _fixed_features_list_from_category_rows(unique_rows.tolist(), cat_dims)
    except ImportError:
        pass

    try:
        import pandas as pd

        if isinstance(train_X, pd.DataFrame):
            X_cat = train_X.iloc[:, cat_dims]
            unique_rows = X_cat.drop_duplicates().to_numpy()
            if unique_rows.size == 0:
                return None
            return _fixed_features_list_from_category_rows(unique_rows.tolist(), cat_dims)
    except ImportError:
        pass

    raise TypeError(
        "Could not infer fixed_features_list from train_X. " "Pass OptimizeConfig.fixed_features_list explicitly."
    )


def _resolve_mixed_fixed_features_from_train_X(
    *,
    opt_config: OptimizeConfig,
    train_X: Any,
    cat_dims: Sequence[int] | None,
) -> OptimizeConfig:
    """mixed optimizer で fixed_features_list 未指定なら train_X から補完する。"""
    if not _uses_mixed_fixed_features(opt_config.optimizer):
        return opt_config
    if opt_config.fixed_features_list is not None:
        return opt_config

    inferred = _infer_fixed_features_list_from_train_X(train_X, cat_dims)
    if not inferred:
        return opt_config
    return replace(opt_config, fixed_features_list=inferred)


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
