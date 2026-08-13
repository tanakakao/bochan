from __future__ import annotations

import importlib.util

import bochan.api as api
from bochan.api import configs, engine, factory, optimizer, registry
from bochan.api.acquisition import defaults as acquisition_defaults
from bochan.api.acquisition import service as acquisition_service
from bochan.api.candidate import output as candidate_output
from bochan.api.configs import acquisition as acquisition_config
from bochan.api.configs import fit as fit_config
from bochan.api.configs import optimize as optimize_config
from bochan.api.llm import LLMCandidateExplanationMixin, LLMSuggestionMixin
from bochan.api.observation import service as observation_service
from bochan.api.observation import state as observation_state
from bochan.api.registry import acquisition as acquisition_registry
from bochan.api.registry import model as model_registry
from bochan.serving.fastapi.routers import tabular as tabular_router
from bochan.serving.fastapi.services import tabular as tabular_service


def test_public_bayesian_optimizer_has_one_canonical_definition() -> None:
    assert api.BayesianOptimizer is optimizer.BayesianOptimizer
    assert api.BayesianOptimizer.__module__ == "bochan.api.optimizer"
    assert issubclass(api.BayesianOptimizer, engine.BayesianOptimizer)


def test_base_configs_are_owned_by_package() -> None:
    assert hasattr(configs, "__path__")
    assert configs.ModelConfig is api.ModelConfig
    assert configs.MultiOutputConfig is api.MultiOutputConfig
    assert configs.ModelConfig.__module__ == "bochan.api.configs.base"


def test_specialized_configs_are_owned_by_package_modules() -> None:
    assert api.AcquisitionConfig is acquisition_config.AcquisitionConfig
    assert api.FitConfig is fit_config.FitConfig
    assert api.OptimizeConfig is optimize_config.OptimizeConfig
    assert api.AcquisitionConfig.__module__ == "bochan.api.configs.acquisition"
    assert api.FitConfig.__module__ == "bochan.api.configs.fit"
    assert api.OptimizeConfig.__module__ == "bochan.api.configs.optimize"


def test_registries_are_owned_by_registry_package() -> None:
    assert hasattr(registry, "__path__")
    assert api.MODEL_REGISTRY is model_registry.MODEL_REGISTRY
    assert api.DEFAULT_MODEL_REGISTRY is model_registry.DEFAULT_MODEL_REGISTRY
    assert api.resolve_acqf_cls is acquisition_registry.resolve_acqf_cls
    assert api.available_acqf_names is acquisition_registry.available_acqf_names
    assert model_registry.MODEL_REGISTRY.__class__.__module__ == (
        "bochan.api.registry.model"
    )
    assert acquisition_registry.resolve_acqf_cls.__module__ == (
        "bochan.api.registry.acquisition"
    )


def test_acquisition_defaults_have_one_owner() -> None:
    assert callable(acquisition_defaults.resolve_acquisition_defaults)
    assert callable(acquisition_defaults.resolve_multi_output_model_config)
    assert acquisition_defaults.__name__ == "bochan.api.acquisition.defaults"


def test_api_import_does_not_replace_engine_or_factory_symbols() -> None:
    assert engine.BayesianOptimizer is not api.BayesianOptimizer
    assert engine.BayesianOptimizer.__module__ == "bochan.api.engine"
    assert factory.build_acquisition.__module__ == "bochan.api.factory"
    assert acquisition_service.build_acquisition.__module__ == (
        "bochan.api.acquisition.service"
    )
    assert api.build_acquisition is acquisition_service.build_acquisition


def test_runtime_services_live_in_responsibility_subpackages() -> None:
    assert candidate_output.select_best_candidate_set.__module__ == (
        "bochan.api.candidate.output"
    )
    assert observation_state.ObservationData.__module__ == (
        "bochan.api.observation.state"
    )
    assert observation_service.build_objective_bundle.__module__ == (
        "bochan.api.observation.service"
    )
    assert LLMSuggestionMixin.__module__ == "bochan.api.llm.suggestion"
    assert LLMCandidateExplanationMixin.__module__ == "bochan.api.llm.explanation"


def test_fastapi_tabular_domain_logic_is_owned_by_application_service() -> None:
    service_functions = {
        "build_fit_response",
        "candidate_response",
        "compute_feature_importance_response",
        "fit_tabular_optimizer",
        "predict_response",
        "to_dataframe",
    }
    assert service_functions.issubset(vars(tabular_service))
    for name in service_functions:
        assert getattr(tabular_service, name).__module__ == (
            "bochan.serving.fastapi.services.tabular"
        )

    removed_router_helpers = {
        "_candidate_direct_kwargs",
        "_candidate_optimize_config",
        "_experiment_failure_config",
        "_fit_response",
        "_frame_records",
        "_generate_candidates",
        "_normalize_string_dtypes",
        "_schema_dict",
        "_to_dataframe",
    }
    assert removed_router_helpers.isdisjoint(vars(tabular_router))


def test_removed_compatibility_and_patch_modules_do_not_exist() -> None:
    removed = {
        "bochan.api.acquisition_config",
        "bochan.api.acquisition_registry",
        "bochan.api.acquisition_service",
        "bochan.api.candidate_output",
        "bochan.api.classification_perturbation_defaults",
        "bochan.api.engine_defaults",
        "bochan.api.fit_config",
        "bochan.api.kronecker_input_perturbation_defaults",
        "bochan.api.llm_candidate_explanation",
        "bochan.api.llm_selected_acquisition",
        "bochan.api.llm_suggestion",
        "bochan.api.model_registry",
        "bochan.api.observation_engine",
        "bochan.api.observation_service",
        "bochan.api.optimizer_config",
        "bochan.acquisition.objective.regression_perturbation",
    }
    for module_name in removed:
        assert importlib.util.find_spec(module_name) is None


def test_canonical_optimizer_owns_observation_and_candidate_entry_points() -> None:
    methods = {
        "fit",
        "fit_observations",
        "refit",
        "tell",
        "tell_observations",
        "acquisition",
        "candidate",
        "ask",
    }
    assert methods.issubset(vars(api.BayesianOptimizer))
