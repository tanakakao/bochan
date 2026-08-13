from __future__ import annotations

import importlib.util

import bochan.api as api
from bochan.api import (
    acquisition_config,
    acquisition_registry,
    engine,
    factory,
    fit_config,
    model_registry,
    optimizer,
    optimizer_config,
)
from bochan.api.acquisition import defaults as acquisition_defaults
from bochan.api.acquisition import service as acquisition_service
from bochan.api.candidate import output as candidate_output
from bochan.api.config import acquisition as config_acquisition
from bochan.api.config import fit as config_fit
from bochan.api.config import optimize as config_optimize
from bochan.api.llm import LLMCandidateExplanationMixin, LLMSuggestionMixin
from bochan.api.observation import service as observation_service
from bochan.api.observation import state as observation_state
from bochan.api.registry import acquisition as registry_acquisition
from bochan.api.registry import model as registry_model


def test_public_bayesian_optimizer_has_one_canonical_definition() -> None:
    assert api.BayesianOptimizer is optimizer.BayesianOptimizer
    assert api.BayesianOptimizer.__module__ == "bochan.api.optimizer"
    assert issubclass(api.BayesianOptimizer, engine.BayesianOptimizer)


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


def test_registries_have_one_canonical_owner() -> None:
    assert registry_model.LazyModelRegistry.__module__ == "bochan.api.registry.model"
    assert api.LazyModelRegistry is registry_model.LazyModelRegistry
    assert api.MODEL_REGISTRY is registry_model.MODEL_REGISTRY
    assert api.DEFAULT_MODEL_REGISTRY is registry_model.DEFAULT_MODEL_REGISTRY

    assert registry_acquisition.resolve_acqf_cls.__module__ == (
        "bochan.api.registry.acquisition"
    )
    assert api.resolve_acqf_cls is registry_acquisition.resolve_acqf_cls
    assert api.available_acqf_names is registry_acquisition.available_acqf_names


def test_flat_registry_modules_are_declarative_facades() -> None:
    assert model_registry.LazyModelRegistry is registry_model.LazyModelRegistry
    assert model_registry.MODEL_REGISTRY is registry_model.MODEL_REGISTRY
    assert acquisition_registry.resolve_acqf_cls is registry_acquisition.resolve_acqf_cls
    assert acquisition_registry.available_acqf_names is registry_acquisition.available_acqf_names


def test_public_configs_have_canonical_package_owners() -> None:
    assert config_fit.FitConfig.__module__ == "bochan.api.config.fit"
    assert config_acquisition.AcquisitionConfig.__module__ == (
        "bochan.api.config.acquisition"
    )
    assert config_acquisition.OutcomeConstraintConfig.__module__ == (
        "bochan.api.config.acquisition"
    )
    assert config_optimize.OptimizeConfig.__module__ == "bochan.api.config.optimize"

    assert api.FitConfig is config_fit.FitConfig
    assert api.AcquisitionConfig is config_acquisition.AcquisitionConfig
    assert api.OutcomeConstraintConfig is config_acquisition.OutcomeConstraintConfig
    assert api.OptimizeConfig is config_optimize.OptimizeConfig


def test_flat_config_modules_are_declarative_facades() -> None:
    assert fit_config.FitConfig is config_fit.FitConfig
    assert acquisition_config.AcquisitionConfig is config_acquisition.AcquisitionConfig
    assert acquisition_config.OutcomeConstraintConfig is (
        config_acquisition.OutcomeConstraintConfig
    )
    assert optimizer_config.OptimizeConfig is config_optimize.OptimizeConfig
    assert optimizer_config.resolve_optimizer_from_cat_dims is (
        config_optimize.resolve_optimizer_from_cat_dims
    )


def test_removed_compatibility_and_patch_modules_do_not_exist() -> None:
    removed = {
        "bochan.api.acquisition_service",
        "bochan.api.candidate_output",
        "bochan.api.classification_perturbation_defaults",
        "bochan.api.engine_defaults",
        "bochan.api.kronecker_input_perturbation_defaults",
        "bochan.api.llm_candidate_explanation",
        "bochan.api.llm_selected_acquisition",
        "bochan.api.llm_suggestion",
        "bochan.api.observation_engine",
        "bochan.api.observation_service",
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
