"""Capability metadata route for the Web API."""

from typing import Any

from fastapi import APIRouter

from bochan.api.registry.capabilities import BETA_MODEL_TYPES

WEB_CAPABILITIES: dict[str, Any] = {
    "task_types": ["regression", "classification", "ordinal", "hybrid", "multi_objective"],
    "model_types": [
        "base",
        "deepgp",
        "deepkernel",
        "crabnet_gp",
        "crabnet_mixed_gp",
        "crabnet_dkl",
        "crabnet_mixed_dkl",
        "saas",
        "pca",
        "rembo",
        "robust",
        "hetero",
        "random_forest",
        "lightgbm_ensemble",
        "ngboost_ensemble",
        "tabpfn",
        "multitask",
    ],
    "gamma_model_types": [
        "gamma_base",
        "gamma_deepgp",
        "gamma_deepkernel",
        "gamma_saas",
        "gamma_pca",
        "gamma_rembo",
        "gamma_rrp",
        "gamma_hetero",
        "gamma_multitask",
    ],
    "beta_model_types": list(BETA_MODEL_TYPES),
    "acquisitions": [
        "EI",
        "PI",
        "UCB",
        "EHVI",
        "NEHVI",
        "NParEGO",
        "variance",
        "predictive_entropy",
        "BALD",
        "NIPV",
        "straddle",
        "boundary_variance",
        "ICU",
    ],
    "optimizers": [
        "optimize_acqf",
        "torch",
        "ga",
        "sa",
        "pso",
        "cmaes",
        "thompson_sampling",
        "nsgaii",
    ],
    "data_sources": ["csv", "excel", "model_artifact"],
    "visualizations": ["yyplot", "target_relation", "pareto", "prediction-1d", "prediction-2d", "ternary"],
    "model_artifacts": {
        "download_endpoint": "/api/v1/runs/{run_id}/model-artifact",
        "import_endpoint": "/api/v1/model-artifacts/import",
        "format": "bochan.pt",
        "pickle_trust_required": True,
    },
    "logging": {
        "format": "jsonl",
        "request_id_header": "X-Request-ID",
        "recent_logs_endpoint": "/api/v1/logs",
    },
    "composition": {
        "enabled": True,
        "max_formula_columns": 1,
        "sites": False,
        "ratio_total": 1.0,
        "representations": ["fractions", "clr", "alr", "ilr"],
        "normalizations": ["atomic_fraction", "weight_fraction"],
        "element_constraints": ["=", "<=", ">="],
        "visualization_axes": ["element_fraction_1d", "element_fraction_2d", "ternary"],
        "feature_importance": ["composition_total", "element_perturbation"],
        "validation_endpoint": "/api/v1/composition/validate",
        "optimization_endpoint": "/api/v1/composition/regression/run",
    },
    "crabnet": {
        "model_types": ["crabnet_gp", "crabnet_mixed_gp", "crabnet_dkl", "crabnet_mixed_dkl"],
        "checkpoint": True,
        "encoder_training": ["partial", "full"],
        "default_encoder_training": "partial",
        "max_formula_columns": 1,
        "continuous_process_model_types": ["crabnet_gp", "crabnet_dkl"],
        "mixed_process_model_types": ["crabnet_mixed_gp", "crabnet_mixed_dkl"],
        "mixed_categorical_kernel": True,
        "mixed_categorical_embedding": True,
        "single_output_regression_only": False,
        "independent_multi_output_model_types": [
            "crabnet_gp",
            "crabnet_mixed_gp",
            "crabnet_dkl",
            "crabnet_mixed_dkl",
        ],
        "multi_output_structure": "model_list",
        "input_perturbation": False,
    },
}


def create_capabilities_router(*, api_prefix: str) -> APIRouter:
    """Create the Web capability metadata router."""

    router = APIRouter()

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        capabilities_payload = dict(WEB_CAPABILITIES)
        logging_payload = dict(WEB_CAPABILITIES["logging"])
        logging_payload["recent_logs_endpoint"] = f"{api_prefix}/logs"
        capabilities_payload["logging"] = logging_payload
        artifact_payload = dict(WEB_CAPABILITIES["model_artifacts"])
        artifact_payload["download_endpoint"] = f"{api_prefix}/runs/{{run_id}}/model-artifact"
        artifact_payload["import_endpoint"] = f"{api_prefix}/model-artifacts/import"
        capabilities_payload["model_artifacts"] = artifact_payload
        composition_payload = dict(WEB_CAPABILITIES["composition"])
        composition_payload["validation_endpoint"] = f"{api_prefix}/composition/validate"
        composition_payload["optimization_endpoint"] = f"{api_prefix}/composition/regression/run"
        capabilities_payload["composition"] = composition_payload
        return capabilities_payload

    return router


__all__ = ["WEB_CAPABILITIES", "create_capabilities_router"]
