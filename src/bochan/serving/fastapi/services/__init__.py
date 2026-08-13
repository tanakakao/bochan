"""Application services shared by thin FastAPI routers."""

from .candidates import compare_candidate_results, generate_candidate_result
from .tabular import (
    build_fit_response,
    candidate_response,
    compute_feature_importance_response,
    fit_tabular_optimizer,
    predict_response,
    to_dataframe,
)

__all__ = [
    "build_fit_response",
    "candidate_response",
    "compare_candidate_results",
    "compute_feature_importance_response",
    "fit_tabular_optimizer",
    "generate_candidate_result",
    "predict_response",
    "to_dataframe",
]
