"""Application services shared by thin FastAPI routers."""

from .candidates import compare_candidate_results, generate_candidate_result

__all__ = ["compare_candidate_results", "generate_candidate_result"]
