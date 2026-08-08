"""Foundation-model regression surrogates."""

from ._pfns4bo_compat import apply_pfns4bo_torch_compat

apply_pfns4bo_torch_compat()

from .pfn import PFNPosterior, PFNRegressorModel, load_pfns4bo_pretrained  # noqa: E402

__all__ = [
    "PFNPosterior",
    "PFNRegressorModel",
    "load_pfns4bo_pretrained",
]
