"""Foundation-model regression surrogates."""

from . import pfn as _pfn
from .tabpfn import TabPFNMixedRegressorModel, TabPFNRegressorModel

_PFNS4BO_CHECKPOINT_REVISION = "4c0d901e4e0f1d5afd4b12e33974525cf3493dd3"
_pfn._PUBLIC_MODEL_BASE_URL = (
    "https://github.com/automl/PFNs4BO/raw/"
    f"{_PFNS4BO_CHECKPOINT_REVISION}/pfns4bo/final_models"
)

PFNPosterior = _pfn.PFNPosterior
PFNRegressorModel = _pfn.PFNRegressorModel
load_pfns4bo_pretrained = _pfn.load_pfns4bo_pretrained

__all__ = [
    "PFNPosterior",
    "PFNRegressorModel",
    "TabPFNMixedRegressorModel",
    "TabPFNRegressorModel",
    "load_pfns4bo_pretrained",
]
