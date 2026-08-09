"""Foundation-model surrogates for binary classification."""

from .tabpfn import (
    TabPFNBinaryClassificationModel,
    TabPFNMixedBinaryClassificationModel,
)

__all__ = [
    "TabPFNBinaryClassificationModel",
    "TabPFNMixedBinaryClassificationModel",
]
