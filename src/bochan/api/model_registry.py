"""Default model registry for the high-level bochan API.

The public API is intended to expose bochan's implemented model families through
simple string keys such as ``task_type="regression"`` and ``model_type="base"``.

This module keeps the registry lazy: model modules are imported only when the
corresponding registry entry is actually requested. This avoids making
``import bochan.api`` unnecessarily heavy.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ModelPath = tuple[str, str]
RegistryTree = dict[str, Any]


def _import_from_path(path: ModelPath) -> Any:
    import importlib

    module_name, attr_name = path
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class LazyModelRegistry(Mapping[str, Any]):
    """Nested mapping that lazily imports model classes at leaf nodes."""

    def __init__(self, tree: RegistryTree) -> None:
        self._tree = tree

    def __getitem__(self, key: str) -> Any:
        value = self._tree[key]
        if isinstance(value, dict):
            return LazyModelRegistry(value)
        if isinstance(value, tuple) and len(value) == 2:
            return _import_from_path(value)
        return value

    def __iter__(self):
        return iter(self._tree)

    def __len__(self) -> int:
        return len(self._tree)

    def __contains__(self, key: object) -> bool:
        return key in self._tree

    def raw(self) -> RegistryTree:
        """Return the raw path-based registry tree."""
        return self._tree


_MODEL_REGISTRY_TREE: RegistryTree = {
    "normal": {
        "regression": {
            "base": ("botorch.models.gp_regression", "SingleTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCASingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOSingleTaskGP"),
            "vae": ("bochan.models.regression.gaussian.high_dim", "VAESingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticSingleTaskGP"),
        },
        "multi_objective": {
            "base": ("botorch.models.gp_regression", "SingleTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCASingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOSingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticSingleTaskGP"),
        },
        "binary": {
            "base": ("bochan.models.classification.binary.base", "BinaryClassificationGPModel"),
            "deepgp": ("bochan.models.classification.binary.deep", "BinaryClassificationDeepGPModel"),
            "deepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationGPModel"),
            "deepgpdeepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationDeepGPModel"),
            "saas": ("bochan.models.classification.binary.high_dim", "SaasBinaryClassificationGPModel"),
            "pca": ("bochan.models.classification.binary.high_dim", "PCABinaryClassificationGPModel"),
            "rembo": ("bochan.models.classification.binary.high_dim", "REMBOBinaryClassificationGPModel"),
            "rrp": ("bochan.models.classification.binary.robust", "OutlierRelevancePursuitBinaryClassificationGPModel"),
            "hetero": ("bochan.models.classification.binary.robust", "HeteroscedasticBinaryClassificationGPModel"),
        },
        "ordinal": {
            "base": ("bochan.models.ordinal.base", "OrdinalGPModel"),
            "deepgp": ("bochan.models.ordinal.deep", "OrdinalDeepGPModel"),
            "deepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalGPModel"),
            "deepgpdeepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalDeepGPModel"),
            "saas": ("bochan.models.ordinal.high_dim", "SaasOrdinalGPModel"),
            "pca": ("bochan.models.ordinal.high_dim", "PCAOrdinalGPModel"),
            "rembo": ("bochan.models.ordinal.high_dim", "REMBOOrdinalGPModel"),
            "rrp": ("bochan.models.ordinal.robust", "OutlierRelevancePursuitOrdinalGPModel"),
            "hetero": ("bochan.models.ordinal.robust", "HeteroscedasticOrdinalGPModel"),
        },
        "multiclass": {
            "base": ("bochan.models.classification.multiclass.base", "MulticlassClassificationGPModel"),
            "deepgp": ("bochan.models.classification.multiclass.deep", "MulticlassDeepGPModel"),
            "deepkernel": ("bochan.models.classification.multiclass.deep", "DeepKernelMulticlassClassificationGPModel"),
            "saas": ("bochan.models.classification.multiclass.high_dim", "SaasMulticlassClassificationGPModel"),
            "pca": ("bochan.models.classification.multiclass.high_dim", "PCAMulticlassClassificationGPModel"),
            "rembo": ("bochan.models.classification.multiclass.high_dim", "REMBOMulticlassClassificationGPModel"),
            "rrp": ("bochan.models.classification.multiclass.robust", "OutlierRelevancePursuitMulticlassClassificationGPModel"),
            "hetero": ("bochan.models.classification.multiclass.robust", "HeteroscedasticMulticlassClassificationGPModel"),
        },
    },
    "mixed": {
        "regression": {
            "base": ("botorch.models.gp_regression_mixed", "MixedSingleTaskGP"),
            "kronecker": ("bochan.models.regression.gaussian", "MixedKroneckerMultiTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepMixedGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepMixedGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasMixedSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAMixedSingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOMixedSingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitMixedSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticMixedSingleTaskGP"),
        },
        "multi_objective": {
            "base": ("botorch.models.gp_regression_mixed", "MixedSingleTaskGP"),
            "kronecker": ("bochan.models.regression.gaussian", "MixedKroneckerMultiTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepMixedGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepMixedGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasMixedSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAMixedSingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOMixedSingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitMixedSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticMixedSingleTaskGP"),
        },
        "binary": {
            "base": ("bochan.models.classification.binary.base", "BinaryClassificationMixedGPModel"),
            "kronecker": ("bochan.models.classification.binary.base", "KroneckerMultiTaskBinaryClassificationMixedGPModel"),
            "multitask": ("bochan.models.classification.binary.base", "MultiTaskBinaryClassificationMixedGPModel"),
            "deepgp": ("bochan.models.classification.binary.deep", "BinaryClassificationMixedDeepGPModel"),
            "deepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationMixedDeepGPModel"),
            "saas": ("bochan.models.classification.binary.high_dim", "SaasBinaryClassificationMixedGPModel"),
            "pca": ("bochan.models.classification.binary.high_dim", "PCABinaryClassificationMixedGPModel"),
            "rembo": ("bochan.models.classification.binary.high_dim", "REMBOBinaryClassificationMixedGPModel"),
            "rrp": ("bochan.models.classification.binary.robust", "OutlierRelevancePursuitBinaryClassificationMixedGPModel"),
            "hetero": ("bochan.models.classification.binary.robust", "HeteroscedasticBinaryClassificationMixedGPModel"),
        },
        "ordinal": {
            "base": ("bochan.models.ordinal.base", "OrdinalMixedGPModel"),
            "kronecker": ("bochan.models.ordinal.base", "KroneckerMultiTaskOrdinalMixedGPModel"),
            "multitask": ("bochan.models.ordinal.base", "MultiTaskOrdinalMixedGPModel"),
            "deepgp": ("bochan.models.ordinal.deep", "OrdinalMixedDeepGPModel"),
            "deepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalMixedDeepGPModel"),
            "saas": ("bochan.models.ordinal.high_dim", "SaasOrdinalMixedGPModel"),
            "pca": ("bochan.models.ordinal.high_dim", "PCAOrdinalMixedGPModel"),
            "rembo": ("bochan.models.ordinal.high_dim", "REMBOOrdinalMixedGPModel"),
            "rrp": ("bochan.models.ordinal.robust", "OutlierRelevancePursuitOrdinalMixedGPModel"),
            "hetero": ("bochan.models.ordinal.robust", "HeteroscedasticOrdinalMixedGPModel"),
        },
        "multiclass": {
            "base": ("bochan.models.classification.multiclass.base", "MulticlassClassificationMixedGPModel"),
            "kronecker": ("bochan.models.classification.multiclass.base", "KroneckerMultiTaskMulticlassClassificationMixedGPModel"),
            "multitask": ("bochan.models.classification.multiclass.base", "MultiTaskMulticlassClassificationMixedGPModel"),
            "deepgp": ("bochan.models.classification.multiclass.deep", "MulticlassMixedDeepGPModel"),
            "deepkernel": ("bochan.models.classification.multiclass.deep", "DeepKernelMulticlassClassificationMixedGPModel"),
            "saas": ("bochan.models.classification.multiclass.high_dim", "SaasMulticlassClassificationMixedGPModel"),
            "pca": ("bochan.models.classification.multiclass.high_dim", "PCAMulticlassClassificationMixedGPModel"),
            "rembo": ("bochan.models.classification.multiclass.high_dim", "REMBOMulticlassClassificationMixedGPModel"),
            "rrp": ("bochan.models.classification.multiclass.robust", "OutlierRelevancePursuitMulticlassClassificationMixedGPModel"),
            "hetero": ("bochan.models.classification.multiclass.robust", "HeteroscedasticMulticlassClassificationMixedGPModel"),
        },
    },
}


MODEL_REGISTRY = LazyModelRegistry(_MODEL_REGISTRY_TREE)
DEFAULT_MODEL_REGISTRY = MODEL_REGISTRY


__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
]
