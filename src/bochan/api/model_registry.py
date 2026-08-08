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
            "kronecker": (
                "bochan.models.regression.gaussian",
                "PerturbationSupportedKroneckerMultiTaskGP",
            ),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskGP"),
            "multifidelity": (
                "bochan.models.regression.gaussian",
                "WideMultiFidelityGP",
            ),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCASingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOSingleTaskGP"),
            "vae": ("bochan.models.regression.gaussian.high_dim", "VAESingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticSingleTaskGP"),
            "ngboost": ("bochan.models.regression.boosting", "NGBoostRegressorModel"),
            "ngboost_ensemble": ("bochan.models.regression.boosting", "NGBoostEnsembleModel"),
            "random_forest": ("bochan.models.regression.boosting", "RandomForestRegressorModel"),
            "deep_ensemble": ("bochan.models.regression.neural", "DeepEnsembleRegressorModel"),
            "pfn": ("bochan.models.regression.foundation", "PFNRegressorModel"),
            "beta_base": ("bochan.models.regression.non_gaussian.beta.base", "BetaGPModel"),
            "beta_deepgp": ("bochan.models.regression.non_gaussian.beta.deep", "BetaDeepGPModel"),
            "beta_deepkernel": ("bochan.models.regression.non_gaussian.beta.deep", "DeepKernelBetaGPModel"),
            "beta_saas": ("bochan.models.regression.non_gaussian.beta.high_dim", "SaasBetaGPModel"),
            "beta_pca": ("bochan.models.regression.non_gaussian.beta.high_dim", "PCABetaGPModel"),
            "beta_rembo": ("bochan.models.regression.non_gaussian.beta.high_dim", "REMBOSingleTaskGP"),
            "beta_rrp": ("bochan.models.regression.non_gaussian.beta.robust", "OutlierRelevancePursuitBetaGPModel"),
            "beta_hetero": ("bochan.models.regression.non_gaussian.beta.robust", "HeteroscedasticBetaGPModel"),
            "beta_multitask": ("bochan.models.regression.non_gaussian.beta.base", "BetaMultiTaskGPModel"),
            "beta_wide_multitask": ("bochan.models.regression.non_gaussian.beta.base", "WideBetaMultiTaskGPModel"),
            "beta_kronecker": ("bochan.models.regression.non_gaussian.beta.base", "KroneckerMultiTaskBetaGPModel"),
            "gamma_base": ("bochan.models.regression.non_gaussian.gamma.base", "GammaGPModel"),
            "gamma_deepgp": ("bochan.models.regression.non_gaussian.gamma.deep", "GammaDeepGPModel"),
            "gamma_deepkernel": ("bochan.models.regression.non_gaussian.gamma.deep", "DeepKernelGammaGPModel"),
            "gamma_saas": ("bochan.models.regression.non_gaussian.gamma.high_dim", "SaasGammaGPModel"),
            "gamma_pca": ("bochan.models.regression.non_gaussian.gamma.high_dim", "PCAGammaGPModel"),
            "gamma_rembo": ("bochan.models.regression.non_gaussian.gamma.high_dim", "REMBOGammaGPModel"),
            "gamma_rrp": ("bochan.models.regression.non_gaussian.gamma.robust", "OutlierRelevancePursuitGammaGPModel"),
            "gamma_hetero": ("bochan.models.regression.non_gaussian.gamma.robust", "HeteroscedasticGammaGPModel"),
            "gamma_multitask": ("bochan.models.regression.non_gaussian.gamma.base", "GammaMultiTaskGPModel"),
            "gamma_wide_multitask": ("bochan.models.regression.non_gaussian.gamma.base", "WideGammaMultiTaskGPModel"),
            "gamma_kronecker": ("bochan.models.regression.non_gaussian.gamma.base", "KroneckerMultiTaskGammaGPModel"),
            "poisson_base": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonGPModel"),
            "poisson_deepgp": ("bochan.models.regression.non_gaussian.poisson.deep", "PoissonDeepGPModel"),
            "poisson_deepkernel": ("bochan.models.regression.non_gaussian.poisson.deep", "DeepKernelPoissonGPModel"),
            "poisson_saas": ("bochan.models.regression.non_gaussian.poisson.high_dim", "SaasPoissonGPModel"),
            "poisson_pca": ("bochan.models.regression.non_gaussian.poisson.high_dim", "PCAPoissonGPModel"),
            "poisson_rembo": ("bochan.models.regression.non_gaussian.poisson.high_dim", "REMBOPoissonGPModel"),
            "poisson_rrp": ("bochan.models.regression.non_gaussian.poisson.robust", "OutlierRelevancePursuitPoissonGPModel"),
            "poisson_hetero": ("bochan.models.regression.non_gaussian.poisson.robust", "HeteroscedasticPoissonGPModel"),
            "poisson_multitask": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonMultiTaskGPModel"),
            "poisson_wide_multitask": ("bochan.models.regression.non_gaussian.poisson.base", "WidePoissonMultiTaskGPModel"),
            "poisson_kronecker": ("bochan.models.regression.non_gaussian.poisson.base", "KroneckerMultiTaskPoissonGPModel"),
            "negative_binomial_base": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialGPModel"),
            "negative_binomial_deepgp": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "NegativeBinomialDeepGPModel"),
            "negative_binomial_deepkernel": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "DeepKernelNegativeBinomialGPModel"),
            "negative_binomial_saas": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "SaasNegativeBinomialGPModel"),
            "negative_binomial_pca": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "PCANegativeBinomialGPModel"),
            "negative_binomial_rembo": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "REMBONegativeBinomialGPModel"),
            "negative_binomial_rrp": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "OutlierRelevancePursuitNegativeBinomialGPModel"),
            "negative_binomial_hetero": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "HeteroscedasticNegativeBinomialGPModel"),
            "negative_binomial_multitask": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialMultiTaskGPModel"),
            "negative_binomial_wide_multitask": ("bochan.models.regression.non_gaussian.negative_binomial.base", "WideNegativeBinomialGPModel"),
            "negative_binomial_kronecker": ("bochan.models.regression.non_gaussian.negative_binomial.base", "KroneckerMultiTaskNegativeBinomialGPModel"),
        },
        "multi_objective": {
            "base": ("botorch.models.gp_regression", "SingleTaskGP"),
            "kronecker": (
                "bochan.models.regression.gaussian",
                "PerturbationSupportedKroneckerMultiTaskGP",
            ),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCASingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOSingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticSingleTaskGP"),
            "beta_multitask": ("bochan.models.regression.non_gaussian.beta.base", "BetaMultiTaskGPModel"),
            "beta_wide_multitask": ("bochan.models.regression.non_gaussian.beta.base", "WideBetaMultiTaskGPModel"),
            "beta_kronecker": ("bochan.models.regression.non_gaussian.beta.base", "KroneckerMultiTaskBetaGPModel"),
            "gamma_multitask": ("bochan.models.regression.non_gaussian.gamma.base", "GammaMultiTaskGPModel"),
            "gamma_wide_multitask": ("bochan.models.regression.non_gaussian.gamma.base", "WideGammaMultiTaskGPModel"),
            "gamma_kronecker": ("bochan.models.regression.non_gaussian.gamma.base", "KroneckerMultiTaskGammaGPModel"),
            "poisson_multitask": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonMultiTaskGPModel"),
            "poisson_wide_multitask": ("bochan.models.regression.non_gaussian.poisson.base", "WidePoissonMultiTaskGPModel"),
            "poisson_kronecker": ("bochan.models.regression.non_gaussian.poisson.base", "KroneckerMultiTaskPoissonGPModel"),
            "negative_binomial_multitask": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialMultiTaskGPModel"),
            "negative_binomial_wide_multitask": ("bochan.models.regression.non_gaussian.negative_binomial.base", "WideNegativeBinomialMultiTaskGPModel"),
            "negative_binomial_kronecker": ("bochan.models.regression.non_gaussian.negative_binomial.base", "KroneckerMultiTaskNegativeBinomialGPModel"),
        },
        "binary": {
            "base": ("bochan.models.classification.binary.base", "BinaryClassificationGPModel"),
            "kronecker": ("bochan.models.classification.binary.base", "KroneckerMultiTaskBinaryClassificationGPModel"),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskBinaryClassificationGPModel"),
            "multifidelity": (
                "bochan.models.classification.binary.base",
                "WideMultiFidelityBinaryClassificationGPModel",
            ),
            "deepgp": ("bochan.models.classification.binary.deep", "BinaryClassificationDeepGPModel"),
            "deepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationGPModel"),
            "deepgpdeepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationDeepGPModel"),
            "saas": ("bochan.models.classification.binary.high_dim", "SaasBinaryClassificationGPModel"),
            "pca": ("bochan.models.classification.binary.high_dim", "PCABinaryClassificationGPModel"),
            "rembo": ("bochan.models.classification.binary.high_dim", "REMBOBinaryClassificationGPModel"),
            "rrp": ("bochan.models.classification.binary.robust", "OutlierRelevancePursuitBinaryClassificationGPModel"),
            "hetero": ("bochan.models.classification.binary.robust", "HeteroscedasticBinaryClassificationGPModel"),
            "ngboost": ("bochan.models.classification.external", "NGBoostBinaryClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.external", "NGBoostBinaryEnsembleModel"),
            "random_forest": ("bochan.models.classification.external", "RandomForestBinaryClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.neural", "DeepEnsembleBinaryClassificationModel"),
        },
        "ordinal": {
            "base": ("bochan.models.ordinal.base", "OrdinalGPModel"),
            "kronecker": ("bochan.models.ordinal.base", "KroneckerMultiTaskOrdinalGPModel"),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskOrdinalGPModel"),
            "deepgp": ("bochan.models.ordinal.deep", "OrdinalDeepGPModel"),
            "deepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalGPModel"),
            "deepgpdeepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalDeepGPModel"),
            "saas": ("bochan.models.ordinal.high_dim", "SaasOrdinalGPModel"),
            "pca": ("bochan.models.ordinal.high_dim", "PCAOrdinalGPModel"),
            "rembo": ("bochan.models.ordinal.high_dim", "REMBOOrdinalGPModel"),
            "rrp": ("bochan.models.ordinal.robust", "OutlierRelevancePursuitOrdinalGPModel"),
            "hetero": ("bochan.models.ordinal.robust", "HeteroscedasticOrdinalGPModel"),
            "deep_ensemble": ("bochan.models.ordinal.neural", "DeepEnsembleOrdinalModel"),
        },
        "multiclass": {
            "base": ("bochan.models.classification.multiclass.base", "MulticlassClassificationGPModel"),
            "kronecker": ("bochan.models.classification.multiclass.base", "KroneckerMultiTaskMulticlassClassificationGPModel"),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskMulticlassClassificationGPModel"),
            "deepgp": ("bochan.models.classification.multiclass.deep", "MulticlassDeepGPModel"),
            "deepkernel": ("bochan.models.classification.multiclass.deep", "DeepKernelMulticlassClassificationGPModel"),
            "saas": ("bochan.models.classification.multiclass.high_dim", "SaasMulticlassClassificationGPModel"),
            "pca": ("bochan.models.classification.multiclass.high_dim", "PCAMulticlassClassificationGPModel"),
            "rembo": ("bochan.models.classification.multiclass.high_dim", "REMBOMulticlassClassificationGPModel"),
            "rrp": ("bochan.models.classification.multiclass.robust", "OutlierRelevancePursuitMulticlassClassificationGPModel"),
            "hetero": ("bochan.models.classification.multiclass.robust", "HeteroscedasticMulticlassClassificationGPModel"),
            "ngboost": ("bochan.models.classification.external", "NGBoostMulticlassClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.external", "NGBoostMulticlassEnsembleModel"),
            "random_forest": ("bochan.models.classification.external", "RandomForestMulticlassClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.neural", "DeepEnsembleMulticlassClassificationModel"),
        },
    },
    "mixed": {
        "regression": {
            "base": ("botorch.models.gp_regression_mixed", "MixedSingleTaskGP"),
            "kronecker": ("bochan.models.regression.gaussian", "MixedKroneckerMultiTaskGP"),
            "multifidelity": (
                "bochan.models.regression.gaussian",
                "WideMixedMultiFidelityGP",
            ),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepMixedGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepMixedGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasMixedSingleTaskGP"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAMixedSingleTaskGP"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOMixedSingleTaskGP"),
            "vae": ("bochan.models.regression.gaussian.high_dim", "VAEMixedSingleTaskGP"),
            "rrp": ("bochan.models.regression.gaussian.robust", "SafeRobustRelevancePursuitMixedSingleTaskGP"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticMixedSingleTaskGP"),
            "ngboost": ("bochan.models.regression.boosting", "NGBoostMixedRegressorModel"),
            "ngboost_ensemble": ("bochan.models.regression.boosting", "NGBoostMixedEnsembleModel"),
            "random_forest": ("bochan.models.regression.boosting", "RandomForestMixedRegressorModel"),
            "deep_ensemble": ("bochan.models.regression.neural", "DeepEnsembleMixedRegressorModel"),
            "beta_base": ("bochan.models.regression.non_gaussian.beta.base", "BetaMixedGPModel"),
            "beta_deepgp": ("bochan.models.regression.non_gaussian.beta.deep", "BetaMixedDeepGPModel"),
            "beta_deepkernel": ("bochan.models.regression.non_gaussian.beta.deep", "DeepKernelBetaMixedGPModel"),
            "beta_saas": ("bochan.models.regression.non_gaussian.beta.high_dim", "SaasBetaMixedGPModel"),
            "beta_pca": ("bochan.models.regression.non_gaussian.beta.high_dim", "PCABetaMixedGPModel"),
            "beta_rembo": ("bochan.models.regression.non_gaussian.beta.high_dim", "REMBOBetaMixedGPModel"),
            "beta_rrp": ("bochan.models.regression.non_gaussian.beta.robust", "OutlierRelevancePursuitBetaMixedGPModel"),
            "beta_hetero": ("bochan.models.regression.non_gaussian.beta.robust", "HeteroscedasticBetaMixedGPModel"),
            "gamma_base": ("bochan.models.regression.non_gaussian.gamma.base", "GammaMixedGPModel"),
            "gamma_deepgp": ("bochan.models.regression.non_gaussian.gamma.deep", "GammaMixedDeepGPModel"),
            "gamma_deepkernel": ("bochan.models.regression.non_gaussian.gamma.deep", "DeepKernelGammaMixedGPModel"),
            "gamma_saas": ("bochan.models.regression.non_gaussian.gamma.high_dim", "SaasGammaMixedGPModel"),
            "gamma_pca": ("bochan.models.regression.non_gaussian.gamma.high_dim", "PCAGammaMixedGPModel"),
            "gamma_rembo": ("bochan.models.regression.non_gaussian.gamma.high_dim", "REMBOGammaMixedGPModel"),
            "gamma_rrp": ("bochan.models.regression.non_gaussian.gamma.robust", "OutlierRelevancePursuitGammaMixedGPModel"),
            "gamma_hetero": ("bochan.models.regression.non_gaussian.gamma.robust", "HeteroscedasticGammaMixedGPModel"),
            "poisson_base": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonMixedGPModel"),
            "poisson_deepgp": ("bochan.models.regression.non_gaussian.poisson.deep", "PoissonMixedDeepGPModel"),
            "poisson_deepkernel": ("bochan.models.regression.non_gaussian.poisson.deep", "DeepKernelPoissonMixedGPModel"),
            "poisson_saas": ("bochan.models.regression.non_gaussian.poisson.high_dim", "SaasPoissonMixedGPModel"),
            "poisson_pca": ("bochan.models.regression.non_gaussian.poisson.high_dim", "PCAPoissonMixedGPModel"),
            "poisson_rembo": ("bochan.models.regression.non_gaussian.poisson.high_dim", "REMBOPoissonMixedGPModel"),
            "poisson_rrp": ("bochan.models.regression.non_gaussian.poisson.robust", "OutlierRelevancePursuitPoissonMixedGPModel"),
            "poisson_hetero": ("bochan.models.regression.non_gaussian.poisson.robust", "HeteroscedasticPoissonMixedGPModel"),
            "negative_binomial_base": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialMixedGPModel"),
            "negative_binomial_deepgp": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "NegativeBinomialMixedDeepGPModel"),
            "negative_binomial_deepkernel": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "DeepKernelNegativeBinomialMixedGPModel"),
            "negative_binomial_saas": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "SaasNegativeBinomialMixedGPModel"),
            "negative_binomial_pca": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "PCANegativeBinomialMixedGPModel"),
            "negative_binomial_rembo": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "REMBONegativeBinomialMixedGPModel"),
            "negative_binomial_rrp": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "OutlierRelevancePursuitNegativeBinomialMixedGPModel"),
            "negative_binomial_hetero": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "HeteroscedasticNegativeBinomialMixedGPModel"),
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
            "multifidelity": (
                "bochan.models.classification.binary.base",
                "WideMixedMultiFidelityBinaryClassificationGPModel",
            ),
            "deepgp": ("bochan.models.classification.binary.deep", "BinaryClassificationMixedDeepGPModel"),
            "deepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationMixedDeepGPModel"),
            "saas": ("bochan.models.classification.binary.high_dim", "SaasBinaryClassificationMixedGPModel"),
            "pca": ("bochan.models.classification.binary.high_dim", "PCABinaryClassificationMixedGPModel"),
            "rembo": ("bochan.models.classification.binary.high_dim", "REMBOBinaryClassificationMixedGPModel"),
            "rrp": ("bochan.models.classification.binary.robust", "OutlierRelevancePursuitBinaryClassificationMixedGPModel"),
            "hetero": ("bochan.models.classification.binary.robust", "HeteroscedasticBinaryClassificationMixedGPModel"),
            "ngboost": ("bochan.models.classification.external", "NGBoostMixedBinaryClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.external", "NGBoostMixedBinaryEnsembleModel"),
            "random_forest": ("bochan.models.classification.external", "RandomForestMixedBinaryClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.neural", "DeepEnsembleMixedBinaryClassificationModel"),
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
            "deep_ensemble": ("bochan.models.ordinal.neural", "DeepEnsembleMixedOrdinalModel"),
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
            "ngboost": ("bochan.models.classification.external", "NGBoostMixedMulticlassClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.external", "NGBoostMixedMulticlassEnsembleModel"),
            "random_forest": ("bochan.models.classification.external", "RandomForestMixedMulticlassClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.neural", "DeepEnsembleMixedMulticlassClassificationModel"),
        },
    },
}


MODEL_REGISTRY = LazyModelRegistry(_MODEL_REGISTRY_TREE)
DEFAULT_MODEL_REGISTRY = MODEL_REGISTRY


def _install_bayesian_optimizer_llm_api() -> None:
    """Attach LLM suggestion methods after the public optimizer class is finalized."""

    from .engine_defaults import BayesianOptimizer
    from .llm_suggestion import install_bayesian_optimizer_llm_api

    install_bayesian_optimizer_llm_api(BayesianOptimizer)


_install_bayesian_optimizer_llm_api()


__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
]
