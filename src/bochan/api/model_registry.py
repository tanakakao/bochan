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
                "GaussianKroneckerMultiTaskGP",
            ),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskGP"),
            "multifidelity": (
                "bochan.models.regression.gaussian",
                "WideMultiFidelityGP",
            ),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGaussianGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGaussianGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGaussianGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasGaussianGPModel"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAGaussianGPModel"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOGaussianGPModel"),
            "vae": ("bochan.models.regression.gaussian.high_dim", "VAEGaussianGPModel"),
            "rrp": ("bochan.models.regression.gaussian.robust", "RobustRelevancePursuitGaussianGPModel"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticGaussianGPModel"),
            "lightgbm": ("bochan.models.regression.external", "LightGBMRegressorModel"),
            "lightgbm_ensemble": ("bochan.models.regression.external", "LightGBMEnsembleModel"),
            "ngboost": ("bochan.models.regression.external", "NGBoostRegressorModel"),
            "ngboost_ensemble": ("bochan.models.regression.external", "NGBoostEnsembleModel"),
            "random_forest": ("bochan.models.regression.external", "RandomForestRegressorModel"),
            "deep_ensemble": ("bochan.models.regression.neural", "DeepEnsembleRegressorModel"),
            "pfn": ("bochan.models.regression.foundation", "PFNRegressorModel"),
            "tabpfn": ("bochan.models.regression.foundation", "TabPFNRegressorModel"),
            "beta_base": ("bochan.models.regression.non_gaussian.beta.base", "BetaGPModel"),
            "beta_deepgp": ("bochan.models.regression.non_gaussian.beta.deep", "DeepBetaGPModel"),
            "beta_deepkernel": ("bochan.models.regression.non_gaussian.beta.deep", "DeepKernelBetaGPModel"),
            "beta_saas": ("bochan.models.regression.non_gaussian.beta.high_dim", "SaasBetaGPModel"),
            "beta_pca": ("bochan.models.regression.non_gaussian.beta.high_dim", "PCABetaGPModel"),
            "beta_rembo": ("bochan.models.regression.non_gaussian.beta.high_dim", "REMBOBetaGPModel"),
            "beta_rrp": ("bochan.models.regression.non_gaussian.beta.robust", "RobustRelevancePursuitBetaGPModel"),
            "beta_hetero": ("bochan.models.regression.non_gaussian.beta.robust", "HeteroscedasticBetaGPModel"),
            "beta_multitask": ("bochan.models.regression.non_gaussian.beta.base", "BetaMultiTaskGPModel"),
            "beta_wide_multitask": ("bochan.models.regression.non_gaussian.beta.base", "WideBetaMultiTaskGPModel"),
            "beta_kronecker": ("bochan.models.regression.non_gaussian.beta.base", "KroneckerMultiTaskBetaGPModel"),
            "gamma_base": ("bochan.models.regression.non_gaussian.gamma.base", "GammaGPModel"),
            "gamma_deepgp": ("bochan.models.regression.non_gaussian.gamma.deep", "DeepGammaGPModel"),
            "gamma_deepkernel": ("bochan.models.regression.non_gaussian.gamma.deep", "DeepKernelGammaGPModel"),
            "gamma_saas": ("bochan.models.regression.non_gaussian.gamma.high_dim", "SaasGammaGPModel"),
            "gamma_pca": ("bochan.models.regression.non_gaussian.gamma.high_dim", "PCAGammaGPModel"),
            "gamma_rembo": ("bochan.models.regression.non_gaussian.gamma.high_dim", "REMBOGammaGPModel"),
            "gamma_rrp": ("bochan.models.regression.non_gaussian.gamma.robust", "RobustRelevancePursuitGammaGPModel"),
            "gamma_hetero": ("bochan.models.regression.non_gaussian.gamma.robust", "HeteroscedasticGammaGPModel"),
            "gamma_multitask": ("bochan.models.regression.non_gaussian.gamma.base", "GammaMultiTaskGPModel"),
            "gamma_wide_multitask": ("bochan.models.regression.non_gaussian.gamma.base", "WideGammaMultiTaskGPModel"),
            "gamma_kronecker": ("bochan.models.regression.non_gaussian.gamma.base", "KroneckerMultiTaskGammaGPModel"),
            "poisson_base": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonGPModel"),
            "poisson_deepgp": ("bochan.models.regression.non_gaussian.poisson.deep", "DeepPoissonGPModel"),
            "poisson_deepkernel": ("bochan.models.regression.non_gaussian.poisson.deep", "DeepKernelPoissonGPModel"),
            "poisson_saas": ("bochan.models.regression.non_gaussian.poisson.high_dim", "SaasPoissonGPModel"),
            "poisson_pca": ("bochan.models.regression.non_gaussian.poisson.high_dim", "PCAPoissonGPModel"),
            "poisson_rembo": ("bochan.models.regression.non_gaussian.poisson.high_dim", "REMBOPoissonGPModel"),
            "poisson_rrp": ("bochan.models.regression.non_gaussian.poisson.robust", "RobustRelevancePursuitPoissonGPModel"),
            "poisson_hetero": ("bochan.models.regression.non_gaussian.poisson.robust", "HeteroscedasticPoissonGPModel"),
            "poisson_multitask": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonMultiTaskGPModel"),
            "poisson_wide_multitask": ("bochan.models.regression.non_gaussian.poisson.base", "WidePoissonMultiTaskGPModel"),
            "poisson_kronecker": ("bochan.models.regression.non_gaussian.poisson.base", "KroneckerMultiTaskPoissonGPModel"),
            "negative_binomial_base": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialGPModel"),
            "negative_binomial_deepgp": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "DeepNegativeBinomialGPModel"),
            "negative_binomial_deepkernel": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "DeepKernelNegativeBinomialGPModel"),
            "negative_binomial_saas": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "SaasNegativeBinomialGPModel"),
            "negative_binomial_pca": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "PCANegativeBinomialGPModel"),
            "negative_binomial_rembo": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "REMBONegativeBinomialGPModel"),
            "negative_binomial_rrp": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "RobustRelevancePursuitNegativeBinomialGPModel"),
            "negative_binomial_hetero": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "HeteroscedasticNegativeBinomialGPModel"),
            "negative_binomial_multitask": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialMultiTaskGPModel"),
            "negative_binomial_wide_multitask": ("bochan.models.regression.non_gaussian.negative_binomial.base", "WideNegativeBinomialMultiTaskGPModel"),
            "negative_binomial_kronecker": ("bochan.models.regression.non_gaussian.negative_binomial.base", "KroneckerMultiTaskNegativeBinomialGPModel"),
        },
        "multi_objective": {
            "base": ("botorch.models.gp_regression", "SingleTaskGP"),
            "kronecker": (
                "bochan.models.regression.gaussian",
                "GaussianKroneckerMultiTaskGP",
            ),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGaussianGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGaussianGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGaussianGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasGaussianGPModel"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAGaussianGPModel"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOGaussianGPModel"),
            "rrp": ("bochan.models.regression.gaussian.robust", "RobustRelevancePursuitGaussianGPModel"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticGaussianGPModel"),
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
            "deepgp": ("bochan.models.classification.binary.deep", "DeepBinaryClassificationGPModel"),
            "deepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationGPModel"),
            "deepgpdeepkernel": ("bochan.models.classification.binary.deep", "DeepKernelDeepBinaryClassificationGPModel"),
            "saas": ("bochan.models.classification.binary.high_dim", "SaasBinaryClassificationGPModel"),
            "pca": ("bochan.models.classification.binary.high_dim", "PCABinaryClassificationGPModel"),
            "rembo": ("bochan.models.classification.binary.high_dim", "REMBOBinaryClassificationGPModel"),
            "rrp": ("bochan.models.classification.binary.robust", "RobustRelevancePursuitBinaryClassificationGPModel"),
            "hetero": ("bochan.models.classification.binary.robust", "HeteroscedasticBinaryClassificationGPModel"),
            "lightgbm": ("bochan.models.classification.binary.external", "LightGBMBinaryClassificationModel"),
            "lightgbm_ensemble": ("bochan.models.classification.binary.external", "LightGBMBinaryEnsembleModel"),
            "ngboost": ("bochan.models.classification.binary.external", "NGBoostBinaryClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.binary.external", "NGBoostBinaryEnsembleModel"),
            "random_forest": ("bochan.models.classification.binary.external", "RandomForestBinaryClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.binary.neural", "DeepEnsembleBinaryClassificationModel"),
            "tabpfn": ("bochan.models.classification.binary.foundation", "TabPFNBinaryClassificationModel"),
        },
        "ordinal": {
            "base": ("bochan.models.ordinal.base", "OrdinalGPModel"),
            "kronecker": ("bochan.models.ordinal.base", "KroneckerMultiTaskOrdinalGPModel"),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskOrdinalGPModel"),
            "deepgp": ("bochan.models.ordinal.deep", "DeepOrdinalGPModel"),
            "deepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalGPModel"),
            "deepgpdeepkernel": ("bochan.models.ordinal.deep", "DeepKernelDeepOrdinalGPModel"),
            "saas": ("bochan.models.ordinal.high_dim", "SaasOrdinalGPModel"),
            "pca": ("bochan.models.ordinal.high_dim", "PCAOrdinalGPModel"),
            "rembo": ("bochan.models.ordinal.high_dim", "REMBOOrdinalGPModel"),
            "rrp": ("bochan.models.ordinal.robust", "RobustRelevancePursuitOrdinalGPModel"),
            "hetero": ("bochan.models.ordinal.robust", "HeteroscedasticOrdinalGPModel"),
            "lightgbm": ("bochan.models.ordinal.external", "LightGBMOrdinalModel"),
            "lightgbm_ensemble": ("bochan.models.ordinal.external", "LightGBMOrdinalEnsembleModel"),
            "ngboost": ("bochan.models.ordinal.external", "NGBoostOrdinalModel"),
            "ngboost_ensemble": ("bochan.models.ordinal.external", "NGBoostOrdinalEnsembleModel"),
            "random_forest": ("bochan.models.ordinal.external", "RandomForestOrdinalModel"),
            "deep_ensemble": ("bochan.models.ordinal.neural", "DeepEnsembleOrdinalModel"),
        },
        "multiclass": {
            "base": ("bochan.models.classification.multiclass.base", "MulticlassClassificationGPModel"),
            "kronecker": ("bochan.models.classification.multiclass.base", "KroneckerMultiTaskMulticlassClassificationGPModel"),
            "multitask": ("bochan.models.wide_multitask_variants", "WideMultiTaskMulticlassClassificationGPModel"),
            "deepgp": ("bochan.models.classification.multiclass.deep", "DeepMulticlassClassificationGPModel"),
            "deepkernel": ("bochan.models.classification.multiclass.deep", "DeepKernelMulticlassClassificationGPModel"),
            "saas": ("bochan.models.classification.multiclass.high_dim", "SaasMulticlassClassificationGPModel"),
            "pca": ("bochan.models.classification.multiclass.high_dim", "PCAMulticlassClassificationGPModel"),
            "rembo": ("bochan.models.classification.multiclass.high_dim", "REMBOMulticlassClassificationGPModel"),
            "rrp": ("bochan.models.classification.multiclass.robust", "RobustRelevancePursuitMulticlassClassificationGPModel"),
            "hetero": ("bochan.models.classification.multiclass.robust", "HeteroscedasticMulticlassClassificationGPModel"),
            "lightgbm": ("bochan.models.classification.multiclass.external", "LightGBMMulticlassClassificationModel"),
            "lightgbm_ensemble": ("bochan.models.classification.multiclass.external", "LightGBMMulticlassEnsembleModel"),
            "ngboost": ("bochan.models.classification.multiclass.external", "NGBoostMulticlassClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.multiclass.external", "NGBoostMulticlassEnsembleModel"),
            "random_forest": ("bochan.models.classification.multiclass.external", "RandomForestMulticlassClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.multiclass.neural", "DeepEnsembleMulticlassClassificationModel"),
            "tabpfn": ("bochan.models.classification.multiclass.foundation", "TabPFNMulticlassClassificationModel"),
        },
    },
    "mixed": {
        "regression": {
            "base": ("botorch.models.gp_regression_mixed", "MixedSingleTaskGP"),
            "kronecker": ("bochan.models.regression.gaussian", "GaussianMixedKroneckerMultiTaskGP"),
            "multitask": ("bochan.models.wide_mixed_multitask", "WideMixedMultiTaskGP"),
            "multifidelity": (
                "bochan.models.regression.gaussian",
                "WideMixedMultiFidelityGP",
            ),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGaussianMixedGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGaussianMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGaussianMixedGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasGaussianMixedGPModel"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAGaussianMixedGPModel"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOGaussianMixedGPModel"),
            "vae": ("bochan.models.regression.gaussian.high_dim", "VAEGaussianMixedGPModel"),
            "rrp": ("bochan.models.regression.gaussian.robust", "RobustRelevancePursuitGaussianMixedGPModel"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticGaussianMixedGPModel"),
            "lightgbm": ("bochan.models.regression.external", "LightGBMMixedRegressorModel"),
            "lightgbm_ensemble": ("bochan.models.regression.external", "LightGBMMixedEnsembleModel"),
            "ngboost": ("bochan.models.regression.external", "NGBoostMixedRegressorModel"),
            "ngboost_ensemble": ("bochan.models.regression.external", "NGBoostMixedEnsembleModel"),
            "random_forest": ("bochan.models.regression.external", "RandomForestMixedRegressorModel"),
            "deep_ensemble": ("bochan.models.regression.neural", "DeepEnsembleMixedRegressorModel"),
            "tabpfn": ("bochan.models.regression.foundation", "TabPFNMixedRegressorModel"),
            "beta_base": ("bochan.models.regression.non_gaussian.beta.base", "BetaMixedGPModel"),
            "beta_deepgp": ("bochan.models.regression.non_gaussian.beta.deep", "DeepBetaMixedGPModel"),
            "beta_deepkernel": ("bochan.models.regression.non_gaussian.beta.deep", "DeepKernelBetaMixedGPModel"),
            "beta_saas": ("bochan.models.regression.non_gaussian.beta.high_dim", "SaasBetaMixedGPModel"),
            "beta_pca": ("bochan.models.regression.non_gaussian.beta.high_dim", "PCABetaMixedGPModel"),
            "beta_rembo": ("bochan.models.regression.non_gaussian.beta.high_dim", "REMBOBetaMixedGPModel"),
            "beta_rrp": ("bochan.models.regression.non_gaussian.beta.robust", "RobustRelevancePursuitBetaMixedGPModel"),
            "beta_hetero": ("bochan.models.regression.non_gaussian.beta.robust", "HeteroscedasticBetaMixedGPModel"),
            "gamma_base": ("bochan.models.regression.non_gaussian.gamma.base", "GammaMixedGPModel"),
            "gamma_deepgp": ("bochan.models.regression.non_gaussian.gamma.deep", "DeepGammaMixedGPModel"),
            "gamma_deepkernel": ("bochan.models.regression.non_gaussian.gamma.deep", "DeepKernelGammaMixedGPModel"),
            "gamma_saas": ("bochan.models.regression.non_gaussian.gamma.high_dim", "SaasGammaMixedGPModel"),
            "gamma_pca": ("bochan.models.regression.non_gaussian.gamma.high_dim", "PCAGammaMixedGPModel"),
            "gamma_rembo": ("bochan.models.regression.non_gaussian.gamma.high_dim", "REMBOGammaMixedGPModel"),
            "gamma_rrp": ("bochan.models.regression.non_gaussian.gamma.robust", "RobustRelevancePursuitGammaMixedGPModel"),
            "gamma_hetero": ("bochan.models.regression.non_gaussian.gamma.robust", "HeteroscedasticGammaMixedGPModel"),
            "poisson_base": ("bochan.models.regression.non_gaussian.poisson.base", "PoissonMixedGPModel"),
            "poisson_deepgp": ("bochan.models.regression.non_gaussian.poisson.deep", "DeepPoissonMixedGPModel"),
            "poisson_deepkernel": ("bochan.models.regression.non_gaussian.poisson.deep", "DeepKernelPoissonMixedGPModel"),
            "poisson_saas": ("bochan.models.regression.non_gaussian.poisson.high_dim", "SaasPoissonMixedGPModel"),
            "poisson_pca": ("bochan.models.regression.non_gaussian.poisson.high_dim", "PCAPoissonMixedGPModel"),
            "poisson_rembo": ("bochan.models.regression.non_gaussian.poisson.high_dim", "REMBOPoissonMixedGPModel"),
            "poisson_rrp": ("bochan.models.regression.non_gaussian.poisson.robust", "RobustRelevancePursuitPoissonMixedGPModel"),
            "poisson_hetero": ("bochan.models.regression.non_gaussian.poisson.robust", "HeteroscedasticPoissonMixedGPModel"),
            "negative_binomial_base": ("bochan.models.regression.non_gaussian.negative_binomial.base", "NegativeBinomialMixedGPModel"),
            "negative_binomial_deepgp": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "DeepNegativeBinomialMixedGPModel"),
            "negative_binomial_deepkernel": ("bochan.models.regression.non_gaussian.negative_binomial.deep", "DeepKernelNegativeBinomialMixedGPModel"),
            "negative_binomial_saas": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "SaasNegativeBinomialMixedGPModel"),
            "negative_binomial_pca": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "PCANegativeBinomialMixedGPModel"),
            "negative_binomial_rembo": ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "REMBONegativeBinomialMixedGPModel"),
            "negative_binomial_rrp": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "RobustRelevancePursuitNegativeBinomialMixedGPModel"),
            "negative_binomial_hetero": ("bochan.models.regression.non_gaussian.negative_binomial.robust", "HeteroscedasticNegativeBinomialMixedGPModel"),
        },
        "multi_objective": {
            "base": ("botorch.models.gp_regression_mixed", "MixedSingleTaskGP"),
            "kronecker": ("bochan.models.regression.gaussian", "GaussianMixedKroneckerMultiTaskGP"),
            "multitask": ("bochan.models.wide_mixed_multitask", "WideMixedMultiTaskGP"),
            "deepgp": ("bochan.models.regression.gaussian.deep", "DeepGaussianMixedGPModel"),
            "deepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelGaussianMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.regression.gaussian.deep", "DeepKernelDeepGaussianMixedGPModel"),
            "saas": ("bochan.models.regression.gaussian.high_dim", "SaasGaussianMixedGPModel"),
            "pca": ("bochan.models.regression.gaussian.high_dim", "PCAGaussianMixedGPModel"),
            "rembo": ("bochan.models.regression.gaussian.high_dim", "REMBOGaussianMixedGPModel"),
            "rrp": ("bochan.models.regression.gaussian.robust", "RobustRelevancePursuitGaussianMixedGPModel"),
            "hetero": ("bochan.models.regression.gaussian.robust", "HeteroscedasticGaussianMixedGPModel"),
        },
        "binary": {
            "base": ("bochan.models.classification.binary.base", "BinaryClassificationMixedGPModel"),
            "kronecker": ("bochan.models.classification.binary.base", "KroneckerMultiTaskBinaryClassificationMixedGPModel"),
            "multitask": ("bochan.models.classification.binary.base", "MultiTaskBinaryClassificationMixedGPModel"),
            "multifidelity": (
                "bochan.models.classification.binary.base",
                "WideMixedMultiFidelityBinaryClassificationGPModel",
            ),
            "deepgp": ("bochan.models.classification.binary.deep", "DeepBinaryClassificationMixedGPModel"),
            "deepkernel": ("bochan.models.classification.binary.deep", "DeepKernelBinaryClassificationMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.classification.binary.deep", "DeepKernelDeepBinaryClassificationMixedGPModel"),
            "saas": ("bochan.models.classification.binary.high_dim", "SaasBinaryClassificationMixedGPModel"),
            "pca": ("bochan.models.classification.binary.high_dim", "PCABinaryClassificationMixedGPModel"),
            "rembo": ("bochan.models.classification.binary.high_dim", "REMBOBinaryClassificationMixedGPModel"),
            "rrp": ("bochan.models.classification.binary.robust", "RobustRelevancePursuitBinaryClassificationMixedGPModel"),
            "hetero": ("bochan.models.classification.binary.robust", "HeteroscedasticBinaryClassificationMixedGPModel"),
            "lightgbm": ("bochan.models.classification.binary.external", "LightGBMMixedBinaryClassificationModel"),
            "lightgbm_ensemble": ("bochan.models.classification.binary.external", "LightGBMMixedBinaryEnsembleModel"),
            "ngboost": ("bochan.models.classification.binary.external", "NGBoostMixedBinaryClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.binary.external", "NGBoostMixedBinaryEnsembleModel"),
            "random_forest": ("bochan.models.classification.binary.external", "RandomForestMixedBinaryClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.binary.neural", "DeepEnsembleMixedBinaryClassificationModel"),
            "tabpfn": ("bochan.models.classification.binary.foundation", "TabPFNMixedBinaryClassificationModel"),
        },
        "ordinal": {
            "base": ("bochan.models.ordinal.base", "OrdinalMixedGPModel"),
            "kronecker": ("bochan.models.ordinal.base", "KroneckerMultiTaskOrdinalMixedGPModel"),
            "multitask": ("bochan.models.ordinal.base", "MultiTaskOrdinalMixedGPModel"),
            "deepgp": ("bochan.models.ordinal.deep", "DeepOrdinalMixedGPModel"),
            "deepkernel": ("bochan.models.ordinal.deep", "DeepKernelOrdinalMixedGPModel"),
            "deepgpdeepkernel": ("bochan.models.ordinal.deep", "DeepKernelDeepOrdinalMixedGPModel"),
            "saas": ("bochan.models.ordinal.high_dim", "SaasOrdinalMixedGPModel"),
            "pca": ("bochan.models.ordinal.high_dim", "PCAOrdinalMixedGPModel"),
            "rembo": ("bochan.models.ordinal.high_dim", "REMBOOrdinalMixedGPModel"),
            "rrp": ("bochan.models.ordinal.robust", "RobustRelevancePursuitOrdinalMixedGPModel"),
            "hetero": ("bochan.models.ordinal.robust", "HeteroscedasticOrdinalMixedGPModel"),
            "lightgbm": ("bochan.models.ordinal.external", "LightGBMMixedOrdinalModel"),
            "lightgbm_ensemble": ("bochan.models.ordinal.external", "LightGBMMixedOrdinalEnsembleModel"),
            "ngboost": ("bochan.models.ordinal.external", "NGBoostMixedOrdinalModel"),
            "ngboost_ensemble": ("bochan.models.ordinal.external", "NGBoostMixedOrdinalEnsembleModel"),
            "random_forest": ("bochan.models.ordinal.external", "RandomForestMixedOrdinalModel"),
            "deep_ensemble": ("bochan.models.ordinal.neural", "DeepEnsembleMixedOrdinalModel"),
        },
        "multiclass": {
            "base": ("bochan.models.classification.multiclass.base", "MulticlassClassificationMixedGPModel"),
            "kronecker": ("bochan.models.classification.multiclass.base", "KroneckerMultiTaskMulticlassClassificationMixedGPModel"),
            "multitask": ("bochan.models.classification.multiclass.base", "MultiTaskMulticlassClassificationMixedGPModel"),
            "deepgp": ("bochan.models.classification.multiclass.deep", "DeepMulticlassClassificationMixedGPModel"),
            "deepkernel": ("bochan.models.classification.multiclass.deep", "DeepKernelMulticlassClassificationMixedGPModel"),
            "saas": ("bochan.models.classification.multiclass.high_dim", "SaasMulticlassClassificationMixedGPModel"),
            "pca": ("bochan.models.classification.multiclass.high_dim", "PCAMulticlassClassificationMixedGPModel"),
            "rembo": ("bochan.models.classification.multiclass.high_dim", "REMBOMulticlassClassificationMixedGPModel"),
            "rrp": ("bochan.models.classification.multiclass.robust", "RobustRelevancePursuitMulticlassClassificationMixedGPModel"),
            "hetero": ("bochan.models.classification.multiclass.robust", "HeteroscedasticMulticlassClassificationMixedGPModel"),
            "lightgbm": ("bochan.models.classification.multiclass.external", "LightGBMMixedMulticlassClassificationModel"),
            "lightgbm_ensemble": ("bochan.models.classification.multiclass.external", "LightGBMMixedMulticlassEnsembleModel"),
            "ngboost": ("bochan.models.classification.multiclass.external", "NGBoostMixedMulticlassClassificationModel"),
            "ngboost_ensemble": ("bochan.models.classification.multiclass.external", "NGBoostMixedMulticlassEnsembleModel"),
            "random_forest": ("bochan.models.classification.multiclass.external", "RandomForestMixedMulticlassClassificationModel"),
            "deep_ensemble": ("bochan.models.classification.multiclass.neural", "DeepEnsembleMixedMulticlassClassificationModel"),
            "tabpfn": ("bochan.models.classification.multiclass.foundation", "TabPFNMixedMulticlassClassificationModel"),
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
