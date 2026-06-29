from .classification import (
    ClassificationFitResult,
    fit_binary_classifier_mll,
    fit_classification_gp,
    fit_classification_mll,
    fit_multiclass_gp,
    fit_multiclass_mll,
)
from .deep import fit_deep_full_batch_mll, fit_deepgp_mll, fit_deepkernel_mll
from .non_gaussian import (
    FitResult,
    fit_beta_gp,
    fit_beta_mll,
    fit_gamma_gp,
    fit_gamma_mll,
    fit_gpytorch_mll_like_botorch,
    fit_negative_binomial_gp,
    fit_negative_binomial_mll,
    fit_non_gaussian_gp,
    fit_non_gaussian_mll,
    fit_poisson_gp,
    fit_poisson_mll,
)
from .ordinal import (
    fit_ordinal_gp,
    fit_ordinal_mll,
    make_ordinal_mll as _make_ordinal_mll,
)
from .robust import (
    fit_rrp_binary_classifier_mll,
    fit_rrp_binary_classifier_mll_optimizer,
    fit_rrp_multiclass_classifier_mll,
    fit_rrp_multiclass_classifier_mll_optimizer,
    fit_rrp_multiclass_mll,
    fit_rrp_multiclass_mll_optimizer,
    fit_rrp_ordinal_mll,
    fit_rrp_ordinal_mll_optimizer,
)
from .vae import VAEFitResult, fit_vae_gp


def make_ordinal_mll(model, *, beta: float | None = None, **kwargs):
    """Build an ordinal MLL and optionally set its KL-divergence weight.

    Args:
        model: Ordinal model or wrapper accepted by the underlying factory.
        beta: Optional KL-divergence weight used by variational ordinal MLLs.
        **kwargs: Additional arguments forwarded to the ordinal MLL factory.

    Returns:
        Constructed ordinal marginal log likelihood.
    """

    mll = _make_ordinal_mll(model, **kwargs)
    if beta is not None:
        if not hasattr(mll, "beta"):
            raise TypeError(
                f"{type(mll).__name__} does not support the beta parameter."
            )
        mll.beta = float(beta)
    return mll


__all__ = [
    "ClassificationFitResult",
    "FitResult",
    "VAEFitResult",
    "fit_binary_classifier_mll",
    "fit_classification_gp",
    "fit_classification_mll",
    "fit_multiclass_gp",
    "fit_multiclass_mll",
    "fit_deep_full_batch_mll",
    "fit_deepgp_mll",
    "fit_deepkernel_mll",
    "fit_non_gaussian_gp",
    "fit_non_gaussian_mll",
    "fit_gpytorch_mll_like_botorch",
    "fit_beta_gp",
    "fit_beta_mll",
    "fit_gamma_gp",
    "fit_gamma_mll",
    "fit_poisson_gp",
    "fit_poisson_mll",
    "fit_negative_binomial_gp",
    "fit_negative_binomial_mll",
    "fit_vae_gp",
    "make_ordinal_mll",
    "fit_ordinal_mll",
    "fit_ordinal_gp",
    "fit_rrp_binary_classifier_mll",
    "fit_rrp_binary_classifier_mll_optimizer",
    "fit_rrp_multiclass_mll",
    "fit_rrp_multiclass_mll_optimizer",
    "fit_rrp_multiclass_classifier_mll",
    "fit_rrp_multiclass_classifier_mll_optimizer",
    "fit_rrp_ordinal_mll",
    "fit_rrp_ordinal_mll_optimizer",
]
