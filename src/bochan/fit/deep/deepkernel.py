from __future__ import annotations

from .common import fit_deep_full_batch_mll


_DEEPKERNEL_PSD_JITTER_VALUES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def fit_deepkernel_mll(mll, **kwargs):
    """Fit a Deep Kernel wrapper model with bounded stability safeguards.

    The feature extractor and exact GP are optimized jointly. Small or
    duplicated cross-validation folds can therefore create ill-conditioned
    covariance matrices transiently. Deep Kernel fitting enables gradient
    clipping and retries only failed Cholesky factorizations with an ascending,
    bounded jitter schedule.
    """
    kwargs.setdefault("clip_grad_norm", 10.0)
    kwargs.setdefault("psd_jitter_values", _DEEPKERNEL_PSD_JITTER_VALUES)
    return fit_deep_full_batch_mll(
        mll,
        log_prefix="fit_deepkernel_mll",
        **kwargs,
    )
