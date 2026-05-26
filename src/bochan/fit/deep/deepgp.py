from __future__ import annotations

from .common import fit_deep_full_batch_mll


def fit_deepgp_mll(mll, **kwargs):
    """
    Fit a DeepGP approximate MLL.

    This is a thin wrapper around `fit_deep_full_batch_mll` that preserves the
    previous public function name and logging prefix.
    """
    return fit_deep_full_batch_mll(
        mll,
        log_prefix="fit_deepgp_mll",
        **kwargs,
    )
