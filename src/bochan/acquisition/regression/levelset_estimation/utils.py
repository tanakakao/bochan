from __future__ import annotations

import torch
from torch import Tensor


def contour_uncertainty(model, X: Tensor, h: float, eps: float = 1e-9) -> Tensor:
    """Compute simple regression contour uncertainty.

    Args:
        model: BoTorch-compatible regression model.
        X: Candidate tensor.
        h: Level-set threshold.
        eps: Numerical stability.

    Returns:
        Tensor: Pointwise score with shape `batch_shape x q`.
    """
    posterior = model.posterior(X)
    mean = posterior.mean.squeeze(-1)
    std = posterior.variance.clamp_min(eps).sqrt().squeeze(-1)
    return torch.exp(-0.5 * ((mean - float(h)) / std.clamp_min(eps)).pow(2))
