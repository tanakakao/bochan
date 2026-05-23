from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _OneDimensionalLikelihood
from torch import Tensor


class OrdinalLogitLikelihood(_OneDimensionalLikelihood):
    """
    Ordered logit likelihood.

    For latent value f,
        P(y <= k | f) = sigmoid(c_k - f)
    with monotonically increasing cutpoints.
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 3:
            raise ValueError("num_classes must be >= 3")

        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)

        if self.fix_first_cutpoint:
            n_free = self.num_classes - 2
            init_raw = self._inv_softplus(torch.full((n_free,), float(init_gap)))
            self.raw_gaps = nn.Parameter(init_raw)
            self.register_buffer("_fixed_first_cutpoint", torch.tensor(0.0))
        else:
            n_free = self.num_classes - 1
            init_raw = self._inv_softplus(torch.full((n_free,), float(init_gap)))
            self.raw_gaps = nn.Parameter(init_raw)

    @staticmethod
    def _inv_softplus(x: Tensor) -> Tensor:
        return torch.log(torch.expm1(x))

    @property
    def cutpoints(self) -> Tensor:
        gaps = F.softplus(self.raw_gaps) + 1e-6
        if self.fix_first_cutpoint:
            c_rest = torch.cumsum(gaps, dim=0)
            return torch.cat([self._fixed_first_cutpoint.view(1), c_rest], dim=0)
        cuts = torch.cumsum(gaps, dim=0)
        return cuts - cuts.mean()

    def class_probs_from_f(self, f: Tensor) -> Tensor:
        cuts = self.cutpoints.to(device=f.device, dtype=f.dtype)
        cdfs = torch.sigmoid(cuts - f.unsqueeze(-1))
        lower = torch.cat([torch.zeros_like(f).unsqueeze(-1), cdfs], dim=-1)
        upper = torch.cat([cdfs, torch.ones_like(f).unsqueeze(-1)], dim=-1)
        probs = (upper - lower).clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True)

    def expected_utility_from_f(self, f: Tensor, utilities: Tensor) -> Tensor:
        utilities = utilities.to(device=f.device, dtype=f.dtype)
        probs = self.class_probs_from_f(f)
        return (probs * utilities).sum(dim=-1)

    def marginal_class_probs(self, function_dist: MultivariateNormal) -> Tensor:
        probs = []
        for k in range(self.num_classes):
            pk = self.quadrature(
                lambda f, kk=k: self.class_probs_from_f(f)[..., kk],
                function_dist,
            )
            probs.append(pk)
        probs = torch.stack(probs, dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True)

    def marginal_expected_utility(
        self,
        function_dist: MultivariateNormal,
        utilities: Tensor,
    ) -> Tensor:
        utilities = utilities.to(
            device=function_dist.mean.device,
            dtype=function_dist.mean.dtype,
        )
        return self.quadrature(
            lambda f: self.expected_utility_from_f(f, utilities),
            function_dist,
        )

    def forward(self, function_samples: Tensor, *args, **kwargs):
        return torch.distributions.Categorical(
            probs=self.class_probs_from_f(function_samples)
        )

    def probs_from_latent(self, f: Tensor) -> Tensor:
        """
        latent score f から ordinal class probability P(y=k|f) を返す。
    
        Args:
            f: Tensor with shape (...), or (..., 1)
    
        Returns:
            probs: Tensor with shape (..., K)
        """
        if f.shape[-1:] == torch.Size([1]):
            f = f.squeeze(-1)
    
        cutpoints = self.cutpoints
        cutpoints = cutpoints.to(device=f.device, dtype=f.dtype)
    
        # c_j - f
        z = cutpoints.view(*((1,) * f.ndim), -1) - f.unsqueeze(-1)
    
        cdf = torch.sigmoid(z)
    
        p0 = cdf[..., :1]
        pmid = cdf[..., 1:] - cdf[..., :-1]
        plast = 1.0 - cdf[..., -1:]
    
        probs = torch.cat([p0, pmid, plast], dim=-1)
        return probs.clamp_min(self.eps)