from __future__ import annotations

from dataclasses import dataclass
import warnings
from itertools import product
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Any

import torch

@dataclass
class PCAConfig:
    n_components: int
    standardize: bool = True
    eps: float = 1e-8


@dataclass
class REMBOConfig:
    n_components: int
    seed: int | None = None
    normalize: bool = True
    eps: float = 1e-8
    projection_matrix: torch.Tensor | None = None


class PCATransformer:
    def __init__(self, config: PCAConfig):
        self.config = config
        self.mean_: torch.Tensor | None = None
        self.scale_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "PCATransformer":
        if x.dim() != 2:
            raise ValueError("x must be 2D tensor [n, d].")
        if self.config.n_components > x.shape[-1]:
            raise ValueError("n_components must be <= input dimension.")

        self.mean_ = x.mean(dim=0, keepdim=True)
        xc = x - self.mean_
        if self.config.standardize:
            self.scale_ = xc.std(dim=0, keepdim=True).clamp_min(self.config.eps)
            xc = xc / self.scale_
        else:
            self.scale_ = torch.ones_like(self.mean_)

        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        self.components_ = vh[: self.config.n_components].T
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        x_flat, lead_shape = self._flatten_last_dim(x)
        z_flat = ((x_flat - self.mean_) / self.scale_) @ self.components_
        return z_flat.reshape(*lead_shape, self.components_.shape[-1])

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        z_flat, lead_shape = self._flatten_last_dim(z)
        x_flat = (z_flat @ self.components_.T) * self.scale_ + self.mean_
        return x_flat.reshape(*lead_shape, self.components_.shape[0])

    def _flatten_last_dim(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] <= 0:
            raise ValueError("Last dimension must be positive.")
        lead_shape = x.shape[:-1]
        return x.reshape(-1, x.shape[-1]), lead_shape

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("PCATransformer is not fitted yet.")


class REMBOTransformer:
    def __init__(self, config: REMBOConfig):
        self.config = config
        self.mean_: torch.Tensor | None = None
        self.scale_: torch.Tensor | None = None
        self.projection_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "REMBOTransformer":
        if x.dim() != 2:
            raise ValueError("x must be 2D tensor [n, d].")
        d = x.shape[-1]
        if self.config.n_components > d:
            raise ValueError("n_components must be <= input dimension.")

        self.mean_ = x.mean(dim=0, keepdim=True)
        xc = x - self.mean_
        if self.config.normalize:
            self.scale_ = xc.std(dim=0, keepdim=True).clamp_min(self.config.eps)
        else:
            self.scale_ = torch.ones_like(self.mean_)

        proj = self.config.projection_matrix
        if proj is None:
            gen = None
            if self.config.seed is not None:
                gen = torch.Generator(device=x.device)
                gen.manual_seed(self.config.seed)
            proj = torch.randn(d, self.config.n_components, dtype=x.dtype, device=x.device, generator=gen)
        if proj.shape != (d, self.config.n_components):
            raise ValueError(
                f"projection_matrix must have shape {(d, self.config.n_components)}, got {tuple(proj.shape)}."
            )
        self.projection_ = proj / proj.norm(dim=0, keepdim=True).clamp_min(self.config.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        x_flat, lead_shape = self._flatten_last_dim(x)
        z_flat = ((x_flat - self.mean_) / self.scale_) @ self.projection_
        return z_flat.reshape(*lead_shape, self.projection_.shape[-1])

    def _flatten_last_dim(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] <= 0:
            raise ValueError("Last dimension must be positive.")
        lead_shape = x.shape[:-1]
        return x.reshape(-1, x.shape[-1]), lead_shape

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.projection_ is None:
            raise RuntimeError("REMBOTransformer is not fitted yet.")