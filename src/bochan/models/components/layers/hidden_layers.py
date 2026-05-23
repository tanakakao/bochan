from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ProductKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from botorch.models.kernels.categorical import CategoricalKernel
from botorch.utils.transforms import normalize_indices
from gpytorch.utils.grid import ScaleToBounds

try:
    from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
except Exception:
    get_covar_module_with_dim_scaled_prior = None

try:
    from ..layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor
except Exception:
    try:
        from .feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor
    except Exception:
        LargeFeatureExtractor = None
        SkipLargeFeatureExtractor = None


# ============================================================
# Helpers
# ============================================================


def _reduce_deepgp_tensor(tensor: Tensor, ref_X: Tensor) -> Tensor:
    """Average leading DeepGP sample dimensions until shape matches ref_X[:-1]."""
    expected_ndim = ref_X.ndim - 1
    while tensor.ndim > expected_ndim:
        tensor = tensor.mean(dim=0)
    return tensor



def _select_inducing_points_from_data(
    input_data: Optional[Tensor],
    input_dims: int,
    num_inducing: int,
    output_dims: Optional[int] = None,
) -> Tensor:
    """Choose inducing points from data when available; else initialize randomly."""
    if input_data is not None:
        n = input_data.shape[-2]
        m = min(int(num_inducing), int(n))
        perm = torch.randperm(n, device=input_data.device)[:m]
        base_Z = input_data[perm].clone()
        if output_dims is None:
            return base_Z
        return base_Z.unsqueeze(0).expand(output_dims, m, input_dims).contiguous()

    if output_dims is None:
        return torch.randn(num_inducing, input_dims)
    return torch.randn(output_dims, num_inducing, input_dims)



def _expand_original_input_for_skip(x: Tensor, original_input: Tensor) -> Tensor:
    """Expand original_input to match any leading sample dims of x."""
    original_input = original_input.to(device=x.device, dtype=x.dtype)
    while original_input.ndim < x.ndim:
        original_input = original_input.unsqueeze(0)
    target_shape = x.shape[:-1] + (original_input.shape[-1],)
    return original_input.expand(*target_shape)



def _make_cont_kernel(active_dims: Sequence[int], batch_shape: torch.Size):
    if get_covar_module_with_dim_scaled_prior is not None:
        return get_covar_module_with_dim_scaled_prior(
            batch_shape=batch_shape,
            ard_num_dims=len(active_dims),
            active_dims=tuple(active_dims),
        )
    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=len(active_dims),
            active_dims=tuple(active_dims),
            batch_shape=batch_shape,
        ),
        batch_shape=batch_shape,
    )



def _make_cat_kernel(active_dims: Sequence[int], batch_shape: torch.Size):
    return ScaleKernel(
        CategoricalKernel(
            active_dims=tuple(active_dims),
            ard_num_dims=len(active_dims),
            batch_shape=batch_shape,
            lengthscale_constraint=GreaterThan(1e-6),
        ),
        batch_shape=batch_shape,
    )



def build_mixed_deep_kernel(
    input_dims: int,
    ord_dims: Sequence[int],
    cat_dims: Sequence[int],
    batch_shape: torch.Size,
):
    """Continuous + categorical mixed kernel used by DeepMixedGPHiddenLayer."""
    del input_dims  # kept for API symmetry
    ord_dims = sorted(ord_dims)
    cat_dims = sorted(cat_dims)

    if len(cat_dims) == 0:
        return _make_cont_kernel(ord_dims, batch_shape=batch_shape)
    if len(ord_dims) == 0:
        return _make_cat_kernel(cat_dims, batch_shape=batch_shape)

    cont_kernel_1 = _make_cont_kernel(ord_dims, batch_shape=batch_shape)
    cont_kernel_2 = _make_cont_kernel(ord_dims, batch_shape=batch_shape)
    cat_kernel_1 = _make_cat_kernel(cat_dims, batch_shape=batch_shape)
    cat_kernel_2 = _make_cat_kernel(cat_dims, batch_shape=batch_shape)
    return cont_kernel_1 + cat_kernel_1 + ProductKernel(cont_kernel_2, cat_kernel_2)


# ============================================================
# Basic DeepGP layers
# ============================================================


class DeepGPHiddenLayer(DeepGPLayer):
    """Basic hidden layer for a true DeepGP."""

    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int] = None,
        num_inducing: int = 128,
        mean_type: str = "linear",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.input_dims = int(input_dims)
        self.output_dims = output_dims
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        if inducing_points is None:
            inducing_points = _select_inducing_points_from_data(
                input_data=input_data,
                input_dims=self.input_dims,
                num_inducing=num_inducing,
                output_dims=output_dims,
            )
        if input_data is not None:
            inducing_points = inducing_points.to(input_data)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2],
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        # if output_dims is not None:
        #     variational_strategy = IndependentMultitaskVariationalStrategy(
        #         variational_strategy,
        #         num_tasks=output_dims,
        #     )

        super().__init__(
            variational_strategy=variational_strategy,
            input_dims=input_dims,
            output_dims=output_dims,
        )

        if mean_type == "linear":
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        else:
            self.mean_module = ConstantMean(batch_shape=batch_shape)

        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=input_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class SkipDeepGPHiddenLayer(DeepGPHiddenLayer):
    """Hidden layer with deterministic skip reinjection of the original input."""

    def __init__(
        self,
        base_input_dims: int,
        skip_input_dims: int,
        output_dims: Optional[int] = None,
        num_inducing: int = 128,
        mean_type: str = "linear",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.base_input_dims = int(base_input_dims)
        self.skip_input_dims = int(skip_input_dims)
        super().__init__(
            input_dims=self.base_input_dims + self.skip_input_dims,
            output_dims=output_dims,
            num_inducing=num_inducing,
            mean_type=mean_type,
            input_data=input_data,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

    @staticmethod
    def _feature_tensor(x: Union[Tensor, MultivariateNormal], ref: Tensor) -> Tensor:
        if isinstance(x, MultivariateNormal):
            x = x.mean
        return _reduce_deepgp_tensor(x, ref)

    def __call__(self, x, original_input: Tensor, *args, **kwargs):
        x_feat = self._feature_tensor(x, original_input)
        combined = torch.cat([x_feat, original_input], dim=-1)
        return super().__call__(combined, *args, **kwargs)


class DeepMixedGPHiddenLayer(DeepGPLayer):
    """Mixed-input hidden layer for a true DeepGP."""

    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int],
        ord_dims: Sequence[int],
        cat_dims: Sequence[int],
        num_inducing: int = 128,
        mean_type: str = "linear",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.input_dims = int(input_dims)
        self.output_dims = output_dims
        self.ord_dims = sorted(ord_dims)
        self.cat_dims = sorted(cat_dims)
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        if inducing_points is None:
            inducing_points = _select_inducing_points_from_data(
                input_data=input_data,
                input_dims=self.input_dims,
                num_inducing=num_inducing,
                output_dims=output_dims,
            )
        if input_data is not None:
            inducing_points = inducing_points.to(input_data)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2],
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        # if output_dims is not None:
        #     variational_strategy = IndependentMultitaskVariationalStrategy(
        #         variational_strategy,
        #         num_tasks=output_dims,
        #     )

        super().__init__(
            variational_strategy=variational_strategy,
            input_dims=input_dims,
            output_dims=output_dims,
        )

        if mean_type == "linear":
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        else:
            self.mean_module = ConstantMean(batch_shape=batch_shape)

        self.covar_module = build_mixed_deep_kernel(
            input_dims=input_dims,
            ord_dims=self.ord_dims,
            cat_dims=self.cat_dims,
            batch_shape=batch_shape,
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class SkipDeepMixedGPHiddenLayer(DeepMixedGPHiddenLayer):
    """Mixed hidden layer with deterministic skip reinjection."""

    def __init__(
        self,
        base_input_dims: int,
        skip_input_dims: int,
        original_ord_dims: Sequence[int],
        original_cat_dims: Sequence[int],
        output_dims: Optional[int] = None,
        num_inducing: int = 128,
        mean_type: str = "linear",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.base_input_dims = int(base_input_dims)
        self.skip_input_dims = int(skip_input_dims)
        original_ord_dims = list(normalize_indices(indices=original_ord_dims, d=skip_input_dims))
        original_cat_dims = list(normalize_indices(indices=original_cat_dims, d=skip_input_dims))

        ord_dims = list(range(self.base_input_dims)) + [
            self.base_input_dims + j for j in original_ord_dims
        ]
        cat_dims = [self.base_input_dims + j for j in original_cat_dims]

        super().__init__(
            input_dims=self.base_input_dims + self.skip_input_dims,
            output_dims=output_dims,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=num_inducing,
            mean_type=mean_type,
            input_data=input_data,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

    @staticmethod
    def _feature_tensor(x: Union[Tensor, MultivariateNormal], ref: Tensor) -> Tensor:
        if isinstance(x, MultivariateNormal):
            x = x.mean
        return _reduce_deepgp_tensor(x, ref)

    def __call__(self, x, original_input: Tensor, *args, **kwargs):
        x_feat = self._feature_tensor(x, original_input)
        combined = torch.cat([x_feat, original_input], dim=-1)
        return super().__call__(combined, *args, **kwargs)


# ============================================================
# Deep kernel variants
# ============================================================


class DeepKernelDeepGPHiddenLayer(DeepGPHiddenLayer):
    """Deep kernel version of DeepGPHiddenLayer."""

    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int] = None,
        num_inducing: int = 128,
        mean_type: str = "constant",
        ext_type: str = "DEFAULT",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            num_inducing=num_inducing,
            mean_type=mean_type,
            input_data=input_data,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        if LargeFeatureExtractor is None or SkipLargeFeatureExtractor is None:
            raise ImportError("feature_extractor module could not be imported.")

        if str(ext_type).lower() == "skip":
            self.feature_extractor = SkipLargeFeatureExtractor(
                input_dim=input_dims,
                output_dim=input_dims,
            )
        else:
            self.feature_extractor = LargeFeatureExtractor(
                input_dim=input_dims,
                output_dim=input_dims,
            )
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)

    def forward(self, x: Tensor) -> MultivariateNormal:
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)


class DeepKernelDeepMixedGPHiddenLayer(DeepMixedGPHiddenLayer):
    """Mixed-input deep-kernel hidden layer."""

    def __init__(
        self,
        input_dims: int,
        output_dims: Optional[int],
        ord_dims: Sequence[int],
        cat_dims: Sequence[int],
        num_inducing: int = 128,
        mean_type: str = "constant",
        ext_type: str = "DEFAULT",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=num_inducing,
            mean_type=mean_type,
            input_data=input_data,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        if LargeFeatureExtractor is None or SkipLargeFeatureExtractor is None:
            raise ImportError("feature_extractor module could not be imported.")

        self.cat_dims = list(cat_dims)
        self.ord_dims = list(ord_dims)
        cont_dim = len(self.ord_dims)
        if cont_dim == 0:
            raise ValueError("DeepKernelDeepMixedGPHiddenLayer requires at least one continuous dimension.")

        if str(ext_type).lower() == "skip":
            self.feature_extractor = SkipLargeFeatureExtractor(
                input_dim=cont_dim,
                output_dim=cont_dim,
            )
        else:
            self.feature_extractor = LargeFeatureExtractor(
                input_dim=cont_dim,
                output_dim=cont_dim,
            )
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)

    def forward(self, x: Tensor) -> MultivariateNormal:
        x_cont = x[..., self.ord_dims]
        x_cat = x[..., self.cat_dims]
        projected_cont = self.feature_extractor(x_cont)
        projected_cont = self.scale_to_bounds(projected_cont)

        restored_x = x.clone()
        restored_x[..., self.ord_dims] = projected_cont
        restored_x[..., self.cat_dims] = x_cat

        mean_x = self.mean_module(restored_x)
        covar_x = self.covar_module(restored_x)
        return MultivariateNormal(mean_x, covar_x)


class SkipDeepKernelDeepGPHiddenLayer(DeepKernelDeepGPHiddenLayer):
    """Deep-kernel layer with skip reinjection for continuous inputs."""

    def __init__(
        self,
        base_input_dims: int,
        skip_input_dims: int,
        output_dims: Optional[int] = None,
        num_inducing: int = 128,
        mean_type: str = "constant",
        ext_type: str = "DEFAULT",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.base_input_dims = int(base_input_dims)
        self.skip_input_dims = int(skip_input_dims)
        super().__init__(
            input_dims=self.base_input_dims + self.skip_input_dims,
            output_dims=output_dims,
            num_inducing=num_inducing,
            mean_type=mean_type,
            ext_type=ext_type,
            input_data=input_data,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

    def forward(self, x: Tensor, original_input: Optional[Tensor] = None) -> MultivariateNormal:
        if original_input is None:
            raise ValueError("SkipDeepKernelDeepGPHiddenLayer requires original_input.")
        if original_input.dim() < x.dim():
            original_input = _expand_original_input_for_skip(x, original_input)
        x = torch.cat([x, original_input], dim=-1)
        return super().forward(x)


class SkipDeepKernelDeepMixedGPHiddenLayer(DeepKernelDeepMixedGPHiddenLayer):
    """Deep-kernel mixed layer with skip reinjection."""

    def __init__(
        self,
        base_input_dims: int,
        skip_input_dims: int,
        original_ord_dims: Sequence[int],
        original_cat_dims: Sequence[int],
        output_dims: Optional[int] = None,
        num_inducing: int = 128,
        mean_type: str = "constant",
        ext_type: str = "DEFAULT",
        input_data: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.base_input_dims = int(base_input_dims)
        self.skip_input_dims = int(skip_input_dims)
        original_ord_dims = list(normalize_indices(indices=original_ord_dims, d=skip_input_dims))
        original_cat_dims = list(normalize_indices(indices=original_cat_dims, d=skip_input_dims))

        ord_dims = list(range(self.base_input_dims)) + [
            self.base_input_dims + j for j in original_ord_dims
        ]
        cat_dims = [self.base_input_dims + j for j in original_cat_dims]

        super().__init__(
            input_dims=self.base_input_dims + self.skip_input_dims,
            output_dims=output_dims,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=num_inducing,
            mean_type=mean_type,
            ext_type=ext_type,
            input_data=input_data,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

    def forward(self, x: Tensor, original_input: Optional[Tensor] = None) -> MultivariateNormal:
        if original_input is None:
            raise ValueError("SkipDeepKernelDeepMixedGPHiddenLayer requires original_input.")
        if original_input.dim() < x.dim():
            original_input = _expand_original_input_for_skip(x, original_input)
        x = torch.cat([x, original_input], dim=-1)
        return super().forward(x)


__all__ = [
    "build_mixed_deep_kernel",
    "DeepGPHiddenLayer",
    "SkipDeepGPHiddenLayer",
    "DeepMixedGPHiddenLayer",
    "SkipDeepMixedGPHiddenLayer",
    "DeepKernelDeepGPHiddenLayer",
    "DeepKernelDeepMixedGPHiddenLayer",
    "SkipDeepKernelDeepGPHiddenLayer",
    "SkipDeepKernelDeepMixedGPHiddenLayer",
]
