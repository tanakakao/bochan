from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior


def categorical_kernel(cat_dims, ord_dims, batch_shape):
    if len(ord_dims) == 0:
        return ScaleKernel(
            CategoricalKernel(
                batch_shape=batch_shape,
                ard_num_dims=len(cat_dims),
                lengthscale_constraint=GreaterThan(1e-6),
            )
        )
    else:
        sum_kernel = ScaleKernel(
            get_covar_module_with_dim_scaled_prior(
                batch_shape=batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            )
            + ScaleKernel(
                CategoricalKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        )

        prod_kernel = ScaleKernel(
            get_covar_module_with_dim_scaled_prior(
                batch_shape=batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            )
            * CategoricalKernel(
                batch_shape=batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=GreaterThan(1e-6),
            )
        )
        return sum_kernel + prod_kernel