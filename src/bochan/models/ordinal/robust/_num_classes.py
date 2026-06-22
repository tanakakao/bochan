from __future__ import annotations

"""Robust ordinal models with ``num_classes`` inference support.

The underlying robust implementations predate the common ordinal convention that
``num_classes=None`` delegates class-count inference to ``train_Y``.  These thin
public wrappers preserve the existing implementations while aligning their
constructor APIs with the base, PCA, REMBO, and mixed ordinal models.
"""

from typing import Optional, Sequence

import gpytorch
from botorch.models.transforms.input import InputTransform
from torch import Tensor

from bochan.models.components.robust import SparseOutlierOrdinalLogitLikelihood
from bochan.models.ordinal.base.models import (
    _BaseOrdinalGPModel,
    _infer_num_classes_from_train_Y,
)

from .heteroscedastic import (
    HeteroscedasticOrdinalGPModel as _HeteroscedasticOrdinalGPModel,
)
from .heteroscedastic import (
    HeteroscedasticOrdinalMixedGPModel as _HeteroscedasticOrdinalMixedGPModel,
)
from .relevance_pursuit import (
    OutlierRelevancePursuitOrdinalGPModel as _OutlierRelevancePursuitOrdinalGPModel,
)
from .relevance_pursuit import (
    OutlierRelevancePursuitOrdinalMixedGPModel as _OutlierRelevancePursuitOrdinalMixedGPModel,
)


__all__ = [
    "OutlierRelevancePursuitOrdinalGPModel",
    "OutlierRelevancePursuitOrdinalMixedGPModel",
    "HeteroscedasticOrdinalGPModel",
    "HeteroscedasticOrdinalMixedGPModel",
]


def _resolve_num_classes(
    *,
    train_X: Tensor,
    train_Y: Tensor,
    num_classes: Optional[int],
) -> int:
    """Resolve an explicit class count or infer it from canonical ordinal labels."""
    raw_train_X = _BaseOrdinalGPModel._canonicalize_train_X(train_X)
    canonical_train_Y = _BaseOrdinalGPModel._canonicalize_train_Y(
        train_Y,
        n=raw_train_X.shape[-2],
        device=raw_train_X.device,
    )
    if num_classes is None:
        return _infer_num_classes_from_train_Y(canonical_train_Y)
    return int(num_classes)


class OutlierRelevancePursuitOrdinalGPModel(
    _OutlierRelevancePursuitOrdinalGPModel,
):
    """RRP ordinal GP that infers ``num_classes`` from ``train_Y`` when omitted."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        likelihood: Optional[SparseOutlierOrdinalLogitLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        resolved_num_classes = _resolve_num_classes(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=resolved_num_classes,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_predictive_log_likelihood=use_predictive_log_likelihood,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
            label_smoothing=label_smoothing,
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            likelihood=likelihood,
            input_transform=input_transform,
            inducing_points=inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )


class OutlierRelevancePursuitOrdinalMixedGPModel(
    _OutlierRelevancePursuitOrdinalMixedGPModel,
):
    """Mixed RRP ordinal GP with optional ``num_classes`` inference."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        cat_dims: Sequence[int] = (),
        category_counts: Optional[dict[int, int]] = None,
        cont_kernel: str = "matern52",
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        likelihood: Optional[SparseOutlierOrdinalLogitLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        resolved_num_classes = _resolve_num_classes(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=resolved_num_classes,
            cat_dims=cat_dims,
            category_counts=category_counts,
            cont_kernel=cont_kernel,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_predictive_log_likelihood=use_predictive_log_likelihood,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
            label_smoothing=label_smoothing,
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            likelihood=likelihood,
            input_transform=input_transform,
            inducing_points=inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )


class HeteroscedasticOrdinalGPModel(_HeteroscedasticOrdinalGPModel):
    """Heteroscedastic ordinal GP with optional ``num_classes`` inference."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 200,
        aux_batch_size: Optional[int] = None,
        min_noise: float = 1e-6,
        input_transform: Optional[InputTransform] = None,
        train_Yvar: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        resolved_num_classes = _resolve_num_classes(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=resolved_num_classes,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_predictive_log_likelihood=use_predictive_log_likelihood,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
            aux_lr=aux_lr,
            aux_num_epochs=aux_num_epochs,
            aux_batch_size=aux_batch_size,
            min_noise=min_noise,
            input_transform=input_transform,
            train_Yvar=train_Yvar,
            inducing_points=inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )


class HeteroscedasticOrdinalMixedGPModel(
    _HeteroscedasticOrdinalMixedGPModel,
):
    """Mixed heteroscedastic ordinal GP with optional class-count inference."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        cat_dims: Sequence[int] = (),
        category_counts: Optional[dict[int, int]] = None,
        category_values: Optional[dict[int, Sequence[int | float]]] = None,
        cont_kernel: str = "matern52",
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 200,
        aux_batch_size: Optional[int] = None,
        min_noise: float = 1e-6,
        input_transform: Optional[InputTransform] = None,
        train_Yvar: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        resolved_num_classes = _resolve_num_classes(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=resolved_num_classes,
            cat_dims=cat_dims,
            category_counts=category_counts,
            category_values=category_values,
            cont_kernel=cont_kernel,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_predictive_log_likelihood=use_predictive_log_likelihood,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
            aux_lr=aux_lr,
            aux_num_epochs=aux_num_epochs,
            aux_batch_size=aux_batch_size,
            min_noise=min_noise,
            input_transform=input_transform,
            train_Yvar=train_Yvar,
            inducing_points=inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
        )
