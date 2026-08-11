from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from torch import Tensor

from bochan.fit.ordinal import fit_ordinal_gp
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.components.kronecker_multitask import (
    canonicalize_block_design_targets,
)
from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    get_continuous_dims,
    normalize_mixed_dims,
    transform_mixed_inputs,
    validate_mixed_input_transform_for_training,
)

from .kronecker_multitask import KroneckerMultiTaskOrdinalGPModel


class KroneckerMultiTaskOrdinalMixedGPModel(KroneckerMultiTaskOrdinalGPModel):
    r"""Mixed-input block-design ordinal model with ICM task covariance.

    All tasks share the same ordinal class definition and cutpoints. Continuous
    and categorical input effects are represented by an additive-plus-interaction
    mixed kernel, while task dependence is represented by the parent ICM model.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        *,
        rank: Optional[int] = None,
        likelihood: Optional[OrdinalLogitLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        num_inducing: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
    ) -> None:
        raw_train_X = torch.as_tensor(train_X).contiguous()
        normalized_cat_dims = normalize_mixed_dims(
            cat_dims,
            raw_train_X.shape[-1],
        )
        validate_mixed_input_transform_for_training(
            raw_train_X,
            input_transform,
            cat_dims=normalized_cat_dims,
        )
        if data_covar_module is None:
            data_covar_module = build_mixed_kronecker_kernel(
                d=raw_train_X.shape[-1],
                cat_dims=normalized_cat_dims,
            )

        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )
        super().__init__(
            train_X=raw_train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            rank=rank,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )
        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )

    def transform_inputs(self, X: Tensor) -> Tensor:
        return transform_mixed_inputs(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
        )

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "KroneckerMultiTaskOrdinalMixedGPModel":
        """Append complete mixed-input block-design observations."""
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                f"noise is not supported for {self.__class__.__name__}."
            )

        X = self._canonicalize_observation_X(X)
        Y = canonicalize_block_design_targets(X, Y, target_dtype=torch.long)
        if Y.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Y must contain {self.num_tasks} tasks, got {Y.shape[-1]}."
            )
        self._validate_ordinal_targets(Y, num_classes=self.num_classes)

        train_X_full = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        train_Y_full = torch.cat([self.train_targets, Y], dim=-2)

        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            cat_dims=self.cat_dims,
            num_classes=self.num_classes,
            rank=self.rank,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            num_inducing=self.num_inducing,
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            eps=self.eps,
            init_gap=self.init_gap,
            fix_first_cutpoint=self.fix_first_cutpoint,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            refit_batch_size = (
                self.conditioning_batch_size if batch_size is None else batch_size
            )
            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_batch_size,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()
        return new_model


__all__ = ["KroneckerMultiTaskOrdinalMixedGPModel"]
