from __future__ import annotations

"""
Ordinal 用 Robust / Outlier Relevance Pursuit モデル。

配置想定:
    bochan/models/ordinal/robust/relevance_pursuit.py

命名規則:
    - RobustOrdinal* は label smoothing による一般ロバスト化。
    - OutlierRelevancePursuitOrdinal* は train-point sparse offset による RRP。
    - OutlierRelevancePursuitOrdinal* は train-point sparse offset による RRP。
"""

import copy
from typing import Any, Optional, Sequence

import gpytorch
import torch
from torch import Tensor
from botorch.models.transforms.input import InputTransform

from bochan.models.ordinal.base.models import (
    _BaseOrdinalGPModel,
    _MixedOrdinalLatentGP,
    _OrdinalLatentGP,
    _normalize_dims,
)
from bochan.models.components.robust import (
    RobustOrdinalLogitLikelihood,
    SparseOutlierOrdinalLogitLikelihood,
    TrainInputsAliasMixin,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    check_categorical_columns_unchanged,
    clone_input_transform,
    make_raw_inducing_points,
)


__all__ = [
    "RobustOrdinalLogitLikelihood",
    "SparseOutlierOrdinalLogitLikelihood",
    "RobustOrdinalGPModel",
    "RobustOrdinalMixedGPModel",
    "OutlierRelevancePursuitOrdinalGPModel",
    "OutlierRelevancePursuitOrdinalMixedGPModel",
]


class OutlierRelevancePursuitOrdinalGPModel(
    TrainInputsAliasMixin,
    _BaseOrdinalGPModel,
):
    """
    RRP robust ordinal variational GP.

    This model matches the classification-side RRP design:
        - train-point sparse logit offsets are held by the likelihood
        - RRP selects the support of suspicious / outlier-like training points
        - posterior(X), class_probs(X), and predict_class(X) receive raw X

    Compared with RobustOrdinalGPModel:
        - RobustOrdinalGPModel uses label smoothing only
        - this class uses SparseOutlierOrdinalLogitLikelihood + RRP
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
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
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        local_input_transform = clone_input_transform(input_transform)

        raw_inducing_points = make_raw_inducing_points(
            raw_train_X=raw_train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
        )

        transformed_train_X = apply_input_transform_for_training(
            raw_train_X,
            local_input_transform,
            name="OutlierRelevancePursuitOrdinalGPModel.input_transform",
        )
        transformed_inducing_points = apply_input_transform_for_training(
            raw_inducing_points,
            local_input_transform,
            name="OutlierRelevancePursuitOrdinalGPModel.input_transform",
        )

        latent_model = _OrdinalLatentGP(
            train_X=transformed_train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=transformed_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
        )

        if likelihood is None:
            likelihood = SparseOutlierOrdinalLogitLikelihood(
                dim=raw_train_X.shape[-2],
                outlier_indices=outlier_indices,
                delta_init=delta_init,
                expanded_base_indices=None,
                num_classes=num_classes,
                eps=eps,
                init_gap=init_gap,
                fix_first_cutpoint=fix_first_cutpoint,
                label_smoothing=label_smoothing,
            )
        else:
            if hasattr(likelihood, "set_expanded_base_indices"):
                likelihood.set_expanded_base_indices(None)

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.train_inputs = (transformed_train_X,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y
        
        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets
        
        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = transformed_inducing_points

        self.input_transform = local_input_transform

        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)
        self.delta_init = float(delta_init)

        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.verbose = bool(verbose)

        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

    def transform_inputs(self, X: Tensor) -> Tensor:
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=None,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
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
        **kwargs,
    ) -> "OutlierRelevancePursuitOrdinalGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                "noise is not supported for OutlierRelevancePursuitOrdinalGPModel."
            )

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            num_classes=self.num_classes,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            lr=self.lr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_predictive_log_likelihood=self.use_predictive_log_likelihood,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            verbose=self.verbose,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            label_smoothing=self.label_smoothing,
            outlier_indices=list(self.likelihood.support),
            delta_init=self.delta_init,
            input_transform=clone_input_transform(self.input_transform),
            inducing_points=self.inducing_points_raw.detach().clone(),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
        )

        new_model.load_state_dict(
            copy.deepcopy(self.state_dict()),
            strict=True,
        )

        if refit:
            from .fit import fit_ordinal_gp

            steps = self.conditioning_steps if num_steps is None else int(num_steps)

            refit_lr = self.conditioning_lr if lr is None else float(lr)
            if refit_lr is None:
                refit_lr = self.lr

            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            if refit_bs is None:
                refit_bs = self.batch_size

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


class OutlierRelevancePursuitOrdinalMixedGPModel(
    TrainInputsAliasMixin,
    _BaseOrdinalGPModel,
):
    """
    Mixed-input RRP robust ordinal variational GP.

    This is the mixed ordinal analogue of
    OutlierRelevancePursuitClassificationMixedGPModel.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
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
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        cat_dims = _normalize_dims(cat_dims, raw_train_X.shape[-1])

        if category_counts is None:
            category_counts = self._infer_category_counts(
                X=raw_train_X,
                cat_dims=cat_dims,
            )
        else:
            category_counts = {int(k): int(v) for k, v in category_counts.items()}

        self._validate_categorical_values(
            X=raw_train_X,
            cat_dims=cat_dims,
            category_counts=category_counts,
        )

        raw_inducing_points = make_raw_inducing_points(
            raw_train_X=raw_train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
        )

        self._validate_categorical_values(
            X=raw_inducing_points,
            cat_dims=cat_dims,
            category_counts=category_counts,
        )

        local_input_transform = clone_input_transform(input_transform)

        transformed_train_X = apply_input_transform_for_training(
            raw_train_X,
            local_input_transform,
            cat_dims=cat_dims,
            name="OutlierRelevancePursuitOrdinalMixedGPModel.input_transform",
        )
        transformed_inducing_points = apply_input_transform_for_training(
            raw_inducing_points,
            local_input_transform,
            cat_dims=cat_dims,
            name="OutlierRelevancePursuitOrdinalMixedGPModel.input_transform",
        )

        self._validate_transformed_categorical_values(
            raw_X=raw_train_X,
            transformed_X=transformed_train_X,
            cat_dims=cat_dims,
        )
        self._validate_transformed_categorical_values(
            raw_X=raw_inducing_points,
            transformed_X=transformed_inducing_points,
            cat_dims=cat_dims,
        )

        latent_model = _MixedOrdinalLatentGP(
            train_X=transformed_train_X,
            cat_dims=cat_dims,
            inducing_points_num=inducing_points_num,
            inducing_points=transformed_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_name=cont_kernel,
        )

        if likelihood is None:
            likelihood = SparseOutlierOrdinalLogitLikelihood(
                dim=raw_train_X.shape[-2],
                outlier_indices=outlier_indices,
                delta_init=delta_init,
                expanded_base_indices=None,
                num_classes=num_classes,
                eps=eps,
                init_gap=init_gap,
                fix_first_cutpoint=fix_first_cutpoint,
                label_smoothing=label_smoothing,
            )
        else:
            if hasattr(likelihood, "set_expanded_base_indices"):
                likelihood.set_expanded_base_indices(None)

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.train_inputs = (transformed_train_X,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y

        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets
        
        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = transformed_inducing_points

        self.input_transform = local_input_transform

        self.cat_dims = list(cat_dims)
        self.category_counts = copy.deepcopy(category_counts)
        self.cont_kernel = str(cont_kernel)

        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)
        self.delta_init = float(delta_init)

        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.verbose = bool(verbose)

        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

    @staticmethod
    def _validate_categorical_values(
        X: Tensor,
        cat_dims: Sequence[int],
        category_counts: dict[int, int],
    ) -> None:
        d = X.shape[-1]
        cat_dims = _normalize_dims(cat_dims, d)

        for j in cat_dims:
            if j not in category_counts:
                raise ValueError(f"category_counts must contain key {j}.")

            n_cat = int(category_counts[j])
            vals = X[..., j].reshape(-1)

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )

            if vals.numel() == 0:
                continue

            vals_min = vals.min().item()
            vals_max = vals.max().item()

            if vals_min < 0 or vals_max > n_cat - 1:
                raise ValueError(
                    f"Categorical column {j} must be in [0, {n_cat - 1}], "
                    f"got min={vals_min}, max={vals_max}."
                )

    @staticmethod
    def _validate_transformed_categorical_values(
        raw_X: Tensor,
        transformed_X: Tensor,
        cat_dims: Sequence[int],
    ) -> None:
        check_categorical_columns_unchanged(
            X=raw_X,
            X_tf=transformed_X,
            cat_dims=cat_dims,
        )

    @staticmethod
    def _infer_category_counts(
        X: Tensor,
        cat_dims: Sequence[int],
    ) -> dict[int, int]:
        d = X.shape[-1]
        cat_dims = _normalize_dims(cat_dims, d)

        inferred: dict[int, int] = {}

        for j in cat_dims:
            vals = X[..., j].reshape(-1)

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )

            vals_int = vals.long()

            if vals_int.numel() == 0:
                raise ValueError(f"Categorical column {j} is empty.")

            min_v = int(vals_int.min().item())
            max_v = int(vals_int.max().item())

            if min_v < 0:
                raise ValueError(
                    f"Categorical column {j} must be non-negative, got min={min_v}."
                )

            inferred[j] = max_v + 1

        return inferred

    def transform_inputs(self, X: Tensor) -> Tensor:
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
        )

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = super()._canonicalize_observation_X(X)

        self._validate_categorical_values(
            X=X,
            cat_dims=self.cat_dims,
            category_counts=self.category_counts,
        )

        return X

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
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
        **kwargs,
    ) -> "OutlierRelevancePursuitOrdinalMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                "noise is not supported for OutlierRelevancePursuitOrdinalMixedGPModel."
            )

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            num_classes=self.num_classes,
            cat_dims=self.cat_dims,
            category_counts=copy.deepcopy(self.category_counts),
            cont_kernel=self.cont_kernel,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            lr=self.lr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_predictive_log_likelihood=self.use_predictive_log_likelihood,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            verbose=self.verbose,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            label_smoothing=self.label_smoothing,
            outlier_indices=list(self.likelihood.support),
            delta_init=self.delta_init,
            input_transform=clone_input_transform(self.input_transform),
            inducing_points=self.inducing_points_raw.detach().clone(),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
        )

        new_model.load_state_dict(
            copy.deepcopy(self.state_dict()),
            strict=True,
        )

        if refit:
            from .fit import fit_ordinal_gp

            steps = self.conditioning_steps if num_steps is None else int(num_steps)

            refit_lr = self.conditioning_lr if lr is None else float(lr)
            if refit_lr is None:
                refit_lr = self.lr

            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            if refit_bs is None:
                refit_bs = self.batch_size

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


class RobustOrdinalGPModel(TrainInputsAliasMixin, _BaseOrdinalGPModel):
    """
    config を使わない robust ordinal variational GP.

    Notes:
        - train_inputs_raw[0] は raw X。
        - train_inputs[0] は input_transform 後の X。
        - posterior(X) / class_probs(X) / predict_class(X) は raw X を受け取る。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
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
        label_smoothing: float = 0.05,
        input_transform: Optional[InputTransform] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        local_input_transform = clone_input_transform(input_transform)

        raw_inducing_points = make_raw_inducing_points(
            raw_train_X=raw_train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
        )

        transformed_train_X = apply_input_transform_for_training(
            raw_train_X,
            local_input_transform,
            name="RobustOrdinalGPModel.input_transform",
        )
        transformed_inducing_points = apply_input_transform_for_training(
            raw_inducing_points,
            local_input_transform,
            name="RobustOrdinalGPModel.input_transform",
        )

        latent_model = _OrdinalLatentGP(
            train_X=transformed_train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=transformed_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
        )

        likelihood = RobustOrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            label_smoothing=label_smoothing,
        )

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        # BoTorch-style training data attributes.
        # Do not assign self.train_X / self.train_Y directly.
        self.train_inputs = (transformed_train_X,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y

        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = transformed_inducing_points

        self.input_transform = local_input_transform

        # constructor / rebuild 用
        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)

        # fit 用デフォルト
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.verbose = bool(verbose)

        # condition_on_observations 用
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

    def transform_inputs(self, X: Tensor) -> Tensor:
        """raw 空間の X をモデル内部の入力空間へ写像する。"""
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=None,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
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
        **kwargs,
    ) -> "RobustOrdinalGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for RobustOrdinalGPModel.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            num_classes=self.num_classes,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            lr=self.lr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_predictive_log_likelihood=self.use_predictive_log_likelihood,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            verbose=self.verbose,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            label_smoothing=self.label_smoothing,
            input_transform=clone_input_transform(self.input_transform),
            inducing_points=self.inducing_points_raw.detach().clone(),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
        )

        new_model.load_state_dict(
            copy.deepcopy(self.state_dict()),
            strict=True,
        )

        if refit:
            from .fit import fit_ordinal_gp

            steps = self.conditioning_steps if num_steps is None else int(num_steps)

            refit_lr = self.conditioning_lr if lr is None else float(lr)
            if refit_lr is None:
                refit_lr = self.lr

            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            if refit_bs is None:
                refit_bs = self.batch_size

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


class RobustOrdinalMixedGPModel(TrainInputsAliasMixin, _BaseOrdinalGPModel):
    """
    config を使わない mixed robust ordinal variational GP.

    Notes:
        - cat_dims は integer-coded category column を想定します。
        - input_transform を使う場合も、カテゴリ列は変換しない前提です。
          典型的には Normalize(indices=cont_dims) を使ってください。
        - train_inputs_raw[0] は raw X。
        - train_inputs[0] は input_transform 後の X。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
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
        label_smoothing: float = 0.05,
        input_transform: Optional[InputTransform] = None,
        inducing_points: Optional[Tensor] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
    ) -> None:
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        cat_dims = _normalize_dims(cat_dims, raw_train_X.shape[-1])

        if category_counts is None:
            category_counts = self._infer_category_counts(
                X=raw_train_X,
                cat_dims=cat_dims,
            )
        else:
            category_counts = {int(k): int(v) for k, v in category_counts.items()}

        self._validate_categorical_values(
            X=raw_train_X,
            cat_dims=cat_dims,
            category_counts=category_counts,
        )

        raw_inducing_points = make_raw_inducing_points(
            raw_train_X=raw_train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
        )

        self._validate_categorical_values(
            X=raw_inducing_points,
            cat_dims=cat_dims,
            category_counts=category_counts,
        )

        local_input_transform = clone_input_transform(input_transform)

        transformed_train_X = apply_input_transform_for_training(
            raw_train_X,
            local_input_transform,
            cat_dims=cat_dims,
            name="RobustOrdinalMixedGPModel.input_transform",
        )
        transformed_inducing_points = apply_input_transform_for_training(
            raw_inducing_points,
            local_input_transform,
            cat_dims=cat_dims,
            name="RobustOrdinalMixedGPModel.input_transform",
        )

        self._validate_transformed_categorical_values(
            raw_X=raw_train_X,
            transformed_X=transformed_train_X,
            cat_dims=cat_dims,
        )
        self._validate_transformed_categorical_values(
            raw_X=raw_inducing_points,
            transformed_X=transformed_inducing_points,
            cat_dims=cat_dims,
        )

        latent_model = _MixedOrdinalLatentGP(
            train_X=transformed_train_X,
            cat_dims=cat_dims,
            inducing_points_num=inducing_points_num,
            inducing_points=transformed_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_name=cont_kernel,
        )

        likelihood = RobustOrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            label_smoothing=label_smoothing,
        )

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        # BoTorch-style training data attributes.
        # Do not assign self.train_X / self.train_Y directly.
        self.train_inputs = (transformed_train_X,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y

        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = transformed_inducing_points

        self.input_transform = local_input_transform

        # mixed 固有
        self.cat_dims = list(cat_dims)
        self.category_counts = copy.deepcopy(category_counts)
        self.cont_kernel = str(cont_kernel)

        # constructor / rebuild 用
        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)

        # fit 用デフォルト
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.verbose = bool(verbose)

        # condition_on_observations 用
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

    @staticmethod
    def _validate_categorical_values(
        X: Tensor,
        cat_dims: Sequence[int],
        category_counts: dict[int, int],
    ) -> None:
        d = X.shape[-1]
        cat_dims = _normalize_dims(cat_dims, d)

        for j in cat_dims:
            if j not in category_counts:
                raise ValueError(f"category_counts must contain key {j}.")

            n_cat = int(category_counts[j])
            vals = X[..., j].reshape(-1)

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )

            if vals.numel() == 0:
                continue

            vals_min = vals.min().item()
            vals_max = vals.max().item()

            if vals_min < 0 or vals_max > n_cat - 1:
                raise ValueError(
                    f"Categorical column {j} must be in [0, {n_cat - 1}], "
                    f"got min={vals_min}, max={vals_max}."
                )

    @staticmethod
    def _validate_transformed_categorical_values(
        raw_X: Tensor,
        transformed_X: Tensor,
        cat_dims: Sequence[int],
    ) -> None:
        """input_transform がカテゴリ列を変えていないことを確認する。"""
        check_categorical_columns_unchanged(
            X=raw_X,
            X_tf=transformed_X,
            cat_dims=cat_dims,
        )

    @staticmethod
    def _infer_category_counts(
        X: Tensor,
        cat_dims: Sequence[int],
    ) -> dict[int, int]:
        d = X.shape[-1]
        cat_dims = _normalize_dims(cat_dims, d)

        inferred: dict[int, int] = {}

        for j in cat_dims:
            vals = X[..., j].reshape(-1)

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )

            vals_int = vals.long()

            if vals_int.numel() == 0:
                raise ValueError(f"Categorical column {j} is empty.")

            min_v = int(vals_int.min().item())
            max_v = int(vals_int.max().item())

            if min_v < 0:
                raise ValueError(
                    f"Categorical column {j} must be non-negative, got min={min_v}."
                )

            inferred[j] = max_v + 1

        return inferred

    def transform_inputs(self, X: Tensor) -> Tensor:
        """raw 空間の X をモデル内部の入力空間へ写像する。"""
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
        )

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = super()._canonicalize_observation_X(X)

        self._validate_categorical_values(
            X=X,
            cat_dims=self.cat_dims,
            category_counts=self.category_counts,
        )

        return X

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
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
        **kwargs,
    ) -> "RobustOrdinalMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                "noise is not supported for RobustOrdinalMixedGPModel."
            )

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            num_classes=self.num_classes,
            cat_dims=self.cat_dims,
            category_counts=copy.deepcopy(self.category_counts),
            cont_kernel=self.cont_kernel,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            lr=self.lr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            use_predictive_log_likelihood=self.use_predictive_log_likelihood,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            verbose=self.verbose,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            label_smoothing=self.label_smoothing,
            input_transform=clone_input_transform(self.input_transform),
            inducing_points=self.inducing_points_raw.detach().clone(),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
        )

        new_model.load_state_dict(
            copy.deepcopy(self.state_dict()),
            strict=True,
        )

        if refit:
            from .fit import fit_ordinal_gp

            steps = self.conditioning_steps if num_steps is None else int(num_steps)

            refit_lr = self.conditioning_lr if lr is None else float(lr)
            if refit_lr is None:
                refit_lr = self.lr

            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            if refit_bs is None:
                refit_bs = self.batch_size

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model
