from __future__ import annotations

"""
Classification 用 Robust / Outlier Relevance Pursuit モデル。

配置想定:
    bochan/models/classification/robust/relevance_pursuit.py

命名規則:
    - classification / ordinal では feature RRP ではなく train-point outlier RRP として扱う。
    - 主名は OutlierRelevancePursuit* とする。
"""

import copy
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls import VariationalELBO

from bochan.models.classification.binary.base.models import _LatentBinarySVGP, _LatentMixedBinarySVGP
from bochan.posteriors.bernoulli import SimpleBernoulliPosterior
from bochan.models.components.robust import (
    SparseOutlierBernoulliLikelihood,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    check_categorical_columns_unchanged,
    concat_optional_noise,
    flatten_targets,
    make_augmented_targets_and_base_indices,
    prepare_wrapper_conditioning_data,
)


__all__ = [
    "SparseOutlierBernoulliLikelihood",
    "OutlierRelevancePursuitBinaryClassificationGPModel",
    "OutlierRelevancePursuitBinaryClassificationMixedGPModel",
]


class _OutlierRRPBinaryClassificationBase(ApproximateGPyTorchModel):
    """
    train-point outlier RRP 用 binary classification wrapper。

    Public convention:
        train_inputs_raw[0]: raw-space の元訓練 X
        train_inputs[0]: 外側 wrapper が受ける raw-space X
        fit_train_inputs[0]: 実際に latent model の学習に使う X
        fit_train_targets: fit_train_inputs に対応する target
    """

    def __init__(
        self,
        latent_model,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[SparseOutlierBernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        fit_X: Optional[Tensor] = None,
        fit_Y: Optional[Tensor] = None,
        expanded_base_indices: Optional[Tensor] = None,
    ) -> None:
        Y_raw = flatten_targets(train_Y, dtype=train_X.dtype).to(train_X.device)

        if fit_X is None:
            fit_X = train_X
        if fit_Y is None:
            fit_Y = Y_raw.to(dtype=fit_X.dtype, device=fit_X.device)
        else:
            fit_Y = fit_Y.to(dtype=fit_X.dtype, device=fit_X.device)

        if likelihood is None:
            likelihood = SparseOutlierBernoulliLikelihood(
                dim=train_X.shape[-2],
                expanded_base_indices=expanded_base_indices,
            )
        elif hasattr(likelihood, "set_expanded_base_indices"):
            likelihood.set_expanded_base_indices(expanded_base_indices)

        super().__init__(model=latent_model, likelihood=likelihood, num_outputs=1)

        self.input_transform = input_transform
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self._train_targets = Y_raw
        self.train_Yvar = train_Yvar

        self.fit_train_inputs = (fit_X,)
        self.fit_train_targets = fit_Y
        self.model.train_inputs = (fit_X,)
        self.model.train_targets = fit_Y

    def _set_transformed_inputs(self) -> None:
        """BoTorch eval 時の自動 transformed input 更新を無効化する。"""
        return None

    @property
    def train_targets(self) -> Tensor:
        return self._train_targets

    @property
    def train_input_raw(self) -> Tensor:
        return self.train_inputs_raw[0]

    @property
    def fit_train_input(self) -> Tensor:
        return self.fit_train_inputs[0]

    def transform_inputs(self, X: Tensor) -> Tensor:
        """raw X を latent model の評価空間へ写像する。"""
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=getattr(self, "cat_dims", None),
        )

    def latent_posterior(
        self,
        X: Tensor,
        output_indices=None,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """latent f の posterior を返す。"""
        _ = kwargs
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.latent_posterior does not support output_indices."
            )
        self.eval()
        X_tf = self.transform_inputs(X)
        posterior = GPyTorchPosterior(self.model(X_tf))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        """p(y=1|x) の posterior を返す。"""
        _ = observation_noise, kwargs
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )
        if posterior_transform is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support posterior_transform."
            )

        self.eval()
        X_tf = self.transform_inputs(X)
        latent_dist = self.model(X_tf)
        pred_dist = self.likelihood(latent_dist)
        probs = pred_dist.probs
        if probs.ndim == X_tf.ndim - 1:
            probs = probs.unsqueeze(-1)
        return SimpleBernoulliPosterior(probs=probs)

    def probability_posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        """classification acquisition 用に probability posterior を明示名で返す。"""
        return self.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def predict_proba(self, X: Tensor) -> Tensor:
        return self.posterior(X).mean

    def posterior_latent(self, X: Tensor, **kwargs: Any) -> GPyTorchPosterior:
        return self.latent_posterior(X, **kwargs)

    def posterior_f(self, X: Tensor, **kwargs: Any) -> GPyTorchPosterior:
        return self.latent_posterior(X, **kwargs)

    def make_mll(self, beta: float = 1.0) -> VariationalELBO:
        """この wrapper 用の VariationalELBO を返す。"""
        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.fit_train_input.shape[-2],
            beta=beta,
        )


class OutlierRelevancePursuitBinaryClassificationGPModel(_OutlierRRPBinaryClassificationBase):
    """
    連続入力用 train-point outlier RRP binary classification GP。

    Notes:
        - 学習点ごとの sparse logit offset を likelihood が持つ。
        - 予測時には offset は使わない。
        - ``input_transform`` は wrapper 側で管理する。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        likelihood: Optional[SparseOutlierBernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module=None,
        covar_module=None,
        learn_inducing_locations: bool = True,
    ) -> None:
        X_fit = apply_input_transform_for_training(
            train_X,
            input_transform,
            name="OutlierRelevancePursuitClassificationGPModel.input_transform",
        )
        Y_fit, base_indices = make_augmented_targets_and_base_indices(
            train_Y=train_Y,
            X_aug=X_fit,
            n_base=train_X.shape[-2],
        )

        if inducing_points is None:
            Z = X_fit
            self.inducing_points_raw = train_X.detach().clone()
        else:
            self.inducing_points_raw = inducing_points.detach().clone()
            Z = apply_input_transform_for_training(
                inducing_points,
                input_transform,
                name="OutlierRelevancePursuitClassificationGPModel.input_transform",
            )

        latent_model = _LatentBinarySVGP(
            inducing_points=Z,
            train_inputs=X_fit,
            train_targets=Y_fit,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(
            latent_model=latent_model,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=input_transform,
            fit_X=X_fit,
            fit_Y=Y_fit,
            expanded_base_indices=base_indices,
        )

        self.learn_inducing_locations = bool(learn_inducing_locations)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "OutlierRelevancePursuitBinaryClassificationGPModel":
        _ = kwargs
        X_new, Y_new, Yvar_new = prepare_wrapper_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.train_input_raw.shape[-1],
        )
        X_old = self.train_inputs_raw[0]
        Y_old = self.train_targets

        X_full = torch.cat([X_old, X_new.to(X_old)], dim=0)
        Y_full = torch.cat([Y_old, Y_new.to(dtype=Y_old.dtype, device=Y_old.device)], dim=0)
        Yvar_full = concat_optional_noise(
            old_Y=Y_old,
            old_Yvar=self.train_Yvar,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=X_old.dtype,
            device=X_old.device,
        )

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            inducing_points=self.inducing_points_raw.detach().clone(),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            learn_inducing_locations=self.learn_inducing_locations,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


class OutlierRelevancePursuitBinaryClassificationMixedGPModel(_OutlierRRPBinaryClassificationBase):
    """mixed 入力用 train-point outlier RRP binary classification GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        inducing_points: Optional[Tensor] = None,
        likelihood: Optional[SparseOutlierBernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module=None,
        covar_module=None,
        cont_kernel_factory: Optional[Callable[[torch.Size, int, Optional[list[int]]], Any]] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.cat_dims = [int(i) for i in cat_dims]
        self.cont_kernel_factory = cont_kernel_factory

        X_fit = apply_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=self.cat_dims,
            name="OutlierRelevancePursuitClassificationMixedGPModel.input_transform",
        )
        check_categorical_columns_unchanged(
            X=train_X,
            X_tf=X_fit,
            cat_dims=self.cat_dims,
            name="OutlierRelevancePursuitClassificationMixedGPModel.input_transform",
        )
        Y_fit, base_indices = make_augmented_targets_and_base_indices(
            train_Y=train_Y,
            X_aug=X_fit,
            n_base=train_X.shape[-2],
        )

        if inducing_points is None:
            Z = X_fit
            self.inducing_points_raw = train_X.detach().clone()
        else:
            self.inducing_points_raw = inducing_points.detach().clone()
            Z = apply_input_transform_for_training(
                inducing_points,
                input_transform,
                cat_dims=self.cat_dims,
                name="OutlierRelevancePursuitClassificationMixedGPModel.input_transform",
            )

        latent_model = _LatentMixedBinarySVGP(
            inducing_points=Z,
            cat_dims=self.cat_dims,
            train_inputs=X_fit,
            train_targets=Y_fit,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(
            latent_model=latent_model,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=input_transform,
            fit_X=X_fit,
            fit_Y=Y_fit,
            expanded_base_indices=base_indices,
        )

        self.learn_inducing_locations = bool(learn_inducing_locations)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "OutlierRelevancePursuitBinaryClassificationMixedGPModel":
        _ = kwargs
        X_new, Y_new, Yvar_new = prepare_wrapper_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.train_input_raw.shape[-1],
        )
        X_old = self.train_inputs_raw[0]
        Y_old = self.train_targets

        X_full = torch.cat([X_old, X_new.to(X_old)], dim=0)
        Y_full = torch.cat([Y_old, Y_new.to(dtype=Y_old.dtype, device=Y_old.device)], dim=0)
        Yvar_full = concat_optional_noise(
            old_Y=Y_old,
            old_Yvar=self.train_Yvar,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=X_old.dtype,
            device=X_old.device,
        )

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            cat_dims=list(self.cat_dims),
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            inducing_points=self.inducing_points_raw.detach().clone(),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            cont_kernel_factory=self.cont_kernel_factory,
            learn_inducing_locations=self.learn_inducing_locations,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model
