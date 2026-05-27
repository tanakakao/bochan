from __future__ import annotations

"""classification 用 MAP-SAAS GP モデル。

連続入力版は既存の ``BinaryClassificationGPModel`` に SAAS prior 付き Matern
kernel を渡す。mixed 入力版はカテゴリ列を one-hot encode し、内部 classification
GP には encoded-space の入力を渡す。
"""

from copy import deepcopy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import GPyTorchPosterior
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from gpytorch.mlls.variational_elbo import VariationalELBO

from bochan.models.classification.binary.base import BinaryClassificationGPModel
from bochan.posteriors.bernoulli import SimpleBernoulliPosterior
from bochan.models.components.saas import (
    OneHotEncodingMixin,
    build_map_saas_covar_module,
    concat_optional_noise,
    flatten_targets,
    prepare_mixed_conditioning_data,
)


class SaasBinaryClassificationGPModel(BinaryClassificationGPModel):
    """MAP-SAAS style の 2 値分類 GP。

    Notes:
        - 実体は variational classification GP。
        - SAAS は fully Bayesian NUTS ではなく MAP prior として使う。
        - ``make_mll()`` は inner latent SVGP 用の ``VariationalELBO`` を返す。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
    ) -> None:
        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_X_raw = train_X.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()

        if covar_module is None:
            covar_module = build_map_saas_covar_module(
                train_X=train_X,
                input_transform=input_transform,
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_X_raw = train_X.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()

    def make_mll(self, beta: float = 1.0) -> VariationalELBO:
        """inner latent SVGP 用の ``VariationalELBO`` を返す。"""
        inner_train_X = self.model.train_inputs[0]
        inner_train_Y = self.model.train_targets

        if inner_train_X.shape[-2] != inner_train_Y.shape[0]:
            raise RuntimeError(
                "inner train_inputs and train_targets have inconsistent data sizes. "
                f"inner_train_X.shape={tuple(inner_train_X.shape)}, "
                f"inner_train_Y.shape={tuple(inner_train_Y.shape)}."
            )

        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=inner_train_X.shape[-2],
            beta=float(beta),
        )

    def probability_posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        """classification acquisition 用に probability posterior を明示名で返す。"""
        return self.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )


class SaasBinaryClassificationMixedGPModel(OneHotEncodingMixin, SaasBinaryClassificationGPModel):
    """mixed 入力向け MAP-SAAS 2 値分類 GP。

    Notes:
        - public API は raw-space X を受け取る。
        - 内部 GP は encoded-space X で学習される。
        - ``train_inputs_raw`` は raw X、``encoded_train_inputs_raw`` は encoded X。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Optional[Sequence[int]] = None,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device, dtype=train_X.dtype)

        self.train_X_raw = train_X.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()

        encoded_train_X = self._init_one_hot_encoding(train_X=train_X, cat_dims=cat_dims)
        self.encoded_train_inputs_raw = (encoded_train_X.detach().clone(),)
        expanded_input_transform = self._maybe_expand_input_transform(input_transform)
        encoded_inducing_points = self._canonicalize_inducing_points_for_encoded_space(inducing_points)

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=expanded_input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=encoded_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            tau=tau,
            saas_log_scale=saas_log_scale,
            saas_nu=saas_nu,
        )

        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_X_raw = train_X.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.encoded_train_inputs_raw = (encoded_train_X.detach().clone(),)
        self.encoded_train_inputs = (encoded_train_X.detach().clone(),)

        # public 側は raw-space を保持する。
        self.train_inputs = (train_X.detach().clone(),)
        self.train_targets = flatten_targets(train_Y, dtype=train_X.dtype)
        self.encoded_inducing_points_raw = encoded_inducing_points

    @property
    def encoded_train_input_raw(self) -> Tensor:
        return self.encoded_train_inputs_raw[0]

    def make_mll(self, *args: Any, **kwargs: Any):
        """inner encoded-space model 用の MLL を返す。"""
        return super().make_mll(*args, **kwargs)

    def _set_transformed_inputs(self) -> None:
        """BoTorch eval 時の自動 transformed input 更新を無効化する。"""
        return None

    def _get_input_transform_for_eval(self, input_transform: Any | None = None):
        """明示指定または self.input_transform を返す。"""
        return input_transform if input_transform is not None else getattr(self, "input_transform", None)

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Any | None = None,
    ) -> Tensor:
        """raw/encoded X を内部 GP 用の encoded feature space に変換する。"""
        if isinstance(X, tuple):
            X = X[0]

        tf = self._get_input_transform_for_eval(input_transform)

        raw_dim = int(getattr(self, "raw_dim", self.train_X_raw.shape[-1]))
        encoded_dim = int(getattr(self, "encoded_dim", self.encoded_train_input_raw.shape[-1]))

        if X.shape[-1] == raw_dim:
            if callable(tf):
                try:
                    X_tf = tf(X)
                    if isinstance(X_tf, tuple):
                        X_tf = X_tf[0]

                    if X_tf.shape[-1] == raw_dim:
                        return self._to_encoded_feature_space(X_tf).contiguous()

                    if X_tf.shape[-1] == encoded_dim:
                        return X_tf.contiguous()
                except Exception:
                    # encoded-space transform 互換の fallback へ進む。
                    pass

            return self._to_encoded_feature_space(X).contiguous()

        if X.shape[-1] == encoded_dim:
            if callable(tf):
                try:
                    X_tf = tf(X)
                    if isinstance(X_tf, tuple):
                        X_tf = X_tf[0]
                    if X_tf.shape[-1] == encoded_dim:
                        return X_tf.contiguous()
                except Exception:
                    pass
            return X.contiguous()

        raise RuntimeError(
            "Unexpected input dimension for mixed SAAS classification model. "
            f"X.shape={tuple(X.shape)}, raw_dim={raw_dim}, encoded_dim={encoded_dim}."
        )

    def _to_training_feature_space(self, X: Tensor) -> Tensor:
        """内部 GP が使う encoded feature space へ変換する。"""
        return self.transform_inputs(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        """raw-space X に対する probability posterior を返す。"""
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )
        _ = observation_noise, kwargs

        if isinstance(X, tuple):
            X = X[0]

        self.eval()
        self.likelihood.eval()

        X_eval = self.transform_inputs(X)
        latent_dist = self.model(X_eval)
        pred_dist = self.likelihood(latent_dist)

        p = pred_dist.mean
        var = pred_dist.variance

        if p.ndim == X_eval.ndim - 1:
            p = p.unsqueeze(-1)
            var = var.unsqueeze(-1)

        posterior = SimpleBernoulliPosterior(mean=p, variance=var)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def probability_posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        """classification acquisition 用に probability posterior を明示名で返す。"""
        return self.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def latent_posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        apply_input_transform: bool = True,
        **kwargs: Any,
    ) -> Any:
        """raw-space X に対する latent f の posterior を返す。"""
        _ = kwargs
        if isinstance(X, tuple):
            X = X[0]

        self.eval()
        X_eval = self.transform_inputs(X) if apply_input_transform else X
        post = GPyTorchPosterior(self.model(X_eval))

        if posterior_transform is not None:
            post = posterior_transform(post)

        return post

    def posterior_latent(self, X: Tensor, **kwargs: Any) -> Any:
        return self.latent_posterior(X, **kwargs)

    def posterior_f(self, X: Tensor, **kwargs: Any) -> Any:
        return self.latent_posterior(X, **kwargs)

    def forward(self, X: Tensor, apply_input_transform: bool = True):
        if isinstance(X, tuple):
            X = X[0]

        X_eval = self.transform_inputs(X) if apply_input_transform else X
        return self.model(X_eval)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "SaasBinaryClassificationMixedGPModel":
        """raw/encoded X の追加観測で wrapper を再構築する。"""
        _ = kwargs
        X_new_raw, Y_new, Yvar_new = prepare_mixed_conditioning_data(
            X,
            Y,
            noise,
            raw_dim=self.raw_dim,
            encoded_dim=self.encoded_dim,
            decode_fn=self.decode_inputs,
            target_dtype=self.train_X_raw.dtype,
        )
        train_X_old = self.train_inputs_raw[0]
        train_Y_old = flatten_targets(self.train_targets, dtype=train_X_old.dtype)
        X_full = torch.cat(
            [
                train_X_old,
                X_new_raw.to(dtype=train_X_old.dtype, device=train_X_old.device),
            ],
            dim=0,
        )
        Y_full = torch.cat(
            [
                train_Y_old,
                Y_new.to(dtype=train_Y_old.dtype, device=train_Y_old.device),
            ],
            dim=0,
        )
        Yvar_full = concat_optional_noise(
            old_Y=train_Y_old,
            old_Yvar=self.train_Yvar_raw,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        inducing_points = None
        if hasattr(self, "model") and hasattr(self.model, "variational_strategy"):
            inducing_points = self.model.variational_strategy.inducing_points.detach().clone()

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            cat_dims=list(self.cat_dims),
            train_Yvar=Yvar_full,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(getattr(self, "input_transform", None)),
            mean_module=deepcopy(getattr(self.model, "mean_module", None)),
            covar_module=deepcopy(getattr(self.model, "covar_module", None)),
            num_inducing_points=(inducing_points.shape[-2] if inducing_points is not None else 20),
            inducing_points=inducing_points,
            learn_inducing_locations=getattr(
                getattr(self.model, "variational_strategy", None),
                "learn_inducing_locations",
                True,
            ),
            tau=self.tau,
            saas_log_scale=self.saas_log_scale,
            saas_nu=self.saas_nu,
        )
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model


__all__ = [
    "SaasBinaryClassificationGPModel",
    "SaasBinaryClassificationMixedGPModel",
]
