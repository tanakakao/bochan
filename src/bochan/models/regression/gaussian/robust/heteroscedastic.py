
from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
from torch import Tensor

from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean

from bochan.models.components.heteroscedastic import (
    HeteroscedasticLatentPosteriorMixin,
    clone_input_transform,
    check_categorical_columns_unchanged,
    compute_regression_log_var_from_residuals,
    ensure_2d_col,
    fit_exact_mll,
    fit_noise_model_mixed,
    fit_noise_model_single,
    make_normalize_only_transform,
    predict_noise_var_from_log_noise_model,
    prepare_train_yvar,
)


__all__ = [
    "HeteroscedasticSingleTaskGP",
    "HeteroscedasticMixedSingleTaskGP",
]


class HeteroscedasticSingleTaskGP(
    HeteroscedasticLatentPosteriorMixin,
    SingleTaskGP,
):
    """
    入力依存ノイズを持つ SingleTaskGP。

    方針:
        1. 補助 base model を学習する。
        2. 残差二乗または train_Yvar から log variance target を作る。
        3. noise_model が log variance を回帰する。
        4. final model には predicted noise variance を train_Yvar として渡す。
        5. posterior(..., observation_noise=True) のときだけ noise_model の分散を足す。

    Args:
        train_X: raw-space の学習入力。shape は [N, d]。
        train_Y: 学習ターゲット。shape は [N, 1] または [N]。
        train_Yvar: 既知の観測ノイズ分散。指定された場合は補助 residual 推定より優先する。
        likelihood: final model に渡す likelihood。
        covar_module: final model に渡す covariance module。
        mean_module: final model に渡す mean module。
        outcome_transform: final model に渡す outcome transform。
        input_transform: final model に渡す input transform。
            noise_model 学習では、この中から Normalize のみを抽出して使う。
        min_noise: noise variance の下限。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform=None,
        input_transform: Optional[InputTransform] = None,
        min_noise: float = 1e-6,
    ) -> None:
        train_X_raw = train_X.detach().clone()
        train_Y_col = ensure_2d_col(train_Y).to(train_X)

        self.train_inputs_raw = (train_X_raw,)
        self.min_noise = float(min_noise)

        noise_tf = make_normalize_only_transform(
            input_transform=input_transform,
            train_X=train_X_raw,
        )

        provided_yvar = prepare_train_yvar(
            train_Yvar=train_Yvar,
            ref_X=train_X_raw,
            min_noise=self.min_noise,
        )

        if provided_yvar is None:
            base_model = SingleTaskGP(
                train_X=train_X_raw,
                train_Y=train_Y_col,
                likelihood=copy.deepcopy(likelihood),
                covar_module=copy.deepcopy(covar_module),
                mean_module=copy.deepcopy(mean_module),
                outcome_transform=copy.deepcopy(outcome_transform),
                input_transform=clone_input_transform(noise_tf),
            )
            fit_exact_mll(base_model)
            train_log_var = compute_regression_log_var_from_residuals(
                base_model=base_model,
                train_X=train_X_raw,
                train_Y=train_Y_col,
                min_noise=self.min_noise,
            )
        else:
            train_log_var = provided_yvar.log()

        noise_model = fit_noise_model_single(
            train_X=train_X_raw,
            train_Y_log_var=train_log_var,
            input_transform=clone_input_transform(noise_tf),
        )

        predicted_noise_var = predict_noise_var_from_log_noise_model(
            noise_model=noise_model,
            X=train_X_raw,
            ref_like=train_Y_col,
            min_noise=self.min_noise,
        )

        super().__init__(
            train_X=train_X_raw,
            train_Y=train_Y_col,
            train_Yvar=predicted_noise_var.to(train_X_raw),
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.train_inputs_raw = (train_X_raw,)
        self.noise_model = noise_model
        self.noise_input_transform = noise_tf


class HeteroscedasticMixedSingleTaskGP(
    HeteroscedasticLatentPosteriorMixin,
    MixedSingleTaskGP,
):
    """
    mixed 入力対応の入力依存ノイズ付き MixedSingleTaskGP。

    Args:
        train_X: raw-space の学習入力。カテゴリ列は integer-coded を想定。
        train_Y: 学習ターゲット。
        cat_dims: raw-space におけるカテゴリ列 index。
        train_Yvar: 既知の観測ノイズ分散。指定時は residual 推定より優先。
        cont_kernel_factory: MixedSingleTaskGP に渡す continuous kernel factory。
        likelihood: final model に渡す likelihood。
        outcome_transform: final model に渡す outcome transform。
        input_transform: final model に渡す input transform。
            mixed ではカテゴリ列を変換しないものを想定する。
        min_noise: noise variance の下限。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        cont_kernel_factory=None,
        likelihood: Optional[Likelihood] = None,
        outcome_transform=None,
        input_transform: Optional[InputTransform] = None,
        min_noise: float = 1e-6,
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be non-empty for HeteroscedasticMixedSingleTaskGP.")

        train_X_raw = train_X.detach().clone()
        train_Y_col = ensure_2d_col(train_Y).to(train_X_raw)

        self.train_inputs_raw = (train_X_raw,)
        self.cat_dims = [int(i) for i in cat_dims]
        self.min_noise = float(min_noise)

        noise_tf = make_normalize_only_transform(
            input_transform=input_transform,
            train_X=train_X_raw,
            cat_dims=self.cat_dims,
        )

        if noise_tf is not None:
            noise_tf.eval()
            with torch.no_grad():
                X_check = noise_tf(train_X_raw)
            check_categorical_columns_unchanged(
                X=train_X_raw,
                X_tf=X_check,
                cat_dims=self.cat_dims,
                name=f"{self.__class__.__name__}.noise_input_transform",
            )

        provided_yvar = prepare_train_yvar(
            train_Yvar=train_Yvar,
            ref_X=train_X_raw,
            min_noise=self.min_noise,
        )

        if provided_yvar is None:
            base_model = MixedSingleTaskGP(
                train_X=train_X_raw,
                train_Y=train_Y_col,
                cat_dims=list(self.cat_dims),
                cont_kernel_factory=cont_kernel_factory,
                likelihood=copy.deepcopy(likelihood),
                outcome_transform=copy.deepcopy(outcome_transform),
                input_transform=clone_input_transform(noise_tf),
            )
            fit_exact_mll(base_model)
            train_log_var = compute_regression_log_var_from_residuals(
                base_model=base_model,
                train_X=train_X_raw,
                train_Y=train_Y_col,
                min_noise=self.min_noise,
            )
        else:
            train_log_var = provided_yvar.log()

        noise_model = fit_noise_model_mixed(
            train_X=train_X_raw,
            train_Y_log_var=train_log_var,
            cat_dims=list(self.cat_dims),
            input_transform=clone_input_transform(noise_tf),
        )

        predicted_noise_var = predict_noise_var_from_log_noise_model(
            noise_model=noise_model,
            X=train_X_raw,
            ref_like=train_Y_col,
            min_noise=self.min_noise,
        )

        super().__init__(
            train_X=train_X_raw,
            train_Y=train_Y_col,
            train_Yvar=predicted_noise_var.to(train_X_raw),
            cat_dims=list(self.cat_dims),
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.train_inputs_raw = (train_X_raw,)
        self.noise_model = noise_model
        self.noise_input_transform = noise_tf
