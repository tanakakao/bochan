import torch
from torch import Tensor
from typing import Optional, Union, Sequence

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood

from botorch.acquisition.objective import PosteriorTransform
from gpytorch.models.deep_gps import DeepGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices

from bochan.models.components.layers import DeepKernel, DeepKernelMixed


InputTransformArg = Union[str, InputTransform, None]
OutcomeTransformArg = Union[str, OutcomeTransform, None]



def _clone_tensor_tuple(inputs: Union[Tensor, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """Tensor または Tensor tuple を detach + clone して保持する。"""
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    return tuple(x.detach().clone() for x in inputs)


def _expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """InputPerturbation 後の q 展開に合わせて raw X を repeat する。"""
    if X.shape == X_tf.shape:
        return X
    if X.ndim < 2 or X_tf.ndim < 2:
        return X
    if X.shape[-1] != X_tf.shape[-1]:
        return X
    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]
        if q_like == q:
            return X
        if q > 0 and q_like % q == 0:
            return X.repeat_interleave(q_like // q, dim=-2)
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)
    return X


def _check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """mixed model で input_transform がカテゴリ列を変更していないか確認する。"""
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = _expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            "input_transform must not modify categorical columns. "
            "For mixed models, transform only continuous columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def _apply_input_transform_for_training(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
    name: str = "input_transform",
) -> Tensor:
    """学習データ用に input_transform を一度だけ適用する。"""
    if input_transform is None:
        return X

    if hasattr(input_transform, "train"):
        input_transform.train()
    X_tf = input_transform(X)
    if hasattr(input_transform, "eval"):
        input_transform.eval()

    if X_tf.shape[-2] != X.shape[-2]:
        raise RuntimeError(
            f"{name} expanded training inputs. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}. "
            "For InputPerturbation, use transform_on_train=False."
        )

    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """posterior / acquisition 評価用に input_transform を適用する。"""
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


class _BaseDeepKernelGPModel(DeepGP, GPyTorchModel):
    """
    DeepKernel 系回帰モデルの共通 wrapper。

    設計:
        - wrapper が input_transform / outcome_transform を管理
        - inner exact GP は transform 後 train_X を保持
        - forward() は latent GP distribution を返す
        - posterior() は BoTorch 流に予測 posterior を返す
    """

    def __init__(self) -> None:
        super().__init__()
        self._num_outputs = 1
        self._model_dtype = None
        self._model_device = None

    # ------------------------------------------------------------------
    # transform 解決
    # ------------------------------------------------------------------
    def _resolve_outcome_transform(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        outcome_transform: OutcomeTransformArg,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        outcome_transform を解決し、必要なら train_Y / train_Yvar に適用する。
        """
        if isinstance(outcome_transform, str):
            key = outcome_transform.upper()
            if key == "DEFAULT":
                outcome_transform = Standardize(
                    m=train_Y.shape[-1],
                    batch_shape=train_X.shape[:-2],
                )
            elif key in ("NONE", ""):
                outcome_transform = None
            else:
                raise ValueError(f"Unknown outcome_transform: {outcome_transform}")

        self.outcome_transform = outcome_transform

        if self.outcome_transform is not None:
            train_Y, train_Yvar = self.outcome_transform(train_Y, train_Yvar)

        return train_Y, train_Yvar

    def _resolve_input_transform(
        self,
        train_X: Tensor,
        input_transform: InputTransformArg,
    ) -> None:
        """
        input_transform を解決し、モデルへ設定する。
        """
        input_dim = train_X.shape[-1]

        if isinstance(input_transform, str):
            key = input_transform.upper()
            if key == "DEFAULT":
                # mixed model ではカテゴリ列を正規化しない。
                indices = getattr(self, "ord_dims", None)
                if indices is not None:
                    input_transform = Normalize(d=input_dim, indices=indices)
                else:
                    input_transform = Normalize(d=input_dim)
            elif key in ("NONE", ""):
                input_transform = None
            else:
                raise ValueError(f"Unknown input_transform: {input_transform}")

        self.input_transform = input_transform

        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            if hasattr(self.input_transform, "eval"):
                self.input_transform.eval()

    # ------------------------------------------------------------------
    # shape / target 整形
    # ------------------------------------------------------------------
    @staticmethod
    def _get_num_outputs(train_Y: Tensor) -> int:
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            return train_Y.shape[-1]
        return 1

    @staticmethod
    def _prepare_targets_for_inner_model(train_Y: Tensor) -> Tensor:
        """
        inner ExactGP に渡す train_Y の shape を整える。
        単出力の (n, 1) は (n,) にする。
        """
        if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
            return train_Y.squeeze(-1)
        return train_Y

    def _setup_common(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        input_transform: InputTransformArg,
        outcome_transform: OutcomeTransformArg,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        共通初期化処理。

        Returns:
            train_X_raw, train_X_tf, prepared_train_Y
        """
        self._model_dtype = train_X.dtype
        self._model_device = train_X.device

        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        train_Y, train_Yvar = self._resolve_outcome_transform(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
        )
        self._resolve_input_transform(
            train_X=train_X,
            input_transform=input_transform,
        )

        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)

        self._num_outputs = self._get_num_outputs(train_Y)
        prepared_train_Y = self._prepare_targets_for_inner_model(train_Y)

        # wrapper 側は raw 入力を保持する。
        self.train_inputs = (train_X,)
        self.train_inputs_raw = _clone_tensor_tuple(train_X)
        self.train_targets = train_Y
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar

        # inner 側には transform 後の入力を渡す。
        # 学習時は InputPerturbation による q 展開を許さない。
        train_X_tf = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=getattr(self, "cat_dims", None),
            name=f"{self.__class__.__name__}.input_transform",
        )

        return train_X, train_X_tf, prepared_train_Y

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap_inputs(inputs) -> Tensor:
        if isinstance(inputs, tuple):
            return inputs[0]
        return inputs

    def _to_model_dtype_device(self, X: Tensor) -> Tensor:
        return X.to(device=self._model_device, dtype=self._model_dtype)

    def _set_transformed_inputs(self) -> None:
        """BoTorch の eval 時 transformed input 自動更新を無効化する。"""
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """評価時の input_transform を適用する。"""
        return _apply_input_transform_for_eval(
            X,
            getattr(self, "input_transform", None),
            cat_dims=getattr(self, "cat_dims", None),
        )

    def _apply_input_transform(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> Tensor:
        if apply_input_transform and self.input_transform is not None:
            return self.transform_inputs(X)
        return X

    def _apply_observation_noise(self, mvn, observation_noise: Union[bool, Tensor]):
        """
        observation_noise を適用する。
        """
        if observation_noise is False:
            return mvn

        if observation_noise is True:
            return self.likelihood(mvn)

        if torch.is_tensor(observation_noise):
            # GaussianLikelihood は noise=... を受けられるが、
            # shape の扱いがモデルにより異なるのでここでは明示的に対応
            return self.likelihood(mvn, noise=observation_noise)

        raise ValueError("observation_noise は bool または Tensor を想定しています。")

    # ------------------------------------------------------------------
    # boTorch 用 API
    # ------------------------------------------------------------------
    def set_train_data(self, inputs=None, targets=None, strict: bool = True) -> None:
        """
        学習データ更新メソッド。

        inputs は raw 空間で受け取り、inner model 側には transform 後を反映する。
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            raw_X = self._to_model_dtype_device(inputs[0])
            self.train_inputs = (raw_X,)
            self.train_inputs_raw = _clone_tensor_tuple(raw_X)
            self.train_X = raw_X

            X_tf = _apply_input_transform_for_training(
                raw_X,
                self.input_transform,
                cat_dims=getattr(self, "cat_dims", None),
                name=f"{self.__class__.__name__}.input_transform",
            )
            self.deepkernel.set_train_data(inputs=X_tf, targets=None, strict=strict)

        if targets is not None:
            self.train_targets = targets
            self.train_Y = targets
            prepared_targets = self._prepare_targets_for_inner_model(targets)
            self.deepkernel.set_train_data(inputs=None, targets=prepared_targets, strict=strict)

    def forward(
        self,
        inputs,
        apply_input_transform: bool = True,
    ):
        """
        latent GP distribution を返す。

        - training 時:
            transformed X を inner.forward(...) に直接渡す
            -> ExactGP.__call__ の train_inputs 一致チェックを回避
        - eval 時:
            inner(x) を使って通常の posterior を得る
        """
        x = self._unwrap_inputs(inputs)
        x = self._to_model_dtype_device(x)
        x = self._apply_input_transform(
            x,
            apply_input_transform=apply_input_transform,
        )

        if self.training:
            return self.deepkernel.forward(x)

        return self.deepkernel(x)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ):
        """
        posterior を返す。

        - torch.no_grad() は使わない
        - input_transform はここで一度だけ適用
        - outcome_transform の逆変換もここで適用
        """
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )

        self.eval()

        X = self._unwrap_inputs(X)
        X = self._to_model_dtype_device(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)

        with fast_pred_var():
            mvn = self.forward(
                X_tf,
                apply_input_transform=False,
            )
            mvn = self._apply_observation_noise(mvn, observation_noise)

            posterior = GPyTorchPosterior(mvn)

            if getattr(self, "outcome_transform", None) is not None:
                posterior = self.outcome_transform.untransform_posterior(
                    posterior,
                    X=X_tf,
                )

            if posterior_transform is not None:
                posterior = posterior_transform(posterior)

        return posterior

    def predict_latent(
        self,
        X: Tensor,
        observation_noise: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        outcome_transform 後の空間で平均・分散を返す。
        """
        self.eval()

        X = self._unwrap_inputs(X)
        X = self._to_model_dtype_device(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)

        with fast_pred_var():
            mvn = self.forward(
                X_tf,
                apply_input_transform=False,
            )
            if observation_noise:
                mvn = self.likelihood(mvn)

            posterior = GPyTorchPosterior(mvn)
            return posterior.mean, posterior.variance

    def predict(
        self,
        X: Tensor,
        observation_noise: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        元スケールで平均・分散を返す。
        """
        posterior = self.posterior(
            X,
            observation_noise=observation_noise,
        )
        return posterior.mean, posterior.variance

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """
        この wrapper 用の ExactMarginalLogLikelihood を返す。
        """
        return ExactMarginalLogLikelihood(self.likelihood, self.deepkernel)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs


class DeepKernelGPModel(_BaseDeepKernelGPModel):
    """
    連続入力向け Deep Kernel GP 回帰モデル。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
    ) -> None:
        super().__init__()

        train_X_raw, train_X_tf, prepared_train_Y = self._setup_common(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        # likelihood はここで作る
        if likelihood is None:
            if self._num_outputs == 1:
                from botorch.models.utils.gpytorch_modules import (
                    get_gaussian_likelihood_with_lognormal_prior,
                )
                # likelihood = get_gaussian_likelihood_with_lognormal_prior()
                likelihood = GaussianLikelihood()
            else:
                from gpytorch.likelihoods import MultitaskGaussianLikelihood
                likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)

        self.likelihood = likelihood

        self.deepkernel = DeepKernel(
            train_x=train_X_tf,
            train_y=prepared_train_Y,
            likelihood=self.likelihood,
            ext_type=ext_type,
        )
        self.to(train_X)


class DeepKernelMixedGPModel(_BaseDeepKernelGPModel):
    """
    混合入力（連続 + カテゴリ）向け Deep Kernel GP 回帰モデル。

    注意:
        input_transform はカテゴリ列を触らないものを使うこと。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
    ) -> None:
        super().__init__()

        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims

        train_X_raw, train_X_tf, prepared_train_Y = self._setup_common(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        if likelihood is None:
            if self._num_outputs == 1:
                from botorch.models.utils.gpytorch_modules import (
                    get_gaussian_likelihood_with_lognormal_prior,
                )
                likelihood = get_gaussian_likelihood_with_lognormal_prior()
            else:
                from gpytorch.likelihoods import MultitaskGaussianLikelihood
                likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)

        self.likelihood = likelihood

        self.deepkernel = DeepKernelMixed(
            train_x=train_X_tf,
            train_y=prepared_train_Y,
            cat_dims=cat_dims,
            likelihood=self.likelihood,
            ext_type=ext_type,
        )
        self.to(train_X)