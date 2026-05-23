"""BoTorch 形式に寄せた Deep Gaussian Process 回帰モデル群。

このモジュールでは、連続入力用および連続値・カテゴリ値混合入力用の
DeepGP 回帰モデルを提供します。変分 DeepGP で実用上可能な範囲で
BoTorch のモデル規約に近づけています。

重要な設計方針として、``train_inputs`` と ``train_inputs_raw`` は
raw-space の入力を保持します。一方、``input_transform`` は必要に応じて
``forward`` または ``posterior`` の内部で適用します。

公開クラス:
    DeepGPModel: 連続入力向けの DeepGP 回帰モデル。
    DeepMixedGPModel: 連続値 + カテゴリ値の混合入力向け DeepGP 回帰モデル。

使用例:
    >>> model = DeepGPModel(train_X, train_Y, list_hidden_dims=[16, 16])
    >>> mll = model.make_mll(beta=1.0)
    >>> output = model(train_X)
    >>> loss = -mll(output, train_Y)
    >>> posterior = model.posterior(test_X)

Notes:
    - ``forward`` は学習用の latent DeepGP distribution を返します。
      主に ``DeepApproximateMLL`` での学習に使います。
    - ``posterior`` は BoTorch の獲得関数や予測で使いやすい
      ``GPyTorchPosterior`` を返します。
    - mixed 入力モデルでは、``input_transform`` によってカテゴリ列が
      変更されてはいけません。独自 transform を渡す場合は
      ``Normalize(..., indices=continuous_dims)`` のように連続列だけを
      変換してください。
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP
from gpytorch.settings import fast_pred_var

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices

from bochan.models.components.layers import (
    DeepGPHiddenLayer,
    DeepMixedGPHiddenLayer,
    SkipDeepGPHiddenLayer,
    SkipDeepMixedGPHiddenLayer,
)


InputTransformArg = Union[str, InputTransform, None]
OutcomeTransformArg = Union[str, OutcomeTransform, None]


# ============================================================
# ヘルパー
# ============================================================


def _expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """
    InputPerturbation などで X_tf の q 次元が q*n_w に展開された場合に、
    raw X 側を比較可能な形へ展開する。
    """
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
    """
    学習データで input_transform を初期化するための処理。

    train_inputs は raw-space のまま保持する。ここでの変換値は、
    inducing point 初期化など、transformed-space の参照が必要な箇所だけで使う。
    """
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
    """
    posterior / acquisition 評価用の input_transform。

    eval mode の InputPerturbation による q -> q*n_w 展開は許容する。
    mixed model ではカテゴリ列が変更されていないか確認する。
    """
    if input_transform is None:
        return X

    X_tf = input_transform(X)
    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _clone_train_inputs(inputs: Union[Tensor, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    return tuple(x.detach().clone() for x in inputs)


# ============================================================
# 基底モデル
# ============================================================


class _BaseDeepGPModel(DeepGP, GPyTorchModel):
    """
    DeepGP regression 系モデルの共通基底クラス。

    設計方針:
        - train_inputs / train_inputs_raw は raw-space を保持する
        - forward() は学習用 latent distribution を返す
        - posterior() は BoTorch acquisition 用の GPyTorchPosterior を返す
    """

    def __init__(self) -> None:
        super().__init__()

    def _set_transformed_inputs(self) -> None:
        """
        BoTorch Model.eval() が呼ぶ transformed input 自動更新を無効化する。

        DeepGP wrapper では train_inputs を raw-space のまま保持し、
        forward / posterior 内で必要に応じて input_transform を適用する。
        """
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """予測または獲得関数評価用に input_transform を適用する。

        Args:
            X: raw-space の候補点入力。形状は ``batch_shape x q x d`` または
                ``q x d`` を想定する。

        Returns:
            変換後の入力。``InputPerturbation`` を使う場合は、q 次元が
            ``q`` から ``q * n_w`` に展開されることがある。
        """
        return _apply_input_transform_for_eval(
            X,
            getattr(self, "input_transform", None),
            cat_dims=getattr(self, "cat_dims", None),
        )

    def _setup_transforms(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        input_transform: InputTransformArg,
        outcome_transform: OutcomeTransformArg,
        *,
        input_transform_indices: Optional[Sequence[int]] = None,
        cat_dims: Optional[Sequence[int]] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """input_transform / outcome_transform を解決してモデルへ設定する。"""
        input_dim = train_X.shape[-1]

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

        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
            self.outcome_transform = outcome_transform
        else:
            self.outcome_transform = None

        if isinstance(input_transform, str):
            key = input_transform.upper()
            if key == "DEFAULT":
                if input_transform_indices is None:
                    input_transform = Normalize(d=input_dim)
                else:
                    input_transform = Normalize(
                        d=input_dim,
                        indices=list(input_transform_indices),
                    )
            elif key in ("NONE", ""):
                input_transform = None
            else:
                raise ValueError(f"Unknown input_transform: {input_transform}")

        self.input_transform = input_transform
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)
            _apply_input_transform_for_training(
                train_X,
                self.input_transform,
                cat_dims=cat_dims,
                name=f"{self.__class__.__name__}.input_transform",
            )

        return train_Y, train_Yvar

    def _setup_training_data(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
    ) -> None:
        """学習データを検証し、BoTorch / GPyTorch 互換の属性を設定する。"""
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        _, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X,
            train_Y=train_Y,
        )
        self.train_inputs = (train_X,)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets = train_Y

    def _setup_likelihood(self, likelihood, num_outputs: int) -> None:
        if likelihood is not None:
            self.likelihood = likelihood
            return

        if num_outputs == 1:
            self.likelihood = get_gaussian_likelihood_with_lognormal_prior(
                batch_shape=self._aug_batch_shape
            )
        else:
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)

    @staticmethod
    def _unwrap_inputs(inputs: Union[Tensor, tuple[Tensor, ...]]) -> Tensor:
        if isinstance(inputs, tuple):
            return inputs[0]
        return inputs

    def _apply_input_transform(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> Tensor:
        if apply_input_transform and self.input_transform is not None:
            return self.transform_inputs(X)
        return X

    def _average_deepgp_output_distribution(self, output, X: Tensor):
        """
        DeepGP の extra batch / sample 次元を平均化して、通常の MVN / MTMVN に変換する。
        """
        mean = output.mean
        covar = output.covariance_matrix

        if self._num_outputs > 1:
            expected_mean_ndim = X.ndim
            expected_covar_ndim = X.ndim
        else:
            expected_mean_ndim = X.ndim - 1
            expected_covar_ndim = X.ndim

        while mean.ndim > expected_mean_ndim:
            mean = mean.mean(dim=0)
        while covar.ndim > expected_covar_ndim:
            covar = covar.mean(dim=0)

        if self._num_outputs > 1:
            return MultitaskMultivariateNormal(mean, covar)
        return MultivariateNormal(mean, covar)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> GPyTorchPosterior:
        """回帰出力に対する BoTorch 互換の posterior を返す。

        Args:
            X: raw-space のテスト点または候補点。形状は
                ``batch_shape x q x d`` または ``q x d`` を想定する。
            output_indices: 返したい出力次元の指定。この wrapper では現在未対応。
            observation_noise: ``True`` または noise tensor の場合、likelihood を
                適用して観測ノイズ込みの posterior を返す。
            posterior_transform: 任意の BoTorch posterior transform。

        Returns:
            ``GPyTorchPosterior``。``outcome_transform`` がある場合は、
            元の目的変数スケールに戻した posterior を返す。

        Raises:
            NotImplementedError: ``output_indices`` が指定された場合。
        """
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )

        self.eval()
        if hasattr(self, "likelihood"):
            self.likelihood.eval()

        X = self._unwrap_inputs(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)

        with fast_pred_var():
            latent_dist = self.forward(X_tf, apply_input_transform=False)
            mvn = self._average_deepgp_output_distribution(latent_dist, X_tf)

            if observation_noise is not False:
                mvn = self.likelihood(mvn)

            posterior = GPyTorchPosterior(mvn)

            if getattr(self, "outcome_transform", None) is not None:
                posterior = self.outcome_transform.untransform_posterior(
                    posterior,
                    X=X_tf,
                )

            if posterior_transform is not None:
                posterior = posterior_transform(posterior)

        return posterior

    def make_mll(self, beta: float = 1.0) -> DeepApproximateMLL:
        """DeepGP の学習に使う marginal log likelihood を作成する。

        Args:
            beta: ``VariationalELBO`` に渡す KL divergence の重み。

        Returns:
            ``VariationalELBO`` を ``DeepApproximateMLL`` で包んだ MLL。

        Example:
            >>> model = DeepGPModel(train_X, train_Y)
            >>> mll = model.make_mll()
            >>> output = model(train_X)
            >>> loss = -mll(output, model.train_targets)
        """
        base_mll = VariationalELBO(
            likelihood=self.likelihood,
            model=self,
            num_data=self.train_inputs[0].shape[-2],
            beta=float(beta),
        )
        return DeepApproximateMLL(base_mll)

    def set_train_data(self, inputs=None, targets=None, strict: bool = True) -> None:
        _ = strict
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
            self.train_inputs_raw = _clone_train_inputs(inputs)

        if targets is not None:
            self.train_targets = targets

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @staticmethod
    def get_batch_dimensions(train_X: Tensor, train_Y: Tensor) -> tuple[torch.Size, torch.Size]:
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape


# ============================================================
# 連続入力モデル
# ============================================================


class DeepGPModel(_BaseDeepGPModel):
    """連続入力向けの Deep Gaussian Process 回帰モデル。

    このモデルは、連続値の設計変数を対象とします。学習入力は
    ``train_inputs`` / ``train_inputs_raw`` に raw-space のまま保持し、
    ``input_transform`` は ``forward`` および ``posterior`` の内部で適用します。
    単一出力・多出力の両方に対応します。多出力の場合、最終 layer は
    ``output_dims=num_outputs`` となり、デフォルト likelihood には
    ``MultitaskGaussianLikelihood`` を使います。

    Args:
        train_X: 学習入力。形状は ``batch_shape x n x d`` または ``n x d``。
            raw-space の値として保持される。
        train_Y: 学習目的変数。形状は ``batch_shape x n x m`` または
            ``n x m``。``m`` は出力次元数。
        train_Yvar: 既知の観測ノイズ分散。BoTorch 形式の constructor 互換性と
            入力検証のために受け取る。``likelihood`` が ``None`` の場合は
            デフォルト likelihood を自動構築する。
        likelihood: 任意の GPyTorch likelihood。省略時は、単一出力では
            Gaussian likelihood、多出力では multitask Gaussian likelihood を使う。
        input_transform: モデル内部で適用する入力変換。``"DEFAULT"`` では
            ``Normalize(d=train_X.shape[-1])`` を作成する。``"NONE"`` または
            ``None`` では入力変換を無効化する。
        outcome_transform: 目的変数変換。``"DEFAULT"`` では
            ``Standardize(m=train_Y.shape[-1])`` を作成する。``"NONE"`` または
            ``None`` では目的変数変換を無効化する。
        list_hidden_dims: hidden layer の出力次元リスト。デフォルトは ``[10]``。
            各要素につき 1 つの DeepGP hidden layer を作成する。
        model_type: モデル構造の指定。``"DEFAULT"`` では通常の層状 DeepGP、
            ``"skip"`` では元入力を skip-compatible layer に再注入する。
        num_inducing: 各 DeepGP layer の inducing point 数。

    Attributes:
        train_inputs: ``(train_X,)`` として保持する raw-space 学習入力。
        train_inputs_raw: raw-space 学習入力の detached clone。
        train_targets: 必要に応じて outcome_transform 済みの学習目的変数。
        input_transform: ``forward`` / ``posterior`` 内で適用する入力変換。
        outcome_transform: posterior 予測を元スケールへ戻すための目的変数変換。
        likelihood: 変分目的関数で使う GPyTorch likelihood。
        num_outputs: モデルの出力次元数。

    Notes:
        ``forward`` は学習用の latent GPyTorch distribution を返します。
        BoTorch の獲得関数には ``posterior`` を使ってください。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        list_hidden_dims: Optional[Sequence[int]] = None,
        model_type: str = "DEFAULT",
        num_inducing: int = 128,
    ) -> None:
        super().__init__()

        input_dim = train_X.shape[-1]
        num_outputs = train_Y.shape[-1]
        hidden_dims = list(list_hidden_dims) if list_hidden_dims is not None else [10]
        if len(hidden_dims) == 0:
            raise ValueError("list_hidden_dims には少なくとも1つの要素が必要です。")

        train_Y, train_Yvar = self._setup_transforms(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self._setup_training_data(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)

        self.hidden_layers = torch.nn.ModuleList()
        self.use_skip = model_type.lower() == "skip"
        self.original_input_dim = input_dim
        self.num_inducing = int(num_inducing)

        current_input_dim = input_dim
        for hidden_dim in hidden_dims:
            if self.use_skip:
                self.hidden_layers.append(
                    SkipDeepGPHiddenLayer(
                        base_input_dims=current_input_dim,
                        skip_input_dims=self.original_input_dim,
                        output_dims=hidden_dim,
                        num_inducing=self.num_inducing,
                        mean_type="linear",
                    )
                )
            else:
                self.hidden_layers.append(
                    DeepGPHiddenLayer(
                        input_dims=current_input_dim,
                        output_dims=hidden_dim,
                        num_inducing=self.num_inducing,
                        mean_type="linear",
                    )
                )
            current_input_dim = hidden_dim

        if self.use_skip:
            self.last_layer = SkipDeepGPHiddenLayer(
                base_input_dims=current_input_dim,
                skip_input_dims=self.original_input_dim,
                output_dims=None if num_outputs == 1 else num_outputs,
                num_inducing=self.num_inducing,
                mean_type="constant",
            )
        else:
            self.last_layer = DeepGPHiddenLayer(
                input_dims=current_input_dim,
                output_dims=None if num_outputs == 1 else num_outputs,
                num_inducing=self.num_inducing,
                mean_type="constant",
            )

        self._num_outputs = num_outputs
        self._setup_likelihood(likelihood=likelihood, num_outputs=num_outputs)
        self.to(train_X)

    def forward(self, inputs, apply_input_transform: bool = True):
        """latent DeepGP distribution を評価する。

        Args:
            inputs: 入力 tensor、または ``(input_tensor,)`` 形式の tuple。
                ``apply_input_transform=False`` でない限り、raw-space の入力を想定する。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。
                通常の学習時および外部呼び出しでは ``True`` のままでよい。
                ``posterior`` では明示的に変換済みの入力を渡すため、
                ``False`` として ``forward`` を呼ぶ。

        Returns:
            最終 DeepGP layer から得られる latent GPyTorch distribution。
        """
        x = self._unwrap_inputs(inputs)
        x = self._apply_input_transform(x, apply_input_transform=apply_input_transform)

        original_input = x
        h = x
        for layer in self.hidden_layers:
            if isinstance(layer, SkipDeepGPHiddenLayer):
                h = layer(h, original_input=original_input)
            else:
                h = layer(h)

        if isinstance(self.last_layer, SkipDeepGPHiddenLayer):
            return self.last_layer(h, original_input=original_input)
        return self.last_layer(h)


# ============================================================
# 混合入力モデル
# ============================================================


class DeepMixedGPModel(_BaseDeepGPModel):
    """連続値・カテゴリ値の混合入力向け DeepGP 回帰モデル。

    最初の layer には、連続列とカテゴリ列の両方を扱える mixed-aware な
    DeepGP layer を使います。最終 layer は ``model_type`` に応じて、
    通常の DeepGP layer または skip mixed layer になります。
    カテゴリ列は raw の整数エンコード空間で保持し、``input_transform`` によって
    カテゴリ列が変更されていないことをチェックします。

    Args:
        train_X: raw-space の学習入力。形状は ``batch_shape x n x d`` または
            ``n x d``。カテゴリ列は整数エンコードされている必要がある。
        train_Y: 学習目的変数。形状は ``batch_shape x n x m`` または ``n x m``。
        cat_dims: ``train_X`` におけるカテゴリ列の index。負の index は
            BoTorch の ``normalize_indices`` で正規化される。
        train_Yvar: 既知の観測ノイズ分散。BoTorch 形式の constructor 互換性のために受け取る。
        likelihood: 任意の GPyTorch likelihood。省略時は、単一出力では
            Gaussian likelihood、多出力では multitask Gaussian likelihood を使う。
        input_transform: 入力変換。``"DEFAULT"`` では連続列だけに対して
            ``Normalize`` を作成する。独自 transform はカテゴリ列を変更してはいけない。
        outcome_transform: 目的変数変換。``"DEFAULT"`` では ``train_Y`` 用の
            ``Standardize`` を作成する。
        hidden_dim: 最初の mixed hidden layer の出力次元。
        model_type: モデル構造の指定。``"DEFAULT"`` では通常の最終 layer、
            ``"skip"`` では raw/transformed の元入力を最終 mixed layer に再注入する。
        num_inducing: 各 DeepGP layer の inducing point 数。

    Attributes:
        cat_dims: 正規化済みのカテゴリ列 index。
        ord_dims: 正規化・スケーリング対象となる非カテゴリ列 index。
        train_inputs_raw: raw-space 学習入力の detached clone。
        input_layer: mixed-aware な最初の DeepGP hidden layer。
        last_layer: latent 出力を返す最終 DeepGP layer。

    Raises:
        ValueError: ``cat_dims`` が空の場合、または入力変換がカテゴリ列を変更した場合。
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
        hidden_dim: int = 8,
        model_type: str = "DEFAULT",
        num_inducing: int = 128,
    ) -> None:
        super().__init__()

        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        input_dim = train_X.shape[-1]
        num_outputs = train_Y.shape[-1]
        d = train_X.shape[-1]
        cat_dims = list(normalize_indices(indices=cat_dims, d=d))
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims

        train_Y, train_Yvar = self._setup_transforms(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            input_transform_indices=ord_dims,
            cat_dims=cat_dims,
        )
        self._setup_training_data(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)

        self.use_skip = model_type.lower() == "skip"
        self.original_input_dim = input_dim
        self.original_ord_dims = ord_dims
        self.original_cat_dims = cat_dims
        self.num_inducing = int(num_inducing)

        train_X_for_input_layer = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=cat_dims,
            name=f"{self.__class__.__name__}.input_transform",
        )

        self.input_layer = DeepMixedGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=self.num_inducing,
            mean_type="linear",
            input_data=train_X_for_input_layer,
        )

        if self.use_skip:
            with torch.no_grad():
                hidden_init_dist = self.input_layer(train_X_for_input_layer)
                hidden_init = hidden_init_dist.mean
                while hidden_init.ndim > train_X_for_input_layer.ndim:
                    hidden_init = hidden_init.mean(dim=0)
                combined_input_data = torch.cat([hidden_init, train_X_for_input_layer], dim=-1)

            self.last_layer = SkipDeepMixedGPHiddenLayer(
                base_input_dims=hidden_dim,
                skip_input_dims=input_dim,
                original_ord_dims=ord_dims,
                original_cat_dims=cat_dims,
                output_dims=None if num_outputs == 1 else num_outputs,
                num_inducing=self.num_inducing,
                mean_type="constant",
                input_data=combined_input_data,
            )
        else:
            self.last_layer = DeepGPHiddenLayer(
                input_dims=hidden_dim,
                output_dims=None if num_outputs == 1 else num_outputs,
                num_inducing=self.num_inducing,
                mean_type="constant",
            )

        self._num_outputs = num_outputs
        self._setup_likelihood(likelihood=likelihood, num_outputs=num_outputs)
        self.to(train_X)

    def forward(self, inputs, apply_input_transform: bool = True):
        """混合入力に対する latent DeepGP distribution を評価する。

        Args:
            inputs: 入力 tensor、または ``(input_tensor,)`` 形式の tuple。
                カテゴリ列は整数エンコードのままである必要がある。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。

        Returns:
            最終 DeepGP layer から得られる latent GPyTorch distribution。
        """
        x = self._unwrap_inputs(inputs)
        x = self._apply_input_transform(x, apply_input_transform=apply_input_transform)

        original_input = x
        h = self.input_layer(x)

        if isinstance(self.last_layer, SkipDeepMixedGPHiddenLayer):
            return self.last_layer(h, original_input=original_input)
        return self.last_layer(h)
