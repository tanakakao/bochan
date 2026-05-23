"""Deep Gaussian Process による 2 値分類モデル群。

このモジュールでは、連続入力用および連続値・カテゴリ値混合入力用の
2 値分類 DeepGP モデルを提供します。BoTorch 形式のモデルメソッドを意識した
設計になっています。

公開 ``posterior`` メソッドは、分類用獲得関数で使いやすいように
``p(y=1 | x)`` に対する ``SimpleBernoulliPosterior`` を返します。
一方、latent 関数 ``f(x)`` に基づく獲得関数を使う場合は、
``latent_posterior`` が ``GPyTorchPosterior`` を返します。

公開クラス:
    ClassificationDeepGPModel: 連続入力向け 2 値分類 DeepGP モデル。
    ClassificationMixedDeepGPModel: 混合入力向け 2 値分類 DeepGP モデル。

使用例:
    >>> model = ClassificationDeepGPModel(train_X, train_Y)
    >>> mll = model.make_mll()
    >>> latent_dist = model(train_X)
    >>> loss = -mll(latent_dist, model.train_targets)
    >>> prob = model.predict_proba(test_X)

Notes:
    - 目的変数は 0/1 にエンコードされた 2 値ラベルを想定する。
    - ``forward`` は変分学習用の latent distribution を返す。
    - ``posterior`` は latent 正規分布ではなく、クラス 1 確率を返す。
    - latent 関数上で定義された獲得関数を使う場合は、
      ``latent_posterior`` / ``posterior_f`` を使う。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP
from gpytorch.settings import fast_pred_var

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices

from bochan.posteriors.bernoulli import SimpleBernoulliPosterior
from bochan.models.components.layers import (
    DeepGPHiddenLayer,
    DeepMixedGPHiddenLayer,
    SkipDeepGPHiddenLayer,
    SkipDeepMixedGPHiddenLayer,
)


# ============================================================
# ヘルパー
# ============================================================


def _to_device_dtype_transform(
    input_transform: Optional[InputTransform],
    X: Tensor,
) -> Optional[InputTransform]:
    """input_transform を X の device / dtype に合わせる。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        input_transform = input_transform.to(X)
    return input_transform


def _expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
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
    学習データで input_transform を初期化する。

    train_inputs は raw-space のまま保持する。InputPerturbation を使う場合は
    transform_on_train=False を想定する。
    """
    if input_transform is None:
        return X

    if hasattr(input_transform, "train"):
        input_transform.train()
    X_tf = input_transform(X)
    if isinstance(X_tf, tuple):
        X_tf = X_tf[0]
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
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    if isinstance(X_tf, tuple):
        X_tf = X_tf[0]
    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _reduce_deepgp_tensor(tensor: Tensor, X: Tensor) -> Tensor:
    """DeepGP の先頭 sample 次元を平均化して X.shape[:-1] に揃える。

    DeepGP / likelihood の実装によっては、
        sample_shape x batch_shape x q
        sample_shape x batch_shape x q x 1
    のどちらもあり得る。末尾の output singleton は q 次元ではないため、
    先に落としてから leading sample dims を平均する。
    """
    target_shape = torch.Size(X.shape[:-1])
    out = tensor

    # output singleton だけを落とす。q=1 の q 次元は X.shape[:-1] に含まれるので落とさない。
    if out.ndim >= len(target_shape) + 1 and out.shape[-1] == 1:
        if torch.Size(out.shape[-len(target_shape)-1:-1]) == target_shape:
            out = out.squeeze(-1)

    if out.shape == target_shape:
        return out

    # leading DeepGP sample dims を平均する。
    while out.ndim > len(target_shape):
        out = out.mean(dim=0)
        if out.shape == target_shape:
            return out

    if out.shape == target_shape:
        return out

    if out.numel() == int(torch.tensor(target_shape).prod().item()):
        return out.reshape(target_shape)

    raise RuntimeError(
        "Could not reduce DeepGP tensor to candidate shape. "
        f"tensor.shape={tuple(tensor.shape)}, target_shape={tuple(target_shape)}, "
        f"X.shape={tuple(X.shape)}."
    )


def _clone_train_inputs(inputs: Union[Tensor, tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    return tuple(x.detach().clone() for x in inputs)


def _prepare_binary_targets(train_Y: Tensor, train_X: Tensor) -> Tensor:
    if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
        train_Y = train_Y.squeeze(-1)
    return train_Y.to(device=train_X.device, dtype=train_X.dtype).contiguous()


# ============================================================
# 基底モデル
# ============================================================


class _BaseDeepGPBinaryClassificationModel(DeepGP, GPyTorchModel):
    """
    DeepGP binary classification 用の共通基底クラス。

    - forward(): 学習用 latent DeepGP distribution
    - posterior(): p(y=1|x) の SimpleBernoulliPosterior
    - latent_posterior(): latent f の GPyTorchPosterior
    """

    def __init__(self) -> None:
        super().__init__()

    def _set_transformed_inputs(self) -> None:
        """train_inputs を raw-space のまま保持するため、自動変換を無効化する。"""
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        return _apply_input_transform_for_eval(
            X,
            getattr(self, "input_transform", None),
            cat_dims=getattr(self, "cat_dims", None),
        )

    @staticmethod
    def _unwrap_inputs(inputs) -> Tensor:
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

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> SimpleBernoulliPosterior:
        """Bernoulli 確率 posterior ``p(y=1 | x)`` を返す。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。
            output_indices: 返したい出力次元の指定。単一出力の 2 値分類モデルのため未対応。
            observation_noise: Bernoulli 分類では使用しない。BoTorch model API 互換性のために受け取る。
            posterior_transform: 任意の posterior transform。
            **kwargs: 互換性のために受け取る未使用の追加引数。

        Returns:
            ``SimpleBernoulliPosterior``。``mean`` はクラス 1 確率、
            ``variance`` は ``p * (1 - p)`` となる。

        Raises:
            NotImplementedError: ``output_indices`` が指定された場合。
        """
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )
        _ = observation_noise
        _ = kwargs

        self.eval()
        self.likelihood.eval()

        X = self._unwrap_inputs(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)

        with fast_pred_var():
            latent_dist = self.forward(X_tf, apply_input_transform=False)
            pred_dist = self.likelihood(latent_dist)

        p = _reduce_deepgp_tensor(pred_dist.mean, X_tf)
        var = (p * (1.0 - p)).clamp_min(0.0)

        if p.ndim == X_tf.ndim - 1:
            p = p.unsqueeze(-1)
            var = var.unsqueeze(-1)

        posterior = SimpleBernoulliPosterior(mean=p, variance=var)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def probability_posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> SimpleBernoulliPosterior:
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
        **kwargs,
    ) -> GPyTorchPosterior:
        """latent 関数 ``f(x)`` の posterior を返す。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。
            posterior_transform: 任意の BoTorch posterior transform。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。
            **kwargs: 互換性のために受け取る未使用の追加引数。

        Returns:
            latent 関数に対する ``GPyTorchPosterior``。DeepGP 由来の余分な
            sample 次元は、BoTorch 形式の posterior shape に合わせるため平均化する。
        """
        _ = kwargs
        self.eval()

        X = self._unwrap_inputs(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=apply_input_transform)

        with fast_pred_var():
            latent_dist = self.forward(X_tf, apply_input_transform=False)
            mvn = self._average_deepgp_latent_distribution(latent_dist, X_tf)
            posterior = GPyTorchPosterior(mvn)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def posterior_latent(self, X, **kwargs):
        return self.latent_posterior(X, **kwargs)

    def posterior_f(self, X, **kwargs):
        return self.latent_posterior(X, **kwargs)

    def predict_proba(self, X: Tensor) -> Tensor:
        """クラス 1 確率を予測する。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。

        Returns:
            末尾に出力次元 ``1`` を持つ確率 tensor。
        """
        return self.posterior(X).mean

    def set_train_data(
        self,
        inputs: Optional[Union[Tensor, tuple[Tensor, ...]]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        _ = strict
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
            self.train_inputs_raw = _clone_train_inputs(inputs)

        if targets is not None:
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            targets = targets.to(device=self.train_inputs[0].device, dtype=self.train_inputs[0].dtype)
            self.train_targets = targets.contiguous()
            self._train_targets = self.train_targets

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def make_mll(self, beta: float = 1.0) -> DeepApproximateMLL:
        """分類 DeepGP の学習に使う marginal log likelihood を作成する。

        Args:
            beta: ``VariationalELBO`` に渡す KL divergence の重み。

        Returns:
            ``VariationalELBO`` を ``DeepApproximateMLL`` で包んだ MLL。
        """
        if self.train_inputs[0].shape[-2] != self.train_targets.shape[0]:
            raise RuntimeError(
                "train_inputs and train_targets have inconsistent data sizes. "
                f"train_inputs[0].shape={tuple(self.train_inputs[0].shape)}, "
                f"train_targets.shape={tuple(self.train_targets.shape)}. "
                "For InputPerturbation, use transform_on_train=False or keep train inputs raw."
            )

        base_mll = VariationalELBO(
            likelihood=self.likelihood,
            model=self,
            num_data=self.train_inputs[0].shape[-2],
            beta=float(beta),
        )
        return DeepApproximateMLL(base_mll)

    def _average_deepgp_latent_distribution(self, latent_dist, X: Tensor) -> MultivariateNormal:
        """DeepGP の余分な sample 次元を平均し、BoTorch posterior 用 MVN に揃える。"""
        mean = _reduce_deepgp_tensor(latent_dist.mean, X)
        covar = latent_dist.covariance_matrix

        target_covar_shape = torch.Size(X.shape[:-2]) + torch.Size([X.shape[-2], X.shape[-2]])

        if covar.shape == target_covar_shape:
            return MultivariateNormal(mean, covar)

        while covar.ndim > len(target_covar_shape):
            covar = covar.mean(dim=0)
            if covar.shape == target_covar_shape:
                return MultivariateNormal(mean, covar)

        if covar.shape == target_covar_shape:
            return MultivariateNormal(mean, covar)

        if covar.numel() == int(torch.tensor(target_covar_shape).prod().item()):
            covar = covar.reshape(target_covar_shape)
            return MultivariateNormal(mean, covar)

        # Fallback to diagonal covariance from latent variance if covariance shape is incompatible.
        var = _reduce_deepgp_tensor(latent_dist.variance, X).clamp_min(1e-12)
        covar = torch.diag_embed(var)
        return MultivariateNormal(mean, covar)


# ============================================================
# 連続入力モデル
# ============================================================


class BinaryClassificationDeepGPModel(_BaseDeepGPBinaryClassificationModel):
    """連続入力向けの 2 値分類 DeepGP モデル。

    latent DeepGP と ``BernoulliLikelihood`` を組み合わせたモデルです。
    公開 ``posterior`` メソッドは、``SimpleBernoulliPosterior`` として
    クラス 1 確率を返します。latent GP posterior は
    ``latent_posterior``、``posterior_latent``、または ``posterior_f`` から取得できます。

    Args:
        train_X: raw-space の学習入力。形状は ``batch_shape x n x d`` または ``n x d``。
        train_Y: 2 値ラベル。形状は ``n`` または ``n x 1``。値は 0/1 エンコードを想定する。
        likelihood: 任意の ``BernoulliLikelihood``。省略時はデフォルトの Bernoulli likelihood を作成する。
        input_transform: 任意の BoTorch input transform。raw-space 入力を
            ``train_inputs`` と ``train_inputs_raw`` に保持したまま、
            ``forward`` / ``posterior`` 内で適用する。
        list_hidden_dims: hidden layer の出力次元リスト。デフォルトは ``[16]``。
        model_type: モデル構造の指定。``"DEFAULT"`` では通常の層状 DeepGP、
            ``"skip"`` では元入力を skip-compatible layer に再注入する。
        num_inducing: hidden layer の inducing point 数。
        num_inducing_last: 最終 latent layer の inducing point 数。
            ``None`` の場合は ``num_inducing`` を使う。

    Attributes:
        train_inputs: ``(train_X,)`` として保持する raw-space 学習入力。
        train_inputs_raw: raw-space 学習入力の detached clone。
        train_targets: 1 次元 tensor として保持する 2 値ラベル。
        likelihood: latent ``f`` を確率に写像する Bernoulli likelihood。
        hidden_layers: DeepGP hidden layers。
        last_layer: スカラー latent 出力を返す最終 DeepGP layer。

    Notes:
        学習用 MLL の作成には ``make_mll`` を使ってください。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        list_hidden_dims: Optional[Sequence[int]] = None,
        model_type: str = "DEFAULT",
        num_inducing: int = 128,
    ) -> None:
        super().__init__()

        train_Y = _prepare_binary_targets(train_Y, train_X)

        self.train_inputs = (train_X,)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets = train_Y
        self._train_targets = train_Y
        self.input_transform = _to_device_dtype_transform(input_transform, train_X)
        _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )
        self.likelihood = likelihood or BernoulliLikelihood()

        hidden_dims = list(list_hidden_dims) if list_hidden_dims is not None else [16]
        if len(hidden_dims) == 0:
            raise ValueError("list_hidden_dims には少なくとも1つの要素が必要です。")

        self.use_skip = model_type.lower() == "skip"
        self.original_input_dim = train_X.shape[-1]
        self.num_inducing = int(num_inducing)
        self.hidden_layers = torch.nn.ModuleList()

        current_dim = train_X.shape[-1]
        for hidden_dim in hidden_dims:
            if self.use_skip:
                self.hidden_layers.append(
                    SkipDeepGPHiddenLayer(
                        base_input_dims=current_dim,
                        skip_input_dims=self.original_input_dim,
                        output_dims=hidden_dim,
                        num_inducing=self.num_inducing,
                        mean_type="linear",
                    )
                )
            else:
                self.hidden_layers.append(
                    DeepGPHiddenLayer(
                        input_dims=current_dim,
                        output_dims=hidden_dim,
                        num_inducing=self.num_inducing,
                        mean_type="linear",
                    )
                )
            current_dim = hidden_dim

        if self.use_skip:
            self.last_layer = SkipDeepGPHiddenLayer(
                base_input_dims=current_dim,
                skip_input_dims=self.original_input_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
            )
        else:
            self.last_layer = DeepGPHiddenLayer(
                input_dims=current_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
            )

        self.to(train_X)

    def forward(self, X: Tensor, apply_input_transform: bool = True):
        """latent DeepGP distribution を評価する。

        Args:
            X: 入力 tensor、または ``(input_tensor,)`` 形式の tuple。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。

        Returns:
            変分学習に使う latent GPyTorch distribution。
        """
        X = self._unwrap_inputs(X)
        X = self._apply_input_transform(X, apply_input_transform=apply_input_transform)

        original_input = X
        h = X
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


class BinaryClassificationMixedDeepGPModel(_BaseDeepGPBinaryClassificationModel):
    """混合入力向けの 2 値分類 DeepGP モデル。

    連続列とカテゴリ列を同時に含む入力を扱います。最初の layer には
    mixed-aware な DeepGP layer を使います。カテゴリ列は整数エンコードを想定し、
    ``input_transform`` によって誤って変更されないようにチェックします。

    Args:
        train_X: raw-space の学習入力。形状は ``batch_shape x n x d`` または
            ``n x d``。カテゴリ列は整数エンコードされている必要がある。
        train_Y: 2 値ラベル。形状は ``n`` または ``n x 1``。値は 0/1 エンコードを想定する。
        cat_dims: ``train_X`` におけるカテゴリ列の index。負の index は
            BoTorch の ``normalize_indices`` で正規化される。
        likelihood: 任意の ``BernoulliLikelihood``。
        input_transform: 任意の input transform。独自 transform はカテゴリ列を
            変更してはいけない。正規化する場合は連続列のみに適用する。
        hidden_dim: 最初の mixed hidden layer の出力次元。
        model_type: モデル構造の指定。``"DEFAULT"`` では通常の最終 layer、
            ``"skip"`` では元入力を最終 mixed layer に再注入する。
        num_inducing: 最初の mixed layer の inducing point 数。
        num_inducing_last: 最終 latent layer の inducing point 数。
            ``None`` の場合は ``num_inducing`` を使う。

    Attributes:
        cat_dims: 正規化済みのカテゴリ列 index。
        ord_dims: 非カテゴリ列 index。
        input_layer: mixed-aware な DeepGP 入力 layer。
        last_layer: スカラー latent 出力を返す最終 layer。

    Raises:
        ValueError: ``cat_dims`` が空の場合、または入力変換がカテゴリ列を変更した場合。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        hidden_dim: int = 16,
        model_type: str = "DEFAULT",
        num_inducing: int = 128,
        num_inducing_last: Optional[int] = None,
    ) -> None:
        super().__init__()

        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        train_Y = _prepare_binary_targets(train_Y, train_X)

        self.train_inputs = (train_X,)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets = train_Y
        self._train_targets = train_Y
        self.input_transform = _to_device_dtype_transform(input_transform, train_X)
        self.likelihood = likelihood or BernoulliLikelihood()

        d = train_X.shape[-1]
        cat_dims = list(normalize_indices(indices=cat_dims, d=d))
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims
        self.use_skip = model_type.lower() == "skip"
        self.original_input_dim = d
        self.original_ord_dims = ord_dims
        self.original_cat_dims = cat_dims
        self.num_inducing = int(num_inducing)
        self.num_inducing_last = int(num_inducing_last or num_inducing)

        train_X_for_input_layer = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=cat_dims,
            name=f"{self.__class__.__name__}.input_transform",
        )

        self.input_layer = DeepMixedGPHiddenLayer(
            input_dims=d,
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
                skip_input_dims=d,
                original_ord_dims=ord_dims,
                original_cat_dims=cat_dims,
                output_dims=None,
                num_inducing=self.num_inducing_last,
                mean_type="constant",
                input_data=combined_input_data,
            )
        else:
            self.last_layer = DeepGPHiddenLayer(
                input_dims=hidden_dim,
                output_dims=None,
                num_inducing=self.num_inducing_last,
                mean_type="constant",
            )

        self.to(train_X)

    def forward(self, X: Tensor, apply_input_transform: bool = True):
        """混合入力に対する latent DeepGP distribution を評価する。

        Args:
            X: 入力 tensor、または ``(input_tensor,)`` 形式の tuple。
                カテゴリ列は整数エンコードされている必要がある。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。

        Returns:
            変分学習に使う latent GPyTorch distribution。
        """
        X = self._unwrap_inputs(X)
        X = self._apply_input_transform(X, apply_input_transform=apply_input_transform)

        original_input = X
        h = self.input_layer(X)

        if isinstance(self.last_layer, SkipDeepMixedGPHiddenLayer):
            return self.last_layer(h, original_input=original_input)
        return self.last_layer(h)
