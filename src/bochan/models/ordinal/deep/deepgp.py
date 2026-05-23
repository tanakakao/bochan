"""Deep Gaussian Process による順序回帰モデル群。

このモジュールでは、連続入力用および連続値・カテゴリ値混合入力用の
順序回帰 DeepGP モデルを提供します。latent DeepGP と
``OrdinalLogitLikelihood`` を組み合わせて使います。

公開 ``posterior`` メソッドは、BoTorch 互換性のために latent 関数の
``GPyTorchPosterior`` を返します。一方、順序クラス確率や期待効用は
``class_probs`` と ``expected_utility`` から取得します。

公開クラス:
    OrdinalDeepGPModel: 連続入力向け順序回帰 DeepGP モデル。
    OrdinalMixedDeepGPModel: 混合入力向け順序回帰 DeepGP モデル。

使用例:
    >>> model = OrdinalDeepGPModel(train_X, train_Y, num_classes=3)
    >>> fit_true_deep_ordinal_gp(model, num_epochs=200)
    >>> probs = model.class_probs(test_X)
    >>> pred = model.predict_class(test_X)

Notes:
    - 目的変数は ``0, ..., num_classes - 1`` の整数クラスラベルを想定する。
    - ``posterior`` はクラス確率ではなく latent posterior を返す。
    - ``class_probs`` は ``OrdinalLogitLikelihood`` を使って latent posterior を
      順序クラス確率に変換する。
    - ``InputPerturbation`` は、学習入力を展開しない設定であれば利用できる。
      通常は ``transform_on_train=False`` を使う。
"""

from __future__ import annotations

import copy
import inspect
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from linear_operator.operators import DiagLinearOperator

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices

# あなたの環境に合わせて import path は調整してください
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood

# 既存の layer 実装を参照する
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
    """input_transform を X の device / dtype に揃える。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        input_transform = input_transform.to(X)
    return input_transform



def _clone_input_transform(
    input_transform: Optional[InputTransform],
    X: Optional[Tensor] = None,
) -> Optional[InputTransform]:
    """condition_on_observations 用に input_transform を複製する。"""
    if input_transform is None:
        return None
    cloned = copy.deepcopy(input_transform)
    if X is not None and hasattr(cloned, "to"):
        cloned = cloned.to(X)
    return cloned


def _expand_raw_X_to_match_transformed_q(
    X: Tensor,
    X_tf: Tensor,
) -> Tensor:
    """
    InputPerturbation 後の X_tf と比較できるように raw X の q 次元を展開する。

    想定:
        X.shape    = (*batch, q, d)
        X_tf.shape = (*batch, q_like, d)

    通常:
        q_like = q

    InputPerturbation:
        q_like = q * n_w
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
            n_w = q_like // q
            return X.repeat_interleave(n_w, dim=-2)

    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)

    return X


def _check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """
    mixed model 用に input_transform がカテゴリ列を変更していないか確認する。

    InputPerturbation では q -> q*n_w に展開され得るため、
    raw X 側も q 次元を repeat してから比較する。
    """
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = _expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}. "
            "This usually means input_transform changed the batch/q shape in a "
            "non-repeatable way."
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
    classification model と同じ規約で学習用 X_tf を作る。

    手順:
        1. input_transform.train()
        2. X_tf = input_transform(X)
        3. input_transform.eval()

    これにより、InputPerturbation は学習時には通常 q*n_w 展開されず、
    posterior / acquisition 評価時だけ展開される。
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
            "This will not match ordinal train_Y. "
            "For InputPerturbation, ensure transform_on_train=False."
        )

    _check_categorical_columns_unchanged(
        X=X,
        X_tf=X_tf,
        cat_dims=cat_dims,
    )
    return X_tf


def _apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    posterior / acquisition 評価用の input_transform。

    eval mode の InputPerturbation では q -> q*n_w 展開を許す。
    mixed の場合はカテゴリ列が変化していないか確認する。
    """
    if input_transform is None:
        return X

    X_tf = input_transform(X)
    _check_categorical_columns_unchanged(
        X=X,
        X_tf=X_tf,
        cat_dims=cat_dims,
    )
    return X_tf


def _clone_train_inputs(
    inputs: Union[Tensor, tuple[Tensor, ...]]
) -> tuple[Tensor, ...]:
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    return tuple(x.detach().clone() for x in inputs)

def _clone_tensor_tuple(
    inputs: Union[Tensor, tuple[Tensor, ...]],
) -> tuple[Tensor, ...]:
    """生の train inputs を参照共有せず保持する。"""
    if torch.is_tensor(inputs):
        inputs = (inputs,)
    return tuple(x.detach().clone() for x in inputs)

def _prepare_ordinal_targets(train_Y: Tensor, train_X: Tensor) -> Tensor:
    """順序ラベルを [n] の long tensor に整形する。"""
    if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
        train_Y = train_Y.squeeze(-1)
    return train_Y.to(device=train_X.device, dtype=torch.long).contiguous()



def _reduce_deepgp_tensor(tensor: Tensor, X: Tensor) -> Tensor:
    """DeepGP の先頭 sample 次元を平均化して X[:-1] に対応させる。"""
    expected_ndim = X.ndim - 1
    while tensor.ndim > expected_ndim:
        tensor = tensor.mean(dim=0)
    return tensor



def _moment_match_latent_distribution(
    latent_dist: MultivariateNormal,
    X: Tensor,
) -> MultivariateNormal:
    """
    DeepGP の latent mixture を平均・分散で近似した対角 MVN に落とす。
    """
    mean = _reduce_deepgp_tensor(latent_dist.mean, X)
    var = _reduce_deepgp_tensor(latent_dist.variance, X).clamp_min(1e-10)
    return MultivariateNormal(mean, DiagLinearOperator(var))



def _normalize_cat_dims(cat_dims: Sequence[int], d: int) -> list[int]:
    return list(normalize_indices(indices=cat_dims, d=d))



def _validate_categorical_values(
    X: Tensor,
    cat_dims: Sequence[int],
    category_counts: Dict[int, int],
) -> None:
    """mixed 入力でカテゴリ列が整数エンコード (0..K-1) か確認する。"""
    d = X.shape[-1]
    norm_cat_dims = _normalize_cat_dims(cat_dims, d)
    for j in norm_cat_dims:
        if j not in category_counts:
            raise ValueError(f"category_counts must contain key {j}")
        n_cat = int(category_counts[j])
        vals = X[..., j]
        if not torch.allclose(vals, vals.round()):
            raise ValueError(
                f"Categorical column {j} must be integer-coded (0..K-1)."
            )
        if vals.min().item() < 0 or vals.max().item() > n_cat - 1:
            raise ValueError(
                f"Categorical column {j} must be in [0, {n_cat - 1}], "
                f"got min={vals.min().item()}, max={vals.max().item()}"
            )



def _default_hidden_dims(list_hidden_dims: Optional[Sequence[int]]) -> list[int]:
    hidden_dims = list(list_hidden_dims) if list_hidden_dims is not None else [16]
    if len(hidden_dims) == 0:
        raise ValueError("list_hidden_dims must contain at least one element.")
    return hidden_dims



def _filter_kwargs_for_callable(cls, kwargs: dict) -> dict:
    """外部 layers の版差を吸収するため、受け取れる引数だけ渡す。"""
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}



def _make_layer(cls, **kwargs):
    """版差を吸収して layer を構築する。"""
    return cls(**_filter_kwargs_for_callable(cls, kwargs))


# ============================================================
# 基底モデル
# ============================================================


class _BaseDeepOrdinalGPModel(DeepGP, GPyTorchModel):
    """
    DeepGP ordinal model の共通基底クラス。

    - forward(): 学習用の latent distribution を返す
    - posterior(): latent GP posterior を BoTorch 互換で返す
    - class_probs(): ordinal class probability を返す
    """

    _num_outputs = 1

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _unwrap_inputs(inputs) -> Tensor:
        if isinstance(inputs, tuple):
            return inputs[0]
        return inputs

    def _set_transformed_inputs(self) -> None:
        """
        BoTorch Model.eval() が呼ぶ transformed input 自動更新を無効化する。

        この DeepGP ordinal wrapper では、
            train_inputs_raw[0]: raw-space X
            train_inputs[0]:     raw-space X for fitting dataloader
            forward():           必要時に input_transform を適用
        という管理にする。

        BoTorch 標準の eval-time transform が走ると、InputPerturbation により
        train_inputs が n_w 倍に展開され、train_targets と不整合になる可能性がある。
        """
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """
        posterior / acquisition 評価用 transform。

        eval mode では InputPerturbation により q -> q*n_w 展開されてよい。
        mixed model では cat_dims を使ってカテゴリ列保持を確認する。
        """
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

    def latent_distribution(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> MultivariateNormal:
        X = self._unwrap_inputs(X)
        X = self._apply_input_transform(X, apply_input_transform=apply_input_transform)
        return self.forward(X, apply_input_transform=False)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> GPyTorchPosterior:
        """BoTorch の獲得関数で使う latent posterior を返す。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。
            output_indices: 返したい出力次元の指定。単一出力の latent ordinal モデルのため未対応。
            observation_noise: 順序回帰の latent posterior では使用しない。
                BoTorch model API 互換性のために受け取る。
            posterior_transform: 任意の BoTorch posterior transform。
            **kwargs: 互換性のために受け取る未使用の追加引数。

        Returns:
            latent 関数に対する ``GPyTorchPosterior``。DeepGP の mixture は、
            平均と分散の moment matching により対角多変量正規分布として近似する。

        Raises:
            NotImplementedError: ``output_indices`` が指定された場合。
        """
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )
        _ = observation_noise

        self.eval()
        self.likelihood.eval()

        X = self._unwrap_inputs(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)
        latent_dist = self.forward(X_tf, apply_input_transform=False)
        approx_dist = _moment_match_latent_distribution(latent_dist, X_tf)
        posterior = GPyTorchPosterior(approx_dist)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    @property
    def ordinal_likelihood(self) -> OrdinalLogitLikelihood:
        return self.likelihood

    @torch.no_grad()
    def class_probs(self, X: Tensor) -> Tensor:
        """順序クラス確率を予測する。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。

        Returns:
            クラス確率を格納した tensor。形状は
            ``batch_shape x q x num_classes``。
        """
        X = self._unwrap_inputs(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)
        latent_dist = self.forward(X_tf, apply_input_transform=False)
        approx_dist = _moment_match_latent_distribution(latent_dist, X_tf)
        return self.ordinal_likelihood.marginal_class_probs(approx_dist)

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        """最も確率の高い順序クラスを予測する。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。

        Returns:
            予測クラス index を格納した long tensor。
        """
        return self.class_probs(X).argmax(dim=-1)

    @torch.no_grad()
    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        """順序予測分布の下で期待効用を計算する。

        Args:
            X: raw-space の入力。形状は ``batch_shape x q x d`` または ``q x d``。
            utilities: 各クラスに対応する効用値。形状は ``num_classes``。

        Returns:
            期待効用 tensor。形状は ``batch_shape x q``。
        """
        X = self._unwrap_inputs(X)
        X_tf = self._apply_input_transform(X, apply_input_transform=True)
        latent_dist = self.forward(X_tf, apply_input_transform=False)
        approx_dist = _moment_match_latent_distribution(latent_dist, X_tf)
        return self.ordinal_likelihood.marginal_expected_utility(approx_dist, utilities)

    def set_train_data(
        self,
        inputs: Optional[Union[Tensor, tuple[Tensor, ...]]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        """
        明示的な train data 更新用。

        inputs は raw-space X を想定する。
        DeepGP 本体は forward 内で input_transform を適用するため、
        dataloader 用の train_inputs も raw-space のまま保持する。
        """
        _ = strict

        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)

            X_raw = inputs[0].to(
                device=self.train_inputs_raw[0].device,
                dtype=self.train_inputs_raw[0].dtype,
            )
            self.train_inputs = (X_raw,)
            self.train_inputs_raw = (X_raw.detach().clone(),)

        if targets is not None:
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            targets = targets.to(
                device=self.train_inputs[0].device,
                dtype=torch.long,
            ).contiguous()
            self.train_targets = targets
            self._train_targets = targets

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def make_mll(self, beta: Optional[float] = None) -> DeepApproximateMLL:
        """順序回帰 DeepGP の学習に使う marginal log likelihood を作成する。

        Args:
            beta: 任意の KL divergence 重み。``None`` の場合は ``self.beta`` を使う。

        Returns:
            ``VariationalELBO`` を ``DeepApproximateMLL`` で包んだ MLL。
        """
        beta = self.beta if beta is None else float(beta)
        base_mll = VariationalELBO(
            likelihood=self.likelihood,
            model=self,
            num_data=self.train_inputs[0].shape[-2],
            beta=beta,
        )
        return DeepApproximateMLL(base_mll)

    def _get_rebuild_kwargs(self) -> dict:
        raise NotImplementedError

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
    ):
        """追加の順序ラベル観測で条件付けした新しいモデルを返す。

        Args:
            X: 追加する raw-space 入力。形状は ``n x d``。
            Y: 追加する順序ラベル。形状は ``n`` または ``n x 1``。
            refit: 追加データを結合した後、返すモデルを再学習するかどうか。
            num_steps: 再学習 epoch 数。``None`` の場合は ``self.conditioning_steps`` を使う。
            lr: 再学習時の学習率。``None`` の場合は ``self.conditioning_lr``、
                それも ``None`` なら ``self.lr`` を使う。
            batch_size: 再学習時の batch size。``None`` の場合は
                ``self.conditioning_batch_size``、それも ``None`` なら ``self.batch_size`` を使う。
            verbose: 再学習の進捗を表示するかどうか。
            **kwargs: 互換性のための追加引数。順序回帰モデルでは ``noise`` は未対応。

        Returns:
            学習データを追加した新しいモデルインスタンス。

        Raises:
            NotImplementedError: ``noise`` に ``None`` 以外が指定された場合。
        """
        if "noise" in kwargs and kwargs["noise"] is not None:
            raise NotImplementedError("noise is not supported for DeepGP ordinal models.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        # raw-space を基準に追加する。
        # train_inputs[0] が将来 transformed-space に変わっても安全にするため、
        # train_inputs_raw[0] を優先する。
        base_train_X = (
            self.train_inputs_raw[0]
            if hasattr(self, "train_inputs_raw")
            else self.train_inputs[0]
        )
        new_train_X = torch.cat([base_train_X, X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            **self._get_rebuild_kwargs(),
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=True)

        if refit:
            if num_steps is None:
                num_steps = int(self.conditioning_steps)
            if lr is None:
                lr = self.conditioning_lr if self.conditioning_lr is not None else self.lr
            if batch_size is None:
                batch_size = (
                    self.conditioning_batch_size
                    if self.conditioning_batch_size is not None
                    else self.batch_size
                )
            fit_true_deep_ordinal_gp(
                new_model,
                num_epochs=num_steps,
                lr=lr,
                batch_size=batch_size,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


# ============================================================
# 連続入力モデル
# ============================================================


class OrdinalDeepGPModel(_BaseDeepOrdinalGPModel):
    """連続入力向けの順序回帰 DeepGP モデル。

    latent DeepGP と ``OrdinalLogitLikelihood`` を組み合わせたモデルです。
    評価・順位・重症度・離散化した品質クラスなど、順序性を持つカテゴリ目的変数を
    想定しています。``posterior`` は latent 関数の posterior を返し、
    順序回帰固有の予測は ``class_probs`` と ``expected_utility`` から取得します。

    Args:
        train_X: raw-space の学習入力。形状は ``batch_shape x n x d`` または ``n x d``。
        train_Y: 順序ラベル。形状は ``n`` または ``n x 1``。
            ラベルは ``0, ..., num_classes - 1`` の整数エンコードである必要がある。
        num_classes: 順序クラス数。
        list_hidden_dims: hidden layer の出力次元リスト。デフォルトは ``[16]``。
        num_inducing: 各 DeepGP layer の inducing point 数。
        learn_inducing_locations: inducing point の位置を学習するかどうか。
        lr: ``fit_true_deep_ordinal_gp`` および
            ``condition_on_observations(refit=True)`` で使うデフォルト学習率。
        num_epochs: ``fit_true_deep_ordinal_gp`` で使うデフォルト epoch 数。
        batch_size: デフォルト mini-batch size。``None`` の場合は full-batch 学習を行う。
        beta: ``VariationalELBO`` に渡す KL divergence の重み。
        model_type: モデル構造の指定。``"DEFAULT"`` では通常の層状 DeepGP、
            ``"skip"`` では元入力を skip-compatible layer に再注入する。
        fix_first_cutpoint: likelihood 実装において最初の ordinal cutpoint を固定するかどうか。
        init_gap: ordinal cutpoint 間隔の初期値。
        eps: ordinal likelihood で使う数値安定化定数。
        verbose: デフォルト学習ヘルパーで進捗を表示するかどうか。
        conditioning_steps: ``condition_on_observations(refit=True)`` で使う再学習 step 数。
        conditioning_lr: 条件付け時の任意の学習率上書き。
        conditioning_batch_size: 条件付け時の任意の batch size 上書き。
        input_transform: 任意の BoTorch input transform。学習データは
            ``train_inputs`` / ``train_inputs_raw`` に raw-space のまま保持する。
        likelihood: 任意の ordinal likelihood。省略時は ``OrdinalLogitLikelihood`` を作成する。

    Attributes:
        train_inputs: ``(train_X,)`` として保持する raw-space 学習入力。
        train_inputs_raw: raw-space 学習入力の detached clone。
        train_targets: 1 次元 long tensor として保持する順序ラベル。
        likelihood: latent 値をクラス確率に写像する ordinal likelihood。
        num_classes: 順序クラス数。
        list_hidden_dims: hidden layer の次元リスト。

    Notes:
        ``posterior`` は設計上 latent-space の posterior を返します。
        クラス確率には ``class_probs``、効用重み付きのスカラー予測には
        ``expected_utility`` を使ってください。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        list_hidden_dims: Optional[Sequence[int]] = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.01,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        beta: float = 1.0,
        model_type: str = "DEFAULT",
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood: Optional[_OneDimensionalLikelihood] = None,
    ) -> None:
        super().__init__()

        train_Y = _prepare_ordinal_targets(train_Y, train_X)
        input_transform = _to_device_dtype_transform(input_transform, train_X)

        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self._train_targets = train_Y
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.input_transform = input_transform
        self.likelihood = likelihood or OrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        )

        self.num_classes = int(num_classes)
        self.list_hidden_dims = list(_default_hidden_dims(list_hidden_dims))
        self.num_inducing = int(num_inducing)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.beta = float(beta)
        self.model_type = str(model_type)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        train_X_tf = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name="OrdinalDeepGPModel.input_transform",
        )
        hidden_dims = list(self.list_hidden_dims)
        self.use_skip = self.model_type.lower() == "skip"
        self.original_input_dim = train_X.shape[-1]

        self.hidden_layers = nn.ModuleList()
        current_dim = train_X.shape[-1]
        for i, hidden_dim in enumerate(hidden_dims):
            if self.use_skip:
                layer_input_data = (
                    torch.cat([train_X_tf, train_X_tf], dim=-1)
                    if i == 0
                    else None
                )
                layer = _make_layer(
                    SkipDeepGPHiddenLayer,
                    base_input_dims=current_dim,
                    skip_input_dims=self.original_input_dim,
                    output_dims=hidden_dim,
                    num_inducing=self.num_inducing,
                    mean_type="linear",
                    input_data=layer_input_data,
                    learn_inducing_locations=self.learn_inducing_locations,
                )
            else:
                layer = _make_layer(
                    DeepGPHiddenLayer,
                    input_dims=current_dim,
                    output_dims=hidden_dim,
                    num_inducing=self.num_inducing,
                    mean_type="linear",
                    input_data=train_X_tf if i == 0 else None,
                    learn_inducing_locations=self.learn_inducing_locations,
                )
            self.hidden_layers.append(layer)
            current_dim = hidden_dim

        if self.use_skip:
            self.last_layer = _make_layer(
                SkipDeepGPHiddenLayer,
                base_input_dims=current_dim,
                skip_input_dims=self.original_input_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )
        else:
            self.last_layer = _make_layer(
                DeepGPHiddenLayer,
                input_dims=current_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )

        self.to(train_X)

    def _get_rebuild_kwargs(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "list_hidden_dims": list(self.list_hidden_dims),
            "num_inducing": self.num_inducing,
            "learn_inducing_locations": self.learn_inducing_locations,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "beta": self.beta,
            "model_type": self.model_type,
            "fix_first_cutpoint": self.fix_first_cutpoint,
            "init_gap": self.init_gap,
            "eps": self.eps,
            "verbose": self.verbose,
            "conditioning_steps": self.conditioning_steps,
            "conditioning_lr": self.conditioning_lr,
            "conditioning_batch_size": self.conditioning_batch_size,
            "input_transform": _clone_input_transform(self.input_transform, self.train_inputs[0]),
        }

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = X.to(self.train_inputs[0])
        if X.ndim == 1:
            if self.train_inputs[0].shape[-1] != 1:
                raise ValueError(
                    f"1D X can only be used when input dim is 1, got d={self.train_inputs[0].shape[-1]}"
                )
            X = X.unsqueeze(-1)
        if X.ndim != 2:
            raise ValueError(f"Observation X must be [n, d], got shape={tuple(X.shape)}")
        if X.shape[-1] != self.train_inputs[0].shape[-1]:
            raise ValueError(
                f"X feature dim mismatch: expected {self.train_inputs[0].shape[-1]}, got {X.shape[-1]}"
            )
        return X.contiguous()

    def _canonicalize_new_Y(self, Y: Tensor, n: int) -> Tensor:
        Y = torch.as_tensor(Y, device=self.train_inputs[0].device)
        if Y.ndim == 0:
            Y = Y.view(1)
        elif Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        elif Y.ndim != 1:
            raise ValueError(f"Y must be [n] or [n, 1], got shape={tuple(Y.shape)}")
        if Y.shape[0] != n:
            raise ValueError(f"Y length mismatch: expected {n}, got {Y.shape[0]}")
        return Y.long().contiguous()

    def forward(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> MultivariateNormal:
        """latent DeepGP distribution を評価する。

        Args:
            X: 入力 tensor、または ``(input_tensor,)`` 形式の tuple。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。

        Returns:
            最終 DeepGP layer から得られる latent GPyTorch 多変量正規分布。
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


class OrdinalMixedDeepGPModel(_BaseDeepOrdinalGPModel):
    """連続値・カテゴリ値の混合入力向け順序回帰 DeepGP モデル。

    ``OrdinalDeepGPModel`` の mixed 入力版です。最初の layer は mixed-aware な
    layer で、カテゴリ列は整数エンコードされた raw 値として保持します。
    ``input_transform`` は、カテゴリ列を変更しない場合にのみ許容します。

    Args:
        train_X: raw-space の学習入力。形状は ``batch_shape x n x d`` または
            ``n x d``。カテゴリ列は整数エンコードされている必要がある。
        train_Y: 順序ラベル。形状は ``n`` または ``n x 1``。
            ラベルは ``0, ..., num_classes - 1`` の整数エンコードである必要がある。
        num_classes: 順序クラス数。
        cat_dims: ``train_X`` におけるカテゴリ列の index。
        category_counts: カテゴリ列 index からカテゴリ数への任意の対応表。
            省略時は各カテゴリ列について ``max(train_X[..., j]) + 1`` として推定する。
        list_hidden_dims: hidden layer の出力次元リスト。最初の値は mixed input layer で使う。
            デフォルトは ``[16]``。
        num_inducing: 各 DeepGP layer の inducing point 数。
        learn_inducing_locations: inducing point の位置を学習するかどうか。
        lr: 学習ヘルパーで使うデフォルト学習率。
        num_epochs: 学習ヘルパーで使うデフォルト epoch 数。
        batch_size: デフォルト mini-batch size。``None`` の場合は full-batch 学習を行う。
        beta: ``VariationalELBO`` に渡す KL divergence の重み。
        model_type: モデル構造の指定。``"DEFAULT"`` では通常の最終 layer、
            ``"skip"`` では元入力を skip-compatible layer に再注入する。
        fix_first_cutpoint: likelihood 実装において最初の ordinal cutpoint を固定するかどうか。
        init_gap: ordinal cutpoint 間隔の初期値。
        eps: ordinal likelihood で使う数値安定化定数。
        verbose: デフォルト学習ヘルパーで進捗を表示するかどうか。
        conditioning_steps: ``condition_on_observations(refit=True)`` で使う再学習 step 数。
        conditioning_lr: 条件付け時の任意の学習率上書き。
        conditioning_batch_size: 条件付け時の任意の batch size 上書き。
        input_transform: 任意の BoTorch input transform。独自 transform はカテゴリ列を変更してはいけない。
        likelihood: 任意の ordinal likelihood。省略時は ``OrdinalLogitLikelihood`` を作成する。

    Attributes:
        cat_dims: 正規化済みのカテゴリ列 index。
        ord_dims: 非カテゴリ列 index。
        category_counts: 各カテゴリ列のカテゴリ数。
        input_layer: mixed-aware な最初の DeepGP layer。
        hidden_layers: 追加の hidden DeepGP layers。
        last_layer: スカラー latent 出力を返す最終 layer。

    Raises:
        ValueError: カテゴリ値が整数エンコードでない場合、または設定されたカテゴリ範囲外の場合。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int],
        category_counts: Optional[Dict[int, int]] = None,
        list_hidden_dims: Optional[Sequence[int]] = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.01,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        beta: float = 1.0,
        model_type: str = "DEFAULT",
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood: Optional[_OneDimensionalLikelihood] = None,
    ) -> None:
        super().__init__()

        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be specified for the mixed DeepGP model.")

        train_Y = _prepare_ordinal_targets(train_Y, train_X)
        cat_dims = _normalize_cat_dims(cat_dims, train_X.shape[-1])
        category_counts = self._infer_category_counts(
            X=train_X,
            cat_dims=cat_dims,
            category_counts=category_counts,
        )
        _validate_categorical_values(train_X, cat_dims, category_counts)
        input_transform = _to_device_dtype_transform(input_transform, train_X)

        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self._train_targets = train_Y
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.input_transform = input_transform
        self.likelihood = likelihood or OrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        )

        self.num_classes = int(num_classes)
        self.cat_dims = list(cat_dims)
        self.category_counts = copy.deepcopy(category_counts)
        self.list_hidden_dims = list(_default_hidden_dims(list_hidden_dims))
        self.num_inducing = int(num_inducing)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.beta = float(beta)
        self.model_type = str(model_type)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        d = train_X.shape[-1]
        self.ord_dims = sorted(set(range(d)) - set(self.cat_dims))
        self._ignore_X_dims_scaling_check = self.cat_dims
        self.use_skip = self.model_type.lower() == "skip"
        self.original_input_dim = d

        train_X_tf = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name="OrdinalMixedDeepGPModel.input_transform",
        )
        hidden_dims = list(self.list_hidden_dims)

        self.input_layer = _make_layer(
            DeepMixedGPHiddenLayer,
            input_dims=d,
            output_dims=hidden_dims[0],
            ord_dims=self.ord_dims,
            cat_dims=self.cat_dims,
            num_inducing=self.num_inducing,
            mean_type="linear",
            input_data=train_X_tf,
            learn_inducing_locations=self.learn_inducing_locations,
        )

        self.hidden_layers = nn.ModuleList()
        current_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            if self.use_skip:
                layer = _make_layer(
                    SkipDeepGPHiddenLayer,
                    base_input_dims=current_dim,
                    skip_input_dims=self.original_input_dim,
                    output_dims=hidden_dim,
                    num_inducing=self.num_inducing,
                    mean_type="linear",
                    input_data=None,
                    learn_inducing_locations=self.learn_inducing_locations,
                )
            else:
                layer = _make_layer(
                    DeepGPHiddenLayer,
                    input_dims=current_dim,
                    output_dims=hidden_dim,
                    num_inducing=self.num_inducing,
                    mean_type="linear",
                    input_data=None,
                    learn_inducing_locations=self.learn_inducing_locations,
                )
            self.hidden_layers.append(layer)
            current_dim = hidden_dim

        if self.use_skip:
            self.last_layer = _make_layer(
                SkipDeepMixedGPHiddenLayer,
                base_input_dims=current_dim,
                skip_input_dims=d,
                original_ord_dims=self.ord_dims,
                original_cat_dims=self.cat_dims,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )
        else:
            self.last_layer = _make_layer(
                DeepGPHiddenLayer,
                input_dims=current_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )

        self.to(train_X)

    @staticmethod
    def _infer_category_counts(
        X: Tensor,
        cat_dims: Sequence[int],
        category_counts: Optional[Dict[int, int]] = None,
    ) -> Dict[int, int]:
        cat_dims = _normalize_cat_dims(cat_dims, X.shape[-1])

        inferred: Dict[int, int] = {}
        if category_counts is not None:
            inferred.update({int(k): int(v) for k, v in category_counts.items()})

        for j in cat_dims:
            vals = X[..., j]

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )

            if vals.numel() == 0:
                raise ValueError(f"Categorical column {j} is empty.")

            min_val = vals.min().item()
            max_val = vals.max().item()

            if min_val < 0:
                raise ValueError(
                    f"Categorical column {j} must be non-negative integer-coded, "
                    f"got min={min_val}"
                )

            if j not in inferred:
                inferred[j] = int(max_val) + 1

            if inferred[j] <= 0:
                raise ValueError(
                    f"category_counts[{j}] must be positive, got {inferred[j]}"
                )

        return inferred

    def _get_rebuild_kwargs(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "cat_dims": list(self.cat_dims),
            "category_counts": copy.deepcopy(self.category_counts),
            "list_hidden_dims": list(self.list_hidden_dims),
            "num_inducing": self.num_inducing,
            "learn_inducing_locations": self.learn_inducing_locations,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "beta": self.beta,
            "model_type": self.model_type,
            "fix_first_cutpoint": self.fix_first_cutpoint,
            "init_gap": self.init_gap,
            "eps": self.eps,
            "verbose": self.verbose,
            "conditioning_steps": self.conditioning_steps,
            "conditioning_lr": self.conditioning_lr,
            "conditioning_batch_size": self.conditioning_batch_size,
            "input_transform": _clone_input_transform(self.input_transform, self.train_inputs[0]),
        }

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = X.to(self.train_inputs[0])
        if X.ndim == 1:
            if self.train_inputs[0].shape[-1] != 1:
                raise ValueError(
                    f"1D X can only be used when input dim is 1, got d={self.train_inputs[0].shape[-1]}"
                )
            X = X.unsqueeze(-1)
        if X.ndim != 2:
            raise ValueError(f"Observation X must be [n, d], got shape={tuple(X.shape)}")
        if X.shape[-1] != self.train_inputs[0].shape[-1]:
            raise ValueError(
                f"X feature dim mismatch: expected {self.train_inputs[0].shape[-1]}, got {X.shape[-1]}"
            )
        _validate_categorical_values(X, self.cat_dims, self.category_counts)
        return X.contiguous()

    def _canonicalize_new_Y(self, Y: Tensor, n: int) -> Tensor:
        Y = torch.as_tensor(Y, device=self.train_inputs[0].device)
        if Y.ndim == 0:
            Y = Y.view(1)
        elif Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        elif Y.ndim != 1:
            raise ValueError(f"Y must be [n] or [n, 1], got shape={tuple(Y.shape)}")
        if Y.shape[0] != n:
            raise ValueError(f"Y length mismatch: expected {n}, got {Y.shape[0]}")
        return Y.long().contiguous()

    def forward(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> MultivariateNormal:
        """混合入力に対する latent DeepGP distribution を評価する。

        Args:
            X: 入力 tensor、または ``(input_tensor,)`` 形式の tuple。
                カテゴリ列は整数エンコードされている必要がある。
            apply_input_transform: ``self.input_transform`` を適用するかどうか。

        Returns:
            最終 DeepGP layer から得られる latent GPyTorch 多変量正規分布。
        """
        X = self._unwrap_inputs(X)
        X = self._apply_input_transform(X, apply_input_transform=apply_input_transform)

        original_input = X
        h = self.input_layer(X)
        for layer in self.hidden_layers:
            if isinstance(layer, SkipDeepGPHiddenLayer):
                h = layer(h, original_input=original_input)
            else:
                h = layer(h)

        if isinstance(self.last_layer, SkipDeepMixedGPHiddenLayer):
            return self.last_layer(h, original_input=original_input)
        return self.last_layer(h)


# ============================================================
# 学習ヘルパー
# ============================================================


def fit_true_deep_ordinal_gp(
    model: Union[OrdinalDeepGPModel, OrdinalMixedDeepGPModel],
    num_epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    beta: Optional[float] = None,
    verbose: Optional[bool] = None,
):
    """``DeepApproximateMLL`` を使って順序回帰 DeepGP を学習する。

    Args:
        model: 学習対象の順序回帰 DeepGP モデル。
        num_epochs: 学習 epoch 数。``None`` の場合は ``model.num_epochs`` を使う。
        lr: Adam の学習率。``None`` の場合は ``model.lr`` を使う。
        batch_size: mini-batch size。``None`` の場合は ``model.batch_size`` を使う。
            それも ``None`` の場合は full-batch 学習を行う。
        beta: KL divergence の重み。``None`` の場合は ``model.beta`` を使う。
        verbose: 進捗を表示するかどうか。``None`` の場合は ``model.verbose`` を使う。

    Returns:
        学習済みモデル。同じオブジェクトを in-place に更新して返す。

    Notes:
        ``train_inputs`` は raw input space のまま保持し、``input_transform`` は
        ``forward`` 内で適用します。``InputPerturbation`` を使う場合は、
        学習入力が展開されないように ``transform_on_train=False`` を設定してください。
    """
    num_epochs = model.num_epochs if num_epochs is None else int(num_epochs)
    lr = model.lr if lr is None else float(lr)
    verbose = model.verbose if verbose is None else bool(verbose)
    beta = model.beta if beta is None else float(beta)

    train_X = model.train_inputs[0]
    train_Y = model.train_targets

    if num_epochs <= 0:
        model.eval()
        model.likelihood.eval()
        return model

    dataset = TensorDataset(train_X, train_Y)
    if batch_size is None:
        batch_size = model.batch_size
    if batch_size is None:
        batch_size = len(dataset)
    batch_size = int(batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = model.make_mll(beta=beta)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            latent_dist = model(xb)
            loss = -mll(latent_dist, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
            avg_loss = total_loss / train_X.shape[-2]
            cuts = model.ordinal_likelihood.cutpoints.detach().cpu().numpy()
            print(f"[deep-ordinal-fit] epoch={epoch+1:03d} loss={avg_loss:.4f} cutpoints={cuts}")

    model.eval()
    model.likelihood.eval()
    return model
