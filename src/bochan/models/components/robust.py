from __future__ import annotations

"""
Robust / Relevance Pursuit 系モデルで共有する helper と likelihood。

このファイルは task 非依存の共通部品を集約する。
各 task 側のモデルは、以下の責務だけを持つ想定。

- regression: BoTorch の feature relevance pursuit を活かす
- classification: train-point outlier relevance pursuit を使う
- ordinal: label smoothing robust / train-point outlier relevance pursuit を使う

Public convention:
    train_inputs_raw[0]: raw-space の訓練入力
    train_inputs[0]: モデル内部で使う入力。通常は input_transform 後
    fit_train_inputs[0]: train-point outlier RRP などで実際に likelihood 学習に使う入力
"""

import copy
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from botorch.models.relevance_pursuit import RelevancePursuitMixin
from botorch.models.transforms.input import InputTransform
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood


__all__ = [
    "SafeDeepcopyMixin",
    "TrainInputsAliasMixin",
    "RobustOrdinalLogitLikelihood",
    "SparseOutlierBernoulliLikelihood",
    "SparseOutlierOrdinalLogitLikelihood",
    "align_like",
    "clone_input_transform",
    "expand_raw_X_to_match_transformed_q",
    "check_categorical_columns_unchanged",
    "apply_input_transform_for_eval",
    "apply_input_transform_for_training",
    "canonicalize_inducing_points",
    "make_raw_inducing_points",
    "make_augmented_targets_and_base_indices",
    "flatten_targets",
    "flatten_optional_noise",
    "concat_optional_noise",
    "prepare_wrapper_conditioning_data",
]


# =============================================================================
# deepcopy safety
# =============================================================================


class SafeDeepcopyMixin:
    """
    deepcopy / fantasize 用に non-leaf Tensor を leaf Tensor 化する mixin。

    BoTorch の RRP 系モデルでは、BMC probability などの中間計算結果が
    non-leaf Tensor として attribute に残ることがある。この状態で
    ``copy.deepcopy`` すると失敗することがあるため、pickle state 生成時だけ
    leaf Tensor に置き換える。
    """

    @staticmethod
    def leafify_tensor_for_deepcopy(x: Any) -> Any:
        """non-leaf Tensor を detach clone して deepcopy-safe にする。"""
        if isinstance(x, torch.nn.Parameter):
            return x
        if torch.is_tensor(x) and not x.is_leaf:
            return x.detach().clone()
        return x

    @classmethod
    def sanitize_for_deepcopy(cls, obj: Any) -> Any:
        """dict / list / tuple を再帰的に deepcopy-safe にする。"""
        if isinstance(obj, torch.nn.Parameter):
            return obj
        if torch.is_tensor(obj):
            return cls.leafify_tensor_for_deepcopy(obj)
        if isinstance(obj, dict):
            return {k: cls.sanitize_for_deepcopy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls.sanitize_for_deepcopy(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(cls.sanitize_for_deepcopy(v) for v in obj)
        return obj

    def __getstate__(self):
        """copy / pickle 時に non-leaf Tensor を安全化した state を返す。"""
        state = super().__getstate__().copy()
        skip_keys = {"_parameters", "_buffers", "_modules"}
        for k, v in list(state.items()):
            if k in skip_keys:
                continue
            state[k] = self.sanitize_for_deepcopy(v)
        return state


# =============================================================================
# tensor / transform helper
# =============================================================================


def clone_input_transform(
    input_transform: Optional[InputTransform],
) -> Optional[InputTransform]:
    """input_transform を安全に複製する。"""
    return None if input_transform is None else copy.deepcopy(input_transform)


def align_like(t: Tensor, ref: Tensor) -> Tensor:
    """vector-like tensor を ref と同じ shape に揃える。"""
    while t.dim() < ref.dim():
        t = t.unsqueeze(0)
    if t.shape == ref.shape:
        return t
    if t.ndim >= 2 and t.transpose(-1, -2).shape == ref.shape:
        return t.transpose(-1, -2)
    if t.numel() == ref.numel():
        return t.reshape_as(ref)
    return t.expand_as(ref)


def flatten_targets(y: Tensor, *, dtype: Optional[torch.dtype] = None) -> Tensor:
    """target を [N] にそろえる。"""
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    y = y.reshape(-1)
    return y if dtype is None else y.to(dtype=dtype)


def flatten_optional_noise(noise: Optional[Tensor]) -> Optional[Tensor]:
    """optional noise を [N] にそろえる。"""
    if noise is None:
        return None
    if noise.ndim > 1 and noise.shape[-1] == 1:
        noise = noise.squeeze(-1)
    return noise.reshape(-1)


def expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """
    InputPerturbation 後の X_tf と比較できるよう raw X の q 次元を展開する。

    通常は ``q_like == q``。InputPerturbation では ``q_like == q * n_w``
    になるため、raw 側を ``repeat_interleave`` して比較する。
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


def check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
    *,
    name: str = "input_transform",
) -> None:
    """
    mixed model 用に input_transform がカテゴリ列を変更していないか確認する。

    InputPerturbation で q が増える場合も、raw X 側を repeat して比較する。
    """
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]

    if X.shape[-1] != X_tf.shape[-1]:
        raise ValueError(
            f"{name} must preserve feature dimension for mixed models. "
            f"raw dim={X.shape[-1]}, transformed dim={X_tf.shape[-1]}."
        )

    X_cmp = expand_raw_X_to_match_transformed_q(X, X_tf)
    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            f"Could not align raw X with transformed X in {name}. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            f"{name} must not modify categorical columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """posterior / acquisition 評価用 transform。eval mode の q 展開を許す。"""
    if input_transform is None:
        return X
    X_tf = input_transform(X)
    check_categorical_columns_unchanged(
        X=X,
        X_tf=X_tf,
        cat_dims=cat_dims,
        name="input_transform",
    )
    return X_tf


def apply_input_transform_for_training(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
    name: str = "input_transform",
) -> Tensor:
    """
    学習データ用 transform。

    ``input_transform.train()`` で適用し、適用後に ``eval()`` に戻す。
    InputPerturbation は通常 train mode では点数を増やさない想定。
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
            "This will not match train targets. "
            "For InputPerturbation, ensure transform_on_train=False."
        )

    check_categorical_columns_unchanged(
        X=X,
        X_tf=X_tf,
        cat_dims=cat_dims,
        name=name,
    )
    return X_tf


def canonicalize_inducing_points(
    inducing_points: Tensor,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """raw-space inducing_points を [m, d] に正規化する。"""
    inducing_points = torch.as_tensor(inducing_points, device=device, dtype=dtype)
    if inducing_points.ndim != 2:
        raise ValueError(
            f"inducing_points must be [m, d], got shape={tuple(inducing_points.shape)}."
        )
    if inducing_points.shape[-1] != d:
        raise ValueError(
            f"inducing_points feature dim mismatch: expected {d}, got {inducing_points.shape[-1]}."
        )
    return inducing_points.contiguous()


def make_raw_inducing_points(
    raw_train_X: Tensor,
    inducing_points_num: int,
    inducing_points: Optional[Tensor],
) -> Tensor:
    """raw-space の inducing points を作る。"""
    if inducing_points is not None:
        return canonicalize_inducing_points(
            inducing_points,
            d=raw_train_X.shape[-1],
            device=raw_train_X.device,
            dtype=raw_train_X.dtype,
        )
    n = raw_train_X.shape[-2]
    m = min(int(inducing_points_num), n)
    perm = torch.randperm(n, device=raw_train_X.device)[:m]
    return raw_train_X[perm].detach().clone().contiguous()


def make_augmented_targets_and_base_indices(
    train_Y: Tensor,
    X_aug: Tensor,
    *,
    n_base: int,
) -> tuple[Tensor, Optional[Tensor]]:
    """
    InputPerturbation 等で X が n_base -> n_aug に増えた場合に train_Y も展開する。
    """
    Y_base = flatten_targets(train_Y).to(dtype=X_aug.dtype, device=X_aug.device)
    n_aug = X_aug.shape[-2]

    if n_aug == n_base:
        return Y_base, None
    if n_aug % n_base != 0:
        raise RuntimeError(
            "Cannot infer perturbation repeats. "
            f"n_base={n_base}, n_aug={n_aug}."
        )

    n_w = n_aug // n_base
    base_indices = torch.arange(n_base, dtype=torch.long, device=X_aug.device).repeat_interleave(n_w)
    return Y_base[base_indices], base_indices


def concat_optional_noise(
    old_Y: Tensor,
    old_Yvar: Optional[Tensor],
    new_Y: Tensor,
    new_Yvar: Optional[Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    """old/new の train_Yvar を連結する。片方だけある場合は 0 埋めする。"""
    if old_Yvar is None and new_Yvar is None:
        return None
    if old_Yvar is None:
        old_Yvar = torch.zeros_like(old_Y, dtype=dtype, device=device)
    else:
        old_Yvar = flatten_optional_noise(old_Yvar).to(dtype=dtype, device=device)
    if new_Yvar is None:
        new_Yvar = torch.zeros_like(new_Y, dtype=dtype, device=device)
    else:
        new_Yvar = flatten_optional_noise(new_Yvar).to(dtype=dtype, device=device)
    return torch.cat([old_Yvar, new_Yvar], dim=0)


def prepare_wrapper_conditioning_data(
    X: Tensor,
    Y: Tensor,
    noise: Optional[Tensor] = None,
    *,
    expected_input_dim: Optional[int] = None,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    wrapper の condition_on_observations 用に raw X / Y / noise を flatten する。

    fantasy batch ではなく、通常の新規観測追加を想定する。
    """
    if isinstance(X, tuple):
        X = X[0]
    if X.dim() < 2:
        raise ValueError("X must have shape [q, d] or [batch, q, d].")
    if expected_input_dim is not None and X.shape[-1] != expected_input_dim:
        raise ValueError(
            f"Expected X last dim {expected_input_dim}, got {X.shape[-1]}."
        )

    expected_y_shape = X.shape[:-1]
    if Y.dim() == X.dim() and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)
    if Y.shape != expected_y_shape:
        raise NotImplementedError(
            "Only non-fantasy observations with Y.shape == X.shape[:-1] are supported. "
            f"Got X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}."
        )

    if noise is not None:
        if noise.dim() == X.dim() and noise.shape[-1] == 1:
            noise = noise.squeeze(-1)
        if noise.shape != expected_y_shape:
            raise ValueError(
                "noise must match X.shape[:-1]. "
                f"Got X.shape={tuple(X.shape)}, noise.shape={tuple(noise.shape)}."
            )

    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1).to(dtype=X.dtype, device=X.device)
    noise_flat = None if noise is None else noise.reshape(-1).to(dtype=X.dtype, device=X.device)
    return X_flat, Y_flat, noise_flat


# =============================================================================
# train input alias mixin
# =============================================================================


class TrainInputsAliasMixin:
    """
    train_inputs 系の共通 alias と set_train_data 実装。

    Public convention:
        train_inputs_raw[0]: raw-space X
        train_inputs[0]: input_transform 後の internal X
        train_targets: 外側から見える target
    """

    def _set_transformed_inputs(self) -> None:
        """BoTorch eval 時の自動 transformed input 更新を無効化する。"""
        return None

    def set_train_data(
        self,
        inputs: Optional[Tensor | tuple[Tensor, ...]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        """raw-space inputs を受け取り、internal train_inputs と latent model を同期する。"""
        _ = strict
        if inputs is not None:
            X_raw = inputs[0] if isinstance(inputs, tuple) else inputs
            X_raw = torch.as_tensor(
                X_raw,
                device=self.train_inputs_raw[0].device,
                dtype=self.train_inputs_raw[0].dtype,
            )
            if X_raw.ndim == 1:
                X_raw = X_raw.unsqueeze(0)
            X_tf = apply_input_transform_for_training(
                X_raw,
                getattr(self, "input_transform", None),
                cat_dims=getattr(self, "cat_dims", None),
                name=f"{self.__class__.__name__}.input_transform",
            )
            self.train_inputs_raw = (X_raw,)
            self.train_inputs = (X_tf,)
            if hasattr(self, "model"):
                try:
                    self.model.train_inputs = (X_tf,)
                except Exception:
                    pass

        if targets is not None:
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            targets = targets.to(
                device=self.train_inputs_raw[0].device,
                dtype=torch.long,
            ).contiguous()
            self.train_targets = targets
            if hasattr(self, "model"):
                try:
                    self.model.train_targets = targets
                except Exception:
                    pass

    @property
    def train_input_raw(self) -> Tensor:
        return self.train_inputs_raw[0]

    @property
    def train_input(self) -> Tensor:
        return self.train_inputs[0]

    @property
    def transformed_train_input(self) -> Tensor:
        return self.train_inputs[0]

    @property
    def transformed_train_inputs(self) -> tuple[Tensor]:
        return self.train_inputs

    @property
    def raw_train_X(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_X_original(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_X(self) -> Tensor:
        return self.train_input

    @property
    def train_Y(self) -> Tensor:
        return self.train_targets


# =============================================================================
# robust likelihoods
# =============================================================================


class SparseOutlierBernoulliLikelihood(BernoulliLikelihood, RelevancePursuitMixin):
    """
    train-point sparse logit offset 付き Bernoulli likelihood。

    学習時だけ training point i に対して sparse offset ``delta_i`` を加える。
    予測時は test X に delta を使わない。
    """

    def __init__(
        self,
        dim: int,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        expanded_base_indices: Optional[Tensor] = None,
    ) -> None:
        BernoulliLikelihood.__init__(self)
        RelevancePursuitMixin.__init__(self, dim=int(dim), support=outlier_indices)

        init = torch.full((len(self.support),), float(delta_init), dtype=torch.get_default_dtype())
        self.register_parameter("raw_delta", Parameter(init))

        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long)
        self.register_buffer("expanded_base_indices", expanded_base_indices.to(dtype=torch.long))

        self._expansion_modifier = torch.abs
        self._contraction_modifier = torch.abs

    @property
    def sparse_parameter(self) -> Parameter:
        return self.raw_delta

    def set_sparse_parameter(self, value: Parameter) -> None:
        self.raw_delta = Parameter(value.to(self.raw_delta))

    def set_expanded_base_indices(self, expanded_base_indices: Optional[Tensor]) -> None:
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long, device=self.raw_delta.device)
        self.expanded_base_indices = expanded_base_indices.to(dtype=torch.long, device=self.raw_delta.device)

    @property
    def dense_delta(self) -> Tensor:
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense

    def _delta_for_function_dist(self, function_dist: MultivariateNormal) -> Optional[Tensor]:
        mean = function_dist.mean
        n = mean.shape[-1]
        if n == self.dim:
            return self.dense_delta.to(device=mean.device, dtype=mean.dtype)
        if self.expanded_base_indices.numel() > 0 and n == self.expanded_base_indices.numel():
            base_idx = self.expanded_base_indices.to(device=mean.device)
            dense_delta = self.dense_delta.to(device=mean.device, dtype=mean.dtype)
            return dense_delta[base_idx]
        return None

    def _shift_train_function_dist(self, function_dist: MultivariateNormal) -> MultivariateNormal:
        delta = self._delta_for_function_dist(function_dist)
        if delta is None:
            return function_dist
        delta = align_like(delta, function_dist.mean)
        shifted_mean = function_dist.mean + delta
        return function_dist.__class__(shifted_mean, function_dist.lazy_covariance_matrix)

    def expected_log_prob(self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        function_dist = self._shift_train_function_dist(function_dist)
        return super().expected_log_prob(observations, function_dist, *params, **kwargs)


class RobustOrdinalLogitLikelihood(OrdinalLogitLikelihood):
    """
    label smoothing 付き ordinal likelihood。

    厳密な外れ値モデルではなく、ordinal label に対する pragmatic なロバスト化。
    """

    def __init__(self, *args: Any, label_smoothing: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label_smoothing = float(label_smoothing)
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")

    def _smoothed_target_probs(self, observations: Tensor) -> Tensor:
        obs = observations.long()
        target = F.one_hot(obs, num_classes=self.num_classes).to(
            dtype=torch.get_default_dtype(), device=obs.device
        )
        if self.label_smoothing <= 0:
            return target
        smooth = self.label_smoothing / self.num_classes
        return target * (1.0 - self.label_smoothing) + smooth

    def expected_log_prob(self, observations: Tensor, function_dist, *params: Any, **kwargs: Any) -> Tensor:
        target_probs = self._smoothed_target_probs(observations).to(
            device=function_dist.mean.device, dtype=function_dist.mean.dtype
        )
        return self.quadrature(
            lambda f: (target_probs * torch.log(self.class_probs_from_f(f).clamp_min(self.eps))).sum(dim=-1),
            function_dist,
        )

    def log_marginal(self, observations: Tensor, function_dist, *params: Any, **kwargs: Any) -> Tensor:
        target_probs = self._smoothed_target_probs(observations).to(
            device=function_dist.mean.device, dtype=function_dist.mean.dtype
        )
        marginal_probs = self.marginal_class_probs(function_dist)
        return torch.log((target_probs * marginal_probs).sum(dim=-1).clamp_min(self.eps))


class SparseOutlierOrdinalLogitLikelihood(OrdinalLogitLikelihood, RelevancePursuitMixin):
    """
    train-point sparse logit offset 付き ordinal likelihood。

    classification の SparseOutlierBernoulliLikelihood に対応する ordinal 版。
    """

    def __init__(
        self,
        *args: Any,
        dim: int,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        expanded_base_indices: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        OrdinalLogitLikelihood.__init__(self, *args, **kwargs)
        RelevancePursuitMixin.__init__(self, dim=int(dim), support=outlier_indices)

        init = torch.full((len(self.support),), float(delta_init), dtype=torch.get_default_dtype())
        self.register_parameter("raw_delta", Parameter(init))

        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long)
        self.register_buffer("expanded_base_indices", expanded_base_indices.to(dtype=torch.long))

        self.label_smoothing = float(label_smoothing)
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")

        self._expansion_modifier = torch.abs
        self._contraction_modifier = torch.abs

    @property
    def sparse_parameter(self) -> Parameter:
        return self.raw_delta

    def set_sparse_parameter(self, value: Parameter) -> None:
        self.raw_delta = Parameter(value.to(self.raw_delta))

    def set_expanded_base_indices(self, expanded_base_indices: Optional[Tensor]) -> None:
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long, device=self.raw_delta.device)
        self.expanded_base_indices = expanded_base_indices.to(dtype=torch.long, device=self.raw_delta.device)

    @property
    def dense_delta(self) -> Tensor:
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense

    def _delta_for_function_dist(self, function_dist) -> Optional[Tensor]:
        mean = function_dist.mean
        n = mean.shape[-1]
        if n == self.dim:
            return self.dense_delta.to(device=mean.device, dtype=mean.dtype)
        if self.expanded_base_indices.numel() > 0 and n == self.expanded_base_indices.numel():
            base_idx = self.expanded_base_indices.to(device=mean.device)
            dense_delta = self.dense_delta.to(device=mean.device, dtype=mean.dtype)
            return dense_delta[base_idx]
        return None

    def _shift_train_function_dist(self, function_dist):
        delta = self._delta_for_function_dist(function_dist)
        if delta is None:
            return function_dist
        delta = align_like(delta, function_dist.mean)
        shifted_mean = function_dist.mean + delta
        return function_dist.__class__(shifted_mean, function_dist.lazy_covariance_matrix)

    def _smoothed_target_probs(self, observations: Tensor) -> Tensor:
        obs = observations.long()
        target = F.one_hot(obs, num_classes=self.num_classes).to(
            dtype=torch.get_default_dtype(), device=obs.device
        )
        if self.label_smoothing <= 0:
            return target
        smooth = self.label_smoothing / self.num_classes
        return target * (1.0 - self.label_smoothing) + smooth

    def expected_log_prob(self, observations: Tensor, function_dist, *params: Any, **kwargs: Any) -> Tensor:
        function_dist = self._shift_train_function_dist(function_dist)
        target_probs = self._smoothed_target_probs(observations).to(
            device=function_dist.mean.device, dtype=function_dist.mean.dtype
        )
        return self.quadrature(
            lambda f: (target_probs * torch.log(self.class_probs_from_f(f).clamp_min(self.eps))).sum(dim=-1),
            function_dist,
        )

    def log_marginal(self, observations: Tensor, function_dist, *params: Any, **kwargs: Any) -> Tensor:
        function_dist = self._shift_train_function_dist(function_dist)
        target_probs = self._smoothed_target_probs(observations).to(
            device=function_dist.mean.device, dtype=function_dist.mean.dtype
        )
        marginal_probs = self.marginal_class_probs(function_dist)
        return torch.log((target_probs * marginal_probs).sum(dim=-1).clamp_min(self.eps))
