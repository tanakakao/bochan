from __future__ import annotations

"""ordinal 用 MAP-SAAS GP モデル。

連続入力版は ``OrdinalGPModel`` に SAAS prior 付き Matern kernel を渡す。
Mixed 版はカテゴリ列を one-hot encode し、内部 ordinal GP には encoded-space の
入力を渡す。
"""

from copy import deepcopy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.ordinal.base import OrdinalGPModel
from bochan.models.components.saas import (
    OneHotEncodingMixin,
    build_map_saas_covar_module,
    concat_optional_noise,
    flatten_optional_noise,
    flatten_targets,
    prepare_mixed_conditioning_data,
    to_device_dtype_transform,
)


def _infer_num_classes(train_Y: Tensor, num_classes: Optional[int]) -> int:
    """ordinal label から class 数を推定する。"""
    if num_classes is not None:
        return int(num_classes)
    y = flatten_targets(train_Y).long()
    if y.numel() == 0:
        raise ValueError("Cannot infer num_classes from empty train_Y.")
    if y.min().item() < 0:
        raise ValueError("Ordinal labels must be non-negative integers starting at 0.")
    return int(y.max().item()) + 1


def _flatten_ordinal_targets(y: Tensor) -> Tensor:
    """ordinal target を [n] の long tensor にそろえる。"""
    return flatten_targets(y).long()


class SaasOrdinalGPModel(OrdinalGPModel):
    """MAP-SAAS style の ordinal GP。

    Args:
        train_X: 学習入力。
        train_Y: ordinal label。0 始まりの整数ラベルを想定。
        num_classes: クラス数。None の場合は train_Y から推定。
        train_Yvar: optional noise。通常は None。
        ordinal_likelihood: ordinal likelihood。
        likelihood: ``ordinal_likelihood`` の alias。
        input_transform: 入力変換。
        mean_module: mean module。
        covar_module: covar module。None の場合、SAAS prior 付き Matern kernel を使う。
        num_inducing_points: inducing point 数。
        inducing_points: inducing points。
        learn_inducing_locations: inducing point を学習するか。
        tau: SAAS global shrinkage。
        saas_log_scale: log-scale SAAS flag。
        saas_nu: Matern kernel の smoothness。
        fix_first_cutpoint: 最初の cutpoint を固定するか。
        init_gap: cutpoint 初期 gap。
        eps: 数値安定化 epsilon。

    Notes:
        - 実体は通常の ordinal GP + MAP-SAAS kernel。
        - fully Bayesian SAAS ではないため MCMC batch 次元は持たない。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        train_Yvar: Optional[Tensor] = None,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        likelihood: Optional[OrdinalLogitLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        self.num_classes = _infer_num_classes(train_Y=train_Y, num_classes=num_classes)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.train_inputs_raw = (train_X.detach().clone(),)

        if ordinal_likelihood is None:
            ordinal_likelihood = likelihood
        if ordinal_likelihood is None:
            ordinal_likelihood = OrdinalLogitLikelihood(
                num_classes=self.num_classes,
                eps=float(eps),
                init_gap=float(init_gap),
                fix_first_cutpoint=bool(fix_first_cutpoint),
            ).to(train_X)

        input_transform = to_device_dtype_transform(input_transform, train_X)
        if covar_module is None:
            covar_module = build_map_saas_covar_module(
                train_X=train_X,
                input_transform=input_transform,
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )

        # OrdinalGPModel の版差を吸収する。
        try:
            super().__init__(
                train_X=train_X,
                train_Y=train_Y,
                num_classes=self.num_classes,
                train_Yvar=train_Yvar,
                ordinal_likelihood=ordinal_likelihood,
                input_transform=input_transform,
                mean_module=mean_module,
                covar_module=covar_module,
                num_inducing_points=num_inducing_points,
                inducing_points=inducing_points,
                learn_inducing_locations=learn_inducing_locations,
                **kwargs,
            )
        except TypeError:
            try:
                super().__init__(
                    train_X=train_X,
                    train_Y=train_Y,
                    num_classes=self.num_classes,
                    likelihood=ordinal_likelihood,
                    input_transform=input_transform,
                    mean_module=mean_module,
                    covar_module=covar_module,
                    inducing_points_num=num_inducing_points,
                    inducing_points=inducing_points,
                    learn_inducing_locations=learn_inducing_locations,
                    **kwargs,
                )
            except TypeError:
                super().__init__(
                    train_X=train_X,
                    train_Y=train_Y,
                    num_classes=self.num_classes,
                    input_transform=input_transform,
                    mean_module=mean_module,
                    covar_module=covar_module,
                    inducing_points_num=num_inducing_points,
                    inducing_points=inducing_points,
                    learn_inducing_locations=learn_inducing_locations,
                    **kwargs,
                )

        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets = _flatten_ordinal_targets(train_Y).to(device=train_X.device)

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    @property
    def num_outputs(self) -> int:
        return 1


class SaasOrdinalMixedGPModel(OneHotEncodingMixin, SaasOrdinalGPModel):
    """mixed 入力向け MAP-SAAS ordinal GP。

    Args:
        train_X: raw-space の学習入力。カテゴリ列は整数エンコードを想定。
        train_Y: ordinal label。
        num_classes: クラス数。None の場合は train_Y から推定。
        cat_dims: raw-space におけるカテゴリ列 index。
        train_Yvar: optional noise。
        ordinal_likelihood: ordinal likelihood。
        likelihood: ``ordinal_likelihood`` の alias。
        input_transform: raw-space または encoded-space 用 input transform。
        mean_module: mean module。
        covar_module: covar module。None の場合、encoded-space に SAAS prior を貼る。
        num_inducing_points: inducing point 数。
        inducing_points: raw-space または encoded-space の inducing points。
        learn_inducing_locations: inducing point を学習するか。
        tau: SAAS global shrinkage。
        saas_log_scale: log-scale SAAS flag。
        saas_nu: Matern kernel の smoothness。

    Notes:
        - public ``train_inputs_raw`` と ``train_inputs`` は raw-space。
        - 内部 GP の学習は one-hot encoded-space で行う。
        - encoded-space の情報は ``encoded_train_inputs_raw`` に保持する。
        - raw-space の input_transform は one-hot encode 前に適用する。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        cat_dims: Optional[Sequence[int]] = None,
        train_Yvar: Optional[Tensor] = None,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        likelihood: Optional[OrdinalLogitLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()

        encoded_train_X = self._init_one_hot_encoding(train_X=train_X, cat_dims=cat_dims)
        self.encoded_train_inputs_raw = (encoded_train_X.detach().clone(),)
        expanded_input_transform = self._maybe_expand_input_transform(input_transform)
        encoded_inducing_points = self._canonicalize_inducing_points_for_encoded_space(inducing_points)

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            train_Yvar=train_Yvar,
            ordinal_likelihood=ordinal_likelihood,
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
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            **kwargs,
        )

        self.encoded_train_inputs = getattr(self, "train_inputs", (encoded_train_X,))
        if len(self.encoded_train_inputs) > 0:
            self._check_encoded_categorical_blocks_unchanged(
                X_encoded=encoded_train_X,
                X_tf=self.encoded_train_inputs[0],
                name=f"{self.__class__.__name__}.training_input_transform",
            )
        self.encoded_inducing_points_raw = encoded_inducing_points

        # public 側は raw-space に戻す。
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X.detach().clone(),)
        self.train_targets = _flatten_ordinal_targets(train_Y).to(device=train_X.device)

    @property
    def train_input_raw(self) -> Tensor:
        return self.train_inputs_raw[0]

    @property
    def train_input(self) -> Tensor:
        return self.train_inputs[0]

    @property
    def raw_train_X(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_X(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_Y(self) -> Tensor:
        return self.train_targets

    @property
    def encoded_train_input_raw(self) -> Tensor:
        return self.encoded_train_inputs_raw[0]

    @property
    def encoded_train_input(self) -> Tensor:
        return self.encoded_train_inputs[0]

    def _set_transformed_inputs(self) -> None:
        """BoTorch eval 時の自動 transformed input 更新を無効化する。"""
        return None

    def _canonicalize_posterior_X(self, X: Tensor) -> Tensor:
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_input_raw.device, dtype=self.train_input_raw.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim < 2:
            raise ValueError(f"X must have at least 2 dims, got shape={tuple(X.shape)}.")
        if X.shape[-1] not in (self.raw_dim, self.encoded_dim):
            raise ValueError(
                f"Expected raw dim {self.raw_dim} or encoded dim {self.encoded_dim}, got {X.shape[-1]}."
            )
        return X.contiguous()

    def _get_input_transform_for_eval(self, input_transform=None):
        """明示指定または self.input_transform を返す。"""
        return input_transform if input_transform is not None else getattr(self, "input_transform", None)

    def _apply_transform_raw_first(self, X: Tensor, input_transform=None) -> Tensor:
        """raw/encoded X を内部 GP が受け取る encoded-space へ変換する。

        Mixed + InputPerturbation では、raw 3次元用の input_transform を
        one-hot 後の encoded 4次元に適用すると次元不一致になる。
        そのため raw 入力の場合は、

            raw X -> input_transform -> one-hot encode -> inner GP

        の順を優先する。encoded 入力の場合は、既に内部特徴空間にいるものとして
        raw-space transform は原則再適用しない。
        """
        X = self._canonicalize_posterior_X(X)
        tf = self._get_input_transform_for_eval(input_transform)

        if tf is None:
            return self._to_encoded_feature_space(X).contiguous()

        # raw-space 入力: raw-space transform を先に適用し、その後 one-hot encode する。
        if X.shape[-1] == self.raw_dim:
            raw_transform_error = None
            try:
                X_raw_tf = tf(X)
                if isinstance(X_raw_tf, tuple):
                    X_raw_tf = X_raw_tf[0]
                X_raw_tf = self._canonicalize_posterior_X(X_raw_tf)

                if X_raw_tf.shape[-1] == self.raw_dim:
                    return self._to_encoded_feature_space(X_raw_tf).contiguous()

                if X_raw_tf.shape[-1] == self.encoded_dim:
                    self._check_encoded_categorical_blocks_unchanged(
                        X_encoded=self._to_encoded_feature_space(X),
                        X_tf=X_raw_tf,
                        name=f"{self.__class__.__name__}.raw_input_transform",
                    )
                    return X_raw_tf.contiguous()
            except Exception as exc:
                raw_transform_error = exc

            # 旧実装互換: transform が encoded-space 用の場合のみ fallback する。
            X_encoded = self._to_encoded_feature_space(X)
            try:
                X_tf = tf(X_encoded)
                if isinstance(X_tf, tuple):
                    X_tf = X_tf[0]
                self._check_encoded_categorical_blocks_unchanged(
                    X_encoded=X_encoded,
                    X_tf=X_tf,
                    name=f"{self.__class__.__name__}.encoded_input_transform",
                )
                return X_tf.contiguous()
            except Exception:
                if raw_transform_error is not None:
                    raise raw_transform_error
                raise

        # encoded-space 入力:
        # raw-space InputPerturbation は encoded dim に適用できないため、無理に再適用しない。
        # ただし encoded-space 用 transform であれば適用できる。
        try:
            X_tf = tf(X)
            if isinstance(X_tf, tuple):
                X_tf = X_tf[0]
            if X_tf.shape[-1] == self.encoded_dim:
                self._check_encoded_categorical_blocks_unchanged(
                    X_encoded=X,
                    X_tf=X_tf,
                    name=f"{self.__class__.__name__}.encoded_input_transform",
                )
                return X_tf.contiguous()
        except Exception:
            pass

        return X.contiguous()

    def transform_inputs(self, X: Tensor, input_transform=None) -> Tensor:
        """raw/encoded X を内部 GP が受け取る encoded-space にそろえる。

        raw-space の input_transform、特に InputPerturbation は one-hot encode 前に
        適用する。これにより raw dim と encoded dim の不一致を避ける。
        """
        return self._apply_transform_raw_first(X, input_transform=input_transform)

    def _to_training_feature_space(self, X: Tensor) -> Tensor:
        """raw/encoded X を input_transform 適用済み encoded-space へ変換する。"""
        return self._apply_transform_raw_first(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        # ここで _to_training_feature_space を先に呼ぶと、super().posterior -> self(X)
        # の中で forward 側の変換が再度走り、mixed feature 変換や input_transform が
        # 二重適用される。posterior には raw/encoded X をそのまま渡し、forward で
        # 一度だけ raw transform -> encode -> inner GP の順に処理する。
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def forward(self, X: Tensor):
        # super().forward(...) は transform_inputs を再度呼ぶため使わない。
        # ここで一度だけ内部 GP 用の feature space に変換する。
        X_model = self._to_training_feature_space(X)
        return self.model(X_model)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "SaasOrdinalMixedGPModel":
        """raw/encoded X の追加観測で wrapper を再構築する。"""
        X_new_raw, Y_new, Yvar_new = prepare_mixed_conditioning_data(
            X,
            Y,
            noise,
            raw_dim=self.raw_dim,
            encoded_dim=self.encoded_dim,
            decode_fn=self.decode_inputs,
            target_dtype=torch.long,
        )
        train_X_old = self.train_inputs_raw[0]
        train_Y_old = _flatten_ordinal_targets(self.train_targets)
        X_full = torch.cat([
            train_X_old,
            X_new_raw.to(dtype=train_X_old.dtype, device=train_X_old.device),
        ], dim=0)
        Y_full = torch.cat([
            train_Y_old,
            Y_new.to(device=train_Y_old.device).long(),
        ], dim=0)
        Yvar_full = concat_optional_noise(
            old_Y=train_Y_old.to(dtype=train_X_old.dtype),
            old_Yvar=self.train_Yvar_raw,
            new_Y=Y_new.to(dtype=train_X_old.dtype, device=train_X_old.device),
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        learn_inducing_locations = True
        if hasattr(self, "model") and hasattr(self.model, "variational_strategy"):
            learn_inducing_locations = getattr(self.model.variational_strategy, "learn_inducing_locations", True)

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            num_classes=self.num_classes,
            cat_dims=list(self.cat_dims),
            train_Yvar=Yvar_full,
            ordinal_likelihood=deepcopy(getattr(self, "ordinal_likelihood", None)),
            input_transform=deepcopy(getattr(self, "input_transform", None)),
            mean_module=deepcopy(getattr(getattr(self, "model", None), "mean_module", None)),
            covar_module=deepcopy(getattr(getattr(self, "model", None), "covar_module", None)),
            num_inducing_points=(
                int(self.encoded_inducing_points_raw.shape[-2])
                if isinstance(self.encoded_inducing_points_raw, Tensor)
                else 20
            ),
            inducing_points=(
                self.encoded_inducing_points_raw.detach().clone()
                if isinstance(self.encoded_inducing_points_raw, Tensor)
                else None
            ),
            learn_inducing_locations=learn_inducing_locations,
            tau=self.tau,
            saas_log_scale=self.saas_log_scale,
            saas_nu=self.saas_nu,
        )
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model


__all__ = [
    "SaasOrdinalGPModel",
    "SaasOrdinalMixedGPModel",
]
