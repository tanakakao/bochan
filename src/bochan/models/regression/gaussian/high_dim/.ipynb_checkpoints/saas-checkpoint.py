from __future__ import annotations

"""regression 用 SAAS / Additive MAP-SAAS モデル。

BoTorch の ``AdditiveMapSaasSingleTaskGP`` を regression の標準 SAAS 実装として使う。
Mixed 版では raw-space のカテゴリ列を one-hot encode し、内部の
``AdditiveMapSaasSingleTaskGP`` には encoded-space の入力を渡す。
"""

from copy import deepcopy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from bochan.models.components.saas import (
    OneHotEncodingMixin,
    concat_optional_noise,
    flatten_targets,
    prepare_mixed_conditioning_data,
)


class SaasSingleTaskGP(AdditiveMapSaasSingleTaskGP):
    """連続入力向け Additive MAP-SAAS single-task GP。

    Args:
        train_X: raw-space の学習入力。shape は ``batch_shape x n x d``。
        train_Y: 学習 target。shape は ``batch_shape x n x m``。
        train_Yvar: 既知観測ノイズ。None の場合は Gaussian likelihood 側で推定。
        outcome_transform: outcome transform。BoTorch 既定では Standardize。
        input_transform: raw-space に適用する input transform。
        num_taus: additive MAP-SAAS kernel に含める tau の個数。

    Notes:
        - 中核実装は BoTorch の ``AdditiveMapSaasSingleTaskGP``。
        - ``train_inputs_raw`` には raw-space の clone を保持する。
        - exact GP としての性質は BoTorch 実装に従う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: OutcomeTransform | Any | None = None,
        input_transform: Optional[InputTransform] = None,
        num_taus: int = 4,
    ) -> None:
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.num_taus = int(num_taus)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            num_taus=num_taus,
        )


class SaasMixedSingleTaskGP(OneHotEncodingMixin, AdditiveMapSaasSingleTaskGP):
    """mixed 入力向け Additive MAP-SAAS single-task GP。

    カテゴリ列を one-hot encode してから BoTorch の
    ``AdditiveMapSaasSingleTaskGP`` に渡す wrapper。

    Args:
        train_X: raw-space の学習入力。カテゴリ列は整数エンコードを想定。
        train_Y: 学習 target。
        cat_dims: raw-space におけるカテゴリ列 index。
        train_Yvar: 既知観測ノイズ。
        outcome_transform: outcome transform。
        input_transform: raw-space 用、または encoded-space 用の input transform。
            raw-space 用 ``Normalize`` は encoded-space に拡張される。
        num_taus: additive MAP-SAAS kernel に含める tau の個数。

    Attributes:
        train_inputs_raw: raw-space の training X。
        encoded_train_inputs_raw: one-hot encode 後の training X。
        encoded_cat_dims: raw カテゴリ列から encoded one-hot block への対応。

    Notes:
        - public API は raw-space X を受け取る。
        - encoded-space X も許容するため、最適化後の encoded 候補確認にも使える。
        - ``get_optimize_acqf_mixed_fixed_features_list`` で encoded-space の
          ``fixed_features_list`` を取得できる。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Optional[Sequence[int]] = None,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: OutcomeTransform | Any | None = None,
        input_transform: Optional[InputTransform] = None,
        num_taus: int = 4,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.num_taus = int(num_taus)

        encoded_train_X = self._init_one_hot_encoding(train_X=train_X, cat_dims=cat_dims)
        self.encoded_train_inputs_raw = (encoded_train_X.detach().clone(),)

        expanded_input_transform = self._maybe_expand_input_transform(input_transform)

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=expanded_input_transform,
            num_taus=num_taus,
        )

        # public 側は raw-space を明示する。
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X.detach().clone(),)
        self.train_targets = train_Y
        self.input_transform = expanded_input_transform

    @property
    def encoded_train_input_raw(self) -> Tensor:
        """one-hot encode 後、input_transform 前の training X。"""
        return self.encoded_train_inputs_raw[0]

    def _set_transformed_inputs(self) -> None:
        """BoTorch の eval-time 自動 train_inputs 更新を無効化する。"""
        return None

    def transform_inputs(self, X: Tensor, input_transform=None) -> Tensor:  # noqa: N802
        """raw/encoded X を内部 encoded feature space に変換する。"""
        if isinstance(X, tuple):
            X = X[0]
        X_encoded = self._to_encoded_feature_space(X)
        if input_transform is not None:
            X_tf = input_transform(X_encoded)
        else:
            try:
                X_tf = AdditiveMapSaasSingleTaskGP.transform_inputs(self, X=X_encoded)
            except TypeError:
                X_tf = AdditiveMapSaasSingleTaskGP.transform_inputs(self, X_encoded)
        self._check_encoded_categorical_blocks_unchanged(
            X_encoded=X_encoded,
            X_tf=X_tf,
            name=f"{self.__class__.__name__}.input_transform",
        )
        return X_tf

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ):
        """raw-space または encoded-space の X に対する posterior を返す。"""
        X_encoded = self._to_encoded_feature_space(X)
        return super().posterior(
            X_encoded,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def forward(self, X: Tensor):
        """raw/encoded X を encoded-space にそろえてから base forward を呼ぶ。"""
        if isinstance(X, tuple):
            X = X[0]
        return super().forward(self._to_encoded_feature_space(X))

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "SaasMixedSingleTaskGP":
        """raw/encoded X の追加観測で wrapper を再構築する。

        Notes:
            - 戻り値の ``train_inputs_raw`` も raw-space になる。
            - Exact GP の fantasy batch には対応せず、通常の観測追加用途を想定する。
        """
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
        train_Y_old = self.train_Y_raw
        if train_Y_old.ndim == 1:
            train_Y_old = train_Y_old.unsqueeze(-1)
        if Y_new.ndim == 1:
            Y_new = Y_new.unsqueeze(-1)

        X_full = torch.cat([
            train_X_old,
            X_new_raw.to(dtype=train_X_old.dtype, device=train_X_old.device),
        ], dim=-2)
        Y_full = torch.cat([
            train_Y_old,
            Y_new.to(dtype=train_Y_old.dtype, device=train_Y_old.device),
        ], dim=-2)
        Yvar_full = concat_optional_noise(
            old_Y=flatten_targets(train_Y_old, dtype=train_X_old.dtype),
            old_Yvar=self.train_Yvar_raw,
            new_Y=flatten_targets(Y_new, dtype=train_X_old.dtype),
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )
        if Yvar_full is not None:
            Yvar_full = Yvar_full.unsqueeze(-1)

        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            cat_dims=list(self.cat_dims),
            train_Yvar=Yvar_full,
            outcome_transform=deepcopy(getattr(self, "outcome_transform", None)),
            input_transform=deepcopy(getattr(self, "input_transform", None)),
            num_taus=self.num_taus,
        )
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model


# 旧名を残す。
MixedAdditiveMapSaasSingleTaskGP = SaasMixedSingleTaskGP
AdditiveSaasSingleTaskGP = SaasSingleTaskGP


__all__ = [
    "SaasSingleTaskGP",
    "SaasMixedSingleTaskGP",
    "AdditiveSaasSingleTaskGP",
    "MixedAdditiveMapSaasSingleTaskGP",
]
