from __future__ import annotations

"""
Regression 用 Robust Relevance Pursuit モデル。

配置想定:
    bochan/models/regression/robust/relevance_pursuit.py

命名規則:
    - regression では BoTorch の feature relevance pursuit を使う。
    - deepcopy / fantasize 安全化を入れたものは Safe* とする。
"""

import copy
from typing import Optional, Sequence

import torch
from torch import Tensor

from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitMixin,
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood

from bochan.models.components.robust import SafeDeepcopyMixin


__all__ = [
    "SafeRobustRelevancePursuitSingleTaskGP",
    "SafeRobustRelevancePursuitMixedSingleTaskGP",
]


class SafeRobustRelevancePursuitSingleTaskGP(
    SafeDeepcopyMixin,
    RobustRelevancePursuitSingleTaskGP,
):
    """
    deepcopy-safe な BoTorch RobustRelevancePursuitSingleTaskGP。

    Args:
        train_X: 訓練入力。shape は通常 ``[n, d]``。
        train_Y: 訓練ターゲット。shape は通常 ``[n, 1]``。
        train_Yvar: 既知観測ノイズ。
        likelihood: 任意の likelihood。
        outcome_transform: outcome transform。
        input_transform: input transform。
        convex_parameterization: BoTorch RRP の convex parameterization 設定。
        prior_mean_of_support: support size の prior mean。
        cache_model_trace: model trace を cache するか。

    Notes:
        - ``train_inputs_raw[0]`` には raw train_X を保持する。
        - ``to_standard_model()`` で通常の ``SingleTaskGP`` に戻せる。
        - qNIPV / fantasize / deepcopy で落ちにくいよう non-leaf Tensor を安全化する。
        - ``_set_transformed_inputs`` は上書きしない。BoTorch 標準の
          input_transform 更新を止めると、学習時・予測時の入力空間がずれることがある。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: Optional[InputTransform] = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: Optional[float] = None,
        cache_model_trace: bool = False,
    ) -> None:
        self._original_X = train_X.detach().clone()
        self._original_Y = train_Y.detach().clone()
        self._original_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )

        self.train_inputs_raw = (self._original_X,)
        self.train_targets_raw = self._original_Y
        self.train_Yvar_raw = self._original_Yvar

    def to_standard_model(self) -> Model:
        """通常の ``SingleTaskGP`` として再構築したモデルを返す。"""
        is_training = self.training

        likelihood = copy.deepcopy(self.likelihood) if getattr(self, "likelihood", None) is not None else None
        outcome_transform = (
            copy.deepcopy(self.outcome_transform)
            if getattr(self, "outcome_transform", None) is not None
            else None
        )
        input_transform = (
            copy.deepcopy(self.input_transform)
            if getattr(self, "input_transform", None) is not None
            else None
        )

        model = SingleTaskGP(
            train_X=self._original_X,
            train_Y=self._original_Y,
            train_Yvar=self._original_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        model.train(is_training)
        return model


class SafeRobustRelevancePursuitMixedSingleTaskGP(
    SafeDeepcopyMixin,
    MixedSingleTaskGP,
    RobustRelevancePursuitMixin,
):
    """
    deepcopy-safe な MixedSingleTaskGP + RobustRelevancePursuitMixin。

    Args:
        train_X: raw-space の mixed 入力。shape は ``[n, d]``。
        train_Y: 訓練ターゲット。
        cat_dims: raw-space におけるカテゴリ列 index。
        train_Yvar: 既知観測ノイズ。
        likelihood: 任意の likelihood。
        outcome_transform: outcome transform。
        input_transform: input transform。カテゴリ列を変換しないものを推奨。
        convex_parameterization: BoTorch RRP の convex parameterization 設定。
        prior_mean_of_support: support size の prior mean。
        cache_model_trace: model trace を cache するか。

    Notes:
        - regression の RRP は feature relevance pursuit として扱う。
        - そのため RRP mixin の ``dim`` は特徴量次元 ``d`` にする。
        - 旧実装の ``dim=n`` は train-point outlier RRP の意味に近いため、
          classification / ordinal 側とは分けて考える。
        - ``_set_transformed_inputs`` は上書きしない。カテゴリ列を除外した
          input_transform などの BoTorch 標準更新をそのまま使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: Optional[InputTransform] = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: Optional[float] = None,
        cache_model_trace: bool = False,
    ) -> None:
        self._original_X = train_X.detach().clone()
        self._original_Y = train_Y.detach().clone()
        self._original_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()
        self._cat_dims = [int(i) for i in cat_dims]

        MixedSingleTaskGP.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=list(self._cat_dims),
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        RobustRelevancePursuitMixin.__init__(
            self,
            base_likelihood=self.likelihood,
            dim=train_X.shape[-1],
            prior_mean_of_support=prior_mean_of_support,
            convex_parameterization=convex_parameterization,
            cache_model_trace=cache_model_trace,
        )

        self.train_inputs_raw = (self._original_X,)
        self.train_targets_raw = self._original_Y
        self.train_Yvar_raw = self._original_Yvar

    def to_standard_model(self) -> Model:
        """通常の ``MixedSingleTaskGP`` として再構築したモデルを返す。"""
        is_training = self.training

        likelihood = copy.deepcopy(self.likelihood) if getattr(self, "likelihood", None) is not None else None
        outcome_transform = (
            copy.deepcopy(self.outcome_transform)
            if getattr(self, "outcome_transform", None) is not None
            else None
        )
        input_transform = (
            copy.deepcopy(self.input_transform)
            if getattr(self, "input_transform", None) is not None
            else None
        )

        model = MixedSingleTaskGP(
            train_X=self._original_X,
            train_Y=self._original_Y,
            cat_dims=list(self._cat_dims),
            train_Yvar=self._original_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        model.train(is_training)
        return model
