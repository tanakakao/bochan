from __future__ import annotations

"""PCA / REMBO などの固定射影 wrapper の共通 base。

このモジュールは、regression / classification / ordinal に共通する
raw-space / preproject-space / projected-space の管理を担当する。

各タスク固有の posterior 解釈、分類確率、順序 utility などは、各タスク側の
``high_dim/decomposition.py`` に残す。
"""

from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.components.projected_utils import (
    _apply_input_transform_for_eval,
    _apply_input_transform_for_training,
    _check_categorical_columns_unchanged,
    _get_cont_dims,
    flatten_projected_one_to_many_point_axes,
    _normalize_dims,
)


__all__ = [
    "_BaseProjectedModel",
    "_BaseProjectedMixedModel",
]


class _BaseProjectedModel(Model):
    """固定射影付きモデルの共通 wrapper。

    この wrapper は、以下の3つの入力空間を明示的に分けて保持する。

    1. raw-space:
        ユーザーが ``posterior(X)`` や ``condition_on_observations(X, Y)`` に渡す
        元の特徴量空間。
    2. preproject-space:
        raw-space に ``input_transform`` を適用した後、PCA / REMBO 射影前の空間。
    3. projected-space:
        PCA / REMBO 後、内部の ``base_model`` が実際に見る空間。

    Notes:
        BoTorch の ``Model.eval()`` は ``input_transform`` を持つモデルに対して
        transformed training inputs の更新を試みることがある。この wrapper では
        独自の変換経路を持つため、``_set_transformed_inputs`` を no-op にしている。
    """

    def __init__(self) -> None:
        super().__init__()

    def _set_transformed_inputs(self) -> None:
        """BoTorch の eval 時 transformed input 自動更新を無効化する。"""
        return None

    @property
    def model(self):
        """内部 base_model の ``model`` 属性があれば返す。"""
        return getattr(self.base_model, "model", self.base_model)

    @property
    def likelihood(self):
        """内部 base_model の likelihood。"""
        return self.base_model.likelihood

    @property
    def num_outputs(self) -> int:
        """出力次元数。"""
        return int(getattr(self.base_model, "num_outputs", 1))

    @property
    def batch_shape(self) -> torch.Size:
        """base_model の batch_shape。なければ空 batch。"""
        return getattr(self.base_model, "batch_shape", torch.Size())

    def make_mll(self, **kwargs: Any):
        """内部 ``base_model`` 用の MLL を構築する。

        ``base_model`` が独自の ``make_mll`` を持つ場合は、``beta`` などの
        タスク固有引数を含めてそのまま委譲する。ordinal projected model では
        ``make_ordinal_mll`` を使い、それ以外の exact GP では ``beta`` を無視して
        ``ExactMarginalLogLikelihood`` を構築する。
        """
        if hasattr(self.base_model, "make_mll"):
            return self.base_model.make_mll(**kwargs)

        mll_kwargs = dict(kwargs)
        mll_kwargs.pop("beta", None)

        if hasattr(self, "ordinal_likelihood"):
            from bochan.fit import make_ordinal_mll

            return make_ordinal_mll(self, **mll_kwargs)

        if mll_kwargs:
            names = ", ".join(sorted(mll_kwargs))
            raise TypeError(
                "Exact projected models received unsupported MLL keyword arguments: "
                f"{names}."
            )
        return ExactMarginalLogLikelihood(self.base_model.likelihood, self.base_model)

    @property
    def train_input_raw(self) -> Tensor:
        return self._raw_train_X

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        return (self.train_input_raw,)

    @property
    def train_input(self) -> Tensor:
        return self.train_input_raw

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self.train_input,)

    @property
    def preproject_train_input(self) -> Tensor:
        return self._preproject_train_X

    @property
    def preproject_train_inputs(self) -> tuple[Tensor]:
        return (self.preproject_train_input,)

    @property
    def projected_train_input(self) -> Tensor:
        return self._projected_train_X

    @property
    def projected_train_inputs(self) -> tuple[Tensor]:
        return (self.projected_train_input,)

    @property
    def train_targets(self) -> Tensor:
        return self._train_targets

    @property
    @property
    @property
    def _to_preprojection_space(self, X: Tensor) -> Tensor:
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_input_raw.device,
            dtype=self.train_input_raw.dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.shape[-1] != self.train_input_raw.shape[-1]:
            raise ValueError(
                f"Expected raw input dim {self.train_input_raw.shape[-1]}, got {X.shape[-1]}."
            )
        return _apply_input_transform_for_eval(
            X,
            getattr(self, "input_transform", None),
            cat_dims=getattr(self, "cat_dims", None),
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def transform_inputs(self, X: Tensor) -> Tensor:
        X_raw = X[0] if isinstance(X, tuple) else X
        X_pre = self._to_preprojection_space(X_raw)
        projected = self._project_preprojected_inputs(X_pre)
        return flatten_projected_one_to_many_point_axes(X_raw, projected)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return self.base_model.posterior(self.transform_inputs(X), *args, **kwargs)

    def forward(self, X: Tensor):
        return self.base_model(self.transform_inputs(X))

    def set_train_data(
        self,
        inputs: Optional[Tensor | tuple[Tensor, ...]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        if inputs is not None:
            if torch.is_tensor(inputs):
                X_raw = inputs
            else:
                X_raw = inputs[0]
            X_raw = torch.as_tensor(
                X_raw,
                device=self.train_input_raw.device,
                dtype=self.train_input_raw.dtype,
            )
            if X_raw.ndim == 1:
                X_raw = X_raw.unsqueeze(0)
            if X_raw.shape[-1] != self.train_input_raw.shape[-1]:
                raise ValueError(
                    f"Expected raw input dim {self.train_input_raw.shape[-1]}, got {X_raw.shape[-1]}."
                )
            X_pre = _apply_input_transform_for_training(
                X_raw,
                getattr(self, "input_transform", None),
                cat_dims=getattr(self, "cat_dims", None),
                name=f"{self.__class__.__name__}.input_transform",
            )
            X_proj = self._project_preprojected_inputs(X_pre)
            self._raw_train_X = X_raw.detach().clone()
            self._preproject_train_X = X_pre.detach().clone()
            self._projected_train_X = X_proj.detach().clone()
            if hasattr(self.base_model, "set_train_data"):
                self.base_model.set_train_data(inputs=X_proj, strict=strict)
            else:
                self.base_model.train_inputs = (X_proj,)

        if targets is not None:
            self._train_targets = targets
            if hasattr(self.base_model, "set_train_data"):
                self.base_model.set_train_data(targets=targets, strict=strict)
            else:
                self.base_model.train_targets = targets


class _BaseProjectedMixedModel(_BaseProjectedModel):
    def _setup_mixed_dims(
        self,
        *,
        input_dim: int,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]] = None,
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be specified for mixed projected models.")
        self.input_dim_original = int(input_dim)
        self.cat_dims = _normalize_dims(cat_dims, input_dim)
        self.cat_dims_original = list(self.cat_dims)
        self.cont_dims = _get_cont_dims(input_dim, self.cat_dims)
        self.cont_dims_original = list(self.cont_dims)
        if len(self.cont_dims) == 0:
            raise ValueError("At least one continuous dimension is required.")
        self.category_counts = self._infer_category_counts(
            getattr(self, "_raw_train_X", None),
            category_counts=category_counts,
        ) if getattr(self, "_raw_train_X", None) is not None else category_counts
        self._ignore_X_dims_scaling_check = self.cat_dims

    def _infer_category_counts(
        self,
        X: Optional[Tensor],
        *,
        category_counts: Optional[dict[int, int]] = None,
    ) -> dict[int, int]:
        inferred: dict[int, int] = {}
        if category_counts is not None:
            inferred.update({int(k): int(v) for k, v in category_counts.items()})
        if X is None:
            return inferred
        for j in self.cat_dims:
            vals = X[..., j]
            if not torch.allclose(vals, vals.round()):
                raise ValueError(f"Categorical column {j} must be integer-coded.")
            if vals.min().item() < 0:
                raise ValueError(f"Categorical column {j} must be non-negative.")
            inferred.setdefault(j, int(vals.max().item()) + 1)
            if inferred[j] <= 0:
                raise ValueError(f"category_counts[{j}] must be positive.")
        return inferred

    def _validate_categorical_values(
        self,
        X: Tensor,
        *,
        category_counts: Optional[dict[int, int]] = None,
    ) -> None:
        counts = self.category_counts if category_counts is None else category_counts
        if counts is None:
            return
        for j in self.cat_dims:
            if j not in counts:
                raise ValueError(f"category_counts must contain key {j}.")
            n_cat = int(counts[j])
            vals = X[..., j]
            if not torch.allclose(vals, vals.round()):
                raise ValueError(f"Categorical column {j} must be integer-coded.")
            if vals.min().item() < 0 or vals.max().item() > n_cat - 1:
                raise ValueError(
                    f"Categorical column {j} must be in [0, {n_cat - 1}], "
                    f"got min={vals.min().item()}, max={vals.max().item()}."
                )

    def _project_continuous_and_concat_categorical(self, X_pre: Tensor, x_cont_projected: Tensor) -> Tensor:
        x_cat = X_pre[..., self.cat_dims]
        return torch.cat([x_cont_projected, x_cat], dim=-1)

    def _to_preprojection_space(self, X: Tensor) -> Tensor:
        X_pre = super()._to_preprojection_space(X)
        _check_categorical_columns_unchanged(
            X=_BaseProjectedModel._to_preprojection_space(self, X) if False else X_pre,
            X_tf=X_pre,
            cat_dims=self.cat_dims,
        )
        self._validate_categorical_values(X_pre)
        return X_pre
