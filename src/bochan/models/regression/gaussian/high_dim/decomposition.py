from __future__ import annotations

"""PCA / REMBO による高次元 regression GP wrapper。

配置想定:
    ``bochan.models.regression.high_dim.decomposition``

このファイルでは、PCA / REMBO の射影器そのものは
``bochan.models.components.decomposition`` に置く前提とする。共通 wrapper / helper は
``bochan.models.components.projected`` および
``bochan.models.components.projected_utils`` を利用する。
"""

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.input import InputTransform

from bochan.models.components.decomposition import (
    PCAConfig,
    REMBOConfig,
    PCATransformer,
    REMBOTransformer,
)
from bochan.models.components.projected import _BaseProjectedModel, _BaseProjectedMixedModel
from bochan.models.components.projected_utils import (
    _apply_input_transform_for_training,
    _clone_fitted_pca,
    _clone_fitted_rembo,
    _clone_input_transform,
    _concat_optional_noise,
    _ensure_2d_train_Y,
    _prepare_original_space_conditioning_data,
    _prepare_raw_input_transform_for_mixed,
    _resolve_latent_dim,
)


__all__ = [
    "PCASingleTaskGP",
    "REMBOSingleTaskGP",
    "PCAMixedSingleTaskGP",
    "REMBOMixedSingleTaskGP",
]


class _BaseProjectedSingleTaskGP(_BaseProjectedModel):
    """SingleTaskGP を base_model とする射影 regression wrapper。"""

    projector_name: str = "projector"

    def _init_common_state(
        self,
        *,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        input_transform: Optional[InputTransform],
    ) -> Tensor:
        if train_X.ndim != 2:
            raise ValueError("train_X must be 2D tensor [n, d].")
        train_Y = _ensure_2d_train_Y(train_Y)
        self.input_dim_original = train_X.shape[-1]
        self.input_transform = _clone_input_transform(input_transform)
        self._raw_train_X = train_X.detach().clone()
        self._train_targets = train_Y
        self.train_Yvar_original = train_Yvar
        self._preproject_train_X = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )
        return train_Y

    def _build_base_model(
        self,
        *,
        projected_train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        likelihood: Any | None,
        covar_module: Any | None,
        mean_module: Any | None,
        outcome_transform: Any | None,
    ) -> SingleTaskGP:
        return SingleTaskGP(
            train_X=projected_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=None,
        )

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ):
        """raw-space の新規観測で条件付けした wrapper を返す。"""
        X_new, Y_new, Yvar_new = _prepare_original_space_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.input_dim_original,
            force_2d_Y=True,
        )
        X_new = X_new.to(self.train_input_raw)
        Y_new = Y_new.to(dtype=self.train_targets.dtype, device=self.train_targets.device)
        if Yvar_new is not None:
            Yvar_new = Yvar_new.to(dtype=X_new.dtype, device=X_new.device)

        X_full = torch.cat([self.train_input_raw, X_new], dim=0)
        Y_full = torch.cat([self.train_targets, Y_new], dim=0)
        Yvar_full = _concat_optional_noise(
            old_Y=self.train_targets,
            old_Yvar=self.train_Yvar_original,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=X_full.dtype,
            device=X_full.device,
        )

        new_base = self.base_model.condition_on_observations(
            self.transform_inputs(X_new),
            Y_new,
            noise=Yvar_new,
            **kwargs,
        )

        return self._rebuild_with_new_data(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            base_model=new_base,
        )

    def _rebuild_with_new_data(
        self,
        *,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        base_model: SingleTaskGP,
    ):
        raise NotImplementedError


class PCASingleTaskGP(_BaseProjectedSingleTaskGP):
    """PCA 射影後の低次元空間で学習する SingleTaskGP wrapper。

    Args:
        train_X: raw-space の訓練入力。shape は ``[n, d_original]``。
        train_Y: 訓練ターゲット。``[n]`` または ``[n, m]``。
        train_Yvar: 既知観測ノイズ。
        likelihood: base ``SingleTaskGP`` に渡す likelihood。
        covar_module: base ``SingleTaskGP`` に渡す kernel。
        mean_module: base ``SingleTaskGP`` に渡す mean module。
        outcome_transform: base ``SingleTaskGP`` に渡す outcome transform。
        input_transform: raw-space に適用する input transform。内部 GP には渡さない。
        pca_config: 既存の PCAConfig。
        latent_dim: PCA の低次元次元数。
        n_components: ``latent_dim`` の後方互換 alias。
        fitted_pca: fit 済み PCA transformer。condition 時の projector 固定に使う。
        base_model: 既存 base model。condition_on_observations 後の再構築で使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Any | None = None,
        covar_module: Any | None = None,
        mean_module: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: InputTransform | None = None,
        pca_config: PCAConfig | None = None,
        latent_dim: int | None = None,
        n_components: int | None = None,
        fitted_pca: PCATransformer | None = None,
        base_model: SingleTaskGP | None = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
        )
        dim = _resolve_latent_dim(
            latent_dim=latent_dim,
            n_components=n_components,
            default=train_X.shape[-1],
        )
        self.pca_config = copy.deepcopy(pca_config) if pca_config is not None else PCAConfig(n_components=dim)
        self.pca = _clone_fitted_pca(fitted_pca) if fitted_pca is not None else PCATransformer(self.pca_config)
        if fitted_pca is None:
            self.pca.fit(self.preproject_train_input)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        return self.pca.transform(X)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor], base_model: SingleTaskGP):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=_clone_input_transform(self.input_transform),
            pca_config=copy.deepcopy(self.pca_config),
            fitted_pca=_clone_fitted_pca(self.pca),
            base_model=base_model,
        )


class REMBOSingleTaskGP(_BaseProjectedSingleTaskGP):
    """REMBO 固定ランダム射影後の低次元空間で学習する SingleTaskGP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Any | None = None,
        covar_module: Any | None = None,
        mean_module: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: InputTransform | None = None,
        rembo_config: REMBOConfig | None = None,
        latent_dim: int | None = None,
        n_components: int | None = None,
        fitted_rembo: REMBOTransformer | None = None,
        seed: int = 42,
        base_model: SingleTaskGP | None = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
        )
        dim = _resolve_latent_dim(
            latent_dim=latent_dim,
            n_components=n_components,
            default=train_X.shape[-1],
        )
        self.rembo_config = copy.deepcopy(rembo_config) if rembo_config is not None else REMBOConfig(n_components=dim, seed=seed)
        self.rembo = _clone_fitted_rembo(fitted_rembo) if fitted_rembo is not None else REMBOTransformer(self.rembo_config)
        if fitted_rembo is None:
            self.rembo.fit(self.preproject_train_input)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        return self.rembo.transform(X)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor], base_model: SingleTaskGP):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=_clone_input_transform(self.input_transform),
            rembo_config=copy.deepcopy(self.rembo_config),
            fitted_rembo=_clone_fitted_rembo(self.rembo),
            base_model=base_model,
        )


class _BaseProjectedMixedSingleTaskGP(_BaseProjectedMixedModel):
    """MixedSingleTaskGP を base_model とする射影 regression wrapper。"""

    def _init_common_state(
        self,
        *,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]],
        train_Yvar: Optional[Tensor],
        input_transform: Optional[InputTransform],
    ) -> Tensor:
        if train_X.ndim != 2:
            raise ValueError("train_X must be 2D tensor [n, d].")
        train_Y = _ensure_2d_train_Y(train_Y)
        self._raw_train_X = train_X.detach().clone()
        self._setup_mixed_dims(input_dim=train_X.shape[-1], cat_dims=cat_dims, category_counts=category_counts)
        self._validate_categorical_values(train_X)
        self.input_transform = _prepare_raw_input_transform_for_mixed(
            _clone_input_transform(input_transform),
            input_dim=train_X.shape[-1],
            cont_dims=self.cont_dims,
            cat_dims=self.cat_dims,
        )
        self._train_targets = train_Y
        self.train_Yvar_original = train_Yvar
        self._preproject_train_X = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name=f"{self.__class__.__name__}.input_transform",
        )
        self._validate_categorical_values(self.preproject_train_input)
        return train_Y

    def _build_base_model(
        self,
        *,
        projected_train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor],
        cont_kernel_factory: Any | None,
        likelihood: Any | None,
        outcome_transform: Any | None,
    ) -> MixedSingleTaskGP:
        latent_cat_dims = list(range(self.latent_dim, self.latent_dim + len(self.cat_dims)))
        return MixedSingleTaskGP(
            train_X=projected_train_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=None,
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, noise: Optional[Tensor] = None, **kwargs: Any):
        X_new, Y_new, Yvar_new = _prepare_original_space_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.input_dim_original,
            force_2d_Y=True,
        )
        X_new = X_new.to(self.train_input_raw)
        self._validate_categorical_values(X_new)
        Y_new = Y_new.to(dtype=self.train_targets.dtype, device=self.train_targets.device)
        if Yvar_new is not None:
            Yvar_new = Yvar_new.to(dtype=X_new.dtype, device=X_new.device)

        X_full = torch.cat([self.train_input_raw, X_new], dim=0)
        Y_full = torch.cat([self.train_targets, Y_new], dim=0)
        Yvar_full = _concat_optional_noise(
            old_Y=self.train_targets,
            old_Yvar=self.train_Yvar_original,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=X_full.dtype,
            device=X_full.device,
        )
        new_base = self.base_model.condition_on_observations(
            self.transform_inputs(X_new),
            Y_new,
            noise=Yvar_new,
            **kwargs,
        )
        return self._rebuild_with_new_data(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            base_model=new_base,
        )

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor], base_model: MixedSingleTaskGP):
        raise NotImplementedError


class PCAMixedSingleTaskGP(_BaseProjectedMixedSingleTaskGP):
    """連続列だけ PCA 射影し、カテゴリ列を末尾に保持する MixedSingleTaskGP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        cont_kernel_factory: Any | None = None,
        likelihood: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: InputTransform | None = None,
        pca_config: PCAConfig | None = None,
        latent_dim: int | None = None,
        n_components: int | None = None,
        fitted_pca: PCATransformer | None = None,
        category_counts: Optional[dict[int, int]] = None,
        base_model: MixedSingleTaskGP | None = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            category_counts=category_counts,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
        )
        dim = _resolve_latent_dim(latent_dim=latent_dim, n_components=n_components, default=len(self.cont_dims))
        self.pca_config = copy.deepcopy(pca_config) if pca_config is not None else PCAConfig(n_components=dim)
        self.pca = _clone_fitted_pca(fitted_pca) if fitted_pca is not None else PCATransformer(self.pca_config)
        if fitted_pca is None:
            self.pca.fit(self.preproject_train_input[..., self.cont_dims])
        self.latent_dim = int(self.pca_config.n_components)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
        )
        self._cont_kernel_factory = cont_kernel_factory

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        x_cont = self.pca.transform(X[..., self.cont_dims])
        return self._project_continuous_and_concat_categorical(X, x_cont)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor], base_model: MixedSingleTaskGP):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=list(self.cat_dims),
            category_counts=copy.deepcopy(self.category_counts),
            train_Yvar=train_Yvar,
            cont_kernel_factory=self._cont_kernel_factory,
            input_transform=_clone_input_transform(self.input_transform),
            pca_config=copy.deepcopy(self.pca_config),
            fitted_pca=_clone_fitted_pca(self.pca),
            base_model=base_model,
        )


class REMBOMixedSingleTaskGP(_BaseProjectedMixedSingleTaskGP):
    """連続列だけ REMBO 射影し、カテゴリ列を末尾に保持する MixedSingleTaskGP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        cont_kernel_factory: Any | None = None,
        likelihood: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: InputTransform | None = None,
        rembo_config: REMBOConfig | None = None,
        latent_dim: int | None = None,
        n_components: int | None = None,
        fitted_rembo: REMBOTransformer | None = None,
        seed: int = 42,
        category_counts: Optional[dict[int, int]] = None,
        base_model: MixedSingleTaskGP | None = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            category_counts=category_counts,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
        )
        dim = _resolve_latent_dim(latent_dim=latent_dim, n_components=n_components, default=len(self.cont_dims))
        self.rembo_config = copy.deepcopy(rembo_config) if rembo_config is not None else REMBOConfig(n_components=dim, seed=seed)
        self.rembo = _clone_fitted_rembo(fitted_rembo) if fitted_rembo is not None else REMBOTransformer(self.rembo_config)
        if fitted_rembo is None:
            self.rembo.fit(self.preproject_train_input[..., self.cont_dims])
        self.latent_dim = int(self.rembo_config.n_components)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
        )
        self._cont_kernel_factory = cont_kernel_factory

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        x_cont = self.rembo.transform(X[..., self.cont_dims])
        return self._project_continuous_and_concat_categorical(X, x_cont)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor], base_model: MixedSingleTaskGP):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=list(self.cat_dims),
            category_counts=copy.deepcopy(self.category_counts),
            train_Yvar=train_Yvar,
            cont_kernel_factory=self._cont_kernel_factory,
            input_transform=_clone_input_transform(self.input_transform),
            rembo_config=copy.deepcopy(self.rembo_config),
            fitted_rembo=_clone_fitted_rembo(self.rembo),
            base_model=base_model,
        )
