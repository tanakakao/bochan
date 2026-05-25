from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform

from bochan.models.components.decomposition import (
    PCAConfig,
    REMBOConfig,
    PCATransformer,
    REMBOTransformer,
)
from bochan.models.components.gamma import (
    GammaLink,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    check_categorical_columns_unchanged,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    prepare_positive_targets,
)
from bochan.models.regression.non_gaussian.gamma import (
    GammaGPModel,
    GammaLogLikelihood,
    GammaMixedGPModel,
)


def _clone_fitted_pca(pca: PCATransformer) -> PCATransformer:
    """fit 済み PCA transformer を複製する。"""
    new = PCATransformer(copy.deepcopy(pca.config))
    for name in ("mean_", "scale_", "components_"):
        if hasattr(pca, name):
            value = getattr(pca, name)
            setattr(new, name, None if value is None else value.detach().clone())
    return new


def _clone_fitted_rembo(rembo: REMBOTransformer) -> REMBOTransformer:
    """fit 済み REMBO transformer を複製する。"""
    new = REMBOTransformer(copy.deepcopy(rembo.config))
    for name in ("mean_", "scale_", "projection_"):
        if hasattr(rembo, name):
            value = getattr(rembo, name)
            setattr(new, name, None if value is None else value.detach().clone())
    return new


class _BaseProjectedGammaModel(Model):
    """PCA / REMBO Gamma wrapper の共通基底。"""

    def _set_transformed_inputs(self) -> None:
        return None

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        return (self._raw_train_X,)

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self._raw_train_X,)

    @property
    def preproject_train_inputs(self) -> tuple[Tensor]:
        return (self._preproject_train_X,)

    @property
    def projected_train_inputs(self) -> tuple[Tensor]:
        return (self._projected_train_X,)

    @property
    def train_targets(self) -> Tensor:
        return self._train_targets

    @property
    def likelihood(self):
        return self.base_model.likelihood

    @property
    def model(self):
        return self.base_model.model

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def latent_posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return self.base_model.latent_posterior(self.transform_inputs(X), *args, **kwargs)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return self.base_model.posterior(self.transform_inputs(X), *args, **kwargs)

    def predict_mean(self, X: Tensor) -> Tensor:
        return self.base_model.predict_mean(self.transform_inputs(X))

    def predict_concentration(self) -> Tensor:
        return self.base_model.predict_concentration()

    def make_mll(self):
        return self.base_model.make_mll()


class _ContinuousProjectedGammaModel(_BaseProjectedGammaModel):
    """連続入力用 projected Gamma wrapper。"""

    def _to_preprojection_space(self, X: Tensor) -> Tensor:
        X = torch.as_tensor(X, device=self._raw_train_X.device, dtype=self._raw_train_X.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.shape[-1] == self.input_dim_original:
            return apply_input_transform_for_eval(X, self.input_transform)
        if X.shape[-1] == self.latent_dim:
            return X
        raise ValueError(
            f"Expected raw dim {self.input_dim_original} or latent dim {self.latent_dim}, got {X.shape[-1]}."
        )

    def transform_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.latent_dim:
            return X
        X_pre = self._to_preprojection_space(X)
        return self.projector.transform(X_pre)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for projected Gamma models.")
        X = torch.as_tensor(X, device=self._raw_train_X.device, dtype=self._raw_train_X.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_positive_targets(Y, X, min_value=self.min_mean)

        new_X = torch.cat([self._raw_train_X, X], dim=0)
        new_Y = torch.cat([self._train_targets, Y], dim=0)
        return self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            likelihood=copy.deepcopy(self.base_model.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            projector=self._clone_projector(),
            latent_dim=self.latent_dim,
            num_inducing_points=self.num_inducing_points,
            link=self.link,
            exp_clip=self.exp_clip,
            min_mean=self.min_mean,
        )


class PCAGammaGPModel(_ContinuousProjectedGammaModel):
    """PCA 射影後の低次元空間で学習する Gamma GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        latent_dim: int = 8,
        n_components: Optional[int] = None,
        pca_config: Optional[PCAConfig] = None,
        projector: Optional[PCATransformer] = None,
        likelihood: Optional[GammaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        link: GammaLink = "softplus",
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
        self.input_dim_original = train_X.shape[-1]
        self.latent_dim = int(n_components if n_components is not None else latent_dim)
        self.input_transform = clone_input_transform(input_transform)
        pre_X = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name="PCAGammaGPModel.input_transform",
        )

        if projector is None:
            cfg = pca_config or PCAConfig(n_components=self.latent_dim)
            projector = PCATransformer(cfg)
            projector.fit(pre_X)
        self.projector = projector
        projected_X = projector.transform(pre_X)
        self.latent_dim = projected_X.shape[-1]

        self._raw_train_X = train_X.detach().clone()
        self._preproject_train_X = pre_X.detach().clone()
        self._projected_train_X = projected_X.detach().clone()
        self._train_targets = train_Y

        self.num_inducing_points = int(num_inducing_points)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)

        self.base_model = GammaGPModel(
            train_X=projected_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=None,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )

    def _clone_projector(self):
        return _clone_fitted_pca(self.projector)


class REMBOGammaGPModel(_ContinuousProjectedGammaModel):
    """REMBO 射影後の低次元空間で学習する Gamma GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        latent_dim: int = 8,
        n_components: Optional[int] = None,
        rembo_config: Optional[REMBOConfig] = None,
        projector: Optional[REMBOTransformer] = None,
        likelihood: Optional[GammaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        seed: int = 42,
        link: GammaLink = "softplus",
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
        self.input_dim_original = train_X.shape[-1]
        self.latent_dim = int(n_components if n_components is not None else latent_dim)
        self.input_transform = clone_input_transform(input_transform)
        pre_X = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name="REMBOGammaGPModel.input_transform",
        )

        if projector is None:
            cfg = rembo_config or REMBOConfig(n_components=self.latent_dim, seed=seed)
            projector = REMBOTransformer(cfg)
            projector.fit(pre_X)
        self.projector = projector
        projected_X = projector.transform(pre_X)
        self.latent_dim = projected_X.shape[-1]

        self._raw_train_X = train_X.detach().clone()
        self._preproject_train_X = pre_X.detach().clone()
        self._projected_train_X = projected_X.detach().clone()
        self._train_targets = train_Y

        self.num_inducing_points = int(num_inducing_points)
        self.seed = int(seed)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)

        self.base_model = GammaGPModel(
            train_X=projected_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=None,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )

    def _clone_projector(self):
        return _clone_fitted_rembo(self.projector)


class _MixedProjectedGammaModel(_BaseProjectedGammaModel):
    """mixed 入力用 projected Gamma wrapper。"""

    def _to_internal(self, X: Tensor) -> Tensor:
        X = torch.as_tensor(X, device=self._raw_train_X.device, dtype=self._raw_train_X.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        internal_dim = self.latent_dim + len(self.cat_dims)
        if X.shape[-1] == internal_dim:
            return X
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(f"Expected raw dim {self.input_dim_original} or internal dim {internal_dim}.")
        X_pre = apply_input_transform_for_eval(X, self.input_transform, cat_dims=self.cat_dims)
        x_cont = self.projector.transform(X_pre[..., self.cont_dims])
        x_cat = X_pre[..., self.cat_dims]
        return torch.cat([x_cont, x_cat], dim=-1)

    def transform_inputs(self, X: Tensor) -> Tensor:
        return self._to_internal(X)


class PCAGammaMixedGPModel(_MixedProjectedGammaModel):
    """連続列だけ PCA 射影し、カテゴリ列を保持する mixed Gamma GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        latent_dim: int = 8,
        n_components: Optional[int] = None,
        pca_config: Optional[PCAConfig] = None,
        projector: Optional[PCATransformer] = None,
        likelihood: Optional[GammaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        link: GammaLink = "softplus",
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
        self.input_dim_original = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, self.input_dim_original)
        self.cont_dims = get_cont_dims(self.input_dim_original, self.cat_dims)
        self.latent_dim = int(n_components if n_components is not None else latent_dim)
        self.input_transform = clone_input_transform(input_transform)
        pre_X = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name="PCAGammaMixedGPModel.input_transform",
        )
        check_categorical_columns_unchanged(train_X, pre_X, self.cat_dims)

        if projector is None:
            cfg = pca_config or PCAConfig(n_components=self.latent_dim)
            projector = PCATransformer(cfg)
            projector.fit(pre_X[..., self.cont_dims])
        self.projector = projector

        x_cont = projector.transform(pre_X[..., self.cont_dims])
        self.latent_dim = x_cont.shape[-1]
        projected_X = torch.cat([x_cont, pre_X[..., self.cat_dims]], dim=-1)
        latent_cat_dims = list(range(self.latent_dim, projected_X.shape[-1]))

        self._raw_train_X = train_X.detach().clone()
        self._preproject_train_X = pre_X.detach().clone()
        self._projected_train_X = projected_X.detach().clone()
        self._train_targets = train_Y

        self.num_inducing_points = int(num_inducing_points)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)

        self.base_model = GammaMixedGPModel(
            train_X=projected_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            likelihood=likelihood,
            input_transform=None,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )


class REMBOGammaMixedGPModel(PCAGammaMixedGPModel):
    """連続列だけ REMBO 射影し、カテゴリ列を保持する mixed Gamma GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        latent_dim: int = 8,
        n_components: Optional[int] = None,
        rembo_config: Optional[REMBOConfig] = None,
        projector: Optional[REMBOTransformer] = None,
        likelihood: Optional[GammaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        seed: int = 42,
        link: GammaLink = "softplus",
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
    ) -> None:
        Model.__init__(self)
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
        self.input_dim_original = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, self.input_dim_original)
        self.cont_dims = get_cont_dims(self.input_dim_original, self.cat_dims)
        self.latent_dim = int(n_components if n_components is not None else latent_dim)
        self.input_transform = clone_input_transform(input_transform)
        pre_X = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name="REMBOGammaMixedGPModel.input_transform",
        )
        check_categorical_columns_unchanged(train_X, pre_X, self.cat_dims)

        if projector is None:
            cfg = rembo_config or REMBOConfig(n_components=self.latent_dim, seed=seed)
            projector = REMBOTransformer(cfg)
            projector.fit(pre_X[..., self.cont_dims])
        self.projector = projector

        x_cont = projector.transform(pre_X[..., self.cont_dims])
        self.latent_dim = x_cont.shape[-1]
        projected_X = torch.cat([x_cont, pre_X[..., self.cat_dims]], dim=-1)
        latent_cat_dims = list(range(self.latent_dim, projected_X.shape[-1]))

        self._raw_train_X = train_X.detach().clone()
        self._preproject_train_X = pre_X.detach().clone()
        self._projected_train_X = projected_X.detach().clone()
        self._train_targets = train_Y

        self.num_inducing_points = int(num_inducing_points)
        self.seed = int(seed)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)

        self.base_model = GammaMixedGPModel(
            train_X=projected_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            likelihood=likelihood,
            input_transform=None,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )


__all__ = [
    "PCAGammaGPModel",
    "REMBOGammaGPModel",
    "PCAGammaMixedGPModel",
    "REMBOGammaMixedGPModel",
]
