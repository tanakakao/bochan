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
from bochan.models.components.poisson import (
    PoissonLink,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    check_categorical_columns_unchanged,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    prepare_count_targets,
)
from bochan.models.regression.non_gaussian.poisson import (
    PoissonGPModel,
    PoissonLogLikelihood,
    PoissonMixedGPModel,
)


def _clone_fitted_transformer(transformer):
    """PCATransformer / REMBOTransformer を簡易 clone する。"""
    new = transformer.__class__(copy.deepcopy(transformer.config))
    for name in ("mean_", "scale_", "components_", "projection_"):
        if hasattr(transformer, name):
            v = getattr(transformer, name)
            setattr(new, name, None if v is None else v.detach().clone())
    return new


class _BaseProjectedPoissonModel(Model):
    """PCA / REMBO Poisson wrapper の共通基底。"""

    def _set_transformed_inputs(self) -> None:
        return None

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        return (self._raw_train_X,)

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self._raw_train_X,)

    @property
    def projected_train_inputs(self) -> tuple[Tensor]:
        return (self._projected_train_X,)

    @property
    def preproject_train_inputs(self) -> tuple[Tensor]:
        return (self._preproject_train_X,)

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

    def predict_rate(self, X: Tensor) -> Tensor:
        return self.base_model.predict_rate(self.transform_inputs(X))

    def predict_count(self, X: Tensor) -> Tensor:
        return self.predict_rate(X)

    def make_mll(self):
        return self.base_model.make_mll()


class _ContinuousProjectedPoissonModel(_BaseProjectedPoissonModel):
    transformer_cls = None
    config_cls = None

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        latent_dim: int = 8,
        n_components: Optional[int] = None,
        config: Any | None = None,
        projector: Any | None = None,
        likelihood: Optional[PoissonLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        seed: int = 42,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_count_targets(train_Y, train_X)
        self.input_dim_original = train_X.shape[-1]
        self.latent_dim = int(n_components if n_components is not None else latent_dim)
        self.input_transform = clone_input_transform(input_transform)
        self.num_inducing_points = int(num_inducing_points)
        self.seed = int(seed)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_rate = float(min_rate)

        pre_X = apply_input_transform_for_training(train_X, self.input_transform, name=f"{self.__class__.__name__}.input_transform")
        if projector is None:
            if config is None:
                try:
                    config = self.config_cls(n_components=self.latent_dim, seed=seed)
                except TypeError:
                    config = self.config_cls(n_components=self.latent_dim)
            projector = self.transformer_cls(config)
            projector.fit(pre_X)
        self.projector = projector
        projected_X = projector.transform(pre_X)
        self.latent_dim = projected_X.shape[-1]

        self._raw_train_X = train_X.detach().clone()
        self._preproject_train_X = pre_X.detach().clone()
        self._projected_train_X = projected_X.detach().clone()
        self._train_targets = train_Y

        self.base_model = PoissonGPModel(
            train_X=projected_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=None,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )

    def transform_inputs(self, X: Tensor) -> Tensor:
        X = torch.as_tensor(X, device=self._raw_train_X.device, dtype=self._raw_train_X.dtype)
        if X.shape[-1] == self.latent_dim:
            return X
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(f"Expected raw dim {self.input_dim_original} or latent dim {self.latent_dim}, got {X.shape[-1]}.")
        X_pre = apply_input_transform_for_eval(X, self.input_transform)
        return self.projector.transform(X_pre)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for projected Poisson models.")
        X = torch.as_tensor(X, device=self._raw_train_X.device, dtype=self._raw_train_X.dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_count_targets(Y, X)
        return self.__class__(
            train_X=torch.cat([self._raw_train_X, X], dim=0),
            train_Y=torch.cat([self._train_targets, Y], dim=0),
            latent_dim=self.latent_dim,
            projector=_clone_fitted_transformer(self.projector),
            likelihood=copy.deepcopy(self.base_model.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            num_inducing_points=self.num_inducing_points,
            seed=self.seed,
            link=self.link,
            exp_clip=self.exp_clip,
            min_rate=self.min_rate,
        )


class PCAPoissonGPModel(_ContinuousProjectedPoissonModel):
    """PCA 射影後の低次元空間で学習する Poisson GP。"""
    transformer_cls = PCATransformer
    config_cls = PCAConfig

    def __init__(self, *args, pca_config: Optional[PCAConfig] = None, **kwargs) -> None:
        super().__init__(*args, config=pca_config, **kwargs)


class REMBOPoissonGPModel(_ContinuousProjectedPoissonModel):
    """REMBO 射影後の低次元空間で学習する Poisson GP。"""
    transformer_cls = REMBOTransformer
    config_cls = REMBOConfig

    def __init__(self, *args, rembo_config: Optional[REMBOConfig] = None, **kwargs) -> None:
        super().__init__(*args, config=rembo_config, **kwargs)


class _MixedProjectedPoissonModel(_BaseProjectedPoissonModel):
    transformer_cls = None
    config_cls = None

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        latent_dim: int = 8,
        n_components: Optional[int] = None,
        config: Any | None = None,
        projector: Any | None = None,
        likelihood: Optional[PoissonLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        seed: int = 42,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_count_targets(train_Y, train_X)
        self.input_dim_original = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, self.input_dim_original)
        self.cont_dims = get_cont_dims(self.input_dim_original, self.cat_dims)
        self.latent_dim = int(n_components if n_components is not None else latent_dim)
        self.input_transform = clone_input_transform(input_transform)
        self.num_inducing_points = int(num_inducing_points)
        self.seed = int(seed)
        self.link = link
        self.exp_clip = float(exp_clip)
        self.min_rate = float(min_rate)

        pre_X = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name=f"{self.__class__.__name__}.input_transform",
        )
        check_categorical_columns_unchanged(train_X, pre_X, self.cat_dims)

        if projector is None:
            if config is None:
                try:
                    config = self.config_cls(n_components=self.latent_dim, seed=seed)
                except TypeError:
                    config = self.config_cls(n_components=self.latent_dim)
            projector = self.transformer_cls(config)
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

        self.base_model = PoissonMixedGPModel(
            train_X=projected_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            likelihood=likelihood,
            input_transform=None,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )

    def transform_inputs(self, X: Tensor) -> Tensor:
        X = torch.as_tensor(X, device=self._raw_train_X.device, dtype=self._raw_train_X.dtype)
        internal_dim = self.latent_dim + len(self.cat_dims)
        if X.shape[-1] == internal_dim:
            return X
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(f"Expected raw dim {self.input_dim_original} or internal dim {internal_dim}.")
        X_pre = apply_input_transform_for_eval(X, self.input_transform, cat_dims=self.cat_dims)
        x_cont = self.projector.transform(X_pre[..., self.cont_dims])
        return torch.cat([x_cont, X_pre[..., self.cat_dims]], dim=-1)


class PCAPoissonMixedGPModel(_MixedProjectedPoissonModel):
    """連続列だけ PCA 射影し、カテゴリ列を保持する mixed Poisson GP。"""
    transformer_cls = PCATransformer
    config_cls = PCAConfig

    def __init__(self, *args, pca_config: Optional[PCAConfig] = None, **kwargs) -> None:
        super().__init__(*args, config=pca_config, **kwargs)


class REMBOPoissonMixedGPModel(_MixedProjectedPoissonModel):
    """連続列だけ REMBO 射影し、カテゴリ列を保持する mixed Poisson GP。"""
    transformer_cls = REMBOTransformer
    config_cls = REMBOConfig

    def __init__(self, *args, rembo_config: Optional[REMBOConfig] = None, **kwargs) -> None:
        super().__init__(*args, config=rembo_config, **kwargs)


__all__ = [
    "PCAPoissonGPModel",
    "REMBOPoissonGPModel",
    "PCAPoissonMixedGPModel",
    "REMBOPoissonMixedGPModel",
]
