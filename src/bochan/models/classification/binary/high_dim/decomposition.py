from __future__ import annotations

"""PCA / REMBO による高次元 binary classification GP wrapper。

配置想定:
    ``bochan.models.classification.high_dim.decomposition``

このファイルは、PCA / REMBO の共通射影処理を components 側へ寄せ、
classification 固有の posterior / latent_posterior / condition_on_observations だけを扱う。
"""

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.transforms.input import InputTransform

from bochan.models.classification.base import BinaryClassificationGPModel, BinaryClassificationMixedGPModel
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
    _flatten_targets,
    _prepare_original_space_conditioning_data,
    _prepare_raw_input_transform_for_mixed,
    _resolve_latent_dim,
)


__all__ = [
    "PCABinaryClassificationGPModel",
    "REMBOBinaryClassificationGPModel",
    "PCABinaryClassificationMixedGPModel",
    "REMBOBinaryClassificationMixedGPModel",
]


def _get_variational_inducing_points(model) -> Optional[Tensor]:
    """classification model / projected wrapper から inducing points を安全に取得する。"""
    try:
        obj = getattr(model, "base_model", model)
        inner = getattr(obj, "model", obj)
        return inner.variational_strategy.inducing_points.detach().clone()
    except Exception:
        return None


class _BaseProjectedClassificationGP(_BaseProjectedModel):
    """ClassificationGPModel を base_model とする射影 wrapper。"""

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
        self.input_dim_original = train_X.shape[-1]
        self.input_transform = _clone_input_transform(input_transform)
        self._raw_train_X = train_X.detach().clone()
        train_Y = _flatten_targets(train_Y, dtype=train_X.dtype)
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
        mean_module: Any | None,
        covar_module: Any | None,
        num_inducing_points: int,
        inducing_points: Optional[Tensor],
        learn_inducing_locations: bool,
    ) -> BinaryClassificationGPModel:
        return BinaryClassificationGPModel(
            train_X=projected_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=None,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Any:
        """raw X を受け取り、projected-space 上の probability posterior を返す。"""
        post = self.base_model.posterior(
            self.transform_inputs(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        return post

    def probability_posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Any:
        """classification acquisition 用に probability posterior を明示名で返す。"""
        return self.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def latent_posterior(self, X: Tensor, posterior_transform=None, **kwargs: Any) -> Any:
        """raw X を受け取り、projected-space 上の latent f posterior を返す。"""
        post = self.base_model.latent_posterior(self.transform_inputs(X), **kwargs)
        if posterior_transform is not None:
            post = posterior_transform(post)
        return post

    def posterior_latent(self, X: Tensor, **kwargs: Any) -> Any:
        return self.latent_posterior(X, **kwargs)

    def posterior_f(self, X: Tensor, **kwargs: Any) -> Any:
        return self.latent_posterior(X, **kwargs)

    def predict_proba(self, X: Tensor) -> Tensor:
        """p(y=1|x) を返す。"""
        if hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict_proba(self.transform_inputs(X))
        return self.posterior(X).mean

    def make_mll(self, *args: Any, **kwargs: Any) -> Any:
        """projected-space の base_model 用 MLL を返す。

        fit_classifier_mll には、この MLL を渡す。MLL の model は base_model 側の
        inner latent GP になり、wrapper 側では raw-space X を保持する。
        """
        if not hasattr(self.base_model, "make_mll"):
            raise AttributeError("base_model does not define make_mll().")
        return self.base_model.make_mll(*args, **kwargs)

    @property
    def model(self):
        """base_model の inner latent model を返す。"""
        return getattr(self.base_model, "model", self.base_model)

    @property
    def likelihood(self):
        return getattr(self.base_model, "likelihood", None)

    @property
    def train_inputs(self):
        return (self.train_input_raw,)

    @property
    def train_targets(self):
        return self._train_targets

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ):
        """raw-space の新規観測を追加した wrapper を返す。"""
        _ = kwargs
        X_new, Y_new, Yvar_new = _prepare_original_space_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.input_dim_original,
            force_2d_Y=False,
        )
        X_new = X_new.to(self.train_input_raw)
        Y_new = _flatten_targets(Y_new, dtype=self.train_targets.dtype).to(self.train_targets.device)
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

        new_model = self._rebuild_with_new_data(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
        )
        try:
            new_model.load_state_dict(self.state_dict(), strict=False)
        except Exception:
            pass
        new_model.eval()
        return new_model

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]):
        raise NotImplementedError


class PCABinaryClassificationGPModel(_BaseProjectedClassificationGP):
    """PCA 射影後の低次元空間で学習する binary classification GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Any | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Any | None = None,
        covar_module: Any | None = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        pca_config: Optional[PCAConfig] = None,
        latent_dim: Optional[int] = None,
        n_components: Optional[int] = None,
        fitted_pca: Optional[PCATransformer] = None,
        base_model: Optional[BinaryClassificationGPModel] = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
        )
        dim = _resolve_latent_dim(latent_dim=latent_dim, n_components=n_components, default=train_X.shape[-1])
        self.pca_config = copy.deepcopy(pca_config) if pca_config is not None else PCAConfig(n_components=dim)
        self.pca = _clone_fitted_pca(fitted_pca) if fitted_pca is not None else PCATransformer(self.pca_config)
        if fitted_pca is None:
            self.pca.fit(self.preproject_train_input)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=self.num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=self.learn_inducing_locations,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        return self.pca.transform(X)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=copy.deepcopy(getattr(self.base_model, "likelihood", None)),
            input_transform=_clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "mean_module", None)),
            covar_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "covar_module", None)),
            num_inducing_points=self.num_inducing_points,
            inducing_points=_get_variational_inducing_points(self),
            learn_inducing_locations=self.learn_inducing_locations,
            pca_config=copy.deepcopy(self.pca_config),
            fitted_pca=_clone_fitted_pca(self.pca),
        )


class REMBOBinaryClassificationGPModel(_BaseProjectedClassificationGP):
    """REMBO 固定ランダム射影後の低次元空間で学習する binary classification GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Any | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Any | None = None,
        covar_module: Any | None = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        rembo_config: Optional[REMBOConfig] = None,
        latent_dim: Optional[int] = None,
        n_components: Optional[int] = None,
        fitted_rembo: Optional[REMBOTransformer] = None,
        seed: int = 42,
        base_model: Optional[BinaryClassificationGPModel] = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
        )
        dim = _resolve_latent_dim(latent_dim=latent_dim, n_components=n_components, default=train_X.shape[-1])
        self.rembo_config = copy.deepcopy(rembo_config) if rembo_config is not None else REMBOConfig(n_components=dim, seed=seed)
        self.rembo = _clone_fitted_rembo(fitted_rembo) if fitted_rembo is not None else REMBOTransformer(self.rembo_config)
        if fitted_rembo is None:
            self.rembo.fit(self.preproject_train_input)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=self.num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=self.learn_inducing_locations,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        return self.rembo.transform(X)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=copy.deepcopy(getattr(self.base_model, "likelihood", None)),
            input_transform=_clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "mean_module", None)),
            covar_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "covar_module", None)),
            num_inducing_points=self.num_inducing_points,
            inducing_points=_get_variational_inducing_points(self),
            learn_inducing_locations=self.learn_inducing_locations,
            rembo_config=copy.deepcopy(self.rembo_config),
            fitted_rembo=_clone_fitted_rembo(self.rembo),
        )


class _BaseProjectedMixedClassificationGP(_BaseProjectedMixedModel):
    """ClassificationMixedGPModel を base_model とする射影 wrapper。"""

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
        self._raw_train_X = train_X.detach().clone()
        self._setup_mixed_dims(input_dim=train_X.shape[-1], cat_dims=cat_dims, category_counts=category_counts)
        self._validate_categorical_values(train_X)
        self.input_transform = _prepare_raw_input_transform_for_mixed(
            _clone_input_transform(input_transform),
            input_dim=train_X.shape[-1],
            cont_dims=self.cont_dims,
            cat_dims=self.cat_dims,
        )
        train_Y = _flatten_targets(train_Y, dtype=train_X.dtype)
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
        likelihood: Any | None,
        mean_module: Any | None,
        covar_module: Any | None,
        cont_kernel_factory: Any | None,
        num_inducing_points: int,
        inducing_points: Optional[Tensor],
        learn_inducing_locations: bool,
    ) -> BinaryClassificationMixedGPModel:
        latent_cat_dims = list(range(self.latent_dim, self.latent_dim + len(self.cat_dims)))
        return BinaryClassificationMixedGPModel(
            train_X=projected_train_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=None,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Any:
        """raw mixed X を受け取り、projected mixed-space 上の probability posterior を返す。"""
        return self.base_model.posterior(
            self.transform_inputs(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def probability_posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Any:
        """classification acquisition 用に probability posterior を明示名で返す。"""
        return self.posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def latent_posterior(self, X: Tensor, posterior_transform=None, **kwargs: Any) -> Any:
        post = self.base_model.latent_posterior(self.transform_inputs(X), **kwargs)
        if posterior_transform is not None:
            post = posterior_transform(post)
        return post

    def posterior_latent(self, X: Tensor, **kwargs: Any) -> Any:
        return self.latent_posterior(X, **kwargs)

    def posterior_f(self, X: Tensor, **kwargs: Any) -> Any:
        return self.latent_posterior(X, **kwargs)

    def predict_proba(self, X: Tensor) -> Tensor:
        if hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict_proba(self.transform_inputs(X))
        return self.posterior(X).mean

    def make_mll(self, *args: Any, **kwargs: Any) -> Any:
        """projected mixed-space の base_model 用 MLL を返す。"""
        if not hasattr(self.base_model, "make_mll"):
            raise AttributeError("base_model does not define make_mll().")
        return self.base_model.make_mll(*args, **kwargs)

    @property
    def model(self):
        """base_model の inner latent model を返す。"""
        return getattr(self.base_model, "model", self.base_model)

    @property
    def likelihood(self):
        return getattr(self.base_model, "likelihood", None)

    @property
    def train_inputs(self):
        return (self.train_input_raw,)

    @property
    def train_targets(self):
        return self._train_targets

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def condition_on_observations(self, X: Tensor, Y: Tensor, noise: Optional[Tensor] = None, **kwargs: Any):
        _ = kwargs
        X_new, Y_new, Yvar_new = _prepare_original_space_conditioning_data(
            X,
            Y,
            noise,
            expected_input_dim=self.input_dim_original,
            force_2d_Y=False,
        )
        X_new = X_new.to(self.train_input_raw)
        self._validate_categorical_values(X_new)
        Y_new = _flatten_targets(Y_new, dtype=self.train_targets.dtype).to(self.train_targets.device)
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
        new_model = self._rebuild_with_new_data(train_X=X_full, train_Y=Y_full, train_Yvar=Yvar_full)
        try:
            new_model.load_state_dict(self.state_dict(), strict=False)
        except Exception:
            pass
        new_model.eval()
        return new_model

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]):
        raise NotImplementedError


class PCABinaryClassificationMixedGPModel(_BaseProjectedMixedClassificationGP):
    """連続列だけ PCA 射影し、カテゴリ列を保持する binary mixed classification GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood: Any | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Any | None = None,
        covar_module: Any | None = None,
        cont_kernel_factory: Any | None = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        pca_config: Optional[PCAConfig] = None,
        latent_dim: Optional[int] = None,
        n_components: Optional[int] = None,
        fitted_pca: Optional[PCATransformer] = None,
        category_counts: Optional[dict[int, int]] = None,
        base_model: Optional[BinaryClassificationMixedGPModel] = None,
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
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self._cont_kernel_factory = cont_kernel_factory
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing_points=self.num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=self.learn_inducing_locations,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        x_cont = self.pca.transform(X[..., self.cont_dims])
        return self._project_continuous_and_concat_categorical(X, x_cont)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=list(self.cat_dims),
            category_counts=copy.deepcopy(self.category_counts),
            train_Yvar=train_Yvar,
            likelihood=copy.deepcopy(getattr(self.base_model, "likelihood", None)),
            input_transform=_clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "mean_module", None)),
            covar_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "covar_module", None)),
            cont_kernel_factory=self._cont_kernel_factory,
            num_inducing_points=self.num_inducing_points,
            inducing_points=_get_variational_inducing_points(self),
            learn_inducing_locations=self.learn_inducing_locations,
            pca_config=copy.deepcopy(self.pca_config),
            fitted_pca=_clone_fitted_pca(self.pca),
        )


class REMBOBinaryClassificationMixedGPModel(_BaseProjectedMixedClassificationGP):
    """連続列だけ REMBO 射影し、カテゴリ列を保持する binary mixed classification GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood: Any | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Any | None = None,
        covar_module: Any | None = None,
        cont_kernel_factory: Any | None = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        rembo_config: Optional[REMBOConfig] = None,
        latent_dim: Optional[int] = None,
        n_components: Optional[int] = None,
        fitted_rembo: Optional[REMBOTransformer] = None,
        seed: int = 42,
        category_counts: Optional[dict[int, int]] = None,
        base_model: Optional[BinaryClassificationMixedGPModel] = None,
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
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self._cont_kernel_factory = cont_kernel_factory
        self.base_model = base_model or self._build_base_model(
            projected_train_X=self.projected_train_input,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing_points=self.num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=self.learn_inducing_locations,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        x_cont = self.rembo.transform(X[..., self.cont_dims])
        return self._project_continuous_and_concat_categorical(X, x_cont)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=list(self.cat_dims),
            category_counts=copy.deepcopy(self.category_counts),
            train_Yvar=train_Yvar,
            likelihood=copy.deepcopy(getattr(self.base_model, "likelihood", None)),
            input_transform=_clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "mean_module", None)),
            covar_module=copy.deepcopy(getattr(getattr(self.base_model, "model", None), "covar_module", None)),
            cont_kernel_factory=self._cont_kernel_factory,
            num_inducing_points=self.num_inducing_points,
            inducing_points=_get_variational_inducing_points(self),
            learn_inducing_locations=self.learn_inducing_locations,
            rembo_config=copy.deepcopy(self.rembo_config),
            fitted_rembo=_clone_fitted_rembo(self.rembo),
        )
