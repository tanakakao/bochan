from __future__ import annotations

"""PCA / REMBO による高次元 ordinal GP wrapper。

配置想定:
    ``bochan.models.ordinal.high_dim.decomposition``

PCA / REMBO の射影器は ``bochan.models.components.decomposition``、
raw/preproject/projected 管理は ``bochan.models.components.projected`` に寄せる。

Notes:
    - 外部 API は ``n_components`` に統一する。
    - 旧 API の ``latent_dim`` は ``__init__`` 引数から削除する。
    - mixed wrapper 内部では、カテゴリ列の offset として ``projected_dim`` を使う。
"""

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.transforms.input import InputTransform

from bochan.models.ordinal.base.models import OrdinalGPModel, OrdinalMixedGPModel, _BaseOrdinalGPModel
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
    _prepare_raw_input_transform_for_mixed,
)


__all__ = [
    "PCAOrdinalGPModel",
    "REMBOOrdinalGPModel",
    "PCAOrdinalMixedGPModel",
    "REMBOOrdinalMixedGPModel",
]


def _canonicalize_ordinal_X(X: Tensor, *, like: Tensor) -> Tensor:
    """ordinal wrapper 用に raw X を [n, d] にそろえる。"""
    X = torch.as_tensor(X, device=like.device, dtype=like.dtype)
    if X.ndim == 1:
        X = X.unsqueeze(0)
    if X.ndim != 2:
        raise ValueError(f"X must be [n, d], got shape={tuple(X.shape)}.")
    return X.contiguous()


def _canonicalize_ordinal_Y(Y: Tensor, *, n: int, device: torch.device) -> Tensor:
    """ordinal target を [n] long tensor にそろえる。"""
    Y = torch.as_tensor(Y, device=device)
    if Y.ndim == 0:
        Y = Y.view(1)
    elif Y.ndim == 2 and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)
    elif Y.ndim != 1:
        raise ValueError(f"Y must be [n] or [n, 1], got shape={tuple(Y.shape)}.")
    if Y.shape[0] != n:
        raise ValueError(f"Y length mismatch: expected {n}, got {Y.shape[0]}.")
    return Y.long().contiguous()


def _resolve_n_components(
    *,
    n_components: Optional[int],
    default: int,
    name: str = "n_components",
) -> int:
    """射影後の連続次元数を決定する。

    Args:
        n_components: PCA / REMBO の射影後次元数。None の場合は ``default`` を使う。
        default: ``n_components`` が None のときに使うデフォルト値。
        name: エラーメッセージ用の引数名。

    Returns:
        正の整数として解決された射影後次元数。

    Raises:
        ValueError: 解決された次元数が 1 未満の場合。
    """
    dim = int(default if n_components is None else n_components)
    if dim <= 0:
        raise ValueError(f"{name} must be a positive integer, got {dim}.")
    return dim


def _resolve_pca_config(
    *,
    pca_config: Optional[PCAConfig],
    n_components: Optional[int],
    default: int,
) -> PCAConfig:
    """PCAConfig と n_components の指定を一元的に解決する。

    ``pca_config`` が与えられた場合はそれを優先する。ただし、同時に
    ``n_components`` も指定されている場合は、値の不一致をエラーにする。
    """
    if pca_config is None:
        return PCAConfig(n_components=_resolve_n_components(n_components=n_components, default=default))

    cfg = copy.deepcopy(pca_config)
    cfg_n_components = int(cfg.n_components)
    if cfg_n_components <= 0:
        raise ValueError(f"pca_config.n_components must be positive, got {cfg_n_components}.")

    if n_components is not None and int(n_components) != cfg_n_components:
        raise ValueError(
            "pca_config.n_components and n_components are inconsistent: "
            f"pca_config.n_components={cfg_n_components}, n_components={int(n_components)}."
        )
    return cfg


def _resolve_rembo_config(
    *,
    rembo_config: Optional[REMBOConfig],
    n_components: Optional[int],
    default: int,
    seed: int,
) -> REMBOConfig:
    """REMBOConfig と n_components の指定を一元的に解決する。

    ``rembo_config`` が与えられた場合はそれを優先する。ただし、同時に
    ``n_components`` も指定されている場合は、値の不一致をエラーにする。
    """
    if rembo_config is None:
        return REMBOConfig(
            n_components=_resolve_n_components(n_components=n_components, default=default),
            seed=seed,
        )

    cfg = copy.deepcopy(rembo_config)
    cfg_n_components = int(cfg.n_components)
    if cfg_n_components <= 0:
        raise ValueError(f"rembo_config.n_components must be positive, got {cfg_n_components}.")

    if n_components is not None and int(n_components) != cfg_n_components:
        raise ValueError(
            "rembo_config.n_components and n_components are inconsistent: "
            f"rembo_config.n_components={cfg_n_components}, n_components={int(n_components)}."
        )
    return cfg


class _BaseProjectedOrdinalGP(_BaseProjectedModel):
    """OrdinalGPModel を base_model とする射影 wrapper。"""

    @property
    def ordinal_likelihood(self):
        return self.base_model.ordinal_likelihood

    @torch.no_grad()
    def class_probs(self, X: Tensor) -> Tensor:
        return self.base_model.class_probs(self.transform_inputs(X))

    @torch.no_grad()
    def class_probs_from_posterior(self, posterior) -> Tensor:
        return self.base_model.class_probs_from_posterior(posterior)

    @torch.no_grad()
    def predict_proba(self, X: Tensor) -> Tensor:
        return self.class_probs(X)

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        return self.base_model.predict_class(self.transform_inputs(X))

    @torch.no_grad()
    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        return self.base_model.expected_utility(self.transform_inputs(X), utilities)

    def _init_common_state(
        self,
        *,
        train_X: Tensor,
        train_Y: Tensor,
        input_transform: Optional[InputTransform],
    ) -> Tensor:
        raw_train_X = _BaseOrdinalGPModel._canonicalize_train_X(train_X)
        train_Y = _BaseOrdinalGPModel._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )
        self.input_dim_original = raw_train_X.shape[-1]
        self.input_transform = _clone_input_transform(input_transform)
        self._raw_train_X = raw_train_X.detach().clone()
        self._train_targets = train_Y
        self._preproject_train_X = _apply_input_transform_for_training(
            raw_train_X,
            self.input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )
        return train_Y

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """raw-space の新規観測で条件付けした wrapper を返す。"""
        X = _canonicalize_ordinal_X(X, like=self.train_input_raw)
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(
                f"Expected raw input dim {self.input_dim_original}, got {X.shape[-1]}."
            )
        Y = _canonicalize_ordinal_Y(Y, n=X.shape[-2], device=X.device)

        new_base = self.base_model.condition_on_observations(
            self.transform_inputs(X),
            Y,
            refit=refit,
            num_steps=num_steps,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )
        return self._rebuild_with_new_data(
            train_X=torch.cat([self.train_input_raw, X], dim=0),
            train_Y=torch.cat([self.train_targets, Y], dim=0),
            base_model=new_base,
        )

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, base_model: OrdinalGPModel):
        raise NotImplementedError


class PCAOrdinalGPModel(_BaseProjectedOrdinalGP):
    """PCA 射影後の低次元空間で学習する ordinal GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        n_components: Optional[int] = 8,
        pca_config: Optional[PCAConfig] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        fitted_pca: Optional[PCATransformer] = None,
        base_model: Optional[OrdinalGPModel] = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
        )
        self.pca_config = _resolve_pca_config(
            pca_config=pca_config,
            n_components=n_components,
            default=self.preproject_train_input.shape[-1],
        )
        self.projected_dim = int(self.pca_config.n_components)
        self.n_components = self.projected_dim
        # 内部互換用。外部 API からは latent_dim を削除する。
        self.latent_dim = self.projected_dim

        self.pca = _clone_fitted_pca(fitted_pca) if fitted_pca is not None else PCATransformer(self.pca_config)
        if fitted_pca is None:
            self.pca.fit(self.preproject_train_input)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()

        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.base_model = base_model or OrdinalGPModel(
            train_X=self.projected_train_input,
            train_Y=train_Y,
            num_classes=self.num_classes,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        return self.pca.transform(X)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, base_model: OrdinalGPModel):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=self.num_classes,
            pca_config=copy.deepcopy(self.pca_config),
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            input_transform=_clone_input_transform(self.input_transform),
            fitted_pca=_clone_fitted_pca(self.pca),
            base_model=base_model,
        )


class REMBOOrdinalGPModel(_BaseProjectedOrdinalGP):
    """REMBO 固定ランダム射影後の低次元空間で学習する ordinal GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        n_components: Optional[int] = 8,
        rembo_config: Optional[REMBOConfig] = None,
        seed: int = 42,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        fitted_rembo: Optional[REMBOTransformer] = None,
        base_model: Optional[OrdinalGPModel] = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
        )
        self.rembo_config = _resolve_rembo_config(
            rembo_config=rembo_config,
            n_components=n_components,
            default=self.preproject_train_input.shape[-1],
            seed=seed,
        )
        self.projected_dim = int(self.rembo_config.n_components)
        self.n_components = self.projected_dim
        # 内部互換用。外部 API からは latent_dim を削除する。
        self.latent_dim = self.projected_dim

        self.rembo = _clone_fitted_rembo(fitted_rembo) if fitted_rembo is not None else REMBOTransformer(self.rembo_config)
        if fitted_rembo is None:
            self.rembo.fit(self.preproject_train_input)
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()

        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.base_model = base_model or OrdinalGPModel(
            train_X=self.projected_train_input,
            train_Y=train_Y,
            num_classes=self.num_classes,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        return self.rembo.transform(X)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, base_model: OrdinalGPModel):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=self.num_classes,
            rembo_config=copy.deepcopy(self.rembo_config),
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            input_transform=_clone_input_transform(self.input_transform),
            fitted_rembo=_clone_fitted_rembo(self.rembo),
            base_model=base_model,
        )


class _BaseProjectedOrdinalMixedGP(_BaseProjectedMixedModel):
    """OrdinalMixedGPModel を base_model とする射影 wrapper。"""

    @property
    def ordinal_likelihood(self):
        return self.base_model.ordinal_likelihood

    @torch.no_grad()
    def class_probs(self, X: Tensor) -> Tensor:
        return self.base_model.class_probs(self.transform_inputs(X))

    @torch.no_grad()
    def class_probs_from_posterior(self, posterior) -> Tensor:
        return self.base_model.class_probs_from_posterior(posterior)

    @torch.no_grad()
    def predict_proba(self, X: Tensor) -> Tensor:
        return self.class_probs(X)

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        return self.base_model.predict_class(self.transform_inputs(X))

    @torch.no_grad()
    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        return self.base_model.expected_utility(self.transform_inputs(X), utilities)

    def _init_common_state(
        self,
        *,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]],
        input_transform: Optional[InputTransform],
    ) -> Tensor:
        raw_train_X = _BaseOrdinalGPModel._canonicalize_train_X(train_X)
        train_Y = _BaseOrdinalGPModel._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )
        self._raw_train_X = raw_train_X.detach().clone()
        self._setup_mixed_dims(
            input_dim=raw_train_X.shape[-1],
            cat_dims=cat_dims,
            category_counts=category_counts,
        )
        self._validate_categorical_values(raw_train_X)
        self.input_transform = _prepare_raw_input_transform_for_mixed(
            _clone_input_transform(input_transform),
            input_dim=raw_train_X.shape[-1],
            cont_dims=self.cont_dims,
            cat_dims=self.cat_dims,
        )
        self._train_targets = train_Y
        self._preproject_train_X = _apply_input_transform_for_training(
            raw_train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name=f"{self.__class__.__name__}.input_transform",
        )
        self._validate_categorical_values(self.preproject_train_input)
        return train_Y

    def _make_remapped_counts(self) -> dict[int, int]:
        """射影後のカテゴリ列 index に合わせて category_counts を作り直す。"""
        return {
            self.projected_dim + i: int(self.category_counts[j])
            for i, j in enumerate(self.cat_dims)
        }

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        X = _canonicalize_ordinal_X(X, like=self.train_input_raw)
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(
                f"Expected raw input dim {self.input_dim_original}, got {X.shape[-1]}."
            )
        self._validate_categorical_values(X)
        Y = _canonicalize_ordinal_Y(Y, n=X.shape[-2], device=X.device)
        new_base = self.base_model.condition_on_observations(
            self.transform_inputs(X),
            Y,
            refit=refit,
            num_steps=num_steps,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )
        return self._rebuild_with_new_data(
            train_X=torch.cat([self.train_input_raw, X], dim=0),
            train_Y=torch.cat([self.train_targets, Y], dim=0),
            base_model=new_base,
        )

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, base_model: OrdinalMixedGPModel):
        raise NotImplementedError


class PCAOrdinalMixedGPModel(_BaseProjectedOrdinalMixedGP):
    """連続列だけ PCA 射影し、カテゴリ列を保持する mixed ordinal GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int] = (),
        category_counts: Optional[dict[int, int]] = None,
        cont_kernel: str = "matern52",
        n_components: Optional[int] = 8,
        pca_config: Optional[PCAConfig] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        fitted_pca: Optional[PCATransformer] = None,
        base_model: Optional[OrdinalMixedGPModel] = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            category_counts=category_counts,
            input_transform=input_transform,
        )
        self.pca_config = _resolve_pca_config(
            pca_config=pca_config,
            n_components=n_components,
            default=len(self.cont_dims),
        )
        self.projected_dim = int(self.pca_config.n_components)
        self.n_components = self.projected_dim
        # 内部互換用。外部 API からは latent_dim を削除する。
        self.latent_dim = self.projected_dim

        self.pca = _clone_fitted_pca(fitted_pca) if fitted_pca is not None else PCATransformer(self.pca_config)
        if fitted_pca is None:
            self.pca.fit(self.preproject_train_input[..., self.cont_dims])
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()

        self.num_classes = int(num_classes)
        self.cont_kernel = str(cont_kernel)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        remapped_cat_dims = list(range(self.projected_dim, self.projected_dim + len(self.cat_dims)))
        self.base_model = base_model or OrdinalMixedGPModel(
            train_X=self.projected_train_input,
            train_Y=train_Y,
            num_classes=self.num_classes,
            cat_dims=remapped_cat_dims,
            category_counts=self._make_remapped_counts(),
            cont_kernel=self.cont_kernel,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        x_cont = self.pca.transform(X[..., self.cont_dims])
        return self._project_continuous_and_concat_categorical(X, x_cont)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, base_model: OrdinalMixedGPModel):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=self.num_classes,
            cat_dims=list(self.cat_dims),
            category_counts=copy.deepcopy(self.category_counts),
            cont_kernel=self.cont_kernel,
            pca_config=copy.deepcopy(self.pca_config),
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            input_transform=_clone_input_transform(self.input_transform),
            fitted_pca=_clone_fitted_pca(self.pca),
            base_model=base_model,
        )


class REMBOOrdinalMixedGPModel(_BaseProjectedOrdinalMixedGP):
    """連続列だけ REMBO 射影し、カテゴリ列を保持する mixed ordinal GP wrapper。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int] = (),
        category_counts: Optional[dict[int, int]] = None,
        cont_kernel: str = "matern52",
        n_components: Optional[int] = 8,
        rembo_config: Optional[REMBOConfig] = None,
        seed: int = 42,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        fitted_rembo: Optional[REMBOTransformer] = None,
        base_model: Optional[OrdinalMixedGPModel] = None,
    ) -> None:
        super().__init__()
        train_Y = self._init_common_state(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            category_counts=category_counts,
            input_transform=input_transform,
        )
        self.rembo_config = _resolve_rembo_config(
            rembo_config=rembo_config,
            n_components=n_components,
            default=len(self.cont_dims),
            seed=seed,
        )
        self.projected_dim = int(self.rembo_config.n_components)
        self.n_components = self.projected_dim
        # 内部互換用。外部 API からは latent_dim を削除する。
        self.latent_dim = self.projected_dim

        self.rembo = _clone_fitted_rembo(fitted_rembo) if fitted_rembo is not None else REMBOTransformer(self.rembo_config)
        if fitted_rembo is None:
            self.rembo.fit(self.preproject_train_input[..., self.cont_dims])
        self._projected_train_X = self._project_preprojected_inputs(self.preproject_train_input).detach().clone()

        self.num_classes = int(num_classes)
        self.cont_kernel = str(cont_kernel)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        remapped_cat_dims = list(range(self.projected_dim, self.projected_dim + len(self.cat_dims)))
        self.base_model = base_model or OrdinalMixedGPModel(
            train_X=self.projected_train_input,
            train_Y=train_Y,
            num_classes=self.num_classes,
            cat_dims=remapped_cat_dims,
            category_counts=self._make_remapped_counts(),
            cont_kernel=self.cont_kernel,
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        x_cont = self.rembo.transform(X[..., self.cont_dims])
        return self._project_continuous_and_concat_categorical(X, x_cont)

    def _rebuild_with_new_data(self, *, train_X: Tensor, train_Y: Tensor, base_model: OrdinalMixedGPModel):
        return self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=self.num_classes,
            cat_dims=list(self.cat_dims),
            category_counts=copy.deepcopy(self.category_counts),
            cont_kernel=self.cont_kernel,
            rembo_config=copy.deepcopy(self.rembo_config),
            inducing_points_num=self.inducing_points_num,
            learn_inducing_locations=self.learn_inducing_locations,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
            input_transform=_clone_input_transform(self.input_transform),
            fitted_rembo=_clone_fitted_rembo(self.rembo),
            base_model=base_model,
        )
