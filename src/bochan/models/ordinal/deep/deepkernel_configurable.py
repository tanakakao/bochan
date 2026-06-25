from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import PredictiveLogLikelihood, VariationalELBO

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.components.layers.feature_extractor import (
    LargeFeatureExtractor,
    SkipLargeFeatureExtractor,
)
from .deepkernel import (
    DeepKernelMixedOrdinal,
    DeepKernelOrdinal,
    InputTransformArg,
    _BaseDeepKernelOrdinalGPModel,
    _clone_input_transform,
    _get_cont_dims,
    _make_train_X_tf_like_classification,
    _normalize_dims,
    _prepare_ordinal_targets,
    _resolve_input_transform,
    _to_device_dtype_transform,
)


def _make_feature_extractor(
    input_dim: int,
    ext_type: str = "DEFAULT",
    hidden_dims: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Ordinal DeepKernel 用 feature extractor を作る。"""
    hidden_dims = (
        [input_dim * 8, input_dim * 4, input_dim * 2]
        if hidden_dims is None
        else [int(h) for h in hidden_dims]
    )
    if ext_type.lower() == "skip":
        return SkipLargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims,
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
            use_global_skip=True,
        )

    return LargeFeatureExtractor(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dims=hidden_dims,
        activation="leaky_relu",
        dropout=0.0,
        use_bn=False,
    )


class DeepKernelOrdinalGPModel(_BaseDeepKernelOrdinalGPModel):
    """Regression-style deep-kernel ordinal GP for continuous inputs.

    Args:
        hidden_dims: DeepKernel feature extractor の隠れ層次元。
            None の場合は従来通り [input_dim * 8, input_dim * 4, input_dim * 2] を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        likelihood: Optional[_OneDimensionalLikelihood] = None,
        input_transform: InputTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
    ) -> None:
        train_Y = _prepare_ordinal_targets(train_Y, train_X)
        input_transform = _resolve_input_transform(train_X, input_transform)
        input_transform = _to_device_dtype_transform(input_transform, train_X)

        train_X_tf = _make_train_X_tf_like_classification(
            train_X,
            input_transform,
            name="DeepKernelOrdinalGPModel.input_transform",
        )

        if likelihood is None:
            likelihood = OrdinalLogitLikelihood(
                num_classes=num_classes,
                eps=eps,
                init_gap=init_gap,
                fix_first_cutpoint=fix_first_cutpoint,
            )

        if feature_extractor is None:
            feature_extractor = _make_feature_extractor(
                input_dim=train_X_tf.size(-1),
                ext_type=ext_type,
                hidden_dims=hidden_dims,
            )

        latent_model = DeepKernelOrdinal(
            train_x=train_X_tf,
            train_y=train_Y,
            likelihood=likelihood,
            ext_type=ext_type,
            feature_extractor=feature_extractor,
            mean_module=mean_module,
            covar_module=covar_module,
            inducing_points=inducing_points,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_predictive_log_likelihood=use_predictive_log_likelihood,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )

        self.num_classes = int(num_classes)
        self.ext_type = str(ext_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]

    def make_mll(self, beta: Optional[float] = None):
        kwargs = {
            "likelihood": self.likelihood,
            "model": self.deepkernel,
            "num_data": self.train_X.shape[-2],
        }
        if beta is not None:
            kwargs["beta"] = float(beta)

        mll_cls = (
            PredictiveLogLikelihood
            if self.use_predictive_log_likelihood
            else VariationalELBO
        )
        return mll_cls(**kwargs)

    def _get_rebuild_kwargs(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "input_transform": _clone_input_transform(self.input_transform),
            "ext_type": self.ext_type,
            "hidden_dims": copy.deepcopy(self.hidden_dims),
            "inducing_points_num": self.inducing_points_num,
            "learn_inducing_locations": self.learn_inducing_locations,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "use_predictive_log_likelihood": self.use_predictive_log_likelihood,
            "fix_first_cutpoint": self.fix_first_cutpoint,
            "init_gap": self.init_gap,
            "eps": self.eps,
            "verbose": self.verbose,
            "conditioning_steps": self.conditioning_steps,
            "conditioning_lr": self.conditioning_lr,
            "conditioning_batch_size": self.conditioning_batch_size,
            "feature_extractor": copy.deepcopy(
                getattr(self.deepkernel, "feature_extractor", self.deepkernel.deepkernel)
            ),
            "mean_module": copy.deepcopy(self.deepkernel.mean_module),
            "covar_module": copy.deepcopy(self.deepkernel.covar_module),
            "inducing_points": self.deepkernel.variational_strategy.inducing_points.detach().clone(),
        }


class DeepKernelOrdinalMixedGPModel(_BaseDeepKernelOrdinalGPModel):
    """Regression-style deep-kernel ordinal GP for mixed continuous + categorical inputs.

    Args:
        hidden_dims: 連続変数側 feature extractor の隠れ層次元。
            None の場合は従来通り [cont_dim * 8, cont_dim * 4, cont_dim * 2] を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]] = None,
        likelihood: Optional[_OneDimensionalLikelihood] = None,
        input_transform: InputTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
        cont_kernel: str = "matern52",
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.03,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        use_predictive_log_likelihood: bool = False,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        feature_extractor: Optional[nn.Module] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        train_Y = _prepare_ordinal_targets(train_Y, train_X)
        norm_cat_dims = _normalize_dims(cat_dims, train_X.shape[-1])

        norm_category_counts = self._infer_category_counts(
            X=train_X,
            cat_dims=norm_cat_dims,
            category_counts=category_counts,
        )

        self._validate_categorical_values(
            X=train_X,
            cat_dims=norm_cat_dims,
            category_counts=norm_category_counts,
        )

        cont_dims = _get_cont_dims(train_X.shape[-1], norm_cat_dims)
        input_transform = _resolve_input_transform(
            train_X,
            input_transform,
            indices=cont_dims,
        )
        input_transform = _to_device_dtype_transform(input_transform, train_X)
        train_X_tf = _make_train_X_tf_like_classification(
            train_X,
            input_transform,
            cat_dims=norm_cat_dims,
            name="DeepKernelOrdinalMixedGPModel.input_transform",
        )

        if likelihood is None:
            likelihood = OrdinalLogitLikelihood(
                num_classes=num_classes,
                eps=eps,
                init_gap=init_gap,
                fix_first_cutpoint=fix_first_cutpoint,
            )

        if feature_extractor is None and len(cont_dims) > 0:
            feature_extractor = _make_feature_extractor(
                input_dim=len(cont_dims),
                ext_type=ext_type,
                hidden_dims=hidden_dims,
            )

        latent_model = DeepKernelMixedOrdinal(
            train_x=train_X_tf,
            train_y=train_Y,
            cat_dims=norm_cat_dims,
            likelihood=likelihood,
            ext_type=ext_type,
            feature_extractor=feature_extractor,
            covar_module=covar_module,
            inducing_points=inducing_points,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            cont_kernel=cont_kernel,
        )

        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            inducing_points_num=inducing_points_num,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_predictive_log_likelihood=use_predictive_log_likelihood,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )

        self.num_classes = int(num_classes)
        self.cat_dims = list(norm_cat_dims)
        self.category_counts = copy.deepcopy(norm_category_counts)
        self.ext_type = str(ext_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
        self.cont_kernel = str(cont_kernel)
        self._ignore_X_dims_scaling_check = self.cat_dims

    def make_mll(self, beta: Optional[float] = None):
        kwargs = {
            "likelihood": self.likelihood,
            "model": self.deepkernel,
            "num_data": self.train_X.shape[-2],
        }
        if beta is not None:
            kwargs["beta"] = float(beta)

        mll_cls = (
            PredictiveLogLikelihood
            if self.use_predictive_log_likelihood
            else VariationalELBO
        )
        return mll_cls(**kwargs)

    @staticmethod
    def _infer_category_counts(
        X: Tensor,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]] = None,
    ) -> dict[int, int]:
        d = X.shape[-1]
        cat_dims = _normalize_dims(cat_dims, d)

        inferred = {}
        if category_counts is not None:
            inferred.update({int(k): int(v) for k, v in category_counts.items()})

        for j in cat_dims:
            vals = X[..., j]

            if not torch.allclose(vals, vals.round()):
                raise ValueError(f"Categorical column {j} must be integer-coded (0..K-1).")
            if vals.min().item() < 0:
                raise ValueError(
                    f"Categorical column {j} must be non-negative integer-coded, got min={vals.min().item()}"
                )
            if j not in inferred:
                inferred[j] = int(vals.max().item()) + 1
            if inferred[j] <= 0:
                raise ValueError(f"category_counts[{j}] must be positive, got {inferred[j]}")

        return inferred

    @staticmethod
    def _validate_categorical_values(
        X: Tensor,
        cat_dims: Sequence[int],
        category_counts: dict[int, int],
    ) -> None:
        d = X.shape[-1]
        cat_dims = _normalize_dims(cat_dims, d)
        for j in cat_dims:
            if j not in category_counts:
                raise ValueError(f"category_counts must contain key {j}")
            n_cat = int(category_counts[j])
            vals = X[..., j]
            if not torch.allclose(vals, vals.round()):
                raise ValueError(f"Categorical column {j} must be integer-coded (0..K-1).")
            if vals.min().item() < 0 or vals.max().item() > n_cat - 1:
                raise ValueError(
                    f"Categorical column {j} must be in [0, {n_cat - 1}], "
                    f"got min={vals.min().item()}, max={vals.max().item()}"
                )

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = super()._canonicalize_observation_X(X)
        self._validate_categorical_values(
            X=X,
            cat_dims=self.cat_dims,
            category_counts=self.category_counts,
        )
        return X

    def _get_rebuild_kwargs(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "cat_dims": copy.deepcopy(self.cat_dims),
            "category_counts": copy.deepcopy(self.category_counts),
            "input_transform": _clone_input_transform(self.input_transform),
            "ext_type": self.ext_type,
            "hidden_dims": copy.deepcopy(self.hidden_dims),
            "cont_kernel": self.cont_kernel,
            "inducing_points_num": self.inducing_points_num,
            "learn_inducing_locations": self.learn_inducing_locations,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "use_predictive_log_likelihood": self.use_predictive_log_likelihood,
            "fix_first_cutpoint": self.fix_first_cutpoint,
            "init_gap": self.init_gap,
            "eps": self.eps,
            "verbose": self.verbose,
            "conditioning_steps": self.conditioning_steps,
            "conditioning_lr": self.conditioning_lr,
            "conditioning_batch_size": self.conditioning_batch_size,
            "feature_extractor": copy.deepcopy(
                getattr(self.deepkernel, "feature_extractor", self.deepkernel.deepkernel)
            ),
            "covar_module": copy.deepcopy(self.deepkernel.covar_module),
            "inducing_points": self.deepkernel.variational_strategy.inducing_points.detach().clone(),
        }


__all__ = ["DeepKernelOrdinalGPModel", "DeepKernelOrdinalMixedGPModel"]
