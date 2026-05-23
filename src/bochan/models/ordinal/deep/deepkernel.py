from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import PredictiveLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.settings import fast_pred_var
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior

from bochan.models.components.layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.ordinal.base.models import (
    _get_cont_dims,
    _normalize_dims,
    build_mixed_ordinal_kernel,
)

InputTransformArg = Union[str, InputTransform, None]


# ============================================================
# Helpers
# ============================================================


def _clone_input_transform(
    input_transform: Optional[InputTransform],
) -> Optional[InputTransform]:
    return None if input_transform is None else copy.deepcopy(input_transform)



def _prepare_ordinal_targets(train_Y: Tensor, train_X: Tensor) -> Tensor:
    """Convert ordinal targets to shape=(n,) long tensor."""
    if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
        train_Y = train_Y.squeeze(-1)
    return train_Y.to(device=train_X.device, dtype=torch.long).contiguous()



def _resolve_input_transform(
    train_X: Tensor,
    input_transform: InputTransformArg,
    *,
    indices: Optional[Sequence[int]] = None,
) -> Optional[InputTransform]:
    """string / object の input_transform を解決する。"""
    input_dim = train_X.shape[-1]

    if isinstance(input_transform, str):
        key = input_transform.upper()
        if key == "DEFAULT":
            if indices is None:
                input_transform = Normalize(d=input_dim)
            else:
                input_transform = Normalize(d=input_dim, indices=list(indices))
        elif key in ("NONE", ""):
            input_transform = None
        else:
            raise ValueError(f"Unknown input_transform: {input_transform}")

    if input_transform is not None and hasattr(input_transform, "to"):
        input_transform = input_transform.to(train_X)

    return input_transform



def _to_device_dtype_transform(
    input_transform: Optional[InputTransform],
    X: Tensor,
) -> Optional[InputTransform]:
    """Move input transform to match X if possible."""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        input_transform = input_transform.to(X)
    return input_transform




def _expand_raw_X_to_match_transformed_q(
    X: Tensor,
    X_tf: Tensor,
) -> Tensor:
    """
    InputPerturbation 後の X_tf と比較できるように raw X の q 次元を展開する。

    想定:
        X.shape    = (*batch, q, d)
        X_tf.shape = (*batch, q_like, d)

    通常:
        q_like = q

    InputPerturbation:
        q_like = q * n_w
    """
    if X.shape == X_tf.shape:
        return X

    if X.ndim < 2 or X_tf.ndim < 2:
        return X

    if X.shape[-1] != X_tf.shape[-1]:
        return X

    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]

        if q_like == q:
            return X

        if q > 0 and q_like % q == 0:
            n_w = q_like // q
            return X.repeat_interleave(n_w, dim=-2)

    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)

    return X


def _check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """
    mixed model 用に、input_transform がカテゴリ列を変更していないか確認する。

    InputPerturbation では q -> q*n_w に展開され得るため、
    raw X 側も q 次元を repeat してから比較する。
    """
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = _expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}. "
            "This usually means input_transform changed the batch/q shape in a "
            "non-repeatable way."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            "input_transform must not modify categorical columns. "
            "For mixed models, transform only continuous columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def _make_train_X_tf_like_classification(
    train_X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
    name: str = "input_transform",
) -> Tensor:
    """
    Classification DeepKernel model と同じ規約で train_X_tf を作る。

    重要:
        - train_X_tf 作成時だけ input_transform.train() にする
        - その後 input_transform.eval() に戻す
        - InputPerturbation は通常 train mode では展開されず、
          eval mode で候補点評価時に展開される

    これにより、学習時には
        train_X_tf.shape[-2] == train_X.shape[-2]
    を保ち、ordinal target との shape 不一致を避ける。
    """
    if input_transform is None:
        return train_X

    input_transform.train()
    train_X_tf = input_transform(train_X)
    input_transform.eval()

    if train_X_tf.shape[-2] != train_X.shape[-2]:
        raise RuntimeError(
            f"{name} expanded train_X during training-mode transform. "
            f"train_X.shape={tuple(train_X.shape)}, "
            f"train_X_tf.shape={tuple(train_X_tf.shape)}. "
            "This will not match ordinal train_Y. "
            "Check that InputPerturbation has transform_on_train=False, or disable "
            "perturbation during training."
        )

    _check_categorical_columns_unchanged(
        X=train_X,
        X_tf=train_X_tf,
        cat_dims=cat_dims,
    )

    return train_X_tf


def _apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    posterior / acquisition 評価用の input_transform。

    eval mode の InputPerturbation では q -> q*n_w 展開を許す。
    mixed model ではカテゴリ列が変わっていないか確認する。
    """
    if input_transform is None:
        return X

    X_tf = input_transform(X)

    _check_categorical_columns_unchanged(
        X=X,
        X_tf=X_tf,
        cat_dims=cat_dims,
    )

    return X_tf


def _select_inducing_points(
    X: Tensor,
    num_inducing_points: int,
    inducing_points: Optional[Tensor] = None,
) -> Tensor:
    """Select inducing points from input-space candidates if not provided."""
    if inducing_points is not None:
        return inducing_points.to(X)
    n = X.shape[-2]
    m = min(int(num_inducing_points), n)
    perm = torch.randperm(n, device=X.device)[:m]
    return X[perm].clone()



def _make_feature_extractor(
    input_dim: int,
    ext_type: str = "DEFAULT",
) -> nn.Module:
    """Return the regression-style feature extractor."""
    if ext_type.lower() == "skip":
        return SkipLargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dims=[input_dim * 8, input_dim * 4, input_dim * 2],
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
            use_global_skip=True,
        )

    return LargeFeatureExtractor(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dims=[input_dim * 8, input_dim * 4, input_dim * 2],
        activation="leaky_relu",
        dropout=0.0,
        use_bn=False,
    )


# ============================================================
# Inner latent models
# ============================================================


class DeepKernelOrdinal(ApproximateGP):
    """
    Continuous-input latent SVGP used by the ordinal deep-kernel wrapper.

    Notes:
        - This class itself does not own ``input_transform``.
        - The wrapper passes transformed inputs to this class.
        - Inducing points live in the transformed input space, not in the
          feature-extractor output space.
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: _OneDimensionalLikelihood,
        ext_type: str = "DEFAULT",
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
    ) -> None:
        input_space_inducing = _select_inducing_points(
            train_x,
            num_inducing_points=inducing_points_num,
            inducing_points=inducing_points,
        )

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=input_space_inducing.shape[-2]
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=input_space_inducing,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.feature_extractor = (feature_extractor or _make_feature_extractor(
            input_dim=train_x.size(-1),
            ext_type=ext_type,
        )).to(train_x)
        # backward-compatible alias
        self.deepkernel = self.feature_extractor
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)
        
        with torch.no_grad():
            sample_feat = self.scale_to_bounds(self.deepkernel(train_x[:1]))
        latent_dim = sample_feat.shape[-1]

        self.mean_module = mean_module or torch.nn.Identity()
        if isinstance(self.mean_module, torch.nn.Identity):
            from gpytorch.means import ConstantMean
            self.mean_module = ConstantMean()

        if covar_module is None:
            from gpytorch.constraints import GreaterThan
            from gpytorch.kernels import MaternKernel, ScaleKernel
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=latent_dim,
                    lengthscale_constraint=GreaterThan(1e-4),
                )
            )
        self.covar_module = covar_module

        self.likelihood = likelihood
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def forward(self, x: Tensor) -> MultivariateNormal:
        projected_x = self.deepkernel(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)


class DeepKernelMixedOrdinal(ApproximateGP):
    """
    Mixed-input latent SVGP used by the ordinal deep-kernel wrapper.

    Notes:
        - Only continuous columns are passed through the feature extractor.
        - Categorical columns remain unchanged.
        - Inducing points live in the transformed input space.
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        cat_dims: Sequence[int],
        likelihood: _OneDimensionalLikelihood,
        ext_type: str = "DEFAULT",
        feature_extractor: Optional[nn.Module] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
        inducing_points_num: int = 128,
        learn_inducing_locations: bool = True,
        cont_kernel: str = "matern52",
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        d = train_x.size(-1)
        self.cat_dims = _normalize_dims(cat_dims, d)
        self.ord_dims = _get_cont_dims(d, self.cat_dims)
        self._ignore_X_dims_scaling_check = self.cat_dims

        input_space_inducing = _select_inducing_points(
            train_x,
            num_inducing_points=inducing_points_num,
            inducing_points=inducing_points,
        )

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=input_space_inducing.shape[-2]
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=input_space_inducing,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        if len(self.ord_dims) > 0:
            self.feature_extractor = (feature_extractor or _make_feature_extractor(
                input_dim=len(self.ord_dims),
                ext_type=ext_type,
            )).to(train_x)
            # backward-compatible alias
            self.deepkernel = self.feature_extractor
            self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)
        else:
            self.feature_extractor = nn.Identity()
            # backward-compatible alias
            self.deepkernel = self.feature_extractor
            self.scale_to_bounds = nn.Identity()

        from gpytorch.means import ConstantMean
        self.mean_module = ConstantMean()
        self.covar_module = covar_module or build_mixed_ordinal_kernel(
            d=d,
            cat_dims=self.cat_dims,
            cont_kernel_name=cont_kernel,
        )

        self.likelihood = likelihood
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def _combine_cont_and_cat(self, x: Tensor) -> Tensor:
        if len(self.ord_dims) == 0:
            return x

        cont_x = x[..., self.ord_dims]
        cat_x = x[..., self.cat_dims]

        projected_cont_x = self.deepkernel(cont_x)
        projected_cont_x = self.scale_to_bounds(projected_cont_x)

        out = torch.empty_like(x)
        out[..., self.ord_dims] = projected_cont_x
        out[..., self.cat_dims] = cat_x
        return out

    def forward(self, x: Tensor) -> MultivariateNormal:
        mixed_x = self._combine_cont_and_cat(x)
        mean_x = self.mean_module(mixed_x)
        covar_x = self.covar_module(mixed_x)
        return MultivariateNormal(mean_x, covar_x)


# ============================================================
# Base wrapper
# ============================================================


class _BaseDeepKernelOrdinalGPModel(ApproximateGPyTorchModel):
    """
    Regression-style wrapper for ordinal deep-kernel GP models.

    Design:
        - wrapper manages ``input_transform``
        - inner latent SVGP stores transformed train_X
        - ``forward`` returns the latent GP distribution
        - ``posterior`` returns the latent posterior for BoTorch
        - ordinal-specific helpers expose class probabilities / utilities
    """

    def __init__(
        self,
        latent_model: ApproximateGP,
        likelihood: OrdinalLogitLikelihood,
        train_X: Tensor,
        train_Y: Tensor,
        input_transform: Optional[InputTransform] = None,
        *,
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
    ) -> None:
        super().__init__(model=latent_model, likelihood=likelihood, num_outputs=1)
        self.deepkernel = self.model

        self._num_outputs = 1
        self._model_dtype = train_X.dtype
        self._model_device = train_X.device

        self.input_transform = input_transform

        self.train_inputs = (train_X,)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets = train_Y
        self._train_targets = train_Y
        self.train_X = train_X
        self.train_Y = train_Y

        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_predictive_log_likelihood = bool(use_predictive_log_likelihood)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.init_gap = float(init_gap)
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.to(train_X)

    @staticmethod
    def _unwrap_inputs(inputs) -> Tensor:
        if isinstance(inputs, tuple):
            return inputs[0]
        return inputs

    def _to_model_dtype_device(self, X: Tensor) -> Tensor:
        return X.to(device=self._model_device, dtype=self._model_dtype)

    def _set_transformed_inputs(self) -> None:
        """
        BoTorch Model.eval() が呼ぶ transformed input 自動更新を無効化する。

        この DeepKernel ordinal wrapper では、
            wrapper.train_X / train_inputs_raw: raw-space X
            inner deepkernel.train_inputs:      transformed-space X
        を明示的に分けて管理する。

        BoTorch 標準の eval-time transform が走ると、InputPerturbation により
        train_inputs が n_w 倍に展開され、train_targets と不整合になる可能性がある。
        """
        return None

    def transform_inputs(self, X: Tensor) -> Tensor:
        """
        posterior / acquisition 評価用 transform。

        eval mode では InputPerturbation により q -> q*n_w 展開されてよい。
        mixed model では cat_dims を使ってカテゴリ列保持を確認する。
        """
        return _apply_input_transform_for_eval(
            X,
            getattr(self, "input_transform", None),
            cat_dims=getattr(self, "cat_dims", None),
        )

    def _apply_input_transform(
        self,
        X: Tensor,
        apply_input_transform: bool = True,
    ) -> Tensor:
        if apply_input_transform and self.input_transform is not None:
            return self.transform_inputs(X)
        return X

    def _canonicalize_posterior_X(self, X: Tensor) -> Tensor:
        X = self._to_model_dtype_device(X)
        if X.ndim == 1:
            if self.train_X.shape[-1] != 1:
                raise ValueError(
                    f"1D X can only be used when input dim is 1, got d={self.train_X.shape[-1]}"
                )
            X = X.unsqueeze(-1)
        if X.ndim < 2:
            raise ValueError(f"X for posterior must have at least 2 dims, got shape={tuple(X.shape)}")
        if X.shape[-1] != self.train_X.shape[-1]:
            raise ValueError(
                f"X feature dim mismatch: expected {self.train_X.shape[-1]}, got {X.shape[-1]}"
            )
        return X.contiguous()

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = self._canonicalize_posterior_X(X)
        if X.ndim != 2:
            raise ValueError(f"Observation X must be [n, d], got shape={tuple(X.shape)}")
        return X

    def _canonicalize_new_Y(self, Y: Tensor, n: int) -> Tensor:
        Y = torch.as_tensor(Y, device=self.train_X.device)
        if Y.ndim == 0:
            Y = Y.view(1)
        elif Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        elif Y.ndim != 1:
            raise ValueError(f"Y must be [n] or [n, 1], got shape={tuple(Y.shape)}")
        if Y.shape[0] != n:
            raise ValueError(f"Y length mismatch: expected {n}, got {Y.shape[0]}")
        return Y.long().contiguous()

    @property
    def ordinal_likelihood(self) -> OrdinalLogitLikelihood:
        return self.likelihood

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def forward(
        self,
        inputs,
        apply_input_transform: bool = True,
    ) -> MultivariateNormal:
        x = self._unwrap_inputs(inputs)
        x = self._to_model_dtype_device(x)
        x = self._apply_input_transform(x, apply_input_transform=apply_input_transform)
        return self.deepkernel(x)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ):
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )
        if observation_noise is not False:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support observation_noise."
            )

        self.eval()
        self.likelihood.eval()

        X = self._canonicalize_posterior_X(self._unwrap_inputs(X))
        X_tf = self._apply_input_transform(X, apply_input_transform=True)

        with fast_pred_var():
            latent_dist = self.deepkernel(X_tf)
            posterior = GPyTorchPosterior(latent_dist)

            if posterior_transform is not None:
                posterior = posterior_transform(posterior)

        return posterior

    def predict_latent(self, X: Tensor) -> tuple[Tensor, Tensor]:
        posterior = self.posterior(X)
        return posterior.mean, posterior.variance

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Regression-style alias: returns latent mean / variance."""
        return self.predict_latent(X)

    @torch.no_grad()
    def class_probs_from_posterior(self, posterior) -> Tensor:
        return self.ordinal_likelihood.marginal_class_probs(posterior.distribution)

    @torch.no_grad()
    def class_probs(self, X: Tensor) -> Tensor:
        posterior = self.posterior(X)
        return self.class_probs_from_posterior(posterior)

    @torch.no_grad()
    def predict_proba(self, X: Tensor) -> Tensor:
        return self.class_probs(X)

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)

    @torch.no_grad()
    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        posterior = self.posterior(X)
        return self.ordinal_likelihood.marginal_expected_utility(
            posterior.distribution,
            utilities,
        )

    def set_train_data(
        self,
        inputs: Optional[Union[Tensor, tuple[Tensor, ...]]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        _ = strict
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_X = self._to_model_dtype_device(inputs[0])
            self.train_inputs = (self.train_X,)
            self.train_inputs_raw = (self.train_X.detach().clone(),)

            X_tf = _make_train_X_tf_like_classification(
                self.train_X,
                self.input_transform,
                cat_dims=getattr(self, "cat_dims", None),
                name=f"{self.__class__.__name__}.input_transform",
            )
            self.deepkernel.train_inputs = (X_tf,)

        if targets is not None:
            prepared_targets = _prepare_ordinal_targets(targets, self.train_X)
            self.train_targets = prepared_targets
            self._train_targets = prepared_targets
            self.train_Y = prepared_targets
            self.deepkernel.train_targets = prepared_targets

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def make_mll(self):
        if self.use_predictive_log_likelihood:
            return PredictiveLogLikelihood(
                likelihood=self.likelihood,
                model=self.deepkernel,
                num_data=self.train_X.shape[-2],
            )
        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.deepkernel,
            num_data=self.train_X.shape[-2],
        )

    def _get_rebuild_kwargs(self) -> dict:
        raise NotImplementedError

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        if "noise" in kwargs and kwargs["noise"] is not None:
            raise NotImplementedError(f"noise is not supported for {self.__class__.__name__}.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_X, X], dim=-2)
        new_train_Y = torch.cat([self.train_Y, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            **self._get_rebuild_kwargs(),
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=True)

        if refit:
            if num_steps is None:
                num_steps = self.conditioning_steps
            if lr is None:
                lr = self.conditioning_lr if self.conditioning_lr is not None else self.lr
            if batch_size is None:
                batch_size = (
                    self.conditioning_batch_size
                    if self.conditioning_batch_size is not None
                    else self.batch_size
                )
            fit_deepkernel_ordinal_gp(
                new_model,
                num_epochs=int(num_steps),
                lr=float(lr),
                batch_size=batch_size,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


class DeepKernelOrdinalGPModel(_BaseDeepKernelOrdinalGPModel):
    """Regression-style deep-kernel ordinal GP for continuous inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        likelihood: Optional[_OneDimensionalLikelihood] = None,
        input_transform: InputTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
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

    def _get_rebuild_kwargs(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "input_transform": _clone_input_transform(self.input_transform),
            "ext_type": self.ext_type,
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
    """
    Regression-style deep-kernel ordinal GP for mixed continuous + categorical inputs.

    Notes:
        input_transform はカテゴリ列を触らないものを使うこと。
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
        self.cont_kernel = str(cont_kernel)
        self._ignore_X_dims_scaling_check = self.cat_dims

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
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )

            if vals.min().item() < 0:
                raise ValueError(
                    f"Categorical column {j} must be non-negative integer-coded, "
                    f"got min={vals.min().item()}"
                )

            if j not in inferred:
                inferred[j] = int(vals.max().item()) + 1

            if inferred[j] <= 0:
                raise ValueError(
                    f"category_counts[{j}] must be positive, got {inferred[j]}"
                )

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
                raise ValueError(
                    f"Categorical column {j} must be integer-coded (0..K-1)."
                )
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

# ============================================================
# Fitting helper
# ============================================================


# def fit_deepkernel_ordinal_gp(
#     model: _BaseDeepKernelOrdinalGPModel,
#     num_epochs: Optional[int] = None,
#     lr: Optional[float] = None,
#     batch_size: Optional[int] = None,
#     verbose: Optional[bool] = None,
# ) -> _BaseDeepKernelOrdinalGPModel:
#     """Train a regression-style deep-kernel ordinal GP model."""
#     num_epochs = model.num_epochs if num_epochs is None else int(num_epochs)
#     lr = model.lr if lr is None else float(lr)
#     verbose = model.verbose if verbose is None else bool(verbose)

#     train_X = model.train_X
#     train_Y = model.train_Y

#     if num_epochs <= 0:
#         return model

#     dataset = TensorDataset(train_X, train_Y)
#     if batch_size is None:
#         batch_size = model.batch_size
#     if batch_size is None:
#         batch_size = len(dataset)
#     batch_size = int(batch_size)

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model.train()
#     model.likelihood.train()

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     mll = model.make_mll()

#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         for xb, yb in loader:
#             optimizer.zero_grad()
#             latent_dist = model(xb, apply_input_transform=True)
#             loss = -mll(latent_dist, yb)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * xb.shape[0]

#         if verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
#             avg_loss = total_loss / train_X.shape[-2]
#             cuts = model.ordinal_likelihood.cutpoints.detach().cpu().numpy()
#             print(f"[fit-deepkernel-ordinal] epoch={epoch + 1:03d} loss={avg_loss:.4f} cutpoints={cuts}")

#     model.eval()
#     model.likelihood.eval()
#     return model


# backward-compatible alias
# fit_deep_ordinal_gp = fit_deepkernel_ordinal_gp


def fit_deepkernel_ordinal_gp(
    model: _BaseDeepKernelOrdinalGPModel,
    num_epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    verbose: Optional[bool] = None,
) -> _BaseDeepKernelOrdinalGPModel:
    """DeepKernel ordinal GP を学習する。"""
    num_epochs = model.num_epochs if num_epochs is None else int(num_epochs)
    lr = model.lr if lr is None else float(lr)
    verbose = model.verbose if verbose is None else bool(verbose)

    train_X = model.train_X
    train_Y = model.train_Y

    if num_epochs <= 0:
        return model

    dataset = TensorDataset(train_X, train_Y)
    if batch_size is None:
        batch_size = model.batch_size
    if batch_size is None:
        batch_size = len(dataset)
    batch_size = int(batch_size)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = model.make_mll()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            latent_dist = model(xb, apply_input_transform=True)
            loss = -mll(latent_dist, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
            avg_loss = total_loss / train_X.shape[-2]
            cuts = model.ordinal_likelihood.cutpoints.detach().cpu().numpy()
            print(
                f"[fit-deepkernel-ordinal] epoch={epoch + 1:03d} "
                f"loss={avg_loss:.4f} cutpoints={cuts}"
            )

    model.eval()
    model.likelihood.eval()
    return model


# backward-compatible alias
fit_deep_ordinal_gp = fit_deepkernel_ordinal_gp


__all__ = [
    "DeepKernelOrdinal",
    "DeepKernelMixedOrdinal",
    "DeepKernelOrdinalGPModel",
    "DeepKernelMixedOrdinalGPModel",
    "fit_deepkernel_ordinal_gp",
    "fit_deep_ordinal_gp",
]
