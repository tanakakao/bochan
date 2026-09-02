from collections.abc import Sequence

import torch
import torch.nn as nn
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.transforms import normalize_indices
from gpytorch import settings
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from linear_operator import to_linear_operator
from linear_operator.operators import MaskedLinearOperator
from torch import Tensor

from ..layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor


class StableScaleToBounds(nn.Module):
    """Scale features to fixed bounds without dividing by a zero range.

    GPyTorch's ``ScaleToBounds`` uses the current batch minimum and maximum.
    Deep-kernel feature extractors can temporarily collapse to a nearly constant
    representation, especially in small cross-validation folds. Clamping the
    denominator keeps the projected features finite while preserving the usual
    min-max behavior outside the degenerate case.
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        *,
        min_range: float = 1e-6,
    ) -> None:
        super().__init__()
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        if min_range <= 0:
            raise ValueError("min_range must be positive.")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.min_range = float(min_range)
        self.register_buffer("min_val", torch.tensor(lower_bound))
        self.register_buffer("max_val", torch.tensor(upper_bound))

    def forward(self, x: Tensor) -> Tensor:
        """Return finite min-max-scaled features within the configured bounds."""
        if x.numel() == 0:
            return x
        if not torch.isfinite(x).all():
            raise FloatingPointError(
                "Deep Kernel feature extractor produced non-finite values. "
                "Reduce the learning rate or inspect the input data."
            )

        if self.training:
            min_val = x.amin()
            max_val = x.amax()
            self.min_val.data = min_val.detach()
            self.max_val.data = max_val.detach()
        else:
            min_val = self.min_val.to(device=x.device, dtype=x.dtype)
            max_val = self.max_val.to(device=x.device, dtype=x.dtype)
            x = x.clamp(min=min_val, max=max_val)

        min_range = torch.as_tensor(
            self.min_range,
            device=x.device,
            dtype=x.dtype,
        )
        scale = 0.95 * (self.upper_bound - self.lower_bound)
        safe_range = (max_val - min_val).clamp_min(min_range)
        scaled = (x - min_val) * (scale / safe_range) + 0.95 * self.lower_bound
        if not torch.isfinite(scaled).all():
            raise FloatingPointError("Deep Kernel feature scaling produced non-finite values.")
        return scaled


def _make_feature_extractor(
    input_dim: int,
    output_dim: int | None = None,
    ext_type: str = "DEFAULT",
    hidden_dims: Sequence[int] | None = None,
) -> nn.Module:
    """
    特徴抽出器を返す。

    Args:
        input_dim (int): 入力次元
        output_dim (int | None): 出力特徴次元。None の場合は input_dim。
        ext_type (str): "DEFAULT" または "skip"
        hidden_dims (Sequence[int] | None): 隠れ層の次元数。
            None の場合は従来通り [input_dim * 8, input_dim * 4, input_dim * 2] を使う。
    """
    output_dim = input_dim if output_dim is None else int(output_dim)
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    hidden_dims = (
        [input_dim * 8, input_dim * 4, input_dim * 2] if hidden_dims is None else [int(h) for h in hidden_dims]
    )

    if ext_type.lower() == "skip":
        return SkipLargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
            use_global_skip=True,
        )

    return LargeFeatureExtractor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation="leaky_relu",
        dropout=0.0,
        use_bn=False,
    )


def _feature_output_dim(
    *,
    train_x: Tensor,
    feature_extractor: nn.Module,
    latent_dim: int | None,
) -> int:
    """Resolve and validate the last dimension produced by a feature extractor."""

    configured = None if latent_dim is None else int(latent_dim)
    if configured is not None and configured <= 0:
        raise ValueError("latent_dim must be positive.")

    declared = getattr(feature_extractor, "output_dim", None)
    if declared is not None:
        declared = int(declared)
        if declared <= 0:
            raise ValueError("feature_extractor.output_dim must be positive.")
        if configured is not None and configured != declared:
            raise ValueError(f"latent_dim does not match feature_extractor.output_dim: {configured} != {declared}.")
        return declared

    if configured is not None:
        return configured

    sample_input = train_x[..., :1, :]
    was_training = feature_extractor.training
    feature_extractor.eval()
    try:
        with torch.no_grad():
            sample = feature_extractor(sample_input)
    finally:
        feature_extractor.train(was_training)
    if not torch.is_tensor(sample) or sample.ndim == 0:
        raise ValueError("feature_extractor must return a Tensor with a feature dimension.")
    if sample.shape[:-1] != sample_input.shape[:-1]:
        raise ValueError(
            "feature_extractor must preserve all leading input dimensions. "
            f"Got input shape {tuple(sample_input.shape)} and "
            f"output shape {tuple(sample.shape)}."
        )
    inferred = int(sample.shape[-1])
    if inferred <= 0:
        raise ValueError("feature_extractor must produce at least one feature.")
    return inferred


def _validate_projected_features(
    *,
    x: Tensor,
    projected_x: Tensor,
    latent_dim: int,
) -> None:
    """Validate the feature extractor contract before evaluating a GP kernel."""

    if not torch.is_tensor(projected_x):
        raise TypeError("feature_extractor must return a Tensor.")
    if projected_x.shape[:-1] != x.shape[:-1]:
        raise ValueError(
            "feature_extractor must preserve all leading input dimensions. "
            f"Got input shape {tuple(x.shape)} and output shape {tuple(projected_x.shape)}."
        )
    if projected_x.shape[-1] != latent_dim:
        raise ValueError(
            f"feature_extractor output width does not match latent_dim: {projected_x.shape[-1]} != {latent_dim}."
        )


class _PartialObservationPredictionStrategy(DefaultPredictionStrategy):
    """Exact prediction strategy that conditions only on observed task cells."""

    def _observed_event_mask(self) -> Tensor | None:
        if not bool(torch.isnan(self.train_labels).any()):
            return None
        return settings.observation_nan_policy._get_observed(
            self.train_labels,
            torch.Size((self.train_labels.shape[-1],)),
        ).reshape(-1)

    def exact_prediction(self, test_mean, test_test_covar, test_train_covar):
        if self._observed_event_mask() is None:
            return super().exact_prediction(
                test_mean,
                test_test_covar,
                test_train_covar,
            )
        with settings.observation_nan_policy("mask"):
            return super().exact_prediction(
                test_mean,
                test_test_covar,
                test_train_covar,
            )

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        observed = self._observed_event_mask()
        if observed is None:
            return super().exact_predictive_covar(
                test_test_covar,
                test_train_covar,
            )

        train_covar = MaskedLinearOperator(
            to_linear_operator(self.lik_train_train_covar),
            observed,
            observed,
        )
        test_rows = torch.ones(
            test_train_covar.shape[-2],
            dtype=torch.bool,
            device=observed.device,
        )
        test_train_observed = MaskedLinearOperator(
            to_linear_operator(test_train_covar),
            test_rows,
            observed,
        )
        test_train_dense = test_train_observed.to_dense()
        correction_rhs = train_covar.solve(test_train_dense.transpose(-1, -2))
        correction = test_train_dense @ correction_rhs
        return to_linear_operator(test_test_covar) + to_linear_operator(
            correction.mul(-1)
        )


class _PartialObservationMultitaskKernel(MultitaskKernel):
    """Multitask kernel selecting the partial-observation prediction strategy."""

    def prediction_strategy(
        self,
        train_inputs,
        train_prior_dist,
        train_labels,
        likelihood,
    ):
        return _PartialObservationPredictionStrategy(
            train_inputs,
            train_prior_dist,
            train_labels,
            likelihood,
        )


class DeepKernel(ExactGP):
    """
    連続入力向け Deep Kernel Learning 回帰モデル。

    注意:
        - このクラス自身は input_transform を持たない
        - wrapper 側で変換済みの train_x / x を受け取る前提
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood,
        ext_type: str = "DEFAULT",
        hidden_dims: Sequence[int] | None = None,
        feature_extractor: nn.Module | None = None,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        input_dim = train_x.size(-1)
        self.input_dim = int(input_dim)
        self.feature_extractor = (
            _make_feature_extractor(
                input_dim=input_dim,
                output_dim=latent_dim,
                ext_type=ext_type,
                hidden_dims=hidden_dims,
            )
            if feature_extractor is None
            else feature_extractor
        ).to(train_x)
        self.latent_dim = _feature_output_dim(
            train_x=train_x,
            feature_extractor=self.feature_extractor,
            latent_dim=latent_dim,
        )
        num_outputs = train_y.shape[-1] if (train_y.ndim > 1) and (train_y.shape[-1] != 1) else None
        self.num_outputs = 1 if num_outputs is None else num_outputs

        batch_shape = torch.Size([] if num_outputs is None else [num_outputs])

        if num_outputs is None:
            self.mean_module = ConstantMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                RBFKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=self.latent_dim,
                ),
                batch_shape=batch_shape,
            )
        else:
            self.mean_module = MultitaskMean(
                ConstantMean(),
                num_tasks=train_y.shape[-1],
            )
            self.covar_module = _PartialObservationMultitaskKernel(
                ScaleKernel(RBFKernel(ard_num_dims=self.latent_dim)),
                num_tasks=train_y.shape[-1],
            )

        # NN の出力特徴を [-1, 1] に押し込む。
        # fold内で特徴がほぼ一定になっても0除算しない。
        self.scale_to_bounds = StableScaleToBounds(-1.0, 1.0)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): すでに wrapper 側で整形済みの入力

        Returns:
            MultivariateNormal | MultitaskMultivariateNormal
        """
        projected_x = self.feature_extractor(x)
        _validate_projected_features(
            x=x,
            projected_x=projected_x,
            latent_dim=self.latent_dim,
        )
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        if self.num_outputs == 1:
            return MultivariateNormal(mean_x, covar_x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class DeepKernelMixed(BatchedMultiOutputGPyTorchModel, ExactGP):
    """
    混合入力（連続 + カテゴリ）向け Deep Kernel Learning 回帰モデル。

    設計:
        - 連続列だけを feature_extractor に通す
        - カテゴリ列はそのまま使う
        - 結果を元の列順に復元して mixed kernel に渡す

    注意:
        - このクラス自身は input_transform を持たない
        - wrapper 側で、連続列だけ transform 済みの train_x / x を受け取る前提
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        cat_dims,
        likelihood,
        ext_type: str = "DEFAULT",
        hidden_dims: Sequence[int] | None = None,
        feature_extractor: nn.Module | None = None,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        d = train_x.shape[-1]
        self._num_outputs = train_y.shape[-1] if (train_y.ndim > 1) and (train_y.shape[-1] != 1) else 1

        self._ignore_X_dims_scaling_check = cat_dims

        self.cat_dims = normalize_indices(indices=cat_dims, d=d)
        self.ord_dims = sorted(set(range(d)) - set(self.cat_dims))

        aug_batch_shape = train_x.shape[:-2]

        # 連続列の次元数
        cont_dim = len(self.ord_dims)

        # 連続列がある場合のみ feature extractor を使う
        if cont_dim > 0:
            self.feature_extractor = (
                _make_feature_extractor(
                    input_dim=cont_dim,
                    output_dim=latent_dim,
                    ext_type=ext_type,
                    hidden_dims=hidden_dims,
                )
                if feature_extractor is None
                else feature_extractor
            ).to(train_x)
            self.latent_dim = _feature_output_dim(
                train_x=train_x[..., self.ord_dims],
                feature_extractor=self.feature_extractor,
                latent_dim=latent_dim,
            )
            self.scale_to_bounds = StableScaleToBounds(-1.0, 1.0)
        else:
            if feature_extractor is not None:
                raise ValueError("feature_extractor cannot be used when all inputs are categorical.")
            if latent_dim not in (None, 0):
                raise ValueError("latent_dim cannot be set when all inputs are categorical.")
            self.feature_extractor = nn.Identity()
            self.scale_to_bounds = nn.Identity()
            self.latent_dim = 0

        self._preserve_input_layout = self.latent_dim == cont_dim
        if self._preserve_input_layout:
            self.kernel_ord_dims = self.ord_dims
            self.kernel_cat_dims = self.cat_dims
        else:
            self.kernel_ord_dims = list(range(self.latent_dim))
            self.kernel_cat_dims = list(range(self.latent_dim, self.latent_dim + len(self.cat_dims)))

        if self._num_outputs == 1:
            self.mean_module = ConstantMean(batch_shape=aug_batch_shape)
        else:
            self.mean_module = MultitaskMean(
                ConstantMean(),
                num_tasks=train_y.shape[-1],
            )

        # --- mixed kernel 構築 ---
        if len(self.ord_dims) == 0:
            # カテゴリのみ
            base_kernel = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(self.cat_dims),
                    active_dims=self.kernel_cat_dims,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        else:
            cont_kernel_factory = get_covar_module_with_dim_scaled_prior

            cont_kernel = cont_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=self.latent_dim,
                active_dims=self.kernel_ord_dims,
            )

            cat_kernel = CategoricalKernel(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(self.cat_dims),
                active_dims=self.kernel_cat_dims,
                lengthscale_constraint=GreaterThan(1e-6),
            )

            sum_kernel = ScaleKernel(cont_kernel + ScaleKernel(cat_kernel))
            prod_kernel = ScaleKernel(cont_kernel * cat_kernel)
            base_kernel = sum_kernel + prod_kernel

        if self._num_outputs == 1:
            self.covar_module = base_kernel
        else:
            self.covar_module = _PartialObservationMultitaskKernel(
                base_kernel,
                num_tasks=train_y.shape[-1],
            )

    def _combine_cont_and_cat(self, x: Tensor) -> Tensor:
        """
        連続列を feature extractor に通し、カテゴリ列をそのまま残して、
        元の列順へ戻す。
        """
        if len(self.ord_dims) == 0:
            return x

        cont_x = x[..., self.ord_dims]
        cat_x = x[..., self.cat_dims]

        projected_cont_x = self.feature_extractor(cont_x)
        _validate_projected_features(
            x=cont_x,
            projected_x=projected_cont_x,
            latent_dim=self.latent_dim,
        )
        projected_cont_x = self.scale_to_bounds(projected_cont_x)

        if self._preserve_input_layout:
            out = torch.empty_like(x)
            out[..., self.ord_dims] = projected_cont_x
            out[..., self.cat_dims] = cat_x
            return out
        return torch.cat([projected_cont_x, cat_x], dim=-1)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): wrapper 側で整形済みの mixed 入力

        Returns:
            MultivariateNormal | MultitaskMultivariateNormal
        """
        mixed_x = self._combine_cont_and_cat(x)

        mean_x = self.mean_module(mixed_x)
        covar_x = self.covar_module(mixed_x)

        if self._num_outputs == 1:
            return MultivariateNormal(mean_x, covar_x)
        return MultitaskMultivariateNormal(mean_x, covar_x)
