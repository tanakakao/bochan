import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Sequence

from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.constraints import GreaterThan
from gpytorch.utils.grid import ScaleToBounds

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.transforms import normalize_indices

from ..layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor


def _make_feature_extractor(
    input_dim: int,
    ext_type: str = "DEFAULT",
    hidden_dims: Optional[Sequence[int]] = None,
) -> nn.Module:
    """
    特徴抽出器を返す。

    Args:
        input_dim (int): 入力次元
        ext_type (str): "DEFAULT" または "skip"
        hidden_dims (Optional[Sequence[int]]): 隠れ層の次元数。
            None の場合は従来通り [input_dim * 8, input_dim * 4, input_dim * 2] を使う。
    """
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
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        input_dim = train_x.size(-1)
        num_outputs = (
            train_y.shape[-1]
            if (train_y.ndim > 1) and (train_y.shape[-1] != 1)
            else None
        )
        self.num_outputs = 1 if num_outputs is None else num_outputs

        batch_shape = torch.Size([] if num_outputs is None else [num_outputs])

        if num_outputs is None:
            self.mean_module = ConstantMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                RBFKernel(
                    batch_shape=batch_shape,
                    ard_num_dims=input_dim,
                ),
                batch_shape=batch_shape,
            )
        else:
            self.mean_module = MultitaskMean(
                ConstantMean(),
                num_tasks=train_y.shape[-1],
            )
            self.covar_module = MultitaskKernel(
                ScaleKernel(
                    RBFKernel(ard_num_dims=input_dim)
                ),
                num_tasks=train_y.shape[-1],
            )

        self.feature_extractor = _make_feature_extractor(
            input_dim=input_dim,
            ext_type=ext_type,
            hidden_dims=hidden_dims,
        )

        # NN の出力特徴を [-1, 1] に押し込む
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): すでに wrapper 側で整形済みの入力

        Returns:
            MultivariateNormal | MultitaskMultivariateNormal
        """
        projected_x = self.feature_extractor(x)
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
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        d = train_x.size(-1)
        self._num_outputs = (
            train_y.shape[-1]
            if (train_y.ndim > 1) and (train_y.shape[-1] != 1)
            else 1
        )

        self._ignore_X_dims_scaling_check = cat_dims

        self.cat_dims = normalize_indices(indices=cat_dims, d=d)
        self.ord_dims = sorted(set(range(d)) - set(self.cat_dims))

        aug_batch_shape = train_x.shape[:-2]

        # 連続列の次元数
        cont_dim = len(self.ord_dims)

        # 連続列がある場合のみ feature extractor を使う
        if cont_dim > 0:
            self.feature_extractor = _make_feature_extractor(
                input_dim=cont_dim,
                ext_type=ext_type,
                hidden_dims=hidden_dims,
            )
            self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)
        else:
            self.feature_extractor = nn.Identity()
            self.scale_to_bounds = nn.Identity()

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
                    active_dims=self.cat_dims,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            )
        else:
            cont_kernel_factory = get_covar_module_with_dim_scaled_prior

            cont_kernel = cont_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(self.ord_dims),
                active_dims=self.ord_dims,
            )

            cat_kernel = CategoricalKernel(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(self.cat_dims),
                active_dims=self.cat_dims,
                lengthscale_constraint=GreaterThan(1e-6),
            )

            sum_kernel = ScaleKernel(
                cont_kernel + ScaleKernel(cat_kernel)
            )
            prod_kernel = ScaleKernel(
                cont_kernel * cat_kernel
            )
            base_kernel = sum_kernel + prod_kernel

        if self._num_outputs == 1:
            self.covar_module = base_kernel
        else:
            self.covar_module = MultitaskKernel(
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
        projected_cont_x = self.scale_to_bounds(projected_cont_x)

        out = torch.empty_like(x)
        out[..., self.ord_dims] = projected_cont_x
        out[..., self.cat_dims] = cat_x
        return out

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
