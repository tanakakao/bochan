from typing import Optional, Sequence, Union

from gpytorch.likelihoods import BernoulliLikelihood
from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import normalize_indices

from .deepgp import (
    BinaryClassificationDeepGPModel,
    BinaryClassificationMixedDeepGPModel,
)
from bochan.models.components.layers import (
    DeepKernelDeepGPHiddenLayer,
    DeepKernelDeepMixedGPHiddenLayer,
)
from bochan.models.components.layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor


def _make_deepkernel_feature_extractor(input_dim: int, ext_type: str, hidden_dims: Optional[Sequence[int]] = None):
    hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
    if str(ext_type).lower() == "skip":
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


class DeepKernelBinaryClassificationDeepGPModel(BinaryClassificationDeepGPModel):
    """
    連続入力向け Deep Kernel + DeepGP の 2値分類モデル。

    Args:
        kernel_hidden_dims: 最終 DeepKernel feature extractor の隠れ層次元。
            None の場合は feature extractor 側の既定値を使う。
    """

    def __init__(
        self,
        train_X,
        train_Y,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        ext_type: str = "DEFAULT",
        list_hidden_dims: Optional[Sequence[int]] = None,
        model_type: str = "DEFAULT",
        kernel_hidden_dims: Optional[Sequence[int]] = None,
    ):
        hidden_dims = list(list_hidden_dims) if list_hidden_dims is not None else [16]

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            list_hidden_dims=hidden_dims,
            model_type=model_type,
        )

        # binary classification の latent はスカラー
        self.last_layer = DeepKernelDeepGPHiddenLayer(
            input_dims=hidden_dims[-1],
            output_dims=None,
            ext_type=ext_type,
            mean_type="constant",
        )
        self.last_layer.feature_extractor = _make_deepkernel_feature_extractor(
            input_dim=hidden_dims[-1],
            ext_type=ext_type,
            hidden_dims=kernel_hidden_dims,
        )
        self.kernel_hidden_dims = None if kernel_hidden_dims is None else [int(h) for h in kernel_hidden_dims]


class DeepKernelBinaryClassificationMixedDeepGPModel(
    BinaryClassificationMixedDeepGPModel
):
    """
    混合入力（連続 + カテゴリ）向け Deep Kernel + DeepGP の 2値分類モデル。

    Args:
        kernel_hidden_dims: 入力 DeepKernel feature extractor の隠れ層次元。
            None の場合は feature extractor 側の既定値を使う。
    """

    def __init__(
        self,
        train_X,
        train_Y,
        cat_dims,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        ext_type: str = "DEFAULT",
        hidden_dim: int = 16,
        model_type: str = "DEFAULT",
        kernel_hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            likelihood=likelihood,
            input_transform=input_transform,
            hidden_dim=hidden_dim,
            model_type=model_type,
        )

        input_dim = train_X.shape[-1]
        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        # input_layer 内部の初期化に使う input_data は、
        # 実際に forward 時に入る空間とそろえる
        train_X_for_input_layer = self._apply_input_transform(
            train_X,
            apply_input_transform=True,
        )

        self.input_layer = DeepKernelDeepMixedGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=128,
            mean_type="linear",
            ext_type=ext_type,
            input_data=train_X_for_input_layer,
        )
        self.input_layer.feature_extractor = _make_deepkernel_feature_extractor(
            input_dim=len(ord_dims),
            ext_type=ext_type,
            hidden_dims=kernel_hidden_dims,
        )
        self.kernel_hidden_dims = None if kernel_hidden_dims is None else [int(h) for h in kernel_hidden_dims]
