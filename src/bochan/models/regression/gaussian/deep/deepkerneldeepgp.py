from typing import List, Optional, Sequence, Union
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import normalize_indices
from .deepgp import DeepGaussianGPModel, DeepGaussianMixedGPModel
from bochan.models.components.layers import DeepKernelDeepGPHiddenLayer, DeepKernelDeepMixedGPHiddenLayer
from bochan.models.components.layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor
import warnings
warnings.simplefilter('ignore')


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


class DeepKernelDeepGaussianGPModel(DeepGaussianGPModel):
    """
    Deep Gaussian Processモデルクラス。

    Args:
        kernel_hidden_dims: 最終 DeepKernel feature extractor の隠れ層次元。
            None の場合は feature extractor 側の既定値を使う。
    """
    def __init__(
        self,
        train_X,
        train_Y,
        train_Yvar=None,
        likelihood=None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",
        ext_type="DEFAULT",
        hidden_dims=[10, 10],
        kernel_hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            likelihood,
            input_transform,
            outcome_transform,
            hidden_dims,
        )
        num_outputs = train_Y.shape[-1]
        self.last_layer = DeepKernelDeepGPHiddenLayer(
            input_dims=hidden_dims[-1],
            output_dims=None if num_outputs == 1 else num_outputs,
            ext_type=ext_type,
            mean_type="constant",
        )
        self.last_layer.feature_extractor = _make_deepkernel_feature_extractor(
            input_dim=hidden_dims[-1],
            ext_type=ext_type,
            hidden_dims=kernel_hidden_dims,
        )
        self.kernel_hidden_dims = None if kernel_hidden_dims is None else [int(h) for h in kernel_hidden_dims]


class DeepKernelDeepGaussianMixedGPModel(DeepGaussianMixedGPModel):
    """
    Deep Gaussian Processモデル（混合データ対応）。

    Args:
        kernel_hidden_dims: 入力 DeepKernel feature extractor の隠れ層次元。
            None の場合は feature extractor 側の既定値を使う。
    """
    def __init__(
        self,
        train_X,
        train_Y,
        cat_dims,
        train_Yvar=None,
        likelihood=None,
        input_transform: Union[str, InputTransform, None] = "DEFAULT",
        outcome_transform: Union[str, OutcomeTransform, None] = "DEFAULT",
        ext_type="DEFAULT",
        hidden_dim=10,
        kernel_hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__(train_X, train_Y, cat_dims, train_Yvar, likelihood, input_transform, outcome_transform, hidden_dim)

        input_dim = train_X.shape[-1]
        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.input_layer = DeepKernelDeepMixedGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            ord_dims=ord_dims,
            cat_dims=cat_dims,
            num_inducing=128,
            mean_type="linear",
            ext_type=ext_type,
            input_data=train_X,
        )
        self.input_layer.feature_extractor = _make_deepkernel_feature_extractor(
            input_dim=len(ord_dims),
            ext_type=ext_type,
            hidden_dims=kernel_hidden_dims,
        )
        self.kernel_hidden_dims = None if kernel_hidden_dims is None else [int(h) for h in kernel_hidden_dims]
