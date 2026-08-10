from __future__ import annotations

import copy
from typing import Optional, Sequence

from torch import Tensor
from botorch.models.transforms.input import InputTransform

from .deepgp import (
    DeepOrdinalMixedGPModel,
    DeepOrdinalGPModel,
    _normalize_cat_dims,
)
from bochan.models.components.layers import (
    DeepKernelDeepGPHiddenLayer,
    DeepKernelDeepMixedGPHiddenLayer,
    SkipDeepKernelDeepGPHiddenLayer,
    SkipDeepKernelDeepMixedGPHiddenLayer,
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


class DeepKernelDeepOrdinalGPModel(DeepOrdinalGPModel):
    """Continuous-input Deep Kernel + DeepGP ordinal model.

    Args:
        kernel_hidden_dims: 最終 DeepKernel feature extractor の隠れ層次元。
            None の場合は feature extractor 側の既定値を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        hidden_dims: Optional[Sequence[int]] = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.01,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        beta: float = 1.0,
        model_type: str = "DEFAULT",
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        ext_type: str = "DEFAULT",
        kernel_hidden_dims: Optional[Sequence[int]] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood=None,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            beta=beta,
            model_type=model_type,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
            input_transform=input_transform,
            likelihood=likelihood,
        )
        self.ext_type = str(ext_type)
        self.kernel_hidden_dims = None if kernel_hidden_dims is None else [int(h) for h in kernel_hidden_dims]

        hidden_dims = list(self.hidden_dims)
        use_skip = self.ext_type.lower() == "skip"
        last_input_dim = hidden_dims[-1]

        if use_skip:
            self.last_layer = SkipDeepKernelDeepGPHiddenLayer(
                base_input_dims=last_input_dim,
                skip_input_dims=self.original_input_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                ext_type=self.ext_type,
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )
            self.last_layer.feature_extractor = _make_deepkernel_feature_extractor(
                input_dim=last_input_dim + self.original_input_dim,
                ext_type=self.ext_type,
                hidden_dims=kernel_hidden_dims,
            )
        else:
            self.last_layer = DeepKernelDeepGPHiddenLayer(
                input_dims=last_input_dim,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                ext_type=self.ext_type,
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )
            self.last_layer.feature_extractor = _make_deepkernel_feature_extractor(
                input_dim=last_input_dim,
                ext_type=self.ext_type,
                hidden_dims=kernel_hidden_dims,
            )
        self.to(train_X)

    def _get_rebuild_kwargs(self) -> dict:
        kwargs = super()._get_rebuild_kwargs()
        kwargs.update({
            "ext_type": self.ext_type,
            "kernel_hidden_dims": copy.deepcopy(self.kernel_hidden_dims),
        })
        return kwargs


class DeepKernelDeepOrdinalMixedGPModel(DeepOrdinalMixedGPModel):
    """Mixed-input Deep Kernel + DeepGP ordinal model.

    Args:
        kernel_hidden_dims: DeepKernel feature extractor の隠れ層次元。
            None の場合は feature extractor 側の既定値を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        num_inducing: int = 128,
        learn_inducing_locations: bool = True,
        lr: float = 0.01,
        num_epochs: int = 300,
        batch_size: Optional[int] = None,
        beta: float = 1.0,
        model_type: str = "DEFAULT",
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
        ext_type: str = "DEFAULT",
        kernel_hidden_dims: Optional[Sequence[int]] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood=None,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            cat_dims=cat_dims,
            category_counts=category_counts,
            hidden_dims=hidden_dims,
            num_inducing=num_inducing,
            learn_inducing_locations=learn_inducing_locations,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            beta=beta,
            model_type=model_type,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            verbose=verbose,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
            input_transform=input_transform,
            likelihood=likelihood,
        )
        self.ext_type = str(ext_type)
        self.kernel_hidden_dims = None if kernel_hidden_dims is None else [int(h) for h in kernel_hidden_dims]

        d = train_X.shape[-1]
        self.cat_dims = _normalize_cat_dims(self.cat_dims, d)
        self.ord_dims = sorted(set(range(d)) - set(self.cat_dims))

        hidden_dims = list(self.hidden_dims)
        train_X_for_input_layer = self._apply_input_transform(
            train_X,
            apply_input_transform=True,
        )

        self.input_layer = DeepKernelDeepMixedGPHiddenLayer(
            input_dims=d,
            output_dims=hidden_dims[0],
            ord_dims=self.ord_dims,
            cat_dims=self.cat_dims,
            num_inducing=self.num_inducing,
            mean_type="linear",
            ext_type=self.ext_type,
            input_data=train_X_for_input_layer,
            learn_inducing_locations=self.learn_inducing_locations,
        )
        self.input_layer.feature_extractor = _make_deepkernel_feature_extractor(
            input_dim=len(self.ord_dims),
            ext_type=self.ext_type,
            hidden_dims=kernel_hidden_dims,
        )

        if self.ext_type.lower() == "skip":
            self.last_layer = SkipDeepKernelDeepMixedGPHiddenLayer(
                base_input_dims=hidden_dims[-1],
                skip_input_dims=d,
                original_ord_dims=self.ord_dims,
                original_cat_dims=self.cat_dims,
                output_dims=None,
                num_inducing=self.num_inducing,
                mean_type="constant",
                ext_type=self.ext_type,
                input_data=None,
                learn_inducing_locations=self.learn_inducing_locations,
            )
            self.last_layer.feature_extractor = _make_deepkernel_feature_extractor(
                input_dim=hidden_dims[-1] + d,
                ext_type=self.ext_type,
                hidden_dims=kernel_hidden_dims,
            )

        self.to(train_X)

    def _get_rebuild_kwargs(self) -> dict:
        kwargs = super()._get_rebuild_kwargs()
        kwargs.update({
            "ext_type": self.ext_type,
            "kernel_hidden_dims": copy.deepcopy(self.kernel_hidden_dims),
        })
        return kwargs


__all__ = [
    "DeepKernelDeepOrdinalGPModel",
    "DeepKernelDeepOrdinalMixedGPModel",
]
