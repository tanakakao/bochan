from __future__ import annotations

import copy
from typing import Optional, Sequence

from torch import Tensor
from botorch.models.transforms.input import InputTransform

from .deepgp import (
    OrdinalMixedDeepGPModel,
    OrdinalDeepGPModel,
    _normalize_cat_dims,
    # fit_true_deep_ordinal_gp,
)
from bochan.models.components.layers import (
    DeepKernelDeepGPHiddenLayer,
    DeepKernelDeepMixedGPHiddenLayer,
    SkipDeepKernelDeepGPHiddenLayer,
    SkipDeepKernelDeepMixedGPHiddenLayer,
)

class DeepKernelOrdinalDeepGPModel(OrdinalDeepGPModel):
    """Continuous-input Deep Kernel + DeepGP ordinal model.

    Design:
        - Inherits the regression-style ordinal DeepGP wrapper.
        - Replaces only the final latent layer with a deep-kernel hidden layer.
        - Uses the shared layers module instead of redefining layer classes here.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        list_hidden_dims: Optional[Sequence[int]] = None,
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
        input_transform: Optional[InputTransform] = None,
        likelihood=None,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            list_hidden_dims=list_hidden_dims,
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

        hidden_dims = list(self.list_hidden_dims)
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
        self.to(train_X)

    def _get_rebuild_kwargs(self) -> dict:
        kwargs = super()._get_rebuild_kwargs()
        kwargs.update({"ext_type": self.ext_type})
        return kwargs


class DeepKernelOrdinalMixedDeepGPModel(OrdinalMixedDeepGPModel):
    """Mixed-input Deep Kernel + DeepGP ordinal model.

    Design:
        - Inherits the regression-style mixed ordinal DeepGP wrapper.
        - Replaces the mixed input layer with a deep-kernel mixed hidden layer.
        - Optionally replaces the final layer with a skip deep-kernel layer when
          ``ext_type='skip'`` so the design matches the shared layers module.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]] = None,
        list_hidden_dims: Optional[Sequence[int]] = None,
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
        input_transform: Optional[InputTransform] = None,
        likelihood=None,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            cat_dims=cat_dims,
            category_counts=category_counts,
            list_hidden_dims=list_hidden_dims,
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

        d = train_X.shape[-1]
        self.cat_dims = _normalize_cat_dims(self.cat_dims, d)
        self.ord_dims = sorted(set(range(d)) - set(self.cat_dims))

        hidden_dims = list(self.list_hidden_dims)
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

        self.to(train_X)

    def _get_rebuild_kwargs(self) -> dict:
        kwargs = super()._get_rebuild_kwargs()
        kwargs.update({"ext_type": self.ext_type})
        return kwargs


__all__ = [
    "DeepKernelOrdinalDeepGPModel",
    "DeepKernelOrdinalMixedDeepGPModel",
    # "fit_true_deep_ordinal_gp",
]
