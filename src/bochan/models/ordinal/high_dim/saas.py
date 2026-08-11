from __future__ import annotations

"""MAP-SAAS ordinal GP models.

The continuous model combines :class:`OrdinalGPModel` with a MAP-SAAS kernel.
The mixed-input model one-hot encodes categorical inputs and keeps the latent GP
in encoded feature space while exposing raw-space candidate inputs publicly.
"""

import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from torch import Tensor

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.components.saas import (
    OneHotEncodingMixin,
    build_map_saas_covar_module,
    concat_optional_noise,
    flatten_targets,
    prepare_mixed_conditioning_data,
    to_device_dtype_transform,
)
from bochan.models.ordinal.base import OrdinalGPModel


def _labels_as_long(train_Y: Tensor) -> Tensor:
    """Validate ordinal labels and return a flat integer tensor."""
    y_raw = flatten_targets(torch.as_tensor(train_Y))
    if y_raw.numel() == 0:
        raise ValueError("Cannot infer num_classes from empty train_Y.")
    if y_raw.dtype.is_floating_point and not torch.allclose(y_raw, y_raw.round()):
        raise ValueError("Ordinal labels must be integer-valued.")
    y = y_raw.long()
    if y.min().item() < 0:
        raise ValueError("Ordinal labels must be non-negative integers.")
    return y


def _infer_num_classes(train_Y: Tensor, num_classes: int | None) -> int:
    """Resolve and validate the number of ordinal classes."""
    y = _labels_as_long(train_Y)
    if num_classes is not None:
        k = int(num_classes)
        if k < 3:
            raise ValueError("num_classes must be >= 3 for ordinal GP models.")
        if y.max().item() >= k:
            raise ValueError(
                "Ordinal labels must be in [0, num_classes - 1]. "
                f"Got max label {int(y.max().item())} for num_classes={k}."
            )
        return k

    unique_y = torch.unique(y).sort().values
    if unique_y.numel() < 3:
        raise ValueError(
            "Ordinal GP requires at least 3 observed classes when num_classes is None. "
            "Pass num_classes explicitly if some classes are currently unobserved."
        )
    expected = torch.arange(
        unique_y.numel(),
        device=unique_y.device,
        dtype=unique_y.dtype,
    )
    if not torch.equal(unique_y, expected):
        raise ValueError(
            "When num_classes is None, ordinal labels must be consecutive integers starting at 0. "
            f"Got labels {unique_y.detach().cpu().tolist()}."
        )
    return int(unique_y.numel())


def _resolve_num_classes(
    train_Y: Tensor,
    num_classes: int | None,
    likelihood: OrdinalLogitLikelihood | None,
) -> int:
    """Resolve ``num_classes`` against an optional custom ordinal likelihood."""
    if likelihood is not None:
        likelihood_num_classes = int(likelihood.num_classes)
        if num_classes is None:
            num_classes = likelihood_num_classes
        elif int(num_classes) != likelihood_num_classes:
            raise ValueError(
                "num_classes and likelihood.num_classes are inconsistent. "
                f"num_classes={int(num_classes)}, likelihood.num_classes={likelihood_num_classes}."
            )
    return _infer_num_classes(train_Y, num_classes)


def _warn_if_train_yvar_is_provided(train_Yvar: Tensor | None) -> None:
    if train_Yvar is not None:
        warnings.warn(
            "train_Yvar is accepted for the common model constructor contract but is ignored "
            "by ordinal SAAS models because OrdinalLogitLikelihood does not use Gaussian "
            "observation-noise variances.",
            UserWarning,
            stacklevel=3,
        )


def _flatten_ordinal_targets(y: Tensor) -> Tensor:
    return flatten_targets(y).long()


class SaasOrdinalGPModel(OrdinalGPModel):
    """Continuous-input MAP-SAAS ordinal GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        num_classes: int | None = None,
        likelihood: OrdinalLogitLikelihood | None = None,
        input_transform: Any | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        num_inducing: int = 20,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        conditioning_steps: int = 50,
        conditioning_lr: float | None = None,
        conditioning_batch_size: int | None = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        _warn_if_train_yvar_is_provided(train_Yvar)

        resolved_num_classes = _resolve_num_classes(train_Y, num_classes, likelihood)
        input_transform = to_device_dtype_transform(input_transform, train_X)
        if covar_module is None:
            covar_module = build_map_saas_covar_module(
                train_X=train_X,
                input_transform=input_transform,
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=resolved_num_classes,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )

        if likelihood is not None:
            self.likelihood = to_device_dtype_transform(likelihood, train_X)

        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        self.train_Yvar_raw = (
            None if train_Yvar is None else torch.as_tensor(train_Yvar, device=train_X.device).detach().clone()
        )
        self.train_targets = _flatten_ordinal_targets(train_Y).to(device=train_X.device)
        self.model.train_targets = self.train_targets

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def num_outputs(self) -> int:
        return 1

    def probability_posterior(self, X: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        return self.class_probs(X)


class SaasOrdinalMixedGPModel(OneHotEncodingMixin, SaasOrdinalGPModel):
    """Mixed continuous/categorical MAP-SAAS ordinal GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
        num_classes: int | None = None,
        likelihood: OrdinalLogitLikelihood | None = None,
        input_transform: Any | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        num_inducing: int = 20,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        conditioning_steps: int = 50,
        conditioning_lr: float | None = None,
        conditioning_batch_size: int | None = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        raw_train_X = train_X.detach().clone()

        encoded_train_X = self._init_one_hot_encoding(
            train_X=train_X,
            cat_dims=list(cat_dims),
        )
        self.encoded_train_inputs_raw = (encoded_train_X.detach().clone(),)
        expanded_input_transform = self._maybe_expand_input_transform(input_transform)
        encoded_inducing_points = self._canonicalize_inducing_points_for_encoded_space(
            inducing_points
        )

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            num_classes=num_classes,
            likelihood=likelihood,
            input_transform=expanded_input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing=num_inducing,
            inducing_points=encoded_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            tau=tau,
            saas_log_scale=saas_log_scale,
            saas_nu=saas_nu,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )

        self.encoded_train_inputs = tuple(
            x.detach().clone() for x in self.train_inputs
        )
        if self.encoded_train_inputs:
            self._check_encoded_categorical_blocks_unchanged(
                X_encoded=encoded_train_X,
                X_tf=self.encoded_train_inputs[0],
                name=f"{self.__class__.__name__}.training_input_transform",
            )
        self.encoded_inducing_points_raw = self.inducing_points_raw.detach().clone()

        # The wrapper accepts raw-space candidates publicly; the latent model keeps
        # its encoded-space training inputs established by the parent constructor.
        self.train_inputs_raw = (raw_train_X,)
        self.train_inputs = (raw_train_X.detach().clone(),)
        self.train_targets = _flatten_ordinal_targets(train_Y).to(device=train_X.device)
        self.model.train_targets = self.train_targets

    @property
    def train_input_raw(self) -> Tensor:
        return self.train_inputs_raw[0]

    @property
    def train_input(self) -> Tensor:
        return self.train_inputs[0]

    @property
    def encoded_train_input_raw(self) -> Tensor:
        return self.encoded_train_inputs_raw[0]

    @property
    def encoded_train_input(self) -> Tensor:
        return self.encoded_train_inputs[0]

    def _set_transformed_inputs(self) -> None:
        return None

    def _canonicalize_posterior_X(self, X: Tensor) -> Tensor:
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_input_raw.device,
            dtype=self.train_input_raw.dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim < 2:
            raise ValueError(f"X must have at least 2 dims, got shape={tuple(X.shape)}.")
        if X.shape[-1] not in (self.raw_dim, self.encoded_dim):
            raise ValueError(
                f"Expected raw dim {self.raw_dim} or encoded dim {self.encoded_dim}, "
                f"got {X.shape[-1]}."
            )
        return X.contiguous()

    def _get_input_transform_for_eval(self, input_transform=None):
        return input_transform if input_transform is not None else getattr(self, "input_transform", None)

    def _apply_transform_raw_first(self, X: Tensor, input_transform=None) -> Tensor:
        """Map raw or encoded inputs into the latent GP's encoded feature space."""
        X = self._canonicalize_posterior_X(X)
        transform = self._get_input_transform_for_eval(input_transform)
        if transform is None:
            return self._to_encoded_feature_space(X).contiguous()

        if X.shape[-1] == self.raw_dim:
            raw_transform_error: Exception | None = None
            try:
                transformed = transform(X)
                if isinstance(transformed, tuple):
                    transformed = transformed[0]
                transformed = self._canonicalize_posterior_X(transformed)
                if transformed.shape[-1] == self.raw_dim:
                    return self._to_encoded_feature_space(transformed).contiguous()
                if transformed.shape[-1] == self.encoded_dim:
                    self._check_encoded_categorical_blocks_unchanged(
                        X_encoded=self._to_encoded_feature_space(X),
                        X_tf=transformed,
                        name=f"{self.__class__.__name__}.raw_input_transform",
                    )
                    return transformed.contiguous()
            except Exception as exc:
                raw_transform_error = exc

            encoded = self._to_encoded_feature_space(X)
            try:
                transformed = transform(encoded)
                if isinstance(transformed, tuple):
                    transformed = transformed[0]
                self._check_encoded_categorical_blocks_unchanged(
                    X_encoded=encoded,
                    X_tf=transformed,
                    name=f"{self.__class__.__name__}.encoded_input_transform",
                )
                return transformed.contiguous()
            except Exception:
                if raw_transform_error is not None:
                    raise raw_transform_error
                raise

        try:
            transformed = transform(X)
            if isinstance(transformed, tuple):
                transformed = transformed[0]
            if transformed.shape[-1] == self.encoded_dim:
                self._check_encoded_categorical_blocks_unchanged(
                    X_encoded=X,
                    X_tf=transformed,
                    name=f"{self.__class__.__name__}.encoded_input_transform",
                )
                return transformed.contiguous()
        except Exception:
            pass
        return X.contiguous()

    def transform_inputs(self, X: Tensor, input_transform=None) -> Tensor:
        return self._apply_transform_raw_first(X, input_transform=input_transform)

    def _to_training_feature_space(self, X: Tensor) -> Tensor:
        return self._apply_transform_raw_first(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ):
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def forward(self, X: Tensor):
        return self.model(self._to_training_feature_space(X))

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Tensor | None = None,
        **kwargs: Any,
    ) -> "SaasOrdinalMixedGPModel":
        """Rebuild the mixed model after appending raw-space observations."""
        X_new_raw, Y_new, Yvar_new = prepare_mixed_conditioning_data(
            X,
            Y,
            noise,
            raw_dim=self.raw_dim,
            encoded_dim=self.encoded_dim,
            decode_fn=self.decode_inputs,
            target_dtype=torch.long,
        )
        train_X_old = self.train_inputs_raw[0]
        train_Y_old = _flatten_ordinal_targets(self.train_targets)
        X_full = torch.cat(
            [
                train_X_old,
                X_new_raw.to(dtype=train_X_old.dtype, device=train_X_old.device),
            ],
            dim=0,
        )
        Y_full = torch.cat(
            [
                train_Y_old,
                Y_new.to(device=train_Y_old.device).long(),
            ],
            dim=0,
        )
        Yvar_full = concat_optional_noise(
            old_Y=train_Y_old.to(dtype=train_X_old.dtype),
            old_Yvar=self.train_Yvar_raw,
            new_Y=Y_new.to(dtype=train_X_old.dtype, device=train_X_old.device),
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        learn_inducing_locations = bool(
            getattr(
                getattr(self.model, "variational_strategy", None),
                "learn_inducing_locations",
                True,
            )
        )
        new_model = self.__class__(
            train_X=X_full,
            train_Y=Y_full,
            train_Yvar=Yvar_full,
            cat_dims=list(self.cat_dims),
            num_classes=self.num_classes,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(getattr(self, "input_transform", None)),
            mean_module=deepcopy(getattr(self.model, "mean_module", None)),
            covar_module=deepcopy(getattr(self.model, "covar_module", None)),
            num_inducing=int(self.encoded_inducing_points_raw.shape[-2]),
            inducing_points=self.encoded_inducing_points_raw.detach().clone(),
            learn_inducing_locations=learn_inducing_locations,
            tau=self.tau,
            saas_log_scale=self.saas_log_scale,
            saas_nu=self.saas_nu,
            fix_first_cutpoint=self.fix_first_cutpoint,
            init_gap=self.init_gap,
            eps=self.eps,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model


__all__ = ["SaasOrdinalGPModel", "SaasOrdinalMixedGPModel"]
