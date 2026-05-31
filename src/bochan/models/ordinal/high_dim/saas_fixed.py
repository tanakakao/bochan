from __future__ import annotations

"""Compatibility-fixed ordinal MAP-SAAS wrappers.

This module keeps the public class names used by ``bochan.models.ordinal.high_dim``
while avoiding the constructor fallbacks in ``saas.py``.  The main fixes are:

- align constructor calls with the current ``OrdinalGPModel`` API,
- respect custom ``OrdinalLogitLikelihood`` / cutpoint kwargs,
- validate ordinal labels and explicit ``num_classes`` consistently.
"""

from copy import deepcopy
from typing import Any, Optional, Sequence
import warnings

import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.components.saas import (
    build_map_saas_covar_module,
    flatten_targets,
    to_device_dtype_transform,
)
from bochan.models.ordinal.base import OrdinalGPModel
from .saas import (
    SaasOrdinalGPModel as _LegacySaasOrdinalGPModel,
    SaasOrdinalMixedGPModel as _LegacySaasOrdinalMixedGPModel,
)


def _labels_as_long(train_Y: Tensor) -> Tensor:
    y_raw = flatten_targets(torch.as_tensor(train_Y))
    if y_raw.numel() == 0:
        raise ValueError("Cannot infer num_classes from empty train_Y.")
    if y_raw.dtype.is_floating_point and not torch.allclose(y_raw, y_raw.round()):
        raise ValueError("Ordinal labels must be integer-valued.")
    y = y_raw.long()
    if y.min().item() < 0:
        raise ValueError("Ordinal labels must be non-negative integers.")
    return y


def _infer_num_classes(train_Y: Tensor, num_classes: Optional[int]) -> int:
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
    expected = torch.arange(unique_y.numel(), device=unique_y.device, dtype=unique_y.dtype)
    if not torch.equal(unique_y, expected):
        raise ValueError(
            "When num_classes is None, ordinal labels must be consecutive integers starting at 0. "
            f"Got labels {unique_y.detach().cpu().tolist()}."
        )
    return int(unique_y.numel())


def _resolve_num_classes(
    train_Y: Tensor,
    num_classes: Optional[int],
    ordinal_likelihood: Optional[OrdinalLogitLikelihood],
) -> int:
    if ordinal_likelihood is not None:
        likelihood_num_classes = int(getattr(ordinal_likelihood, "num_classes"))
        if num_classes is None:
            num_classes = likelihood_num_classes
        elif int(num_classes) != likelihood_num_classes:
            raise ValueError(
                "num_classes and ordinal_likelihood.num_classes are inconsistent. "
                f"num_classes={int(num_classes)}, likelihood.num_classes={likelihood_num_classes}."
            )
    return _infer_num_classes(train_Y, num_classes)


def _warn_if_train_yvar_is_provided(train_Yvar: Optional[Tensor]) -> None:
    if train_Yvar is not None:
        warnings.warn(
            "train_Yvar is accepted for API compatibility but is ignored by ordinal SAAS models. "
            "OrdinalLogitLikelihood does not use Gaussian observation-noise variances.",
            UserWarning,
            stacklevel=3,
        )


def _pop_inducing_points_num_alias(kwargs: dict[str, Any], num_inducing_points: int) -> int:
    if "inducing_points_num" not in kwargs:
        return int(num_inducing_points)
    alias_value = int(kwargs.pop("inducing_points_num"))
    if int(num_inducing_points) != 20 and int(num_inducing_points) != alias_value:
        raise ValueError(
            "Both num_inducing_points and inducing_points_num were specified with different values. "
            f"num_inducing_points={num_inducing_points}, inducing_points_num={alias_value}."
        )
    return alias_value


def _flatten_ordinal_targets(y: Tensor) -> Tensor:
    return flatten_targets(y).long()


class SaasOrdinalGPModel(_LegacySaasOrdinalGPModel):
    """MAP-SAAS style ordinal GP aligned with the current ``OrdinalGPModel`` API."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        train_Yvar: Optional[Tensor] = None,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        likelihood: Optional[OrdinalLogitLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        num_inducing_points = _pop_inducing_points_num_alias(kwargs, num_inducing_points)
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        _warn_if_train_yvar_is_provided(train_Yvar)

        if ordinal_likelihood is None:
            ordinal_likelihood = likelihood
        resolved_num_classes = _resolve_num_classes(train_Y, num_classes, ordinal_likelihood)

        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.train_inputs_raw = (train_X.detach().clone(),)

        input_transform = to_device_dtype_transform(input_transform, train_X)
        if covar_module is None:
            covar_module = build_map_saas_covar_module(
                train_X=train_X,
                input_transform=input_transform,
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )

        OrdinalGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            num_classes=resolved_num_classes,
            inducing_points_num=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            **kwargs,
        )

        if ordinal_likelihood is not None:
            self.likelihood = to_device_dtype_transform(ordinal_likelihood, train_X)

        self.num_classes = int(resolved_num_classes)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_targets = _flatten_ordinal_targets(train_Y).to(device=train_X.device)
        self.model.train_targets = self.train_targets

    def probability_posterior(self, X: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        return self.class_probs(X)


class SaasOrdinalMixedGPModel(_LegacySaasOrdinalMixedGPModel):
    """Mixed-input MAP-SAAS ordinal GP with the fixed parent constructor."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        cat_dims: Optional[Sequence[int]] = None,
        train_Yvar: Optional[Tensor] = None,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        likelihood: Optional[OrdinalLogitLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | Tensor | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        fix_first_cutpoint: bool = True,
        init_gap: float = 1.0,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        num_inducing_points = _pop_inducing_points_num_alias(kwargs, num_inducing_points)
        train_X = torch.as_tensor(train_X)
        train_Y = torch.as_tensor(train_Y, device=train_X.device)
        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()

        encoded_train_X = self._init_one_hot_encoding(train_X=train_X, cat_dims=cat_dims)
        self.encoded_train_inputs_raw = (encoded_train_X.detach().clone(),)
        expanded_input_transform = self._maybe_expand_input_transform(input_transform)
        encoded_inducing_points = self._canonicalize_inducing_points_for_encoded_space(inducing_points)

        SaasOrdinalGPModel.__init__(
            self,
            train_X=encoded_train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            train_Yvar=train_Yvar,
            ordinal_likelihood=ordinal_likelihood,
            likelihood=likelihood,
            input_transform=expanded_input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=encoded_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            tau=tau,
            saas_log_scale=saas_log_scale,
            saas_nu=saas_nu,
            fix_first_cutpoint=fix_first_cutpoint,
            init_gap=init_gap,
            eps=eps,
            **kwargs,
        )

        self.encoded_train_inputs = getattr(self.model, "train_inputs", getattr(self, "train_inputs", (encoded_train_X,)))
        if len(self.encoded_train_inputs) > 0:
            self._check_encoded_categorical_blocks_unchanged(
                X_encoded=encoded_train_X,
                X_tf=self.encoded_train_inputs[0],
                name=f"{self.__class__.__name__}.training_input_transform",
            )
        self.encoded_inducing_points_raw = encoded_inducing_points

        self.train_X_raw = train_X.detach().clone()
        self.train_Y_raw = train_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else train_Yvar.detach().clone()
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X.detach().clone(),)
        self.train_targets = _flatten_ordinal_targets(train_Y).to(device=train_X.device)
        self.model.train_targets = self.train_targets

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "SaasOrdinalMixedGPModel":
        if noise is not None:
            warnings.warn(
                "noise is ignored by ordinal SAAS models because OrdinalLogitLikelihood "
                "does not use Gaussian observation-noise variances.",
                UserWarning,
                stacklevel=2,
            )
        new_model = super().condition_on_observations(X=X, Y=Y, noise=None, **kwargs)
        if not isinstance(new_model, SaasOrdinalMixedGPModel):
            new_model.__class__ = SaasOrdinalMixedGPModel
        return new_model


__all__ = ["SaasOrdinalGPModel", "SaasOrdinalMixedGPModel"]
