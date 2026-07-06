"""Constraint compatibility for multiclass hypervolume acquisitions."""

# ruff: noqa: I001

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

from botorch.utils.safe_math import fatmoid
import torch
from torch import Tensor


_APPLIED = False


def _own_init_signature(acquisition_cls: type) -> inspect.Signature | None:
    """Return the class's own initializer signature without inherited metadata."""

    initializer = acquisition_cls.__dict__.get("__init__")
    if initializer is None:
        return None
    initializer = inspect.unwrap(initializer)
    try:
        signature = inspect.signature(initializer, follow_wrapped=False)
    except (TypeError, ValueError):
        return None
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    return signature.replace(parameters=parameters)


def _patch_eta_buffer_assignment(acquisition_cls: type) -> None:
    """Pass eta with the shape required by BoTorch's registered buffer."""

    if acquisition_cls.__dict__.get("_bochan_eta_buffer_patched", False):
        return

    from bochan.acquisition.classification_constraint_compat import (
        _install_init_patch,
    )

    original_init = acquisition_cls.__init__
    public_signature = _own_init_signature(acquisition_cls)

    @wraps(original_init)
    def compatible_init(self, *args, eta=1e-3, **kwargs) -> None:
        constraints = kwargs.get("constraints")
        num_constraints = 0 if constraints is None else len(constraints)
        if torch.is_tensor(eta):
            eta_tensor = eta
            if eta_tensor.ndim == 0 and num_constraints:
                eta_tensor = eta_tensor.expand(num_constraints).clone()
        elif num_constraints:
            eta_tensor = torch.full((num_constraints,), float(eta))
        else:
            eta_tensor = torch.as_tensor(float(eta))
        original_init(self, *args, eta=eta_tensor, **kwargs)

    _install_init_patch(acquisition_cls, compatible_init, public_signature)
    acquisition_cls._bochan_eta_buffer_patched = True
    acquisition_cls._bochan_original_init_before_eta_buffer = original_init


def _constraint_parameter_list(
    value: Tensor | float | bool | list[Any],
    *,
    count: int,
) -> list[Any]:
    """Normalize scalar and per-constraint parameters to a Python list."""

    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(
                "Constraint parameter length must match the number of constraints."
            )
        return value
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value for _ in range(count)]
        if value.numel() != count:
            raise ValueError(
                "Constraint parameter length must match the number of constraints."
            )
        return list(value.unbind())
    return [value for _ in range(count)]


def _multiclass_feasibility_weights(
    *,
    constraints: list[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: Tensor | float,
    fat: list[bool | None] | bool,
) -> Tensor:
    """Compute ``[..., q]`` feasibility without using the class axis as q."""

    eta_values = _constraint_parameter_list(eta, count=len(constraints))
    fat_values = _constraint_parameter_list(fat, count=len(constraints))
    feasibility: Tensor | None = None

    for constraint, eta_value, fat_value in zip(
        constraints,
        eta_values,
        fat_values,
        strict=True,
    ):
        constraint_value = constraint(samples)
        if fat_value is None:
            weight = constraint_value
        else:
            eta_tensor = torch.as_tensor(
                eta_value,
                dtype=constraint_value.dtype,
                device=constraint_value.device,
            )
            if fat_value:
                weight = fatmoid(-constraint_value, tau=eta_tensor)
            else:
                weight = torch.sigmoid(-constraint_value / eta_tensor)
        feasibility = weight if feasibility is None else feasibility * weight

    if feasibility is None:
        raise RuntimeError("At least one constraint is required.")
    return feasibility


def _compute_multiclass_qehvi(
    self,
    samples: Tensor,
    X: Tensor | None = None,
) -> Tensor:
    """Compute qEHVI while keeping multiclass class and candidate axes separate."""

    from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation_compat import (
        validate_hypervolume_objective_q,
    )

    obj = self.objective(samples, X=X)
    validate_hypervolume_objective_q(obj, X)
    q = obj.shape[-2]
    feasibility = None
    if self.constraints is not None:
        feasibility = _multiclass_feasibility_weights(
            constraints=self.constraints,
            samples=samples,
            eta=self.eta,
            fat=self.fat,
        )

    device = self.ref_point.device
    q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
    batch_shape = obj.shape[:-2]
    areas_per_segment = torch.zeros(
        *batch_shape,
        self.cell_lower_bounds.shape[-2],
        dtype=obj.dtype,
        device=device,
    )
    cell_batch_ndim = self.cell_lower_bounds.ndim - 2
    sample_batch_view_shape = torch.Size(
        [
            batch_shape[0] if cell_batch_ndim > 0 else 1,
            *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
            *self.cell_lower_bounds.shape[1:-2],
        ]
    )
    view_shape = (
        *sample_batch_view_shape,
        self.cell_upper_bounds.shape[-2],
        1,
        self.cell_upper_bounds.shape[-1],
    )

    for subset_size in range(1, self.q_out + 1):
        subset_indices = q_subset_indices[f"q_choose_{subset_size}"]
        obj_subsets = obj.index_select(dim=-2, index=subset_indices.view(-1))
        obj_subsets = obj_subsets.view(
            obj.shape[:-2] + subset_indices.shape + obj.shape[-1:]
        )
        overlap_vertices = obj_subsets.min(dim=-2).values
        overlap_vertices = torch.min(
            overlap_vertices.unsqueeze(-3),
            self.cell_upper_bounds.view(view_shape),
        )
        lengths = (
            overlap_vertices - self.cell_lower_bounds.view(view_shape)
        ).clamp_min(0.0)
        areas = lengths.prod(dim=-1)
        if feasibility is not None:
            feasibility_subsets = feasibility.index_select(
                dim=-1,
                index=subset_indices.view(-1),
            ).view(feasibility.shape[:-1] + subset_indices.shape)
            areas = areas * feasibility_subsets.unsqueeze(-3).prod(dim=-1)
        areas = areas.sum(dim=-1)
        areas_per_segment += (-1) ** (subset_size + 1) * areas

    return areas_per_segment.sum(dim=-1).mean(dim=0)


def _patch_multiclass_qehvi(acquisition_cls: type) -> None:
    """Replace BoTorch's raw-sample feasibility shape assumption."""

    if acquisition_cls.__dict__.get("_bochan_multiclass_qehvi_patched", False):
        return
    acquisition_cls._compute_qehvi = _compute_multiclass_qehvi
    acquisition_cls._bochan_multiclass_qehvi_patched = True


def apply_multiclass_constraint_compat() -> None:
    """Patch multiclass EHVI and NEHVI constraint handling."""

    global _APPLIED
    if _APPLIED:
        return

    from bochan.acquisition.multiclass.bayesian_optimization import multi_output

    for acquisition_cls in (
        multi_output.qMultiOutputMulticlassExpectedHypervolumeImprovement,
        multi_output.qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    ):
        _patch_eta_buffer_assignment(acquisition_cls)
        _patch_multiclass_qehvi(acquisition_cls)

    _APPLIED = True


__all__ = ["apply_multiclass_constraint_compat"]
