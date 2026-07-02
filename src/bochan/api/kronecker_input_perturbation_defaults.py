"""High-level objective defaults for perturbed Kronecker regression models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .acquisition_config import AcquisitionConfig
from .configs import ModelBundle, ObjectiveConfig


def _is_multi_output(bundle: ModelBundle) -> bool:
    """Return whether a bundle represents more than one regression output."""

    if bool(bundle.metadata.get("multi_output", False)):
        return True
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _has_explicit_objective(config: AcquisitionConfig) -> bool:
    """Return whether the caller supplied objective behavior explicitly."""

    return bool(
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is not None
    )


def install_kronecker_input_perturbation_objective_defaults() -> None:
    """Install risk-neutral ``n_w`` aggregation for Kronecker multi-output BO.

    The generic high-level default intentionally avoids creating objectives for
    arbitrary multi-output models. For a Kronecker regression model, however,
    every output is part of the same posterior and an input perturbation expands
    each candidate from ``q`` to ``q * n_w``. EHVI, NEHVI, NParEGO, and NSGA-II
    therefore require a multi-output preprocessing objective that averages the
    perturbation dimension before applying their own multi-objective logic.
    """

    from . import engine as engine_module
    from . import engine_defaults as defaults_module

    current = engine_module._resolve_objective_config_n_w_from_input_transform
    marker = "_bochan_kronecker_multi_output_default"
    if bool(getattr(current, marker, False)):
        return

    def resolve_with_kronecker_default(
        *,
        acq_config: AcquisitionConfig,
        bundle: ModelBundle | None,
    ) -> AcquisitionConfig:
        if bundle is not None and not _has_explicit_objective(acq_config):
            task_type = str(bundle.task_type)
            model_type = str(bundle.model_type).replace("-", "").replace("_", "").lower()
            if (
                model_type == "kronecker"
                and task_type in {"regression", "multi_objective"}
                and _is_multi_output(bundle)
            ):
                n_w = engine_module._input_transform_n_w_from_bundle(bundle)
                if n_w is not None:
                    return replace(
                        acq_config,
                        objective_config=ObjectiveConfig(
                            mode="multi_output",
                            n_w=int(n_w),
                            risk_type=None,
                        ),
                    )

        return current(
            acq_config=acq_config,
            bundle=bundle,
        )

    setattr(resolve_with_kronecker_default, marker, True)
    setattr(resolve_with_kronecker_default, "_wrapped", current)
    engine_module._resolve_objective_config_n_w_from_input_transform = (
        resolve_with_kronecker_default
    )
    defaults_module._resolve_objective_config_n_w_from_input_transform = (
        resolve_with_kronecker_default
    )


__all__ = ["install_kronecker_input_perturbation_objective_defaults"]
