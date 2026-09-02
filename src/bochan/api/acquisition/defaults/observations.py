"""Observation-aware acquisition baselines for partially observed outcomes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...configs import AcquisitionConfig, DataContext, ModelBundle


def _as_output_matrix(values: Any) -> Any | None:
    """Return ``values`` as a 2-D tensor when outcome masking is meaningful.

    Multi-output wrappers may expose outcomes either as one ``[n, m]`` tensor or
    as a sequence of per-output tensors. Normalize both representations without
    forcing nested tensors through ``torch.as_tensor`` scalar conversion.
    """

    import torch

    if values is None:
        return None

    if torch.is_tensor(values):
        tensor = values
    elif isinstance(values, (list, tuple)) and values and all(torch.is_tensor(value) for value in values):
        columns = []
        n_rows = None
        for value in values:
            column = value
            if column.ndim == 1:
                column = column.unsqueeze(-1)
            if column.ndim != 2:
                return None
            if n_rows is None:
                n_rows = column.shape[0]
            elif column.shape[0] != n_rows:
                return None
            columns.append(column)
        tensor = torch.cat(columns, dim=-1)
    else:
        try:
            tensor = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError):
            return None

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        return None
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor


def _output_names(bundle: ModelBundle, n_outputs: int) -> list[str] | None:
    """Resolve public output names without imposing a model implementation."""

    candidates = [
        getattr(bundle.model, "output_names", None),
        bundle.metadata.get("output_names") if bundle.metadata is not None else None,
    ]
    multi = getattr(bundle.model_config, "multi_output_config", None)
    if multi is not None:
        candidates.append(getattr(multi, "output_names", None))

    for values in candidates:
        if values is None:
            continue
        names = [str(value) for value in values]
        if len(names) == n_outputs:
            return names
    return None


def _resolve_output_index(
    bundle: ModelBundle,
    output: Any,
    *,
    n_outputs: int,
) -> int:
    """Resolve an integer or named objective output to a column index."""

    if isinstance(output, str):
        names = _output_names(bundle, n_outputs)
        if names is None:
            raise ValueError(
                "String objective outputs require model.output_names or configured "
                "multi-output names before an observation-aware baseline can be built."
            )
        try:
            return names.index(output)
        except ValueError as exc:
            raise ValueError(
                f"Unknown objective output {output!r}. Available outputs: {names}."
            ) from exc

    try:
        index = int(output)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid objective output {output!r}.") from exc
    if index < 0 or index >= n_outputs:
        raise ValueError(
            f"Objective output index {index} is out of range for {n_outputs} outputs."
        )
    return index


def _scalar_objective_indices(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    *,
    n_outputs: int,
) -> list[int] | None:
    """Return outputs that must be observed for a scalar objective baseline.

    Vector / multi-objective acquisitions deliberately keep the full baseline.
    NEHVI can use the model posterior over partially observed baseline points, so
    requiring a complete row there would discard useful observations.
    """

    objective_config = config.objective_config
    if objective_config is None:
        return [0] if n_outputs == 1 else None

    mode = str(objective_config.mode)
    if mode == "multi_output":
        return None

    if objective_config.output is not None:
        return [
            _resolve_output_index(
                bundle,
                objective_config.output,
                n_outputs=n_outputs,
            )
        ]

    if objective_config.outputs is not None and mode == "scalar":
        indices = [
            _resolve_output_index(bundle, output, n_outputs=n_outputs)
            for output in objective_config.outputs
        ]
        return list(dict.fromkeys(indices))

    if mode == "scalar" and objective_config.weights is not None:
        weights = list(objective_config.weights)
        if len(weights) == n_outputs:
            indices = [
                index
                for index, weight in enumerate(weights)
                if getattr(weight, "numel", lambda: 1)() == 1 and float(weight) != 0.0
            ]
            return indices or None

    if n_outputs == 1:
        return [0]
    return None


def resolve_observation_aware_baselines(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> DataContext:
    """Restrict automatic scalar baselines to rows observed for that objective.

    The optimizer's default context points ``X_baseline`` / ``Y_baseline`` at
    the complete regression training rows. With partially observed multi-output
    targets this may include a row where the selected scalar objective is NaN.
    Such a row did not observe that objective and should not become an automatic
    baseline for EI/PI/NEI-style scalar acquisition.

    Explicit user baselines are preserved. Training data and ``train_Yvar`` are
    never sliced or mutated here; this helper only derives acquisition context.
    """

    import torch

    train_X = getattr(bundle, "train_X", None)
    train_Y_raw = getattr(bundle, "train_Y", None)
    train_Y = _as_output_matrix(train_Y_raw)
    if train_X is None or train_Y is None or train_Y.shape[0] == 0:
        return context

    # Complete outcome data must be a strict no-op. Besides preserving legacy
    # object identity, this avoids resolving named outputs for workflows that do
    # not need observation masking at all.
    if bool(torch.isfinite(train_Y).all()):
        return context

    n_outputs = int(train_Y.shape[-1])
    indices = _scalar_objective_indices(bundle, config, n_outputs=n_outputs)
    if not indices:
        return context

    # Only replace baselines that are absent or are the optimizer-generated
    # defaults. A custom experimental baseline must remain authoritative.
    automatic_x = context.X_baseline is None or context.X_baseline is train_X
    automatic_y = context.Y_baseline is None or context.Y_baseline is train_Y_raw
    if not (automatic_x and automatic_y):
        return context

    selected = train_Y[:, indices]
    mask = torch.isfinite(selected).all(dim=-1)
    if bool(mask.all()):
        return context
    if not bool(mask.any()):
        raise ValueError(
            "Cannot build an acquisition baseline because the selected scalar "
            f"objective outputs {indices} have no jointly observed rows."
        )

    rows = torch.where(mask)[0].detach().cpu().tolist()
    try:
        x_baseline = train_X[mask]
    except (IndexError, TypeError):
        x_baseline = train_X[rows]

    if torch.is_tensor(train_Y_raw):
        y_baseline = train_Y_raw[mask]
    elif isinstance(train_Y_raw, (list, tuple)) and train_Y_raw and all(
        torch.is_tensor(value) for value in train_Y_raw
    ):
        masked_columns = [
            value[mask.to(device=value.device)]
            for value in train_Y_raw
        ]
        y_baseline = type(train_Y_raw)(masked_columns)
    else:
        try:
            y_baseline = train_Y_raw[mask.detach().cpu().numpy()]
        except (IndexError, TypeError, AttributeError):
            y_baseline = train_Y_raw[rows]

    return replace(
        context,
        X_baseline=x_baseline,
        Y_baseline=y_baseline,
    )


__all__ = ["resolve_observation_aware_baselines"]
