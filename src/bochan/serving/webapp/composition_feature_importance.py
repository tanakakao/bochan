"""Composition-aware permutation importance for Web optimization results.

The fitted model consumes CLR / ALR / ILR coordinates, but those coordinates are
not individually interpretable as elemental effects.  This adapter evaluates a
second raw-composition importance view after the normal Web workflow finishes:

* all composition coordinates are permuted jointly and reported as ``組成全体``;
* one element fraction is permuted at a time while the remaining fractions are
  adjusted to keep the composition closed;
* invalid or constraint-violating perturbations fall back to the observed row,
  so the model is never evaluated outside the configured composition domain.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .composition_visualization import (
    _composition_context,
    _composition_validity,
    _formula_strings,
    _observed_composition_frame,
)

_GROUP_NAME = "組成全体"
_DEFAULT_MODE = "proportional"
_MODE_LABELS = {
    "proportional": "残りの元素比を維持",
    "balance": "バランス元素で調整",
}


def _as_mapping(value: Any) -> dict[str, Any]:
    """Return a JSON-like mapping from Pydantic objects or dictionaries."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return dict(vars(value))


def _is_closed(fractions: np.ndarray, *, atol: float = 1e-8) -> np.ndarray:
    """Return rows that are finite, non-negative, and sum to one."""

    values = np.asarray(fractions, dtype=float)
    return (
        np.isfinite(values).all(axis=1)
        & (values >= -atol).all(axis=1)
        & (np.abs(values.sum(axis=1) - 1.0) <= atol)
    )


def _observed_row_permutation(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    decimals: int = 12,
) -> bool:
    """Detect a joint permutation of complete observed composition rows."""

    if not _is_closed(proposed).all():
        return False
    library = {
        tuple(np.round(row, decimals=decimals).tolist())
        for row in np.asarray(baseline, dtype=float)
    }
    return all(
        tuple(np.round(row, decimals=decimals).tolist()) in library
        for row in np.asarray(proposed, dtype=float)
    )


def _resolve_perturbed_fractions(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    mode: str = _DEFAULT_MODE,
    balance_index: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Close element-wise perturbations while preserving their intended axis.

    Args:
        baseline: Observed closed compositions with shape ``[n, d]``.
        proposed: Matrix supplied by the permutation-importance engine.
        mode: ``proportional`` preserves the ratios among all non-selected
            elements. ``balance`` keeps the other elements fixed and assigns the
            residual to ``balance_index``.
        balance_index: Element receiving the residual in balance mode.

    Returns:
        The closed composition matrix and whether the proposal was a joint
        permutation of complete observed composition rows.
    """

    base = np.asarray(baseline, dtype=float)
    values = np.asarray(proposed, dtype=float)
    if base.shape != values.shape or base.ndim != 2:
        raise ValueError("baseline and proposed must be two-dimensional arrays with equal shape.")

    joint = _observed_row_permutation(base, values)
    if joint:
        return np.clip(values, 0.0, None), True

    changed = np.any(np.abs(values - base) > 1e-10, axis=0)
    changed_indices = np.flatnonzero(changed)
    if len(changed_indices) != 1:
        # The core permutation engine calls this path only for one element or
        # the complete group. Keep an explicit, safe fallback for unexpected
        # custom calls instead of sending an open composition to the model.
        clipped = np.clip(values, 0.0, None)
        totals = clipped.sum(axis=1)
        valid = totals > 0.0
        clipped[valid] = clipped[valid] / totals[valid, None]
        clipped[~valid] = base[~valid]
        return clipped, False

    selected = int(changed_indices[0])
    resolved = base.copy()
    selected_values = values[:, selected]
    resolved[:, selected] = selected_values
    residual = 1.0 - selected_values

    use_balance = (
        mode == "balance"
        and balance_index is not None
        and 0 <= int(balance_index) < base.shape[1]
        and int(balance_index) != selected
    )
    if use_balance:
        balance = int(balance_index)
        fixed_indices = [
            index
            for index in range(base.shape[1])
            if index not in {selected, balance}
        ]
        fixed_total = (
            base[:, fixed_indices].sum(axis=1)
            if fixed_indices
            else np.zeros(base.shape[0], dtype=float)
        )
        resolved[:, balance] = residual - fixed_total
        return resolved, False

    remaining = [index for index in range(base.shape[1]) if index != selected]
    weights = np.clip(base[:, remaining], 0.0, None)
    totals = weights.sum(axis=1)
    zero_rows = totals <= 0.0
    if zero_rows.any():
        weights[zero_rows] = 1.0
        totals = weights.sum(axis=1)
    weights = weights / totals[:, None]
    resolved[:, remaining] = residual[:, None] * weights
    return resolved, False


@dataclass
class _CompositionFractionPredictor:
    """Predictor accepting element fractions instead of model coordinates."""

    session: Any
    context: Any
    baseline: np.ndarray
    targets: tuple[str, ...]
    mode: str = _DEFAULT_MODE
    balance_element: str | None = None

    def _repair(self, fractions: np.ndarray, *, joint: bool) -> np.ndarray:
        """Apply configured composition repair and reject invalid constraints."""

        optimizer = self.session.tabular_optimizer
        repaired = np.asarray(fractions, dtype=float).copy()
        if not joint:
            search_space = dict(
                getattr(optimizer, "composition_search_spaces_", None) or {}
            ).get(self.context.site_name)
            if search_space is not None:
                total = float(self.context.config.get("total", 1.0))
                for row_index, row in enumerate(repaired):
                    try:
                        scaled = {
                            element: float(value) * total
                            for element, value in zip(
                                self.context.elements,
                                row,
                                strict=True,
                            )
                        }
                        repaired_row = search_space.repair(scaled)
                        repaired[row_index] = np.asarray(
                            [
                                float(repaired_row[element]) / total
                                for element in self.context.elements
                            ],
                            dtype=float,
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        repaired[row_index] = self.baseline[row_index]

        closed = _is_closed(repaired)
        valid = closed & _composition_validity(
            self.session,
            self.context,
            repaired,
        )
        repaired[~valid] = self.baseline[~valid]
        return repaired

    def _source_frame(self, fractions: np.ndarray) -> Any:
        source = self.session.data.copy()
        source.loc[:, self.context.column] = _formula_strings(
            self.context,
            fractions,
        )
        return source

    def predict(self, X: Any, *, return_result: bool = True, **_: Any) -> Any:
        """Return output means in the format used by the inspection engine."""

        import torch

        proposed = torch.as_tensor(X).detach().cpu().numpy().astype(float)
        balance_index = (
            self.context.elements.index(self.balance_element)
            if self.balance_element in self.context.elements
            else None
        )
        fractions, joint = _resolve_perturbed_fractions(
            self.baseline,
            proposed,
            mode=self.mode,
            balance_index=balance_index,
        )
        repaired = self._repair(fractions, joint=joint)
        prediction = self.session.tabular_optimizer.predict(
            self._source_frame(repaired)
        )
        columns = [f"{target}_mean" for target in self.targets]
        missing = [column for column in columns if column not in prediction]
        if missing:
            raise ValueError(
                "Composition importance requires numeric regression means; "
                f"missing columns: {missing!r}."
            )
        mean = torch.as_tensor(
            prediction.loc[:, columns].to_numpy(dtype=float),
            dtype=torch.double,
        )
        result = SimpleNamespace(mean=mean)
        return result if return_result else mean


def _importance_config(request: Any, n_elements: int) -> Any:
    """Build a predictive-only config matching the Web PI settings."""

    from bochan.inspection import FeatureGroup, FeatureImportanceConfig

    settings = getattr(request, "feature_importance", None)
    raw = _as_mapping(getattr(settings, "config", None))
    allowed = {
        "n_repeats",
        "random_state",
        "scoring",
        "scoring_direction",
        "normalize_importance",
        "clip_negative_importance",
        "return_per_repeat_values",
        "batch_size",
        "unsupported_method_policy",
        "error_policy",
    }
    values = {key: raw[key] for key in allowed if key in raw}
    return FeatureImportanceConfig(
        predictive_methods=("permutation",),
        diagnostic_methods=(),
        compute_noise_importance=False,
        compute_classwise_importance=False,
        feature_groups=(
            FeatureGroup(
                _GROUP_NAME,
                tuple(range(n_elements)),
                "composition",
            ),
        ),
        feature_roles={index: "composition_element" for index in range(n_elements)},
        **values,
    )


def _records_by_scope(importance: Any, targets: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert the core result and normalize elemental importance internally."""

    from bochan.visualization import feature_importance_dataframe

    overall: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []
    for target in targets:
        frame = feature_importance_dataframe(
            importance,
            output_name=target,
            normalized=False,
            sort=False,
            include_negative=True,
        )
        records = frame.drop(columns=["display_value"], errors="ignore").to_dict(
            orient="records"
        )
        element_records = [
            record for record in records if str(record.get("feature")) != _GROUP_NAME
        ]
        positive_total = sum(
            max(float(record.get("mean") or 0.0), 0.0)
            for record in element_records
        )
        for record in records:
            record["evaluation_source"] = "training"
            if str(record.get("feature")) == _GROUP_NAME:
                record["normalized_mean"] = None
                overall.append(record)
                continue
            mean = float(record.get("mean") or 0.0)
            record["normalized_mean"] = (
                max(mean, 0.0) / positive_total if positive_total > 0.0 else 0.0
            )
            record["label"] = f"{record['feature']} 比率"
            elements.append(record)
    return overall, elements


def _importance_enabled(request: Any) -> bool:
    settings = getattr(request, "feature_importance", None)
    return bool(settings is not None and getattr(settings, "enabled", False))


def attach_composition_feature_importance(
    result: dict[str, Any],
    request: Any,
    session: Any,
) -> None:
    """Attach composition-total and element-wise PI to one completed Web result."""

    if not _importance_enabled(request):
        return
    context = _composition_context(session)
    if context is None:
        return

    regression_targets = tuple(
        target
        for target in session.target_columns
        if str(session.target_metadata[target].get("internal_task")) == "regression"
    )
    if not regression_targets:
        return

    settings = getattr(request, "feature_importance", None)
    config_mapping = _as_mapping(getattr(settings, "config", None))
    error_policy = str(config_mapping.get("error_policy", "warn"))
    warnings: list[str] = []
    try:
        import torch

        from bochan.inspection import compute_feature_importance

        observed = _observed_composition_frame(session)
        baseline = observed.loc[
            :, list(context.fraction_features)
        ].to_numpy(dtype=float)
        if baseline.shape[0] < 2:
            raise ValueError("At least two observations are required for permutation importance.")

        mode = _DEFAULT_MODE
        balance_element = None
        predictor = _CompositionFractionPredictor(
            session=session,
            context=context,
            baseline=baseline,
            targets=regression_targets,
            mode=mode,
            balance_element=balance_element,
        )
        target_values = torch.as_tensor(
            session.encoded_targets.loc[
                :, list(regression_targets)
            ].to_numpy(dtype=float),
            dtype=torch.double,
        )
        importance = compute_feature_importance(
            predictor=predictor,
            X=torch.as_tensor(baseline, dtype=torch.double),
            y=target_values,
            task_type=("regression",) * len(regression_targets),
            feature_names=context.elements,
            output_names=regression_targets,
            config=_importance_config(request, len(context.elements)),
            training_data=True,
        )
        overall, elements = _records_by_scope(importance, regression_targets)
        requested_source = result.get("feature_importance_source")
        if requested_source == "cross_validation":
            warnings.append(
                "組成全体・元素別重要度は、制約付き組成摂動を行うため、"
                "最終モデルの学習データ上で評価しています。通常PIはcross-validation集約です。"
            )
        coordinate_features = list(
            dict(
                getattr(
                    session.tabular_optimizer,
                    "composition_transformers_",
                    None,
                )
                or {}
            )[context.site_name].feature_names_
            or ()
        )
        payload = {
            "column": context.column,
            "elements": list(context.elements),
            "coordinate_features": coordinate_features,
            "mode": mode,
            "mode_label": _MODE_LABELS[mode],
            "balance_element": balance_element,
            "evaluation_source": "training",
            "requested_source": requested_source,
            "n_repeats": int(config_mapping.get("n_repeats", 10)),
            "overall": overall,
            "elements": elements,
            "warnings": warnings,
        }
        result["composition_feature_importance"] = payload
        metadata = dict(result.get("metadata") or {})
        metadata["composition_feature_importance"] = {
            "enabled": True,
            "mode": mode,
            "evaluation_source": "training",
            "coordinate_features": coordinate_features,
        }
        result["metadata"] = metadata
    except Exception as exc:
        if error_policy == "raise":
            raise
        warning = f"Composition feature importance failed: {exc}"
        existing = list(result.get("feature_importance_warnings") or [])
        if warning not in existing:
            existing.append(warning)
        result["feature_importance_warnings"] = existing

    if warnings:
        existing = list(result.get("feature_importance_warnings") or [])
        existing.extend(warning for warning in warnings if warning not in existing)
        result["feature_importance_warnings"] = existing


def install_composition_feature_importance() -> None:
    """Wrap the lifecycle workflow before ``app.py`` binds its route callable."""

    from . import workflows

    if getattr(workflows, "_composition_feature_importance_installed", False):
        return
    original = workflows.run_regression_web_workflow

    def workflow_adapter(request: Any, store: Any) -> dict[str, Any]:
        result = original(request, store)
        from .logging import current_request_id
        from .visualization_sessions import get_visualization_session

        run_id = current_request_id()
        if not run_id:
            return result
        try:
            session = get_visualization_session(run_id)
            attach_composition_feature_importance(result, request, session)
            session.result = copy.deepcopy(result)
        except KeyError:
            pass
        return result

    workflows.run_regression_web_workflow = workflow_adapter
    workflows._composition_feature_importance_installed = True


__all__ = [
    "attach_composition_feature_importance",
    "install_composition_feature_importance",
]
