"""Tabular feature-importance diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..config import TabularFeatureGroup
from ..data import resolve_column_indices


class DiagnosticsService:
    """Delegate feature-importance calculation after tabular conversion."""

    def _groups(self, owner: Any, groups: Any) -> list[Any] | None:
        if groups is None:
            return None
        from bochan.inspection import FeatureGroup

        resolved = []
        for value in groups:
            group = (
                value
                if isinstance(value, TabularFeatureGroup)
                else TabularFeatureGroup(
                    name=str(value["name"]),
                    columns=tuple(value["columns"]),
                    role=str(value.get("role", "group")),
                )
            )
            try:
                indices = resolve_column_indices(
                    group.columns,
                    owner.dataset.feature_names,
                )
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            resolved.append(FeatureGroup(group.name, tuple(indices or ()), group.role))
        return resolved

    def feature_importance(
        self,
        owner: Any,
        data: Any | None = None,
        y: Any | None = None,
        *,
        config: Any | Mapping[str, Any] | None = None,
        feature_groups: Any = None,
        feature_names: Any = None,
        target_names: Any = None,
        return_type: str = "result",
    ) -> Any:
        owner._check_fitted()
        from bochan.inspection import FeatureImportanceConfig

        if return_type not in {"result", "dataframe", "both"}:
            raise ValueError("return_type must be result, dataframe, or both.")
        resolved_config = (
            FeatureImportanceConfig()
            if config is None
            else FeatureImportanceConfig(**dict(config))
            if isinstance(config, Mapping)
            else config
        )
        groups = self._groups(owner, feature_groups)
        if groups is not None:
            resolved_config = replace(resolved_config, feature_groups=groups)

        if data is None:
            if y is not None:
                raise ValueError("y cannot be provided when data is None.")
            X_eval, y_eval = None, None
        else:
            model_data = (
                owner.transform_compositions(data)
                if owner.composition.enabled
                else data
            )
            evaluation = owner._to_dataset(
                model_data,
                y,
                data_config=replace(
                    owner.data_config,
                    input_cols=list(owner.dataset.feature_names),
                    target_cols=list(owner.dataset.target_names),
                    category_maps=owner.dataset.category_maps,
                    target_category_maps=owner.dataset.target_category_maps,
                ),
                feature_names=feature_names or owner.dataset.feature_names,
                target_names=target_names or owner.dataset.target_names,
            )
            if evaluation.Y is None:
                raise ValueError("Evaluation targets are required for permutation importance.")
            X_eval, y_eval = evaluation.X, evaluation.Y

        result = owner.bo.feature_importance(
            X_eval,
            y_eval,
            config=resolved_config,
            feature_names=[str(name) for name in owner.dataset.feature_names],
            output_names=[str(name) for name in (target_names or owner.dataset.target_names)],
        )
        owner.feature_importance_result_ = result
        if return_type == "result":
            return result
        frame = self.dataframe(
            owner,
            result=result,
            output_name=None if len(result.outputs) == 1 else "__all__",
        )
        return frame if return_type == "dataframe" else (result, frame)

    def dataframe(
        self,
        owner: Any,
        result: Any | None = None,
        *,
        output_name: str | None = None,
        method: str = "permutation",
        importance_kind: str = "predictive",
        class_label: Any | None = None,
        normalized: bool = False,
        sort: bool = True,
        top_k: int | None = None,
    ) -> Any:
        import pandas as pd
        from bochan.visualization import feature_importance_dataframe

        result = owner.feature_importance_result_ if result is None else result
        if result is None:
            raise RuntimeError("No feature importance result is available.")
        names = (
            list(result.outputs)
            if output_name in {None, "__all__"} and len(result.outputs) > 1
            else [output_name]
        )
        frames = [
            feature_importance_dataframe(
                result,
                output_name=name,
                method=method,
                importance_kind=importance_kind,
                class_label=class_label,
                normalized=normalized,
                sort=sort,
                top_k=top_k,
            )
            for name in names
        ]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


__all__ = ["DiagnosticsService"]
