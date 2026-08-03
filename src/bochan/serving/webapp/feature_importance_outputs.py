"""Feature-importance output naming helpers for the Web workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def relabel_feature_importance_outputs(
    result: Any,
    target_columns: Sequence[str],
) -> Any:
    """Replace positional output names with source target-column names.

    Native wide multitask models do not use ``MultiOutputConfig``. Core
    cross-validation therefore names their outputs ``output_0``, ``output_1``,
    and so on, even though permutation importance is evaluated independently
    for each posterior output. The Web response should use the original target
    columns so the summary table and generated figure identifiers stay aligned.

    The supplied result is mutated in place and returned for convenience. If
    its output count does not match the target-column count, no changes are
    made because a positional mapping would be ambiguous.
    """

    outputs = getattr(result, "outputs", None)
    targets = [str(column) for column in target_columns]
    if not isinstance(outputs, dict) or len(outputs) != len(targets):
        return result
    if len(set(targets)) != len(targets):
        return result

    original_items = list(outputs.items())
    name_map: dict[str, str] = {}
    renamed: dict[str, Any] = {}
    for (original_name, output), target_name in zip(
        original_items,
        targets,
        strict=True,
    ):
        name_map[str(original_name)] = target_name
        if hasattr(output, "output_name"):
            output.output_name = target_name
        renamed[target_name] = output

    result.outputs = renamed
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata["output_name_map"] = name_map
        metadata["output_names_source"] = "target_columns"
    return result


__all__ = ["relabel_feature_importance_outputs"]
