from __future__ import annotations

import pytest
import torch
from botorch.models.transforms.input import Normalize

from bochan.models.regression.gaussian.materials import (
    MixedProcessLayout,
    resolve_mixed_process_input_transform,
    resolve_mixed_process_layout,
    select_continuous_process_branch,
)


def test_structure_mixed_layout_preserves_selector_and_numeric_process_order() -> None:
    layout = resolve_mixed_process_layout(5, [2, 4], material_dims=[0])

    assert layout == MixedProcessLayout(
        input_dim=5,
        material_dims=(0,),
        categorical_dims=(2, 4),
        continuous_dims=(0, 1, 3),
        numeric_process_dims=(1, 3),
    )
    assert layout.numeric_process_dim == 2
    assert layout.categorical_process_dim == 2


def test_layout_accepts_negative_botorch_indices() -> None:
    layout = resolve_mixed_process_layout(4, [-1], material_dims=[0])
    assert layout.categorical_dims == (3,)
    assert layout.continuous_dims == (0, 1, 2)


def test_layout_rejects_material_categorical_overlap() -> None:
    with pytest.raises(ValueError, match="cannot be categorical"):
        resolve_mixed_process_layout(3, [0], material_dims=[0])


def test_default_transform_normalizes_numeric_process_only() -> None:
    X = torch.tensor(
        [[0.0, 10.0, 0.0, 100.0], [1.0, 20.0, 1.0, 200.0]],
        dtype=torch.double,
    )
    layout = resolve_mixed_process_layout(4, [2], material_dims=[0])

    transform = resolve_mixed_process_input_transform(X, layout, "DEFAULT")

    assert isinstance(transform, Normalize)
    assert transform.indices.tolist() == [1, 3]


def test_default_transform_is_none_without_numeric_process() -> None:
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    layout = resolve_mixed_process_layout(2, [1], material_dims=[0])
    assert resolve_mixed_process_input_transform(X, layout, "DEFAULT") is None


def test_custom_transform_is_preserved() -> None:
    X = torch.zeros(2, 3, dtype=torch.double)
    layout = resolve_mixed_process_layout(3, [2], material_dims=[0])
    custom = Normalize(d=3, indices=[1])
    assert resolve_mixed_process_input_transform(X, layout, custom) is custom


def test_select_continuous_branch_drops_categories_only() -> None:
    X = torch.tensor([[2.0, 10.0, 1.0, 20.0]], dtype=torch.double)
    layout = resolve_mixed_process_layout(4, [2], material_dims=[0])
    selected = select_continuous_process_branch(X, layout)
    assert torch.equal(selected, X[:, [0, 1, 3]])
