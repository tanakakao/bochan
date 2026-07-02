from __future__ import annotations

from typing import Optional, Sequence

from torch.nn import Module

from bochan.models.hybrid.multi_output import HybridMultiOutputModel
from bochan.models.hybrid.specs import OutputSpec


class _DummyModel(Module):
    """HybridMultiOutputModel の初期化だけを検証する最小モデル。"""

    def __init__(self, cat_dims: Optional[Sequence[int]]) -> None:
        super().__init__()
        self.cat_dims = None if cat_dims is None else list(cat_dims)
        self.input_transform = None


def _make_spec(name: str, cat_dims: Optional[Sequence[int]]) -> OutputSpec:
    return OutputSpec(
        name=name,
        task_type="multiclass",
        model=_DummyModel(cat_dims),
    )


def test_none_cat_dims_are_normalized_to_empty_list() -> None:
    model = HybridMultiOutputModel(
        specs=[
            _make_spec("output_0", None),
            _make_spec("output_1", None),
        ]
    )

    assert model.cat_dims == []


def test_matching_cat_dims_are_preserved() -> None:
    model = HybridMultiOutputModel(
        specs=[
            _make_spec("output_0", [1, 3]),
            _make_spec("output_1", (1, 3)),
        ]
    )

    assert model.cat_dims == [1, 3]


def test_mismatched_cat_dims_fall_back_to_empty_list() -> None:
    model = HybridMultiOutputModel(
        specs=[
            _make_spec("output_0", None),
            _make_spec("output_1", [1]),
        ]
    )

    assert model.cat_dims == []
