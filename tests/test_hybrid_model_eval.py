from __future__ import annotations

import torch
from torch import nn

from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec


class _InputTransform:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def preprocess_transform(self, X: torch.Tensor) -> torch.Tensor:
        raise AssertionError("wrapper-level transform cache should not be used")


class _SubModel(nn.Module):
    def __init__(self, input_transform: object) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.train_inputs = (torch.zeros(3, 2, dtype=torch.double),)
        self.train_targets = torch.zeros(3, dtype=torch.double)
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()


def test_hybrid_eval_skips_wrapper_level_transformed_input_cache() -> None:
    model_a = _SubModel(_InputTransform())
    model_b = _SubModel(_InputTransform())
    hybrid = HybridMultiOutputModel(
        [
            OutputSpec(name="a", task_type="regression", model=model_a),
            OutputSpec(name="b", task_type="regression", model=model_b),
        ]
    )

    assert hybrid.input_transform is None

    returned = hybrid.eval()

    assert returned is hybrid
    assert hybrid.training is False
    assert model_a.eval_called
    assert model_b.eval_called


def test_hybrid_eval_keeps_shared_input_transform_available() -> None:
    shared_transform = _InputTransform()
    model_a = _SubModel(shared_transform)
    model_b = _SubModel(shared_transform)
    hybrid = HybridMultiOutputModel(
        [
            OutputSpec(name="a", task_type="regression", model=model_a),
            OutputSpec(name="b", task_type="regression", model=model_b),
        ]
    )

    assert hybrid.input_transform is shared_transform

    returned = hybrid.eval()

    assert returned is hybrid
    assert hybrid.input_transform is shared_transform
    assert model_a.eval_called
    assert model_b.eval_called
