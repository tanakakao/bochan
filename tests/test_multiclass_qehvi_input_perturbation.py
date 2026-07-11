from __future__ import annotations

import pytest
import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from torch import Tensor

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
)
from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation import (
    InputPerturbationMultiOutputObjectiveAdapter,
    infer_input_perturbation_n_w,
    validate_hypervolume_objective_q,
)
from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
    MulticlassTargetProbabilityObjective,
)


class _DummyModel(Model):
    def __init__(self, *, n_w: int = 8) -> None:
        super().__init__()
        self.input_transform = InputPerturbation(
            perturbation_set=torch.zeros(n_w, 2, dtype=torch.double)
        )

    @property
    def num_outputs(self) -> int:
        return 2

    def posterior(self, X: Tensor, *args, **kwargs):  # pragma: no cover - init only
        raise NotImplementedError


class _IdentityMultiOutputObjective(MCMultiOutputObjective):
    def __init__(self) -> None:
        super().__init__()
        self._verify_output_shape = False

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        return samples


def _partitioning() -> FastNondominatedPartitioning:
    return FastNondominatedPartitioning(
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        Y=torch.tensor([[0.5, 0.5]], dtype=torch.double),
    )


def test_infers_n_w_from_model_input_transform() -> None:
    assert infer_input_perturbation_n_w(_DummyModel(n_w=8)) == 8


def test_adapter_reduces_q_times_n_w_to_raw_q() -> None:
    sample_shape = 4
    batch_size = 5
    q = 3
    n_w = 8
    m = 2
    c = 3

    base_objective = MulticlassTargetProbabilityObjective(
        output_target_classes=[1, 2],
        num_outputs=m,
    )
    adapter = InputPerturbationMultiOutputObjectiveAdapter(
        base_objective,
        n_w=n_w,
    )

    logits = torch.randn(
        sample_shape,
        batch_size,
        q * n_w,
        m,
        c,
        dtype=torch.double,
    )
    probabilities = torch.softmax(logits, dim=-1)
    X = torch.zeros(batch_size, q, 2, dtype=torch.double)

    values = adapter(probabilities, X=X)
    expected_expanded = base_objective(probabilities, X=X)
    expected = expected_expanded.reshape(
        sample_shape,
        batch_size,
        q,
        n_w,
        m,
    ).mean(dim=-2)

    assert values.shape == torch.Size([sample_shape, batch_size, q, m])
    assert torch.allclose(values, expected)


def test_qehvi_wraps_default_objective_automatically() -> None:
    acquisition = qMultiOutputMulticlassExpectedHypervolumeImprovement(
        model=_DummyModel(n_w=8),
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        partitioning=_partitioning(),
        output_target_classes=[1, 1],
    )

    assert acquisition.input_perturbation_n_w == 8
    assert isinstance(
        acquisition.objective,
        InputPerturbationMultiOutputObjectiveAdapter,
    )


def test_explicit_custom_objective_is_not_replaced() -> None:
    objective = _IdentityMultiOutputObjective()
    acquisition = qMultiOutputMulticlassExpectedHypervolumeImprovement(
        model=_DummyModel(n_w=8),
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        partitioning=_partitioning(),
        objective=objective,
    )

    assert acquisition.objective is objective


def test_unaggregated_objective_is_rejected_before_subset_enumeration() -> None:
    X = torch.zeros(16, 3, 2, dtype=torch.double)
    objective_values = torch.zeros(64, 16, 24, 2, dtype=torch.double)

    with pytest.raises(RuntimeError, match="did not reduce"):
        validate_hypervolume_objective_q(objective_values, X)


def test_aggregated_objective_q_is_accepted() -> None:
    X = torch.zeros(16, 3, 2, dtype=torch.double)
    objective_values = torch.zeros(64, 16, 3, 2, dtype=torch.double)

    validate_hypervolume_objective_q(objective_values, X)
