from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from botorch.optim.optimize import optimize_acqf

from bochan.acquisition.binary.active_learning import (
    qBinaryBALD,
    qBinaryIntegratedPosteriorVarianceProxy,
    qBinaryPredictiveEntropy,
    qBinaryProbabilityVariance,
)
from bochan.acquisition.multiclass.active_learning import (
    qMulticlassBALD,
    qMulticlassIntegratedPosteriorVarianceProxy,
    qMulticlassPredictiveEntropy,
    qMulticlassProbabilityVariance,
)
from bochan.acquisition.ordinal.active_learning import (
    qOrdinalBALD,
    qOrdinalFantasyNegIntegratedPosteriorVariance,
    qOrdinalPredictiveEntropy,
    qOrdinalUtilityVariance,
)
from bochan.models.classification.binary.base import BinaryClassificationGPModel
from bochan.models.classification.multiclass.base import MulticlassClassificationGPModel
from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.models.ordinal.base import OrdinalGPModel

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
MC_POINTS = torch.linspace(0.1, 0.9, 5, dtype=DTYPE).unsqueeze(-1)


def _binary_model() -> BinaryClassificationGPModel:
    train_x = torch.linspace(0.05, 0.95, 8, dtype=DTYPE).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=DTYPE).unsqueeze(-1)
    model = BinaryClassificationGPModel(train_X=train_x, train_Y=train_y, num_inducing=6)
    model.eval()
    model.likelihood.eval()
    return model


def _multiclass_model() -> MulticlassClassificationGPModel:
    train_x = torch.linspace(0.05, 0.95, 9, dtype=DTYPE).unsqueeze(-1)
    train_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
    model = MulticlassClassificationGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        num_inducing=6,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _ordinal_model() -> OrdinalGPModel:
    train_x = torch.linspace(0.05, 0.95, 9, dtype=DTYPE).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    model = OrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        num_inducing=6,
        conditioning_steps=1,
        conditioning_lr=0.03,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _hybrid(task_type: str, model) -> HybridMultiOutputModel:
    utility_values = [0.0, 1.0, 2.0] if task_type in {"ordinal", "multiclass"} else None
    return HybridMultiOutputModel(
        [
            OutputSpec(
                name="target",
                task_type=task_type,
                model=model,
                utility_values=utility_values,
                positive_class=1 if task_type == "binary" else None,
            )
        ]
    )


def _make_acquisition(task_type: str, name: str):
    if task_type == "binary":
        native = _binary_model()
        hybrid = _hybrid(task_type, native)
        factories: dict[str, Callable] = {
            "variance": lambda: qBinaryProbabilityVariance(model=hybrid, num_samples=8),
            "predictive_entropy": lambda: qBinaryPredictiveEntropy(model=hybrid, num_samples=8),
            "BALD": lambda: qBinaryBALD(model=hybrid, num_samples=8),
            "NIPV": lambda: qBinaryIntegratedPosteriorVarianceProxy(
                model=hybrid,
                mc_points=MC_POINTS,
            ),
        }
    elif task_type == "multiclass":
        native = _multiclass_model()
        hybrid = _hybrid(task_type, native)
        factories = {
            "variance": lambda: qMulticlassProbabilityVariance(model=hybrid, num_samples=8),
            "predictive_entropy": lambda: qMulticlassPredictiveEntropy(model=hybrid, num_samples=8),
            "BALD": lambda: qMulticlassBALD(model=hybrid, num_samples=8),
            "NIPV": lambda: qMulticlassIntegratedPosteriorVarianceProxy(
                model=hybrid,
                mc_points=MC_POINTS,
                num_samples=8,
            ),
        }
    elif task_type == "ordinal":
        native = _ordinal_model()
        hybrid = _hybrid(task_type, native)
        factories = {
            "variance": lambda: qOrdinalUtilityVariance(model=hybrid),
            "predictive_entropy": lambda: qOrdinalPredictiveEntropy(model=hybrid),
            "BALD": lambda: qOrdinalBALD(model=hybrid, num_samples=8),
            "NIPV": lambda: qOrdinalFantasyNegIntegratedPosteriorVariance(
                model=hybrid,
                mc_points=MC_POINTS,
                num_fantasies=1,
                conditioning_steps=1,
                conditioning_lr=0.03,
            ),
        }
    else:
        raise AssertionError(f"Unknown task_type={task_type!r}")

    acquisition = factories[name]()
    return native, acquisition


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "ordinal"])
@pytest.mark.parametrize("name", ["variance", "predictive_entropy", "BALD", "NIPV"])
def test_one_output_hybrid_single_output_acquisition_optimizes(
    task_type: str,
    name: str,
) -> None:
    """One-output Hybrid wrappers must reach native single-output AL acquisitions."""
    torch.manual_seed(0)
    native, acquisition = _make_acquisition(task_type, name)
    assert acquisition.model is native

    options: dict[str, object] = {"maxiter": 8, "batch_limit": 1}
    if task_type == "ordinal" and name == "NIPV":
        options["with_grad"] = False

    candidates, acq_value = optimize_acqf(
        acq_function=acquisition,
        bounds=BOUNDS,
        q=1,
        num_restarts=1,
        raw_samples=8,
        options=options,
    )

    assert candidates.shape == torch.Size([1, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(torch.as_tensor(acq_value)).all()
    assert torch.all(candidates >= BOUNDS[0])
    assert torch.all(candidates <= BOUNDS[1])
