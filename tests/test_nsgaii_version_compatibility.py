from __future__ import annotations

import torch

from bochan.optim.nsgaii_adapter import _make_version_compatible_optimizer


class _TwoOutputAcquisition:
    def __init__(self) -> None:
        self.model = type("Model", (), {"num_outputs": 2})()

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([X[..., :1], 1.0 - X[..., :1]], dim=-1)


def test_legacy_botorch_signature_omits_new_keywords() -> None:
    received: dict[str, object] = {}

    def legacy_optimize_with_nsgaii(
        acq_function,
        bounds,
        num_objectives,
        q=None,
        ref_point=None,
        objective=None,
        constraints=None,
        population_size=250,
        max_gen=None,
        seed=None,
        fixed_features=None,
    ):
        received.update(locals())
        X = torch.tensor([[0.2], [0.8]], dtype=bounds.dtype, device=bounds.device)
        Y = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=bounds.dtype, device=bounds.device)
        return X, Y

    compatible = _make_version_compatible_optimizer(
        legacy_optimize_with_nsgaii
    )
    X, Y = compatible(
        acq_function=_TwoOutputAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_objectives=2,
        q=2,
        inequality_constraints=None,
        max_attempts=4,
        discrete_choices=None,
        post_processing_func=None,
    )

    assert X.shape == torch.Size([2, 1])
    assert Y.shape == torch.Size([2, 2])
    assert "inequality_constraints" not in received
    assert "max_attempts" not in received
    assert "discrete_choices" not in received
    assert "post_processing_func" not in received


def test_modern_botorch_signature_receives_supported_keywords() -> None:
    received: dict[str, object] = {}

    def modern_optimize_with_nsgaii(
        acq_function,
        bounds,
        num_objectives,
        q=None,
        ref_point=None,
        objective=None,
        constraints=None,
        inequality_constraints=None,
        population_size=250,
        max_gen=None,
        seed=None,
        fixed_features=None,
        max_attempts=2,
        discrete_choices=None,
        post_processing_func=None,
    ):
        received.update(locals())
        X = torch.tensor([[0.2], [0.8]], dtype=bounds.dtype, device=bounds.device)
        Y = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=bounds.dtype, device=bounds.device)
        return X, Y

    linear_constraint = (
        torch.tensor([0]),
        torch.tensor([1.0], dtype=torch.double),
        0.1,
    )
    repair = lambda X: X
    compatible = _make_version_compatible_optimizer(
        modern_optimize_with_nsgaii
    )
    compatible(
        acq_function=_TwoOutputAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_objectives=2,
        q=2,
        inequality_constraints=[linear_constraint],
        max_attempts=4,
        discrete_choices={0: [0.0, 0.5, 1.0]},
        post_processing_func=repair,
    )

    assert received["inequality_constraints"] is not None
    assert received["max_attempts"] == 4
    assert received["discrete_choices"] == {0: [0.0, 0.5, 1.0]}
    assert received["post_processing_func"] is repair


def test_legacy_signature_rejects_requested_input_constraints() -> None:
    def legacy_optimize_with_nsgaii(
        acq_function,
        bounds,
        num_objectives,
        q=None,
        ref_point=None,
        objective=None,
        constraints=None,
        population_size=250,
        max_gen=None,
        seed=None,
        fixed_features=None,
    ):
        raise AssertionError("legacy optimizer must not be called")

    linear_constraint = (
        torch.tensor([0]),
        torch.tensor([1.0], dtype=torch.double),
        0.1,
    )
    compatible = _make_version_compatible_optimizer(
        legacy_optimize_with_nsgaii
    )
    try:
        compatible(
            acq_function=_TwoOutputAcquisition(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            num_objectives=2,
            q=2,
            inequality_constraints=[linear_constraint],
        )
    except NotImplementedError as exc:
        assert "inequality_constraints" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError")


def test_legacy_signature_applies_post_processing_and_recomputes_values() -> None:
    def legacy_optimize_with_nsgaii(
        acq_function,
        bounds,
        num_objectives,
        q=None,
        ref_point=None,
        objective=None,
        constraints=None,
        population_size=250,
        max_gen=None,
        seed=None,
        fixed_features=None,
    ):
        X = torch.tensor([[0.24], [0.76]], dtype=bounds.dtype, device=bounds.device)
        Y = torch.tensor([[0.24, 0.76], [0.76, 0.24]], dtype=bounds.dtype, device=bounds.device)
        return X, Y

    compatible = _make_version_compatible_optimizer(
        legacy_optimize_with_nsgaii
    )
    X, Y = compatible(
        acq_function=_TwoOutputAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_objectives=2,
        q=2,
        discrete_choices={0: [0.0, 0.5, 1.0]},
        post_processing_func=lambda X: X,
    )

    expected_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    expected_Y = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    torch.testing.assert_close(X, expected_X)
    torch.testing.assert_close(Y, expected_Y)
