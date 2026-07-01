from __future__ import annotations

from bochan.api.optimizer_api import (
    OptimizeConfig,
    _force_sequential_for_kronecker,
)


class KroneckerMultiTaskGP:
    pass


class SingleTaskGP:
    pass


class Acquisition:
    def __init__(self, model):
        self.model = model


def test_kronecker_q_batch_forces_sequential_optimization() -> None:
    config = OptimizeConfig(q=3, sequential=False)

    resolved = _force_sequential_for_kronecker(
        Acquisition(KroneckerMultiTaskGP()),
        config,
    )

    assert resolved is not config
    assert resolved.q == 3
    assert resolved.sequential is True
    assert config.sequential is False


def test_nested_kronecker_model_is_detected() -> None:
    wrapped_model = Acquisition(Acquisition(KroneckerMultiTaskGP()))
    config = OptimizeConfig(q=2, sequential=False)

    resolved = _force_sequential_for_kronecker(wrapped_model, config)

    assert resolved.sequential is True


def test_non_kronecker_model_keeps_joint_q_batch() -> None:
    config = OptimizeConfig(q=3, sequential=False)

    resolved = _force_sequential_for_kronecker(
        Acquisition(SingleTaskGP()),
        config,
    )

    assert resolved is config
    assert resolved.sequential is False


def test_kronecker_q_one_does_not_need_sequential_override() -> None:
    config = OptimizeConfig(q=1, sequential=False)

    resolved = _force_sequential_for_kronecker(
        Acquisition(KroneckerMultiTaskGP()),
        config,
    )

    assert resolved is config
    assert resolved.sequential is False


def test_kronecker_non_scipy_backend_is_not_overridden() -> None:
    config = OptimizeConfig(q=3, optimizer="torch", sequential=False)

    resolved = _force_sequential_for_kronecker(
        Acquisition(KroneckerMultiTaskGP()),
        config,
    )

    assert resolved is config
    assert resolved.sequential is False
