from __future__ import annotations

import pytest
import torch

from bochan.models.multifidelity import FidelitySpec, ResolvedFidelitySpec


def test_fidelity_spec_resolves_negative_index_and_target() -> None:
    spec = FidelitySpec(
        fidelity_features=(-1,),
        target_fidelities={-1: 1.0},
    )

    resolved = spec.resolve(
        4,
        bounds=torch.tensor(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 1.0, 5.0, 1.0],
            ]
        ),
    )

    assert isinstance(resolved, ResolvedFidelitySpec)
    assert resolved.fidelity_features == (3,)
    assert resolved.primary_fidelity_feature == 3
    assert resolved.target_fidelities == {3: 1.0}


def test_fidelity_spec_rejects_duplicate_indices_after_resolution() -> None:
    spec = FidelitySpec(fidelity_features=(2, -1))

    with pytest.raises(ValueError, match="Duplicate fidelity dim"):
        spec.resolve(3)


def test_fidelity_spec_rejects_categorical_collision() -> None:
    spec = FidelitySpec(fidelity_features=(-1,))

    with pytest.raises(ValueError, match="must be disjoint"):
        spec.resolve(4, cat_dims=[1, 3])


def test_fidelity_spec_rejects_target_key_outside_fidelity_features() -> None:
    spec = FidelitySpec(
        fidelity_features=(-1,),
        target_fidelities={0: 1.0},
    )

    with pytest.raises(ValueError, match="is not a fidelity feature"):
        spec.resolve(3)


def test_fidelity_spec_rejects_target_outside_bounds() -> None:
    spec = FidelitySpec(
        fidelity_features=(-1,),
        target_fidelities={-1: 1.5},
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="outside bounds"):
        spec.resolve(2, bounds=bounds)


def test_fidelity_spec_rejects_invalid_bounds_shape() -> None:
    spec = FidelitySpec(fidelity_features=(-1,))

    with pytest.raises(ValueError, match=r"bounds must have shape \[2, 3\]"):
        spec.resolve(3, bounds=torch.zeros(3, 2))


def test_fidelity_spec_phase59_accepts_multiple_fidelity_features() -> None:
    spec = FidelitySpec(
        fidelity_features=(-2, -1),
        target_fidelities={-2: 0.8, -1: 1.0},
    )

    resolved = spec.resolve(5)

    assert resolved.fidelity_features == (3, 4)
    assert resolved.target_fidelities == {3: 0.8, 4: 1.0}


def test_fidelity_spec_single_fidelity_compatibility_guard() -> None:
    spec = FidelitySpec(fidelity_features=(1, 2))

    with pytest.raises(ValueError, match="supports exactly one continuous fidelity"):
        spec.resolve(4, single_fidelity_only=True)


def test_fidelity_spec_rejects_nonfinite_target() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        FidelitySpec(
            fidelity_features=(-1,),
            target_fidelities={-1: float("nan")},
        )
