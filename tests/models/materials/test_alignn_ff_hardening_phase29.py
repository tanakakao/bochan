from __future__ import annotations

import pytest
import torch

from bochan.models.regression.gaussian.materials.structure import alignn_ff_residual


def test_alignn_ff_structure_indices_validate_range_on_cpu() -> None:
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    indices, leading = alignn_ff_residual._structure_indices(X, num_structures=2)

    assert indices.tolist() == [0, 1]
    assert leading == torch.Size([2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_alignn_ff_structure_indices_validate_range_on_cuda() -> None:
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double, device="cuda")

    indices, leading = alignn_ff_residual._structure_indices(X, num_structures=2)

    assert indices.cpu().tolist() == [0, 1]
    assert leading == torch.Size([2])


def test_alignn_ff_structure_indices_reject_out_of_range() -> None:
    X = torch.tensor([[2.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="outside the configured structure bank"):
        alignn_ff_residual._structure_indices(X, num_structures=2)


def test_alignn_ff_missing_dependency_message_matches_install_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_alignn(name: str) -> object:
        if name.startswith("alignn.ff"):
            raise ImportError(name)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(alignn_ff_residual, "import_module", missing_alignn)

    with pytest.raises(ImportError, match=r"python -m pip install alignn==2026\.8\.11"):
        alignn_ff_residual._load_alignn_ff_calculator("alignnff_wt10")
