from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from bochan.models.regression.gaussian.materials.common.relaxation import StructureRelaxationResult
from bochan.models.regression.gaussian.materials.structure import factory
from bochan.models.regression.gaussian.materials.structure.relax_acquisition import (
    MaterialRelaxationAcquisitionSelector,
)
from bochan.models.regression.gaussian.materials.structure.relax_rank import MaterialRelaxationRanker


class _FakeRelaxer:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def relax(self, structure: Any, **kwargs: Any) -> StructureRelaxationResult:
        raise AssertionError("Factory tests must not execute a physical relaxation.")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("mace", "mace"),
        (" CHGNet ", "chgnet"),
        ("M3GNET", "m3gnet"),
        ("alignn-ff", "alignn-ff"),
        ("alignn_ff", "alignn-ff"),
        ("alignnff", "alignn-ff"),
    ],
)
def test_normalize_material_backend(value: str, expected: str) -> None:
    assert factory.normalize_material_backend(value) == expected


def test_normalize_material_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Supported backends: mace, chgnet, m3gnet, alignn-ff"):
        factory.normalize_material_backend("matgl")


def test_create_structure_relaxer_uses_lazy_backend_import(monkeypatch: pytest.MonkeyPatch) -> None:
    class_names = {
        ".mace_relaxation": "MACEStructureRelaxer",
        ".chgnet_relaxation": "CHGNetStructureRelaxer",
        ".m3gnet_relaxation": "M3GNetStructureRelaxer",
        ".alignn_ff_relaxation": "ALIGNNFFStructureRelaxer",
    }
    imports: list[tuple[str, str | None]] = []

    def fake_import_module(name: str, package: str | None = None) -> Any:
        imports.append((name, package))
        return SimpleNamespace(**{class_names[name]: _FakeRelaxer})

    monkeypatch.setattr(factory, "import_module", fake_import_module)

    for backend in factory.SUPPORTED_MLIP_BACKENDS:
        relaxer = factory.create_structure_relaxer(backend, model_name="test-model")
        assert isinstance(relaxer, _FakeRelaxer)
        assert relaxer.kwargs == {"model_name": "test-model"}

    assert [name for name, _ in imports] == list(class_names)


def test_workflow_factories_accept_injected_relaxer() -> None:
    relaxer = _FakeRelaxer()

    ranker = factory.create_relaxation_ranker("mace", relaxer=relaxer)
    selector = factory.create_relaxation_acquisition_selector("alignn_ff", relaxer=relaxer)

    assert isinstance(ranker, MaterialRelaxationRanker)
    assert isinstance(selector, MaterialRelaxationAcquisitionSelector)
    assert ranker.relaxer is relaxer
    assert selector.relaxer is relaxer


def test_workflow_factories_reject_mixed_injection_and_kwargs() -> None:
    relaxer = _FakeRelaxer()
    with pytest.raises(ValueError, match="either relaxer or backend keyword arguments"):
        factory.create_relaxation_ranker("mace", relaxer=relaxer, device="cpu")
    with pytest.raises(ValueError, match="either relaxer or backend keyword arguments"):
        factory.create_relaxation_acquisition_selector("mace", relaxer=relaxer, device="cpu")
