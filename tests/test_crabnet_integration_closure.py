from __future__ import annotations

import tomllib
from pathlib import Path

from bochan.api.registry.model import DEFAULT_MODEL_REGISTRY
from bochan.composition import CrabNetEncoder, MaterialEncoder, TorchSimplexTransform
from bochan.models.regression.gaussian.deep import (
    CrabNetDKLModel,
    CrabNetGPModel,
    CrabNetInputTransform,
)
from bochan.serving.webapp.routers.capabilities import WEB_CAPABILITIES

ROOT = Path(__file__).resolve().parents[1]
CRABNET_REVISION = "d6906fed634a34d9a7cb5f35db2199629fdfd939"
CRABNET_REQUIREMENT = f"crabnet @ git+https://github.com/sparks-baird/CrabNet.git@{CRABNET_REVISION}"


def _registry_locations(model_type: str) -> list[tuple[str, str]]:
    locations: list[tuple[str, str]] = []
    for input_type, task_registry in DEFAULT_MODEL_REGISTRY.raw().items():
        for task_type, models in task_registry.items():
            if model_type in models:
                locations.append((input_type, task_type))
    return locations


def test_crabnet_dependency_is_reproducibly_pinned_for_every_supported_extra() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    extras = project["project"]["optional-dependencies"]
    for extra in ("materials", "web", "all"):
        crabnet_requirements = [
            requirement for requirement in extras[extra] if requirement.lower().startswith("crabnet ")
        ]
        assert crabnet_requirements == [CRABNET_REQUIREMENT], extra

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    locked_source = f"https://github.com/sparks-baird/CrabNet.git?rev={CRABNET_REVISION}#{CRABNET_REVISION}"
    assert 'name = "crabnet"' in lock
    assert locked_source in lock


def test_crabnet_public_stack_has_one_canonical_model_contract() -> None:
    assert issubclass(CrabNetEncoder, MaterialEncoder)
    assert CrabNetInputTransform.__module__.endswith(".deep.crabnet")
    assert TorchSimplexTransform.__module__ == "bochan.composition.simplex"
    assert CrabNetGPModel.__module__.endswith(".deep.crabnet")
    assert CrabNetDKLModel.__module__.endswith(".deep.crabnet")

    for model_type in ("crabnet_gp", "crabnet_dkl"):
        assert _registry_locations(model_type) == [("normal", "regression")]
        assert model_type in WEB_CAPABILITIES["model_types"]
        assert model_type in WEB_CAPABILITIES["crabnet"]["model_types"]

    assert WEB_CAPABILITIES["crabnet"]["default_encoder_training"] == "partial"


def test_crabnet_final_acceptance_contract_is_documented_and_ci_guarded() -> None:
    guide = (ROOT / "docs" / "crabnet_fastapi_web.md").read_text(encoding="utf-8")
    for heading in (
        "## Python and Tabular API",
        "## Tabular FastAPI request",
        "## React workbench",
        "## Reproducibility and final verification",
    ):
        assert heading in guide

    workflow = (ROOT / ".github" / "workflows" / "composition-smoke.yml").read_text(encoding="utf-8")
    for test_path in (
        "tests/test_crabnet_encoder.py",
        "tests/test_crabnet_gp.py",
        "tests/test_tabular_crabnet_models.py",
        "tests/test_crabnet_integration_closure.py",
    ):
        assert test_path in workflow
