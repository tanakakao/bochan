from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from botorch.acquisition.monte_carlo import qExpectedImprovement

from bochan.serving.webapp import hybrid_bo_routing


def test_prepare_hybrid_bo_request_attaches_request_local_resolver(monkeypatch) -> None:
    acquisition = SimpleNamespace(
        name="EI",
        acqf_kwargs={"web_family": "bayesian_optimization", "eta": 1e-3},
    )
    request = SimpleNamespace(acquisition=acquisition)
    monkeypatch.setattr(
        hybrid_bo_routing,
        "_uses_single_classification_or_ordinal_objective",
        lambda _request: True,
    )

    prepared = hybrid_bo_routing.prepare_hybrid_objective_bo_request(request)

    assert prepared is not request
    assert prepared.acquisition is not acquisition
    assert "thresholds" not in acquisition.acqf_kwargs
    resolver = prepared.acquisition.acqf_kwargs["thresholds"]
    resolved_kwargs, acqf_cls = resolver._resolve_acqf_kwargs(
        name="EI",
        kwargs=prepared.acquisition.acqf_kwargs,
    )
    assert "thresholds" not in resolved_kwargs
    assert resolved_kwargs["eta"] == 1e-3
    assert acqf_cls is qExpectedImprovement


def test_prepare_hybrid_bo_request_ignores_non_bo_family() -> None:
    request = SimpleNamespace(
        acquisition=SimpleNamespace(
            name="EI",
            acqf_kwargs={"web_family": "active_learning"},
        )
    )

    assert hybrid_bo_routing.prepare_hybrid_objective_bo_request(request) is request


def test_hybrid_bo_routing_is_native_workflow_not_runtime_patch() -> None:
    root = Path("src/bochan/serving/webapp")
    workflows_root = root / "workflows"
    workflow_source = (workflows_root / "__init__.py").read_text(encoding="utf-8")
    routing_source = (root / "hybrid_bo_routing.py").read_text(encoding="utf-8")
    init_source = (root / "__init__.py").read_text(encoding="utf-8")

    assert "def _run_tabular_web_workflow" in workflow_source
    assert "prepare_hybrid_objective_bo_request(request)" in workflow_source
    assert (workflows_root / "tabular.py").is_file()
    assert not (root / "workflows.py").exists()
    assert not (root / "workflows_tabular.py").exists()
    assert "def install_web_hybrid_objective_bo_routing" not in routing_source
    assert "workflows_tabular.run_regression_web_workflow =" not in routing_source
    assert not (root / "runtime_adapters.py").exists()
    assert "install_web_runtime_adapters" not in init_source
