from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from bochan.api import ModelConfig, MultiOutputConfig, OutputConfig
from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.serving.webapp.services.model_reuse import (
    model_reuse_run,
    model_reuse_signature,
    prepare_model_reuse_request,
    register_fitted_model,
    reuse_fitted_tabular_optimizer,
)
from bochan.serving.webapp.services.visualization_sessions import (
    VisualizationSession,
    register_visualization_session,
)


def _request(*, model_type: str = "base", acquisition: str = "EI") -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id="dataset-1",
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        direction="maximize",
        directions={"y": "maximize"},
        model_type=model_type,
        model_kwargs={"web_target_settings": [{"target": "y", "task_type": "regression"}]},
        fit_maxiter=32,
        normalize=True,
        outcome_transform=True,
        input_perturbation=False,
        n_w=8,
        perturbation_std=0.1,
        search_space=[{"name": "x", "type": "numeric", "lower": 0.0, "upper": 1.0}],
        drop_missing=True,
        acquisition=SimpleNamespace(name=acquisition),
    )


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(task_type="regression", multi_output_config=None)


class _FakeTabularOptimizer:
    def __init__(self) -> None:
        self.bo = SimpleNamespace(model=object())
        self.dataset = SimpleNamespace(
            X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            Y=torch.tensor([[1.0], [2.0]], dtype=torch.double),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            cat_dims=[],
            feature_names=["x"],
            target_names=["y"],
        )

    def candidate(self, *args, **kwargs):
        return SimpleNamespace(
            candidates=torch.tensor([[0.5]], dtype=torch.double),
            acq_value=torch.tensor([1.0], dtype=torch.double),
        )


class _FakeElementResolver:
    def normalize(self, value):
        return list(value or [])


class _FakeCandidateService:
    def __init__(self, constraints) -> None:
        self.composition = object()
        self.element_resolver = _FakeElementResolver()
        self.element_constraints = list(constraints)
        self.validated = False

    def projector(self):
        owner = self

        class _Projector:
            def validate(self) -> None:
                owner.validated = True

        return _Projector()


def _register_optimizer(
    run_id: str,
    request: SimpleNamespace,
    optimizer: _FakeTabularOptimizer,
    *,
    hybrid_model: bool = False,
) -> _FakeTabularOptimizer:
    session = VisualizationSession(
        optimizer=optimizer.bo,
        tabular_optimizer=optimizer,
        data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
        encoded_targets=pd.DataFrame({"y": [1.0, 2.0]}),
        feature_columns=["x"],
        target_columns=["y"],
        target_metadata={"y": {"internal_task": "regression"}},
        hybrid_model=hybrid_model,
    )
    register_visualization_session(run_id, session)
    with model_reuse_run(request, None):
        register_fitted_model(run_id)
    return optimizer


def _register_source(run_id: str, request: SimpleNamespace) -> _FakeTabularOptimizer:
    return _register_optimizer(run_id, request, _FakeTabularOptimizer())


def test_prepare_model_reuse_request_removes_web_only_key() -> None:
    request = _request()
    request.model_kwargs = {
        **request.model_kwargs,
        "web_reuse_model_run_id": "source-run",
    }

    cleaned, source_run_id = prepare_model_reuse_request(request)

    assert source_run_id == "source-run"
    assert "web_reuse_model_run_id" not in cleaned.model_kwargs
    assert "web_target_settings" in cleaned.model_kwargs


def test_reuse_clones_optimizer_shell_and_skips_fitting() -> None:
    request = _request()
    source = _register_source("source-run", request)
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})

    with model_reuse_run(request, "source-run") as report:
        reused = reuse_fitted_tabular_optimizer(
            source_run_id="source-run",
            current_run_id="next-run",
            data=data,
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            model_config=_model_config(),
            hybrid_model=False,
        )

    assert reused is not source
    assert reused.bo is not source.bo
    assert reused.bo.model is source.bo.model
    assert reused.dataset is not source.dataset
    assert torch.equal(reused.dataset.X, source.dataset.X)
    assert torch.equal(reused.dataset.Y, source.dataset.Y)
    assert reused.web_model_reused is True
    assert report["model_reused"] is True
    assert report["fit_skipped"] is True
    assert report["source_run_id"] == "source-run"


def test_reuse_rejects_changed_model_settings() -> None:
    source_request = _request()
    _register_source("mismatch-source", source_request)
    changed_request = _request(model_type="saas")

    with (
        model_reuse_run(changed_request, "mismatch-source"),
        pytest.raises(ValueError, match="cannot be reused"),
    ):
        reuse_fitted_tabular_optimizer(
            source_run_id="mismatch-source",
            current_run_id="mismatch-next",
            data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            model_config=_model_config(),
            hybrid_model=False,
        )


def test_acquisition_changes_do_not_change_model_fingerprint() -> None:
    source_request = _request(acquisition="EI")
    source = _register_source("acq-source", source_request)
    changed_acquisition = _request(acquisition="UCB")

    with model_reuse_run(changed_acquisition, "acq-source") as report:
        reused = reuse_fitted_tabular_optimizer(
            source_run_id="acq-source",
            current_run_id="acq-next",
            data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            model_config=_model_config(),
            hybrid_model=False,
        )

    assert reused.bo is not source.bo
    assert reused.bo.model is source.bo.model
    assert report["model_reused"] is True


def test_candidate_target_roles_do_not_change_model_fingerprint() -> None:
    source = _request()
    source.model_kwargs["web_target_settings"] = [
        {
            "target": "y",
            "task_type": "regression",
            "optimize": True,
            "direction": "maximize",
            "goal": "none",
            "value": None,
        }
    ]
    changed = _request()
    changed.direction = "minimize"
    changed.directions = {"y": "minimize"}
    changed.model_kwargs["web_target_settings"] = [
        {
            "target": "y",
            "task_type": "regression",
            "optimize": False,
            "direction": "minimize",
            "goal": "below",
            "value": 0.25,
        }
    ]
    changed.model_kwargs["web_target_roles"] = {
        "y": {"optimize": False, "direction": "minimize"}
    }

    assert model_reuse_signature(source) == model_reuse_signature(changed)


def test_classification_target_class_is_candidate_time_not_fit_time() -> None:
    positive_a = _request()
    positive_b = _request()
    positive_a.model_kwargs["web_target_settings"] = [
        {
            "target": "y",
            "task_type": "classification",
            "target_class": "A",
            "target_classes": ["A"],
        }
    ]
    positive_b.model_kwargs["web_target_settings"] = [
        {
            "target": "y",
            "task_type": "classification",
            "target_class": "B",
            "target_classes": ["B"],
        }
    ]

    assert model_reuse_signature(positive_a) == model_reuse_signature(positive_b)


def test_ordinal_class_order_still_changes_model_fingerprint() -> None:
    order_a = _request()
    order_b = _request()
    order_a.model_kwargs["web_target_settings"] = [
        {"target": "y", "task_type": "ordinal", "class_order": ["low", "high"]}
    ]
    order_b.model_kwargs["web_target_settings"] = [
        {"target": "y", "task_type": "ordinal", "class_order": ["high", "low"]}
    ]

    assert model_reuse_signature(order_a) != model_reuse_signature(order_b)


def test_fixed_value_and_step_do_not_change_model_fingerprint() -> None:
    source = _request()
    changed = _request()
    changed.search_space = [
        {
            "name": "x",
            "type": "numeric",
            "lower": 0.0,
            "upper": 1.0,
            "step": 0.1,
            "fixed": True,
            "fixed_value": 0.5,
        }
    ]

    assert model_reuse_signature(source) == model_reuse_signature(changed)


def test_composition_element_constraints_do_not_change_model_fingerprint() -> None:
    source = _request()
    changed = _request()
    common = {
        "enabled": True,
        "column": "formula",
        "elements": ["Li", "O"],
        "representation": "ilr",
        "normalization": "atomic_fraction",
        "coordinate_bounds": [-8.0, 8.0],
    }
    source.model_kwargs["web_composition"] = {
        **common,
        "element_constraints": [{"rhs": 0.2}],
    }
    changed.model_kwargs["web_composition"] = {
        **common,
        "element_constraints": [{"rhs": 0.8}],
    }

    assert model_reuse_signature(source) == model_reuse_signature(changed)


def test_composition_representation_change_still_rejects_reuse() -> None:
    source = _request()
    changed = _request()
    source.model_kwargs["web_composition"] = {
        "enabled": True,
        "column": "formula",
        "elements": ["Li", "O"],
        "representation": "ilr",
    }
    changed.model_kwargs["web_composition"] = {
        "enabled": True,
        "column": "formula",
        "elements": ["Li", "O"],
        "representation": "fractions",
    }

    assert model_reuse_signature(source) != model_reuse_signature(changed)


def test_reuse_rebuilds_hybrid_target_view_with_current_threshold() -> None:
    source_request = _request()
    source_request.model_kwargs["web_target_settings"] = [
        {
            "target": "y",
            "task_type": "regression",
            "goal": "target",
            "value": 1.0,
        }
    ]
    current_request = _request()
    current_request.model_kwargs["web_target_settings"] = [
        {
            "target": "y",
            "task_type": "regression",
            "goal": "target",
            "value": 3.75,
        }
    ]

    submodel = nn.Linear(1, 1).double()
    source_model = HybridMultiOutputModel(
        [
            OutputSpec(
                name="y",
                task_type="regression",
                model=submodel,
                eq_target=1.0,
            )
        ]
    )
    source_config = ModelConfig(
        task_type="hybrid",
        model_type="base",
        outcome_transform=False,
        multi_output_config=MultiOutputConfig(
            output_configs=[
                OutputConfig(
                    task_type="regression",
                    model_type="base",
                    name="y",
                    output_spec_kwargs={"eq_target": 1.0, "sign": 1.0},
                )
            ],
            output_names=["y"],
            use_hybrid=True,
        ),
    )
    current_config = ModelConfig(
        task_type="hybrid",
        model_type="base",
        outcome_transform=False,
        multi_output_config=MultiOutputConfig(
            output_configs=[
                OutputConfig(
                    task_type="regression",
                    model_type="base",
                    name="y",
                    output_spec_kwargs={"eq_target": 3.75, "sign": 1.0},
                )
            ],
            output_names=["y"],
            use_hybrid=True,
        ),
    )

    source = _FakeTabularOptimizer()
    source.bo = SimpleNamespace(
        model=source_model,
        model_config=source_config,
        bundle=SimpleNamespace(
            model=source_model,
            model_config=source_config,
            metadata={},
        ),
        history=[],
    )
    _register_optimizer("hybrid-source", source_request, source, hybrid_model=True)

    with model_reuse_run(current_request, "hybrid-source") as report:
        reused = reuse_fitted_tabular_optimizer(
            source_run_id="hybrid-source",
            current_run_id="hybrid-next",
            data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={
                "y": {
                    "internal_task": "regression",
                    "goal": "target",
                    "configured_value": 3.75,
                }
            },
            model_config=current_config,
            hybrid_model=True,
        )

    assert reused.bo is not source.bo
    assert reused.bo.model is not source_model
    assert reused.bo.model.models[0] is submodel
    assert reused.bo.model.specs[0].eq_target == pytest.approx(3.75)
    assert reused.bo.bundle.model is reused.bo.model
    assert source_model.specs[0].eq_target == pytest.approx(1.0)
    assert report["fit_skipped"] is True


def test_reuse_refreshes_composition_element_constraints_without_mutating_source() -> None:
    request = _request()
    source = _FakeTabularOptimizer()
    source.composition = object()
    source.candidates = _FakeCandidateService([{"rhs": 0.2}])
    source.candidates.composition = source.composition
    _register_optimizer("composition-source", request, source)

    with model_reuse_run(request, "composition-source"):
        reused = reuse_fitted_tabular_optimizer(
            source_run_id="composition-source",
            current_run_id="composition-next",
            data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            model_config=_model_config(),
            hybrid_model=False,
            composition_config={"element_constraints": [{"rhs": 0.8}]},
        )

    assert reused.candidates is not source.candidates
    assert reused.candidates.element_constraints == [{"rhs": 0.8}]
    assert reused.candidates.validated is True
    assert source.candidates.element_constraints == [{"rhs": 0.2}]
