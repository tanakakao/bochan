from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from bochan.serving.webapp.app import (
    RegressionRunRequest,
    _profile_with_category_values,
)
from bochan.serving.webapp.search_settings import (
    botorch_linear_constraints,
    feature_constraint_results,
    normalize_feature_constraints,
)
from bochan.serving.webapp.visualization_sessions import (
    VisualizationSession,
    model_details,
    visualization_options,
)


def test_multi_term_feature_constraint_uses_weighted_sum() -> None:
    constraints = normalize_feature_constraints(
        [
            {
                "name": "mixture",
                "terms": [
                    {"column": "x1", "coefficient": 2.0},
                    {"column": "x2", "coefficient": -0.5},
                ],
                "sense": "le",
                "rhs": 3.0,
            }
        ],
        feature_columns=["x1", "x2"],
    )

    equality, inequality = botorch_linear_constraints(
        constraints,
        feature_columns=["x1", "x2"],
    )
    assert equality == []
    assert len(inequality) == 1
    assert inequality[0][0].tolist() == [0, 1]
    assert inequality[0][1].tolist() == [-2.0, 0.5]
    assert inequality[0][2] == -3.0

    results = feature_constraint_results({"x1": 1.0, "x2": 1.0}, constraints)
    assert results[0]["lhs"] == 1.5
    assert results[0]["ok"] is True


def test_regression_request_accepts_selection_count_constraint() -> None:
    request = RegressionRunRequest(
        dataset_id="dataset",
        feature_columns=["x1", "x2", "x3"],
        target_column="y",
        target_columns=["y"],
        k_sparse={
            "enabled": True,
            "columns": ["x1", "x2", "x3"],
            "k": 2,
            "score": "abs",
            "support_selection": "topk",
            "final_priority": "grid",
        },
    )

    assert request.k_sparse is not None
    assert request.k_sparse.enabled is True
    assert request.k_sparse.columns == ["x1", "x2", "x3"]
    assert request.k_sparse.k == 2


def test_web_profile_includes_low_cardinality_numeric_values() -> None:
    data = pd.DataFrame({"category_code": [2, 1, 2, 3], "continuous": [0.1, 0.2, 0.3, 0.4]})
    record = SimpleNamespace(
        data=data,
        profile={
            "n_rows": 4,
            "n_columns": 2,
            "columns": [
                {"name": "category_code", "kind": "numeric", "unique_count": 3},
                {"name": "continuous", "kind": "numeric", "unique_count": 4},
            ],
        },
    )

    profile = _profile_with_category_values(record)
    assert profile["columns"][0]["values"] == [2, 1, 3]
    assert profile["columns"][1]["values"] == [0.1, 0.2, 0.3, 0.4]


def test_visualization_options_include_numeric_features_and_ternary_group() -> None:
    constraint = SimpleNamespace(
        sense="eq",
        rhs=1.0,
        terms=[
            SimpleNamespace(column="a", coefficient=1.0),
            SimpleNamespace(column="b", coefficient=1.0),
            SimpleNamespace(column="c", coefficient=1.0),
        ],
    )
    session = VisualizationSession(
        optimizer=SimpleNamespace(model=SimpleNamespace()),
        tabular_optimizer=SimpleNamespace(dataset=SimpleNamespace(cat_dims=[3])),
        data=pd.DataFrame(),
        encoded_targets=pd.DataFrame(),
        feature_columns=["a", "b", "c", "category"],
        target_columns=["strength", "class"],
        target_metadata={
            "strength": {"internal_task": "regression"},
            "class": {"internal_task": "binary"},
        },
        hybrid_model=True,
        feature_constraints=[constraint],
    )

    options = visualization_options(session)
    assert options["numeric_features"] == ["a", "b", "c"]
    assert options["regression_targets"] == ["strength"]
    assert options["ternary_groups"] == [
        {"features": ["a", "b", "c"], "sum_value": 1.0}
    ]
    assert options["feature_controls"] == {}


def test_visualization_options_build_feature_controls_from_observed_data() -> None:
    """Feature controls use robust defaults from numeric and categorical data."""

    session = VisualizationSession(
        optimizer=SimpleNamespace(model=SimpleNamespace()),
        tabular_optimizer=SimpleNamespace(dataset=SimpleNamespace(cat_dims=[1])),
        data=pd.DataFrame({"amount": [1.0, 3.0, 9.0], "grade": ["A", "B", "A"]}),
        encoded_targets=pd.DataFrame(),
        feature_columns=["amount", "grade"],
        target_columns=["strength"],
        target_metadata={"strength": {"internal_task": "regression"}},
        hybrid_model=False,
    )

    controls = visualization_options(session)["feature_controls"]

    assert controls["amount"] == {
        "kind": "numeric",
        "min": 1.0,
        "max": 9.0,
        "default": 3.0,
    }
    assert controls["grade"] == {
        "kind": "categorical",
        "values": ["A", "B"],
        "default": "A",
    }


def test_model_details_report_hybrid_submodels_and_effective_acquisition() -> None:
    class SubmodelA:
        pass

    class SubmodelB:
        pass

    model = SimpleNamespace(
        models=[SubmodelA(), SubmodelB()],
        specs=[
            SimpleNamespace(name="y1", task_type="regression", model=SubmodelA()),
            SimpleNamespace(name="y2", task_type="binary", model=SubmodelB()),
        ],
    )
    session = VisualizationSession(
        optimizer=SimpleNamespace(model=model),
        tabular_optimizer=SimpleNamespace(dataset=SimpleNamespace(cat_dims=[])),
        data=pd.DataFrame(),
        encoded_targets=pd.DataFrame(),
        feature_columns=["x"],
        target_columns=["y1", "y2"],
        target_metadata={
            "y1": {"internal_task": "regression"},
            "y2": {"internal_task": "binary"},
        },
        hybrid_model=True,
        candidate_result=SimpleNamespace(acqf=SimpleNamespace()),
        request_details={
            "requested_model_type": "base",
            "requested_optimizer": "normal",
            "normalize": True,
            "input_perturbation": False,
            "n_w": 16,
            "perturbation_std": 0.1,
            "model_kwargs": {},
        },
    )
    result = {
        "metadata": {
            "internal_model_type": "base",
            "requested_acquisition": "EHVI",
            "acquisition": "EHVI",
            "acquisition_family": "bayesian_optimization",
            "optimizer": "optimize_acqf",
        }
    }

    details = model_details(session, result)
    assert details["hybrid_model"] is True
    assert len(details["submodel_classes"]) == 2
    assert details["effective_acquisition"] == "EHVI"
    assert details["normalize"] is True
