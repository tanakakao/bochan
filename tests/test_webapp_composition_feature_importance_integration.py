from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from bochan.serving.webapp.composition_feature_importance import (
    attach_composition_feature_importance,
)
from bochan.tabular.composition import (
    CompositionSearchSpace,
    CompositionTransformer,
    normalize_composition,
    parse_formula,
)


def _fraction_frame(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for formula in data["formula"]:
        normalized = normalize_composition(
            parse_formula(formula),
            mode="atomic_fraction",
        )
        rows.append(
            {
                "formula__fraction__Fe": normalized.get("Fe", 0.0),
                "formula__fraction__Co": normalized.get("Co", 0.0),
                "formula__fraction__Ni": normalized.get("Ni", 0.0),
            }
        )
    return pd.concat(
        [
            data.reset_index(drop=True),
            pd.DataFrame(rows),
        ],
        axis=1,
    )


def test_attach_composition_feature_importance_runs_core_permutation() -> None:
    data = pd.DataFrame(
        {
            "formula": [
                "Fe2Co3Ni5",
                "Fe4Co4Ni2",
                "Fe6Co1Ni3",
                "Fe3Co5Ni2",
            ],
            "temperature": [900.0, 920.0, 940.0, 960.0],
            "property": [2.0, 4.0, 6.0, 3.0],
        }
    )
    transformer = CompositionTransformer(
        elements=["Fe", "Co", "Ni"],
        representation="ilr",
        prefix="formula",
    ).fit(data["formula"])
    config = {
        "column": "formula",
        "elements": ("Fe", "Co", "Ni"),
        "normalization": "atomic_fraction",
        "representation": "ilr",
        "total": 1.0,
        "precision": 6,
        "bounds": {},
        "steps": {},
        "min_components": 1,
        "max_components": 3,
        "required_components": (),
    }

    class Optimizer:
        composition_sites = {"composition": config}
        composition_transformers_ = {"composition": transformer}
        composition_search_spaces_ = {
            "composition": CompositionSearchSpace(
                components=["Fe", "Co", "Ni"],
                total=1.0,
            )
        }
        composition_element_constraints: list[dict[str, object]] = []

        @staticmethod
        def transform_compositions(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

        @staticmethod
        def inverse_compositions(
            frame: pd.DataFrame,
            **_: object,
        ) -> pd.DataFrame:
            return _fraction_frame(frame)

        @staticmethod
        def predict(frame: pd.DataFrame) -> pd.DataFrame:
            fractions = _fraction_frame(frame)
            return pd.DataFrame(
                {
                    "property_mean": fractions["formula__fraction__Fe"] * 10.0,
                    "property_variance": 0.01,
                }
            )

    session = SimpleNamespace(
        tabular_optimizer=Optimizer(),
        data=data,
        encoded_targets=data[["property"]].copy(),
        feature_columns=["formula", "temperature"],
        target_columns=["property"],
        target_metadata={"property": {"internal_task": "regression"}},
    )
    request = SimpleNamespace(
        feature_importance=SimpleNamespace(
            enabled=True,
            config=SimpleNamespace(
                n_repeats=3,
                random_state=0,
                scoring="rmse",
                scoring_direction="auto",
                normalize_importance=False,
                clip_negative_importance=False,
                return_per_repeat_values=True,
                batch_size=None,
                unsupported_method_policy="warn",
                error_policy="raise",
            ),
        )
    )
    result = {
        "feature_importance_source": "training",
        "feature_importance_warnings": [],
        "metadata": {},
    }

    attach_composition_feature_importance(result, request, session)

    payload = result["composition_feature_importance"]
    assert payload["mode"] == "proportional"
    assert payload["overall"][0]["feature"] == "組成全体"
    assert {row["feature"] for row in payload["elements"]} == {
        "Fe",
        "Co",
        "Ni",
    }
    assert all(row["metric_name"] == "rmse" for row in payload["elements"])
    assert any(float(row["mean"]) > 0.0 for row in payload["elements"])
