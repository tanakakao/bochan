from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

from bochan.serving.webapp.composition.support import (
    repair_composition_candidate_result,
)
from bochan.tabular.composition.constraints import CompositionElementConstraintProjector
from bochan.tabular.config import TabularDataConfig


class _FakeAcquisition:
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., 0].sum(dim=-1)


class _FractionOptimizer:
    feature_names = [
        "formula__fraction__Fe",
        "formula__fraction__Co",
        "temperature",
    ]

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(feature_names=list(self.feature_names))
        self.data_config = TabularDataConfig(
            input_cols=list(self.feature_names),
            bounds={
                "formula__fraction__Fe": [0.0, 1.0],
                "formula__fraction__Co": [0.0, 1.0],
                "temperature": [800.0, 1000.0],
            },
        )
        self.composition = SimpleNamespace(
            transformers={
                "composition": SimpleNamespace(
                    feature_names_=tuple(self.feature_names[:2])
                )
            }
        )

    def candidate(self, *_: Any, **__: Any) -> Any:
        return SimpleNamespace(
            candidates=torch.tensor(
                [[0.4, 0.6, 900.0], [0.7, 0.3, 950.0]],
                dtype=torch.double,
            ),
            acq_value=torch.zeros(2, dtype=torch.double),
            acqf=_FakeAcquisition(),
        )

    def candidates_to_dataframe(self, candidates: torch.Tensor) -> pd.DataFrame:
        return pd.DataFrame(
            candidates.detach().cpu().numpy(),
            columns=self.feature_names,
        )

    def inverse_compositions(
        self,
        candidates: pd.DataFrame,
        **_: Any,
    ) -> pd.DataFrame:
        formulas = [
            f"Fe{row[0]:.6f}Co{row[1]:.6f}"
            for row in candidates.loc[:, self.feature_names[:2]].to_numpy()
        ]
        return pd.DataFrame(
            {
                "formula": formulas,
                self.feature_names[0]: candidates[self.feature_names[0]],
                self.feature_names[1]: candidates[self.feature_names[1]],
                "temperature": candidates["temperature"],
            }
        )

    def transform_compositions(self, frame: pd.DataFrame) -> pd.DataFrame:
        fractions = frame["formula"].str.extract(
            r"Fe(?P<Fe>[0-9.]+)Co(?P<Co>[0-9.]+)"
        ).astype(float)
        fractions.columns = self.feature_names[:2]
        base = frame.drop(columns=["formula"])
        return pd.concat([base, fractions], axis=1)


def test_fraction_candidate_repair_preserves_model_dimension() -> None:
    optimizer = _FractionOptimizer()

    result = repair_composition_candidate_result(
        optimizer,
        optimizer.candidate(return_result=True),
    )

    assert result.candidates.shape == (2, 3)
    assert torch.allclose(
        result.candidates,
        torch.tensor(
            [[0.4, 0.6, 900.0], [0.7, 0.3, 950.0]],
            dtype=torch.double,
        ),
    )
    assert result.acq_value.shape == (2,)


def test_constraint_projector_exposes_native_row_values() -> None:
    projector = CompositionElementConstraintProjector(
        composition_sites={
            "composition": {
                "elements": ("Fe", "Co"),
                "total": 1.0,
                "input_kind": "formula",
            }
        },
        composition_element_constraints=(),
        composition_transformers={
            "composition": SimpleNamespace(prefix="formula")
        },
        max_supports=16,
    )
    restored = pd.DataFrame(
        {
            "formula__fraction__Fe": [0.4],
            "formula__fraction__Co": [0.6],
        }
    )

    raw, totals = projector.row_native_values(restored, 0)

    assert raw == {
        ("composition", "Fe"): 0.4,
        ("composition", "Co"): 0.6,
    }
    assert totals == {"composition": 1.0}


def test_web_composition_support_uses_canonical_component_state() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "src/bochan/serving/webapp/composition/support.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {
        "_make_element_constraint_projector",
        "_require_fitted",
        "_row_native_values",
        "composition_element_constraints",
        "composition_sites",
        "composition_transformers_",
    }
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden
    }

    assert not used, (
        "Web composition support reaches stale or private composition state: "
        f"{sorted(used)!r}."
    )
