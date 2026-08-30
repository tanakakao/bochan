from __future__ import annotations

import pandas as pd
import pytest
import torch

from bochan.composition import parse_formula
from bochan.tabular import TabularBayesianOptimizer


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "formula": [
                "Al0.40Ti0.25V0.20Nb0.15",
                "Al0.35Ti0.30V0.15Nb0.20",
                "Al0.30Ti0.20V0.30Nb0.20",
                "Al0.25Ti0.25V0.20Nb0.30",
                "Al0.45Ti0.15V0.25Nb0.15",
                "Al0.32Ti0.28V0.22Nb0.18",
                "Al0.38Ti0.18V0.18Nb0.26",
                "Al0.28Ti0.22V0.28Nb0.22",
            ],
            "temperature": [
                850.0,
                900.0,
                950.0,
                1000.0,
                1050.0,
                1100.0,
                1150.0,
                1200.0,
            ],
            "property": [0.8, 1.0, 1.15, 1.25, 1.35, 1.5, 1.65, 1.8],
        }
    )


def _optimizer(representation: str = "ilr") -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        fit_config={"maxiter": 32},
        input_cols=["formula", "temperature"],
        target_cols="property",
        composition_sites={
            "alloy": {
                "column": "formula",
                "elements": ["Al", "Ti", "V", "Nb"],
                "representation": representation,
                "reference_element": "Nb" if representation == "alr" else None,
                "pseudocount": 1e-8,
                "bounds": {
                    "Al": [0.05, 0.8],
                    "Ti": [0.0, 0.8],
                    "V": [0.0, 0.8],
                    "Nb": [0.0, 0.8],
                },
                "min_components": 3,
                "max_components": 3,
                "required_components": ["Al"],
                "support_selection": "best_subset",
                "best_subset_strategy": "exact",
                "best_subset_max_combinations": 20,
            }
        },
        bounds={"temperature": [800.0, 1250.0]},
    )


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_tabular_candidate_optimizes_logratio_support_in_raw_fraction_space(
    representation: str,
) -> None:
    optimizer = _optimizer(representation).fit(_frame())

    result = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=2,
        raw_samples=16,
        optimizer_kwargs={
            "best_subset_strategy": "exact",
            "options": {"maxiter": 12, "batch_limit": 2},
        },
        return_result=True,
    )

    assert result.candidates.shape[-1] == len(optimizer.dataset.feature_names)
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(torch.as_tensor(result.acq_value)).all()

    raw = result.raw_composition_candidates
    bridge = result.composition_raw_bridge
    fractions = raw[..., bridge.fraction_slice]
    assert fractions.shape[-1] == 4
    assert int((fractions > 1e-8).sum().item()) == 3
    assert fractions[..., 0].item() > 0.0  # required Al
    assert fractions.sum().item() == pytest.approx(1.0, abs=1e-7)

    model_frame = optimizer.candidates_to_dataframe(result.candidates)
    restored = optimizer.inverse_compositions(model_frame, repair=True)
    parsed = parse_formula(restored.loc[0, "formula"])
    assert "Al" in parsed
    assert len(parsed) == 3
    assert 800.0 <= float(restored.loc[0, "temperature"]) <= 1250.0
