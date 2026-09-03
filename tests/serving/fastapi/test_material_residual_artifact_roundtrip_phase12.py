from __future__ import annotations

import pandas as pd
import torch

from bochan.models.regression.gaussian.materials.common import (
    MaterialBaselineSpec,
    MaterialPropertyContract,
    MultipleBaselineModelListGP,
    assert_residual_posterior_equivalent,
    validate_residual_production_model,
)
from bochan.serving.fastapi.services.tabular_artifacts import (
    load_tabular_artifact,
    save_tabular_artifact,
)
from bochan.serving.fastapi.stores import FileOptimizerStore
from bochan.tabular import TabularBayesianOptimizer
from tests._material_residual_hardening_utils import resolve_toy_material_model

_STRUCTURE = {
    "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    "coords": [[0.0, 0.0, 0.0]],
    "elements": ["Si"],
}


def _optimizer(monkeypatch) -> TabularBayesianOptimizer:
    import bochan.tabular.structure.material_multi_baseline as routing

    monkeypatch.setattr(routing, "_resolve_model_class", resolve_toy_material_model)
    energy = MaterialBaselineSpec(
        family="mace",
        output_name="energy",
        property=MaterialPropertyContract("energy", "eV", "total"),
        model_name="medium-mpa-0",
    )
    gap = MaterialBaselineSpec(
        family="m3gnet",
        output_name="band_gap",
        property=MaterialPropertyContract("band_gap", "eV", "intensive"),
        model_name="M3GNet-PES-MatPES-PBE-2025.2",
    )
    optimizer = TabularBayesianOptimizer(
        model_config={
            "task_type": "multi_objective",
            "model_type": "material_multi_baseline_residual_gp",
            "model_kwargs": {
                "baseline_routes": [
                    {"spec": energy, "model_kwargs": {}},
                    {"spec": gap, "model_kwargs": {}},
                ],
                "ordinary_family": "chgnet",
                "ordinary_model_kwargs": {"model_name": "0.3.0"},
            },
        },
        fit_config={"skip_fit": True},
        input_cols=["structure", "temperature"],
        target_cols=["energy", "band_gap", "strength"],
        bounds={"temperature": [300.0, 900.0]},
        structure_col="structure",
        structure_catalog={"s0": _STRUCTURE, "s1": _STRUCTURE},
    )
    frame = pd.DataFrame(
        [
            {"structure": "s0", "temperature": 350.0, "energy": -1.0, "band_gap": 1.0, "strength": 2.0},
            {"structure": "s1", "temperature": 500.0, "energy": -0.8, "band_gap": 1.2, "strength": 2.2},
            {"structure": "s0", "temperature": 650.0, "energy": -0.7, "band_gap": 1.4, "strength": 2.4},
            {"structure": "s1", "temperature": 800.0, "energy": -0.5, "band_gap": 1.5, "strength": 2.5},
        ]
    )
    optimizer.fit(frame)
    return optimizer


def test_full_bochan_artifact_roundtrip_preserves_multiple_baselines(monkeypatch, tmp_path) -> None:
    optimizer = _optimizer(monkeypatch)
    model = optimizer.bo.bundle.model
    assert isinstance(model, MultipleBaselineModelListGP)
    query = pd.DataFrame(
        [
            {"structure": "s0", "temperature": 425.0},
            {"structure": "s1", "temperature": 725.0},
        ]
    )
    X, _ = optimizer._prediction_input(query)  # noqa: SLF001
    before_report = validate_residual_production_model(model, X, expected_num_outputs=3)

    store = FileOptimizerStore(tmp_path)
    path = save_tabular_artifact(
        optimizer,
        store,
        filename="multi-baseline.bochan.pt",
        default_stem="unused",
        overwrite=False,
        metadata={"surface": "phase12-hardening"},
    )
    assert path.exists()

    restored, restored_path = load_tabular_artifact(
        store,
        filename="multi-baseline.bochan.pt",
        map_location="cpu",
        trust_pickle=True,
    )
    assert restored_path == path
    restored_model = restored.bo.bundle.model
    assert isinstance(restored_model, MultipleBaselineModelListGP)
    after_report = validate_residual_production_model(restored_model, X, expected_num_outputs=3)

    assert before_report.baseline_output_indices == (0, 1)
    assert after_report.baseline_output_indices == (0, 1)
    assert restored_model.output_names == ("energy", "band_gap", "strength")
    assert [item["family"] for item in restored_model.baseline_metadata] == ["mace", "m3gnet"]
    assert_residual_posterior_equivalent(model, restored_model, X)

    before = optimizer.predict(query, return_type="mean")
    after = restored.predict(query, return_type="mean")
    torch.testing.assert_close(after, before)
