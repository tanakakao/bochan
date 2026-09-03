"""Cross-cutting architecture contract for material-aware Gaussian models."""

from __future__ import annotations

import importlib
import pickle

import torch
from torch import Tensor, nn

from bochan.composition.encoders.base import MaterialEncoder as LegacyMaterialEncoder
from bochan.composition.encoders.fusion import MaterialProcessFusion as LegacyMaterialProcessFusion
from bochan.models.regression.gaussian.materials import (
    LEGACY_MATERIAL_MODEL_PATHS,
    MaterialEncoder,
    MaterialMultiTaskSpec,
    MaterialProcessFusion,
    MaterialSurrogateSpec,
    PretrainedMaterialCapabilities,
    PretrainedMaterialSpec,
    compute_material_residual_targets,
    get_material_family,
    list_material_families,
    resolve_material_latent_dim,
    resolve_mixed_process_layout,
)
from bochan.models.regression.gaussian.materials.common.residual import DirectMaterialPredictor


class _FeatureExtractor(nn.Module):
    output_dim = 3

    def forward(self, X: Tensor) -> Tensor:
        return X[..., : self.output_dim]


class _Predictor(DirectMaterialPredictor):
    @property
    def output_dim(self) -> int:
        return 2

    def forward(self, X: Tensor) -> Tensor:
        return torch.stack((X[..., 0], X[..., 0] + 1.0), dim=-1)


def _resolve(path: str):
    module_name, attribute = path.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def test_phase13_public_contracts_compose_without_backend_specific_setup() -> None:
    """Core neutral contracts should compose through the public materials API."""

    assert MaterialEncoder is LegacyMaterialEncoder
    assert MaterialProcessFusion is LegacyMaterialProcessFusion

    layout = resolve_mixed_process_layout(
        input_dim=5,
        cat_dims=(4,),
        material_dims=(0,),
    )
    assert layout.material_dims == (0,)
    assert layout.numeric_process_dims == (1, 2, 3)
    assert layout.categorical_dims == (4,)

    multitask = MaterialMultiTaskSpec(mode="correlated", num_tasks=2)
    assert multitask.num_tasks == 2

    surrogate = MaterialSurrogateSpec(kind="dkl", mixed=True, latent_dim=3)
    assert surrogate.kind == "dkl"
    assert surrogate.mixed is True
    assert resolve_material_latent_dim(_FeatureExtractor(), surrogate.latent_dim) == 3


def test_phase13_partial_observations_survive_residual_target_construction() -> None:
    """Residual preprocessing must preserve missing observations for the GP layer."""

    train_X = torch.tensor([[1.0], [2.0], [3.0]])
    train_Y = torch.tensor([[2.0, 3.0], [float("nan"), 4.0], [5.0, 6.0]])

    residual = compute_material_residual_targets(train_X, train_Y, _Predictor())

    assert torch.isnan(residual[1, 0])
    assert torch.allclose(residual[[0, 2]], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_phase13_pretrained_capability_semantics_match_residual_boundary() -> None:
    """Residual-GP support must require a verified direct prediction capability."""

    representation_only = PretrainedMaterialSpec(
        family="example",
        domain="structure",
        capabilities=PretrainedMaterialCapabilities(
            representation=True,
            direct_prediction=False,
            loading_modes=frozenset({"injected"}),
            fine_tuning=True,
            residual_gp=False,
        ),
    )
    assert representation_only.supports_gp is True
    assert representation_only.supports_dkl is True
    assert representation_only.supports_residual_gp is False

    residual_ready = PretrainedMaterialSpec(
        family="example-residual",
        domain="structure",
        capabilities=PretrainedMaterialCapabilities(
            representation=True,
            direct_prediction=True,
            loading_modes=frozenset({"injected"}),
            fine_tuning=True,
            residual_gp=True,
        ),
    )
    assert residual_ready.supports_residual_gp is True


def test_phase13_registry_matches_current_material_family_matrix() -> None:
    """Registry metadata should describe the concrete family matrix conservatively."""

    assert list_material_families() == (
        "crabnet",
        "roost",
        "alignn",
        "chgnet",
        "m3gnet",
        "mace",
    )
    assert list_material_families(domain="composition") == ("crabnet", "roost")
    assert list_material_families(domain="structure") == (
        "alignn",
        "chgnet",
        "m3gnet",
        "mace",
    )

    full_matrix = {
        "gp",
        "dkl",
        "mixed_gp",
        "mixed_dkl",
        "multitask_gp",
        "multitask_dkl",
        "mixed_multitask_gp",
        "mixed_multitask_dkl",
    }
    for family in ("crabnet", "alignn", "chgnet", "m3gnet", "mace"):
        registration = get_material_family(family)
        assert registration.variants == frozenset(full_matrix)
        assert registration.pretrained.capabilities.direct_prediction is False
        assert registration.pretrained.capabilities.residual_gp is False

    assert get_material_family("roost").variants == frozenset({"gp", "dkl"})


def test_phase13_registry_classes_and_legacy_paths_share_identity() -> None:
    """Canonical, registry, and historical imports must keep identical class objects."""

    compatibility_by_class = {
        item.canonical.rsplit(":", 1)[1]: item
        for item in LEGACY_MATERIAL_MODEL_PATHS
    }

    for family in list_material_families():
        registration = get_material_family(family)
        for variant in ("gp", "dkl"):
            registered_class = registration.resolve_model_class(variant)
            compatibility = compatibility_by_class[registered_class.__name__]
            canonical_class = _resolve(compatibility.canonical)
            legacy_class = _resolve(compatibility.legacy)

            assert registered_class is canonical_class is legacy_class
            assert registered_class.__module__.startswith(
                "bochan.models.regression.gaussian.deep."
            )
            assert pickle.loads(pickle.dumps(registered_class)) is registered_class
