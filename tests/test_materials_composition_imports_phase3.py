"""Phase 3 import-contract tests for composition material models."""

from bochan.models.regression.gaussian.deep.crabnet import (
    CrabNetDKLModel as LegacyCrabNetDKLModel,
)
from bochan.models.regression.gaussian.deep.crabnet import (
    CrabNetGPModel as LegacyCrabNetGPModel,
)
from bochan.models.regression.gaussian.deep.crabnet_mixed import (
    CrabNetMixedGPModel as LegacyCrabNetMixedGPModel,
)
from bochan.models.regression.gaussian.deep.crabnet_mixed_dkl import (
    CrabNetMixedDKLModel as LegacyCrabNetMixedDKLModel,
)
from bochan.models.regression.gaussian.deep.crabnet_multitask import (
    CrabNetMixedMultiTaskDKLModel as LegacyCrabNetMixedMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.crabnet_multitask import (
    CrabNetMixedMultiTaskGPModel as LegacyCrabNetMixedMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.crabnet_multitask import (
    CrabNetMultiTaskDKLModel as LegacyCrabNetMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.crabnet_multitask import (
    CrabNetMultiTaskGPModel as LegacyCrabNetMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.roost import (
    RoostDKLModel as LegacyRoostDKLModel,
)
from bochan.models.regression.gaussian.deep.roost import (
    RoostGPModel as LegacyRoostGPModel,
)
from bochan.models.regression.gaussian.materials.composition import (
    CrabNetDKLModel,
    CrabNetGPModel,
    CrabNetMixedDKLModel,
    CrabNetMixedGPModel,
    CrabNetMixedMultiTaskDKLModel,
    CrabNetMixedMultiTaskGPModel,
    CrabNetMultiTaskDKLModel,
    CrabNetMultiTaskGPModel,
    RoostDKLModel,
    RoostGPModel,
)


def test_crabnet_canonical_imports_preserve_class_identity() -> None:
    assert CrabNetGPModel is LegacyCrabNetGPModel
    assert CrabNetDKLModel is LegacyCrabNetDKLModel
    assert CrabNetMixedGPModel is LegacyCrabNetMixedGPModel
    assert CrabNetMixedDKLModel is LegacyCrabNetMixedDKLModel
    assert CrabNetMultiTaskGPModel is LegacyCrabNetMultiTaskGPModel
    assert CrabNetMultiTaskDKLModel is LegacyCrabNetMultiTaskDKLModel
    assert CrabNetMixedMultiTaskGPModel is LegacyCrabNetMixedMultiTaskGPModel
    assert CrabNetMixedMultiTaskDKLModel is LegacyCrabNetMixedMultiTaskDKLModel


def test_roost_canonical_imports_preserve_class_identity() -> None:
    assert RoostGPModel is LegacyRoostGPModel
    assert RoostDKLModel is LegacyRoostDKLModel
