"""Phase 4 import-contract tests for structure material models."""

from bochan.models.regression.gaussian.deep.alignn import ALIGNNDKLModel as OldALIGNNDKLModel
from bochan.models.regression.gaussian.deep.alignn import ALIGNNGPModel as OldALIGNNGPModel
from bochan.models.regression.gaussian.deep.alignn_mixed import (
    ALIGNNMixedDKLModel as OldALIGNNMixedDKLModel,
)
from bochan.models.regression.gaussian.deep.alignn_mixed import (
    ALIGNNMixedGPModel as OldALIGNNMixedGPModel,
)
from bochan.models.regression.gaussian.deep.alignn_multitask import (
    ALIGNNMixedMultiTaskDKLModel as OldALIGNNMixedMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.alignn_multitask import (
    ALIGNNMixedMultiTaskGPModel as OldALIGNNMixedMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.alignn_multitask import (
    ALIGNNMultiTaskDKLModel as OldALIGNNMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.alignn_multitask import (
    ALIGNNMultiTaskGPModel as OldALIGNNMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.chgnet import CHGNetDKLModel as OldCHGNetDKLModel
from bochan.models.regression.gaussian.deep.chgnet import CHGNetGPModel as OldCHGNetGPModel
from bochan.models.regression.gaussian.deep.chgnet import (
    CHGNetMixedDKLModel as OldCHGNetMixedDKLModel,
)
from bochan.models.regression.gaussian.deep.chgnet import CHGNetMixedGPModel as OldCHGNetMixedGPModel
from bochan.models.regression.gaussian.deep.chgnet_multitask import (
    CHGNetMixedMultiTaskDKLModel as OldCHGNetMixedMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.chgnet_multitask import (
    CHGNetMixedMultiTaskGPModel as OldCHGNetMixedMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.chgnet_multitask import (
    CHGNetMultiTaskDKLModel as OldCHGNetMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.chgnet_multitask import (
    CHGNetMultiTaskGPModel as OldCHGNetMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.m3gnet import M3GNetDKLModel as OldM3GNetDKLModel
from bochan.models.regression.gaussian.deep.m3gnet import M3GNetGPModel as OldM3GNetGPModel
from bochan.models.regression.gaussian.deep.m3gnet import (
    M3GNetMixedDKLModel as OldM3GNetMixedDKLModel,
)
from bochan.models.regression.gaussian.deep.m3gnet import M3GNetMixedGPModel as OldM3GNetMixedGPModel
from bochan.models.regression.gaussian.deep.m3gnet_multitask import (
    M3GNetMixedMultiTaskDKLModel as OldM3GNetMixedMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.m3gnet_multitask import (
    M3GNetMixedMultiTaskGPModel as OldM3GNetMixedMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.m3gnet_multitask import (
    M3GNetMultiTaskDKLModel as OldM3GNetMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.m3gnet_multitask import (
    M3GNetMultiTaskGPModel as OldM3GNetMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.mace import MACEDKLModel as OldMACEDKLModel
from bochan.models.regression.gaussian.deep.mace import MACEGPModel as OldMACEGPModel
from bochan.models.regression.gaussian.deep.mace_mixed import MACEMixedDKLModel as OldMACEMixedDKLModel
from bochan.models.regression.gaussian.deep.mace_mixed import MACEMixedGPModel as OldMACEMixedGPModel
from bochan.models.regression.gaussian.deep.mace_multitask import (
    MACEMixedMultiTaskDKLModel as OldMACEMixedMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.mace_multitask import (
    MACEMixedMultiTaskGPModel as OldMACEMixedMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.mace_multitask import (
    MACEMultiTaskDKLModel as OldMACEMultiTaskDKLModel,
)
from bochan.models.regression.gaussian.deep.mace_multitask import (
    MACEMultiTaskGPModel as OldMACEMultiTaskGPModel,
)
from bochan.models.regression.gaussian.materials.structure import (
    ALIGNNDKLModel,
    ALIGNNGPModel,
    ALIGNNMixedDKLModel,
    ALIGNNMixedGPModel,
    ALIGNNMixedMultiTaskDKLModel,
    ALIGNNMixedMultiTaskGPModel,
    ALIGNNMultiTaskDKLModel,
    ALIGNNMultiTaskGPModel,
    CHGNetDKLModel,
    CHGNetGPModel,
    CHGNetMixedDKLModel,
    CHGNetMixedGPModel,
    CHGNetMixedMultiTaskDKLModel,
    CHGNetMixedMultiTaskGPModel,
    CHGNetMultiTaskDKLModel,
    CHGNetMultiTaskGPModel,
    M3GNetDKLModel,
    M3GNetGPModel,
    M3GNetMixedDKLModel,
    M3GNetMixedGPModel,
    M3GNetMixedMultiTaskDKLModel,
    M3GNetMixedMultiTaskGPModel,
    M3GNetMultiTaskDKLModel,
    M3GNetMultiTaskGPModel,
    MACEDKLModel,
    MACEGPModel,
    MACEMixedDKLModel,
    MACEMixedGPModel,
    MACEMixedMultiTaskDKLModel,
    MACEMixedMultiTaskGPModel,
    MACEMultiTaskDKLModel,
    MACEMultiTaskGPModel,
)


def test_alignn_old_and_new_imports_share_class_identity() -> None:
    assert ALIGNNGPModel is OldALIGNNGPModel
    assert ALIGNNDKLModel is OldALIGNNDKLModel
    assert ALIGNNMixedGPModel is OldALIGNNMixedGPModel
    assert ALIGNNMixedDKLModel is OldALIGNNMixedDKLModel
    assert ALIGNNMultiTaskGPModel is OldALIGNNMultiTaskGPModel
    assert ALIGNNMultiTaskDKLModel is OldALIGNNMultiTaskDKLModel
    assert ALIGNNMixedMultiTaskGPModel is OldALIGNNMixedMultiTaskGPModel
    assert ALIGNNMixedMultiTaskDKLModel is OldALIGNNMixedMultiTaskDKLModel


def test_chgnet_old_and_new_imports_share_class_identity() -> None:
    assert CHGNetGPModel is OldCHGNetGPModel
    assert CHGNetDKLModel is OldCHGNetDKLModel
    assert CHGNetMixedGPModel is OldCHGNetMixedGPModel
    assert CHGNetMixedDKLModel is OldCHGNetMixedDKLModel
    assert CHGNetMultiTaskGPModel is OldCHGNetMultiTaskGPModel
    assert CHGNetMultiTaskDKLModel is OldCHGNetMultiTaskDKLModel
    assert CHGNetMixedMultiTaskGPModel is OldCHGNetMixedMultiTaskGPModel
    assert CHGNetMixedMultiTaskDKLModel is OldCHGNetMixedMultiTaskDKLModel


def test_m3gnet_old_and_new_imports_share_class_identity() -> None:
    assert M3GNetGPModel is OldM3GNetGPModel
    assert M3GNetDKLModel is OldM3GNetDKLModel
    assert M3GNetMixedGPModel is OldM3GNetMixedGPModel
    assert M3GNetMixedDKLModel is OldM3GNetMixedDKLModel
    assert M3GNetMultiTaskGPModel is OldM3GNetMultiTaskGPModel
    assert M3GNetMultiTaskDKLModel is OldM3GNetMultiTaskDKLModel
    assert M3GNetMixedMultiTaskGPModel is OldM3GNetMixedMultiTaskGPModel
    assert M3GNetMixedMultiTaskDKLModel is OldM3GNetMixedMultiTaskDKLModel


def test_mace_old_and_new_imports_share_class_identity() -> None:
    assert MACEGPModel is OldMACEGPModel
    assert MACEDKLModel is OldMACEDKLModel
    assert MACEMixedGPModel is OldMACEMixedGPModel
    assert MACEMixedDKLModel is OldMACEMixedDKLModel
    assert MACEMultiTaskGPModel is OldMACEMultiTaskGPModel
    assert MACEMultiTaskDKLModel is OldMACEMultiTaskDKLModel
    assert MACEMixedMultiTaskGPModel is OldMACEMixedMultiTaskGPModel
    assert MACEMixedMultiTaskDKLModel is OldMACEMixedMultiTaskDKLModel


def test_canonical_classes_keep_historical_module_paths_for_pickle_compatibility() -> None:
    assert ALIGNNGPModel.__module__ == "bochan.models.regression.gaussian.deep.alignn"
    assert CHGNetGPModel.__module__ == "bochan.models.regression.gaussian.deep.chgnet"
    assert M3GNetGPModel.__module__ == "bochan.models.regression.gaussian.deep.m3gnet"
    assert MACEGPModel.__module__ == "bochan.models.regression.gaussian.deep.mace"
