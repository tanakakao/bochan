"""Structure-model namespace for material-aware Gaussian models.

ALIGNN, CHGNet, M3GNet, and MACE are exposed from this canonical namespace.
Historical GP/DKL implementations remain under ``gaussian.deep`` for saved-model
compatibility; newly introduced residual models may live directly here.
"""

from .alignn import (
    ALIGNNDKLModel,
    ALIGNNGPModel,
    ALIGNNMixedDKLModel,
    ALIGNNMixedGPModel,
    ALIGNNMixedMultiTaskDKLModel,
    ALIGNNMixedMultiTaskGPModel,
    ALIGNNMultiTaskDKLModel,
    ALIGNNMultiTaskGPModel,
)
from .chgnet import (
    CHGNetDKLModel,
    CHGNetGPModel,
    CHGNetMixedDKLModel,
    CHGNetMixedGPModel,
    CHGNetMixedMultiTaskDKLModel,
    CHGNetMixedMultiTaskGPModel,
    CHGNetMultiTaskDKLModel,
    CHGNetMultiTaskGPModel,
)
from .chgnet_relax_acquisition import CHGNetRelaxationAcquisitionSelector
from .chgnet_relax_rank import CHGNetRelaxationRanker
from .chgnet_relaxation import CHGNetStructureRelaxer, relax_structure_chgnet
from .chgnet_residual import CHGNetDirectEnergyPredictor, CHGNetResidualGPModel
from .chgnet_tensor_residual import (
    CHGNetDirectForcePredictor,
    CHGNetDirectStressPredictor,
    CHGNetForceResidualGPModel,
    CHGNetStressResidualGPModel,
)
from .m3gnet import (
    M3GNetDKLModel,
    M3GNetGPModel,
    M3GNetMixedDKLModel,
    M3GNetMixedGPModel,
    M3GNetMixedMultiTaskDKLModel,
    M3GNetMixedMultiTaskGPModel,
    M3GNetMultiTaskDKLModel,
    M3GNetMultiTaskGPModel,
)
from .m3gnet_relax_acquisition import M3GNetRelaxationAcquisitionSelector
from .m3gnet_relax_rank import M3GNetRelaxationRanker
from .m3gnet_relaxation import M3GNetStructureRelaxer, relax_structure_m3gnet
from .m3gnet_residual import M3GNetDirectPredictor, M3GNetResidualGPModel
from .m3gnet_tensor_residual import (
    M3GNetDirectForcePredictor,
    M3GNetDirectStressPredictor,
    M3GNetForceResidualGPModel,
    M3GNetStressResidualGPModel,
)
from .mace import (
    MACEDKLModel,
    MACEGPModel,
    MACEMixedDKLModel,
    MACEMixedGPModel,
    MACEMixedMultiTaskDKLModel,
    MACEMixedMultiTaskGPModel,
    MACEMultiTaskDKLModel,
    MACEMultiTaskGPModel,
)
from .mace_relax_acquisition import MACERelaxationAcquisitionSelector
from .mace_relax_rank import MACERelaxationRanker
from .mace_relaxation import MACEStructureRelaxer, OptimizerName, relax_structure_mace
from .mace_residual import MACEDirectEnergyPredictor, MACEResidualGPModel
from .mace_tensor_residual import (
    MACEDirectForcePredictor,
    MACEDirectStressPredictor,
    MACEForceResidualGPModel,
    MACEStressResidualGPModel,
)
from .mixed_residual import (
    CHGNetMixedResidualGPModel,
    M3GNetMixedResidualGPModel,
    MACEMixedResidualGPModel,
)
from .multitask_residual import (
    CHGNetMixedMultiTaskResidualGPModel,
    CHGNetMultiTaskResidualGPModel,
    M3GNetMixedMultiTaskResidualGPModel,
    M3GNetMultiTaskResidualGPModel,
    MACEMixedMultiTaskResidualGPModel,
    MACEMultiTaskResidualGPModel,
)
from .relax_acquisition import (
    MaterialRelaxationAcquisitionSelector,
    RelaxedStructureAcquisitionCandidate,
    RelaxedStructureAcquisitionResult,
)
from .relax_rank import (
    MaterialRelaxationRanker,
    RankingCriterion,
    RankingDirection,
    RelaxedStructureRank,
    RelaxedStructureRankingResult,
)

__all__ = [
    "ALIGNNDKLModel",
    "ALIGNNGPModel",
    "ALIGNNMixedDKLModel",
    "ALIGNNMixedGPModel",
    "ALIGNNMixedMultiTaskDKLModel",
    "ALIGNNMixedMultiTaskGPModel",
    "ALIGNNMultiTaskDKLModel",
    "ALIGNNMultiTaskGPModel",
    "CHGNetDKLModel",
    "CHGNetDirectEnergyPredictor",
    "CHGNetDirectForcePredictor",
    "CHGNetDirectStressPredictor",
    "CHGNetForceResidualGPModel",
    "CHGNetGPModel",
    "CHGNetMixedDKLModel",
    "CHGNetMixedGPModel",
    "CHGNetMixedMultiTaskDKLModel",
    "CHGNetMixedMultiTaskGPModel",
    "CHGNetMixedMultiTaskResidualGPModel",
    "CHGNetMixedResidualGPModel",
    "CHGNetMultiTaskDKLModel",
    "CHGNetMultiTaskGPModel",
    "CHGNetMultiTaskResidualGPModel",
    "CHGNetRelaxationAcquisitionSelector",
    "CHGNetRelaxationRanker",
    "CHGNetResidualGPModel",
    "CHGNetStressResidualGPModel",
    "CHGNetStructureRelaxer",
    "M3GNetDKLModel",
    "M3GNetDirectForcePredictor",
    "M3GNetDirectPredictor",
    "M3GNetDirectStressPredictor",
    "M3GNetForceResidualGPModel",
    "M3GNetGPModel",
    "M3GNetMixedDKLModel",
    "M3GNetMixedGPModel",
    "M3GNetMixedMultiTaskDKLModel",
    "M3GNetMixedMultiTaskGPModel",
    "M3GNetMixedMultiTaskResidualGPModel",
    "M3GNetMixedResidualGPModel",
    "M3GNetMultiTaskDKLModel",
    "M3GNetMultiTaskGPModel",
    "M3GNetMultiTaskResidualGPModel",
    "M3GNetRelaxationAcquisitionSelector",
    "M3GNetRelaxationRanker",
    "M3GNetResidualGPModel",
    "M3GNetStressResidualGPModel",
    "M3GNetStructureRelaxer",
    "MACEDKLModel",
    "MACEDirectEnergyPredictor",
    "MACEDirectForcePredictor",
    "MACEDirectStressPredictor",
    "MACEForceResidualGPModel",
    "MACEGPModel",
    "MACEMixedDKLModel",
    "MACEMixedGPModel",
    "MACEMixedMultiTaskDKLModel",
    "MACEMixedMultiTaskGPModel",
    "MACEMixedMultiTaskResidualGPModel",
    "MACEMixedResidualGPModel",
    "MACEMultiTaskDKLModel",
    "MACEMultiTaskGPModel",
    "MACEMultiTaskResidualGPModel",
    "MACERelaxationAcquisitionSelector",
    "MACERelaxationRanker",
    "MACEResidualGPModel",
    "MACEStressResidualGPModel",
    "MACEStructureRelaxer",
    "MaterialRelaxationAcquisitionSelector",
    "MaterialRelaxationRanker",
    "OptimizerName",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureAcquisitionCandidate",
    "RelaxedStructureAcquisitionResult",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
    "relax_structure_chgnet",
    "relax_structure_m3gnet",
    "relax_structure_mace",
]
