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
from .alignn_ff_relax_acquisition import ALIGNNFFRelaxationAcquisitionSelector
from .alignn_ff_relax_rank import ALIGNNFFRelaxationRanker
from .alignn_ff_relaxation import ALIGNNFFStructureRelaxer, relax_structure_alignn_ff
from .alignn_ff_residual import (
    ALIGNNFFDirectEnergyPredictor,
    ALIGNNFFDirectForcePredictor,
    ALIGNNFFDirectStressPredictor,
    ALIGNNFFEnergyResidualGPModel,
    ALIGNNFFForceResidualGPModel,
    ALIGNNFFStressResidualGPModel,
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
from .factory import (
    SUPPORTED_MLIP_BACKENDS,
    MaterialMLIPBackend,
    create_relaxation_acquisition_selector,
    create_relaxation_ranker,
    create_structure_relaxer,
    normalize_material_backend,
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
from .model_factory import (
    SUPPORTED_MATERIAL_MODEL_MODES,
    MaterialModelMode,
    MaterialModelSpec,
    create_material_model,
    normalize_material_model_mode,
)
from .multitask_residual import (
    CHGNetMixedMultiTaskResidualGPModel,
    CHGNetMultiTaskResidualGPModel,
    M3GNetMixedMultiTaskResidualGPModel,
    M3GNetMultiTaskResidualGPModel,
    MACEMixedMultiTaskResidualGPModel,
    MACEMultiTaskResidualGPModel,
)
from .property_factory import (
    SUPPORTED_MATERIAL_QUANTITIES,
    MaterialQuantity,
    create_direct_material_predictor,
    create_material_residual_gp,
    normalize_material_quantity,
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
from .workflow_factory import (
    SUPPORTED_MATERIAL_WORKFLOW_MODES,
    MaterialWorkflow,
    MaterialWorkflowMode,
    MaterialWorkflowSpec,
    create_material_workflow,
    normalize_material_workflow_mode,
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
    "ALIGNNFFDirectEnergyPredictor",
    "ALIGNNFFDirectForcePredictor",
    "ALIGNNFFDirectStressPredictor",
    "ALIGNNFFEnergyResidualGPModel",
    "ALIGNNFFForceResidualGPModel",
    "ALIGNNFFRelaxationAcquisitionSelector",
    "ALIGNNFFRelaxationRanker",
    "ALIGNNFFStressResidualGPModel",
    "ALIGNNFFStructureRelaxer",
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
    "MaterialMLIPBackend",
    "MaterialModelMode",
    "MaterialModelSpec",
    "MaterialQuantity",
    "MaterialRelaxationAcquisitionSelector",
    "MaterialRelaxationRanker",
    "MaterialWorkflow",
    "MaterialWorkflowMode",
    "MaterialWorkflowSpec",
    "OptimizerName",
    "RankingCriterion",
    "RankingDirection",
    "RelaxedStructureAcquisitionCandidate",
    "RelaxedStructureAcquisitionResult",
    "RelaxedStructureRank",
    "RelaxedStructureRankingResult",
    "SUPPORTED_MATERIAL_MODEL_MODES",
    "SUPPORTED_MATERIAL_QUANTITIES",
    "SUPPORTED_MATERIAL_WORKFLOW_MODES",
    "SUPPORTED_MLIP_BACKENDS",
    "create_direct_material_predictor",
    "create_material_model",
    "create_material_residual_gp",
    "create_material_workflow",
    "create_relaxation_acquisition_selector",
    "create_relaxation_ranker",
    "create_structure_relaxer",
    "normalize_material_backend",
    "normalize_material_model_mode",
    "normalize_material_quantity",
    "normalize_material_workflow_mode",
    "relax_structure_alignn_ff",
    "relax_structure_chgnet",
    "relax_structure_m3gnet",
    "relax_structure_mace",
]
