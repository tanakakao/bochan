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
from .chgnet_residual import CHGNetDirectEnergyPredictor, CHGNetResidualGPModel
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
from .m3gnet_residual import M3GNetDirectPredictor, M3GNetResidualGPModel
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
    "CHGNetResidualGPModel",
    "M3GNetDKLModel",
    "M3GNetDirectPredictor",
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
    "M3GNetResidualGPModel",
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
    "MACEResidualGPModel",
    "MACEStressResidualGPModel",
]
