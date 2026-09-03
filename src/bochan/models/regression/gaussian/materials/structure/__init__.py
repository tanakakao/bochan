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
    "CHGNetMultiTaskDKLModel",
    "CHGNetMultiTaskGPModel",
    "CHGNetResidualGPModel",
    "M3GNetDKLModel",
    "M3GNetDirectPredictor",
    "M3GNetGPModel",
    "M3GNetMixedDKLModel",
    "M3GNetMixedGPModel",
    "M3GNetMixedMultiTaskDKLModel",
    "M3GNetMixedMultiTaskGPModel",
    "M3GNetMultiTaskDKLModel",
    "M3GNetMultiTaskGPModel",
    "M3GNetResidualGPModel",
    "MACEDKLModel",
    "MACEGPModel",
    "MACEMixedDKLModel",
    "MACEMixedGPModel",
    "MACEMixedMultiTaskDKLModel",
    "MACEMixedMultiTaskGPModel",
    "MACEMultiTaskDKLModel",
    "MACEMultiTaskGPModel",
]
