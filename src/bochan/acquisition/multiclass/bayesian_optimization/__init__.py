from bochan.acquisition._nparego_shape import (
    reduce_nparego_sample_and_q_to_tbatch,
)

from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassExpectedImprovement,
    qHeteroMultiOutputMulticlassNParEGO,
    qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassProbabilityOfFeasibility,
    qHeteroMultiOutputMulticlassProbabilityOfImprovement,
    qHeteroMultiOutputMulticlassUpperConfidenceBound,
)
from .hetero_single_output import (
    NoiseCombineType,
    NoiseWeightMode,
    qHeteroMulticlassExpectedImprovement,
    qHeteroMulticlassProbabilityOfFeasibility,
    qHeteroMulticlassProbabilityOfImprovement,
    qHeteroMulticlassUpperConfidenceBound,
)
from .input_perturbation_compat import (
    patch_multiclass_hypervolume_input_perturbation,
)
from .nehvi_baseline_compat import patch_multiclass_nehvi_baseline_input
from .nparego_input_perturbation_compat import (
    patch_multiclass_nparego_input_perturbation,
)
from .output_compat import apply_bayesian_optimization_output_compat

# Keep q=1 sequential optimization shape handling aligned across classification
# and ordinal NParEGO implementations.
_multi_output._reduce_sample_and_q_to_tbatch = (
    reduce_nparego_sample_and_q_to_tbatch
)

from .multi_output import (
    MulticlassTargetProbabilityObjective,
    OutputReductionType,
    compute_observed_multiclass_target_probability_values,
    compute_observed_multiclass_utility,
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassNParEGO,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
)
from .single_output import (
    compute_multiclass_target_probability_best_f,
    compute_multiclass_target_probability_values,
    qMulticlassExpectedImprovement,
    qMulticlassProbabilityOfFeasibility,
    qMulticlassProbabilityOfImprovement,
    qMulticlassUpperConfidenceBound,
)

# qNEHVI builds its baseline partitioning before qEHVI installs the automatic
# InputPerturbation objective adapter. Preserve raw X_baseline for an explicitly
# pre-wrapped objective so it can distinguish raw q from q * n_w.
patch_multiclass_nehvi_baseline_input(_multi_output)

# qNParEGO also calls its objective with X=None for baseline and candidate
# evaluation. Supply raw X so the adapter can distinguish q from q * n_w.
patch_multiclass_nparego_input_perturbation(_multi_output)

# A one-to-many InputPerturbation transform expands q to q*n_w. qEHVI subset
# enumeration is exponential in that effective q, so aggregate the built-in
# multiclass objective back to raw q before BoTorch enters the subset loop.
patch_multiclass_hypervolume_input_perturbation(
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
    default_objective_type=MulticlassTargetProbabilityObjective,
)

# DeepGP などで qEHVI の戻り値に extra sample / latent 次元が残る場合の出力整形 patch。
apply_bayesian_optimization_output_compat()

__all__ = [
    "NoiseCombineType",
    "NoiseWeightMode",
    "OutputReductionType",
    "apply_bayesian_optimization_output_compat",
    "compute_multiclass_target_probability_best_f",
    "compute_multiclass_target_probability_values",
    "compute_observed_multiclass_utility",
    "compute_observed_multiclass_target_probability_values",
    "qMulticlassProbabilityOfFeasibility",
    "qMulticlassExpectedImprovement",
    "qMulticlassProbabilityOfImprovement",
    "qMulticlassUpperConfidenceBound",
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassNParEGO",
    "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
    "qHeteroMulticlassProbabilityOfFeasibility",
    "qHeteroMulticlassExpectedImprovement",
    "qHeteroMulticlassProbabilityOfImprovement",
    "qHeteroMulticlassUpperConfidenceBound",
    "qHeteroMultiOutputMulticlassProbabilityOfFeasibility",
    "qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputMulticlassNParEGO",
    "qHeteroMultiOutputMulticlassExpectedImprovement",
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
    "qHeteroMultiOutputMulticlassUpperConfidenceBound",
]
