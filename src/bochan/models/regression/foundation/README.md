# PFN regression surrogate

`PFNRegressorModel` integrates the public PFNs4BO pretrained models as an
in-context single-output regression surrogate.

## Version-1 scope

- continuous inputs only
- single-output regression only
- public PFNs4BO checkpoints (`hebo_plus`, `bnn`, `hebo_plus_userprior`)
- differentiable q=1 native EI / PI / UCB
- marginal predictive mean, variance, and bucket probabilities
- no task-specific neural-network fitting

The PFN consumes the observed `(train_X, train_Y)` as context on every forward
pass. Candidate inputs are normalized to `[0, 1]` using the configured search
bounds. Targets are affine-standardized by default and mapped back to the raw
scale for predictive moments and EI.

## Optional upstream code package

The official PFNs4BO checkpoints are full PyTorch pickles and therefore require
the original `pfns4bo` Python modules when they are deserialized. The upstream
`pfns4bo==0.1.5` package declares an old `scikit-learn<1.2` constraint that is
not compatible with bochan's current dependency stack, so PFNs4BO is deliberately
not a hard bochan dependency.

Install only the upstream Python code when public checkpoints are used:

```bash
pip install pfns4bo==0.1.5 --no-deps
```

bochan downloads the selected official compressed checkpoint itself and stores
it under `~/.cache/bochan/pfns4bo` (or `$BOCHAN_CACHE_DIR/pfns4bo`). A local
checkpoint can instead be supplied with `model_path=`.

## High-level model construction

Always pass the real search-space bounds when possible. Inferring bounds from
observed points is supported only when every observed dimension already has a
non-zero range.

```python
import torch

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bounds = torch.tensor([
    [0.0, 700.0],
    [1.0, 1200.0],
])

optimizer = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="pfn",
        model_kwargs={"bounds": bounds, "model_name": "hebo_plus"},
    ),
    fit_config=FitConfig(),
    bounds=bounds,
)
optimizer.fit(train_X, train_Y)
```

`fit()` loads and freezes the pretrained transformer. It does not optimize PFN
parameters on the current task.

## Native acquisitions

PFNs4BO represents the predictive distribution with a bar distribution rather
than a Gaussian posterior. Version 1 therefore computes EI, PI, and UCB directly
from that distribution instead of approximating it as Gaussian.

```python
from bochan.acquisition.regression.pfn import (
    PFNExpectedImprovement,
    PFNProbabilityOfImprovement,
    PFNUpperConfidenceBound,
)
from bochan.api import AcquisitionConfig, OptimizeConfig

candidates, value = optimizer.candidate(
    AcquisitionConfig(
        name="pfn_ei",
        acqf_cls=PFNExpectedImprovement,
    ),
    OptimizeConfig(q=1),
)
```

`PFNPosterior` exposes `mean`, `variance`, and `probabilities` for prediction and
inspection. It intentionally does not expose a fake joint `rsample()` contract:
PFNs4BO v1 in bochan does not yet model a reparameterized joint posterior across
multiple candidate points. Native acquisitions consequently require `q=1`.

Mixed inputs, multi-output regression, classification/ordinal targets, joint q>1
sampling, and custom/materials priors are intentionally deferred to later phases.
