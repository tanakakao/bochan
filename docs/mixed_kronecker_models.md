# Mixed Kronecker multi-task models

The mixed Kronecker models use block-design data:

```text
train_X: [n, d]
train_Y: [n, m]
```

Categorical columns are specified with `cat_dims`. Input transforms must target
continuous columns only.

```python
from botorch.models.transforms.input import Normalize

cat_dims = [2]
input_transform = Normalize(d=4, indices=[0, 1, 3])
```

Available models:

```python
from bochan.models.regression.gaussian import GaussianMixedKroneckerMultiTaskGP
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalMixedGPModel
```

The default input kernel is:

```text
continuous + categorical + continuous * categorical
```

The multiclass model uses one mixed data kernel per class and returns task
covariance matrices with shape `[C, m, m]`. Binary, ordinal, and Gaussian models
return one task covariance matrix with shape `[m, m]`.

For candidate optimization, enumerate categorical assignments with
`optimize_acqf_mixed` or use bochan's mixed optimization backend.

```python
from botorch.optim import optimize_acqf_mixed

candidates, value = optimize_acqf_mixed(
    acq_function=acq_function,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=[{2: 0.0}, {2: 1.0}, {2: 2.0}],
)
```

The high-level model registry exposes these models with
`model_type="kronecker"` under mixed regression, multi-objective, binary,
multiclass, and ordinal task types.
