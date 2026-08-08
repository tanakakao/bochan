# Ordinal Knowledge Gradient

`qOrdinalKnowledgeGradient` is a **likelihood-aware one-step look-ahead acquisition for ordinal outcomes** whose terminal decision value is expected utility.

## Terminal utility

Assign a utility `u_k` to each ordinal class and select the final input with the largest posterior expected utility.

```math
U(x)=\sum_k u_k P(y=k\mid x,D)
```

```math
V(D)=\max_x U(x)
```

A new experiment returns one of the ordinal classes. Ordinal KG likelihood-reweights coherent posterior function samples for each hypothetical class, evaluates the best updated expected utility, and averages over predictive class probabilities.

```math
KG(x)=\sum_kP(y=k\mid x,D)V(D\cup\{(x,k)\})-V(D)
```

## Utility values

The default utilities are `[0, 1, ..., K-1]`. Specify nonuniform utilities when the value difference between adjacent ranks is not uniform.

```python
utility_values = [0.0, 0.2, 1.0, 5.0]
```

An `OrdinalExpectedUtilityMCObjective` can also provide the utility vector.

## Example

```python
from bochan.acquisition.ordinal.bayesian_optimization import qOrdinalKnowledgeGradient
from bochan.api import AcquisitionConfig, OptimizeConfig

acq_config = AcquisitionConfig(
    name="ordinal_kg",
    acqf_cls=qOrdinalKnowledgeGradient,
    acqf_kwargs={
        "bounds": bounds,
        "utility_values": [0.0, 0.5, 2.0, 5.0],
        "terminal_size": 128,
        "num_samples": 64,
    },
)

X_next, value = optimizer.candidate(
    acq_config,
    OptimizeConfig(q=1),
)
```

## v1 scope

- single-output ordinal models
- an ordinal likelihood exposing `class_probs_from_f`
- q=1
- expected-utility maximization
- continuous terminal sets may be generated from bounds with Sobol samples
- mixed/categorical spaces require an explicit valid terminal set
- pending-class conditioning is intentionally not approximated

The implementation does not reinterpret ordinal labels as Gaussian regression observations; hypothetical observations are handled through the ordinal likelihood.
