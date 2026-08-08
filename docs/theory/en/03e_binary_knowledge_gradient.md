# Binary Knowledge Gradient

`qBinaryKnowledgeGradient` is a **likelihood-aware one-step look-ahead acquisition for binary classification**.

## Terminal decision value

For binary BO, the final decision is the input with the largest target-class probability. By default the target is class 1.

```math
V(D)=\max_x P(y=1\mid x,D)
```

A new experiment at `x` produces a discrete label `y=0` or `y=1`. Binary KG evaluates the terminal value after each possible label and subtracts the current terminal value.

```math
KG(x)=\sum_{y\in\{0,1\}}P(y\mid x,D)V(D\cup\{(x,y)\})-V(D)
```

## Implementation

The implementation does not route a binary model through regression `qKnowledgeGradient` or a Gaussian fantasy approximation. It draws coherent latent posterior function samples jointly at the candidate and a finite terminal decision set, then Bayes-reweights those function samples by the Bernoulli likelihood of each hypothetical label.

This yields a sample-average one-step discrete-observation KG without refitting the classifier for every acquisition evaluation.

## Example

```python
from bochan.acquisition.binary.bayesian_optimization import qBinaryKnowledgeGradient
from bochan.api import AcquisitionConfig, DataContext, OptimizeConfig

acq_config = AcquisitionConfig(
    name="binary_kg",
    acqf_cls=qBinaryKnowledgeGradient,
    acqf_kwargs={
        "bounds": bounds,
        "target_class": 1,
        "terminal_size": 128,
        "num_samples": 64,
    },
)

X_next, value = optimizer.candidate(
    acq_config,
    OptimizeConfig(q=1),
    data_context=DataContext(),
)
```

An explicit `terminal_set` can be used for discrete or categorical search spaces.

## v1 scope

- single-output binary classification
- q=1
- target class 0 or 1
- continuous terminal sets may be generated from bounds with Sobol samples
- mixed/categorical spaces require an explicit valid terminal set
- pending-label conditioning is intentionally not approximated

For more than one experiment, observe the selected label and refit before requesting the next Binary KG candidate.
