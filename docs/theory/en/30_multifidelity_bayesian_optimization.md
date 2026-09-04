# 30. Multi-Fidelity Bayesian Optimization

Multi-fidelity BO models both design and fidelity:

```math
y=f(x,s)+\epsilon,
```

where `s` denotes an evaluation level such as pretrained model, MLIP, DFT, or experiment. The optimization target is usually the highest-fidelity function.

The next action chooses both components:

```math
(x_{t+1},s_{t+1})=\arg\max_{x,s}\alpha_{MF}(x,s).
```

Useful information from cheap fidelities depends on cross-fidelity correlation. A cost model `c(x,s)` enables cost-aware value of information or entropy reduction.

Discrete fidelities model named evaluation sources; continuous fidelity can represent mesh density, simulation accuracy, training budget, or similar tunable accuracy parameters.

For `bochan`, a clean future architecture separates fidelity specification, cost model, multi-fidelity surrogate, acquisition, and optimizer.