# 23. Joint Composition, Structure, and Process Optimization

A unified materials decision variable may contain composition, structure, and process information:

```math
x=(c,S,p).
```

This creates a heterogeneous search space with continuous, discrete, categorical, and conditional variables.

## Conditional structure

Some variables only exist when a particular composition or structure family is selected. Feasibility constraints and candidate-generation rules must therefore be represented explicitly rather than reduced to a simple box.

## Multi-objective and robust settings

Materials development commonly trades performance, stability, cost, and manufacturability. Multi-objective BO, constraints, and robust objectives such as VaR/CVaR can be layered on top of the same surrogate architecture.

## bochan perspective

`bochan` is best viewed as the probabilistic decision layer. Material generation, physical simulation, and instrument control may remain external while composition/structure/process representations and acquisitions are combined through common interfaces.