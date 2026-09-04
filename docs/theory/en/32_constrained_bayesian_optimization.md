# 32. Constrained Bayesian Optimization

Constrained BO optimizes an objective while respecting feasible regions. Constraints may be deterministic in the input, modeled as uncertain outcomes, or represented by a classifier.

A probabilistic feasibility term is

```math
P(g_j(x)\le 0\mid D).
```

Improvement-based utilities can be weighted or masked by joint feasibility. When no feasible observation exists, a feasibility-first phase is often more stable than pretending an objective incumbent exists.

Chance constraints specify a required feasibility probability rather than deterministic satisfaction. Multiple constraints require careful aggregation and calibration because multiplying poorly calibrated probabilities can become overconfident.

In `bochan`, optimizer-side input constraints and model-based outcome constraints should remain distinct. The feasibility wrappers and classification probability objectives support reusable constrained-acquisition composition.