# 38. Level-Set Estimation and Boundary Search

Level-set estimation identifies regions relative to a threshold rather than maximizing the response:

```math
L_h^+=\{x:f(x)\ge h\}.
```

Boundary-search acquisitions prioritize points whose posterior places substantial mass near the threshold. Straddle combines distance to the threshold with uncertainty. Margin, ICU-style criteria, and boundary variance provide related ways to target uncertain decision boundaries.

For classification, the relevant boundary may be a probability threshold rather than a latent-score threshold. The correct space must be explicit.

Stopping criteria can use confidence-set stability, boundary uncertainty, or estimated classification loss rather than improvement in the best objective.

Applications include specification windows, phase boundaries, pass/fail regions, and safe operating envelopes.