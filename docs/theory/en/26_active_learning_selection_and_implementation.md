# 26. Active Learning Selection and Implementation

Active Learning chooses observations to improve the model rather than directly maximize an objective.

Posterior variance is the simplest regression baseline. Predictive entropy targets ambiguous predictions. BALD focuses on information about model parameters and separates epistemic from irreducible uncertainty more explicitly. Margin and latent-straddle criteria concentrate on decision boundaries. NIPV-style criteria value expected reduction of integrated posterior uncertainty.

## Practical order

Start with variance or entropy baselines before using more expensive information-theoretic methods. For batch AL, avoid selecting multiple nearly identical points by using joint acquisition or diversity-aware selection.

## bochan perspective

AL criteria are exposed separately for regression, binary, multiclass, and ordinal model families so the score is computed in the appropriate posterior space.