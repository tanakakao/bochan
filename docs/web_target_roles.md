# Web target roles and acquisition families

Each selected target is modeled, but its role in candidate generation is configured independently.

- **Optimization target**: included in the acquisition objective.
- **Constraint-only target**: modeled and evaluated for feasibility, but excluded from the acquisition objective.
- **Direction**: maximize or minimize; independent from an above/below feasibility constraint.
- **Target value**: treated as a distance objective, so no maximize/minimize selector is required.

At least one selected target must remain an optimization target.

The Optimize page exposes a two-stage acquisition selection:

1. Bayesian optimization: EI, NEI, UCB, EHVI, or NEHVI depending on objective count.
2. Active learning: posterior variance, predictive entropy, or BALD.
3. Level-set estimation: straddle or boundary variance.

The web request stores the family marker in `acquisition.acqf_kwargs.web_family` to remain compatible with the strict FastAPI request schema. The workflow removes this marker before constructing the acquisition function.
