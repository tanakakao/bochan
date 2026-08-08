# Web target roles and acquisition families

Each selected target is modeled, but its role in candidate generation is configured independently.

- **Optimization target**: included in the acquisition objective.
- **Constraint-only target**: modeled and evaluated for feasibility, but excluded from the acquisition objective.
- **Direction**: maximize or minimize for Bayesian optimization. Level-set estimation does not expose a maximize/minimize selector.
- **Target value**: treated as a distance objective for Bayesian optimization and as a zero-distance contour for level-set estimation.

At least one selected target must remain an optimization target.

The Optimize page exposes three acquisition families:

1. Bayesian optimization: EI, PI, UCB, EHVI, NEHVI, or NParEGO depending on objective count.
2. Active learning: posterior variance, predictive entropy, BALD, or NIPV.
3. Level-set estimation: Straddle, Boundary Variance, or ICU.

## Level-set estimation in the Web workbench

For an optimized LSE target, `above`, `below`, or `target` defines the contour to learn. It does **not** also become a hard feasibility constraint; sampling on both sides of the contour remains possible. An output whose optimization checkbox is cleared can still use `above` / `below` as a constraint-only feasibility rule.

The Web UI exposes the acquisition-specific scalar parameter:

- Straddle: `beta` (default 1.96)
- Boundary Variance: `tau` (default 1.0)
- ICU: `bandwidth`; Web value `0` keeps the class default and uses posterior standard deviation as the bandwidth

For multiple modeled outputs, optimized targets have non-negative relative `level_set_weight` values. Constraint-only outputs receive zero acquisition weight. The multi-output LSE implementation normalizes positive weights internally.

With InputPerturbation, mean aggregation is always supported. VaR / CVaR are available for Bayesian optimization and LSE. BO applies risk to the objective through `ObjectiveConfig`; LSE applies it to perturbation-expanded level-set scores through the regression LSE score objective. For `q > 1`, the Web UI forces sequential LSE candidate generation so each nominal candidate keeps its own perturbation-risk aggregation before it is added as pending. These paths are wired directly in the Web workflow and do not replace engine or acquisition functions at runtime.

The Web request stores integration-only markers such as `web_family`, `web_level_set_parameter`, `web_risk_type`, and `web_risk_alpha` inside `acquisition.acqf_kwargs` to keep the FastAPI request schema backward compatible. The workflow consumes these markers before constructing the acquisition function.
