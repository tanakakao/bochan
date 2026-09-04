# 31. Robust Bayesian Optimization: Input Perturbation, VaR, and CVaR

Robust BO optimizes performance under implementation variability rather than only the nominal setting.

```math
\tilde x=x+\xi.
```

A mean-robust objective uses `E[f(x+ξ)]`. VaR summarizes a tail quantile, while CVaR summarizes average performance inside the adverse tail. For maximization, the lower tail is typically the risk-relevant region; the exact `alpha` convention must be stated explicitly.

Monte Carlo input perturbations produce `n_w` scenarios per nominal candidate. Posterior-sample uncertainty and perturbation uncertainty are different axes and must not be reduced accidentally.

Input uncertainty is also distinct from observation noise. A heteroscedastic likelihood models variability in `y|x`; robust BO models uncertainty in realized input.

In `bochan`, `n_w`, `risk_type`, and `alpha` connect input perturbation with mean/VaR/CVaR objective aggregation.