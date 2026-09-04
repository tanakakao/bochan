# 41. Bayesian Optimization Diagnostics and Failure Modes

When BO proposes implausible candidates, debug the complete decision pipeline rather than only the acquisition formula.

```text
data
 -> transforms
 -> model fit
 -> posterior
 -> objective transform
 -> acquisition
 -> acquisition optimizer
 -> post-processing
```

Common problems include collapsed or extreme GP length scales, poorly estimated noise, overconfident posteriors, incorrect objective direction, inconsistent `best_f` or reference points, acquisition saturation, duplicate candidates, and post-processing that invalidates optimizer assumptions.

Extrapolation and domain shift are particularly important for pretrained material models.

Useful checks include posterior calibration, train/validation residuals, uncertainty versus distance, acquisition values on known points, duplicate distance, constraint satisfaction, and sequential regret or hypervolume traces.

A good BO system monitors decision quality, not only predictive RMSE.