# 27. GP Model Selection: Single-Task, Multi-Task, SAAS, and Deep Kernels

Use the simplest GP whose assumptions match the problem.

`SingleTaskGP` is the default for continuous single-task regression. Known-noise and learned-noise variants should be distinguished. Multi-output problems can use independent models or correlated multitask models. `MultiTaskGP` and `KroneckerMultiTaskGP` encode cross-task covariance when observations support it.

High-dimensional ARD can be statistically weak with limited data. SAAS priors encourage sparse effective dimensionality. Deep Kernel Learning combines a neural feature map with GP uncertainty but requires enough data and careful validation.

A frozen pretrained encoder plus GP is often a useful compromise between fixed descriptors and fully trainable DKL.

Model selection should be driven by calibration and sequential performance, not training likelihood alone.