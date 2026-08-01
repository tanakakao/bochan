# Negative Binomial regression

Bochan exposes `negative_binomial_base`, `negative_binomial_deepgp`,
`negative_binomial_deepkernel`, `negative_binomial_saas`,
`negative_binomial_pca`, `negative_binomial_rembo`, `negative_binomial_rrp`,
`negative_binomial_hetero`, and `negative_binomial_multitask` through the model
registry and the optimizer factory. Continuous and mixed variants already exist
for every single-output family; multitask currently supports continuous inputs.

The public parameterization is `Y | mu, r ~ NegativeBinomial(mu, r)`, with
`E[Y]=mu` and `Var[Y]=mu + mu**2/r`. Thus `alpha=1/r`, PyTorch
`total_count=r`, and `logits=log(mu/r)`. Larger `r` approaches the Poisson
variance. Targets are raw, finite, non-negative integer-like counts; zero is an
observation, while only `NaN` denotes a missing wide multitask cell. Outcome
standardization is intentionally rejected.

`latent_posterior` represents the latent link scale, `mean_posterior` is the
positive, reparameterizable conditional-mean posterior used by MC acquisition,
and `predictive_posterior` adds Negative Binomial observation variance.
`sample_observations` is deliberately non-reparameterized. The correlated
multitask implementation uses one variational ICM GP over observed wide cells,
preserves the full latent task covariance, and learns a positive dispersion per
task. It does not impute missing training cells or split tasks into a ModelList.

Exposure/offsets, mixed-input multitask, exact non-Gaussian fantasy conditioning,
and a true input-dependent dispersion GP remain future work. Current models
assume equal exposure. The heteroscedastic legacy model adds an auxiliary
variance term rather than learning `r(X)`. Negative Binomial overdispersion does
not imply zero inflation; a dedicated zero-inflated model may be needed when
structural zeros remain unexplained.
