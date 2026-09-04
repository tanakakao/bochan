# 39. Mixed, Discrete, and Combinatorial Bayesian Optimization

Many materials and process spaces mix continuous, integer, categorical, and combinatorial variables.

Continuous variables can use gradient-based acquisition optimization. Mixed categorical spaces may use models such as `MixedSingleTaskGP` with `optimize_acqf_mixed` or explicit enumeration of categorical assignments. Finite candidate pools can be scored directly.

Combinatorial selection should not be disguised as unconstrained continuous optimization. In composition discovery, selecting which elements are present is a combinatorial problem, while optimizing their fractions is a continuous simplex problem.

Conditional variables require validity-aware candidate generation or repair.

`bochan` can combine discrete candidate sets, mixed optimization, post-processing, and materials-specific subset selection while keeping each search-space mechanism explicit.