# Multiclass foundation models

## TabPFN

`TabPFNMulticlassClassificationModel` wraps the official `TabPFNClassifier`
through bochan's multiclass probability-posterior protocol.

- `model_type="tabpfn"`
- normal and mixed inputs
- mixed models pass `cat_dims` to TabPFN as native `categorical_features_indices`
- final class probabilities are exposed as one finite posterior member
- the public TabPFN inference ensemble is not treated as independent Bayesian posterior samples
- epistemic probability variance is therefore zero in this first integration

TabPFN crosses a Tensor-to-NumPy estimator boundary in bochan. Candidate
optimization should therefore use derivative-free optimization such as `evo` or
`evo_mixed` rather than gradient-based `optimize_acqf`.
