# Binary foundation models

## TabPFN

`TabPFNBinaryClassificationModel` wraps the official `TabPFNClassifier` through
bochan's probability-posterior protocol.

- `model_type="tabpfn"`
- normal and mixed inputs
- mixed models pass `cat_dims` to TabPFN as native `categorical_features_indices`
- public `predict_proba` is exposed as one probability-posterior member
- the public TabPFN inference ensemble is not reinterpreted as independent Bayesian posterior samples
- epistemic probability variance is therefore zero in this first integration

TabPFN crosses a Tensor-to-NumPy estimator boundary in bochan. Candidate
optimization should therefore use derivative-free optimization such as `evo` or
`evo_mixed` rather than gradient-based `optimize_acqf`.
