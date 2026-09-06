# Phase 71 — MFKG per-fidelity score diagnostics

Phase 70 confirmed on Augmented Hartmann that MFKG target-fidelity selection remains strongly sensitive to the affine cost intercept. Phase 71 separates the two mechanisms that can produce low-fidelity preference:

1. the raw knowledge-gradient score itself prefers a low fidelity;
2. the cost-aware utility changes the ranking after cost normalization.

`run_mfkg_fidelity_score_diagnostic` fixes the optimizer to each allowed fidelity before every production MFKG step and records the best unweighted KG and cost-aware KG score for that fidelity. It then executes the normal cost-aware MFKG optimization over all fidelities and marks the fidelity actually selected.

The diagnostic intentionally leaves the production acquisition implementation unchanged.
