# Residual GP production hardening

Phase 12 validates material residual Gaussian-process workflows at three levels.

## 1. Real backend integration

`tests/models/materials/test_residual_real_backends_phase12.py` loads one real pretrained backend selected by `BOCHAN_MATERIAL_BACKEND` and validates:

- pretrained model loading,
- direct pretrained prediction,
- residual target construction,
- exact residual GP construction,
- corrected posterior mean and variance.

GitHub Actions runs CHGNet, M3GNet, and MACE as separate matrix jobs so backend-specific failures are isolated.

## 2. Artifact round-trip

`tests/serving/fastapi/test_material_residual_artifact_roundtrip_phase12.py` exercises the common `.bochan.pt` envelope with a cross-family multiple-baseline ModelList. It checks:

- tabular fit,
- deterministic baseline assignments,
- ModelList posterior,
- save through `save_tabular_artifact`,
- trusted load through `load_tabular_artifact`,
- output names and baseline metadata after load,
- posterior mean/variance equality before and after serialization,
- tabular prediction equality after load.

The test uses lightweight deterministic structure baselines so the serialization contract remains fast and deterministic in normal PR CI.

## 3. FastAPI lifecycle E2E

`tests/serving/fastapi/test_material_residual_e2e_phase12.py` uses the actual FastAPI router and dependency injection to cover:

1. fit,
2. predict,
3. candidates,
4. ask / pending registration,
5. tell,
6. save,
7. load,
8. predict after load.

The lifecycle test shares the production residual, tabular, artifact, candidate, and router code while replacing only the expensive pretrained backend with a deterministic test backend.

## Runtime health report

`validate_residual_production_model` provides a common validation contract for tests and diagnostics. It checks posterior shape and finite values and reports:

- number of outputs,
- deterministic-baseline output indices,
- posterior shape,
- shared parameter aliases.

`shared_parameter_aliases` intentionally calls `named_parameters(remove_duplicate=False)` so duplicate ownership of a shared pretrained encoder is visible instead of silently de-duplicated by PyTorch.

`assert_residual_posterior_equivalent` compares posterior mean and variance across serialization or transport boundaries.

## Scope

This phase does not add Web UI exposure, automatic physical-unit conversion, or force/stress tensor residuals. Those remain separate features after the scalar-property residual production path is stable.
