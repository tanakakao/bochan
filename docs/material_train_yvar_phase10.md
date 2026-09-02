# Material `train_Yvar` Phase 10: acquisition diagnostics

## Goal

Phase 10 makes the observation semantics used during candidate generation visible without changing acquisition mathematics.

The diagnostics are intended for Web/API clients that need to explain why an acquisition baseline contains fewer rows than the fitted objective table, or whether failed / pending experiments were excluded before model fitting.

## Reported fields

After acquisition resolution, `BayesianOptimizer.last_acquisition_diagnostics` contains a JSON-serializable dictionary when diagnostics can be derived safely.

Typical fields are:

- `training_rows`: objective-model training rows available to the acquisition layer.
- `baseline_rows`: rows in the effective `X_baseline` after acquisition defaults are resolved.
- `baseline_source`: `automatic` or `explicit`.
- `baseline_filtered`: whether an automatic baseline was reduced by observation-aware scalar-objective semantics.
- `partial_observation`: whether any target output is only partially observed.
- `observed_per_output`: observed target counts for each output.
- `objective_output_indices`: scalar objective outputs that determine observation-aware baseline eligibility, when inferable.
- `known_observation_variance`: whether known observation variance is present in the fitted bundle.
- `observation_rows`, `completed_rows`, `success_rows`, `failed_rows`, `pending_rows`: experiment-state counts when `ObservationData` is available.
- `failed_excluded_from_objective_training` and `pending_excluded_from_objective_training`: explicit indicators that failed / pending rows are not objective-model training observations.

## FastAPI exposure

`CandidateResponse` now has an optional `diagnostics` field. The normal `/candidates` and `/ask` endpoints return the latest acquisition diagnostics together with candidates and acquisition value.

The field is additive and optional so non-observation workflows and integrations that cannot derive diagnostics continue to work.

## Safety / compatibility

Diagnostics are read-only and non-blocking. Failure to derive a diagnostic report must not make candidate generation fail. The acquisition configuration, baseline resolution, model fitting, `train_Yvar`, failure-probability weighting, and optimization backend are not modified by Phase 10.

## Tests

Focused tests cover:

- partially observed scalar baselines;
- effective baseline row counts;
- per-output observation counts;
- failed / pending experiment counts and exclusion flags;
- explicit baseline behavior;
- complete-data backward compatibility.
