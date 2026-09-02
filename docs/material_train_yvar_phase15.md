# Material `train_Yvar` Phase 15: end-to-end observation lifecycle

Phase 15 closes the observation-aware material optimization series with an end-to-end regression test that exercises the state transitions used by real experimental loops.

## Covered lifecycle

The test starts from a multi-output material data set containing all of the following at once:

- partially observed targets,
- cell-aligned known observation variance (`Yvar`),
- one failed experiment,
- one pending experiment,
- an independent success/failure classifier,
- scalar acquisition baselines for different outputs,
- vector / multi-output acquisition baseline semantics,
- candidate-level acquisition provenance.

It then executes:

1. `fit_observations(...)`
2. `ask(...)`
3. `tell(...)` for the previously pending experiment
4. refit of the objective and failure models
5. a second `ask(...)`
6. `compare_acquisitions(...)`
7. inspection of diagnostics retained by each `CandidateResult`

## Assertions

The Phase 15 E2E test verifies that:

- failed and pending experiments never enter objective-model training;
- completed failed experiments remain available to the success classifier;
- pending rows are propagated to `X_pending` before `tell(...)`;
- a matching `tell(...)` resolves the existing pending row rather than duplicating it;
- partially observed outputs keep independent observed-row counts;
- known `Yvar` reaches each output-specific objective model;
- diagnostics report known observation variance from the canonical observation state;
- scalar acquisition baselines contain only rows observed for the selected output;
- multi-output/vector acquisitions preserve the full partial baseline;
- diagnostics from the first `ask(...)` remain unchanged after later `tell(...)` and `ask(...)` calls;
- comparison results retain distinct diagnostics for each acquisition;
- complete known-variance data remains a baseline-filtering no-op.

## Diagnostic correction

During the Phase 15 integration pass, the diagnostics contract exposed a representation mismatch: partial multi-output objective bundles do not necessarily retain a top-level `train_Yvar` attribute even though canonical `ObservationData` retains known variance and passes it to output-specific models.

`build_acquisition_observation_diagnostics(...)` now treats `ObservationData.report()["known_observation_variance"]` as an authoritative source in addition to a bundle-level `train_Yvar` attribute. This keeps diagnostics aligned with the actual observation state without changing model fitting or acquisition mathematics.

The existing acquisition diagnostics fixture was also corrected so it no longer passes an unsupported `train_Yvar` keyword directly to `ModelBundle`.

## Compatibility

Phase 15 does not change:

- objective values,
- acquisition-function mathematics,
- candidate optimization,
- failure-probability weighting,
- pending-row matching semantics,
- partial-observation model construction,
- known-variance model inputs.

The only production change is diagnostic provenance: known observation variance is now reported correctly for canonical observation-aware workflows.
