# Material `train_Yvar` Phase 12: Web acquisition diagnostics view

## Goal

Phase 12 makes the observation-aware acquisition diagnostics from Phases 10 and 11 directly consumable by the bochan Web results screen.

The Web regression response now carries a compact `acquisition_diagnostics_view` payload. This is presentation metadata only; it does not rebuild the acquisition function or change optimizer state.

## Web payload

`acquisition_diagnostics_view` contains:

- `available`: whether acquisition diagnostics were resolved for the run.
- `cards`: compact summary values for objective training rows, acquisition baseline rows, failed experiments, pending experiments, and known `Yvar` presence.
- `warnings`: explanatory messages for partial observations, automatic baseline filtering, and failed / pending exclusions.
- `details`: baseline source/filtering, per-output observation counts, selected objective outputs, and known observation variance.
- `observation_report`: the current canonical `ObservationData.report()` payload when available.

A Web client can render `cards` directly near the candidate table and expose `details` in an expandable diagnostics section.

## Compatibility

The view is additive and best-effort. If a request has no visualization session or diagnostics cannot be resolved, the workflow still succeeds and returns `available=false`.

No candidate-generation settings, acquisition mathematics, baseline resolution, model fitting, failure weighting, pending handling, or `train_Yvar` semantics are changed.

## Tests

Focused tests cover:

- compact cards for partial-target diagnostics;
- warning generation for baseline filtering and failed / pending exclusions;
- safe unavailable state;
- propagation of the canonical observation report.
