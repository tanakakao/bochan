# Material `train_Yvar` Phase 11: diagnostics inspection endpoint

## Goal

Phase 11 makes the observation-aware acquisition diagnostics introduced in Phase 10 directly inspectable after candidate generation, without requiring clients to retain the previous candidate response.

## Endpoint

`GET /models/{model_id}/acquisition/diagnostics`

The response contains:

- `diagnostics`: the latest read-only acquisition diagnostics generated during acquisition resolution;
- `observation_report`: the current canonical `ObservationData.report()` summary when observation state is attached.

The observation report includes row counts for completed, successful, failed, and pending experiments plus per-output observation counts and known observation-variance availability when `Yvar` is present.

## Semantics

This endpoint does not build a new acquisition and does not change optimizer state. It only exposes already-resolved diagnostics and current observation state.

Before the first acquisition is generated, `diagnostics` is `null`. Non-observation workflows return `observation_report=null`.

## Compatibility

The endpoint is additive. Existing `/candidates` and `/ask` responses continue to include their optional Phase 10 `diagnostics` payload. Acquisition mathematics, baseline filtering, model fitting, failure weighting, pending handling, and known `train_Yvar` behavior are unchanged.

## Tests

Focused tests cover:

- latest acquisition diagnostics retrieval;
- successful / failed / pending observation counts;
- safe inspection before any candidate has been generated.
