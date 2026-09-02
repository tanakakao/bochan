# Material `train_Yvar` Phase 13: per-candidate acquisition provenance

## Goal

Phase 13 makes the observation-aware acquisition diagnostics from Phases 10-12 belong to the exact candidate-generation call that produced them.

Until Phase 12, the serving layer could read `optimizer.last_acquisition_diagnostics`. That is sufficient for a single immediate `/candidates` or `/ask` response, but it is mutable optimizer-level state. During acquisition comparison, only the last acquisition call remained available, so one diagnostics payload could not safely describe every returned candidate set.

## Candidate-level snapshot

`resolve_acquisition(...)` now snapshots the resolved diagnostics on the exact `DataContext` instance returned from acquisition resolution. The high-level optimizer stores that same context inside each `CandidateResult`, so `optimizer.history` retains the diagnostics used by every candidate-generation call.

The snapshot deliberately uses a private `DataContext` provenance attribute rather than `DataContext.extra`. `extra` is forwarded to acquisition constructors and therefore must not contain transport or provenance-only metadata.

Use:

```python
from bochan.api.acquisition.diagnostics import candidate_acquisition_diagnostics

result = optimizer.ask(..., return_result=True)
diagnostics = candidate_acquisition_diagnostics(result)
```

The helper returns a defensive deep copy. Mutating a response or a later diagnostics dictionary cannot change a previously stored candidate result.

## FastAPI behavior

`POST /models/{model_id}/candidates` and `POST /models/{model_id}/ask` now request the canonical `CandidateResult` internally and serialize diagnostics from that exact result.

`POST /models/{model_id}/candidates/compare` now returns diagnostics independently for every acquisition configuration. For example, the EI and UCB entries can carry different baseline row counts or objective-output indices without relying on mutable `optimizer.last_acquisition_diagnostics`.

The candidate service keeps its historical tuple return as the default. `return_result=True` is an additive internal option used by the serving routes.

## Compatibility

This phase does not change:

- acquisition mathematics;
- Phase 9 automatic baseline filtering;
- objective-model fitting;
- known `train_Yvar` handling;
- failed-experiment weighting;
- pending-experiment handling;
- `CandidateResponse` schema;
- the default tuple return of the FastAPI candidate application service.

If observation-aware diagnostics cannot be produced, candidate generation still succeeds and the per-candidate diagnostics value remains absent.

## Tests

`tests/test_candidate_acquisition_provenance.py` covers:

- defensive per-candidate diagnostics snapshots;
- attachment during acquisition resolution;
- the additive canonical-result path in the FastAPI service;
- independent diagnostics for every entry returned by `/candidates/compare`.
