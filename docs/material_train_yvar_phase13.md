# Material train_Yvar Phase 13: candidate acquisition provenance

## Goal

Persist the exact observation-aware acquisition diagnostics used by each candidate-generation call so optimizer history remains auditable after later asks mutate `last_acquisition_diagnostics`.

## Design

Phase 10 introduced best-effort acquisition diagnostics and Phase 11 exposed the latest optimizer-level state. Phase 13 adds a per-call snapshot to the resolved `DataContext` retained by `CandidateResult`.

The snapshot is stored on the context as a private attribute rather than inside `DataContext.extra`. This is deliberate: `extra` participates in acquisition-function keyword construction, while provenance must remain read-only metadata and must never change acquisition behavior.

Use:

```python
from bochan.api.acquisition.provenance import candidate_acquisition_diagnostics

diagnostics = candidate_acquisition_diagnostics(optimizer.history[-1])
```

The helper returns a defensive deep copy. Older candidate results without a Phase 13 snapshot return `None`.

## Semantics

Each result preserves the diagnostics resolved for that ask, including:

- objective training row count
- acquisition baseline row count and source
- automatic baseline filtering
- partial-observation status
- observed count per output
- selected objective output indices
- known observation variance (`train_Yvar`)
- success / failed / pending counts when observation state is available

Later asks may replace `optimizer.last_acquisition_diagnostics`, but previous `CandidateResult` objects keep their own snapshot.

## Compatibility

Phase 13 does not modify:

- acquisition mathematics
- acquisition-function kwargs
- model fitting or `train_Yvar`
- failure/success weighting
- pending handling
- candidate values or acquisition values

The feature is additive and backward compatible.
