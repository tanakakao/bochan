# Material train_Yvar Phase 14: comparison diagnostics provenance

## Goal

Expose the exact acquisition diagnostics captured for each result returned by the FastAPI acquisition-comparison endpoint.

Phase 13 made acquisition diagnostics persistent per `CandidateResult`. Phase 14 consumes that stored provenance in:

- `POST /models/{model_id}/candidates/compare`

instead of relying on the optimizer-level `last_acquisition_diagnostics` value, which only represents the most recently resolved acquisition.

## Behavior

Each entry in `CompareCandidatesResponse.results` now includes the diagnostics snapshot belonging to that specific candidate result.

For example, two acquisition configurations can legitimately report different:

- `baseline_rows`
- `baseline_filtered`
- `objective_output_indices`
- partial-observation state
- failed / pending counts
- known observation variance state

The response therefore preserves the provenance of each acquisition independently.

## Compatibility

This phase is additive only.

- Candidate values are unchanged.
- Acquisition values are unchanged.
- Acquisition construction and optimization are unchanged.
- `train_Yvar` handling is unchanged.
- Failure weighting and pending-observation behavior are unchanged.
- Results created before Phase 13, or custom candidate results without a diagnostics snapshot, continue to serialize with `diagnostics: null`.

## Why this matters

Before Phase 13/14, a comparison route could only inspect the optimizer's mutable latest diagnostics value. With several acquisitions evaluated sequentially, that value corresponds to the last acquisition only and cannot safely describe the earlier comparison entries.

Phase 14 removes that ambiguity by reading diagnostics from each `CandidateResult` through `candidate_acquisition_diagnostics(result)`.

## Tests

`tests/test_fastapi_compare_acquisition_diagnostics.py` verifies that:

1. each compared acquisition returns its own stored diagnostics;
2. optimizer-level latest diagnostics cannot overwrite earlier comparison provenance;
3. candidate results without a stored snapshot remain backward-compatible with `diagnostics=None`.
