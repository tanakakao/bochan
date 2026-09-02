# Material model architecture Phase 6

Phase 6 introduces a neutral mixed-process layout contract without changing material-model mathematics or categorical-kernel behavior.

## Scope

The shared contract now resolves a raw mixed input into:

- material selector/coordinate dimensions;
- categorical process dimensions;
- continuous branch dimensions;
- numeric process dimensions.

`DEFAULT` input transforms normalize numeric process dimensions only. Material selectors/coordinates and categorical process codes remain untouched.

This matches the existing ALIGNN, CHGNet, M3GNet and MACE mixed-model convention, where the structure selector stays in the continuous material branch and categorical process variables remain in BoTorch's mixed kernel. It also supports composition models by supplying their composition-coordinate dimensions as `material_dims`.

## New API

- `MixedProcessLayout`
- `resolve_mixed_process_layout`
- `resolve_mixed_process_input_transform`
- `select_continuous_process_branch`

## Compatibility boundary

Phase 6 intentionally does not replace categorical kernels with learned embeddings and does not alter `optimize_acqf_mixed` semantics. CrabNet's learned categorical-embedding DKL path remains an explicit model variant.

No changes are made to public model names, model types, GP/DKL equations, partial targets, known `train_Yvar`, structure caches, FastAPI/Web schemas, or saved-model module paths.

## Follow-up

Existing family-specific mixed wrappers can migrate to this layout helper incrementally. Once all families consume it, duplicated cat-dimension normalization and process-column selection code can be removed safely.
