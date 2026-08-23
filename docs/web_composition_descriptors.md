# EWB composition descriptors

The React experiment workbench can augment a composition model with descriptors
that are recomputed from each composition. Descriptor values are **derived model
features**, not independent optimization variables.

## Data flow

```text
formula
  -> Fraction / CLR / ALR / ILR decision coordinates
  -> candidate optimization in composition/process space
  -> atomic fractions reconstructed inside the model InputTransform
  -> elemental-property descriptors recomputed
  -> surrogate posterior / acquisition value
```

This keeps the optimizer physically consistent. It cannot propose a descriptor
value that disagrees with the proposed composition.

## EWB controls

On **Model > 組成式のモデル変換**, enable **元素物性記述子を追加**.

Built-in elemental properties:

- atomic number
- atomic weight

Available weighted statistics:

- mean
- standard deviation
- minimum
- maximum
- range

Composition-only descriptors:

- number of active elements
- ideal mixing entropy, `-sum(x * log(x))`

The settings are saved in the normal `web_composition` payload and participate
in the model-reuse signature.

## Optimization contract

The tabular `composition_sites` adapter intentionally keeps
`include_descriptors=False`. Only Fraction/CLR/ALR/ILR coordinates are included
in the optimizer decision tensor. A Web-specific BoTorch `InputTransform`
appends descriptors at model evaluation time.

As a result:

- training rows use descriptors calculated from their observed formulas;
- `predict()`/posterior evaluation recalculates descriptors from the supplied
  composition coordinates;
- every acquisition-function evaluation recalculates descriptors from the
  candidate composition;
- repaired candidates are rescored after composition repair;
- candidate CSV/results contain the formula, fractions and process conditions,
  not independently optimized descriptor columns.

## Initial compatibility limits

Descriptor augmentation currently does not combine with:

- `crabnet_gp` / `crabnet_dkl` (CrabNet already owns a learned composition
  representation);
- input perturbation;
- external estimator models (`random_forest`, `lightgbm_ensemble`,
  `ngboost_ensemble`, `tabpfn`).

These combinations are rejected explicitly rather than silently producing
inconsistent features.

The descriptor transform is implemented with Torch operations so gradients from
BoTorch acquisition functions propagate through CLR/ALR/ILR inverse transforms
to the composition decision coordinates.
