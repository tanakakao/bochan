# 18. Machine-Learning Interatomic Potentials

Machine-learning interatomic potentials (MLIPs) approximate a potential-energy surface from atomic structure. The central quantities are energy, force, and stress.

```math
F_i=-\frac{\partial E}{\partial r_i}.
```

Energy is typically scalar per structure, force has `3N` components for `N` atoms, and a full stress tensor has nine components in the representation used by current `bochan` material workflows.

## Training and symmetry

MLIPs often combine energy, force, and stress losses. Structure models must respect translation, permutation, and rotation symmetries; forces are rotationally equivariant rather than invariant.

## Pretrained models

MACE, CHGNet, M3GNet/MatGL, and ALIGNN-FF can provide pretrained physical baselines. A pretrained model can support screening or relaxation without task-specific labels, but domain shift remains a major source of error.

## bochan perspective

`bochan` uses MLIPs in direct prediction, residual-GP correction, and structure-relaxation workflows. The physical baseline and the probabilistic correction are intentionally separate so uncertainty and target-domain calibration can be added without redefining the MLIP itself.