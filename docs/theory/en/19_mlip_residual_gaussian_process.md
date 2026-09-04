# 19. MLIP + Residual Gaussian Process

A residual Gaussian process learns the difference between a deterministic pretrained baseline and target-domain observations:

```math
y(x)=f_{MLIP}(x)+\delta(x)+\epsilon.
```

Training targets for the GP are therefore

```math
r_i=y_i-f_{MLIP}(x_i).
```

The corrected posterior mean is the MLIP baseline plus the GP residual mean. Posterior variance primarily describes uncertainty in the learned correction unless baseline uncertainty is modeled separately.

## Why residual learning can help

If the pretrained MLIP captures broad physical structure but has a systematic task-domain bias, the residual can be smoother and easier to learn than the original target. This can improve data efficiency with a small amount of DFT or experimental data.

## Distinction from multi-fidelity

Residual learning is not automatically multi-fidelity BO. A residual model fixes a baseline and learns a correction. Multi-fidelity BO additionally decides which fidelity to evaluate.

## bochan perspective

The common material factories expose residual-GP construction for energy, force, and stress across supported MLIP backends.