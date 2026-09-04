# 44. Hierarchical MLIP, DFT, and Experiment Workflows

Materials discovery often combines evaluation sources with very different cost and fidelity:

```text
composition/ML screening
 -> MLIP relaxation
 -> DFT
 -> experiment
```

Each stage should filter or improve the candidate set rather than pretending to be a perfect substitute for the final experiment.

A residual GP can calibrate a pretrained baseline to high-fidelity labels:

```math
y_H=f_{base}(x)+\delta_{GP}(x)+\epsilon.
```

When the decision includes which evaluation source to run next, the problem becomes multi-fidelity and a cost-aware acquisition can trade information against computational or experimental expense.

Domain shift, convergence failures, and missing results must be monitored explicitly. Failed DFT or experiments may themselves provide feasibility information.

`bochan` can act as the common representation/surrogate/acquisition layer while DFT codes and laboratory systems remain external evaluators.