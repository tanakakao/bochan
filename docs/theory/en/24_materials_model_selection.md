# 24. Materials Model and Workflow Selection

Model choice should follow the available input information and the decision task.

| Available information | Typical model family |
|---|---|
| Tabular process variables | GP / DKL / tree-based surrogate |
| Composition only | descriptors, CrabNet, Roost, encoder + GP |
| Crystal structure | graph/equivariant model, structure encoder + GP |
| Energy/force/stress physics | MLIP |

## Direct versus residual

Use a direct pretrained model when its predictions are already the target. Use a residual GP when target-domain labels are available and a systematic correction is needed.

## GP versus DKL

A standard GP is the baseline for small structured data. DKL becomes attractive when learned representations are necessary and enough data exist to train them robustly. A frozen pretrained encoder plus GP is often a safer intermediate baseline.

## Workflow choices

Choose between model-only prediction, relaxation + ranking, and relaxation + acquisition according to whether geometry optimization and expensive follow-up evaluations are part of the decision loop.