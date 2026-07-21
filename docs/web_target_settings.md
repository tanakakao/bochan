# Web workbench target settings

The Web workbench stores exactly one setting for every selected target column.

| Task | Above / below | Target value |
| --- | --- | --- |
| Regression | Optimizes in the selected direction and applies the numeric threshold as a feasibility condition. | Maximizes the negative absolute error from the numeric target value. |
| Classification | For binary targets, applies a threshold to the probability of the second class in ascending encoded order. Multiclass targets use target-value mode. | Maximizes the probability of the selected class. |
| Ordinal regression | Uses the ascending observed category order, optimizes expected rank, and applies the selected rank as the threshold. | Maximizes utility based on negative distance from the selected rank. |

Targets with different task types are represented as a `HybridMultiOutputModel` with one submodel per output. The Web `multitask` option is currently limited to multiple regression targets with numeric input variables.
