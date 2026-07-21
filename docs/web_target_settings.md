# Web workbench target settings

The Web workbench stores one task definition for every selected target column. A hard constraint is optional and may be set to `none`.

| Task | Objective definition | Optional constraint |
| --- | --- | --- |
| Regression | Maximizes the raw prediction by default. A target value maximizes the negative absolute error from that value. | `above` or `below` compares the predicted value with a numeric threshold. |
| Binary classification | Maximizes the probability of the selected `target_class`. | `above` or `below` compares the selected-class probability with a threshold in `[0, 1]`. |
| Multiclass classification | Maximizes the summed probability of one or more `target_classes`. | `above` or `below` compares that summed probability with a threshold in `[0, 1]`. |
| Ordinal regression | Encodes classes in the user-defined low-to-high `class_order`. Without a target value it maximizes expected rank. With one or more `target_values`, it maximizes utility based on distance to the nearest target rank. | `above` or `below` compares expected rank with the selected boundary class. |

Example settings:

```json
[
  {
    "target": "yield",
    "task_type": "regression",
    "goal": "none"
  },
  {
    "target": "pass_fail",
    "task_type": "classification",
    "target_class": "pass",
    "goal": "above",
    "value": 0.8
  },
  {
    "target": "phase",
    "task_type": "classification",
    "target_classes": ["alpha", "beta"],
    "goal": "none"
  },
  {
    "target": "grade",
    "task_type": "ordinal",
    "class_order": ["D", "C", "B", "A"],
    "goal": "target",
    "target_values": ["B", "A"]
  }
]
```

Targets with different task types are represented as a `HybridMultiOutputModel` with one submodel per output. The Web `multitask` option remains limited to multiple regression targets with numeric input variables.
