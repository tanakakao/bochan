# Web workbench flow

1. **Data**: upload and inspect CSV or Excel data.
2. **Select**: select target columns and feature columns only.
3. **Settings**:
   - configure exactly one task and condition for every target;
   - configure feature search ranges, categories, steps, and fixed values.
4. **Optimize**: select the surrogate model, acquisition function, and candidate-generation budget.
5. **Results**: inspect target-specific predictions, condition checks, figures, and CSV output.
6. **Logs**: inspect structured FastAPI execution logs.

Keeping variable selection separate from optimization settings makes it possible to revise columns without mixing task semantics or search-space details into the selection step.
