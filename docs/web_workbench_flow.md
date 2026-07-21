# Web workbench flow

The React workbench uses six focused pages:

1. **Data** uploads and profiles CSV or Excel data.
2. **Select** chooses target and explanatory columns only.
3. **Settings** defines one rule per target and configures explanatory columns as search variables.
4. **Optimize** chooses the surrogate model, acquisition function, and candidate-generation budget.
5. **Results** reviews predictions, feasibility, plots, and candidate CSV output.
6. **Logs** inspects structured FastAPI execution logs.

Keeping variable selection separate from optimization settings makes it possible to revise columns without mixing task semantics or search-space details into the selection step.

## Visual system

The shell uses a restrained modern workbench design:

- monochrome `bochan` brand with blue used mainly for interaction and state;
- sticky header, workflow navigation, contextual side rails, and compact status bar;
- shared spacing, typography, radius, border, shadow, and semantic-color tokens;
- consistent cards, forms, tables, upload states, validation messages, and loading states;
- responsive desktop, tablet, and mobile navigation;
- light and dark themes, including Plotly chart surfaces and controls;
- visible keyboard focus and reduced-motion support.
