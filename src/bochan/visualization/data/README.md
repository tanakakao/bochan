# Visualization data boundary

`bochan.visualization.data` owns Plotly-independent data preparation.

- `frames.py`: training, prediction, and candidate DataFrame builders
- `grids.py`: one- and two-dimensional evaluation grids
- `ternary.py`: three-component simplex grids
- `study.py`: study history DataFrame builders

Prediction builders depend directly on the perturbation-aware prediction implementation. Import-time patching from the package root is not part of this boundary.
