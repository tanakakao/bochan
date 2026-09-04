# 42. Composition-Space Exploration

Composition optimization combines at least two different problems: selecting which elements are present and choosing their fractions.

For `K` active components,

```math
c_j\ge0,\qquad \sum_j c_j=1.
```

Element-subset selection is combinatorial; ratio optimization is continuous on a simplex. Process variables can then be added as an additional mixed-variable layer.

Composition representations may use raw fractions, descriptor aggregation, CrabNet/Roost embeddings, or log-ratio coordinates such as CLR/ILR for strictly positive compositions.

Conditional structure matters: if an element is absent, its fraction is fixed at zero. Treating every element fraction as an independent unconstrained variable wastes search effort and can generate invalid candidates.

`bochan` separates subset selection, feasible composition generation, representation, surrogate uncertainty, and acquisition.