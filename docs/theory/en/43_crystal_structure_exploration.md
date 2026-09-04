# 43. Crystal-Structure Exploration

A crystal structure can be represented conceptually as

```math
S=(Z,R,L),
```

with atomic species, coordinates, and lattice. The same composition can have multiple polymorphs with different properties.

Structure generation and structure relaxation are different tasks. Generation proposes topologies/configurations; relaxation locally minimizes an energy surface from an initial structure.

Candidate structures may come from databases, prototype substitution, enumeration, random generation, or generative models. MLIPs can cheaply relax and screen them before DFT.

For expensive follow-up evaluation, a relaxed candidate bank can feed a GP/residual GP and BO or AL acquisition. Posterior-mean ranking is not the same as uncertainty-aware acquisition.

`bochan` primarily covers the relaxation, surrogate, and decision layers while allowing external structure generators and DFT engines to remain modular.