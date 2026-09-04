# 22. From Composition to Crystal Structure

A composition does not uniquely determine a crystal structure. Conceptually, structure generation can be viewed as sampling or optimizing under

```math
p(S\mid c).
```

A practical materials workflow therefore separates composition screening from structure generation and structure-aware evaluation.

## Staged search

A common hierarchy is:

```text
composition model
  -> shortlisted compositions
  -> candidate structures
  -> structure model / MLIP relaxation
  -> residual GP or high-fidelity model
  -> DFT / experiment
```

CrabNet/Roost are composition-level tools; ALIGNN is structure-aware; ALIGNN-FF and other MLIPs support energy/force/stress and relaxation.

## Key implication

Composition prediction can be valuable before a structure is known, but structure-sensitive properties require structure information or an explicit uncertainty model over possible structures.

`bochan` keeps these stages modular so composition and structure decisions can be combined without pretending they are the same representation.