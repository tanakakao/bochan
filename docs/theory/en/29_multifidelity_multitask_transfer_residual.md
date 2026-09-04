# 29. Multi-Fidelity, Multi-Task, Transfer, and Residual Learning

These concepts are related but not interchangeable.

- **Residual learning** fixes a baseline and learns a correction: `y_H=b(x)+δ(x)+ε`.
- **Multi-fidelity** models evaluations at different accuracy/cost levels and may choose both candidate and fidelity.
- **Multi-task** models correlated functions indexed by a task variable; tasks need not form an accuracy hierarchy.
- **Transfer learning** reuses pretrained representations or model parameters in a new domain.

A common multi-fidelity relation is `f_H(x)=ρ f_L(x)+δ(x)`, but other kernels over `(x,s)` are possible.

MLIP + residual GP is therefore not automatically multi-fidelity BO. It becomes multi-fidelity only when evaluation source/fidelity is an explicit modeled decision.

In `bochan`, these should remain separate design axes rather than overloading the residual-GP abstraction.