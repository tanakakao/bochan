# Optimizer package architecture

`bochan.optim` is organized by optimization algorithm family rather than by
individual historical modules. The top-level package only aggregates the public
optimizer surface used by API dispatch.

## Layout

- `gradient/`: gradient-based candidate optimization. `botorch.py` contains
  BoTorch wrappers, `torch.py` contains `torch.optim` implementations, and
  `multitask.py` contains the correlated-multitask restart strategy.
- `evolutionary/`: derivative-free population/search methods such as GA, PSO,
  simulated annealing, and CMA-ES.
- `nsgaii/`: NSGA-II backend, strategy, API adapter, objective-space constraint
  adaptation, output-shape normalization, and final-batch diversity selection.
- `thompson/`: finite-pool Thompson sampling core and the high-level objective /
  constraint adapter.
- `llm/`: LLM-generated candidate-set backends. The acquisition function remains
  responsible for final scoring.

## Extension rule

Add a new optimizer family as a new subpackage when it has its own optimization
semantics or support utilities. Each family owns its algorithm-specific core,
adapters, and post-processing helpers; cross-family problem definitions remain
outside `bochan.optim`.

Keep shared problem constraints under `bochan.constraints`; optimizer packages
should only contain optimizer-facing adaptation. Avoid runtime monkey patches and
forwarding modules for removed paths.
