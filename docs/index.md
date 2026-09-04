# bochan documentation

`bochan` is a BoTorch-oriented toolkit for Bayesian optimization, active learning, level-set estimation, tabular optimization, and materials-informatics workflows.

This page is the canonical entry point for project documentation. The repository also contains implementation-phase notes that are useful for design history, but new users should start from the guides below rather than from files named after individual development phases.

## Start here

| Goal | Recommended entry point |
|---|---|
| Learn the core optimizer API | [Repository README](https://github.com/tanakakao/bochan/blob/main/README.md) |
| Work with material structures and MLIPs | [MLIP workflows](materials/mlip-workflows.md) |
| Use HTTP / JSON APIs | [FastAPI reference](reference/fastapi.md) |
| Work with ALIGNN structures | [ALIGNN structure input](alignn_structure_input.md) |
| Use composition subset search | [Composition best-subset optimization](composition_best_subset.md) |
| Use LLM-assisted workflows | [LLM documentation](llm/README.md) |
| Understand documentation status and maintenance | [Documentation maintenance](maintenance.md) |

## Documentation layers

bochan documentation is organized conceptually into four layers.

### User guides

User guides describe supported public workflows and should be preferred for normal usage. They are expected to stay current as APIs evolve.

- [MLIP workflows](materials/mlip-workflows.md)
- [FastAPI reference](reference/fastapi.md)
- [Composition best-subset optimization](composition_best_subset.md)
- [LLM workflows](llm/README.md)

### Backend-specific notes

These pages provide deeper details for individual material encoders or serving surfaces.

- [ALIGNN FastAPI](alignn_fastapi.md)
- [CHGNet FastAPI](chgnet_fastapi.md)
- [M3GNet FastAPI](m3gnet_fastapi.md)
- [MACE FastAPI](mace_fastapi.md)
- [CrabNet FastAPI / Web](crabnet_fastapi_web.md)

### Model-family notes

These pages describe specialized surrogate-model families or experimental APIs.

- [Beta regression](beta-regression.md)
- [Beta models](beta_models.md)
- [Gamma models](gamma_models.md)

### Historical implementation notes

Files containing names such as `phase1`, `phase8`, `phase13`, or `release_readiness` are retained as engineering history. They document why a capability was introduced and what was validated at that point in development. They are not the preferred source for the current public API.

For materials/MLIP work in particular, use [MLIP workflows](materials/mlip-workflows.md) as the current consolidated reference. The older `material_models_architecture_phase*.md`, `mace_phase*.md`, and residual-GP phase notes remain useful when investigating design decisions or regressions.

## Major capability areas

### Bayesian optimization and active learning

The top-level README contains the most complete overview of tensor-based optimization, `BochanStudy`, tabular optimization, acquisition functions, mixed variables, classification, multi-objective optimization, and ask/tell workflows.

### Materials informatics

bochan supports composition- and structure-aware modeling, pretrained material encoders, residual Gaussian processes, and MLIP-backed structure relaxation. The unified MLIP layer currently covers MACE, CHGNet, M3GNet/MatGL, and ALIGNN-FF for energy, force, and stress workflows.

### Serving

FastAPI endpoints expose model lifecycle, candidate generation, capability discovery, material workflow validation/configuration, and runtime structure relaxation. See the [FastAPI reference](reference/fastapi.md).

## Building these docs locally

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Serve locally:

```bash
mkdocs serve
```

Build strictly before publishing:

```bash
mkdocs build --strict
```

The documentation configuration lives in `mkdocs.yml` at the repository root.
