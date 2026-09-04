# Documentation maintenance

bochan has grown through many incremental implementation phases. To keep the documentation useful as the codebase expands, new documentation should distinguish current user guidance from historical engineering notes.

## Document classes

### Canonical user guide

A canonical guide describes the current supported public workflow. It should avoid phase numbers in the filename and should be updated when the corresponding public API changes.

Examples:

- `docs/materials/mlip-workflows.md`
- `docs/reference/fastapi.md`
- `docs/composition_best_subset.md`

### Backend-specific reference

A backend reference documents behavior that does not fit cleanly in the common guide, such as backend-native structures, model artifacts, or compatibility constraints.

Examples include the existing ALIGNN, MACE, CHGNet, M3GNet, and CrabNet pages.

### Historical implementation note

A historical note records a development phase, migration, performance investigation, or release-readiness checkpoint. These documents remain useful for architecture archaeology and regression analysis, but should not be the only documentation for a current public feature.

Typical filename patterns are:

```text
*_phase1.md
*_phase8_*.md
*_phase13.md
*_release_readiness.md
```

When a multi-phase feature reaches a stable public interface, create or update a non-phase canonical guide and link the older notes from a history section.

## Source-of-truth order

When documentation conflicts, use this order:

1. public source code and schemas on `main`;
2. canonical user guides listed in `mkdocs.yml`;
3. backend-specific reference pages;
4. historical phase notes.

A phase note should not be edited solely to make it look current if its purpose is to preserve what was implemented at that phase. Prefer adding a clear link to the current canonical guide.

## Recommended structure

```text
docs/
  index.md
  materials/
    mlip-workflows.md
  reference/
    fastapi.md
  llm/
    README.md
    ...
  <existing backend-specific references>
  <historical phase notes>
```

The current restructuring is intentionally conservative: historical files are not moved en masse because existing links may depend on their paths. MkDocs navigation provides the organized user-facing hierarchy while preserving old URLs.

## Writing conventions

- Prefer public import paths over private implementation modules.
- State tensor shape contracts explicitly when they are part of the API.
- State unit/sign conventions for physical quantities rather than implying conversion.
- Separate dependency-light validation/configuration from runtime behavior that imports heavy optional packages.
- Include one minimal example before advanced configuration.
- For FastAPI pages, show the HTTP method and full route.
- Avoid naming new canonical documents after implementation phases.

## Updating docs with code changes

For a public API change, check at least the following:

- `README.md` when the top-level usage story changes;
- the relevant canonical guide under `docs/`;
- `docs/reference/fastapi.md` when routes or schemas change;
- `mkdocs.yml` when a new canonical page is introduced;
- examples or backend notes that demonstrate the changed behavior.

## Local validation

Install docs tooling:

```bash
pip install -e ".[docs]"
```

Run a strict build:

```bash
mkdocs build --strict
```

A strict MkDocs build should be part of documentation-focused pull requests so broken internal links and navigation errors are caught before merge.
