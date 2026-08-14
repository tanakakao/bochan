# FastAPI package architecture

The FastAPI adapter follows a transport-first package boundary.

- `app.py` owns application construction.
- `routers/` owns HTTP routes and HTTP error translation.
- `schemas/` owns request and response contracts.
- `services/` owns stateful application workflows used by thin routers.
- `stores/` owns in-process persistence abstractions used by the adapter.
- `converters.py` owns conversion between JSON/Pydantic values and canonical `bochan.api` values.
- `target_categories.py` owns transport-only categorical target encoding and metadata.
- `dependencies.py` owns FastAPI dependency wiring for stores.

The package root must not accumulate workflow or domain implementations. New endpoint behavior belongs in a concrete router and, when it is more than HTTP adaptation, a concrete service module. Package `__init__.py` files must not become forwarding facades for service functions.

`router.py` is intentionally export-free. Router composition is owned by `routers/__init__.py`; the empty root file remains only while existing smoke workflows still reference its path for static linting.
