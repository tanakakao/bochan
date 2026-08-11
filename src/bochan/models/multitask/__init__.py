"""Cross-family infrastructure for correlated multi-task models.

Import concrete helpers from ``task_feature``, ``wide``, ``validation``,
``kronecker``, or ``mixed`` explicitly. Keeping package initialization lightweight
prevents one strategy import from eagerly importing every model family and makes
future strategy packages such as multi-fidelity easier to extend independently.
"""
