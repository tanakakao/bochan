"""Independent multi-output model aggregators.

Canonical implementations live in ``binary``, ``multiclass``, and ``ordinal``.
Unlike ``bochan.models.multitask``, these wrappers combine independently fitted
submodels and do not introduce learned task covariance.
"""
