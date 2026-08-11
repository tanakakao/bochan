"""Negative Binomial count regression models.

Only base models are imported at package initialization to avoid circular
imports with optional deep / high-dimensional / robust modules. Import those
variants from their subpackages directly when needed.
"""

from .base import *
