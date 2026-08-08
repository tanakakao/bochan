"""Compatibility imports for shared external-estimator preprocessing."""

from bochan.models.external.common import (
    _ExternalClassifierMixin,
    _ExternalEstimatorMixin,
    _ExternalRegressorMixin,
    _MixedCategoricalEncoder,
    _MixedCategoricalMixin,
    _check_one_to_one_input_transform,
    _require_classification_targets,
    _require_external_inputs,
    _require_single_output,
    _to_numpy,
    _validate_classification_values,
    _validate_output_indices,
)

__all__ = [
    "_ExternalClassifierMixin",
    "_ExternalEstimatorMixin",
    "_ExternalRegressorMixin",
    "_MixedCategoricalEncoder",
    "_MixedCategoricalMixin",
    "_check_one_to_one_input_transform",
    "_require_classification_targets",
    "_require_external_inputs",
    "_require_single_output",
    "_to_numpy",
    "_validate_classification_values",
    "_validate_output_indices",
]
