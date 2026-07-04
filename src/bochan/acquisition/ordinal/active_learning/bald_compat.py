from __future__ import annotations

from itertools import permutations

import torch
from torch import Tensor

from botorch.utils.transforms import t_batch_mode_transform

from .multi_output import (
    _ensure_class_probs_shape,
    _ensure_q_batch,
    _sample_latent,
    ordinal_entropy_from_probs,
    qMultiOutputOrdinalBALD as _BaseMultiOutputOrdinalBALD,
)


def _align_pointwise_axes(
    values: Tensor,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
    """Align pointwise axes to a reference tensor without reordering values.

    Kronecker task-view posterior samples can expose the transformed point axis
    before the optimizer t-batch axis, while marginal probabilities expose the
    conventional ``(*batch, q_like)`` order. This helper only accepts a true
    permutation of the same axes. It intentionally avoids a blind reshape,
    which could silently change the correspondence between candidates and
    conditional entropies.
    """

    if values.shape == reference.shape:
        return values

    aligned = values
    while aligned.ndim > reference.ndim:
        singleton_dims = [i for i, size in enumerate(aligned.shape) if size == 1]
        if not singleton_dims:
            break
        aligned = aligned.squeeze(singleton_dims[0])
        if aligned.shape == reference.shape:
            return aligned

    if aligned.ndim == reference.ndim:
        matching_permutations = [
            permutation
            for permutation in permutations(range(aligned.ndim))
            if tuple(aligned.shape[i] for i in permutation) == tuple(reference.shape)
        ]
        if matching_permutations:
            # Prefer the least disruptive permutation when repeated dimensions
            # admit more than one shape-compatible ordering.
            permutation = min(
                matching_permutations,
                key=lambda candidate: sum(
                    source_axis != target_axis
                    for target_axis, source_axis in enumerate(candidate)
                ),
            )
            return aligned.permute(*permutation)

    raise RuntimeError(
        f"{name}: could not align pointwise axes. "
        f"values.shape={tuple(values.shape)}, "
        f"reference.shape={tuple(reference.shape)}."
    )


class qMultiOutputOrdinalBALD(_BaseMultiOutputOrdinalBALD):
    """Multi-output ordinal BALD with Kronecker point-axis compatibility."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        scores: list[Tensor] = []

        for sm, lik in zip(self.submodels, self.ordinal_likelihoods):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=X.shape[-2])
            predictive_entropy = ordinal_entropy_from_probs(probs, eps=self.eps)

            latent_samples = _sample_latent(posterior, self.sampler)
            class_probs_given_f = lik.class_probs_from_f(latent_samples)
            cond_entropy = ordinal_entropy_from_probs(
                class_probs_given_f,
                eps=self.eps,
            ).mean(dim=0)
            cond_entropy = _align_pointwise_axes(
                cond_entropy,
                predictive_entropy,
                name="qMultiOutputOrdinalBALD conditional entropy",
            )
            scores.append(predictive_entropy - cond_entropy)

        if len(scores) == 0:
            raise RuntimeError("No submodels were available.")

        score_per_output = torch.stack(scores, dim=-1)
        score = self._aggregate_outputs(score_per_output)
        return self._finalize_pointwise_score(
            score,
            X,
            name="qMultiOutputOrdinalBALD",
        )


__all__ = ["qMultiOutputOrdinalBALD"]
