"""BoTorch-native Deep Ensemble models for binary and multiclass classification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
import torch.nn.functional as F
from botorch.models.ensemble import EnsembleModel
from torch import Tensor, nn
from torch.optim import Optimizer

from bochan.models.classification.external.base import _ExternalProbabilityClassifierMixin
from bochan.models.external.common import _require_classification_targets
from bochan.models.regression.neural.deep_ensemble import (
    _MixedCategoricalEncoder,
    _make_activation,
)


class _DenseClassifier(nn.Module):
    """Configurable MLP classifier used by Deep Ensemble members."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        widths = [int(width) for width in hidden_dims]
        if any(width <= 0 for width in widths):
            raise ValueError("hidden_dims must contain only positive widths.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be within [0, 1).")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in widths:
            layers.append(nn.Linear(previous, width))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        layers.append(nn.Linear(previous, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        return self.network(X)


def _bootstrap_classification_indices(
    labels: Tensor,
    *,
    num_classes: int,
    bootstrap: bool,
    random_state: int | None,
    member_index: int,
) -> Tensor:
    """Bootstrap rows while retaining at least one observation from every class."""
    n = int(labels.numel())
    device = labels.device
    if not bootstrap:
        return torch.arange(n, device=device)

    generator = torch.Generator(device=device)
    if random_state is not None:
        generator.manual_seed(int(random_state) + 10_000 + int(member_index))
    indices = torch.randint(n, size=(n,), generator=generator, device=device)
    sampled = labels.index_select(0, indices).clone()

    for class_index in range(int(num_classes)):
        if torch.any(sampled == class_index):
            continue
        candidates = torch.nonzero(labels == class_index, as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            raise RuntimeError(f"Training labels do not contain class {class_index}.")
        replace_at = torch.randint(
            n,
            size=(1,),
            generator=generator,
            device=device,
        ).item()
        candidate_at = torch.randint(
            int(candidates.numel()),
            size=(1,),
            generator=generator,
            device=device,
        ).item()
        indices[replace_at] = candidates[candidate_at]
        sampled[replace_at] = class_index
    return indices


class _DeepEnsembleClassificationModel(
    _ExternalProbabilityClassifierMixin,
    EnsembleModel,
):
    """Shared differentiable Deep Ensemble classifier.

    The model reuses bochan's finite probability-posterior contract used by the
    external Random Forest / NGBoost classifiers, but all inference remains in
    PyTorch. Consequently, class-probability samples retain gradients with
    respect to candidate inputs.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        binary: bool,
        num_classes: int | None = None,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: nn.Module | None = None,
        members: Sequence[nn.Module] | None = None,
        member_factory: Callable[[int, int], nn.Module] | None = None,
        weights: Tensor | None = None,
        _feature_encoder: nn.Module | None = None,
        _encoded_input_dim: int | None = None,
    ) -> None:
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        requested_classes = 2 if binary else num_classes
        train_Y, inferred_classes = _require_classification_targets(
            train_X,
            train_Y,
            model_name="Deep Ensemble classifier",
            num_classes=requested_classes,
        )
        if weights is not None:
            weights = weights.to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(weights=weights)

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.binary = bool(binary)
        self.num_classes = int(inferred_classes)
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.feature_encoder = _feature_encoder if _feature_encoder is not None else nn.Identity()
        encoded_input_dim = int(_encoded_input_dim or train_X.shape[-1])
        output_dim = 1 if self.binary else self.num_classes

        if members is not None:
            member_list = list(members)
            if not member_list:
                raise ValueError("members must contain at least one neural network.")
            if ensemble_size != 5 and ensemble_size != len(member_list):
                raise ValueError("ensemble_size must match len(members) when both are specified.")
            self.ensemble_size = len(member_list)
        else:
            self.ensemble_size = int(ensemble_size)
            member_list = [
                self._build_member(
                    index=index,
                    input_dim=encoded_input_dim,
                    output_dim=output_dim,
                    hidden_dims=hidden_dims,
                    activation=activation,
                    dropout=dropout,
                    member_factory=member_factory,
                )
                for index in range(self.ensemble_size)
            ]

        if weights is not None and weights.numel() != self.ensemble_size:
            raise ValueError("weights must contain one value per ensemble member.")

        self.members = nn.ModuleList(member_list)
        self.members.to(dtype=train_X.dtype, device=train_X.device)
        self._configure_probability_acquisition_bridge()
        self._is_fitted = False
        self.fit_losses: list[list[float]] = []

    def _build_member(
        self,
        *,
        index: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        member_factory: Callable[[int, int], nn.Module] | None,
    ) -> nn.Module:
        def build() -> nn.Module:
            if member_factory is not None:
                member = member_factory(input_dim, output_dim)
                if not isinstance(member, nn.Module):
                    raise TypeError("member_factory must return a torch.nn.Module.")
                return member
            return _DenseClassifier(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )

        if self.random_state is None:
            return build()
        with torch.random.fork_rng():
            torch.manual_seed(int(self.random_state) + index)
            return build()

    def _encode_features(self, X: Tensor) -> Tensor:
        return self.feature_encoder(X)

    def _member_logits(self, member: nn.Module, X: Tensor) -> Tensor:
        logits = member(X)
        expected = 1 if self.binary else self.num_classes
        if logits.shape[:-1] != X.shape[:-1] or logits.shape[-1] != expected:
            raise RuntimeError(
                f"Each Deep Ensemble classifier member must return shape [..., {expected}]."
            )
        return logits

    def _member_probabilities(self, member: nn.Module, X: Tensor) -> Tensor:
        logits = self._member_logits(member, X)
        if self.binary:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _training_tensors(self) -> tuple[Tensor, Tensor]:
        self.train()
        transformed_X = self.transform_inputs(self.train_X)
        encoded_X = self._encode_features(transformed_X)
        labels = self.train_Y.squeeze(-1).long()
        return encoded_X, labels

    def _epoch_order(self, n: int, member_index: int, epoch: int, device: torch.device) -> Tensor:
        generator = torch.Generator(device=device)
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state) + 100_000 + member_index * 10_000 + epoch)
        return torch.randperm(n, generator=generator, device=device)

    def fit(
        self,
        _fit_target: Any | None = None,
        *,
        num_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int | None = None,
        shuffle: bool = True,
        verbose: bool = False,
        clip_grad_norm: float | None = None,
        optimizer_cls: type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """Fit each classifier member on an independent bootstrap sample."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this Deep Ensemble classifier.")
        if int(num_epochs) <= 0:
            raise ValueError("num_epochs must be positive.")
        if float(lr) <= 0:
            raise ValueError("lr must be positive.")
        if clip_grad_norm is not None and float(clip_grad_norm) <= 0:
            raise ValueError("clip_grad_norm must be positive when provided.")

        train_X, labels = self._training_tensors()
        n = int(train_X.shape[0])
        resolved_batch_size = n if batch_size is None else min(int(batch_size), n)
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be positive when provided.")

        self.fit_losses = []
        for member_index, member in enumerate(self.members):
            member.train()
            sample_indices = _bootstrap_classification_indices(
                labels,
                num_classes=self.num_classes,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                member_index=member_index,
            )
            member_X = train_X.index_select(0, sample_indices)
            member_Y = labels.index_select(0, sample_indices)

            options = dict(optimizer_kwargs or {})
            options.setdefault("lr", float(lr))
            optimizer = optimizer_cls(member.parameters(), **options)
            history: list[float] = []

            for epoch in range(int(num_epochs)):
                order = (
                    self._epoch_order(n, member_index, epoch, train_X.device)
                    if shuffle
                    else torch.arange(n, device=train_X.device)
                )
                epoch_loss = 0.0
                seen = 0
                for start in range(0, n, resolved_batch_size):
                    indices = order[start : start + resolved_batch_size]
                    batch_X = member_X.index_select(0, indices)
                    batch_Y = member_Y.index_select(0, indices)

                    optimizer.zero_grad(set_to_none=True)
                    logits = self._member_logits(member, batch_X)
                    if self.binary:
                        loss = F.binary_cross_entropy_with_logits(
                            logits.squeeze(-1),
                            batch_Y.to(dtype=logits.dtype),
                        )
                    else:
                        loss = F.cross_entropy(logits, batch_Y)
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(member.parameters(), float(clip_grad_norm))
                    optimizer.step()

                    count = int(indices.numel())
                    epoch_loss += float(loss.detach()) * count
                    seen += count
                history.append(epoch_loss / max(seen, 1))

            self.fit_losses.append(history)
            if verbose:
                print(
                    f"DeepEnsemble classifier member {member_index + 1}/{self.ensemble_size}: "
                    f"loss={history[-1]:.6g}"
                )

        self._is_fitted = True
        self.eval()
        return self

    def forward(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() first.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")
        encoded_X = self._encode_features(X)
        values = [self._member_probabilities(member, encoded_X) for member in self.members]
        return torch.stack(values, dim=-3)


class DeepEnsembleBinaryClassificationModel(_DeepEnsembleClassificationModel):
    """Binary Deep Ensemble classifier with epistemic probability samples."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, binary=True, num_classes=2, **kwargs)


class DeepEnsembleMulticlassClassificationModel(_DeepEnsembleClassificationModel):
    """Multiclass Deep Ensemble classifier with simplex-valued member samples."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            binary=False,
            num_classes=num_classes,
            **kwargs,
        )


class DeepEnsembleMixedBinaryClassificationModel(DeepEnsembleBinaryClassificationModel):
    """Binary Deep Ensemble classifier for mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        encoder = _MixedCategoricalEncoder(train_X=train_X, cat_dims=cat_dims, atol=categorical_atol)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            _feature_encoder=encoder,
            _encoded_input_dim=encoder.encoded_dim,
            **kwargs,
        )
        self.cat_dims = list(encoder.cat_dims)

    @property
    def categorical_values(self) -> dict[int, tuple[float, ...]]:
        encoder = self.feature_encoder
        if not isinstance(encoder, _MixedCategoricalEncoder):  # pragma: no cover
            raise RuntimeError("Mixed Deep Ensemble categorical encoder is unavailable.")
        return encoder.categorical_values


class DeepEnsembleMixedMulticlassClassificationModel(DeepEnsembleMulticlassClassificationModel):
    """Multiclass Deep Ensemble classifier for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        encoder = _MixedCategoricalEncoder(train_X=train_X, cat_dims=cat_dims, atol=categorical_atol)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            _feature_encoder=encoder,
            _encoded_input_dim=encoder.encoded_dim,
            **kwargs,
        )
        self.cat_dims = list(encoder.cat_dims)

    @property
    def categorical_values(self) -> dict[int, tuple[float, ...]]:
        encoder = self.feature_encoder
        if not isinstance(encoder, _MixedCategoricalEncoder):  # pragma: no cover
            raise RuntimeError("Mixed Deep Ensemble categorical encoder is unavailable.")
        return encoder.categorical_values


__all__ = [
    "DeepEnsembleBinaryClassificationModel",
    "DeepEnsembleMixedBinaryClassificationModel",
    "DeepEnsembleMixedMulticlassClassificationModel",
    "DeepEnsembleMulticlassClassificationModel",
]
