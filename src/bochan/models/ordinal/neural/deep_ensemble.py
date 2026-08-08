"""BoTorch-native Deep Ensemble models for ordinal regression."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.ensemble import EnsembleModel
from torch import Tensor, nn

from bochan.fit.ordinal import fit_ordinal_mll
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.regression.neural.deep_ensemble import (
    _DenseRegressor,
    _MixedCategoricalEncoder,
)
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior
from bochan.posteriors.ordinal_ensemble import OrdinalEnsemblePosterior


def _require_ordinal_targets(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    num_classes: int | None,
) -> tuple[Tensor, int]:
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if not train_X.is_floating_point():
        raise TypeError("train_X must be a floating-point tensor.")

    labels = torch.as_tensor(train_Y, device=train_X.device)
    if labels.ndim == 2 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    if labels.ndim != 1:
        raise ValueError("Ordinal Deep Ensemble targets must have shape [n] or [n, 1].")
    if labels.shape[0] != train_X.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if labels.is_floating_point():
        if not torch.isfinite(labels).all():
            raise ValueError("Ordinal labels must be finite.")
        rounded = labels.round()
        if not torch.allclose(labels, rounded):
            raise ValueError("Ordinal labels must be integer-valued.")
        labels = rounded
    labels = labels.long()
    if (labels < 0).any():
        raise ValueError("Ordinal labels must be non-negative integers.")

    inferred = int(labels.max().item()) + 1 if labels.numel() else 0
    resolved = inferred if num_classes is None else int(num_classes)
    if resolved < 3:
        raise ValueError("Ordinal Deep Ensemble requires at least 3 classes.")
    if inferred > resolved:
        raise ValueError(f"train_Y contains labels outside num_classes={resolved}.")
    observed = torch.unique(labels).cpu()
    expected = torch.arange(resolved)
    if not torch.equal(observed, expected):
        raise ValueError(
            f"Ordinal labels must cover contiguous values 0..{resolved - 1}; "
            f"observed {observed.tolist()}."
        )
    return labels.unsqueeze(-1), resolved


def _make_generator(
    *,
    device: torch.device,
    seed: int | None,
) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


class _DeepEnsembleOrdinalObjective(nn.Module):
    """Log-likelihood adapter consumed by bochan's standard ordinal fit helper."""

    def __init__(self, model: DeepEnsembleOrdinalModel) -> None:
        super().__init__()
        self.model = model
        self.likelihood = model.likelihood
        self._calls = 0

    def forward(self, latent_values: Tensor, target: Tensor) -> Tensor:
        labels = target.squeeze(-1).long()
        latent = latent_values.squeeze(-1)
        if latent.ndim != 2:
            raise RuntimeError(
                "Ordinal Deep Ensemble training expects latent values with shape [ensemble, batch]."
            )

        member_log_likelihoods = []
        batch_size = int(labels.shape[0])
        for member_index in range(int(latent.shape[0])):
            if self.model.bootstrap:
                seed = (
                    None
                    if self.model.random_state is None
                    else int(self.model.random_state)
                    + 10_000 * member_index
                    + self._calls
                )
                generator = _make_generator(device=labels.device, seed=seed)
                indices = torch.randint(
                    batch_size,
                    size=(batch_size,),
                    generator=generator,
                    device=labels.device,
                )
                member_latent = latent[member_index].index_select(0, indices)
                member_labels = labels.index_select(0, indices)
            else:
                member_latent = latent[member_index]
                member_labels = labels

            probs = self.likelihood.class_probs_from_f(member_latent).clamp_min(
                self.likelihood.eps
            )
            selected = probs.gather(
                dim=-1,
                index=member_labels.unsqueeze(-1),
            ).squeeze(-1)
            member_log_likelihoods.append(selected.log().mean())

        self._calls += 1
        self.model._is_fitted = True
        return torch.stack(member_log_likelihoods).mean()


class DeepEnsembleOrdinalModel(EnsembleModel):
    """Ordinal Deep Ensemble with shared ordered-logit cutpoints.

    Each neural member predicts one latent ordinal score. The finite latent
    predictions are exposed through :class:`OrdinalEnsemblePosterior`. A shared
    ``OrdinalLogitLikelihood`` maps latent scores to ordered class probabilities,
    matching bochan's existing ordinal GP API.

    Existing ordinal acquisition functions already marginalize the posterior
    internally before their ``average_over_ensemble_models`` decorator runs.
    Therefore this wrapper disables BoTorch's outer ensemble marker while still
    returning a finite ``EnsemblePosterior``. This prevents a second reduction
    over the acquisition output's last dimension.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int | None = None,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: nn.Module | None = None,
        members: Sequence[nn.Module] | None = None,
        member_factory: Callable[[int], nn.Module] | None = None,
        weights: Tensor | None = None,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        _feature_encoder: nn.Module | None = None,
        _encoded_input_dim: int | None = None,
    ) -> None:
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        train_Y, resolved_classes = _require_ordinal_targets(
            train_X,
            train_Y,
            num_classes=num_classes,
        )
        if weights is not None:
            weights = weights.to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(weights=weights)
        self._is_ensemble = False

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform

        self.num_classes = int(resolved_classes)
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.feature_encoder = _feature_encoder if _feature_encoder is not None else nn.Identity()
        encoded_input_dim = int(_encoded_input_dim or train_X.shape[-1])

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
        self.likelihood = OrdinalLogitLikelihood(
            num_classes=self.num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        ).to(dtype=train_X.dtype, device=train_X.device)
        self._num_outputs = 1
        self._is_fitted = False

    @property
    def ordinal_likelihood(self) -> OrdinalLogitLikelihood:
        return self.likelihood

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self.train_X,)

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        return (self.train_X,)

    @property
    def train_targets(self) -> Tensor:
        return self.train_Y.squeeze(-1)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _set_transformed_inputs(self) -> None:
        """Keep stored training inputs in the raw search space."""

    def _revert_to_original_inputs(self) -> None:
        """Keep stored training inputs in the raw search space."""

    def _build_member(
        self,
        *,
        index: int,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        member_factory: Callable[[int], nn.Module] | None,
    ) -> nn.Module:
        def build() -> nn.Module:
            if member_factory is not None:
                member = member_factory(input_dim)
                if not isinstance(member, nn.Module):
                    raise TypeError("member_factory must return a torch.nn.Module.")
                return member
            return _DenseRegressor(
                input_dim=input_dim,
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

    @staticmethod
    def _member_prediction(member: nn.Module, X: Tensor) -> Tensor:
        value = member(X)
        if value.shape == X.shape[:-1]:
            value = value.unsqueeze(-1)
        if value.shape[:-1] != X.shape[:-1] or value.shape[-1] != 1:
            raise RuntimeError("Each ordinal Deep Ensemble member must return shape [..., 1].")
        return value

    def _forward_transformed(self, X: Tensor) -> Tensor:
        encoded_X = self._encode_features(X)
        values = [self._member_prediction(member, encoded_X) for member in self.members]
        return torch.stack(values, dim=-3)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")
        transformed_X = self.transform_inputs(X)
        return self._forward_transformed(transformed_X)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **_: Any,
    ) -> OrdinalEnsemblePosterior:
        if output_indices is not None and list(output_indices) != [0]:
            raise UnsupportedError(f"{type(self).__name__} exposes only output index 0.")
        if observation_noise is not False:
            raise UnsupportedError("Ordinal Deep Ensemble does not support observation_noise.")

        self.eval()
        transformed_X = self.transform_inputs(X)
        posterior = OrdinalEnsemblePosterior(
            values=self._forward_transformed(transformed_X),
            weights=getattr(self, "ensemble_weights", None),
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def latent_posterior(self, X: Tensor, **kwargs: Any) -> OrdinalEnsemblePosterior:
        return self.posterior(X, **kwargs)

    def probability_posterior(self, X: Tensor) -> ClassificationEnsemblePosterior:
        latent = self.posterior(X)
        probabilities = self.likelihood.class_probs_from_f(latent.values.squeeze(-1))
        return ClassificationEnsemblePosterior(
            values=probabilities,
            weights=getattr(self, "ensemble_weights", None),
        )

    def epistemic_probability_posterior(self, X: Tensor) -> ClassificationEnsemblePosterior:
        return self.probability_posterior(X)

    def class_probs_from_posterior(self, posterior: OrdinalEnsemblePosterior) -> Tensor:
        probabilities = self.likelihood.class_probs_from_f(posterior.values.squeeze(-1))
        weights = posterior.weights.to(device=probabilities.device, dtype=probabilities.dtype)
        shape = [1] * probabilities.ndim
        shape[-3] = int(weights.numel())
        return (weights.view(*shape) * probabilities).sum(dim=-3)

    def class_probs(self, X: Tensor) -> Tensor:
        return self.probability_posterior(X).mean

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)

    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        probabilities = self.class_probs(X)
        utilities = utilities.to(device=probabilities.device, dtype=probabilities.dtype)
        if utilities.numel() != self.num_classes:
            raise ValueError(f"utilities must contain {self.num_classes} values.")
        return (probabilities * utilities).sum(dim=-1)

    def make_mll(self, **_: Any) -> _DeepEnsembleOrdinalObjective:
        return _DeepEnsembleOrdinalObjective(self)

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
        **kwargs: Any,
    ) -> Self:
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this ordinal Deep Ensemble model.")
        objective = self.make_mll()
        fit_ordinal_mll(
            objective,
            fit_model=self,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=verbose,
            clip_grad_norm=clip_grad_norm,
            **kwargs,
        )
        self._is_fitted = True
        self.eval()
        return self


class DeepEnsembleMixedOrdinalModel(DeepEnsembleOrdinalModel):
    """Ordinal Deep Ensemble with Torch-native one-hot categorical encoding."""

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
    "DeepEnsembleMixedOrdinalModel",
    "DeepEnsembleOrdinalModel",
]
