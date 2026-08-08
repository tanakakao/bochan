"""BoTorch-native Deep Ensemble surrogate models for regression."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
import torch.nn.functional as F
from botorch.models.ensemble import EnsembleModel
from torch import Tensor, nn
from torch.optim import Optimizer


class _DenseRegressor(nn.Module):
    """Small configurable MLP used as the default Deep Ensemble member."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
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
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        return self.network(X)


def _make_activation(name: str) -> nn.Module:
    normalized = str(name).lower().replace("-", "_")
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError("activation must be one of: relu, gelu, silu/swish, tanh.")


def _require_regression_data(train_X: Tensor, train_Y: Tensor) -> Tensor:
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if not train_X.is_floating_point():
        raise TypeError("train_X must be a floating-point tensor.")
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError("Deep Ensemble currently supports single-output regression only.")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of observations.")
    if not train_Y.is_floating_point():
        raise TypeError("train_Y must be a floating-point tensor.")
    if train_X.shape[0] == 0:
        raise ValueError("Deep Ensemble requires at least one training observation.")
    return train_Y


class _MixedCategoricalEncoder(nn.Module):
    """Torch-native one-hot encoder that preserves gradients of continuous inputs."""

    def __init__(
        self,
        train_X: Tensor,
        cat_dims: Sequence[int],
        *,
        atol: float = 1e-8,
    ) -> None:
        super().__init__()
        if atol < 0:
            raise ValueError("categorical_atol must be non-negative.")
        dims = sorted(int(dim) for dim in cat_dims)
        if not dims:
            raise ValueError("cat_dims must contain at least one categorical dimension.")
        if len(set(dims)) != len(dims):
            raise ValueError("cat_dims must not contain duplicate dimensions.")
        if any(dim < 0 or dim >= train_X.shape[-1] for dim in dims):
            raise ValueError(
                f"cat_dims must be within [0, {train_X.shape[-1] - 1}], got {dims}."
            )

        self.input_dim = int(train_X.shape[-1])
        self.cat_dims = tuple(dims)
        self.atol = float(atol)
        self._category_buffer_names: dict[int, str] = {}
        for dim in self.cat_dims:
            values = train_X[:, dim].detach()
            if not torch.isfinite(values).all():
                raise ValueError(f"Categorical dimension {dim} contains non-finite values.")
            categories = torch.unique(values).sort().values
            buffer_name = f"categories_{dim}"
            self.register_buffer(buffer_name, categories)
            self._category_buffer_names[dim] = buffer_name

    @property
    def encoded_dim(self) -> int:
        continuous = self.input_dim - len(self.cat_dims)
        categorical = sum(self._categories(dim).numel() for dim in self.cat_dims)
        return continuous + categorical

    def _categories(self, dim: int) -> Tensor:
        return getattr(self, self._category_buffer_names[dim])

    @property
    def categorical_values(self) -> dict[int, tuple[float, ...]]:
        return {
            dim: tuple(float(value) for value in self._categories(dim).detach().cpu().tolist())
            for dim in self.cat_dims
        }

    def forward(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected mixed inputs with {self.input_dim} features, got {X.shape[-1]}."
            )

        parts: list[Tensor] = []
        for dim in range(self.input_dim):
            column = X[..., dim : dim + 1]
            if dim not in self._category_buffer_names:
                parts.append(column)
                continue

            categories = self._categories(dim).to(dtype=X.dtype, device=X.device)
            view_shape = [1] * (column.ndim - 1) + [categories.numel()]
            matches = torch.isclose(
                column,
                categories.reshape(view_shape),
                rtol=0.0,
                atol=self.atol,
            )
            match_count = matches.sum(dim=-1)
            if not torch.all(match_count == 1):
                invalid = column.squeeze(-1)[match_count != 1].detach().cpu()
                bad_values = torch.unique(invalid).tolist()
                known_values = categories.detach().cpu().tolist()
                raise ValueError(
                    f"Categorical dimension {dim} contains values not observed during training: "
                    f"{bad_values}. Known values are {known_values}."
                )
            parts.append(matches.to(dtype=X.dtype))
        return torch.cat(parts, dim=-1)


class DeepEnsembleRegressorModel(EnsembleModel):
    """Differentiable Deep Ensemble surrogate exposed through BoTorch's ensemble API.

    Each member is an independently initialized neural regressor. By default each
    member is trained on one bootstrap resample of the training data. The public
    ``posterior`` is BoTorch's ``EnsemblePosterior`` and therefore uses member-to-
    member prediction disagreement as epistemic uncertainty.

    The model deliberately keeps observation noise out of the ensemble posterior,
    matching the latent-function semantics used by BoTorch acquisitions. This also
    keeps the uncertainty meaning aligned with bochan's Random Forest and bootstrap
    NGBoost ensemble models.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        bootstrap: bool = True,
        random_state: int | None = None,
        input_transform: nn.Module | None = None,
        outcome_transform: nn.Module | None = None,
        members: Sequence[nn.Module] | None = None,
        member_factory: Callable[[int], nn.Module] | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        weights: Tensor | None = None,
        _feature_encoder: nn.Module | None = None,
        _encoded_input_dim: int | None = None,
    ) -> None:
        train_Y = _require_regression_data(train_X, train_Y)
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        if weights is not None:
            weights = weights.to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(weights=weights)

        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self._num_outputs = 1
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.loss_fn = loss_fn
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
        self._is_fitted = False
        self.fit_losses: list[list[float]] = []

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

    def _set_transformed_inputs(self) -> None:
        """Keep stored training inputs in the public raw search space."""

    def _revert_to_original_inputs(self) -> None:
        """Keep stored training inputs in the public raw search space."""

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

    def make_mll(self, **_: Any) -> None:
        """Deep Ensembles use their direct neural-network training objective, not an MLL."""
        return None

    def _encode_features(self, X: Tensor) -> Tensor:
        return self.feature_encoder(X)

    @staticmethod
    def _member_prediction(member: nn.Module, X: Tensor) -> Tensor:
        prediction = member(X)
        if prediction.shape == X.shape[:-1]:
            prediction = prediction.unsqueeze(-1)
        if prediction.shape[:-1] != X.shape[:-1] or prediction.shape[-1] != 1:
            raise RuntimeError(
                "Each Deep Ensemble member must return shape [..., 1] for inputs [..., d]."
            )
        return prediction

    def _training_tensors(self) -> tuple[Tensor, Tensor]:
        self.train()
        transformed_X = self.transform_inputs(self.train_X)
        encoded_X = self._encode_features(transformed_X)
        transformed_Y = self.train_Y
        outcome_transform = getattr(self, "outcome_transform", None)
        if outcome_transform is not None:
            transformed_Y, _ = outcome_transform(transformed_Y, X=self.train_X)
        return encoded_X, transformed_Y

    def _bootstrap_indices(self, n: int, member_index: int, device: torch.device) -> Tensor:
        if not self.bootstrap:
            return torch.arange(n, device=device)
        if self.random_state is None:
            return torch.randint(n, size=(n,), device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.random_state) + 10_000 + member_index)
        return torch.randint(n, size=(n,), generator=generator, device=device)

    def _epoch_order(self, n: int, member_index: int, epoch: int, device: torch.device) -> Tensor:
        if self.random_state is None:
            return torch.randperm(n, device=device)
        generator = torch.Generator(device=device)
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
        """Fit every ensemble member using differentiable PyTorch training."""
        if _fit_target is not None and _fit_target is not self:
            raise TypeError("The optional fit target must be this Deep Ensemble model instance.")
        if int(num_epochs) <= 0:
            raise ValueError("num_epochs must be positive.")
        if float(lr) <= 0:
            raise ValueError("lr must be positive.")
        if clip_grad_norm is not None and float(clip_grad_norm) <= 0:
            raise ValueError("clip_grad_norm must be positive when provided.")

        train_X, train_Y = self._training_tensors()
        n = int(train_X.shape[0])
        resolved_batch_size = n if batch_size is None else min(int(batch_size), n)
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be positive when provided.")

        self.fit_losses = []
        for member_index, member in enumerate(self.members):
            member.train()
            sample_indices = self._bootstrap_indices(n, member_index, train_X.device)
            member_X = train_X.index_select(0, sample_indices)
            member_Y = train_Y.index_select(0, sample_indices)

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
                    prediction = self._member_prediction(member, batch_X)
                    loss = self.loss_fn(prediction, batch_Y) if self.loss_fn is not None else F.mse_loss(
                        prediction,
                        batch_Y,
                    )
                    if loss.ndim:
                        loss = loss.mean()
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
                    f"DeepEnsemble member {member_index + 1}/{self.ensemble_size}: "
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
        values = [self._member_prediction(member, encoded_X) for member in self.members]
        return torch.stack(values, dim=-3)


class DeepEnsembleMixedRegressorModel(DeepEnsembleRegressorModel):
    """Deep Ensemble regression with internal one-hot encoding of categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        categorical_atol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        encoder = _MixedCategoricalEncoder(
            train_X=train_X,
            cat_dims=cat_dims,
            atol=categorical_atol,
        )
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
        if not isinstance(encoder, _MixedCategoricalEncoder):  # pragma: no cover - constructor invariant
            raise RuntimeError("Mixed Deep Ensemble categorical encoder is unavailable.")
        return encoder.categorical_values


__all__ = [
    "DeepEnsembleMixedRegressorModel",
    "DeepEnsembleRegressorModel",
]
