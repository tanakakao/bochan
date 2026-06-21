from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["VAEProjector"]


def _make_activation(name: str) -> nn.Module:
    """Return an activation module from a compact public string key."""
    key = name.lower()
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if key not in activations:
        raise ValueError(f"Unknown activation={name!r}. Expected one of {sorted(activations)}.")
    return activations[key]()


def _make_output_activation(name: str) -> nn.Module:
    """Return the decoder output activation."""
    key = name.lower()
    if key in {"identity", "none", "linear"}:
        return nn.Identity()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError("decoder_output_activation must be 'identity', 'sigmoid', or 'tanh'.")


class VAEProjector(nn.Module):
    """Tabular variational autoencoder used as a trainable BO projection.

    The encoder returns a diagonal Gaussian distribution in latent space. The
    deterministic latent mean is used by the GP and acquisition functions,
    while a reparameterized sample is used by the reconstruction objective.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "silu",
        decoder_output_activation: str = "identity",
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if latent_dim <= 0 or latent_dim > input_dim:
            raise ValueError("latent_dim must satisfy 1 <= latent_dim <= input_dim.")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max.")

        widths = (
            list(hidden_dims)
            if hidden_dims is not None
            else [max(16, input_dim * 2), max(8, input_dim)]
        )
        if any(width <= 0 for width in widths):
            raise ValueError("hidden_dims must contain only positive integers.")

        encoder_layers: list[nn.Module] = []
        current_dim = input_dim
        for width in widths:
            encoder_layers.extend(
                [nn.Linear(current_dim, int(width)), _make_activation(activation)]
            )
            current_dim = int(width)
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        self.mu_layer = nn.Linear(current_dim, latent_dim)
        self.logvar_layer = nn.Linear(current_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        current_dim = latent_dim
        for width in reversed(widths):
            decoder_layers.extend(
                [nn.Linear(current_dim, int(width)), _make_activation(activation)]
            )
            current_dim = int(width)
        decoder_layers.extend(
            [
                nn.Linear(current_dim, input_dim),
                _make_output_activation(decoder_output_activation),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = widths
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def encode(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return latent mean and log variance for ``X``."""
        hidden = self.encoder(X)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Draw a differentiable sample from a diagonal Gaussian."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, Z: Tensor) -> Tensor:
        """Decode latent values into the VAE input space."""
        return self.decoder(Z)

    def transform(self, X: Tensor) -> Tensor:
        """Return the deterministic latent mean used by the GP."""
        mu, _ = self.encode(X)
        return mu

    def inverse_transform(self, Z: Tensor) -> Tensor:
        """Decode latent values into the VAE input space."""
        return self.decode(Z)

    def forward(
        self,
        X: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(reconstruction, mu, logvar, latent)``."""
        mu, logvar = self.encode(X)
        latent = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(latent), mu, logvar, latent
