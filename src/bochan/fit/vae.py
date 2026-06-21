from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.fit.common import (
    maybe_clip_grad_norm,
    set_model_and_likelihood_eval_mode,
    set_model_and_likelihood_train_mode,
)
from bochan.models.regression.gaussian.high_dim.vae import VAESingleTaskGP


@dataclass
class VAEFitResult:
    """Training diagnostics returned by :func:`fit_vae_gp`."""

    model: VAESingleTaskGP
    gp_mll: ExactMarginalLogLikelihood
    loss_history: list[float]
    gp_loss_history: list[float]
    reconstruction_loss_history: list[float]
    kl_loss_history: list[float]

    @property
    def final_loss(self) -> float:
        """Return the last total loss."""
        return self.loss_history[-1]


def fit_vae_gp(
    model: VAESingleTaskGP,
    *,
    lr: float = 1e-2,
    num_epochs: int | None = None,
    epoch: int | None = None,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: dict[str, Any] | None = None,
    clip_grad_norm: float | None = None,
    verbose: bool = False,
    log_interval: int = 50,
    **_: Any,
) -> VAEFitResult:
    """Jointly fit the VAE encoder/decoder and latent-space exact GP.

    The GP uses the encoder mean, while reconstruction uses a reparameterized
    latent sample. Training is full-batch because the exact GP marginal
    likelihood requires all observations in each optimization step.
    """
    if not isinstance(model, VAESingleTaskGP):
        raise TypeError("fit_vae_gp expects a VAESingleTaskGP instance.")
    if num_epochs is None:
        num_epochs = 300 if epoch is None else int(epoch)
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive.")

    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer = optimizer_cls(model.parameters(), lr=float(lr), **optimizer_kwargs)
    gp_mll = model.make_gp_mll()

    set_model_and_likelihood_train_mode(model, model.likelihood)
    gp_mll.train()

    loss_history: list[float] = []
    gp_loss_history: list[float] = []
    reconstruction_loss_history: list[float] = []
    kl_loss_history: list[float] = []

    for i in range(int(num_epochs)):
        optimizer.zero_grad()
        components = model.joint_loss_components(gp_mll)
        loss = components["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite VAE-GP loss encountered. Consider reducing lr, "
                "lowering kl_weight, or normalizing train_X."
            )
        loss.backward()
        maybe_clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()

        loss_history.append(float(loss.detach().item()))
        gp_loss_history.append(float(components["gp_loss"].detach().item()))
        reconstruction_loss_history.append(
            float(components["reconstruction_loss"].detach().item())
        )
        kl_loss_history.append(float(components["kl_loss"].detach().item()))

        if verbose and (
            i == 0
            or i == num_epochs - 1
            or (i + 1) % log_interval == 0
        ):
            print(
                f"[fit_vae_gp] epoch={i + 1:04d} "
                f"loss={loss_history[-1]:.6f} gp={gp_loss_history[-1]:.6f} "
                f"recon={reconstruction_loss_history[-1]:.6f} "
                f"kl={kl_loss_history[-1]:.6f}"
            )

    model.refresh_latent_train_inputs()
    set_model_and_likelihood_eval_mode(model, model.likelihood)
    gp_mll.eval()

    return VAEFitResult(
        model=model,
        gp_mll=gp_mll,
        loss_history=loss_history,
        gp_loss_history=gp_loss_history,
        reconstruction_loss_history=reconstruction_loss_history,
        kl_loss_history=kl_loss_history,
    )


__all__ = ["VAEFitResult", "fit_vae_gp"]
