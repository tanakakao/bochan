import torch
from typing import Any, Optional
from gpytorch.mlls import VariationalELBO


def fit_beta_mll(
    mll: VariationalELBO,
    lr: float = 1e-2,
    num_epochs: int = 200,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    verbose: bool = False,
) -> None:
    latent_model = mll.model
    likelihood = mll.likelihood

    latent_model.train()
    likelihood.train()

    if not hasattr(latent_model, "train_inputs") or not hasattr(latent_model, "train_targets"):
        raise AttributeError(
            "mll.model must have train_inputs and train_targets. "
            "Create mll via make_beta_mll(model) or VariationalELBO(..., model=model.model, ...)."
        )

    x_all = latent_model.train_inputs[0]
    y_all = latent_model.train_targets

    n = x_all.shape[-2]
    if batch_size is None or batch_size >= n:
        batch_size = n

    params = list(latent_model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(num_epochs):
        if shuffle:
            perm = torch.randperm(n, device=x_all.device)
            x_epoch = x_all.index_select(dim=-2, index=perm)
            y_epoch = y_all.index_select(dim=-1, index=perm)
        else:
            x_epoch = x_all
            y_epoch = y_all

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            xb = x_epoch[..., start:end, :]
            yb = y_epoch[..., start:end]

            optimizer.zero_grad()
            output = latent_model(xb)
            loss = -mll(output, yb).mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach())
            n_batches += 1

        if verbose and ((epoch == 0) or ((epoch + 1) % 20 == 0)):
            scale = float(likelihood.scale.detach().mean().cpu())
            print(
                f"[BetaGP] epoch={epoch+1:4d} "
                f"loss={total_loss / max(n_batches, 1):.6f} "
                f"scale={scale:.6f}"
            )
