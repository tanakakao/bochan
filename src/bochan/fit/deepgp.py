from __future__ import annotations

from typing import Optional

import torch

from .common import (
    get_likelihood_from_mll_or_model,
    get_train_inputs_tensor,
    get_train_targets_tensor,
    maybe_clip_grad_norm,
    set_model_and_likelihood_eval_mode,
    set_model_and_likelihood_train_mode,
    view_single_output_target,
)


def fit_deepgp_mll(
    mll,
    *,
    lr: float = 0.01,
    num_epochs: Optional[int] = None,
    epoch: Optional[int] = None,
    optimizer_cls= torch.optim.Adam,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
    **ignore,
):
    """
    Fit a DeepGP approximate MLL.

    This preserves the original full-batch behavior:
        output = model(x_tensor)
        loss = -mll(output, target)

    Args:
        mll:
            DeepApproximateMLL / VariationalELBO-like MLL.
        num_epochs:
            Preferred epoch argument.
        epoch:
            Backward-compatible alias. Used only when `num_epochs` is None.

    Returns:
        The input `mll`.
    """
    if num_epochs is None:
        num_epochs = 100 if epoch is None else int(epoch)
    else:
        num_epochs = int(num_epochs)

    model = mll.model
    likelihood = get_likelihood_from_mll_or_model(mll, model)

    set_model_and_likelihood_train_mode(model, likelihood)
    if hasattr(mll, "train"):
        mll.train()

    optimizer = optimizer_cls(model.parameters(), lr=lr)

    train_X = get_train_inputs_tensor(model)
    train_Y = get_train_targets_tensor(model)
    target = view_single_output_target(train_Y)

    for i in range(num_epochs):
        optimizer.zero_grad()

        output = model(train_X)
        loss = -mll(output, target)

        if loss.ndim > 0:
            loss = loss.sum()

        loss.backward()
        maybe_clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()

        if verbose and ((i + 1) % 50 == 0 or i == 0 or i == num_epochs - 1):
            print(f"[fit_deepgp_mll] epoch={i + 1:04d} loss={float(loss.detach().item()):.6f}")

    set_model_and_likelihood_eval_mode(model, likelihood)
    if hasattr(mll, "eval"):
        mll.eval()

    return mll
