from __future__ import annotations

from typing import Optional

import torch

from ..common import (
    build_tensor_dataloader,
    get_train_inputs_tensor,
    get_train_targets_tensor,
    maybe_clip_grad_norm,
    move_batch_like,
    set_mll_eval_mode,
    set_mll_train_mode,
    squeeze_single_output_target,
)


def _is_deep_classification_mll(mll) -> bool:
    """Return whether the MLL wraps a DeepGP / DeepKernel classification model."""

    model = getattr(mll, "model", None)
    module_name = type(model).__module__.lower()
    class_name = type(model).__name__.lower()
    return (
        ".deep." in module_name
        or "deepgp" in module_name
        or "deepkernel" in module_name
        or "deepgp" in class_name
        or "deepkernel" in class_name
    )


def fit_binary_classifier_mll(
    mll,
    *,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    optimizer_cls=torch.optim.Adam,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
    **ignore,
):
    """
    Fit a variational binary / multi-output classification model from an MLL.

    Intended MLLs:
        - gpytorch.mlls.VariationalELBO
        - gpytorch.mlls.PredictiveLogLikelihood
        - gpytorch.mlls.DeepApproximateMLL

    Notes:
        - Standard variational classifiers keep the existing mini-batch behavior.
        - DeepGP / DeepKernel classifiers use the dedicated full-batch loop so
          their wrapper-specific transformed training inputs and MLL structure
          are handled consistently.
        - `mll.model.train_inputs` is used as X.
        - `mll.model.train_targets` is used as y.
        - Single-output targets shaped [n, 1] are squeezed to [n].
        - Returns the input `mll`, following BoTorch-style fit helpers.
    """
    if _is_deep_classification_mll(mll):
        from ..deep.common import fit_deep_full_batch_mll

        return fit_deep_full_batch_mll(
            mll,
            lr=lr,
            num_epochs=num_epochs,
            optimizer_cls=optimizer_cls,
            clip_grad_norm=clip_grad_norm,
            verbose=verbose,
            log_prefix="fit_binary_classifier_mll",
        )

    set_mll_train_mode(mll)

    model = mll.model
    train_X = get_train_inputs_tensor(model)
    train_Y = squeeze_single_output_target(get_train_targets_tensor(model))

    optimizer = optimizer_cls(mll.parameters(), lr=lr)
    loader = build_tensor_dataloader(
        train_X=train_X,
        train_Y=train_Y,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    num_epochs = int(num_epochs)
    num_data = int(train_X.shape[-2])

    for epoch in range(num_epochs):
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = move_batch_like(xb, yb, train_X=train_X, train_Y=train_Y)

            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)

            if loss.ndim > 0:
                loss = loss.sum()

            loss.backward()
            maybe_clip_grad_norm(mll.parameters(), clip_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().item()) * xb.shape[0]

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(f"[fit_classifier_mll] epoch={epoch + 1:04d} loss={total_loss / num_data:.6f}")

    set_mll_eval_mode(mll)
    return mll
