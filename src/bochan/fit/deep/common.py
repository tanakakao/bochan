from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import torch
from gpytorch.settings import cholesky_jitter
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from ..common import (
    get_likelihood_from_mll_or_model,
    get_train_inputs_tensor,
    get_train_targets_tensor,
    maybe_clip_grad_norm,
    set_model_and_likelihood_eval_mode,
    set_model_and_likelihood_train_mode,
    view_single_output_target,
)


def _get_deep_full_batch_train_X(model) -> Tensor:
    """Return the training input used by the internal deep model.

    Deep wrappers often keep raw ``train_inputs`` for user-facing APIs while
    also keeping ``transformed_train_inputs`` for the internal model. This is
    important for wrappers that apply ``input_transform`` manually in posterior
    paths but whose ``forward`` expects already-transformed inputs during MLL
    fitting.

    The fallback preserves the previous behavior for regression, binary,
    ordinal, multiclass, and deep-kernel models that do not expose transformed
    training inputs.
    """

    transformed_train_inputs = getattr(model, "transformed_train_inputs", None)
    if transformed_train_inputs is not None:
        if isinstance(transformed_train_inputs, tuple) and len(transformed_train_inputs) > 0:
            train_X = transformed_train_inputs[0]
        else:
            train_X = transformed_train_inputs

        if torch.is_tensor(train_X):
            return train_X

    return get_train_inputs_tensor(model)


def _deep_full_batch_loss(
    mll,
    model,
    train_X: Tensor,
    target: Tensor,
    *,
    psd_jitter_values: Sequence[float] | None,
) -> tuple[Tensor, float | None]:
    """Evaluate one full-batch loss with optional bounded PSD retries."""

    if psd_jitter_values is None:
        output = model(train_X)
        return -mll(output, target), None

    jitter_values = tuple(float(value) for value in psd_jitter_values)
    if not jitter_values or any(value <= 0 for value in jitter_values):
        raise ValueError("psd_jitter_values must contain positive values.")

    last_error: NotPSDError | None = None
    for jitter in jitter_values:
        try:
            context = cholesky_jitter(
                float_value=jitter,
                double_value=jitter,
                half_value=max(jitter, 1e-4),
            )
            with context:
                output = model(train_X)
                return -mll(output, target), jitter
        except NotPSDError as error:
            last_error = error

    if last_error is None:
        raise RuntimeError("Deep model PSD retry failed without an exception.")
    last_error.add_note(
        "Deep Kernel MLL remained non-PSD after bounded jitter retries: "
        f"{jitter_values}."
    )
    raise last_error


def fit_deep_full_batch_mll(
    mll,
    *,
    lr: float = 0.01,
    num_epochs: int | None = None,
    epoch: int | None = None,
    optimizer_cls=torch.optim.Adam,
    clip_grad_norm: float | None = None,
    psd_jitter_values: Sequence[float] | None = None,
    verbose: bool = False,
    log_prefix: str = "fit_deep_full_batch_mll",
    **ignore,
):
    """
    Fit a DeepGP / DeepKernel-style MLL with the existing full-batch loop.

    This helper is intentionally small and conservative. It preserves the
    previous behavior of the old ``fit_deepgp_mll`` and ``fit_deepkernel_mll``
    implementations. Deep Kernel callers may additionally supply a bounded
    jitter schedule for numerically difficult exact-GP covariance matrices.

    Args:
        mll:
            DeepApproximateMLL / VariationalELBO-like MLL.
        num_epochs:
            Preferred epoch argument.
        epoch:
            Backward-supported alias. Used only when ``num_epochs`` is None.
        psd_jitter_values:
            Optional ascending diagonal jitter values. Each epoch is retried
            only when Cholesky factorization raises ``NotPSDError``.
        log_prefix:
            Name used in verbose logs.

    Returns:
        The input ``mll``.
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

    train_X = _get_deep_full_batch_train_X(model)
    train_Y = get_train_targets_tensor(model)
    target = view_single_output_target(train_Y)

    for i in range(num_epochs):
        optimizer.zero_grad()

        loss, used_jitter = _deep_full_batch_loss(
            mll,
            model,
            train_X,
            target,
            psd_jitter_values=psd_jitter_values,
        )

        if loss.ndim > 0:
            loss = loss.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{log_prefix} produced a non-finite loss at epoch {i + 1}."
            )

        loss.backward()
        maybe_clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()

        if verbose and ((i + 1) % 50 == 0 or i == 0 or i == num_epochs - 1):
            jitter_text = "default" if used_jitter is None else f"{used_jitter:.1e}"
            print(
                f"[{log_prefix}] epoch={i + 1:04d} "
                f"loss={float(loss.detach().item()):.6f} jitter={jitter_text}"
            )

    set_model_and_likelihood_eval_mode(model, likelihood)
    if hasattr(mll, "eval"):
        mll.eval()

    return mll
