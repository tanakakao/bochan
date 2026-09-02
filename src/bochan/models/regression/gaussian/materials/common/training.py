"""Shared encoder training-policy utilities for material models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn

EncoderTrainingMode = Literal["frozen", "partial", "full"]


@dataclass(frozen=True)
class EncoderTrainingPolicy:
    """Describe how a material encoder participates in model training.

    ``partial`` keeps the encoder in evaluation mode while selected submodules
    follow the parent model's train/eval state. ``full`` lets the complete
    encoder follow the parent model state. ``frozen`` keeps every encoder
    parameter non-trainable and the encoder in evaluation mode.
    """

    mode: EncoderTrainingMode = "frozen"
    trainable_modules: tuple[nn.Module, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"frozen", "partial", "full"}:
            raise ValueError("mode must be 'frozen', 'partial', or 'full'.")
        if self.mode == "partial" and not self.trainable_modules:
            raise ValueError("Partial encoder training requires at least one trainable module.")
        if self.mode != "partial" and self.trainable_modules:
            raise ValueError("Only partial encoder training accepts trainable_modules.")
        if any(not isinstance(module, nn.Module) for module in self.trainable_modules):
            raise TypeError("trainable_modules must contain torch.nn.Module instances.")


def unique_module_parameters(modules: tuple[nn.Module, ...]) -> tuple[nn.Parameter, ...]:
    """Return parameters from modules once while preserving traversal order."""

    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if not isinstance(module, nn.Module):
            raise TypeError("modules must contain torch.nn.Module instances.")
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return tuple(parameters)


def configure_encoder_parameters(
    encoder: nn.Module,
    policy: EncoderTrainingPolicy,
) -> None:
    """Apply one frozen/partial/full ``requires_grad`` policy to an encoder."""

    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module.")
    if not isinstance(policy, EncoderTrainingPolicy):
        raise TypeError("policy must be an EncoderTrainingPolicy.")

    for parameter in encoder.parameters():
        parameter.requires_grad_(policy.mode == "full")

    if policy.mode == "partial":
        selected_parameters = unique_module_parameters(policy.trainable_modules)
        if not selected_parameters:
            raise ValueError("Selected partial-training modules expose no parameters.")
        for parameter in selected_parameters:
            parameter.requires_grad_(True)


def apply_encoder_train_mode(
    encoder: nn.Module,
    policy: EncoderTrainingPolicy,
    mode: bool,
) -> None:
    """Apply train/eval state without activating frozen encoder components."""

    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module.")
    if not isinstance(policy, EncoderTrainingPolicy):
        raise TypeError("policy must be an EncoderTrainingPolicy.")

    if policy.mode == "full":
        encoder.train(mode)
        return

    encoder.eval()
    if policy.mode == "partial":
        for module in policy.trainable_modules:
            module.train(mode)


def apply_encoder_training_policy(
    encoder: nn.Module,
    policy: EncoderTrainingPolicy,
    *,
    training: bool,
) -> None:
    """Apply both parameter and module-mode portions of an encoder policy."""

    configure_encoder_parameters(encoder, policy)
    apply_encoder_train_mode(encoder, policy, training)


__all__ = [
    "EncoderTrainingMode",
    "EncoderTrainingPolicy",
    "apply_encoder_train_mode",
    "apply_encoder_training_policy",
    "configure_encoder_parameters",
    "unique_module_parameters",
]
