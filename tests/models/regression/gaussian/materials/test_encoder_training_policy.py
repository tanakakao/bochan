"""Contract tests for shared material-encoder training policies."""

from __future__ import annotations

import pytest
from torch import nn

from bochan.models.regression.gaussian.materials.common import (
    EncoderTrainingPolicy,
    apply_encoder_train_mode,
    apply_encoder_training_policy,
    configure_encoder_parameters,
    unique_module_parameters,
)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.tail = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)


def _all_requires_grad(module: nn.Module) -> list[bool]:
    return [parameter.requires_grad for parameter in module.parameters()]


def test_frozen_policy_freezes_parameters_and_keeps_encoder_eval() -> None:
    encoder = _Encoder()
    encoder.train()

    apply_encoder_training_policy(
        encoder,
        EncoderTrainingPolicy(mode="frozen"),
        training=True,
    )

    assert not encoder.training
    assert not any(_all_requires_grad(encoder))


def test_full_policy_trains_complete_encoder() -> None:
    encoder = _Encoder()

    apply_encoder_training_policy(
        encoder,
        EncoderTrainingPolicy(mode="full"),
        training=True,
    )

    assert encoder.training
    assert all(_all_requires_grad(encoder))

    apply_encoder_train_mode(encoder, EncoderTrainingPolicy(mode="full"), False)
    assert not encoder.training


def test_partial_policy_only_unfreezes_selected_modules() -> None:
    encoder = _Encoder()
    policy = EncoderTrainingPolicy(mode="partial", trainable_modules=(encoder.tail,))

    apply_encoder_training_policy(encoder, policy, training=True)

    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.prefix.parameters())
    assert all(parameter.requires_grad for parameter in encoder.tail.parameters())
    assert not any(parameter.requires_grad for parameter in encoder.head.parameters())
    assert encoder.tail.training
    assert not encoder.prefix.training
    assert not encoder.head.training

    apply_encoder_train_mode(encoder, policy, False)
    assert not encoder.tail.training


def test_reconfiguring_from_full_to_frozen_clears_trainable_parameters() -> None:
    encoder = _Encoder()

    configure_encoder_parameters(encoder, EncoderTrainingPolicy(mode="full"))
    assert all(_all_requires_grad(encoder))

    configure_encoder_parameters(encoder, EncoderTrainingPolicy(mode="frozen"))
    assert not any(_all_requires_grad(encoder))


def test_reconfiguring_partial_selection_does_not_leave_stale_parameters() -> None:
    encoder = _Encoder()
    first = EncoderTrainingPolicy(mode="partial", trainable_modules=(encoder.tail,))
    second = EncoderTrainingPolicy(mode="partial", trainable_modules=(encoder.head,))

    configure_encoder_parameters(encoder, first)
    assert all(parameter.requires_grad for parameter in encoder.tail.parameters())

    configure_encoder_parameters(encoder, second)
    assert not any(parameter.requires_grad for parameter in encoder.tail.parameters())
    assert all(parameter.requires_grad for parameter in encoder.head.parameters())


def test_unique_module_parameters_deduplicates_shared_parameters() -> None:
    shared = nn.Linear(2, 2)
    wrapper = nn.Sequential(shared, nn.ReLU())

    parameters = unique_module_parameters((shared, wrapper))

    assert len(parameters) == len(tuple(shared.parameters()))
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in shared.parameters()
    }


def test_partial_policy_requires_modules() -> None:
    with pytest.raises(ValueError, match="requires at least one trainable module"):
        EncoderTrainingPolicy(mode="partial")


def test_frozen_and_full_policies_reject_partial_modules() -> None:
    module = nn.Linear(1, 1)

    for mode in ("frozen", "full"):
        with pytest.raises(ValueError, match="Only partial encoder training"):
            EncoderTrainingPolicy(mode=mode, trainable_modules=(module,))


def test_partial_modules_must_expose_parameters_when_configured() -> None:
    encoder = _Encoder()
    empty_module = nn.ReLU()
    policy = EncoderTrainingPolicy(mode="partial", trainable_modules=(empty_module,))

    with pytest.raises(ValueError, match="expose no parameters"):
        configure_encoder_parameters(encoder, policy)
