from __future__ import annotations

import pytest

from bochan.api import AcquisitionConfig


@pytest.mark.parametrize(
    "name",
    ["ucb", "qUCB", "upper_confidence_bound", "qUpperConfidenceBound"],
)
def test_ucb_uses_default_beta(name: str) -> None:
    config = AcquisitionConfig(name=name)

    assert config.acqf_kwargs["beta"] == 3.0


def test_explicit_ucb_beta_takes_priority() -> None:
    config = AcquisitionConfig(name="ucb", acqf_kwargs={"beta": 0.25})

    assert config.acqf_kwargs["beta"] == 0.25


def test_non_ucb_does_not_receive_beta() -> None:
    config = AcquisitionConfig(name="ei")

    assert "beta" not in config.acqf_kwargs
