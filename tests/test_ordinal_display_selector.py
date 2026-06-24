from __future__ import annotations

import pytest

from bochan.visualization.ordinal_display import _to_internal_display


def test_ordinal_display_latent_maps_to_existing_plot() -> None:
    assert _to_internal_display("latent") == "current"


def test_ordinal_display_probability_maps_to_probability_plot() -> None:
    assert _to_internal_display("probability") == "probability"


def test_ordinal_display_rejects_old_current_name() -> None:
    with pytest.raises(ValueError, match="latent.*probability"):
        _to_internal_display("current")  # type: ignore[arg-type]
