from __future__ import annotations

from dataclasses import dataclass

from bochan.api.optimizer_api import _resolve_thompson_sampling_target


class _PosteriorModel:
    def posterior(self, X, **kwargs):
        return X


class _LatentModel:
    pass


@dataclass
class _Acquisition:
    model: object


def test_resolve_thompson_sampling_target_prefers_configured_public_model() -> None:
    latent_model = _LatentModel()
    public_model = _PosteriorModel()
    acqf = _Acquisition(model=latent_model)

    object.__setattr__(acqf, "_bochan_thompson_model", public_model)

    assert _resolve_thompson_sampling_target(acqf) is public_model


def test_resolve_thompson_sampling_target_uses_model_with_posterior() -> None:
    public_model = _PosteriorModel()
    acqf = _Acquisition(model=public_model)

    assert _resolve_thompson_sampling_target(acqf) is public_model


def test_resolve_thompson_sampling_target_preserves_legacy_nonposterior_fallback() -> None:
    latent_model = _LatentModel()
    acqf = _Acquisition(model=latent_model)

    assert _resolve_thompson_sampling_target(acqf) is latent_model
