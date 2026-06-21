import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.acquisition._nehvi_cache_root import (
    patch_nehvi_cache_root_init,
    resolve_nehvi_cache_root,
)
from bochan.acquisition.binary.bayesian_optimization import (
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
)
from bochan.acquisition.binary.bayesian_optimization.multi_output import (
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement as DirectBinaryNEHVI,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
)
from bochan.acquisition.ordinal.bayesian_optimization.multi_output import (
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement as DirectOrdinalNEHVI,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel


class _CacheRootUnsupportedModel:
    _supports_cache_root = False


class _DefaultModel:
    pass


class _DummyNEHVI:
    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.cache_root = kwargs["cache_root"]


def _make_ordinal_model() -> KroneckerMultiTaskOrdinalGPModel:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    model = KroneckerMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_resolve_cache_root_preserves_explicit_values() -> None:
    model = _CacheRootUnsupportedModel()

    assert resolve_nehvi_cache_root(model, True) is True
    assert resolve_nehvi_cache_root(model, False) is False


def test_resolve_cache_root_uses_model_capability_by_default() -> None:
    assert resolve_nehvi_cache_root(_CacheRootUnsupportedModel()) is False
    assert resolve_nehvi_cache_root(_DefaultModel()) is True


def test_cache_root_constructor_patch_is_idempotent() -> None:
    patched = patch_nehvi_cache_root_init(_DummyNEHVI)
    patched_again = patch_nehvi_cache_root_init(_DummyNEHVI)

    assert patched is patched_again
    assert patched(_CacheRootUnsupportedModel()).cache_root is False
    assert patched(_CacheRootUnsupportedModel(), cache_root=True).cache_root is True
    assert patched(_DefaultModel()).cache_root is True


def test_package_and_direct_imports_share_patched_classes() -> None:
    assert DirectOrdinalNEHVI is qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
    assert DirectBinaryNEHVI is qMultiOutputBinaryNoisyExpectedHypervolumeImprovement
    assert getattr(DirectOrdinalNEHVI, "_bochan_cache_root_compat_patched", False)
    assert getattr(DirectBinaryNEHVI, "_bochan_cache_root_compat_patched", False)


def test_ordinal_kronecker_nehvi_disables_cache_root_automatically() -> None:
    model = _make_ordinal_model()
    utility_values = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=torch.double,
    )
    acquisition = qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=[-0.1, -0.1],
        X_baseline=model.train_inputs_raw[0],
        utility_values=utility_values,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )

    assert acquisition._cache_root is False
