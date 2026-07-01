from bochan.acquisition.ordinal.levelset_estimation import (
    qHeteroMultiOutputOrdinalBoundaryVariance,
    qHeteroMultiOutputOrdinalLevelSetUncertainty,
    qHeteroMultiOutputOrdinalStraddle,
)
from bochan.api import resolve_acqf_cls


def test_hetero_multioutput_ordinal_levelset_short_aliases():
    expected = {
        "straddle": qHeteroMultiOutputOrdinalStraddle,
        "boundaryvariance": qHeteroMultiOutputOrdinalBoundaryVariance,
        "icu": qHeteroMultiOutputOrdinalLevelSetUncertainty,
    }

    for alias, expected_cls in expected.items():
        resolved = resolve_acqf_cls(
            alias,
            task_type="ordinal",
            model_type="hetero",
            multi_output=True,
        )
        assert resolved is expected_cls


def test_hetero_multioutput_ordinal_canonical_compatibility_names():
    expected = {
        "qHeteroMultiOutputOrdinalLatentStraddleAcquisition": (
            qHeteroMultiOutputOrdinalStraddle
        ),
        "qHeteroMultiOutputOrdinalBoundaryVarianceAcquisition": (
            qHeteroMultiOutputOrdinalBoundaryVariance
        ),
        "qHeteroMultiOutputOrdinalICUAcquisition": (
            qHeteroMultiOutputOrdinalLevelSetUncertainty
        ),
    }

    for alias, expected_cls in expected.items():
        assert resolve_acqf_cls(alias) is expected_cls
