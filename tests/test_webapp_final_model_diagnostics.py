from types import SimpleNamespace

import torch
from torch import nn

from bochan.serving.webapp.workflows import _attach_final_model_diagnostics


class _Kernel(nn.Module):
    def __init__(self, lengthscale: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "lengthscale",
            torch.tensor(lengthscale, dtype=torch.double),
        )


class _BaseGP(nn.Module):
    def __init__(self, lengthscale: list[float]) -> None:
        super().__init__()
        self.covar_module = _Kernel(lengthscale)


class _HybridModel(nn.Module):
    def __init__(self, output_name: str, model: nn.Module) -> None:
        super().__init__()
        self.output_names = [output_name]
        self.models = nn.ModuleList([model])


def _request(*, diagnostics: bool) -> SimpleNamespace:
    return SimpleNamespace(
        feature_importance=SimpleNamespace(
            enabled=True,
            config=SimpleNamespace(
                diagnostic_methods=["auto"] if diagnostics else [],
            ),
        )
    )


def _session() -> SimpleNamespace:
    model = _HybridModel("強度", _BaseGP([2.0, 4.0]))
    return SimpleNamespace(
        optimizer=SimpleNamespace(model=model),
        tabular_optimizer=SimpleNamespace(
            dataset=SimpleNamespace(cat_dims=[]),
        ),
        feature_columns=["温度", "時間"],
        target_columns=["強度"],
    )


def test_cv_result_receives_final_base_gp_ard_diagnostics() -> None:
    result = {
        "feature_importance_source": "cross_validation",
        "model_diagnostics": {},
        "feature_importance_warnings": [],
        "metadata": {},
    }

    _attach_final_model_diagnostics(
        result,
        _request(diagnostics=True),
        _session(),
    )

    diagnostics = result["model_diagnostics"]["強度"]
    assert "ard" in diagnostics
    component = diagnostics["ard"]["components"][0]
    assert component["inverse_lengthscale"] == [0.5, 0.25]
    assert result["metadata"]["model_diagnostics_source"] == "final_fitted_model"


def test_pi_only_does_not_attach_final_model_diagnostics() -> None:
    result = {
        "feature_importance_source": "cross_validation",
        "model_diagnostics": {},
        "feature_importance_warnings": [],
    }

    _attach_final_model_diagnostics(
        result,
        _request(diagnostics=False),
        _session(),
    )

    assert result["model_diagnostics"] == {}
