from types import SimpleNamespace

import torch
from botorch.models import SingleTaskGP
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


def _session(model: nn.Module | None = None) -> SimpleNamespace:
    wrapped = _HybridModel("強度", model or _BaseGP([2.0, 4.0]))
    return SimpleNamespace(
        optimizer=SimpleNamespace(model=wrapped),
        tabular_optimizer=SimpleNamespace(
            dataset=SimpleNamespace(cat_dims=[]),
        ),
        feature_columns=["温度", "時間"],
        target_columns=["強度"],
    )


def _empty_cv_result() -> dict[str, object]:
    return {
        "feature_importance_source": "cross_validation",
        "model_diagnostics": {},
        "feature_importance_warnings": [],
        "metadata": {},
    }


def test_cv_result_receives_final_base_gp_ard_diagnostics() -> None:
    result = _empty_cv_result()

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


def test_actual_single_task_gp_exposes_ard_for_each_feature() -> None:
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = (train_x[:, :1] + 0.5 * train_x[:, 1:2]).sin()
    model = SingleTaskGP(train_x, train_y)
    result = _empty_cv_result()

    _attach_final_model_diagnostics(
        result,
        _request(diagnostics=True),
        _session(model),
    )

    diagnostics = result["model_diagnostics"]["強度"]
    components = diagnostics["ard"]["components"]
    inverse_lengthscale = [
        value
        for component in components
        for value in torch.as_tensor(component["inverse_lengthscale"]).reshape(-1).tolist()
    ]
    assert len(inverse_lengthscale) >= 2


def test_pi_only_does_not_attach_final_model_diagnostics() -> None:
    result = _empty_cv_result()

    _attach_final_model_diagnostics(
        result,
        _request(diagnostics=False),
        _session(),
    )

    assert result["model_diagnostics"] == {}
