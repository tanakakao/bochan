import json

import torch
from botorch.models import SingleTaskGP
from torch import nn

from bochan.inspection.diagnostics import extract_model_diagnostics


class DiagnosticKernel(nn.Module):
    def __init__(self, lengthscale: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "lengthscale",
            torch.tensor(lengthscale, dtype=torch.double),
        )


class DiagnosticModel(nn.Module):
    def __init__(self, lengthscale: list[float]) -> None:
        super().__init__()
        self.covar_module = DiagnosticKernel(lengthscale)


class HybridDiagnosticModel(nn.Module):
    def __init__(self, names: list[str], models: list[nn.Module]) -> None:
        super().__init__()
        self.output_names = names
        self.models = nn.ModuleList(models)


class HeteroscedasticDiagnosticModel(DiagnosticModel):
    def __init__(self) -> None:
        super().__init__([1.0, 2.0])
        train_x = torch.rand(6, 2, dtype=torch.double)
        train_y = (train_x[:, :1] - train_x[:, 1:2]).square()
        self.noise_model = SingleTaskGP(train_x, train_y)

    def predict_noise_var(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[:-1], dtype=X.dtype, device=X.device)


def test_single_output_hybrid_model_unwraps_concrete_gp() -> None:
    model = HybridDiagnosticModel(
        ["強度"],
        [DiagnosticModel([2.0, 4.0])],
    )

    diagnostics, warnings = extract_model_diagnostics(
        model,
        methods=("auto",),
        feature_names=("温度", "時間"),
        cat_dims=(),
    )

    assert warnings == []
    assert "ard" in diagnostics
    component = diagnostics["ard"]["components"][0]
    assert component["inverse_lengthscale"] == [0.5, 0.25]


def test_multi_output_hybrid_model_preserves_output_column_names() -> None:
    model = HybridDiagnosticModel(
        ["強度", "密度"],
        [DiagnosticModel([1.0, 2.0]), DiagnosticModel([4.0, 5.0])],
    )

    diagnostics, warnings = extract_model_diagnostics(
        model,
        methods=("ard",),
        feature_names=("温度", "時間"),
        cat_dims=(),
    )

    assert warnings == []
    assert set(diagnostics["ard"]["by_output"]) == {"強度", "密度"}
    assert {
        component["output_name"]
        for component in diagnostics["ard"]["components"]
    } == {"強度", "密度"}


def test_explicit_unsupported_diagnostic_warning_keeps_output_name() -> None:
    model = HybridDiagnosticModel(
        ["強度", "密度"],
        [DiagnosticModel([1.0]), DiagnosticModel([2.0])],
    )

    _, warnings = extract_model_diagnostics(
        model,
        methods=("pca",),
        feature_names=("温度",),
        cat_dims=(),
    )

    assert warnings == [
        "強度: Diagnostic 'pca' is not supported by this model interface.",
        "密度: Diagnostic 'pca' is not supported by this model interface.",
    ]


def test_heteroscedastic_diagnostics_are_json_serializable() -> None:
    diagnostics, warnings = extract_model_diagnostics(
        HeteroscedasticDiagnosticModel(),
        methods=("heteroscedastic",),
        feature_names=("温度", "時間"),
        cat_dims=(),
    )

    assert warnings == []
    heteroscedastic = diagnostics["heteroscedastic"]
    assert heteroscedastic["noise_model"]["kind"] == "module"
    assert heteroscedastic["noise_model"]["class"].endswith(".SingleTaskGP")
    assert heteroscedastic["predict_noise_var"] == {
        "kind": "callable",
        "name": "predict_noise_var",
        "available": True,
    }
    json.dumps(diagnostics)