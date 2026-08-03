import torch
from torch import nn

from bochan.inspection.diagnostics import extract_model_diagnostics
from bochan.models.components.saas import build_map_saas_covar_module


class SaasDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        train_x = torch.rand(8, 3, dtype=torch.double)
        self.covar_module = build_map_saas_covar_module(train_x)


def test_saas_diagnostics_skip_outer_none_lengthscale() -> None:
    diagnostics, warnings = extract_model_diagnostics(
        SaasDiagnosticModel(),
        methods=("auto",),
        feature_names=("原料1", "温度", "時間"),
        cat_dims=(),
    )

    assert warnings == []
    assert "ard" in diagnostics
    components = diagnostics["ard"]["components"]
    assert components
    assert all(component["lengthscale"] is not None for component in components)
    assert any(
        component["path"].endswith("base_kernel")
        and len(torch.as_tensor(component["inverse_lengthscale"]).reshape(-1)) == 3
        for component in components
    )

    kernel_components = diagnostics["kernel_components"]
    scale_component = next(
        component
        for component in kernel_components
        if component["kernel_class"] == "ScaleKernel"
    )
    assert "lengthscale" not in scale_component
