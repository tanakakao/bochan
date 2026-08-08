from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(file_name: str, old: str, new: str) -> None:
    file_path = ROOT / file_name
    text = file_path.read_text(encoding="utf-8")
    if old in text:
        file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"Expected snippet not found in {file_name}:\n{old[:500]}")


# Remove the remaining Web risk runtime adapter call.
replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''install_workflow_adapters(_workflows_tabular)
install_web_risk_adapters(_workflows_tabular)''',
    '''install_workflow_adapters(_workflows_tabular)''',
)

# Prediction row normalization is now a normal source-level helper, so the Web
# lifecycle wrapper no longer rebinds imported module functions at runtime.
replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''from . import target_results as _target_results
from . import target_settings as _target_settings
from .logging import current_request_id, get_logger, log_event''',
    '''from .logging import current_request_id, get_logger, log_event''',
)
replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''from .prediction_shapes import normalize_prediction_rows
from .reuse_dataset import store_for_model_reuse''',
    '''from .reuse_dataset import store_for_model_reuse''',
)
replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''# ``app.py`` imports visualization helpers before this compatibility module, so
# replace both already-bound helper references before loading the tabular workflow.
_target_settings._as_2d = normalize_prediction_rows
_target_results._as_2d = normalize_prediction_rows
_workflows_tabular = import_module(".workflows_tabular", package=__package__)''',
    '''_workflows_tabular = import_module(".workflows_tabular", package=__package__)''',
)

replace_once(
    "src/bochan/serving/webapp/target_settings.py",
    '''def _as_2d(value: Any, *, n_rows: int) -> Any:
    """Normalize posterior mean/variance to shape ``[n, m]``."""

    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim > 2:
        tensor = tensor.reshape(n_rows, -1)
    if tensor.ndim != 2 or tensor.shape[0] != n_rows:
        raise RuntimeError(
            f"Could not normalize prediction to [n, m]. shape={tuple(tensor.shape)}"
        )
    return tensor''',
    '''def _as_2d(value: Any, *, n_rows: int) -> Any:
    """Normalize prediction rows, including InputPerturbation expansion."""

    from .prediction_shapes import normalize_prediction_rows

    return normalize_prediction_rows(value, n_rows=n_rows)''',
)

# With LSE + InputPerturbation, candidate-wise score risk is the intended Web
# contract. Force sequential q generation so q>1 covariance-aware acquisition
# paths do not collapse all perturbation replicas into one joint batch score.
replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''  const sequentialForced = q > 1 && (
    selectedVariables.some((variable) => variable.type === "categorical")
    || searchMethod === "cmaes"
  );''',
    '''  const sequentialForced = q > 1 && (
    selectedVariables.some((variable) => variable.type === "categorical")
    || searchMethod === "cmaes"
    || (acquisitionFamily === "level_set_estimation" && inputPerturbation)
  );''',
)
replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''            q &gt; 1で有効にすると、選択済み候補をpendingとして次候補を順番に探索します。
            カテゴリ変数とCMA-ESでは自動的に有効になります。''',
    '''            q &gt; 1で有効にすると、選択済み候補をpendingとして次候補を順番に探索します。
            カテゴリ変数、CMA-ES、LSE + 入力摂動では自動的に有効になります。''',
)

replace_once(
    "docs/web_target_roles.md",
    '''With InputPerturbation, mean aggregation is always supported. VaR / CVaR are available for Bayesian optimization and LSE. BO applies risk to the objective through `ObjectiveConfig`; LSE applies it to perturbation-expanded level-set scores through the regression LSE score objective. These paths are wired directly in the Web workflow and do not replace engine or acquisition functions at runtime.''',
    '''With InputPerturbation, mean aggregation is always supported. VaR / CVaR are available for Bayesian optimization and LSE. BO applies risk to the objective through `ObjectiveConfig`; LSE applies it to perturbation-expanded level-set scores through the regression LSE score objective. For `q > 1`, the Web UI forces sequential LSE candidate generation so each nominal candidate keeps its own perturbation-risk aggregation before it is added as pending. These paths are wired directly in the Web workflow and do not replace engine or acquisition functions at runtime.''',
)

print("Permanent Web LSE follow-up applied successfully")
