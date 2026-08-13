from pathlib import Path
import py_compile


def replace_once(path_str: str, old: str, new: str) -> None:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected block not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/bochan/api/acquisition/service.py",
    """from .. import factory as _factory\nfrom ..configs import AcquisitionConfig, DataContext, ModelBundle, ObjectiveConfig\nfrom .context import (\n    _filter_context_fields_for_acqf,\n    _input_transform_n_w_from_bundle,\n    _resolve_objective_config_n_w_from_input_transform,\n)\nfrom ..llm import is_llm_selected_acquisition, resolve_llm_selected_acquisition\nfrom ..registry.acquisition import resolve_acqf_cls\nfrom .classification import (\n    build_multiclass_objective,\n    build_ordinal_objective,\n    prepare_objective_instance,\n)\n""",
    """from .. import factory as _factory\nfrom ..configs import AcquisitionConfig, DataContext, ModelBundle, ObjectiveConfig\nfrom ..llm import is_llm_selected_acquisition, resolve_llm_selected_acquisition\nfrom ..registry.acquisition import resolve_acqf_cls\nfrom .classification import (\n    build_multiclass_objective,\n    build_ordinal_objective,\n    prepare_objective_instance,\n)\nfrom .context import (\n    _filter_context_fields_for_acqf,\n    _input_transform_n_w_from_bundle,\n    _resolve_objective_config_n_w_from_input_transform,\n)\n""",
)

replace_once(
    "src/bochan/api/acquisition/defaults/information.py",
    """from .common import _num_outputs\nfrom .multiobjective import (\n    make_default_ref_point,\n    observed_multiobjective_values,\n)\nfrom ...configs import AcquisitionConfig, DataContext, ModelBundle, OptimizeConfig\n""",
    """from ...configs import AcquisitionConfig, DataContext, ModelBundle, OptimizeConfig\nfrom .common import _num_outputs\nfrom .multiobjective import (\n    make_default_ref_point,\n    observed_multiobjective_values,\n)\n""",
)

replace_once(
    "src/bochan/api/optimizer/support.py",
    """from ..configs.optimizer_names import (\n    _ALIASES,\n    _CANONICAL_OPTIMIZERS,\n    _EVOLUTIONARY_METHODS,\n    _MIXED_OPTIMIZERS,\n    EvolutionaryMethod,\n    OptimizerName,\n    _InternalMixedOptimizerName,\n    _optimizer_name,\n)\n""",
    "from ..configs.optimizer_names import _ALIASES, _optimizer_name\n",
)

replace_once(
    "tests/test_api_architecture.py",
    """import bochan.api as api\nfrom bochan.api import configs, factory, optimizer, registry\nfrom bochan.api.optimizer import core as optimizer_core\nfrom bochan.api.acquisition import defaults as acquisition_defaults\n""",
    """import bochan.api as api\nfrom bochan.api import configs, factory, optimizer, registry\nfrom bochan.api.acquisition import defaults as acquisition_defaults\n""",
)
replace_once(
    "tests/test_api_architecture.py",
    "from bochan.api.observation import state as observation_state\n",
    "from bochan.api.observation import state as observation_state\nfrom bochan.api.optimizer import core as optimizer_core\n",
)

path = Path("src/bochan/api/evaluation/cross_validation.py")
text = path.read_text(encoding="utf-8")
marker = "\ndef _aggregate_feature_importance(folds: list[Any]) -> Any:\n"
if marker not in text:
    raise RuntimeError("cross-validation aggregation function missing")
prefix = text.split(marker, 1)[0]
aggregate = '''\ndef _aggregate_feature_importance(folds: list[Any]) -> Any:\n    """Aggregate fold means and ranks without aligning latent diagnostics.\n\n    Args:\n        folds: Validation-fold ``FeatureImportanceResult`` objects.\n\n    Returns:\n        Cross-validated output-oriented importance result.\n    """\n    from bochan.inspection.result_types import (\n        CrossValidatedFeatureImportanceResult,\n        CrossValidatedImportanceSummary,\n        CrossValidatedMethodResult,\n        CrossValidatedOutputImportance,\n    )\n\n    outputs = {}\n    for output_name, first_output in folds[0].outputs.items():\n        methods = {}\n        for method_name in first_output.predictive_methods:\n            method_folds = [\n                fold.outputs[output_name].predictive_methods[method_name]\n                for fold in folds\n            ]\n            entries = {}\n            for entry_name in method_folds[0].entries:\n                fold_entries = [method.entries[entry_name] for method in method_folds]\n                values = torch.tensor(\n                    [entry.importance.mean for entry in fold_entries],\n                    dtype=torch.float64,\n                )\n                ranks = torch.tensor(\n                    [entry.importance.rank for entry in fold_entries],\n                    dtype=torch.float64,\n                )\n                entries[entry_name] = CrossValidatedImportanceSummary(\n                    values,\n                    float(values.mean()),\n                    float(values.std(unbiased=False)),\n                    float(values.min()),\n                    float(values.max()),\n                    float(values.median()),\n                    float(ranks.mean()),\n                    float(ranks.std(unbiased=False)),\n                    len(values),\n                    [entry.importance.std for entry in fold_entries],\n                )\n            methods[method_name] = CrossValidatedMethodResult(\n                method_name, entries, method_folds\n            )\n        outputs[output_name] = CrossValidatedOutputImportance(\n            output_name,\n            first_output.task_type,\n            methods,\n            {},\n            [fold.outputs[output_name].model_diagnostics for fold in folds],\n            warnings=[\n                warning\n                for fold in folds\n                for warning in fold.outputs[output_name].warnings\n            ],\n        )\n    return CrossValidatedFeatureImportanceResult(\n        outputs,\n        folds[0].feature_names,\n        [warning for fold in folds for warning in fold.warnings],\n        {"n_folds": len(folds), "pooled_oof_importance": False},\n    )\n'''
path.write_text(prefix + aggregate, encoding="utf-8")

for target in (
    "src/bochan/api/acquisition/service.py",
    "src/bochan/api/acquisition/defaults/information.py",
    "src/bochan/api/optimizer/support.py",
    "src/bochan/api/evaluation/cross_validation.py",
    "tests/test_api_architecture.py",
):
    py_compile.compile(target, doraise=True)
