from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match in {path}, got {text.count(old)}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/bochan/api/support/best_subset.py",
    '''    seed_config = replace(\n        config,\n        repair_config=seed_repair,\n        fixed_features=fixed,\n        optimizer_kwargs=_inner_optimizer_kwargs(config),\n        # Seed construction is heuristic only. Support-dependent final\n        # postprocessing (for example a composition grid MILP) belongs to the\n        # subsequent fixed-support evaluation.\n        final_candidate_postprocess=None,\n    )\n''',
    '''    seed_replacements: dict[str, Any] = {}\n    if hasattr(config, "final_candidate_postprocess"):\n        # Seed construction is heuristic only. Support-dependent final\n        # postprocessing (for example a composition grid MILP) belongs to the\n        # subsequent fixed-support evaluation. The base API OptimizeConfig does\n        # not expose this tabular/canonical extension, so preserve compatibility.\n        seed_replacements["final_candidate_postprocess"] = None\n    seed_config = replace(\n        config,\n        repair_config=seed_repair,\n        fixed_features=fixed,\n        optimizer_kwargs=_inner_optimizer_kwargs(config),\n        **seed_replacements,\n    )\n''',
)

path = Path("tests/test_k_sparse_best_subset_beam.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from bochan.api.configs import CandidateRepairConfig, OptimizeConfig\n",
    "from bochan.api import OptimizeConfig as PublicOptimizeConfig\n"
    "from bochan.api.configs import CandidateRepairConfig, OptimizeConfig\n",
    1,
)
old = '''    config = OptimizeConfig(\n        optimizer=_seeded_callable_optimizer(\n            seen,\n            base_values=(0.9, 0.8, 0.7, 0.6),\n        ),\n        optimizer_kwargs={\n            "best_subset_strategy": "beam",\n            "best_subset_beam_width": 2,\n            "best_subset_beam_steps": 1,\n            "best_subset_max_evaluations": 5,\n        },\n        final_candidate_postprocess=final_postprocess,\n'''
new = '''    config = PublicOptimizeConfig(\n        optimizer=_seeded_callable_optimizer(\n            seen,\n            base_values=(0.9, 0.8, 0.7, 0.6),\n        ),\n        optimizer_kwargs={\n            "best_subset_strategy": "beam",\n            "best_subset_beam_width": 2,\n            "best_subset_beam_steps": 1,\n            "best_subset_max_evaluations": 5,\n        },\n        final_candidate_postprocess=final_postprocess,\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one new beam test block, got {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
