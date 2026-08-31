from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match in {path}, got {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Generic Beam: seed generation is heuristic, so do not apply a support-dependent
# final projector until the selected support is evaluated normally.
replace_once(
    "src/bochan/api/support/best_subset.py",
    """    seed_config = replace(\n        config,\n        repair_config=seed_repair,\n        fixed_features=fixed,\n        optimizer_kwargs=_inner_optimizer_kwargs(config),\n    )\n""",
    """    seed_config = replace(\n        config,\n        repair_config=seed_repair,\n        fixed_features=fixed,\n        optimizer_kwargs=_inner_optimizer_kwargs(config),\n        # Seed construction is heuristic only. Support-dependent final\n        # postprocessing (for example a composition grid MILP) belongs to the\n        # subsequent fixed-support evaluation.\n        final_candidate_postprocess=None,\n    )\n""",
)

replace_once(
    "src/bochan/api/support/best_subset.py",
    """    if not records:\n        raise InfeasibleBestSubsetSupportError(\n            \"best_subset beam search did not produce any feasible seed support.\"\n        )\n\n    beam = sorted(\n""",
    """    if not records:\n        # A heuristic seed may be grid-infeasible. Recover a feasible starting\n        # support by walking the support graph within the normal evaluation budget.\n        frontier = list(seeds)\n        while frontier and len(visited) < max_evaluations and not records:\n            neighbor_set: set[tuple[int, ...]] = set()\n            for support in frontier:\n                neighbor_set.update(_support_neighbors(support, problem))\n            candidates_to_evaluate = sorted(\n                neighbor_set - visited,\n                key=lambda support: _support_order(support, problem),\n            )\n            if not candidates_to_evaluate:\n                break\n\n            frontier = []\n            for support in candidates_to_evaluate:\n                if len(visited) >= max_evaluations:\n                    break\n                visited.add(support)\n                frontier.append(support)\n                try:\n                    records[support] = _evaluate_support(\n                        support=support,\n                        acqf=acqf,\n                        bounds=bounds,\n                        config=config,\n                        optimize_one=optimize_one,\n                    )\n                except InfeasibleBestSubsetSupportError:\n                    continue\n\n        if not records:\n            raise InfeasibleBestSubsetSupportError(\n                \"best_subset beam search did not find a feasible seed support within \"\n                f\"the evaluation budget ({max_evaluations}).\"\n            )\n\n    beam = sorted(\n""",
)

# Fixed-total / raw-fraction composition path.
replace_once(
    "src/bochan/tabular/composition/support.py",
    """    \"\"\"Keep step-grid search on exhaustive exact-cardinality supports for now.\"\"\"\n    if not config.get(\"steps\") or optional_k == 0:\n        return\n    strategy = str(\n        optimizer_kwargs.get(\"best_subset_strategy\", \"exact\")\n    ).lower()\n    support_count = comb(optional_count, optional_k)\n    maximum = int(\n        optimizer_kwargs.get(\"best_subset_max_combinations\", 2000)\n    )\n    if strategy == \"auto\":\n        strategy = \"exact\" if support_count <= maximum else \"beam\"\n    if strategy != \"exact\":\n        raise ValueError(\n            \"Composition best_subset with component steps currently requires exact \"\n            \"support search. Use best_subset_strategy='exact', or 'auto' only when \"\n            \"the support count is within best_subset_max_combinations.\"\n        )\n    if support_count > maximum:\n        raise ValueError(\n            \"Composition step-grid best_subset exact enumeration would evaluate \"\n            f\"{support_count} supports, exceeding best_subset_max_combinations={maximum}.\"\n        )\n""",
    """    \"\"\"Validate stepped support search without forcing full enumeration.\"\"\"\n    if not config.get(\"steps\") or optional_k == 0:\n        return\n    strategy = str(\n        optimizer_kwargs.get(\"best_subset_strategy\", \"exact\")\n    ).lower()\n    if strategy not in {\"exact\", \"beam\", \"auto\"}:\n        raise ValueError(\"best_subset_strategy must be exact, beam, or auto.\")\n    support_count = comb(optional_count, optional_k)\n    maximum = int(\n        optimizer_kwargs.get(\"best_subset_max_combinations\", 2000)\n    )\n    resolved = (\n        \"exact\" if strategy == \"auto\" and support_count <= maximum\n        else \"beam\" if strategy == \"auto\"\n        else strategy\n    )\n    if resolved == \"exact\" and support_count > maximum:\n        raise ValueError(\n            \"Composition step-grid best_subset exact enumeration would evaluate \"\n            f\"{support_count} supports, exceeding best_subset_max_combinations={maximum}.\"\n        )\n""",
)

replace_once(
    "src/bochan/tabular/composition/support.py",
    """def _validate_grid_supports(\n    *,\n    projector: CompositionGridFinalPostprocess | Any,\n    config: Mapping[str, Any],\n    required: Sequence[str],\n    optional: Sequence[str],\n    optional_k: int,\n) -> None:\n    if not config.get(\"steps\"):\n        return\n\n    feasible_count = 0\n""",
    """def _validate_grid_supports(\n    *,\n    projector: CompositionGridFinalPostprocess | Any,\n    config: Mapping[str, Any],\n    optimizer_kwargs: Mapping[str, Any],\n    required: Sequence[str],\n    optional: Sequence[str],\n    optional_k: int,\n) -> None:\n    if not config.get(\"steps\"):\n        return\n\n    strategy = str(optimizer_kwargs.get(\"best_subset_strategy\", \"exact\")).lower()\n    support_count = comb(len(optional), optional_k)\n    maximum = int(optimizer_kwargs.get(\"best_subset_max_combinations\", 2000))\n    if strategy == \"auto\":\n        strategy = \"exact\" if support_count <= maximum else \"beam\"\n    if strategy == \"beam\":\n        # Beam remains scalable by checking feasibility only when a support is\n        # evaluated. Generic Best Subset skips explicit support-infeasibility.\n        return\n\n    feasible_count = 0\n""",
)

replace_once(
    "src/bochan/tabular/composition/support.py",
    """        _validate_grid_supports(\n            projector=final_candidate_postprocess,\n            config=config,\n            required=ordered_required,\n""",
    """        _validate_grid_supports(\n            projector=final_candidate_postprocess,\n            config=config,\n            optimizer_kwargs=search_optimizer_kwargs,\n            required=ordered_required,\n""",
)

# Variable-total raw-amount path.
replace_once(
    "src/bochan/tabular/composition/variable_total_support.py",
    "from bochan.tabular.data import resolve_optimize_config_columns\n",
    "from bochan.api.support.best_subset import InfeasibleBestSubsetSupportError\n"
    "from bochan.tabular.data import resolve_optimize_config_columns\n",
)

replace_once(
    "src/bochan/tabular/composition/variable_total_support.py",
    """    \"\"\"Keep variable-total step grids on exhaustive support search for now.\"\"\"\n\n    if not config.get(\"steps\") or optional_k == 0:\n        return\n    strategy = str(optimizer_kwargs.get(\"best_subset_strategy\", \"exact\")).lower()\n    support_count = comb(optional_count, optional_k)\n    maximum = int(optimizer_kwargs.get(\"best_subset_max_combinations\", 2000))\n    if strategy == \"auto\":\n        strategy = \"exact\" if support_count <= maximum else \"beam\"\n    if strategy != \"exact\":\n        raise ValueError(\n            \"Variable-total composition best_subset with component steps currently \"\n            \"requires exact support search. Use best_subset_strategy='exact', or \"\n            \"'auto' only when the support count is within best_subset_max_combinations.\"\n        )\n    if support_count > maximum:\n        raise ValueError(\n            \"Variable-total composition step-grid best_subset exact enumeration would \"\n            f\"evaluate {support_count} supports, exceeding \"\n            f\"best_subset_max_combinations={maximum}.\"\n        )\n""",
    """    \"\"\"Validate variable-total stepped support search strategy.\"\"\"\n\n    if not config.get(\"steps\") or optional_k == 0:\n        return\n    strategy = str(optimizer_kwargs.get(\"best_subset_strategy\", \"exact\")).lower()\n    if strategy not in {\"exact\", \"beam\", \"auto\"}:\n        raise ValueError(\"best_subset_strategy must be exact, beam, or auto.\")\n    support_count = comb(optional_count, optional_k)\n    maximum = int(optimizer_kwargs.get(\"best_subset_max_combinations\", 2000))\n    resolved = (\n        \"exact\" if strategy == \"auto\" and support_count <= maximum\n        else \"beam\" if strategy == \"auto\"\n        else strategy\n    )\n    if resolved == \"exact\" and support_count > maximum:\n        raise ValueError(\n            \"Variable-total composition step-grid best_subset exact enumeration would \"\n            f\"evaluate {support_count} supports, exceeding \"\n            f\"best_subset_max_combinations={maximum}.\"\n        )\n""",
)

replace_once(
    "src/bochan/tabular/composition/variable_total_support.py",
    """def _validate_grid_supports(\n    *,\n    projector: CompositionVariableTotalGridFinalPostprocess | Any,\n    site_config: Mapping[str, Any],\n    required: Sequence[str],\n    optional: Sequence[str],\n    optional_k: int,\n) -> None:\n    if not site_config.get(\"steps\"):\n        return\n    for selected in combinations(optional, optional_k):\n        projector.validate_support([*required, *selected])\n""",
    """def _validate_grid_supports(\n    *,\n    projector: CompositionVariableTotalGridFinalPostprocess | Any,\n    site_config: Mapping[str, Any],\n    optimizer_kwargs: Mapping[str, Any],\n    required: Sequence[str],\n    optional: Sequence[str],\n    optional_k: int,\n) -> None:\n    if not site_config.get(\"steps\"):\n        return\n\n    strategy = str(optimizer_kwargs.get(\"best_subset_strategy\", \"exact\")).lower()\n    support_count = comb(len(optional), optional_k)\n    maximum = int(optimizer_kwargs.get(\"best_subset_max_combinations\", 2000))\n    if strategy == \"auto\":\n        strategy = \"exact\" if support_count <= maximum else \"beam\"\n    if strategy == \"beam\":\n        return\n\n    feasible_count = 0\n    last_error: InfeasibleBestSubsetSupportError | None = None\n    for selected in combinations(optional, optional_k):\n        try:\n            projector.validate_support([*required, *selected])\n        except InfeasibleBestSubsetSupportError as exc:\n            last_error = exc\n            continue\n        feasible_count += 1\n\n    if feasible_count == 0 and last_error is not None:\n        raise last_error\n""",
)

replace_once(
    "src/bochan/tabular/composition/variable_total_support.py",
    """        _validate_grid_supports(\n            projector=final_candidate_postprocess,\n            site_config=site_config,\n            required=ordered_required,\n""",
    """        _validate_grid_supports(\n            projector=final_candidate_postprocess,\n            site_config=site_config,\n            optimizer_kwargs=search_optimizer_kwargs,\n            required=ordered_required,\n""",
)

# React validation/UI.
replace_once(
    "web/src/context/workbenchValidation.ts",
    "    if (hasSteps && usesBeam && optionalMax > 0) return false;\n",
    "",
)

replace_once(
    "web/src/components/CompositionBestSubsetSettings.tsx",
    """  const stepGridWouldUseBeam = hasSteps && !variableCardinality && optionalMax > 0 && (\n    settings.bestSubsetStrategy === \"beam\" ||\n    (\n      settings.bestSubsetStrategy === \"auto\" &&\n      supportCount > settings.bestSubsetMaxCombinations\n    )\n  );\n""",
    """  const stepGridUsesBeam = hasSteps && !variableCardinality && optionalMax > 0 && (\n    settings.bestSubsetStrategy === \"beam\" ||\n    (\n      settings.bestSubsetStrategy === \"auto\" &&\n      supportCount > settings.bestSubsetMaxCombinations\n    )\n  );\n""",
)

replace_once(
    "web/src/components/CompositionBestSubsetSettings.tsx",
    """      {enabled && hasSteps && !stepGridWouldUseBeam && !stepGridVariableCardinality && !variableTotal && (\n        <p className=\"settings-note\">\n          元素ごとの刻みはExact support探索で有効です。各supportの連続最適化後に、support・bounds・合計を保った最も近いstep格子点へ投影し、その候補で獲得関数を再評価します。\n        </p>\n      )}\n      {enabled && hasSteps && !stepGridWouldUseBeam && !stepGridVariableCardinality && variableTotal && (\n        <p className=\"settings-note\">\n          Variable totalの元素量stepはExact support探索で有効です。各supportの連続最適化後にraw amount格子へ投影し、support・元素bounds・total_boundsを保った候補で獲得関数を再評価します。投影後の合計量はtotal_bounds内で動きます。\n        </p>\n      )}\n      {enabled && stepGridWouldUseBeam && (\n        <p className=\"settings-note warning-text\">\n          step付きBest Subsetは現在Exact探索のみ対応です。探索戦略をExactにするか、AutoのExact最大組合せ数を{supportCount}以上にしてください。\n        </p>\n      )}\n""",
    """      {enabled && hasSteps && !stepGridVariableCardinality && !variableTotal && (\n        <p className=\"settings-note\">\n          元素ごとの刻みはExact / Beamの両方で有効です。各supportの連続最適化後に、support・bounds・合計を保ったstep格子へMILP投影し、その実験可能候補で獲得関数を再評価します。\n          {stepGridUsesBeam ? \" Beamでは評価予算内のsupportだけを調べ、格子・線形制約を満たせないsupportは探索中にskipします。\" : \"\"}\n        </p>\n      )}\n      {enabled && hasSteps && !stepGridVariableCardinality && variableTotal && (\n        <p className=\"settings-note\">\n          Variable totalの元素量stepもExact / Beamで利用できます。各supportをraw amount格子へ投影し、support・元素bounds・total_boundsを保った候補で獲得関数を再評価します。\n          {stepGridUsesBeam ? \" Beamでは不可能supportをskipしながら評価予算内で探索します。\" : \"\"}\n        </p>\n      )}\n""",
)

# Documentation.
replace_once(
    "docs/composition_best_subset.md",
    """Step-grid Best Subset currently uses exhaustive support search for both fixed and variable totals:\n\n- `best_subset_strategy=\"exact\"` is supported;\n- `\"auto\"` is supported only when the exact-cardinality support count is within `best_subset_max_combinations`, so Auto resolves to Exact;\n- `\"beam\"` with component steps is rejected explicitly;\n- `min_components != max_components` with component steps is rejected explicitly.\n""",
    """Step-grid Best Subset supports both exhaustive and approximate support search for fixed and variable totals:\n\n- `best_subset_strategy=\"exact\"` enumerates every exact-cardinality support within `best_subset_max_combinations`;\n- `\"beam\"` explores only the configured support-evaluation budget;\n- `\"auto\"` uses Exact below `best_subset_max_combinations` and Beam above it;\n- every evaluated support is ranked by the acquisition value of its final MILP-projected experiment-space candidate;\n- grid/linear-constraint-infeasible supports are skipped explicitly during Beam search;\n- `min_components != max_components` with component steps remains rejected explicitly.\n\nExact mode prevalidates the complete support set because it will enumerate it anyway. Beam deliberately avoids that combinatorial pre-scan: feasibility is checked lazily when a support is evaluated. If the heuristic top-k seed is infeasible, Beam walks neighboring supports within `best_subset_max_evaluations` until it finds a feasible starting point or exhausts the budget.\n""",
)
replace_once(
    "docs/composition_best_subset.md",
    "- fixed-total and variable-total component step grids with exact cardinality and Exact support search (or Auto resolving to Exact);\n",
    "- fixed-total and variable-total component step grids with exact cardinality and Exact, Beam, or Auto support search;\n",
)
replace_once(
    "docs/composition_best_subset.md",
    "- Beam search over stepped compositions;\n",
    "",
)

# Update tests that encoded the previous Exact-only contract.
replace_once(
    "tests/test_composition_best_subset_steps.py",
    """def test_logratio_step_grid_rejects_auto_when_support_count_requires_beam() -> None:\n    transformer = _transformer(\"ilr\")\n    with pytest.raises(ValueError, match=\"requires exact support search\"):\n        prepare_logratio_best_subset_config(\n            OptimizeConfig(),\n            site_name=\"alloy\",\n            site_config=_site(\n                \"ilr\",\n                best_subset_strategy=\"auto\",\n                best_subset_max_combinations=2,\n            ),\n            transformer=transformer,\n            model_feature_names=_model_layout(transformer),\n            model_bounds=_model_bounds(transformer),\n            dtype=torch.double,\n            device=None,\n        )\n""",
    """def test_logratio_step_grid_auto_can_switch_to_beam() -> None:\n    transformer = _transformer(\"ilr\")\n    _bridge, config, _bounds = prepare_logratio_best_subset_config(\n        OptimizeConfig(),\n        site_name=\"alloy\",\n        site_config=_site(\n            \"ilr\",\n            best_subset_strategy=\"auto\",\n            best_subset_max_combinations=2,\n        ),\n        transformer=transformer,\n        model_feature_names=_model_layout(transformer),\n        model_bounds=_model_bounds(transformer),\n        dtype=torch.double,\n        device=None,\n    )\n    assert config.optimizer_kwargs[\"best_subset_strategy\"] == \"auto\"\n    assert config.optimizer_kwargs[\"best_subset_max_combinations\"] == 2\n""",
)

replace_once(
    "tests/test_composition_best_subset_steps.py",
    """    with pytest.raises(ValueError, match=\"requires exact support search\"):\n        prepare_logratio_best_subset_config(\n            OptimizeConfig(\n                optimizer_kwargs={\"best_subset_strategy\": \"beam\"}\n            ),\n            site_name=\"alloy\",\n            site_config=_site(\"ilr\", best_subset_strategy=\"exact\"),\n            transformer=transformer,\n            model_feature_names=_model_layout(transformer),\n            model_bounds=_model_bounds(transformer),\n            dtype=torch.double,\n            device=None,\n        )\n""",
    """    _bridge, beam_config, _bounds = prepare_logratio_best_subset_config(\n        OptimizeConfig(optimizer_kwargs={\"best_subset_strategy\": \"beam\"}),\n        site_name=\"alloy\",\n        site_config=_site(\"ilr\", best_subset_strategy=\"exact\"),\n        transformer=transformer,\n        model_feature_names=_model_layout(transformer),\n        model_bounds=_model_bounds(transformer),\n        dtype=torch.double,\n        device=None,\n    )\n    assert beam_config.optimizer_kwargs[\"best_subset_strategy\"] == \"beam\"\n""",
)

# Generic regression test for an infeasible heuristic seed.
path = Path("tests/test_k_sparse_best_subset_beam.py")
text = path.read_text(encoding="utf-8")
if "from bochan.api.support.best_subset import InfeasibleBestSubsetSupportError" not in text:
    text = text.replace(
        "from bochan.api.factory import optimize_candidates\n",
        "from bochan.api.factory import optimize_candidates\n"
        "from bochan.api.support.best_subset import InfeasibleBestSubsetSupportError\n",
        1,
    )
if "test_best_subset_beam_recovers_from_infeasible_topk_seed" not in text:
    text += '''\n\n\ndef test_best_subset_beam_recovers_from_infeasible_topk_seed() -> None:\n    bounds = torch.tensor([[0.0] * 4, [1.0] * 4])\n    comp_idx = (0, 1, 2, 3)\n    seen: list[tuple[int, ...]] = []\n\n    def final_postprocess(candidates: torch.Tensor) -> torch.Tensor:\n        active = tuple(\n            index\n            for index in comp_idx\n            if bool((candidates[..., index].abs() > 1e-8).any().item())\n        )\n        if active == (0, 1):\n            raise InfeasibleBestSubsetSupportError(\n                \"heuristic seed is grid-infeasible\"\n            )\n        return candidates\n\n    config = OptimizeConfig(\n        optimizer=_seeded_callable_optimizer(\n            seen,\n            base_values=(0.9, 0.8, 0.7, 0.6),\n        ),\n        optimizer_kwargs={\n            \"best_subset_strategy\": \"beam\",\n            \"best_subset_beam_width\": 2,\n            \"best_subset_beam_steps\": 1,\n            \"best_subset_max_evaluations\": 5,\n        },\n        final_candidate_postprocess=final_postprocess,\n        repair_config=CandidateRepairConfig(\n            comp_idx=comp_idx,\n            k=2,\n            score=\"value\",\n            support_selection=\"best_subset\",\n        ),\n    )\n    table = {\n        (0, 2): 2.0,\n        (0, 3): 3.0,\n        (1, 2): 10.0,\n        (1, 3): 4.0,\n    }\n\n    candidates, acq_value = optimize_candidates(\n        _support_table_acq(table, comp_idx),\n        bounds,\n        config,\n    )\n\n    active = tuple(torch.nonzero(candidates[0] > 1e-8).flatten().tolist())\n    assert active == (1, 2)\n    assert float(acq_value.item()) == pytest.approx(10.0)\n    assert len(set(seen)) <= 5\n'''
path.write_text(text, encoding="utf-8")

# Focused configuration tests for fixed/log-ratio and variable-total stepped Beam.
Path("tests/test_composition_step_beam.py").write_text(
    '''from __future__ import annotations\n\nfrom typing import Any\n\nimport torch\n\nfrom bochan.api import OptimizeConfig\nfrom bochan.composition import CompositionTransformer\nfrom bochan.tabular.composition.grid import (\n    CompositionGridFinalPostprocess,\n    CompositionVariableTotalGridFinalPostprocess,\n)\nfrom bochan.tabular.composition.logratio_support import prepare_logratio_best_subset_config\nfrom bochan.tabular.composition.variable_total_support import (\n    prepare_variable_total_best_subset_config,\n)\n\n\ndef _transformer() -> CompositionTransformer:\n    transformer = CompositionTransformer(\n        elements=[\"Al\", \"Ti\", \"V\", \"Nb\"],\n        representation=\"ilr\",\n        pseudocount=1e-8,\n        prefix=\"alloy\",\n    )\n    transformer.fit([\"Al4Ti3V2Nb\", \"Al3Ti2V3Nb2\"])\n    return transformer\n\n\ndef _fixed_site(**overrides: Any) -> dict[str, Any]:\n    site: dict[str, Any] = {\n        \"column\": \"formula\",\n        \"elements\": (\"Al\", \"Ti\", \"V\", \"Nb\"),\n        \"representation\": \"ilr\",\n        \"normalization\": \"atomic_fraction\",\n        \"reference_element\": None,\n        \"pseudocount\": 1e-8,\n        \"prefix\": \"alloy\",\n        \"total\": 1.0,\n        \"variable_total\": False,\n        \"bounds\": {e: (0.0, 1.0) for e in (\"Al\", \"Ti\", \"V\", \"Nb\")},\n        \"steps\": {e: 0.1 for e in (\"Al\", \"Ti\", \"V\", \"Nb\")},\n        \"min_components\": 2,\n        \"max_components\": 2,\n        \"required_components\": (),\n        \"forbidden_components\": (),\n        \"support_selection\": \"best_subset\",\n        \"best_subset_strategy\": \"beam\",\n        \"best_subset_max_combinations\": 2,\n        \"best_subset_beam_width\": 2,\n        \"best_subset_beam_steps\": 2,\n        \"best_subset_max_evaluations\": 5,\n    }\n    site.update(overrides)\n    return site\n\n\ndef _fixed_layout(transformer: CompositionTransformer) -> list[str]:\n    return [\"temperature\", *transformer.representation_feature_names_, \"pressure\"]\n\n\ndef _fixed_bounds(transformer: CompositionTransformer) -> torch.Tensor:\n    width = len(transformer.representation_feature_names_)\n    return torch.tensor(\n        [[800.0, *([-8.0] * width), 1.0], [1200.0, *([8.0] * width), 5.0]],\n        dtype=torch.double,\n    )\n\n\ndef _variable_site(**overrides: Any) -> dict[str, Any]:\n    site = _fixed_site(**overrides)\n    site.update({\n        \"variable_total\": True,\n        \"total\": 60.0,\n        \"total_bounds\": (40.0, 90.0),\n        \"total_feature\": \"alloy__total\",\n        \"bounds\": {e: (0.0, 70.0) for e in (\"Al\", \"Ti\", \"V\", \"Nb\")},\n        \"steps\": {e: 5.0 for e in (\"Al\", \"Ti\", \"V\", \"Nb\")},\n    })\n    return site\n\n\ndef _variable_layout(transformer: CompositionTransformer) -> tuple[str, ...]:\n    return (\n        \"temperature\",\n        *transformer.representation_feature_names_,\n        \"alloy__total\",\n        \"pressure\",\n    )\n\n\ndef _variable_bounds(transformer: CompositionTransformer) -> torch.Tensor:\n    width = len(transformer.representation_feature_names_)\n    return torch.tensor(\n        [[800.0, *([-8.0] * width), 40.0, 1.0], [1200.0, *([8.0] * width), 90.0, 5.0]],\n        dtype=torch.double,\n    )\n\n\ndef test_fixed_total_logratio_step_grid_accepts_beam() -> None:\n    transformer = _transformer()\n    _bridge, config, _bounds = prepare_logratio_best_subset_config(\n        OptimizeConfig(),\n        site_name=\"alloy\",\n        site_config=_fixed_site(),\n        transformer=transformer,\n        model_feature_names=_fixed_layout(transformer),\n        model_bounds=_fixed_bounds(transformer),\n        dtype=torch.double,\n        device=None,\n    )\n    assert config.optimizer_kwargs[\"best_subset_strategy\"] == \"beam\"\n    assert isinstance(config.final_candidate_postprocess, CompositionGridFinalPostprocess)\n\n\ndef test_fixed_total_step_grid_auto_above_limit_is_accepted() -> None:\n    transformer = _transformer()\n    _bridge, config, _bounds = prepare_logratio_best_subset_config(\n        OptimizeConfig(),\n        site_name=\"alloy\",\n        site_config=_fixed_site(\n            best_subset_strategy=\"auto\",\n            best_subset_max_combinations=2,\n        ),\n        transformer=transformer,\n        model_feature_names=_fixed_layout(transformer),\n        model_bounds=_fixed_bounds(transformer),\n        dtype=torch.double,\n        device=None,\n    )\n    assert config.optimizer_kwargs[\"best_subset_strategy\"] == \"auto\"\n    assert config.optimizer_kwargs[\"best_subset_max_combinations\"] == 2\n\n\ndef test_variable_total_step_grid_accepts_beam() -> None:\n    transformer = _transformer()\n    _bridge, config, _bounds = prepare_variable_total_best_subset_config(\n        OptimizeConfig(),\n        site_name=\"alloy\",\n        site_config=_variable_site(),\n        transformer=transformer,\n        model_feature_names=_variable_layout(transformer),\n        model_bounds=_variable_bounds(transformer),\n        dtype=torch.double,\n        device=None,\n    )\n    assert config.optimizer_kwargs[\"best_subset_strategy\"] == \"beam\"\n    assert isinstance(\n        config.final_candidate_postprocess,\n        CompositionVariableTotalGridFinalPostprocess,\n    )\n''',
    encoding="utf-8",
)
