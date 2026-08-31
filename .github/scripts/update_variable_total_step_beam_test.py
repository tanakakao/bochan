from pathlib import Path

path = Path("tests/test_composition_variable_total_best_subset.py")
text = path.read_text(encoding="utf-8")
old = '''def test_variable_total_step_grid_rejects_beam_support_search() -> None:\n    transformer = _transformer("ilr")\n    site = _stepped_site("ilr")\n    site["best_subset_strategy"] = "beam"\n    with pytest.raises(ValueError, match="requires exact support search"):\n        prepare_variable_total_best_subset_config(\n            OptimizeConfig(),\n            site_name="alloy",\n            site_config=site,\n            transformer=transformer,\n            model_feature_names=_layout(transformer),\n            model_bounds=_model_bounds(transformer),\n            dtype=torch.double,\n        )\n'''
new = '''def test_variable_total_step_grid_supports_beam_search() -> None:\n    transformer = _transformer("ilr")\n    site = _stepped_site("ilr")\n    site["best_subset_strategy"] = "beam"\n    _bridge, config, _bounds = prepare_variable_total_best_subset_config(\n        OptimizeConfig(),\n        site_name="alloy",\n        site_config=site,\n        transformer=transformer,\n        model_feature_names=_layout(transformer),\n        model_bounds=_model_bounds(transformer),\n        dtype=torch.double,\n    )\n\n    assert isinstance(\n        config.final_candidate_postprocess,\n        CompositionVariableTotalGridFinalPostprocess,\n    )\n    assert config.optimizer_kwargs["best_subset_strategy"] == "beam"\n'''
if text.count(old) != 1:
    raise RuntimeError(f"Expected one stale variable-total Beam test, got {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
