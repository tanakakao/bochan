from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, got {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "tests/test_composition_best_subset.py",
    '''def test_step_grid_best_subset_supports_auto_only_when_it_resolves_exact() -> None:\n    accepted = _resolve(\n        _grid_site(\n            best_subset_strategy="auto",\n            best_subset_max_combinations=3,\n        )\n    )\n    assert accepted.final_candidate_postprocess is not None\n\n    with pytest.raises(ValueError, match="requires exact support search"):\n        _resolve(\n            _grid_site(\n                best_subset_strategy="auto",\n                best_subset_max_combinations=2,\n            )\n        )\n    with pytest.raises(ValueError, match="requires exact support search"):\n        _resolve(_grid_site(best_subset_strategy="beam"))\n''',
    '''def test_step_grid_best_subset_supports_auto_and_beam_resolution() -> None:\n    exact_auto = _resolve(\n        _grid_site(\n            best_subset_strategy="auto",\n            best_subset_max_combinations=3,\n        )\n    )\n    assert exact_auto.final_candidate_postprocess is not None\n    assert exact_auto.optimizer_kwargs["best_subset_strategy"] == "auto"\n    assert exact_auto.optimizer_kwargs["best_subset_max_combinations"] == 3\n\n    beam_auto = _resolve(\n        _grid_site(\n            best_subset_strategy="auto",\n            best_subset_max_combinations=2,\n        )\n    )\n    assert beam_auto.final_candidate_postprocess is not None\n    assert beam_auto.optimizer_kwargs["best_subset_strategy"] == "auto"\n    assert beam_auto.optimizer_kwargs["best_subset_max_combinations"] == 2\n\n    explicit_beam = _resolve(_grid_site(best_subset_strategy="beam"))\n    assert explicit_beam.final_candidate_postprocess is not None\n    assert explicit_beam.optimizer_kwargs["best_subset_strategy"] == "beam"\n''',
)

replace_once(
    "tests/test_webapp_composition_support.py",
    '    assert "step付きBest Subsetは現在Exact探索のみ対応" in best_subset\n',
    '    assert "元素ごとの刻みはExact / Beamの両方で有効" in best_subset\n'
    '    assert "Beamでは評価予算内のsupportだけを調べ" in best_subset\n'
    '    assert "Variable totalの元素量stepもExact / Beamで利用できます" in best_subset\n',
)
