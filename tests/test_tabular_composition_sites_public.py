from bochan.tabular import TabularBayesianOptimizer


def test_public_tabular_optimizer_supports_composition_sites() -> None:
    optimizer = TabularBayesianOptimizer(
        composition_sites={
            "A": {
                "column": "A_formula",
                "elements": ["La", "Sr"],
                "min_components": 1,
                "max_components": 2,
                "required_components": ["La"],
            },
            "B": {
                "column": "B_formula",
                "elements": ["Fe", "Co"],
                "min_components": 1,
                "max_components": 2,
                "required_components": ["Fe"],
            },
        }
    )

    assert optimizer.multi_site_composition_enabled
    assert set(optimizer.composition_sites) == {"A", "B"}
