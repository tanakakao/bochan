from pathlib import Path


SETTINGS = Path("web/src/pages/SettingsPage.tsx")
NOISE_ALPHA = Path("web/src/components/NoiseAlphaSettings.tsx")
OPTIMIZE = Path("web/src/pages/OptimizePage.tsx")
FEATURE_CONSTRAINTS = Path("web/src/components/FeatureConstraints.tsx")


def test_observation_noise_uses_inline_variance_floor_control() -> None:
    settings = SETTINGS.read_text(encoding="utf-8")
    noise = NOISE_ALPHA.read_text(encoding="utf-8")

    observation_noise = settings.index("<h4>観測ノイズ</h4>")
    accuracy = settings.index("<h3>精度評価</h3>", observation_noise)
    alpha_control = settings.index("<NoiseAlphaSettings", observation_noise)

    assert observation_noise < alpha_control < accuracy
    assert "ノイズ分散の下限 α" in noise
    assert 'className="transform-card"' not in noise
    assert "観測ノイズ下限 α" not in noise


def test_suggest_orders_search_method_controls_before_q() -> None:
    source = OPTIMIZE.read_text(encoding="utf-8")

    candidate_column = source.index("candidate-strategy-column")
    details = source.index('className="suggestion-card-details model-output-details"')
    family = source.index("探索手法の大分類", candidate_column)
    method = source.index("最適化手法", candidate_column)
    q = source.index("候補点数 q", candidate_column)

    assert candidate_column < family < method < q < details


def test_feature_constraints_are_normal_card_below_search_space() -> None:
    source = OPTIMIZE.read_text(encoding="utf-8")
    constraints = FEATURE_CONSTRAINTS.read_text(encoding="utf-8")

    details = source.index('className="suggestion-card-details model-output-details"')
    details_end = source.index("</details>", details)
    search_space = source.index("<SearchVariableSettings", details_end)
    feature_constraints = source.index("<FeatureConstraints variables", search_space)
    validation = source.index('className="panel compact-panel validation-panel"', feature_constraints)

    assert "<FeatureConstraints variables" not in source[details:details_end]
    assert "詳細設定（候補制約" not in source
    assert search_space < feature_constraints < validation
    assert "ADVANCED · FEATURE CONSTRAINTS" not in constraints
    assert "FEATURE CONSTRAINTS" in constraints
