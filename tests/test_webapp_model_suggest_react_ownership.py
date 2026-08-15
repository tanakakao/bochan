from pathlib import Path


def test_model_to_suggest_navigation_has_no_legacy_page_hosts() -> None:
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    settings = Path("web/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")

    assert "<CompositionModelSettings />" in settings
    assert "<CompositionSearchSpaceConstraints />" in Path(
        "web/src/components/SearchVariableSettings.tsx"
    ).read_text(encoding="utf-8")
    assert "<CompositionLinearConstraints />" in Path(
        "web/src/components/FeatureConstraints.tsx"
    ).read_text(encoding="utf-8")

    assert "installCompositionExtension" not in main
    assert "installWorkflowLayoutExtension" not in main
    assert "composition-model-settings-host" not in runtime
    assert "composition-constraint-settings-host" not in runtime
    assert "model-primary-grid" not in runtime
    assert "feature-constraint-panel" not in runtime


def test_model_and_suggest_keep_primary_controls_visible_before_details() -> None:
    settings = Path("web/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    optimize = Path("web/src/pages/OptimizePage.tsx").read_text(encoding="utf-8")
    workflow = Path("web/src/styles/workflow.css").read_text(encoding="utf-8")

    target_task = settings.index("<TargetModelSettings")
    model_card = settings.index('className="panel model-workbench-card"')
    model_details = settings.index('className="model-card-details model-output-details"')
    assert target_task < model_card < model_details

    assert settings.index('className="model-config-grid"', model_card) < model_details
    assert settings.index("model-selection-column", model_card) < model_details
    assert settings.index("model-basic-settings", model_card) < model_details
    assert settings.index("checked={normalize}", model_card) < model_details
    assert settings.index("checked={inputPerturbation}", model_card) < model_details
    assert settings.index("checked={crossValidation.enabled}", model_card) < model_details
    assert settings.index("checked={featureImportance.enabled}", model_card) < model_details
    assert settings.index("<FeatureMissingStrategySettings", model_card) < model_details

    assert settings.index("Fit maxiter", model_details) > model_details
    assert settings.index("<FeatureMissingImputationSettings", model_details) > model_details
    assert settings.index("<NoiseAlphaSettings", model_details) > model_details

    training = settings.index("<h4>学習</h4>", model_details)
    robustness = settings.index("<h4>頑健化</h4>", model_details)
    missing_values = settings.index("<h4>欠損値</h4>", model_details)
    observation_noise = settings.index("<h4>観測ノイズ</h4>", model_details)
    accuracy = settings.index("<h3>精度評価</h3>", model_details)
    importance = settings.index("<h3>特徴量重要度</h3>", model_details)
    assert model_details < training < robustness < missing_values < observation_noise < accuracy < importance

    target_proposal = optimize.index("<TargetProposalSettings")
    suggest_card = optimize.index('className="panel suggestion-workbench-card"')
    suggest_details = optimize.index('className="suggestion-card-details model-output-details"')
    search_space = optimize.index("<SearchVariableSettings")
    assert target_proposal < suggest_card < suggest_details < search_space

    assert optimize.index('className="suggestion-config-grid"', suggest_card) < suggest_details
    assert optimize.index("獲得関数", suggest_card) < suggest_details
    assert optimize.index("候補点数 q", suggest_card) < suggest_details
    assert optimize.index("最適化手法", suggest_card) < suggest_details
    assert optimize.index("<FeatureConstraints variables", suggest_details) > suggest_details
    assert optimize.index("num_restarts", suggest_details) > suggest_details
    assert optimize.index("raw_samples", suggest_details) > suggest_details

    assert ".model-config-grid," in workflow
    assert ".suggestion-config-grid" in workflow
    responsive = workflow[workflow.index("@media (max-width: 980px)") :]
    assert ".model-config-grid," in responsive
    assert "grid-template-columns: 1fr;" in responsive


def test_model_defaults_and_detail_copy_are_intentional() -> None:
    settings = Path("web/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    run_settings = Path("web/src/context/useWorkbenchRunSettings.ts").read_text(encoding="utf-8")

    assert 'const [inputPerturbation, setInputPerturbation] = useState(false);' in run_settings
    assert 'const [nW, setNW] = useState(4);' in run_settings
    assert "setInputPerturbation(false);" in run_settings
    assert "setNW(4);" in run_settings

    assert "CVのON/OFFは上のモデル選択欄で切り替え" not in settings
    assert "計算のON/OFFは上のモデル選択欄で切り替え" not in settings
    assert "モデル選択欄でCVを有効にすると" not in settings
    assert "モデル選択欄で特徴量重要度を有効にすると" not in settings
    assert "基本設定で入力摂動を有効にすると" not in settings
