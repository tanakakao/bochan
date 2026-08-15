from pathlib import Path


TARGET_UTILS = Path("web/src/targetSettingUtils.ts")
TARGET_PROPOSAL = Path("web/src/components/TargetProposalSettings.tsx")


def test_lse_target_defaults_are_task_aware() -> None:
    source = TARGET_UTILS.read_text(encoding="utf-8")

    assert "export function levelSetTargetDefaultPatch" in source
    assert 'if (setting.task_type === "regression")' in source
    assert 'patch.goal = "target";' in source
    assert "regressionLevelSetThreshold(column, setting)" in source

    classification = source.index('if (setting.task_type === "classification")')
    ordinal_default = source.rindex('patch.goal = "above";')
    assert "patch.value = 0.5;" in source[classification:ordinal_default]
    assert "classes[Math.floor(classes.length / 2)]" in source


def test_lse_proposal_initializes_defaults_and_forces_regression_target_goal() -> None:
    source = TARGET_PROPOSAL.read_text(encoding="utf-8")

    assert "useEffect(() => {" in source
    assert "levelSetTargetDefaultPatch(column, setting, preview)" in source
    assert 'optimizedLevelSet && setting.task_type === "regression"' in source
    assert '>目標値</span>' in source
    assert 'isLevelSet ? "探索するしきい値"' in source
    assert "`${target}の探索するしきい値`" in source


def test_lse_optimized_rows_do_not_offer_unconfigured_none_goal() -> None:
    source = TARGET_PROPOSAL.read_text(encoding="utf-8")

    assert '!optimizedLevelSet && <option value="none">なし</option>' in source
    assert 'optimizedLevelSet && setting.task_type === "ordinal"' in source
    assert '未設定の分類は対象クラス確率0.5' in source
    assert '順序回帰はクラス順の中央を初期境界' in source
