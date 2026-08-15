import type { FeatureMissingSettings as FeatureMissingSettingsValue } from "../webRunSettings";

interface FeatureMissingSettingsProps {
  settings: FeatureMissingSettingsValue;
  onChange: (next: FeatureMissingSettingsValue) => void;
}

/** Compact strategy selector shown with the other primary model settings. */
export function FeatureMissingStrategySettings({
  settings,
  onChange
}: FeatureMissingSettingsProps) {
  return (
    <label className="compact-setting-row compact-setting-select">
      <span>
        <strong>説明変数の欠損値</strong>
        <small>学習前に欠損行を削除するか補完するかを選択</small>
      </span>
      <select
        aria-label="説明変数の欠損値処理"
        value={settings.strategy}
        onChange={(event) => onChange({
          ...settings,
          strategy: event.target.value === "impute" ? "impute" : "drop"
        })}
      >
        <option value="drop">削除</option>
        <option value="impute">補完</option>
      </select>
    </label>
  );
}

/** Detailed imputation controls shown only in the expanded model settings. */
export function FeatureMissingImputationSettings({
  settings,
  onChange
}: FeatureMissingSettingsProps) {
  if (settings.strategy !== "impute") {
    return (
      <p className="settings-note">
        現在は欠損行を削除します。基本設定で「補完」を選ぶと補完手法を設定できます。
      </p>
    );
  }

  return (
    <>
      <div className="model-settings-grid feature-imputation-settings">
        <label>
          数値変数の補完
          <select
            value={settings.continuousStrategy}
            onChange={(event) => onChange({
              ...settings,
              continuousStrategy: event.target.value === "iterative" ? "iterative" : "mean"
            })}
          >
            <option value="mean">平均値</option>
            <option value="iterative">IterativeImputer</option>
          </select>
        </label>
        <label>
          カテゴリ変数の補完
          <select value="mode" disabled>
            <option value="mode">最頻値</option>
          </select>
        </label>
        {settings.continuousStrategy === "iterative" && (
          <label>
            最大反復回数
            <input
              type="number"
              min={1}
              step={1}
              value={settings.imputeMaxIter}
              onChange={(event) => onChange({
                ...settings,
                imputeMaxIter: Math.max(1, Math.trunc(Number(event.target.value) || 1))
              })}
            />
          </label>
        )}
      </div>
      <p className="settings-note">
        数値変数は平均値またはIterativeImputer、カテゴリ変数は最頻値で補完します。目的変数は別の欠損ポリシーで処理します。
      </p>
    </>
  );
}
