import { useState } from "react";
import {
  loadFeatureMissingSettings,
  saveFeatureMissingSettings,
  type FeatureMissingSettings as FeatureMissingSettingsValue
} from "../webRunSettings";

/** Configures explanatory-variable missing-value handling used before model fitting. */
export default function FeatureMissingSettings() {
  const [settings, setSettings] = useState<FeatureMissingSettingsValue>(
    () => loadFeatureMissingSettings()
  );

  function update(next: FeatureMissingSettingsValue) {
    setSettings(next);
    saveFeatureMissingSettings(next);
  }

  return (
    <article className="panel feature-constraint-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">4 · MISSING VALUES</span>
          <h3>説明変数の欠損値</h3>
          <p>モデル学習前に、説明変数の欠損行を削除するか補完するかを設定します。</p>
        </div>
        <span className="status-chip success">{settings.strategy}</span>
      </div>

      <section className="constraint-section feature-missing-section">
        <div className="constraint-variable-picker" role="group" aria-label="説明変数の欠損値処理">
          <button
            type="button"
            className={settings.strategy === "drop" ? "selected" : ""}
            aria-pressed={settings.strategy === "drop"}
            onClick={() => update({ ...settings, strategy: "drop" })}
          >
            欠損行を削除
          </button>
          <button
            type="button"
            className={settings.strategy === "impute" ? "selected" : ""}
            aria-pressed={settings.strategy === "impute"}
            onClick={() => update({ ...settings, strategy: "impute" })}
          >
            欠損値を補完
          </button>
        </div>

        {settings.strategy === "impute" && (
          <div className="transform-fields">
            <label>
              数値変数の補完
              <select
                value={settings.continuousStrategy}
                onChange={(event) => update({
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
                  onChange={(event) => update({
                    ...settings,
                    imputeMaxIter: Math.max(1, Math.trunc(Number(event.target.value) || 1))
                  })}
                />
              </label>
            )}
          </div>
        )}
        <p className="settings-note">
          数値変数は平均値またはIterativeImputer、カテゴリ変数は最頻値で補完します。目的変数は別の欠損ポリシーで処理します。
        </p>
      </section>
    </article>
  );
}
