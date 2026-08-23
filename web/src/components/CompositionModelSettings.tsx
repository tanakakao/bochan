import { useCallback, useEffect, useState } from "react";
import {
  COMPOSITION_SETTINGS_CHANGE_EVENT,
  loadCompositionSettings,
  saveCompositionSettings,
  type CompositionSettings
} from "../compositionExtension";

type Representation = CompositionSettings["representation"];
type Normalization = CompositionSettings["normalization"];
type DescriptorProperty = CompositionSettings["descriptorProperties"][number];
type DescriptorStatistic = CompositionSettings["descriptorStatistics"][number];
type SettingsUpdater = (settings: CompositionSettings) => CompositionSettings;

const DESCRIPTOR_PROPERTY_OPTIONS: Array<{ value: DescriptorProperty; label: string }> = [
  { value: "atomic_number", label: "原子番号" },
  { value: "atomic_weight", label: "原子量" }
];

const DESCRIPTOR_STATISTIC_OPTIONS: Array<{ value: DescriptorStatistic; label: string }> = [
  { value: "mean", label: "平均" },
  { value: "std", label: "標準偏差" },
  { value: "min", label: "最小" },
  { value: "max", label: "最大" },
  { value: "range", label: "範囲" }
];

function uniqueStrings(value: string): string[] {
  return [...new Set(value.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean))];
}

function toggleValue<T extends string>(values: T[], value: T, checked: boolean): T[] {
  if (checked) return values.includes(value) ? values : [...values, value];
  return values.filter((item) => item !== value);
}

function useCompositionSettings(): [CompositionSettings, (updater: SettingsUpdater) => void] {
  const [settings, setSettings] = useState<CompositionSettings>(() => loadCompositionSettings());

  useEffect(() => {
    const refresh = () => setSettings(loadCompositionSettings());
    window.addEventListener(COMPOSITION_SETTINGS_CHANGE_EVENT, refresh);
    return () => window.removeEventListener(COMPOSITION_SETTINGS_CHANGE_EVENT, refresh);
  }, []);

  const update = useCallback((updater: SettingsUpdater) => {
    saveCompositionSettings(updater(loadCompositionSettings()));
  }, []);

  return [settings, update];
}

/** Configures composition-coordinate conversion inside the React-owned Model page. */
export default function CompositionModelSettings() {
  const [settings, update] = useCompositionSettings();
  const [elementsText, setElementsText] = useState(() => settings.elements.join(", "));

  useEffect(() => {
    setElementsText(settings.elements.join(", "));
  }, [settings.elements]);

  if (!settings.enabled || !settings.column) return null;

  function updateElements(): void {
    const elements = uniqueStrings(elementsText);
    update((current) => ({
      ...current,
      elements,
      requiredComponents: current.requiredComponents.filter((element) => elements.includes(element)),
      bounds: Object.fromEntries(elements.map((element) => [
        element,
        current.bounds[element] ?? [0, 1]
      ])),
      steps: Object.fromEntries(elements.map((element) => [
        element,
        current.steps[element] ?? null
      ])),
      constraints: current.constraints.map((constraint) => ({
        ...constraint,
        terms: constraint.terms.filter((term) => elements.includes(term.element))
      })),
      referenceElement: elements.includes(current.referenceElement) ? current.referenceElement : "",
      maxComponents: elements.length || null
    }));
  }

  const descriptorFeatureSelected = (
    settings.descriptorProperties.length > 0 && settings.descriptorStatistics.length > 0
  ) || settings.descriptorIncludeNumElements || settings.descriptorIncludeMixingEntropy;

  return (
    <article className="panel composition-settings-panel composition-model-settings-panel composition-model-settings-react">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">COMPOSITION MODEL</span>
          <h3>組成式のモデル変換</h3>
          <p>{settings.column}を合計1の組成比へ変換し、学習モデルの入力座標を作成します。</p>
        </div>
        <span className={`status-chip ${settings.elements.length >= 2 ? "success" : "warning"}`}>
          {settings.elements.length} elements
        </span>
      </div>

      <div className="composition-basic-grid">
        <label>
          <span>組成式列</span>
          <input value={settings.column} disabled />
        </label>
        <label>
          <span>変換方法</span>
          <select
            value={settings.representation}
            onChange={(event) => update((current) => ({
              ...current,
              representation: event.target.value as Representation
            }))}
          >
            <option value="fractions">Fraction</option>
            <option value="clr">CLR</option>
            <option value="alr">ALR</option>
            <option value="ilr">ILR</option>
          </select>
        </label>
        <label>
          <span>組成基準</span>
          <select
            value={settings.normalization}
            onChange={(event) => update((current) => ({
              ...current,
              normalization: event.target.value as Normalization
            }))}
          >
            <option value="atomic_fraction">原子比・mol比</option>
            <option value="weight_fraction">重量比</option>
          </select>
        </label>
        <label>
          <span>候補元素</span>
          <input
            value={elementsText}
            placeholder="Fe, Co, Ni"
            onChange={(event) => setElementsText(event.target.value)}
            onBlur={updateElements}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                updateElements();
                event.currentTarget.blur();
              }
            }}
          />
        </label>
        {settings.representation === "alr" && (
          <label>
            <span>参照元素</span>
            <select
              value={settings.referenceElement}
              onChange={(event) => update((current) => ({
                ...current,
                referenceElement: event.target.value
              }))}
            >
              <option value="">自動</option>
              {settings.elements.map((element) => (
                <option key={element} value={element}>{element}</option>
              ))}
            </select>
          </label>
        )}
        <label>
          <span>表示桁数</span>
          <input
            type="number"
            min={1}
            max={12}
            value={settings.precision}
            onChange={(event) => update((current) => ({
              ...current,
              precision: Math.max(1, Math.min(12, Number(event.target.value)))
            }))}
          />
        </label>
        {settings.representation !== "fractions" && (
          <>
            <label>
              <span>変換座標の下限</span>
              <input
                type="number"
                step="any"
                value={settings.coordinateLower}
                onChange={(event) => update((current) => ({
                  ...current,
                  coordinateLower: Number(event.target.value)
                }))}
              />
            </label>
            <label>
              <span>変換座標の上限</span>
              <input
                type="number"
                step="any"
                value={settings.coordinateUpper}
                onChange={(event) => update((current) => ({
                  ...current,
                  coordinateUpper: Number(event.target.value)
                }))}
              />
            </label>
          </>
        )}
      </div>

      <section className="model-advanced-section composition-descriptor-settings">
        <div className="config-column-heading">
          <span className="panel-kicker">DERIVED DESCRIPTORS</span>
          <h4>組成から元素物性記述子を自動計算</h4>
          <p>
            記述子は探索変数にせず、学習・予測・候補評価のたびに組成から再計算します。
          </p>
        </div>
        <div className="compact-setting-list">
          <label className="compact-setting-row">
            <span>
              <strong>元素物性記述子を追加</strong>
              <small>組成座標に派生記述子を追加してモデルへ入力</small>
            </span>
            <input
              type="checkbox"
              checked={settings.includeDescriptors}
              onChange={(event) => update((current) => ({
                ...current,
                includeDescriptors: event.target.checked
              }))}
            />
          </label>
        </div>

        {settings.includeDescriptors && (
          <>
            <div className="model-settings-grid">
              <fieldset>
                <legend>元素物性</legend>
                {DESCRIPTOR_PROPERTY_OPTIONS.map((option) => (
                  <label key={option.value} className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={settings.descriptorProperties.includes(option.value)}
                      onChange={(event) => update((current) => ({
                        ...current,
                        descriptorProperties: toggleValue(
                          current.descriptorProperties,
                          option.value,
                          event.target.checked
                        )
                      }))}
                    />
                    <span>{option.label}</span>
                  </label>
                ))}
              </fieldset>
              <fieldset>
                <legend>統計量</legend>
                {DESCRIPTOR_STATISTIC_OPTIONS.map((option) => (
                  <label key={option.value} className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={settings.descriptorStatistics.includes(option.value)}
                      onChange={(event) => update((current) => ({
                        ...current,
                        descriptorStatistics: toggleValue(
                          current.descriptorStatistics,
                          option.value,
                          event.target.checked
                        )
                      }))}
                    />
                    <span>{option.label}</span>
                  </label>
                ))}
              </fieldset>
              <fieldset>
                <legend>組成固有量</legend>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={settings.descriptorIncludeNumElements}
                    onChange={(event) => update((current) => ({
                      ...current,
                      descriptorIncludeNumElements: event.target.checked
                    }))}
                  />
                  <span>有効元素数</span>
                </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={settings.descriptorIncludeMixingEntropy}
                    onChange={(event) => update((current) => ({
                      ...current,
                      descriptorIncludeMixingEntropy: event.target.checked
                    }))}
                  />
                  <span>混合エントロピー</span>
                </label>
              </fieldset>
            </div>
            {!descriptorFeatureSelected && (
              <p className="settings-note warning-text">
                記述子を1種類以上選択してください。
              </p>
            )}
            <p className="settings-note">
              現在の組み込み元素物性は原子番号と原子量です。候補生成時には組成だけを最適化し、
              記述子は各候補から自動的に導出します。
            </p>
          </>
        )}
      </section>

      {settings.elements.length < 2 && (
        <p className="settings-note warning-text">候補元素を2種類以上指定してください。</p>
      )}
      <p className="settings-note">
        元素比率の上下限、刻み、必須元素、使用元素数、元素間制約は候補提案画面で設定します。
      </p>
    </article>
  );
}
