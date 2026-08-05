import { useCallback, useEffect, useState } from "react";
import { loadCompositionSettings } from "../compositionExtension";

const STORAGE_KEY = "bochan-web-composition-settings";
const CHANGE_EVENT = "bochan-composition-settings-change";

type CompositionSettings = ReturnType<typeof loadCompositionSettings>;
type Representation = CompositionSettings["representation"];
type Normalization = CompositionSettings["normalization"];
type SettingsUpdater = (settings: CompositionSettings) => CompositionSettings;

function uniqueStrings(value: string): string[] {
  return [...new Set(value.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean))];
}

function saveCompositionSettings(settings: CompositionSettings): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

function useCompositionSettings(): [CompositionSettings, (updater: SettingsUpdater) => void] {
  const [settings, setSettings] = useState<CompositionSettings>(() => loadCompositionSettings());

  useEffect(() => {
    const refresh = () => setSettings(loadCompositionSettings());
    window.addEventListener(CHANGE_EVENT, refresh);
    return () => window.removeEventListener(CHANGE_EVENT, refresh);
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

      {settings.elements.length < 2 && (
        <p className="settings-note warning-text">候補元素を2種類以上指定してください。</p>
      )}
      <p className="settings-note">
        元素比率の上下限、刻み、必須元素、使用元素数、元素間制約は候補提案画面で設定します。
      </p>
    </article>
  );
}
