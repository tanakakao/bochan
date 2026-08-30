import { useCallback, useEffect, useState } from "react";
import {
  COMPOSITION_SETTINGS_CHANGE_EVENT,
  loadCompositionSettings,
  saveCompositionSettings,
  type CompositionSettings
} from "../compositionExtension";

type SettingsUpdater = (settings: CompositionSettings) => CompositionSettings;
type SupportSelection = CompositionSettings["supportSelection"];
type BestSubsetStrategy = CompositionSettings["bestSubsetStrategy"];

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

function clampCount(value: number, elementCount: number): number {
  return Math.max(1, Math.min(Math.trunc(value || 1), Math.max(elementCount, 1)));
}

/** Configure acquisition-aware element-combination search for one composition site. */
export default function CompositionBestSubsetSettings() {
  const [settings, update] = useCompositionSettings();
  if (!settings.enabled || !settings.column) return null;

  const enabled = settings.supportSelection === "best_subset";
  const exactCount = settings.maxComponents ?? settings.minComponents;
  const hasSteps = settings.elements.some((element) => (settings.steps[element] ?? 0) > 0);
  const overlap = settings.requiredComponents.filter((element) => (
    settings.forbiddenComponents.includes(element)
  ));

  function setSupportSelection(value: SupportSelection): void {
    update((current) => {
      if (value !== "best_subset") {
        return { ...current, supportSelection: value };
      }
      const count = clampCount(
        current.maxComponents ?? current.minComponents,
        current.elements.length
      );
      return {
        ...current,
        supportSelection: value,
        minComponents: count,
        maxComponents: count,
        steps: Object.fromEntries(current.elements.map((element) => [element, null]))
      };
    });
  }

  function setExactCount(value: number): void {
    update((current) => {
      const count = clampCount(value, current.elements.length);
      return { ...current, minComponents: count, maxComponents: count };
    });
  }

  function toggleForbidden(element: string, checked: boolean): void {
    update((current) => ({
      ...current,
      forbiddenComponents: checked
        ? [...new Set([...current.forbiddenComponents, element])]
        : current.forbiddenComponents.filter((value) => value !== element),
      requiredComponents: checked
        ? current.requiredComponents.filter((value) => value !== element)
        : current.requiredComponents
    }));
  }

  return (
    <section className="constraint-section composition-best-subset-settings-react">
      <div className="constraint-section-heading">
        <div>
          <h4>元素組合せの探索方法</h4>
          <p>
            通常repairに加えて、獲得関数を使って元素の組合せ自体を探索するBest Subsetを選択できます。
          </p>
        </div>
        <span className={`status-chip ${enabled ? "success" : ""}`}>
          {enabled ? "BEST SUBSET" : "REPAIR"}
        </span>
      </div>

      <div className="composition-basic-grid">
        <label>
          <span>Support探索</span>
          <select
            value={settings.supportSelection}
            onChange={(event) => setSupportSelection(event.target.value as SupportSelection)}
          >
            <option value="repair">従来のrepair</option>
            <option value="best_subset">Acquisition-aware Best Subset</option>
          </select>
        </label>

        {enabled && (
          <>
            <label>
              <span>使用元素数</span>
              <input
                type="number"
                min={1}
                max={Math.max(settings.elements.length, 1)}
                value={exactCount}
                onChange={(event) => setExactCount(Number(event.target.value))}
              />
            </label>
            <label>
              <span>探索戦略</span>
              <select
                value={settings.bestSubsetStrategy}
                onChange={(event) => update((current) => ({
                  ...current,
                  bestSubsetStrategy: event.target.value as BestSubsetStrategy
                }))}
              >
                <option value="auto">Auto（小規模Exact / 大規模Beam）</option>
                <option value="exact">Exact（全組合せ）</option>
                <option value="beam">Beam（近似探索）</option>
              </select>
            </label>
            <label>
              <span>Exact最大組合せ数</span>
              <input
                type="number"
                min={1}
                value={settings.bestSubsetMaxCombinations}
                onChange={(event) => update((current) => ({
                  ...current,
                  bestSubsetMaxCombinations: Math.max(1, Math.trunc(Number(event.target.value) || 1))
                }))}
              />
            </label>
          </>
        )}
      </div>

      {enabled && (settings.bestSubsetStrategy === "beam" || settings.bestSubsetStrategy === "auto") && (
        <div className="composition-basic-grid">
          <label>
            <span>Beam幅</span>
            <input
              type="number"
              min={1}
              value={settings.bestSubsetBeamWidth}
              onChange={(event) => update((current) => ({
                ...current,
                bestSubsetBeamWidth: Math.max(1, Math.trunc(Number(event.target.value) || 1))
              }))}
            />
          </label>
          <label>
            <span>Beam反復回数</span>
            <input
              type="number"
              min={0}
              value={settings.bestSubsetBeamSteps}
              onChange={(event) => update((current) => ({
                ...current,
                bestSubsetBeamSteps: Math.max(0, Math.trunc(Number(event.target.value) || 0))
              }))}
            />
          </label>
          <label>
            <span>Support評価上限</span>
            <input
              type="number"
              min={1}
              value={settings.bestSubsetMaxEvaluations}
              onChange={(event) => update((current) => ({
                ...current,
                bestSubsetMaxEvaluations: Math.max(1, Math.trunc(Number(event.target.value) || 1))
              }))}
            />
          </label>
        </div>
      )}

      {enabled && (
        <section className="composition-element-section">
          <div className="constraint-section-heading">
            <div>
              <h4>禁止元素</h4>
              <p>選択した元素は候補supportから除外し、組成比を0に固定します。</p>
            </div>
          </div>
          {settings.elements.length === 0 ? (
            <div className="constraint-empty">候補元素をModel画面で指定してください。</div>
          ) : (
            <div className="compact-setting-list">
              {settings.elements.map((element) => (
                <label className="compact-setting-row" key={element}>
                  <span>
                    <strong>{element}</strong>
                    <small>
                      {settings.requiredComponents.includes(element)
                        ? "現在は必須元素です。禁止にすると必須指定を解除します。"
                        : "Best Subsetの候補から除外"}
                    </small>
                  </span>
                  <input
                    type="checkbox"
                    checked={settings.forbiddenComponents.includes(element)}
                    onChange={(event) => toggleForbidden(element, event.target.checked)}
                  />
                </label>
              ))}
            </div>
          )}
        </section>
      )}

      {enabled && settings.representation !== "fractions" && (
        <p className="settings-note warning-text">
          Best Subsetはraw元素supportを扱うため、Model画面の変換方法をFractionにしてください。CLR / ALR / ILR座標の0は元素不存在を意味しません。
        </p>
      )}
      {enabled && settings.minComponents !== settings.maxComponents && (
        <p className="settings-note warning-text">
          Best Subsetでは使用元素数を1つに固定してください。上の「使用元素数」を変更すると最小・最大を同じ値に戻します。
        </p>
      )}
      {enabled && hasSteps && (
        <p className="settings-note warning-text">
          現在のBest Subsetは連続fractionのみ対応です。元素ごとの刻みを空にしてください。
        </p>
      )}
      {enabled && overlap.length > 0 && (
        <p className="settings-note warning-text">
          必須と禁止を同時に指定できない元素があります: {overlap.join(", ")}
        </p>
      )}
      {enabled && (
        <p className="settings-note">
          qバッチでは現時点で1つの元素supportを共有します。Autoは組合せ数が上限以内ならExact、それを超える場合はBeamへ切り替えます。
        </p>
      )}
    </section>
  );
}
