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

function combinationCount(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  const choose = Math.min(k, n - k);
  let result = 1;
  for (let index = 1; index <= choose; index += 1) {
    result = (result * (n - choose + index)) / index;
  }
  return Math.round(result);
}

function combinationRangeCount(n: number, minimum: number, maximum: number): number {
  let result = 0;
  for (let k = minimum; k <= maximum; k += 1) {
    result += combinationCount(n, k);
  }
  return result;
}

/** Configure acquisition-aware element-combination search for one composition site. */
export default function CompositionBestSubsetSettings() {
  const [settings, update] = useCompositionSettings();
  if (!settings.enabled || !settings.column) return null;

  const enabled = settings.supportSelection === "best_subset";
  const variableTotal = settings.totalMode === "variable";
  const minCount = clampCount(settings.minComponents, settings.elements.length);
  const maxCount = clampCount(settings.maxComponents ?? minCount, settings.elements.length);
  const hasSteps = settings.elements.some((element) => (settings.steps[element] ?? 0) > 0);

  const boundRequired = settings.elements.filter((element) => {
    const pair = settings.bounds[element];
    return pair !== undefined && pair[0] > 0;
  });
  const boundForbidden = settings.elements.filter((element) => {
    const pair = settings.bounds[element];
    return pair !== undefined && pair[1] <= 0;
  });
  const requiredSet = new Set([
    ...settings.requiredComponents,
    ...boundRequired
  ]);
  const forbiddenSet = new Set([
    ...settings.forbiddenComponents,
    ...boundForbidden
  ]);
  const overlap = [...requiredSet].filter((element) => forbiddenSet.has(element));
  const required = [...requiredSet].filter((element) => !forbiddenSet.has(element));
  const optionalCount = settings.elements.filter(
    (element) => !requiredSet.has(element) && !forbiddenSet.has(element)
  ).length;
  const optionalMin = Math.max(0, minCount - required.length);
  const optionalMax = Math.max(
    optionalMin,
    Math.min(optionalCount, maxCount - required.length)
  );
  const supportCount = combinationRangeCount(optionalCount, optionalMin, optionalMax);
  const cardinalityCount = Math.max(0, optionalMax - optionalMin + 1);
  const variableCardinality = minCount !== maxCount;
  const stepGridUsesBeam = hasSteps && optionalMax > 0 && (
    settings.bestSubsetStrategy === "beam" ||
    (
      settings.bestSubsetStrategy === "auto" &&
      supportCount > settings.bestSubsetMaxCombinations
    )
  );
  const beamBudgetTooSmall = enabled && (
    settings.bestSubsetStrategy === "beam" ||
    (
      settings.bestSubsetStrategy === "auto" &&
      supportCount > settings.bestSubsetMaxCombinations
    )
  ) && settings.bestSubsetMaxEvaluations < cardinalityCount;

  function setSupportSelection(value: SupportSelection): void {
    update((current) => {
      if (value !== "best_subset") {
        return { ...current, supportSelection: value };
      }
      const minimum = clampCount(current.minComponents, current.elements.length);
      const maximum = clampCount(
        current.maxComponents ?? minimum,
        current.elements.length
      );
      return {
        ...current,
        supportSelection: value,
        minComponents: Math.min(minimum, maximum),
        maxComponents: Math.max(minimum, maximum)
      };
    });
  }

  function setMinCount(value: number): void {
    update((current) => {
      const minimum = clampCount(value, current.elements.length);
      const maximum = clampCount(
        current.maxComponents ?? minimum,
        current.elements.length
      );
      return {
        ...current,
        minComponents: minimum,
        maxComponents: Math.max(minimum, maximum)
      };
    });
  }

  function setMaxCount(value: number): void {
    update((current) => {
      const maximum = clampCount(value, current.elements.length);
      const minimum = clampCount(current.minComponents, current.elements.length);
      return {
        ...current,
        minComponents: Math.min(minimum, maximum),
        maxComponents: maximum
      };
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
            通常repairに加えて、獲得関数を使って元素の組合せと使用元素数自体を探索するBest Subsetを選択できます。
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
              <span>使用元素数・最小</span>
              <input
                type="number"
                min={1}
                max={Math.max(settings.elements.length, 1)}
                value={minCount}
                onChange={(event) => setMinCount(Number(event.target.value))}
              />
            </label>
            <label>
              <span>使用元素数・最大</span>
              <input
                type="number"
                min={1}
                max={Math.max(settings.elements.length, 1)}
                value={maxCount}
                onChange={(event) => setMaxCount(Number(event.target.value))}
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
              <p>選択した元素は候補supportから除外し、元素量を0に固定します。</p>
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

      {enabled && variableTotal && (
        <p className="settings-note">
          Variable totalでは元素supportをraw absolute-amount空間で探索し、合計量は選択元素量の和として同時に最適化します。モデルには正規化組成と合計featureを入力します。
        </p>
      )}
      {enabled && !variableTotal && settings.representation !== "fractions" && (
        <p className="settings-note">
          元素supportはraw fraction空間で探索し、{settings.representation.toUpperCase()}座標は学習済みモデルと獲得関数の評価だけに使います。非選択元素はraw空間で厳密に0のまま保持されます。
        </p>
      )}
      {enabled && variableCardinality && (
        <p className="settings-note">
          使用元素数は{minCount}〜{maxCount}の各cardinalityを同じ獲得関数で比較します。必須元素は全supportに含まれ、その残りのoptional元素数をBest Subsetが選択します。
          {hasSteps ? " step指定時も、各supportのcardinalityを保持したまま実験格子へMILP投影します。" : ""}
        </p>
      )}
      {enabled && hasSteps && !variableTotal && (
        <p className="settings-note">
          元素ごとの刻みはExact / Beamの両方で有効です。各supportの連続最適化後に、support・cardinality・bounds・合計を保ったstep格子へMILP投影し、その実験可能候補で獲得関数を再評価します。
          {stepGridUsesBeam ? " Beamでは評価予算内のsupportだけを調べ、格子・線形制約を満たせないsupportは探索中にskipします。" : ""}
        </p>
      )}
      {enabled && hasSteps && variableTotal && (
        <p className="settings-note">
          Variable totalの元素量stepもExact / Beamで利用できます。各supportをraw amount格子へ投影し、support・cardinality・元素bounds・total_boundsを保った候補で獲得関数を再評価します。
          {stepGridUsesBeam ? " Beamでは不可能supportをskipしながら評価予算内で探索します。" : ""}
        </p>
      )}
      {enabled && beamBudgetTooSmall && (
        <p className="settings-note warning-text">
          Beam探索では許容cardinalityごとに少なくとも1つseedを評価します。Support評価上限を{cardinalityCount}以上にしてください。
        </p>
      )}
      {enabled && overlap.length > 0 && (
        <p className="settings-note warning-text">
          必須と禁止を同時に指定できない元素があります: {overlap.join(", ")}
        </p>
      )}
      {enabled && (
        <p className="settings-note">
          qバッチでは現時点で1つの元素supportを共有します。Autoは許容元素数全体の組合せ数（現在{supportCount}）が上限以内ならExact、それを超える場合はBeamへ切り替えます。
        </p>
      )}
    </section>
  );
}
