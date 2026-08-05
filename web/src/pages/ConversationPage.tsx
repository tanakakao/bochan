import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useWorkbench } from "../context/WorkbenchContext";
import { getColumnClassValues } from "../targetSettingUtils";
import type { ColumnProfile, Direction, SearchVariable } from "../types";
import {
  loadFeatureConstraints,
  loadFeatureMissingSettings,
  loadSearchMethod,
  loadSelectionCountConstraint,
  saveFeatureConstraints,
  saveFeatureMissingSettings,
  saveSearchMethod,
  saveSelectionCountConstraint
} from "../webRunSettings";
import "../conversation-mode.css";

type ConversationStage = "data" | "target" | "direction" | "features" | "count" | "confirm" | "result";
type MessageRole = "assistant" | "user";

interface ConversationMessage {
  id: number;
  role: MessageRole;
  text: string;
}

interface StoredRunSettingsSnapshot {
  featureConstraints: ReturnType<typeof loadFeatureConstraints>;
  featureMissing: ReturnType<typeof loadFeatureMissingSettings>;
  searchMethod: ReturnType<typeof loadSearchMethod>;
  selectionCount: ReturnType<typeof loadSelectionCountConstraint>;
}

let messageSequence = 0;

function nextMessage(role: MessageRole, text: string): ConversationMessage {
  messageSequence += 1;
  return { id: messageSequence, role, text };
}

function formatNumber(value: unknown): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value ?? "—");
  if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < 0.001)) {
    return number.toExponential(3);
  }
  return number.toFixed(4).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function simpleVariablePatch(
  column: ColumnProfile,
  preview: Record<string, unknown>[],
  current: SearchVariable | undefined
): Partial<SearchVariable> {
  const categorical = column.kind === "categorical" || current?.type === "categorical";
  if (categorical) {
    return {
      type: "categorical",
      categories: getColumnClassValues(column, preview),
      lower: undefined,
      upper: undefined,
      step: undefined,
      fixed: false,
      fixed_value: undefined
    };
  }
  return {
    type: "numeric",
    categories: undefined,
    lower: column.min ?? undefined,
    upper: column.max ?? undefined,
    step: undefined,
    fixed: false,
    fixed_value: undefined
  };
}

function captureStoredRunSettings(): StoredRunSettingsSnapshot {
  return {
    featureConstraints: loadFeatureConstraints(),
    featureMissing: loadFeatureMissingSettings(),
    searchMethod: loadSearchMethod(),
    selectionCount: loadSelectionCountConstraint()
  };
}

function applyConversationDefaults(): void {
  saveFeatureConstraints([]);
  saveSelectionCountConstraint({ enabled: false, variables: [], k: 1 });
  saveFeatureMissingSettings({
    strategy: "drop",
    continuousStrategy: "mean",
    categoricalStrategy: "mode",
    imputeMaxIter: 10,
    imputeRandomState: null,
    multipleImputeSamplePosterior: false
  });
  saveSearchMethod("normal");
}

function restoreStoredRunSettings(snapshot: StoredRunSettingsSnapshot | null): void {
  if (!snapshot) return;
  saveFeatureConstraints(snapshot.featureConstraints);
  saveFeatureMissingSettings(snapshot.featureMissing);
  saveSearchMethod(snapshot.searchMethod);
  saveSelectionCountConstraint(snapshot.selectionCount);
}

function includesColumn(text: string, name: string): boolean {
  return text.toLocaleLowerCase().includes(name.toLocaleLowerCase());
}

export default function ConversationPage() {
  const {
    dataset,
    columns,
    selectableColumns,
    targetColumns,
    targetSettings,
    featureColumns,
    variables,
    q,
    result,
    busy,
    error,
    candidateSettingsValid,
    handleFile,
    toggleTarget,
    toggleFeature,
    patchTargetSetting,
    patchVariable,
    setNormalize,
    setInputPerturbation,
    setNW,
    setPerturbationStd,
    setProjectionDimensions,
    setModelType,
    setAcquisitionFamily,
    setAcquisition,
    setBeta,
    setFitMaxiter,
    crossValidation,
    setCrossValidation,
    featureImportance,
    setFeatureImportance,
    setQ,
    setNumRestarts,
    setRawSamples,
    execute,
    setError,
    setStep
  } = useWorkbench();

  const [stage, setStage] = useState<ConversationStage>(dataset ? "target" : "data");
  const [messages, setMessages] = useState<ConversationMessage[]>([
    nextMessage("assistant", dataset
      ? "データを確認しました。まず、良くしたい値を選びましょう。"
      : "次に試す実験条件を一緒に決めます。まず、CSVまたはExcelの実験データを読み込んでください。")
  ]);
  const [draftTarget, setDraftTarget] = useState(() => targetColumns[0] ?? "");
  const [draftDirection, setDraftDirection] = useState<Direction>(() => (
    targetColumns[0] ? targetSettings[targetColumns[0]]?.direction ?? "maximize" : "maximize"
  ));
  const [draftFeatures, setDraftFeatures] = useState<string[]>(() => [...featureColumns]);
  const [draftQ, setDraftQ] = useState(q);
  const [inputText, setInputText] = useState("");
  const [runRequested, setRunRequested] = useState(false);
  const [awaitingResult, setAwaitingResult] = useState(false);
  const previousResult = useRef(result);
  const storedRunSettings = useRef<StoredRunSettingsSnapshot | null>(null);
  const initializedDatasetId = useRef<string | null>(dataset?.dataset_id ?? null);

  const numericTargets = useMemo(
    () => columns.filter((column) => column.kind === "numeric"),
    [columns]
  );
  const featureCandidates = useMemo(
    () => selectableColumns.filter((column) => column.name !== draftTarget),
    [draftTarget, selectableColumns]
  );
  const resultTarget = draftTarget || result?.target_columns?.[0] || result?.target_column || "";
  const bestCandidate = useMemo(() => {
    if (!result?.candidates.length) return null;
    return [...result.candidates].sort((left, right) => left.rank - right.rank)[0];
  }, [result]);

  function append(role: MessageRole, text: string): void {
    setMessages((current) => [...current, nextMessage(role, text)]);
  }

  function resetConversation(nextDatasetLoaded = Boolean(dataset)): void {
    const initialTarget = targetColumns.find((name) => (
      columns.find((column) => column.name === name)?.kind === "numeric"
    )) ?? numericTargets.at(-1)?.name ?? "";
    const initialFeatures = featureColumns.filter((name) => name !== initialTarget);
    setDraftTarget(initialTarget);
    setDraftDirection(initialTarget ? targetSettings[initialTarget]?.direction ?? "maximize" : "maximize");
    setDraftFeatures(initialFeatures);
    setDraftQ(q);
    setStage(nextDatasetLoaded ? "target" : "data");
    setMessages([
      nextMessage("assistant", nextDatasetLoaded
        ? `「${dataset?.name ?? "データ"}」を確認しました。良くしたい値を選んでください。`
        : "次に試す実験条件を一緒に決めます。まず、CSVまたはExcelの実験データを読み込んでください。")
    ]);
    setInputText("");
  }

  useEffect(() => {
    const datasetId = dataset?.dataset_id ?? null;
    if (datasetId === initializedDatasetId.current) return;
    initializedDatasetId.current = datasetId;
    resetConversation(Boolean(dataset));
  }, [dataset?.dataset_id]);

  useEffect(() => {
    if (!runRequested) return;
    setRunRequested(false);
    if (!candidateSettingsValid) {
      restoreStoredRunSettings(storedRunSettings.current);
      storedRunSettings.current = null;
      setError("対話モードの設定では実行できません。目的変数、説明変数、または探索可能範囲を確認してください。");
      append("assistant", "設定を実行可能な形にできませんでした。右側の設定内容を確認するか、既存の画面で探索範囲を調整してください。");
      return;
    }
    previousResult.current = result;
    setAwaitingResult(true);
    append("assistant", "設定を確定しました。モデルを学習し、次に試す条件を計算します。");
    void execute("retrain").finally(() => {
      restoreStoredRunSettings(storedRunSettings.current);
      storedRunSettings.current = null;
    });
  }, [candidateSettingsValid, runRequested]);

  useEffect(() => {
    if (!awaitingResult || !result || result === previousResult.current) return;
    setAwaitingResult(false);
    setStage("result");
    append("assistant", `${result.candidates.length}件の候補を作成しました。最も推奨する条件を下に表示します。`);
  }, [awaitingResult, result]);

  useEffect(() => {
    if (!awaitingResult || !error) return;
    setAwaitingResult(false);
    setStage("confirm");
    append("assistant", "候補生成を完了できませんでした。表示されたエラーを確認して、設定を修正してください。");
  }, [awaitingResult, error]);

  function selectTarget(name: string): void {
    if (!name) return;
    targetColumns.filter((target) => target !== name).forEach(toggleTarget);
    if (!targetColumns.includes(name)) toggleTarget(name);
    patchTargetSetting(name, {
      task_type: "regression",
      optimize: true,
      direction: draftDirection,
      goal: "none",
      value: null,
      target_class: null,
      target_classes: [],
      class_order: [],
      target_values: []
    });
    setDraftTarget(name);
    setDraftFeatures((current) => current.filter((column) => column !== name));
    append("user", `${name}を良くしたいです。`);
    append("assistant", `了解しました。${name}を大きくするか、小さくするかを選んでください。`);
    setStage("direction");
  }

  function selectDirection(direction: Direction): void {
    if (!draftTarget) return;
    patchTargetSetting(draftTarget, {
      task_type: "regression",
      optimize: true,
      direction,
      goal: "none",
      value: null,
      target_class: null,
      target_classes: [],
      class_order: [],
      target_values: []
    });
    setDraftDirection(direction);
    append("user", direction === "maximize" ? "大きくしたいです。" : "小さくしたいです。");
    append("assistant", "次に、実験で変更できる条件を選んでください。選んだ列の値をbochanが提案します。");
    setStage("features");
  }

  function replaceFeatures(names: string[]): void {
    const desired = new Set(names.filter((name) => name !== draftTarget));
    featureCandidates.forEach((column) => {
      const selected = featureColumns.includes(column.name);
      if (selected !== desired.has(column.name)) toggleFeature(column.name);
      if (desired.has(column.name)) {
        patchVariable(column.name, simpleVariablePatch(column, dataset?.preview ?? [], variables[column.name]));
      }
    });
    setDraftFeatures([...desired]);
  }

  function toggleDraftFeature(name: string): void {
    const selected = draftFeatures.includes(name);
    const next = selected
      ? draftFeatures.filter((column) => column !== name)
      : [...draftFeatures, name];
    replaceFeatures(next);
  }

  function confirmFeatures(): void {
    if (draftFeatures.length === 0) {
      setError("変更できる条件を1つ以上選択してください。");
      return;
    }
    append("user", `${draftFeatures.join("、")}を変更できる条件として使います。`);
    append("assistant", "一度の候補生成で、何件の実験条件を提案しますか？");
    setStage("count");
  }

  function selectCandidateCount(value: number): void {
    const normalized = Math.min(20, Math.max(1, Math.trunc(value)));
    setDraftQ(normalized);
    setQ(normalized);
    append("user", `${normalized}件提案してください。`);
    append("assistant", "設定内容をまとめました。内容を確認して、候補生成を実行してください。");
    setStage("confirm");
  }

  function requestRun(): void {
    if (!dataset || !draftTarget || draftFeatures.length === 0 || busy) return;
    setError(null);
    storedRunSettings.current = captureStoredRunSettings();
    applyConversationDefaults();
    setNormalize(true);
    setInputPerturbation(false);
    setNW(16);
    setPerturbationStd(0.1);
    setProjectionDimensions(Math.min(2, Math.max(draftFeatures.length, 1)));
    setModelType("base");
    setAcquisitionFamily("bayesian_optimization");
    setAcquisition("EI");
    setBeta(2);
    setFitMaxiter(128);
    setCrossValidation({ ...crossValidation, enabled: false });
    setFeatureImportance({ ...featureImportance, enabled: false });
    setQ(draftQ);
    setNumRestarts(10);
    setRawSamples(256);
    window.setTimeout(() => setRunRequested(true), 0);
  }

  function openExistingScreen(step: "prepare" | "settings" | "optimize" | "results"): void {
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    setStep(step);
  }

  function handleTextSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const text = inputText.trim();
    if (!text) return;
    setInputText("");

    if (/やり直|最初|リセット/.test(text)) {
      append("user", text);
      resetConversation(Boolean(dataset));
      return;
    }

    if (stage === "target") {
      const matched = numericTargets.find((column) => includesColumn(text, column.name));
      if (matched) {
        selectTarget(matched.name);
        return;
      }
    }
    if (stage === "direction") {
      if (/最小|小さ|低く|減ら/.test(text)) {
        selectDirection("minimize");
        return;
      }
      if (/最大|大き|高く|増や/.test(text)) {
        selectDirection("maximize");
        return;
      }
    }
    if (stage === "features") {
      if (/すべて|全部/.test(text)) {
        replaceFeatures(featureCandidates.map((column) => column.name));
        append("user", text);
        append("assistant", "すべての候補列を変更できる条件として選択しました。問題なければ「この条件で進む」を押してください。");
        return;
      }
      const matched = featureCandidates
        .filter((column) => includesColumn(text, column.name))
        .map((column) => column.name);
      if (matched.length > 0) {
        replaceFeatures(matched);
        append("user", text);
        append("assistant", `${matched.join("、")}を選択しました。問題なければ「この条件で進む」を押してください。`);
        return;
      }
    }
    if (stage === "count") {
      const count = Number(text.match(/\d+/)?.[0]);
      if (Number.isFinite(count)) {
        selectCandidateCount(count);
        return;
      }
    }
    if (stage === "confirm" && /実行|提案|計算|進め/.test(text)) {
      append("user", text);
      requestRun();
      return;
    }

    append("user", text);
    append("assistant", "この入力から設定を確定できませんでした。下の選択肢を使うか、列名や「最大化」「3件」のように入力してください。");
  }

  return (
    <div className="conversation-page">
      <div className="conversation-heading">
        <div>
          <span className="conversation-kicker">GUIDED CONVERSATION</span>
          <h2>対話モード</h2>
          <p>質問に答えるだけで、既存のbochan設定を組み立てて次の実験候補を提案します。</p>
        </div>
        <div className="conversation-heading-actions">
          <button type="button" className="secondary" onClick={() => resetConversation(Boolean(dataset))}>
            最初からやり直す
          </button>
          {dataset && (
            <button type="button" className="secondary" onClick={() => openExistingScreen("prepare")}>
              画面で設定する
            </button>
          )}
        </div>
      </div>

      <div className="conversation-layout">
        <section className="conversation-thread" aria-label="bochanとの対話">
          <div className="conversation-messages" aria-live="polite">
            {messages.map((message) => (
              <div key={message.id} className={`conversation-message ${message.role}`}>
                <div className="conversation-avatar" aria-hidden="true">
                  {message.role === "assistant" ? "b" : "自"}
                </div>
                <div className="conversation-bubble">{message.text}</div>
              </div>
            ))}

            {stage === "data" && (
              <div className="conversation-action-card">
                <strong>実験データを読み込む</strong>
                <p>CSV、XLSX、XLSに対応しています。読み込み後も既存のData画面はそのまま利用できます。</p>
                <label className="conversation-file-button">
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={(event) => void handleFile(event.target.files?.[0] ?? null)}
                  />
                  ファイルを選択
                </label>
              </div>
            )}

            {stage === "target" && dataset && (
              <div className="conversation-action-card">
                <strong>良くしたい値を選ぶ</strong>
                <p>対話モードでは、まず数値の目的変数を対象にします。分類や順序回帰は既存の詳細画面から設定できます。</p>
                <div className="conversation-choice-grid">
                  {numericTargets.map((column) => (
                    <button key={column.name} type="button" className="secondary" onClick={() => selectTarget(column.name)}>
                      <strong>{column.name}</strong>
                      <small>範囲 {formatNumber(column.min)} ～ {formatNumber(column.max)}</small>
                    </button>
                  ))}
                </div>
                {numericTargets.length === 0 && (
                  <div className="conversation-warning">数値列が見つかりません。既存のSelect画面で目的変数の型を確認してください。</div>
                )}
              </div>
            )}

            {stage === "direction" && (
              <div className="conversation-action-card">
                <strong>{draftTarget}をどうしたいですか？</strong>
                <div className="conversation-choice-grid two-columns">
                  <button type="button" onClick={() => selectDirection("maximize")}>
                    <strong>大きくしたい</strong>
                    <small>高い値が期待できる条件を探す</small>
                  </button>
                  <button type="button" className="secondary" onClick={() => selectDirection("minimize")}>
                    <strong>小さくしたい</strong>
                    <small>低い値が期待できる条件を探す</small>
                  </button>
                </div>
              </div>
            )}

            {stage === "features" && (
              <div className="conversation-action-card">
                <strong>実験で変更できる条件</strong>
                <p>選択した条件だけをbochanが変更して候補を作ります。数値列は過去データの最小値から最大値までを探索します。</p>
                <div className="conversation-feature-list">
                  {featureCandidates.map((column) => {
                    const selected = draftFeatures.includes(column.name);
                    return (
                      <button
                        key={column.name}
                        type="button"
                        className={`conversation-feature ${selected ? "selected" : ""}`}
                        aria-pressed={selected}
                        onClick={() => toggleDraftFeature(column.name)}
                      >
                        <span className="conversation-check">{selected ? "✓" : ""}</span>
                        <span>
                          <strong>{column.name}</strong>
                          <small>{column.kind === "categorical" ? "カテゴリ条件" : `${formatNumber(column.min)} ～ ${formatNumber(column.max)}`}</small>
                        </span>
                      </button>
                    );
                  })}
                </div>
                <button type="button" disabled={draftFeatures.length === 0} onClick={confirmFeatures}>
                  この条件で進む
                </button>
              </div>
            )}

            {stage === "count" && (
              <div className="conversation-action-card">
                <strong>一度に提案する件数</strong>
                <div className="conversation-count-options">
                  {[1, 3, 5, 10].map((value) => (
                    <button key={value} type="button" className={value === 3 ? "" : "secondary"} onClick={() => selectCandidateCount(value)}>
                      {value}件
                    </button>
                  ))}
                </div>
                <label className="conversation-custom-count">
                  その他
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={draftQ}
                    onChange={(event) => setDraftQ(Math.min(20, Math.max(1, Number(event.target.value))))}
                  />
                  <button type="button" className="secondary" onClick={() => selectCandidateCount(draftQ)}>決定</button>
                </label>
              </div>
            )}

            {stage === "confirm" && (
              <div className="conversation-action-card conversation-confirm-card">
                <strong>この内容で候補を提案します</strong>
                <dl>
                  <div><dt>良くしたい値</dt><dd>{draftTarget}</dd></div>
                  <div><dt>方向</dt><dd>{draftDirection === "maximize" ? "大きくする" : "小さくする"}</dd></div>
                  <div><dt>変更する条件</dt><dd>{draftFeatures.join("、")}</dd></div>
                  <div><dt>提案件数</dt><dd>{draftQ}件</dd></div>
                  <div><dt>探索方針</dt><dd>改善と未調査領域のバランス</dd></div>
                  <div><dt>内部設定</dt><dd>Base GP + EI（推奨既定値）</dd></div>
                </dl>
                <button type="button" disabled={Boolean(busy)} onClick={requestRun}>
                  次の実験条件を提案
                </button>
                <button type="button" className="secondary" onClick={() => openExistingScreen("optimize")}>
                  詳細設定を確認
                </button>
              </div>
            )}

            {stage === "result" && result && bestCandidate && (
              <div className="conversation-action-card conversation-result-card">
                <span className="conversation-result-label">第1候補</span>
                <h3>次に試す推奨条件</h3>
                <div className="conversation-result-values">
                  {result.feature_columns.map((column) => (
                    <div key={column}><span>{column}</span><strong>{formatNumber(bestCandidate.values[column])}</strong></div>
                  ))}
                </div>
                <div className="conversation-prediction">
                  <span>{resultTarget || "目的値"}の予測値</span>
                  <strong>{formatNumber(resultTarget ? bestCandidate.predictions?.[resultTarget]?.mean : undefined)}</strong>
                  <small>予測標準偏差 {formatNumber(resultTarget ? bestCandidate.predictions?.[resultTarget]?.std : undefined)}</small>
                </div>
                <p>予測される改善とモデルの不確実性を合わせて評価し、最も優先度が高い候補として選びました。</p>
                <div className="conversation-result-actions">
                  <button type="button" onClick={() => openExistingScreen("results")}>すべての結果を見る</button>
                  <button type="button" className="secondary" onClick={() => setStage("confirm")}>条件を変えて再提案</button>
                </div>
              </div>
            )}
          </div>

          <form className="conversation-composer" onSubmit={handleTextSubmit}>
            <input
              type="text"
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              placeholder={dataset ? "例：強度を最大化、温度と時間を使う、3件提案" : "データ読込後に自然文でも回答できます"}
              disabled={!dataset || Boolean(busy)}
              aria-label="対話モードへの入力"
            />
            <button type="submit" disabled={!dataset || !inputText.trim() || Boolean(busy)}>送信</button>
          </form>
        </section>

        <aside className="conversation-summary" aria-label="現在の探索設定">
          <div className="conversation-summary-card">
            <span>Current plan</span>
            <h3>現在の探索設定</h3>
            <dl>
              <div><dt>データ</dt><dd>{dataset?.name ?? "未読込"}</dd></div>
              <div><dt>件数</dt><dd>{dataset ? `${dataset.profile.n_rows}行` : "—"}</dd></div>
              <div><dt>目的</dt><dd>{draftTarget || "未選択"}</dd></div>
              <div><dt>方向</dt><dd>{draftTarget ? draftDirection === "maximize" ? "最大化" : "最小化" : "—"}</dd></div>
              <div><dt>変更条件</dt><dd>{draftFeatures.length ? `${draftFeatures.length}列` : "未選択"}</dd></div>
              <div><dt>提案件数</dt><dd>{draftQ}件</dd></div>
            </dl>
          </div>

          <div className="conversation-summary-card conversation-progress-card">
            <span>Progress</span>
            <h3>対話の進行</h3>
            <ol>
              {[
                ["data", "データ読込"],
                ["target", "良くしたい値"],
                ["direction", "改善方向"],
                ["features", "変更できる条件"],
                ["count", "提案件数"],
                ["confirm", "内容確認"],
                ["result", "提案結果"]
              ].map(([id, label]) => {
                const order: ConversationStage[] = ["data", "target", "direction", "features", "count", "confirm", "result"];
                const currentIndex = order.indexOf(stage);
                const itemIndex = order.indexOf(id as ConversationStage);
                return (
                  <li key={id} className={itemIndex < currentIndex ? "complete" : itemIndex === currentIndex ? "active" : ""}>
                    <span>{itemIndex < currentIndex ? "✓" : itemIndex + 1}</span>{label}
                  </li>
                );
              })}
            </ol>
          </div>

          <div className="conversation-summary-card conversation-note-card">
            <span>About</span>
            <h3>既存画面との関係</h3>
            <p>対話モードは既存の設定状態とAPIをそのまま使用します。Data、Select、Model、Suggest、Resultsの各画面は変更せず、いつでも切り替えられます。</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
