import { useEffect, useMemo, useRef, useState } from "react";
import {
  type WorkbenchStep,
  useWorkbench
} from "../context/WorkbenchContext";
import {
  setWorkbenchMode,
  type WorkbenchMode
} from "../workbenchMode";
import { TUTORIAL_SAMPLE_DATASET_NAME } from "./sampleDataset";
import {
  TUTORIAL_VERSION,
  type TutorialKind,
  type TutorialProgress,
  readTutorialProgress,
  shouldPromptTutorial,
  writeTutorialProgress
} from "./tutorialStorage";
import "./tutorial.css";

type TutorialPhase = "hidden" | "prompt" | "tour";
type TutorialAdvance = "manual" | "sample_loaded" | "result_ready";

interface TutorialStep {
  id: string;
  selector: string;
  label: string;
  title: string;
  description: string;
  notes: string[];
  page?: WorkbenchStep;
  advance?: TutorialAdvance;
  waitingText?: string;
}

interface TutorialGuideProps {
  requestId: number;
  mode: WorkbenchMode;
  hasDataset: boolean;
  hasResult: boolean;
}

function clampStep(stepIndex: number, stepCount: number): number {
  if (stepCount <= 0) return 0;
  return Math.min(Math.max(0, stepIndex), stepCount - 1);
}

function clearAuxiliaryPage(): void {
  if (!window.location.hash) return;
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

function buildOverviewSteps(
  mode: WorkbenchMode,
  hasDataset: boolean,
  hasResult: boolean
): TutorialStep[] {
  const workflowDescription = mode === "simple"
    ? "簡易モードでは、Data → Select → Results の順に進みます。モデル設定と候補提案は、選択内容から自動構成されます。"
    : "詳細モードでは、Data → Select → Model → Suggest → Results の順に、モデルと獲得関数を明示的に設定します。";

  const workspaceDescription = !hasDataset
    ? "最初にData画面でCSVまたはExcelを読み込みます。読込後は、この領域で変数選択、モデル設定、候補確認を進めます。"
    : hasResult
      ? "候補生成まで完了しています。この領域で候補表、予測値、不確実性、可視化を確認できます。"
      : "データは読み込み済みです。この領域で目的変数・説明変数を選び、候補生成まで進めます。";

  const experimentDescription = hasResult
    ? "候補生成後はExperimentから実験結果を追加できます。追加後に再学習すると、次の候補へループできます。"
    : "候補生成が完了するとExperimentが有効になります。実験結果を追加し、再学習して次候補を得るための入口です。";

  return [
    {
      id: "workflow",
      selector: '[data-tutorial="workflow"]',
      label: "Workflow",
      title: "最適化の進行状況",
      description: workflowDescription,
      notes: [
        "完了した工程は状態表示が変わります。",
        "未準備の工程は無効化され、必要な設定が揃うと開けます。"
      ]
    },
    {
      id: "mode",
      selector: '[data-tutorial="mode"]',
      label: "Mode",
      title: "簡易モードと詳細モード",
      description: "簡易は設定項目を絞って実行し、詳細はモデル・獲得関数・候補生成条件を個別に調整します。",
      notes: [
        "初めて使う場合は簡易モードが適しています。",
        "切替状態は、このブラウザのlocalStorageに保存されます。"
      ]
    },
    {
      id: "navigation",
      selector: '[data-tutorial="navigation"]',
      label: "Navigation",
      title: "工程間の移動",
      description: experimentDescription,
      notes: [
        "Dataで読込後、Selectで目的変数と探索変数を指定します。",
        mode === "advanced"
          ? "ModelとSuggestでモデル・獲得関数・候補数を設定します。"
          : "簡易モードではModelとSuggestの設定を自動化します。"
      ]
    },
    {
      id: "workspace",
      selector: '[data-tutorial="workspace"]',
      label: "Workspace",
      title: "中央の作業領域",
      description: workspaceDescription,
      notes: [
        "入力エラーや注意事項も、この領域の上部に表示されます。",
        "処理中は進行表示が出るため、重複実行を避けられます。"
      ]
    },
    {
      id: "context",
      selector: '[data-tutorial="context"]',
      label: "Context",
      title: "現在の設定を確認",
      description: "右側にはAPI接続、読込データ、目的変数、モデル、獲得関数、候補数などの現在値が表示されます。",
      notes: [
        "設定変更後の確認に使えます。",
        "画面幅が狭い場合、この欄は自動的に非表示になります。"
      ]
    },
    {
      id: "status",
      selector: '[data-tutorial="status"]',
      label: "Status",
      title: "実行状態と再表示",
      description: "画面下部でAPI、モード、データ件数、候補件数を確認できます。チュートリアルは右上の「?」からいつでも再表示できます。",
      notes: [
        `チュートリアル状態のバージョンは ${TUTORIAL_VERSION} です。`,
        "表示済み情報だけをこのブラウザに保存し、サーバー側のユーザー管理は使用しません。"
      ]
    }
  ];
}

function buildSampleSteps(
  targetColumns: string[],
  featureColumns: string[]
): TutorialStep[] {
  const targetReady = targetColumns.length === 1 && targetColumns[0] === "strength";
  const selectedFeatures = ["temperature", "hold_time", "additive_ratio"]
    .filter((name) => featureColumns.includes(name));

  return [
    {
      id: "load-sample",
      selector: '[data-tutorial="sample-data"]',
      label: "1 · Data",
      title: "サンプルデータを読み込む",
      description: "材料の製造条件と強度を含む30行のCSVを、通常のファイル読込APIで読み込みます。",
      notes: [
        "temperature、hold_time、additive_ratioからstrengthを最大化します。",
        "既存データがある場合は、確認後に現在のワークスペースを置き換えます。"
      ],
      page: "data",
      advance: "sample_loaded",
      waitingText: "ハイライトされた「サンプルデータを読み込む」を押してください。"
    },
    {
      id: "confirm-target",
      selector: '[aria-label="目的変数"]',
      label: "2 · Target",
      title: "最大化する目的変数を確認",
      description: "CSVの最後の列であるstrengthが、目的変数として自動選択されています。簡易モードでは数値目的を最大化として扱います。",
      notes: [
        targetReady ? "strengthが選択済みです。" : "strengthが選択されていることを確認してください。",
        "目的変数はモデルが予測し、候補提案で改善を目指す値です。"
      ],
      page: "prepare"
    },
    {
      id: "confirm-features",
      selector: '[aria-label="説明変数"]',
      label: "3 · Features",
      title: "探索する説明変数を確認",
      description: "temperature、hold_time、additive_ratioが説明変数として選択され、観測データの最小値から最大値までが探索範囲になります。",
      notes: [
        `${selectedFeatures.length}/3列が選択済みです。`,
        "このチュートリアルではすべて数値変数として扱います。"
      ],
      page: "prepare"
    },
    {
      id: "run-optimization",
      selector: ".section-actions button",
      label: "4 · Optimize",
      title: "モデル学習と候補生成を実行",
      description: "「既定値で実行」を押すと、Base GPを学習し、EIによって次に試す3条件を提案します。",
      notes: [
        "通常のFastAPI・BoTorch処理をそのまま実行します。",
        "処理中は画面全体に進行表示が出て、完了後にResultsへ移動します。"
      ],
      page: "prepare",
      advance: "result_ready",
      waitingText: "ハイライトされた「既定値で実行」を押してください。"
    },
    {
      id: "review-candidates",
      selector: ".recommended-first",
      label: "5 · Results",
      title: "推奨候補を確認",
      description: "順位1から順に、次に実験する製造条件、strengthの予測値、予測標準偏差、獲得関数値が表示されます。",
      notes: [
        "予測値は期待される強度、標準偏差はモデルの不確実性です。",
        "EIは改善の大きさと不確実性を組み合わせて候補を選びます。"
      ],
      page: "results"
    },
    {
      id: "review-visualization",
      selector: ".interactive-visualization-section",
      label: "6 · Visualize",
      title: "予測モデルと探索空間を見る",
      description: "YY plotで学習データへの当てはまりを確認し、1次元・2次元プロットで条件による予測値や獲得関数の変化を確認します。",
      notes: [
        "右側の図は表示する変数や予測値／獲得関数を切り替えられます。",
        "ここまでが、データ読込から次実験候補を得る基本フローです。"
      ],
      page: "results"
    }
  ];
}

export default function TutorialGuide({
  requestId,
  mode,
  hasDataset,
  hasResult
}: TutorialGuideProps) {
  const {
    step,
    setStep,
    dataset,
    targetColumns,
    featureColumns,
    busy,
    result
  } = useWorkbench();
  const initialProgress = useMemo(() => readTutorialProgress(), []);
  const [savedProgress, setSavedProgress] = useState<TutorialProgress | null>(initialProgress);
  const [phase, setPhase] = useState<TutorialPhase>(() =>
    shouldPromptTutorial(initialProgress) ? "prompt" : "hidden"
  );
  const [tutorialKind, setTutorialKind] = useState<TutorialKind>(initialProgress?.kind ?? "overview");
  const [stepIndex, setStepIndex] = useState(initialProgress?.stepIndex ?? 0);
  const [sampleBaselineDatasetId, setSampleBaselineDatasetId] = useState<string | null>(null);
  const [sampleBaselineResultId, setSampleBaselineResultId] = useState<string | null>(null);
  const lastRequestId = useRef(requestId);
  const dialogRef = useRef<HTMLDivElement>(null);

  const overviewSteps = useMemo(
    () => buildOverviewSteps(mode, hasDataset, hasResult),
    [mode, hasDataset, hasResult]
  );
  const sampleSteps = useMemo(
    () => buildSampleSteps(targetColumns, featureColumns),
    [featureColumns, targetColumns]
  );
  const steps = tutorialKind === "sample" ? sampleSteps : overviewSteps;
  const currentIndex = clampStep(stepIndex, steps.length);
  const currentStep = steps[currentIndex];
  const isLastStep = currentIndex >= steps.length - 1;
  const isTutorialDataset = dataset?.name === TUTORIAL_SAMPLE_DATASET_NAME;
  const datasetId = dataset?.dataset_id ?? null;
  const resultId = result?.visualization_run_id ?? null;
  const isTutorialResult = isTutorialDataset && result?.dataset_name === TUTORIAL_SAMPLE_DATASET_NAME;
  const isResume = savedProgress?.version === TUTORIAL_VERSION && savedProgress.status === "in_progress";
  const waitingForAction = currentStep.advance === "sample_loaded" || currentStep.advance === "result_ready";

  useEffect(() => {
    if (requestId === lastRequestId.current) return;
    lastRequestId.current = requestId;
    setPhase("prompt");
  }, [requestId]);

  useEffect(() => {
    setStepIndex((current) => clampStep(current, steps.length));
  }, [steps.length]);

  useEffect(() => {
    if (phase !== "tour") return;
    const progress = writeTutorialProgress("in_progress", currentIndex, tutorialKind);
    setSavedProgress(progress);
  }, [currentIndex, phase, tutorialKind]);

  useEffect(() => {
    if (phase === "hidden") return;
    dialogRef.current?.focus();
  }, [phase, currentIndex]);

  useEffect(() => {
    if (phase !== "tour" || tutorialKind !== "sample" || !currentStep.page) return;
    if (step === currentStep.page) return;
    clearAuxiliaryPage();
    setStep(currentStep.page);
  }, [currentStep.page, phase, setStep, step, tutorialKind]);

  useEffect(() => {
    if (phase !== "tour" || !currentStep) return;

    const timeoutId = window.setTimeout(() => {
      const target = document.querySelector<HTMLElement>(currentStep.selector);
      if (!target) return;
      target.classList.add("tutorial-focus-target");
      target.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    }, 80);

    return () => {
      window.clearTimeout(timeoutId);
      document.querySelectorAll(".tutorial-focus-target")
        .forEach((target) => target.classList.remove("tutorial-focus-target"));
    };
  }, [currentStep, phase, step]);

  useEffect(() => {
    if (
      phase === "tour" &&
      tutorialKind === "sample" &&
      currentStep.id === "load-sample" &&
      isTutorialDataset &&
      datasetId !== sampleBaselineDatasetId &&
      !busy
    ) {
      setStepIndex(1);
    }
  }, [
    busy,
    currentStep.id,
    datasetId,
    isTutorialDataset,
    phase,
    sampleBaselineDatasetId,
    tutorialKind
  ]);

  useEffect(() => {
    if (
      phase === "tour" &&
      tutorialKind === "sample" &&
      currentStep.id === "run-optimization" &&
      isTutorialResult &&
      resultId !== sampleBaselineResultId &&
      !busy
    ) {
      setStepIndex(4);
    }
  }, [
    busy,
    currentStep.id,
    isTutorialResult,
    phase,
    resultId,
    sampleBaselineResultId,
    tutorialKind
  ]);

  useEffect(() => {
    if (phase === "hidden") return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPhase("hidden");
        return;
      }
      if (phase !== "tour") return;
      if (event.key === "ArrowLeft") {
        goBack();
      }
      if (event.key === "ArrowRight" && !waitingForAction) {
        goNext();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  function startTutorial(kind: TutorialKind, resume: boolean) {
    let nextStepIndex = 0;
    if (resume && savedProgress?.kind === kind) {
      nextStepIndex = clampStep(savedProgress.stepIndex, kind === "sample" ? sampleSteps.length : overviewSteps.length);
      if (kind === "sample") {
        if (!isTutorialDataset) nextStepIndex = 0;
        else if (nextStepIndex >= 4 && !isTutorialResult) nextStepIndex = 3;
      }
    }

    setTutorialKind(kind);
    setStepIndex(nextStepIndex);
    setSampleBaselineDatasetId(datasetId);
    setSampleBaselineResultId(resultId);
    setSavedProgress(writeTutorialProgress("in_progress", nextStepIndex, kind));

    if (kind === "sample") {
      setWorkbenchMode("simple");
      const destination = buildSampleSteps(targetColumns, featureColumns)[nextStepIndex]?.page ?? "data";
      clearAuxiliaryPage();
      setStep(destination);
    }

    setPhase("tour");
  }

  function dismissTutorial() {
    setSavedProgress(writeTutorialProgress("dismissed", 0, tutorialKind));
    setPhase("hidden");
  }

  function finishTutorial() {
    setSavedProgress(writeTutorialProgress("completed", steps.length - 1, tutorialKind));
    setPhase("hidden");
  }

  function goBack() {
    if (currentIndex === 0) return;
    const nextIndex = currentIndex - 1;
    if (tutorialKind === "sample" && nextIndex === 0) {
      setSampleBaselineDatasetId(datasetId);
    }
    if (tutorialKind === "sample" && nextIndex === 3) {
      setSampleBaselineResultId(resultId);
    }
    setStepIndex(nextIndex);
  }

  function goNext() {
    if (waitingForAction) return;
    if (isLastStep) {
      finishTutorial();
      return;
    }
    setStepIndex(currentIndex + 1);
  }

  if (phase === "hidden") return null;

  if (phase === "prompt") {
    const resumeSampleFromStart = savedProgress?.kind === "sample" && !isTutorialDataset;
    return (
      <div className="tutorial-prompt-backdrop">
        <div
          ref={dialogRef}
          className="tutorial-prompt-card tutorial-menu-card"
          role="dialog"
          aria-modal="true"
          aria-labelledby="tutorial-prompt-title"
          tabIndex={-1}
        >
          <div className="tutorial-prompt-icon" aria-hidden="true">?</div>
          <span className="tutorial-eyebrow">Getting started</span>
          <h2 id="tutorial-prompt-title">どのチュートリアルを開始しますか？</h2>
          <p>実際に候補を生成する実践形式と、画面構成だけを確認するガイドを選べます。</p>

          {isResume && (
            <button
              type="button"
              className="tutorial-resume-button"
              onClick={() => startTutorial(savedProgress.kind, true)}
            >
              <strong>
                {savedProgress.kind === "sample"
                  ? resumeSampleFromStart ? "サンプルを最初から再開" : "サンプルの続きから再開"
                  : "画面案内の続きから再開"}
              </strong>
              <span>前回は {savedProgress.stepIndex + 1} ステップ目まで進みました。</span>
            </button>
          )}

          <div className="tutorial-choice-grid">
            <button
              type="button"
              className="tutorial-choice-card primary-choice"
              onClick={() => startTutorial("sample", false)}
            >
              <span className="tutorial-choice-icon" aria-hidden="true">↗</span>
              <strong>サンプルで最適化を体験</strong>
              <small>データ読込 → 変数確認 → 候補生成 → 結果・グラフ確認</small>
              <em>おすすめ · 6ステップ</em>
            </button>
            <button
              type="button"
              className="tutorial-choice-card"
              onClick={() => startTutorial("overview", false)}
            >
              <span className="tutorial-choice-icon" aria-hidden="true">◇</span>
              <strong>画面構成だけ確認</strong>
              <small>ワークフロー、モード、ナビゲーション、設定確認欄を案内</small>
              <em>操作なし · 6ステップ</em>
            </button>
          </div>

          <div className="tutorial-privacy-note">
            {hasDataset && "サンプルデータの読込を選ぶと、確認後に現在のワークスペースを置き換えます。"}
            {!hasDataset && "サンプルデータはブラウザで生成し、通常のCSV読込APIへ送信します。"}
            <br />進捗情報だけを、このPCの現在のブラウザに保存します。
          </div>
          <div className="tutorial-prompt-actions">
            <button type="button" className="secondary" onClick={() => setPhase("hidden")}>後で</button>
          </div>
          <button type="button" className="tutorial-dismiss-button" onClick={dismissTutorial}>
            今後は自動表示しない
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={dialogRef}
      className={`tutorial-guide-card tutorial-kind-${tutorialKind}`}
      role="dialog"
      aria-modal="false"
      aria-labelledby="tutorial-guide-title"
      tabIndex={-1}
    >
      <div className="tutorial-guide-header">
        <div>
          <span className="tutorial-eyebrow">{currentStep.label}</span>
          <h2 id="tutorial-guide-title">{currentStep.title}</h2>
        </div>
        <button
          type="button"
          className="tutorial-close-button"
          aria-label="チュートリアルを閉じる"
          title="閉じる"
          onClick={() => setPhase("hidden")}
        >
          ×
        </button>
      </div>

      <p>{currentStep.description}</p>
      <ul>
        {currentStep.notes.map((note) => <li key={note}>{note}</li>)}
      </ul>

      {waitingForAction && (
        <div className={`tutorial-action-note ${busy ? "busy" : ""}`}>
          <span aria-hidden="true">{busy ? "…" : "→"}</span>
          <strong>{busy ? "処理が完了するまでお待ちください。" : currentStep.waitingText}</strong>
        </div>
      )}

      <div className="tutorial-progress" aria-label={`${currentIndex + 1} / ${steps.length}`}>
        {steps.map((tutorialStep, index) => (
          <span
            key={tutorialStep.id}
            className={index === currentIndex ? "active" : index < currentIndex ? "complete" : ""}
          />
        ))}
      </div>

      <div className="tutorial-guide-footer">
        <span>{currentIndex + 1} / {steps.length}</span>
        <div>
          <button
            type="button"
            className="secondary"
            disabled={currentIndex === 0 || Boolean(busy)}
            onClick={goBack}
          >
            戻る
          </button>
          <button type="button" disabled={waitingForAction || Boolean(busy)} onClick={goNext}>
            {waitingForAction ? busy ? "処理中" : "操作待ち" : isLastStep ? "完了" : "次へ"}
          </button>
        </div>
      </div>
    </div>
  );
}
