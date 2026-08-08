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

function isPracticalTutorial(kind: TutorialKind): boolean {
  return kind === "sample" || kind === "advanced";
}

function executeStepIndex(kind: TutorialKind): number {
  return kind === "advanced" ? 10 : 3;
}

function resultStepIndex(kind: TutorialKind): number {
  return kind === "advanced" ? 11 : 4;
}

function tutorialName(kind: TutorialKind): string {
  if (kind === "advanced") return "詳細設定チュートリアル";
  if (kind === "sample") return "簡易サンプルチュートリアル";
  return "画面案内チュートリアル";
}

function buildOverviewSteps(
  mode: WorkbenchMode,
  hasDataset: boolean,
  hasResult: boolean
): TutorialStep[] {
  const workflowDescription = mode === "simple"
    ? "簡易モードでは、Data → Select → Results の順に進みます。モデル設定と候補提案は選択内容から自動構成され、上部には現在位置だけを表示します。"
    : "詳細モードでは、Data → Select → Model → Suggest → Results の順に進みます。上部には現在位置を表示し、工程間の移動は左側のナビゲーションから行います。";

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
      title: "現在の工程を確認",
      description: workflowDescription,
      notes: [
        "上部のバーは進捗確認専用です。工程の移動には左側のナビゲーションを使います。",
        "左側では、現在の工程・完了済みの工程・まだ開けない工程を確認できます。"
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
        "Dataで読込後、Selectで目的変数と探索変数を指定します。工程間の移動はこの左ナビゲーションから行います。",
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
      title: "目的変数と方向を確認",
      description: "CSVの最後の列であるstrengthが目的変数として自動選択されています。簡易モードでは最大化・最小化を切り替えられ、このサンプルでは最大化のまま進めます。",
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
      title: "変更できる条件を確認",
      description: "temperature、hold_time、additive_ratioが説明変数として選択され、観測データの最小値から最大値までが探索範囲になります。簡易モードでは数値／カテゴリをデータから自動判定します。",
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
      title: "候補を提案",
      description: "「候補を提案」を押すと、bochanがBase GP、EI、入力正規化、BoTorch探索を自動設定し、次に試す3条件を提案します。",
      notes: [
        "通常のFastAPI・BoTorch処理をそのまま実行します。",
        "処理中は画面全体に進行表示が出て、完了後にResultsへ移動します。"
      ],
      page: "prepare",
      advance: "result_ready",
      waitingText: "ハイライトされた「候補を提案」を押してください。"
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

function buildAdvancedSteps(
  targetColumns: string[],
  featureColumns: string[],
  modelType: string,
  acquisition: string,
  q: number
): TutorialStep[] {
  const targetReady = targetColumns.length === 1 && targetColumns[0] === "strength";
  const selectedFeatures = ["temperature", "hold_time", "additive_ratio"]
    .filter((name) => featureColumns.includes(name));

  return [
    {
      id: "load-sample",
      selector: '[data-tutorial="sample-data"]',
      label: "1 · Data",
      title: "詳細設定用サンプルを読み込む",
      description: "材料条件から強度を最大化する30行のCSVを読み込み、詳細モードの全設定を順番に確認します。",
      notes: [
        "サンプルは通常のCSV読込APIを通るため、実データと同じ状態が作られます。",
        "既存データがある場合は、確認後に現在のワークスペースを置き換えます。"
      ],
      page: "data",
      advance: "sample_loaded",
      waitingText: "ハイライトされた「サンプルデータを読み込む」を押してください。"
    },
    {
      id: "advanced-variables",
      selector: ".selection-grid",
      label: "2 · Select",
      title: "目的変数と説明変数を確認",
      description: "strengthを目的変数、temperature・hold_time・additive_ratioを説明変数として使用します。",
      notes: [
        targetReady ? "strengthが目的変数として選択済みです。" : "strengthを目的変数として選択してください。",
        `${selectedFeatures.length}/3列の説明変数が選択済みです。`
      ],
      page: "prepare"
    },
    {
      id: "advanced-target-model",
      selector: ".target-model-settings-wrap",
      label: "3 · Task",
      title: "目的変数のタスクを設定",
      description: "strengthは連続値なので回帰を使用します。分類では正例クラス、順序回帰ではクラス順もここで定義します。",
      notes: [
        "タスク設定は学習するモデルの出力形式を決めます。",
        "このサンプルでは回帰のまま進めます。"
      ],
      page: "settings"
    },
    {
      id: "advanced-surrogate",
      selector: ".model-selection-panel",
      label: "4 · Model",
      title: "代理モデルを選択",
      description: `現在は${modelType}が選択されています。モデルの大分類、種類、学習反復数を変更できます。`,
      notes: [
        "Base GPは少量の連続データに対する標準的な選択です。",
        "PCAやREMBOでは射影次元、Multitaskでは複数目的の条件が追加されます。"
      ],
      page: "settings"
    },
    {
      id: "advanced-transform",
      selector: ".search-transform-grid",
      label: "5 · Transform",
      title: "正規化と入力摂動を設定",
      description: "正規化は探索範囲を基準に説明変数のスケールを揃えます。入力摂動は条件ばらつきを考慮した頑健な候補評価に使います。",
      notes: [
        "このサンプルでは正規化を有効、入力摂動を無効のまま進められます。",
        "入力摂動を使う場合はサンプル数nと標準偏差を設定します。"
      ],
      page: "settings"
    },
    {
      id: "advanced-missing",
      selector: ".feature-missing-section",
      label: "6 · Missing",
      title: "欠損値処理を選択",
      description: "説明変数に欠損がある場合、欠損行の削除または補完を選択します。",
      notes: [
        "サンプルデータには欠損がないため、既定の欠損行削除で問題ありません。",
        "補完では数値変数に平均値またはIterativeImputerを使用できます。"
      ],
      page: "settings"
    },
    {
      id: "advanced-objective",
      selector: ".proposal-target-table",
      label: "7 · Objective",
      title: "最適化方向と目的制約を設定",
      description: "strengthを最適化対象として最大化します。必要に応じて最小化、目標値、以上・以下の実行可能性制約へ変更できます。",
      notes: [
        "モデル学習のタスクと、候補提案で狙う方向は別の設定です。",
        "制約専用の目的変数は最適化対象のチェックを外して使用します。"
      ],
      page: "optimize"
    },
    {
      id: "advanced-methods",
      selector: ".suggestion-method-grid",
      label: "8 · Strategy",
      title: "獲得関数・探索手法・候補数を設定",
      description: `現在は獲得関数${acquisition}、候補数q=${q}です。獲得関数は候補の評価基準、探索手法はその基準を最大化する方法です。`,
      notes: [
        "単目的ベイズ最適化ではEI・PI・UCBを選択できます。",
        "通常のBoTorch探索に加え、GA・PSO・CMA-ES・Thompson samplingも選択できます。"
      ],
      page: "optimize"
    },
    {
      id: "advanced-search-space",
      selector: ".search-variable-table",
      label: "9 · Bounds",
      title: "探索範囲・刻み・固定値を設定",
      description: "各説明変数の下限と上限を確認し、必要に応じて刻みや固定値を指定します。",
      notes: [
        "初期値には観測データの最小値と最大値が設定されています。",
        "製造可能範囲が観測範囲と異なる場合は、ここで実際の上下限へ修正します。"
      ],
      page: "optimize"
    },
    {
      id: "advanced-constraints",
      selector: ".feature-constraint-panel",
      label: "10 · Constraints",
      title: "説明変数の候補制約を確認",
      description: "変数の重み付き和に対する制約や、有効にする変数数の制約を設定できます。",
      notes: [
        "このサンプルでは制約なしのまま実行します。",
        "配合合計、設備上限、使用材料数などの条件を候補生成へ反映できます。"
      ],
      page: "optimize"
    },
    {
      id: "advanced-execute",
      selector: ".train-launcher",
      label: "11 · Run",
      title: "詳細設定で候補を生成",
      description: "検証表示がReadyであることを確認し、モデルを学習して候補を生成します。",
      notes: [
        "設定に矛盾がある場合は、検証欄に修正対象が表示されます。",
        "実行後は自動的にResults画面へ移動します。"
      ],
      page: "optimize",
      advance: "result_ready",
      waitingText: "「モデルを学習して候補を生成」を押してください。"
    },
    {
      id: "advanced-results",
      selector: ".recommended-first",
      label: "12 · Results",
      title: "詳細設定による推奨候補を確認",
      description: "候補ごとの説明変数、strengthの予測値、標準偏差、獲得関数値、制約判定を確認します。",
      notes: [
        "順位1が現在の設定で最も推奨される候補です。",
        "制約を設定した場合は条件列のOK・NGも確認します。"
      ],
      page: "results"
    },
    {
      id: "advanced-visualization",
      selector: ".interactive-visualization-section",
      label: "13 · Visualize",
      title: "モデル評価と探索空間を確認",
      description: "YY plotでモデルの当てはまりを確認し、予測値または獲得関数を1次元・2次元で表示します。",
      notes: [
        "図に使わない変数は固定値を変更して断面を比較できます。",
        "詳細設定を変更して再実行すると、候補と可視化の違いを比較できます。"
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
    modelType,
    acquisition,
    q,
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
  const advancedSteps = useMemo(
    () => buildAdvancedSteps(targetColumns, featureColumns, modelType, acquisition, q),
    [acquisition, featureColumns, modelType, q, targetColumns]
  );
  const steps = tutorialKind === "advanced"
    ? advancedSteps
    : tutorialKind === "sample"
      ? sampleSteps
      : overviewSteps;
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
    if (phase !== "tour" || !isPracticalTutorial(tutorialKind) || !currentStep.page) return;
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
    }, 100);

    return () => {
      window.clearTimeout(timeoutId);
      document.querySelectorAll(".tutorial-focus-target")
        .forEach((target) => target.classList.remove("tutorial-focus-target"));
    };
  }, [currentStep, phase, step]);

  useEffect(() => {
    if (
      phase === "tour" &&
      isPracticalTutorial(tutorialKind) &&
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
      isPracticalTutorial(tutorialKind) &&
      currentStep.advance === "result_ready" &&
      isTutorialResult &&
      resultId !== sampleBaselineResultId &&
      !busy
    ) {
      setStepIndex(resultStepIndex(tutorialKind));
    }
  }, [
    busy,
    currentStep.advance,
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
      if (event.key === "ArrowLeft") goBack();
      if (event.key === "ArrowRight" && !waitingForAction) goNext();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  function stepsForKind(kind: TutorialKind): TutorialStep[] {
    if (kind === "advanced") return advancedSteps;
    if (kind === "sample") return sampleSteps;
    return overviewSteps;
  }

  function startTutorial(kind: TutorialKind, resume: boolean) {
    const selectedSteps = stepsForKind(kind);
    let nextStepIndex = 0;
    if (resume && savedProgress?.kind === kind) {
      nextStepIndex = clampStep(savedProgress.stepIndex, selectedSteps.length);
      if (isPracticalTutorial(kind)) {
        if (!isTutorialDataset) nextStepIndex = 0;
        else if (nextStepIndex >= resultStepIndex(kind) && !isTutorialResult) {
          nextStepIndex = executeStepIndex(kind);
        }
      }
    }

    setTutorialKind(kind);
    setStepIndex(nextStepIndex);
    setSampleBaselineDatasetId(datasetId);
    setSampleBaselineResultId(resultId);
    setSavedProgress(writeTutorialProgress("in_progress", nextStepIndex, kind));

    if (isPracticalTutorial(kind)) {
      setWorkbenchMode(kind === "advanced" ? "advanced" : "simple");
      const destination = selectedSteps[nextStepIndex]?.page ?? "data";
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
    if (isPracticalTutorial(tutorialKind) && nextIndex === 0) {
      setSampleBaselineDatasetId(datasetId);
    }
    if (isPracticalTutorial(tutorialKind) && nextIndex === executeStepIndex(tutorialKind)) {
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
    const resumeFromStart = isPracticalTutorial(savedProgress?.kind ?? "overview") && !isTutorialDataset;
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
          <p>簡易実行、詳細設定、画面構成の3種類から選択できます。</p>

          {isResume && savedProgress && (
            <button
              type="button"
              className="tutorial-resume-button"
              onClick={() => startTutorial(savedProgress.kind, true)}
            >
              <strong>{resumeFromStart ? `${tutorialName(savedProgress.kind)}を最初から再開` : `${tutorialName(savedProgress.kind)}の続きから再開`}</strong>
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
              <strong>簡易モードで最適化を体験</strong>
              <small>データ読込 → 目的・条件を選択 → 候補を提案 → 結果確認</small>
              <em>入門 · 6ステップ</em>
            </button>
            <button
              type="button"
              className="tutorial-choice-card advanced-choice"
              onClick={() => startTutorial("advanced", false)}
            >
              <span className="tutorial-choice-icon" aria-hidden="true">⌘</span>
              <strong>詳細設定で最適化を体験</strong>
              <small>タスク、モデル、前処理、獲得関数、探索範囲、制約まで設定</small>
              <em>詳細 · 13ステップ</em>
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
            {hasDataset && "実践チュートリアルでサンプルデータを読み込むと、確認後に現在のワークスペースを置き換えます。"}
            {!hasDataset && "実践チュートリアルのサンプルデータはブラウザで生成し、通常のCSV読込APIへ送信します。"}
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

      <div
        className="tutorial-progress"
        aria-label={`${currentIndex + 1} / ${steps.length}`}
        style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}
      >
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
