import { useEffect, useMemo, useRef, useState } from "react";
import type { WorkbenchMode } from "../workbenchMode";
import {
  TUTORIAL_VERSION,
  readTutorialProgress,
  shouldPromptTutorial,
  writeTutorialProgress
} from "./tutorialStorage";
import "./tutorial.css";

type TutorialPhase = "hidden" | "prompt" | "tour";

interface TutorialStep {
  id: string;
  selector: string;
  label: string;
  title: string;
  description: string;
  notes: string[];
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

function buildTutorialSteps(
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

export default function TutorialGuide({
  requestId,
  mode,
  hasDataset,
  hasResult
}: TutorialGuideProps) {
  const steps = useMemo(
    () => buildTutorialSteps(mode, hasDataset, hasResult),
    [mode, hasDataset, hasResult]
  );
  const [savedProgress] = useState(readTutorialProgress);
  const [phase, setPhase] = useState<TutorialPhase>(() =>
    shouldPromptTutorial(savedProgress) ? "prompt" : "hidden"
  );
  const [stepIndex, setStepIndex] = useState(() =>
    savedProgress?.version === TUTORIAL_VERSION && savedProgress.status === "in_progress"
      ? clampStep(savedProgress.stepIndex, steps.length)
      : 0
  );
  const [persistProgress, setPersistProgress] = useState(false);
  const lastRequestId = useRef(requestId);
  const dialogRef = useRef<HTMLDivElement>(null);

  const currentStep = steps[clampStep(stepIndex, steps.length)];
  const isLastStep = stepIndex >= steps.length - 1;
  const isResume = savedProgress?.version === TUTORIAL_VERSION && savedProgress.status === "in_progress";

  useEffect(() => {
    if (requestId === lastRequestId.current) return;
    lastRequestId.current = requestId;
    setPersistProgress(false);
    setStepIndex(0);
    setPhase("tour");
  }, [requestId]);

  useEffect(() => {
    setStepIndex((current) => clampStep(current, steps.length));
  }, [steps.length]);

  useEffect(() => {
    if (phase !== "tour" || !persistProgress) return;
    writeTutorialProgress("in_progress", stepIndex);
  }, [phase, persistProgress, stepIndex]);

  useEffect(() => {
    if (phase === "hidden") return;
    dialogRef.current?.focus();
  }, [phase, stepIndex]);

  useEffect(() => {
    if (phase !== "tour" || !currentStep) return;

    const target = document.querySelector<HTMLElement>(currentStep.selector);
    if (!target) return;

    target.classList.add("tutorial-focus-target");
    target.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });

    return () => {
      target.classList.remove("tutorial-focus-target");
    };
  }, [currentStep, phase]);

  useEffect(() => {
    if (phase === "hidden") return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPhase("hidden");
        return;
      }
      if (phase !== "tour") return;
      if (event.key === "ArrowLeft") {
        setStepIndex((current) => Math.max(0, current - 1));
      }
      if (event.key === "ArrowRight") {
        if (isLastStep) {
          writeTutorialProgress("completed", steps.length - 1);
          setPhase("hidden");
        } else {
          setStepIndex((current) => Math.min(steps.length - 1, current + 1));
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isLastStep, phase, steps.length]);

  function startTutorial() {
    const resumeStep = isResume ? clampStep(savedProgress.stepIndex, steps.length) : 0;
    setPersistProgress(true);
    setStepIndex(resumeStep);
    setPhase("tour");
  }

  function dismissTutorial() {
    writeTutorialProgress("dismissed", 0);
    setPhase("hidden");
  }

  function finishTutorial() {
    writeTutorialProgress("completed", steps.length - 1);
    setPhase("hidden");
  }

  function goNext() {
    if (isLastStep) {
      finishTutorial();
      return;
    }
    setStepIndex((current) => Math.min(steps.length - 1, current + 1));
  }

  if (phase === "hidden") return null;

  if (phase === "prompt") {
    return (
      <div className="tutorial-prompt-backdrop">
        <div
          ref={dialogRef}
          className="tutorial-prompt-card"
          role="dialog"
          aria-modal="true"
          aria-labelledby="tutorial-prompt-title"
          tabIndex={-1}
        >
          <div className="tutorial-prompt-icon" aria-hidden="true">?</div>
          <span className="tutorial-eyebrow">Getting started</span>
          <h2 id="tutorial-prompt-title">
            {isResume ? "チュートリアルを続けますか？" : "bochanの使い方を確認しますか？"}
          </h2>
          <p>
            約6ステップで、実行モード、ワークフロー、作業領域、設定確認、実験結果追加の流れを案内します。
          </p>
          <div className="tutorial-privacy-note">
            チュートリアルの表示状態だけを、このPCの現在のブラウザに保存します。
          </div>
          <div className="tutorial-prompt-actions">
            <button type="button" onClick={startTutorial}>
              {isResume ? "続きから再開" : "チュートリアルを開始"}
            </button>
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
      className="tutorial-guide-card"
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

      <div className="tutorial-progress" aria-label={`${stepIndex + 1} / ${steps.length}`}>
        {steps.map((step, index) => (
          <span key={step.id} className={index === stepIndex ? "active" : index < stepIndex ? "complete" : ""} />
        ))}
      </div>

      <div className="tutorial-guide-footer">
        <span>{stepIndex + 1} / {steps.length}</span>
        <div>
          <button
            type="button"
            className="secondary"
            disabled={stepIndex === 0}
            onClick={() => setStepIndex((current) => Math.max(0, current - 1))}
          >
            戻る
          </button>
          <button type="button" onClick={goNext}>
            {isLastStep ? "完了" : "次へ"}
          </button>
        </div>
      </div>
    </div>
  );
}
