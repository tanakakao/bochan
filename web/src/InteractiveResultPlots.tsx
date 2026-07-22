import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { fetchResultVisualization } from "./api";
import { useWorkbench } from "./context/WorkbenchContext";
import { themedPlotLayout } from "./plotLayout";
import type {
  RegressionResult,
  ResultVisualization,
  VisualizationKind,
  VisualizationOptions
} from "./types";

interface Props {
  result: RegressionResult;
}

function PlotCard({
  visualization,
  loading,
  error
}: {
  visualization: ResultVisualization | null;
  loading: boolean;
  error: string | null;
}) {
  const { theme } = useWorkbench();
  return (
    <div className="interactive-plot-stage">
      {loading && <div className="plot-loading"><span className="spinner" />図を生成しています...</div>}
      {error && !loading && <div className="alert warning plot-error">{error}</div>}
      {visualization && !loading && !error && (
        <>
          <div className="visualization-heading">
            <h3>{visualization.title}</h3>
            <p>{visualization.description}</p>
          </div>
          <div className="plot-container">
            <Plot
              data={visualization.figure.data as Data[]}
              layout={themedPlotLayout(visualization.figure.layout, theme)}
              config={{ responsive: true, displaylogo: false }}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
            />
          </div>
        </>
      )}
    </div>
  );
}

function defaults(result: RegressionResult): VisualizationOptions {
  return result.visualization_options ?? {
    feature_columns: result.feature_columns,
    numeric_features: result.feature_columns,
    target_columns: result.target_columns,
    regression_targets: result.target_columns,
    ternary_groups: []
  };
}

/** Two-column interactive Plotly area using fitted FastAPI visualization sessions. */
export default function InteractiveResultPlots({ result }: Props) {
  const options = useMemo(() => defaults(result), [result]);
  const runId = result.visualization_run_id;
  const [leftKind, setLeftKind] = useState<"yyplot" | "pareto">(
    options.regression_targets.length >= 2 ? "pareto" : "yyplot"
  );
  const [leftTarget, setLeftTarget] = useState(options.target_columns[0] ?? "");
  const [targetX, setTargetX] = useState(options.regression_targets[0] ?? "");
  const [targetY, setTargetY] = useState(options.regression_targets[1] ?? options.regression_targets[0] ?? "");
  const [rightKind, setRightKind] = useState<"1d" | "2d" | "ternary">("1d");
  const [rightTarget, setRightTarget] = useState(options.target_columns[0] ?? "");
  const [featureA, setFeatureA] = useState(options.feature_columns[0] ?? "");
  const [featureB, setFeatureB] = useState(options.numeric_features[1] ?? options.numeric_features[0] ?? "");
  const [featureC, setFeatureC] = useState(options.numeric_features[2] ?? options.numeric_features[0] ?? "");
  const [showType, setShowType] = useState<"pred" | "acqf">("pred");
  const initialGroup = options.ternary_groups?.[0];
  const [sumValue, setSumValue] = useState(initialGroup?.sum_value ?? 1);
  const [leftPlot, setLeftPlot] = useState<ResultVisualization | null>(null);
  const [rightPlot, setRightPlot] = useState<ResultVisualization | null>(null);
  const [leftLoading, setLeftLoading] = useState(false);
  const [rightLoading, setRightLoading] = useState(false);
  const [leftError, setLeftError] = useState<string | null>(null);
  const [rightError, setRightError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      setLeftError("可視化セッションがありません。候補を再生成してください。");
      return;
    }
    if (leftKind === "pareto" && (!targetX || !targetY || targetX === targetY)) {
      setLeftError("Pareto図には異なる2つの回帰目的変数が必要です。");
      return;
    }
    let active = true;
    setLeftLoading(true);
    setLeftError(null);
    fetchResultVisualization(runId, leftKind === "yyplot"
      ? { kind: "yyplot", target: leftTarget }
      : { kind: "pareto", target_x: targetX, target_y: targetY })
      .then((value) => { if (active) setLeftPlot(value); })
      .catch((caught) => { if (active) setLeftError(caught instanceof Error ? caught.message : String(caught)); })
      .finally(() => { if (active) setLeftLoading(false); });
    return () => { active = false; };
  }, [leftKind, leftTarget, runId, targetX, targetY]);

  useEffect(() => {
    if (!runId) {
      setRightError("可視化セッションがありません。候補を再生成してください。");
      return;
    }
    const features = rightKind === "1d"
      ? [featureA]
      : rightKind === "2d"
        ? [featureA, featureB]
        : [featureA, featureB, featureC];
    if (features.some((value) => !value) || new Set(features).size !== features.length) {
      setRightError("図に使用する異なる説明変数を選択してください。");
      return;
    }
    let active = true;
    setRightLoading(true);
    setRightError(null);
    fetchResultVisualization(runId, {
      kind: rightKind as VisualizationKind,
      target: rightTarget,
      features,
      show_type: showType,
      sum_value: rightKind === "ternary" ? sumValue : undefined,
      n: rightKind === "2d" ? 30 : 50
    })
      .then((value) => { if (active) setRightPlot(value); })
      .catch((caught) => { if (active) setRightError(caught instanceof Error ? caught.message : String(caught)); })
      .finally(() => { if (active) setRightLoading(false); });
    return () => { active = false; };
  }, [featureA, featureB, featureC, rightKind, rightTarget, runId, showType, sumValue]);

  function changeRightKind(value: "1d" | "2d" | "ternary") {
    setRightKind(value);
    if (value !== "1d" && !options.numeric_features.includes(featureA)) {
      setFeatureA(options.numeric_features[0] ?? "");
    }
    if (value === "ternary" && initialGroup) {
      setFeatureA(initialGroup.features[0] ?? "");
      setFeatureB(initialGroup.features[1] ?? "");
      setFeatureC(initialGroup.features[2] ?? "");
      setSumValue(initialGroup.sum_value);
    }
  }

  const rightFeatures = rightKind === "1d" ? options.feature_columns : options.numeric_features;

  return (
    <section className="interactive-visualization-section">
      <div className="result-subheading">
        <div>
          <span className="eyebrow">Visualization</span>
          <h3>結果の可視化</h3>
          <p>左にモデル評価／目的変数、右に説明変数空間の既存Plotly図を表示します。</p>
        </div>
      </div>

      <div className="interactive-plot-grid">
        <article className="panel interactive-plot-card">
          <div className="plot-controls">
            <label>図<select value={leftKind} onChange={(event) => setLeftKind(event.target.value as "yyplot" | "pareto")}>
              <option value="yyplot">YY plot</option>
              <option value="pareto" disabled={options.regression_targets.length < 2}>Pareto図</option>
            </select></label>
            {leftKind === "yyplot" ? (
              <label>目的変数<select value={leftTarget} onChange={(event) => setLeftTarget(event.target.value)}>
                {options.target_columns.map((target) => <option key={target} value={target}>{target}</option>)}
              </select></label>
            ) : (
              <>
                <label>横軸目的<select value={targetX} onChange={(event) => setTargetX(event.target.value)}>
                  {options.regression_targets.map((target) => <option key={target} value={target}>{target}</option>)}
                </select></label>
                <label>縦軸目的<select value={targetY} onChange={(event) => setTargetY(event.target.value)}>
                  {options.regression_targets.map((target) => <option key={target} value={target}>{target}</option>)}
                </select></label>
              </>
            )}
          </div>
          <PlotCard visualization={leftPlot} loading={leftLoading} error={leftError} />
        </article>

        <article className="panel interactive-plot-card">
          <div className="plot-controls">
            <label>図<select value={rightKind} onChange={(event) => changeRightKind(event.target.value as "1d" | "2d" | "ternary")}>
              <option value="1d">1次元プロット</option>
              <option value="2d" disabled={options.numeric_features.length < 2}>2次元プロット</option>
              <option value="ternary" disabled={options.numeric_features.length < 3}>三角図</option>
            </select></label>
            <label>目的変数<select value={rightTarget} onChange={(event) => setRightTarget(event.target.value)}>
              {options.target_columns.map((target) => <option key={target} value={target}>{target}</option>)}
            </select></label>
            <label>表示<select value={showType} onChange={(event) => setShowType(event.target.value as "pred" | "acqf")}>
              <option value="pred">予測値</option>
              <option value="acqf">獲得関数</option>
            </select></label>
            <label>変数1<select value={featureA} onChange={(event) => setFeatureA(event.target.value)}>
              {rightFeatures.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
            </select></label>
            {rightKind !== "1d" && <label>変数2<select value={featureB} onChange={(event) => setFeatureB(event.target.value)}>
              {options.numeric_features.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
            </select></label>}
            {rightKind === "ternary" && <>
              <label>変数3<select value={featureC} onChange={(event) => setFeatureC(event.target.value)}>
                {options.numeric_features.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
              </select></label>
              <label>合計値<input type="number" step="any" value={sumValue} onChange={(event) => setSumValue(Number(event.target.value))} /></label>
            </>}
          </div>
          <PlotCard visualization={rightPlot} loading={rightLoading} error={rightError} />
        </article>
      </div>
    </section>
  );
}
