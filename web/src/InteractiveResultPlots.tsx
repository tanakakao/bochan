import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { fetchResultVisualization } from "./api";
import { useWorkbench } from "./context/WorkbenchContext";
import { RESULT_PLOT_CONFIG } from "./plotConfig";
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

type LeftVisualizationKind = "yyplot" | "pareto";

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
        <div className="plot-container">
          <Plot
            data={visualization.figure.data as Data[]}
            layout={themedPlotLayout(visualization.figure.layout, theme)}
            config={RESULT_PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
          />
        </div>
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
  const [leftKind, setLeftKind] = useState<LeftVisualizationKind>(
    result.visualization_run_id && options.regression_targets.length >= 2 ? "pareto" : "yyplot"
  );
  const [leftTarget, setLeftTarget] = useState(options.target_columns[0] ?? "");
  const [targetX, setTargetX] = useState(options.regression_targets[0] ?? "");
  const [targetY, setTargetY] = useState(
    options.regression_targets[1] ?? options.regression_targets[0] ?? ""
  );
  const [showParetoFront, setShowParetoFront] = useState(false);
  const [rightKind, setRightKind] = useState<"1d" | "2d" | "ternary">("1d");
  const [rightTarget, setRightTarget] = useState(options.target_columns[0] ?? "");
  const [featureA, setFeatureA] = useState(options.feature_columns[0] ?? "");
  const [featureB, setFeatureB] = useState(
    options.numeric_features[1] ?? options.numeric_features[0] ?? ""
  );
  const [featureC, setFeatureC] = useState(
    options.numeric_features[2] ?? options.numeric_features[0] ?? ""
  );
  const [showType, setShowType] = useState<"pred" | "acqf">("pred");
  const initialGroup = options.ternary_groups?.[0];
  const [sumValue, setSumValue] = useState(initialGroup?.sum_value ?? 1);
  const [fixedValues, setFixedValues] = useState<Record<string, string | number>>(() =>
    Object.fromEntries(
      Object.entries(options.feature_controls ?? {}).map(([name, control]) => [name, control.default])
    )
  );
  const [leftPlot, setLeftPlot] = useState<ResultVisualization | null>(null);
  const [rightPlot, setRightPlot] = useState<ResultVisualization | null>(null);
  const [leftLoading, setLeftLoading] = useState(false);
  const [rightLoading, setRightLoading] = useState(false);
  const [leftError, setLeftError] = useState<string | null>(null);
  const [rightError, setRightError] = useState<string | null>(null);

  useEffect(() => {
    if (leftKind === "pareto" && options.regression_targets.length < 2) {
      setLeftKind("yyplot");
    }
  }, [leftKind, options.regression_targets]);

  useEffect(() => {
    if (!runId) {
      const saved = leftKind === "yyplot"
        ? result.visualizations.find((value) => value.id === `${leftTarget}-yyplot`)
          ?? result.visualizations.find((value) => value.id.endsWith("-yyplot"))
          ?? null
        : null;
      setLeftPlot(saved);
      setLeftError(saved
        ? null
        : "最新モデルの可視化セッションがありません。モデルを信頼してプロジェクトを読み込むか、候補を再生成してください。");
      return;
    }
    if (leftKind === "pareto" && (!targetX || !targetY || targetX === targetY)) {
      setLeftError("パレート図には異なる2つの回帰目的変数が必要です。");
      return;
    }
    let active = true;
    setLeftLoading(true);
    setLeftError(null);
    fetchResultVisualization(runId, leftKind === "yyplot"
      ? { kind: "yyplot", target: leftTarget }
      : {
          kind: "pareto",
          target_x: targetX,
          target_y: targetY,
          show_pareto_front: showParetoFront
        })
      .then((value) => { if (active) setLeftPlot(value); })
      .catch((caught) => {
        if (active) setLeftError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => { if (active) setLeftLoading(false); });
    return () => { active = false; };
  }, [leftKind, leftTarget, result.visualizations, runId, showParetoFront, targetX, targetY]);

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
      fixed_values: fixedValues,
      sum_value: rightKind === "ternary" ? sumValue : undefined,
      n: rightKind === "2d" ? 30 : 50
    })
      .then((value) => { if (active) setRightPlot(value); })
      .catch((caught) => {
        if (active) setRightError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => { if (active) setRightLoading(false); });
    return () => { active = false; };
  }, [featureA, featureB, featureC, fixedValues, rightKind, rightTarget, runId, showType, sumValue]);

  function changeLeftKind(value: LeftVisualizationKind) {
    setLeftKind(value);
    if (value === "pareto") {
      setTargetX(options.regression_targets[0] ?? "");
      setTargetY(options.regression_targets[1] ?? options.regression_targets[0] ?? "");
    }
  }

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
  const plottedFeatures = new Set(rightKind === "1d" ? [featureA] : rightKind === "2d"
    ? [featureA, featureB] : [featureA, featureB, featureC]);

  return (
    <section className="interactive-visualization-section">
      <div className="result-subheading">
        <div>
          <span className="eyebrow">Visualization</span>
          <h3>結果の可視化</h3>
          <p>左にモデル評価／パレート図、右に説明変数空間の既存Plotly図を表示します。</p>
        </div>
      </div>

      <div className="interactive-plot-grid">
        <article className="panel interactive-plot-card">
          <div className="plot-controls">
            <label>図<select
              value={leftKind}
              onChange={(event) => changeLeftKind(event.target.value as LeftVisualizationKind)}
            >
              <option value="yyplot">YY plot</option>
              <option value="pareto" disabled={options.regression_targets.length < 2}>パレート図</option>
            </select></label>
            {leftKind === "yyplot" ? (
              <label>目的変数<select
                value={leftTarget}
                onChange={(event) => setLeftTarget(event.target.value)}
              >
                {options.target_columns.map((target) => (
                  <option key={target} value={target}>{target}</option>
                ))}
              </select></label>
            ) : (
              <>
                <label>横軸目的<select value={targetX} onChange={(event) => setTargetX(event.target.value)}>
                  {options.regression_targets.map((target) => (
                    <option key={target} value={target}>{target}</option>
                  ))}
                </select></label>
                <label>縦軸目的<select value={targetY} onChange={(event) => setTargetY(event.target.value)}>
                  {options.regression_targets.map((target) => (
                    <option key={target} value={target}>{target}</option>
                  ))}
                </select></label>
                <label className="switch-field">
                  <input
                    type="checkbox"
                    checked={showParetoFront}
                    onChange={(event) => setShowParetoFront(event.target.checked)}
                  />
                  現データのパレートフロント
                </label>
              </>
            )}
          </div>
          <PlotCard visualization={leftPlot} loading={leftLoading} error={leftError} />
        </article>

        <article className="panel interactive-plot-card">
          <div className="plot-controls">
            <label>図<select
              value={rightKind}
              onChange={(event) => changeRightKind(event.target.value as "1d" | "2d" | "ternary")}
            >
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
              <label>合計値<input
                type="number"
                step="any"
                value={sumValue}
                onChange={(event) => setSumValue(Number(event.target.value))}
              /></label>
            </>}
          </div>
          <PlotCard visualization={rightPlot} loading={rightLoading} error={rightError} />
          {options.feature_controls && (
            <div className="plot-slice-controls">
              <div>
                <strong>表示外の変数値</strong>
                <p>図に使わない説明変数を固定します。表示中の変数は編集できません。</p>
              </div>
              <div className="plot-slice-control-grid">
                {options.feature_columns.map((feature) => {
                  const control = options.feature_controls?.[feature];
                  if (!control) return null;
                  const disabled = plottedFeatures.has(feature);
                  return <label key={feature} className={disabled ? "plot-slice-disabled" : ""}>
                    <span>{feature}{disabled && "（図で使用中）"}</span>
                    {control.kind === "numeric" ? <>
                      <input
                        type="range"
                        min={control.min}
                        max={control.max}
                        step="any"
                        value={Number(fixedValues[feature] ?? control.default)}
                        disabled={disabled}
                        onChange={(event) => setFixedValues((current) => ({
                          ...current,
                          [feature]: Number(event.target.value)
                        }))}
                      />
                      <output>{Number(fixedValues[feature] ?? control.default).toPrecision(5)}</output>
                    </> : <select
                      value={String(fixedValues[feature] ?? control.default)}
                      disabled={disabled}
                      onChange={(event) => setFixedValues((current) => ({
                        ...current,
                        [feature]: event.target.value
                      }))}
                    >
                      {(control.values ?? []).map((value) => (
                        <option key={String(value)} value={String(value)}>{value}</option>
                      ))}
                    </select>}
                  </label>;
                })}
              </div>
            </div>
          )}
        </article>
      </div>
    </section>
  );
}
