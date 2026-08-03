import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { useWorkbench } from "./context/WorkbenchContext";
import { heteroscedasticFigures } from "./heteroscedasticDiagnosticFigures";
import {
  filterHeteroscedasticFigures,
  HETEROSCEDASTIC_SCOPE_ALL,
  heteroscedasticScopeOptions
} from "./heteroscedasticDiagnosticSelector";
import { RESULT_PLOT_CONFIG } from "./plotConfig";
import { themedPlotLayout } from "./plotLayout";
import type {
  FeatureImportanceSummaryRecord,
  RegressionResult,
  ResultVisualization
} from "./types";

type InspectionView = "permutation" | "model_diagnostic";

interface DiagnosticFigure {
  id: string;
  title: string;
  description: string;
  data: Data[];
  layout: Record<string, unknown>;
}

const DIAGNOSTIC_LABELS: Record<string, string> = {
  ard: "ARD感度",
  kernel_components: "カーネル構成",
  saas: "SAAS診断",
  pca: "PCA診断",
  rembo: "REMBO診断",
  vae: "VAE診断",
  deepkernel: "Deep Kernel診断",
  deepgp: "Deep GP診断",
  heteroscedastic: "入力依存ノイズ診断",
  observation_relevance: "観測関連度（RRP）",
  multitask: "マルチタスク診断",
  multifidelity: "マルチフィデリティ診断"
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number" && Number.isFinite(value)) return [value];
  if (!Array.isArray(value)) return [];
  return value.flatMap(flattenNumbers);
}

function numericMatrix(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  return value
    .map((row) => flattenNumbers(row))
    .filter((row) => row.length > 0);
}

function featureName(value: unknown, featureColumns: string[]): string {
  const raw = String(value);
  if (featureColumns.includes(raw)) return raw;
  const match = raw.match(/^(?:feature|x)(?:[_\s-]|\[)?(\d+)\]?$/i);
  if (match) {
    const index = Number(match[1]);
    if (Number.isInteger(index) && featureColumns[index]) return featureColumns[index];
  }
  return raw;
}

function featureNameAt(index: number, featureColumns: string[]): string {
  return featureColumns[index] ?? `説明変数 ${index + 1}`;
}

function diagnosticLabel(key: string): string {
  return DIAGNOSTIC_LABELS[key] ?? key;
}

function diagnosticsByOutput(result: RegressionResult): Record<string, Record<string, unknown>> {
  const diagnostics = result.model_diagnostics ?? {};
  const keys = Object.keys(diagnostics);
  if (!keys.length) return {};

  const knownDiagnosticKeys = new Set(Object.keys(DIAGNOSTIC_LABELS));
  const isFlat = keys.some((key) => knownDiagnosticKeys.has(key));
  if (isFlat) {
    const fallback = result.target_column || result.target_columns[0] || "output";
    return { [fallback]: diagnostics };
  }

  const nested: Array<[string, Record<string, unknown>]> = [];
  for (const [name, value] of Object.entries(diagnostics)) {
    const record = asRecord(value);
    if (record) nested.push([name, record]);
  }
  return Object.fromEntries(nested);
}

function safeFigureId(value: string): string {
  return value.replace(/[^0-9A-Za-z_-]+/g, "-").replace(/^-+|-+$/g, "") || "target";
}

function relabelValues(value: unknown, featureColumns: string[]): unknown {
  if (!Array.isArray(value)) return value;
  return value.map((item) => (
    typeof item === "string" ? featureName(item, featureColumns) : item
  ));
}

function relabelVisualization(
  visualization: ResultVisualization,
  featureColumns: string[]
): ResultVisualization {
  const data = visualization.figure.data.map((item) => {
    const trace = asRecord(item);
    if (!trace) return item;
    return {
      ...trace,
      x: relabelValues(trace.x, featureColumns),
      y: relabelValues(trace.y, featureColumns)
    };
  });
  const layout = { ...visualization.figure.layout };
  for (const axisName of ["xaxis", "yaxis"]) {
    const axis = asRecord(layout[axisName]);
    if (axis?.ticktext) {
      layout[axisName] = {
        ...axis,
        ticktext: relabelValues(axis.ticktext, featureColumns)
      };
    }
  }
  return {
    ...visualization,
    figure: { ...visualization.figure, data, layout }
  };
}

function ardFigures(
  diagnosticKey: string,
  value: unknown,
  output: string,
  featureColumns: string[]
): DiagnosticFigure[] {
  const record = asRecord(value);
  const rawComponents = diagnosticKey === "ard"
    ? record?.components
    : value;
  const components = Array.isArray(rawComponents)
    ? rawComponents.map(asRecord).filter((item): item is Record<string, unknown> => item !== null)
    : [];
  const traces: Data[] = [];

  components.forEach((component, componentIndex) => {
    const inverse = flattenNumbers(component.inverse_lengthscale);
    const lengths = flattenNumbers(component.lengthscale);
    const values = inverse.length
      ? inverse
      : lengths.map((item) => item === 0 ? Number.NaN : 1 / item);
    if (!values.length) return;
    const activeDims = flattenNumbers(component.active_dims).map((item) => Math.round(item));
    const labels = values.map((_, index) => (
      featureNameAt(activeDims[index] ?? index, featureColumns)
    ));
    traces.push({
      type: "bar",
      orientation: "h",
      x: values,
      y: labels,
      name: String(
        component.path ??
        component.kernel_class ??
        `カーネル ${componentIndex + 1}`
      )
    });
  });

  if (!traces.length) return [];
  return [{
    id: `${output}-${diagnosticKey}-sensitivity`,
    title: `${output}: ${diagnosticLabel(diagnosticKey)}`,
    description: "逆長さ尺度が大きい説明変数ほど、カーネル予測がその変数の変化に敏感です。Permutation Importanceや因果効果とは異なります。",
    data: traces,
    layout: {
      barmode: "group",
      xaxis: { title: "逆長さ尺度" },
      yaxis: { title: "説明変数", autorange: "reversed" },
      margin: { l: 170, r: 30, t: 70, b: 60 }
    }
  }];
}

function pcaFigures(
  value: unknown,
  output: string,
  featureColumns: string[]
): DiagnosticFigure[] {
  const record = asRecord(value);
  if (!record) return [];
  const figures: DiagnosticFigure[] = [];
  const ratios = flattenNumbers(
    record.explained_variance_ratio_ ?? record.explained_variance_ratio
  );
  if (ratios.length) {
    const componentNames = ratios.map((_, index) => `PC${index + 1}`);
    let cumulative = 0;
    const cumulativeRatios = ratios.map((ratio) => {
      cumulative += ratio;
      return cumulative;
    });
    figures.push({
      id: `${output}-pca-explained-variance`,
      title: `${output}: PCA寄与率`,
      description: "各主成分の寄与率と累積寄与率を表示します。",
      data: [
        {
          type: "bar",
          x: componentNames,
          y: ratios,
          name: "寄与率"
        },
        {
          type: "scatter",
          mode: "lines+markers",
          x: componentNames,
          y: cumulativeRatios,
          name: "累積寄与率"
        }
      ],
      layout: {
        xaxis: { title: "主成分" },
        yaxis: { title: "寄与率" },
        margin: { l: 80, r: 30, t: 70, b: 60 }
      }
    });
  }

  const loadings = numericMatrix(record.components_ ?? record.components);
  if (loadings.length) {
    const width = Math.max(...loadings.map((row) => row.length));
    const labels = Array.from(
      { length: width },
      (_, index) => featureNameAt(index, featureColumns)
    );
    figures.push({
      id: `${output}-pca-loadings`,
      title: `${output}: PCA負荷量`,
      description: "横軸には元データの説明変数名を使用しています。",
      data: [{
        type: "heatmap",
        z: loadings,
        x: labels,
        y: loadings.map((_, index) => `PC${index + 1}`)
      }],
      layout: {
        xaxis: { title: "説明変数" },
        yaxis: { title: "主成分" },
        margin: { l: 80, r: 30, t: 70, b: 120 }
      }
    });
  }
  return figures;
}

function multitaskFigures(
  value: unknown,
  output: string,
  targetColumns: string[]
): DiagnosticFigure[] {
  const record = asRecord(value);
  if (!record) return [];
  const matrix = numericMatrix(record.task_correlation ?? record.correlation);
  if (!matrix.length) return [];
  const labelsValue = Array.isArray(record.task_names)
    ? record.task_names.map(String)
    : targetColumns;
  const labels = labelsValue.length === matrix.length
    ? labelsValue
    : matrix.map((_, index) => targetColumns[index] ?? `目的変数 ${index + 1}`);
  return [{
    id: `${output}-multitask-correlation`,
    title: `${output}: 学習タスク相関`,
    description: "軸ラベルには元データの目的変数名を使用しています。",
    data: [{
      type: "heatmap",
      z: matrix,
      x: labels,
      y: labels,
      zmin: -1,
      zmax: 1
    }],
    layout: {
      xaxis: { title: "目的変数" },
      yaxis: { title: "目的変数" },
      margin: { l: 110, r: 30, t: 70, b: 100 }
    }
  }];
}

function diagnosticFigures(
  key: string,
  value: unknown,
  output: string,
  result: RegressionResult
): DiagnosticFigure[] {
  if (key === "ard" || key === "kernel_components") {
    return ardFigures(key, value, output, result.feature_columns);
  }
  if (key === "pca") {
    return pcaFigures(value, output, result.feature_columns);
  }
  if (key === "heteroscedastic") {
    return heteroscedasticFigures(value, output, result.feature_columns);
  }
  if (key === "multitask") {
    return multitaskFigures(value, output, result.target_columns);
  }
  return [];
}

function figureMatches(
  visualization: ResultVisualization,
  output: string,
  kind: string,
  outputCount: number
): boolean {
  const id = visualization.id.toLowerCase();
  const safeOutput = safeFigureId(output).toLowerCase();
  const safeIdMatch = safeOutput !== "target" && id.includes(safeOutput);
  const outputMatch = outputCount === 1 ||
    id.includes(output.toLowerCase()) ||
    safeIdMatch ||
    visualization.title.includes(output);
  return outputMatch && id.includes(kind);
}

function importanceKindLabel(kind: string): string {
  if (kind === "predictive") return "予測重要度";
  if (kind === "noise") return "ノイズ重要度";
  return "クラス別重要度";
}

/** Display permutation importance and available model-specific diagnostics. */
export default function FeatureImportancePanel({ result }: { result: RegressionResult }) {
  const { theme } = useWorkbench();
  const summary = result.feature_importance_summary ?? [];
  const figures = result.feature_importance_visualizations ?? [];
  const warnings = [...new Set(result.feature_importance_warnings ?? [])];
  const diagnostics = useMemo(() => diagnosticsByOutput(result), [result]);
  const permutationOutputs = useMemo(
    () => [...new Set(summary.map((row) => row.output_name))],
    [summary]
  );
  const diagnosticOutputs = useMemo(() => Object.keys(diagnostics), [diagnostics]);
  const permutationAvailable = summary.length > 0 || figures.length > 0;
  const diagnosticAvailable = diagnosticOutputs.length > 0;
  const [view, setView] = useState<InspectionView>(
    permutationAvailable ? "permutation" : "model_diagnostic"
  );
  const viewOutputs = useMemo(() => {
    const selected = view === "model_diagnostic"
      ? diagnosticOutputs
      : permutationOutputs;
    if (selected.length) return selected;
    return result.target_columns.length
      ? result.target_columns
      : [result.target_column].filter(Boolean);
  }, [diagnosticOutputs, permutationOutputs, result.target_column, result.target_columns, view]);
  const [output, setOutput] = useState(
    viewOutputs[0] ?? result.target_column ?? ""
  );
  const kinds = useMemo(
    () => [...new Set(
      summary
        .filter((row) => row.output_name === output)
        .map((row) => row.importance_kind)
    )],
    [summary, output]
  );
  const [kind, setKind] = useState<FeatureImportanceSummaryRecord["importance_kind"]>(
    kinds[0] ?? "predictive"
  );
  const outputDiagnostics = diagnostics[output] ?? {};
  const diagnosticKeys = useMemo(
    () => Object.keys(outputDiagnostics),
    [outputDiagnostics]
  );
  const [diagnosticKey, setDiagnosticKey] = useState(diagnosticKeys[0] ?? "");
  const selectedDiagnostic = outputDiagnostics[diagnosticKey];
  const heteroscedasticScopes = useMemo(
    () => heteroscedasticScopeOptions(selectedDiagnostic, result.feature_columns),
    [selectedDiagnostic, result.feature_columns]
  );
  const [heteroscedasticScope, setHeteroscedasticScope] = useState(
    HETEROSCEDASTIC_SCOPE_ALL
  );

  useEffect(() => {
    if (!viewOutputs.includes(output)) {
      setOutput(viewOutputs[0] ?? result.target_column ?? "");
    }
  }, [output, result.target_column, viewOutputs]);

  useEffect(() => {
    if (!kinds.includes(kind)) {
      setKind(kinds[0] ?? "predictive");
    }
  }, [kind, kinds]);

  useEffect(() => {
    if (!diagnosticKeys.includes(diagnosticKey)) {
      setDiagnosticKey(diagnosticKeys[0] ?? "");
    }
  }, [diagnosticKey, diagnosticKeys]);

  useEffect(() => {
    setHeteroscedasticScope(HETEROSCEDASTIC_SCOPE_ALL);
  }, [diagnosticKey, output]);

  const rows = useMemo(() => summary
    .filter((row) => row.output_name === output && row.importance_kind === kind)
    .map((row) => ({
      ...row,
      feature: featureName(row.feature, result.feature_columns)
    }))
    .sort(
      (left, right) => (right.mean ?? -Infinity) - (left.mean ?? -Infinity)
    ), [summary, output, kind, result.feature_columns]);

  const visibleFigures = useMemo(
    () => figures
      .filter((figure) => figureMatches(
        figure,
        output,
        kind,
        permutationOutputs.length
      ))
      .map((figure) => relabelVisualization(figure, result.feature_columns)),
    [figures, output, kind, permutationOutputs.length, result.feature_columns]
  );

  const diagnosticFigureCandidates = useMemo(
    () => diagnosticFigures(diagnosticKey, selectedDiagnostic, output, result),
    [diagnosticKey, selectedDiagnostic, output, result]
  );
  const selectedDiagnosticFigures = useMemo(
    () => diagnosticKey === "heteroscedastic"
      ? filterHeteroscedasticFigures(
          diagnosticFigureCandidates,
          heteroscedasticScope
        )
      : diagnosticFigureCandidates,
    [diagnosticFigureCandidates, diagnosticKey, heteroscedasticScope]
  );

  if (!summary.length && !figures.length && !warnings.length && !diagnosticOutputs.length) {
    return null;
  }

  return <section className="visualization-section feature-importance-panel">
    <div className="result-subheading">
      <div>
        <span className="eyebrow">MODEL INSPECTION</span>
        <h3>特徴量重要度・モデル診断</h3>
        <p>
          Permutation Importanceと、学習モデルが提供するARD・PCAなどのモデル固有診断を切り替えて確認できます。
          いずれも因果効果ではありません。
        </p>
      </div>
    </div>

    {result.feature_importance_source === "training" &&
      <div className="alert warning">
        学習データ上で評価しているため、Permutation Importanceが楽観的な可能性があります。
      </div>}
    {result.feature_importance_source === "cross_validation" &&
      <p>各foldのValidationデータで計算した重要度を集約しています。エラーバーはfold間標準偏差です。</p>}
    {warnings.length > 0 &&
      <div className="alert warning">
        {warnings.map((warning) => <div key={warning}>{warning}</div>)}
      </div>}

    <div className="model-settings-grid">
      <label>
        表示内容
        <select
          value={view}
          onChange={(event) => setView(event.target.value as InspectionView)}
        >
          <option value="permutation">
            Permutation Importance（PI）{permutationAvailable ? "" : "（未計算）"}
          </option>
          <option value="model_diagnostic">
            モデル固有診断{diagnosticAvailable ? "" : "（未取得）"}
          </option>
        </select>
      </label>
      {viewOutputs.length > 1 &&
        <label>
          出力
          <select value={output} onChange={(event) => setOutput(event.target.value)}>
            {viewOutputs.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>}
      {view === "permutation" &&
        <label>
          重要度種別
          <select
            value={kind}
            disabled={kinds.length === 0}
            onChange={(event) => setKind(
              event.target.value as FeatureImportanceSummaryRecord["importance_kind"]
            )}
          >
            {kinds.length === 0
              ? <option value="predictive">重要度データなし</option>
              : kinds.map((name) =>
                <option key={name} value={name}>{importanceKindLabel(name)}</option>)}
          </select>
        </label>}
      {view === "model_diagnostic" &&
        <label>
          診断種類
          <select
            value={diagnosticKey}
            disabled={diagnosticKeys.length === 0}
            onChange={(event) => setDiagnosticKey(event.target.value)}
          >
            {diagnosticKeys.length === 0
              ? <option value="">診断データなし</option>
              : diagnosticKeys.map((name) =>
                <option key={name} value={name}>{diagnosticLabel(name)}</option>)}
          </select>
        </label>}
      {view === "model_diagnostic" && diagnosticKey === "heteroscedastic" &&
        <label>
          表示対象
          <select
            value={heteroscedasticScope}
            onChange={(event) => setHeteroscedasticScope(event.target.value)}
          >
            {heteroscedasticScopes.map((scope) =>
              <option key={scope.value} value={scope.value}>{scope.label}</option>)}
          </select>
        </label>}
    </div>

    {view === "permutation" && !permutationAvailable &&
      <div className="alert warning">
        この結果にはPermutation Importanceがありません。モデル設定の「特徴量重要度を計算する」を有効にして再実行してください。
      </div>}

    {view === "model_diagnostic" && !diagnosticAvailable &&
      <div className="alert warning">
        この結果にはモデル固有診断がありません。モデル設定で「特徴量重要度を計算する」と
        「モデル固有診断を自動取得」を有効にして再実行してください。既存の結果には診断が後から追加されません。
      </div>}

    {view === "permutation" && permutationAvailable && <>
      <div className="visualization-grid">
        {visibleFigures.map((visualization) =>
          <article className="panel visualization-card" key={`${visualization.id}-${visualization.title}`}>
            <h3>{visualization.title}</h3>
            <p>{visualization.description}</p>
            <div
              className="plot-container"
              style={{ height: Math.min(900, Math.max(360, rows.length * 34 + 140)) }}
            >
              <Plot
                data={visualization.figure.data as Data[]}
                layout={themedPlotLayout({
                  ...visualization.figure.layout,
                  margin: { l: 170, r: 30, t: 70, b: 60 }
                }, theme)}
                config={RESULT_PLOT_CONFIG}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
              />
            </div>
          </article>)}
      </div>
      {rows.length > 0 &&
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>順位</th><th>特徴量</th><th>平均重要度</th><th>標準偏差</th>
                <th>正規化重要度</th><th>役割</th><th>種類</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) =>
                <tr key={`${row.method}-${row.feature}-${index}`}>
                  <td>{row.rank ?? "—"}</td>
                  <td>{row.feature}</td>
                  <td>{row.mean ?? "—"}</td>
                  <td>{result.feature_importance_source === "cross_validation"
                    ? row.between_fold_std ?? "—"
                    : row.std ?? "—"}</td>
                  <td>{row.normalized_mean ?? "—"}</td>
                  <td>{row.role ?? "—"}</td>
                  <td>{row.feature_type ?? "—"}</td>
                </tr>)}
            </tbody>
          </table>
        </div>}
    </>}

    {view === "model_diagnostic" && diagnosticAvailable && <>
      <p className="settings-note">
        {diagnosticKey === "ard"
          ? "ARDはカーネルの感度を表します。値が大きいほど、その説明変数に対して予測が敏感です。"
          : diagnosticKey === "heteroscedastic"
            ? "入力依存ノイズ診断は特徴量重要度ではありません。選択した入力条件と、モデルが推定した観測ノイズ標準偏差の関係を表示します。"
            : `${diagnosticLabel(diagnosticKey)}を表示しています。`}
      </p>
      {selectedDiagnosticFigures.length > 0
        ? <div className="visualization-grid">
            {selectedDiagnosticFigures.map((figure) =>
              <article className="panel visualization-card" key={figure.id}>
                <h3>{figure.title}</h3>
                <p>{figure.description}</p>
                <div className="plot-container" style={{ height: 460 }}>
                  <Plot
                    data={figure.data}
                    layout={themedPlotLayout(figure.layout, theme)}
                    config={RESULT_PLOT_CONFIG}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
              </article>)}
          </div>
        : <div className="alert warning">
            この診断には共通形式のグラフがないため、下の診断データを表示します。
          </div>}
      {selectedDiagnostic !== undefined &&
        <details>
          <summary>{diagnosticLabel(diagnosticKey)}の診断データ</summary>
          <p>説明変数は元データのカラム順です: {result.feature_columns.join(" / ")}</p>
          <pre>{JSON.stringify(selectedDiagnostic, null, 2)}</pre>
        </details>}
    </>}
  </section>;
}
