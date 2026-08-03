import type { Data } from "plotly.js";

export interface HeteroscedasticDiagnosticFigure {
  id: string;
  title: string;
  description: string;
  data: Data[];
  layout: Record<string, unknown>;
}

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

function safeId(value: string): string {
  return value.replace(/[^0-9A-Za-z_-]+/g, "-").replace(/^-+|-+$/g, "") || "feature";
}

/** Build plots from plot-ready input-dependent noise diagnostics. */
export function heteroscedasticFigures(
  value: unknown,
  output: string,
  featureColumns: string[]
): HeteroscedasticDiagnosticFigure[] {
  const record = asRecord(value);
  const profile = asRecord(record?.noise_profile);
  if (!profile) return [];

  const sampleIndex = flattenNumbers(profile.sample_index);
  const noiseStd = flattenNumbers(profile.noise_std);
  const featureValues = numericMatrix(profile.feature_values);
  const rawNames = Array.isArray(profile.feature_names)
    ? profile.feature_names.map(String)
    : [];
  const dimension = featureValues.length
    ? Math.max(...featureValues.map((row) => row.length))
    : 0;
  const featureNames = Array.from(
    { length: dimension },
    (_, index) => rawNames[index] ?? featureColumns[index] ?? `説明変数 ${index + 1}`
  );
  const count = Math.min(sampleIndex.length, noiseStd.length, featureValues.length);
  if (count === 0) return [];

  const totalCount = Number(profile.total_count ?? count);
  const displayedCount = Number(profile.displayed_count ?? count);
  const samplingNote = displayedCount < totalCount
    ? `全${totalCount}件から順序を維持して${displayedCount}件を等間隔抽出しています。`
    : `${displayedCount}件の学習データを表示しています。`;
  const figures: HeteroscedasticDiagnosticFigure[] = [{
    id: `${output}-heteroscedastic-by-row`,
    title: `${output}: 学習データごとの予測ノイズ`,
    description: `入力依存ノイズの予測標準偏差を学習データの行番号順に表示します。${samplingNote}`,
    data: [{
      type: "scatter",
      mode: "markers",
      x: sampleIndex.slice(0, count),
      y: noiseStd.slice(0, count),
      name: "予測ノイズ標準偏差",
      customdata: featureValues.slice(0, count),
      hovertemplate: [
        "行=%{x}",
        "ノイズ標準偏差=%{y:.6g}",
        ...featureNames.map((name, index) => `${name}=%{customdata[${index}]}`),
        "<extra></extra>"
      ].join("<br>")
    }],
    layout: {
      xaxis: { title: "元データの行番号" },
      yaxis: { title: "予測ノイズ標準偏差", rangemode: "tozero" },
      margin: { l: 90, r: 30, t: 70, b: 70 }
    }
  }];

  featureNames.forEach((name, featureIndex) => {
    const x: number[] = [];
    const y: number[] = [];
    const rows: number[] = [];
    for (let rowIndex = 0; rowIndex < count; rowIndex += 1) {
      const featureValue = featureValues[rowIndex]?.[featureIndex];
      const noiseValue = noiseStd[rowIndex];
      if (!Number.isFinite(featureValue) || !Number.isFinite(noiseValue)) continue;
      x.push(featureValue);
      y.push(noiseValue);
      rows.push(sampleIndex[rowIndex]);
    }
    if (!x.length) return;
    figures.push({
      id: `${output}-heteroscedastic-${featureIndex}-${safeId(name)}`,
      title: `${output}: ${name}と予測ノイズ`,
      description: `横軸には元データのカラム「${name}」を使用しています。値が大きい点ほど、その入力領域で観測ノイズが大きいとモデルが推定しています。`,
      data: [{
        type: "scatter",
        mode: "markers",
        x,
        y,
        customdata: rows,
        name: name,
        hovertemplate: `${name}=%{x}<br>ノイズ標準偏差=%{y:.6g}<br>行=%{customdata}<extra></extra>`
      }],
      layout: {
        xaxis: { title: name },
        yaxis: { title: "予測ノイズ標準偏差", rangemode: "tozero" },
        margin: { l: 90, r: 30, t: 70, b: 80 }
      }
    });
  });

  return figures;
}
