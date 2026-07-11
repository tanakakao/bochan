import { EmptyState, MetricCard, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function DataPage() {
  const { dataset, columns, handleFile, setStep } = useWorkbench();
  const numericCount = columns.filter((column) => column.kind === "numeric").length;
  const categoricalCount = columns.filter((column) => column.kind === "categorical").length;

  return (
    <>
      <SectionHeader
        step="1 · DATA"
        title="最適化データを読み込む"
        text="CSVまたはExcelをFastAPIへ送信し、列型・欠損・基本統計を確認します。"
        action={
          dataset ? (
            <button onClick={() => setStep("prepare")}>変数設定へ</button>
          ) : undefined
        }
      />

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">DATA SOURCE</span>
            <h3>{dataset ? "データを入れ替える" : "データファイル"}</h3>
            <p>対応形式: CSV / XLSX / XLS</p>
          </div>
          {dataset && <span className="status-chip success">Loaded</span>}
        </div>
        <label className="dropzone">
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(event) => void handleFile(event.target.files?.[0] ?? null)}
          />
          <span className="upload-symbol">⇧</span>
          <strong>{dataset ? "別のファイルを選択" : "CSVまたはExcelを選択"}</strong>
          <span>ファイルはAPIで解析され、現在のFastAPIプロセス内に保持されます。</span>
        </label>
        {dataset && (
          <div className="file-summary">
            <strong>{dataset.name}</strong>
            <span>{dataset.profile.n_rows} rows × {dataset.profile.n_columns} columns</span>
          </div>
        )}
      </article>

      {dataset ? (
        <>
          <div className="cards metric-grid">
            <MetricCard icon="▦" label="Rows" value={dataset.profile.n_rows} detail="学習候補行" />
            <MetricCard icon="▥" label="Columns" value={dataset.profile.n_columns} detail="全列数" />
            <MetricCard icon="#" label="Numeric" value={numericCount} detail="目的・数値特徴量候補" />
            <MetricCard icon="A" label="Categorical" value={categoricalCount} detail="カテゴリ特徴量候補" />
          </div>

          <article className="panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">PREVIEW</span>
                <h3>データプレビュー</h3>
                <p>先頭{Math.min(dataset.preview.length, 20)}行を表示します。</p>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>{columns.map((column) => <th key={column.name}>{column.name}</th>)}</tr>
                </thead>
                <tbody>
                  {dataset.preview.slice(0, 20).map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {columns.map((column) => (
                        <td key={column.name}>{String(row[column.name] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">COLUMN PROFILE</span>
                <h3>列プロファイル</h3>
                <p>型、欠損率、ユニーク数、数値要約を確認します。</p>
              </div>
            </div>
            <div className="table-wrap compact">
              <table>
                <thead>
                  <tr>
                    <th>列</th><th>判定型</th><th>dtype</th><th>欠損</th><th>ユニーク</th>
                    <th>最小</th><th>最大</th><th>平均</th><th>標準偏差</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((column) => (
                    <tr key={column.name}>
                      <td><strong>{column.name}</strong></td>
                      <td><span className="status-chip">{column.kind}</span></td>
                      <td>{column.dtype}</td>
                      <td>{Math.round(column.missing_rate * 1000) / 10}%</td>
                      <td>{column.unique_count}</td>
                      <td>{column.min ?? "—"}</td>
                      <td>{column.max ?? "—"}</td>
                      <td>{column.mean ?? "—"}</td>
                      <td>{column.std ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </>
      ) : (
        <EmptyState>ファイルを読み込むと、データ概要とプレビューを表示します。</EmptyState>
      )}
    </>
  );
}
