import { EmptyState, MetricCard, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  createTutorialSampleFile,
  TUTORIAL_SAMPLE_DATASET_NAME
} from "../tutorial/sampleDataset";

export default function DataPage() {
  const { busy, dataset, columns, handleFile, handleModelArtifact, setError, setStep } = useWorkbench();
  const numericCount = columns.filter((column) => column.kind === "numeric").length;
  const categoricalCount = columns.filter((column) => column.kind === "categorical").length;
  const tutorialSampleLoaded = dataset?.name === TUTORIAL_SAMPLE_DATASET_NAME;

  async function selectModelArtifact(file: File | null, input: HTMLInputElement) {
    try {
      if (!file) return;
      const trusted = window.confirm(
        "保存モデルの読込にはtorch.load / pickleを使用します。\n\n" +
        "pickleファイルは読込時にコードを実行できるため、このbochan Webアプリから自分で保存した信頼できるファイルだけを選択してください。\n\n" +
        "このモデルファイルを信頼して読み込みますか？"
      );
      if (trusted) await handleModelArtifact(file);
    } finally {
      input.value = "";
    }
  }

  async function selectProjectArchive(file: File | null, input: HTMLInputElement) {
    try {
      if (!file) return;
      const trusted = window.confirm(
        "プロジェクトZIPには学習済みモデルが含まれる場合があります。\n\n" +
        "モデルの復元にはtorch.load / pickleを使用するため、このbochan Webアプリから自分で保存した信頼できるプロジェクトだけを選択してください。\n\n" +
        "このプロジェクトと含まれるモデルを信頼して読み込みますか？"
      );
      if (!trusted) return;
      setError(null);
      await handleModelArtifact(file);
    } finally {
      input.value = "";
    }
  }

  function loadTutorialSample() {
    if (dataset && !window.confirm(
      "サンプルデータを読み込むと、現在のワークスペースのデータと候補結果が置き換わります。\n\nサンプルデータを読み込みますか？"
    )) {
      return;
    }
    void handleFile(createTutorialSampleFile());
  }

  return (
    <>
      <SectionHeader
        step="1 · DATA"
        title="最適化データ、保存モデル、またはプロジェクトを読み込む"
        text="CSV・Excelから新規学習するか、保存モデルまたは履歴付きプロジェクトを読み込んで作業を再開します。"
        action={
          dataset ? (
            <button onClick={() => setStep("prepare")}>変数設定へ</button>
          ) : undefined
        }
      />

      <div className="data-source-grid">
        <article className="panel tutorial-sample-panel" data-tutorial="sample-data">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">SAMPLE TUTORIAL</span>
              <h3>サンプルデータで試す</h3>
              <p>材料の製造条件から強度を最大化する流れを体験します。</p>
            </div>
            {tutorialSampleLoaded && <span className="status-chip success">Loaded</span>}
          </div>
          <div className="tutorial-sample-specs">
            <span><strong>30 rows</strong> 実験データ</span>
            <span><strong>3 features</strong> 温度・保持時間・添加量</span>
            <span><strong>1 target</strong> 強度を最大化</span>
          </div>
          <button
            type="button"
            className="tutorial-sample-load"
            disabled={Boolean(busy)}
            onClick={loadTutorialSample}
          >
            {tutorialSampleLoaded ? "サンプルを読み直す" : "サンプルデータを読み込む"}
          </button>
          <p className="tutorial-sample-note">
            生成したCSVを通常のファイル読込APIへ渡すため、実データと同じ処理経路を使用します。
          </p>
        </article>

        <article className="panel data-file-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">DATA SOURCE</span>
              <h3>{dataset ? "データを入れ替える" : "データファイル"}</h3>
              <p>対応形式: CSV / XLSX / XLS</p>
            </div>
            {dataset && dataset.source_type !== "model_artifact" && !tutorialSampleLoaded && (
              <span className="status-chip success">Loaded</span>
            )}
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
        </article>

        <article className="panel model-artifact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">SAVED MODEL</span>
              <h3>保存モデルを読み込む</h3>
              <p>モデル、学習データ、設定、候補結果を復元します。</p>
            </div>
            {dataset?.source_type === "model_artifact" && (
              <span className="status-chip success">Restored</span>
            )}
          </div>
          <label className="dropzone model-dropzone">
            <input
              type="file"
              accept=".pt,.bochan.pt,application/octet-stream"
              onChange={(event) => void selectModelArtifact(
                event.target.files?.[0] ?? null,
                event.currentTarget
              )}
            />
            <span className="upload-symbol">↺</span>
            <strong>bochan保存モデルを選択</strong>
            <span>信頼できる`.bochan.pt`ファイルだけを読み込んでください。</span>
          </label>
          <div className="alert warning artifact-security-note">
            保存モデルはpickle形式です。メールや外部サイトから入手した不明なファイルは読み込まないでください。
          </div>
        </article>

        <article className="panel model-artifact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">EXPERIMENT PROJECT</span>
              <h3>履歴付きプロジェクトを開く</h3>
              <p>データセット系譜、実験サイクル、設定、保存された学習済みモデルを復元します。</p>
            </div>
          </div>
          <label className="dropzone model-dropzone">
            <input
              type="file"
              accept=".bochan-project.zip,.zip,application/zip"
              onChange={(event) => void selectProjectArchive(
                event.target.files?.[0] ?? null,
                event.currentTarget
              )}
            />
            <span className="upload-symbol">▣</span>
            <strong>bochanプロジェクトZIPを選択</strong>
            <span>ファイル名が変更されたZIPも、内部のプロジェクト情報を検証して読み込みます。</span>
          </label>
          <div className="alert warning artifact-security-note">
            通常のプロジェクトZIPには最新モデルが含まれます。モデルを含む場合はpickle形式のため、信頼できるプロジェクトだけを読み込んでください。
          </div>
        </article>
      </div>

      {dataset && (
        <div className="file-summary">
          <strong>{dataset.name}</strong>
          <span>{dataset.profile.n_rows} rows × {dataset.profile.n_columns} columns</span>
          <span className="status-chip">{dataset.source_type}</span>
        </div>
      )}

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
        <EmptyState>データ、保存モデル、または履歴付きプロジェクトを読み込むと概要を表示します。</EmptyState>
      )}
    </>
  );
}
