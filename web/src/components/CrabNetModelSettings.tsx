import { useWorkbench } from "../context/WorkbenchContext";
import { isCrabNetModelType } from "../modelOptions";

/** Configures the server-side checkpoint and CrabNet-DKL fine-tuning policy. */
export default function CrabNetModelSettings() {
  const {
    modelType,
    compositionSettings,
    crabnetCheckpoint,
    setCrabnetCheckpoint,
    crabnetEncoderTraining,
    setCrabnetEncoderTraining
  } = useWorkbench();

  if (!isCrabNetModelType(modelType)) return null;

  const compositionReady = Boolean(
    compositionSettings.enabled &&
    compositionSettings.column &&
    compositionSettings.elements.length >= 2
  );

  return (
    <article className="panel model-advanced-section crabnet-model-settings">
      <div className="config-column-heading">
        <span className="panel-kicker">CRABNET</span>
        <h4>{modelType === "crabnet_gp" ? "CrabNet-GP" : "CrabNet-DKL"} 設定</h4>
        <p>組成式と連続process条件を同じGaussian GPで学習します。</p>
      </div>
      <div className="model-settings-grid">
        <label>
          Checkpoint（任意）
          <input
            value={crabnetCheckpoint}
            placeholder="/srv/checkpoints/crabnet.pth"
            onChange={(event) => setCrabnetCheckpoint(event.target.value)}
          />
          <small>Webサーバーから読み取れるcheckpointパスを指定します。空欄ではランダム初期化です。</small>
        </label>
        {modelType === "crabnet_dkl" && (
          <label>
            Encoder training
            <select
              value={crabnetEncoderTraining}
              onChange={(event) => setCrabnetEncoderTraining(
                event.target.value as "partial" | "full"
              )}
            >
              <option value="partial">Partial（推奨）</option>
              <option value="full">Full</option>
            </select>
            <small>Partialは最終Transformer層のみ、FullはCrabNet encoder全体を微調整します。</small>
          </label>
        )}
      </div>
      {!compositionReady && (
        <p className="settings-note warning-text">
          Select画面で組成式列を選び、候補元素を2種類以上設定してください。
        </p>
      )}
      <p className="settings-note">
        対応範囲は単一の回帰目的、1つの組成式列、連続process列です。カテゴリprocess、入力摂動、複数目的には対応していません。
      </p>
    </article>
  );
}
