import { useWorkbench } from "../context/WorkbenchContext";
import {
  isCrabNetDKLModelType,
  isCrabNetMixedModelType,
  isCrabNetModelType
} from "../modelOptions";

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
  const mixedModel = isCrabNetMixedModelType(modelType);
  const dklModel = isCrabNetDKLModelType(modelType);
  const title = modelType === "crabnet_mixed_dkl"
    ? "CrabNet-Mixed DKL"
    : modelType === "crabnet_mixed_gp"
      ? "CrabNet-Mixed GP"
      : modelType === "crabnet_gp"
        ? "CrabNet-GP"
        : "CrabNet-DKL";

  return (
    <article className="panel model-advanced-section crabnet-model-settings">
      <div className="config-column-heading">
        <span className="panel-kicker">CRABNET</span>
        <h4>{title} 設定</h4>
        <p>
          {modelType === "crabnet_mixed_dkl"
            ? "CrabNet組成表現、連続process条件、学習可能なカテゴリEmbeddingをニューラル融合し、その潜在表現上でGaussian GPを学習します。"
            : modelType === "crabnet_mixed_gp"
              ? "組成式と連続process条件をCrabNet側の連続表現へ変換し、カテゴリprocess条件をCategorical kernelで統合します。"
              : "組成式と連続process条件を同じGaussian GPで学習します。"}
        </p>
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
        {dklModel && (
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
        {modelType === "crabnet_mixed_dkl"
          ? "対応範囲は単一の回帰目的、1つの組成式列、連続process列＋カテゴリprocess列です。カテゴリ候補はmixed optimizerで列挙し、カテゴリ値は学習可能なEmbeddingとして潜在表現へ統合します。CrabNet encoderはPartialまたはFullで微調整できます。入力摂動、複数目的には対応していません。"
          : modelType === "crabnet_mixed_gp"
            ? "対応範囲は単一の回帰目的、1つの組成式列、連続process列＋カテゴリprocess列です。カテゴリ候補はmixed optimizerで列挙し、CrabNet encoderは凍結します。入力摂動、複数目的には対応していません。"
            : mixedModel
              ? "対応範囲は単一の回帰目的、1つの組成式列、連続process列＋カテゴリprocess列です。"
              : "対応範囲は単一の回帰目的、1つの組成式列、連続process列です。カテゴリprocessを含む場合はCrabNet-Mixed GPまたはCrabNet-Mixed DKLを使用してください。入力摂動、複数目的には対応していません。"}
      </p>
    </article>
  );
}
