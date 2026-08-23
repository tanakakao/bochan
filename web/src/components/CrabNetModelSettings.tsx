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
  const multitaskModel = modelType === "crabnet_multitask";
  const title = modelType === "crabnet_multitask"
    ? "CrabNet-MultiTask"
    : modelType === "crabnet_mixed_dkl"
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
          {multitaskModel
            ? "共有したCrabNet組成表現と連続process条件の潜在空間上で、複数の回帰目的間の相関をMultiTask kernelとして学習します。"
            : modelType === "crabnet_mixed_dkl"
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
        {multitaskModel
          ? "対応範囲は2つ以上の連続回帰目的、1つの組成式列、連続process列です。目的間相関をtask covarianceで学習し、CrabNet encoderは全目的で共有・凍結します。カテゴリprocessと入力摂動には未対応です。"
          : modelType === "crabnet_mixed_dkl"
            ? "1つ以上の連続回帰目的、1つの組成式列、連続process列＋カテゴリprocess列に対応します。複数目的では目的ごとに独立したモデルを構築します。カテゴリ候補はmixed optimizerで列挙し、カテゴリ値は学習可能なEmbeddingとして潜在表現へ統合します。"
            : modelType === "crabnet_mixed_gp"
              ? "1つ以上の連続回帰目的、1つの組成式列、連続process列＋カテゴリprocess列に対応します。複数目的では目的ごとに独立したモデルを構築し、CrabNet encoderは凍結します。"
              : mixedModel
                ? "1つ以上の連続回帰目的、1つの組成式列、連続process列＋カテゴリprocess列に対応します。"
                : "1つ以上の連続回帰目的、1つの組成式列、連続process列に対応します。通常のCrabNet系で複数目的を選ぶと独立ModelListとなり、CrabNet-MultiTaskでは目的間相関を学習します。カテゴリprocessを含む場合はCrabNet-Mixed GPまたはCrabNet-Mixed DKLを使用してください。"}
      </p>
    </article>
  );
}
