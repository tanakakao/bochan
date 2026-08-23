import { useWorkbench } from "../context/WorkbenchContext";
import {
  isCrabNetDKLModelType,
  isCrabNetMixedModelType,
  isCrabNetModelType,
  isCrabNetMultitaskModelType
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
  const multitaskModel = isCrabNetMultitaskModelType(modelType);
  const title = modelType === "crabnet_mixed_multitask_dkl"
    ? "CrabNet-Mixed MultiTask DKL"
    : modelType === "crabnet_mixed_multitask"
      ? "CrabNet-Mixed MultiTask GP"
      : modelType === "crabnet_multitask_dkl"
        ? "CrabNet-MultiTask DKL"
        : modelType === "crabnet_multitask"
          ? "CrabNet-MultiTask GP"
          : modelType === "crabnet_mixed_dkl"
            ? "CrabNet-Mixed DKL"
            : modelType === "crabnet_mixed_gp"
              ? "CrabNet-Mixed GP"
              : modelType === "crabnet_gp"
                ? "CrabNet-GP"
                : "CrabNet-DKL";

  const architectureDescription = multitaskModel
    ? mixedModel
      ? dklModel
        ? "共有CrabNet encoder、連続process条件、カテゴリEmbeddingを共同学習し、共有潜在表現上で複数目的間のtask covarianceを学習します。"
        : "共有した凍結CrabNet表現と連続process条件を連続kernelへ、カテゴリprocess条件をCategorical kernelへ接続し、複数目的間のtask covarianceを学習します。"
      : dklModel
        ? "共有CrabNet encoderを微調整しながら連続process条件との共有潜在表現を学習し、複数目的間のtask covarianceも同時学習します。"
        : "共有した凍結CrabNet組成表現と連続process条件の潜在空間上で、複数の回帰目的間の相関をMultiTask kernelとして学習します。"
    : modelType === "crabnet_mixed_dkl"
      ? "CrabNet組成表現、連続process条件、学習可能なカテゴリEmbeddingをニューラル融合し、その潜在表現上でGaussian GPを学習します。"
      : modelType === "crabnet_mixed_gp"
        ? "組成式と連続process条件をCrabNet側の連続表現へ変換し、カテゴリprocess条件をCategorical kernelで統合します。"
        : "組成式と連続process条件を同じGaussian GPで学習します。";

  return (
    <article className="panel model-advanced-section crabnet-model-settings">
      <div className="config-column-heading">
        <span className="panel-kicker">CRABNET</span>
        <h4>{title} 設定</h4>
        <p>{architectureDescription}</p>
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
          ? `2つ以上の連続回帰目的と1つの組成式列に対応します。${mixedModel ? "連続process列＋カテゴリprocess列を扱います。" : "process列は連続値に対応します。"} 目的ごとにモデルを分割せず、共有表現とtask covarianceを1つの相関付きモデルとして学習します。`
          : mixedModel
            ? "1つ以上の連続回帰目的、1つの組成式列、連続process列＋カテゴリprocess列に対応します。複数目的では目的ごとに独立したモデルを構築します。"
            : "1つ以上の連続回帰目的、1つの組成式列、連続process列に対応します。通常のCrabNet系で複数目的を選ぶと独立ModelListとなり、CrabNet-MultiTask系では目的間相関を学習します。"}
      </p>
    </article>
  );
}
