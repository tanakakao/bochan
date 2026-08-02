import { useState } from "react";
import {
  loadNoiseAlpha,
  saveNoiseAlpha,
  supportsNoiseAlpha
} from "../noiseAlphaSettings";

interface NoiseAlphaSettingsProps {
  modelType: string;
  hasRegressionTargets: boolean;
}

/** Configure the Gaussian observation-noise variance floor for supported models. */
export default function NoiseAlphaSettings({
  modelType,
  hasRegressionTargets
}: NoiseAlphaSettingsProps) {
  const [alpha, setAlpha] = useState(() => loadNoiseAlpha());
  const supported = hasRegressionTargets && supportsNoiseAlpha(modelType);

  function update(value: number) {
    setAlpha(value);
    saveNoiseAlpha(value);
  }

  return (
    <section className="transform-card">
      <div className="transform-card-heading">
        <div>
          <span className="panel-kicker">OBSERVATION NOISE</span>
          <h4>観測ノイズ下限 α</h4>
        </div>
        <span className={`status-chip ${supported ? "success" : "warning"}`}>
          {supported ? "有効" : "対象外"}
        </span>
      </div>
      <label>
        ノイズ分散の下限
        <input
          type="number"
          min={1e-12}
          step="any"
          value={alpha}
          disabled={!supported}
          onChange={(event) => update(Number(event.target.value))}
        />
      </label>
      <p>
        Gaussian likelihoodが学習する観測ノイズ分散の下限です。目的変数を標準化する場合は、標準化後のスケールに適用されます。
      </p>
      {!supported && (
        <p className="settings-note">
          この設定はBase GP、Deep GP、Deep Kernel、PCA、REMBO、Robust RRPの回帰出力にのみ適用されます。
        </p>
      )}
    </section>
  );
}
