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
    <>
      <div className="model-settings-grid observation-noise-fields">
        <label>
          ノイズ分散の下限 α
          <input
            type="number"
            min={1e-12}
            step="any"
            value={alpha}
            disabled={!supported}
            onChange={(event) => update(Number(event.target.value))}
          />
        </label>
      </div>
      <p className="settings-note">
        Gaussian likelihoodが学習する観測ノイズ分散の下限です。目的変数を標準化する場合は、標準化後のスケールに適用されます。
      </p>
      {!supported && (
        <p className="settings-note">
          この設定はBase GP、Deep GP、Deep Kernel、PCA、REMBO、Robust RRPの回帰出力にのみ適用されます。
        </p>
      )}
    </>
  );
}
