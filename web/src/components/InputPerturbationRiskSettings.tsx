import { useState } from "react";
import type { AcquisitionFamily } from "../types";
import {
  loadInputPerturbationRiskSettings,
  saveInputPerturbationRiskSettings,
  type InputPerturbationRiskSettings,
  type InputPerturbationRiskType
} from "../webRunSettings";

interface InputPerturbationRiskSettingsProps {
  disabled?: boolean;
  acquisitionFamily?: AcquisitionFamily;
}

function riskDescription(
  riskType: InputPerturbationRiskType,
  acquisitionFamily: AcquisitionFamily | undefined
): string {
  if (acquisitionFamily === "level_set_estimation") {
    if (riskType === "var") return "各入力摂動で計算した境界探索スコアの悪い側α割合の境界値で候補を評価します。";
    if (riskType === "cvar") return "各入力摂動で計算した境界探索スコアの悪い側α割合の平均値で候補を評価します。";
    return "各入力摂動で計算した境界探索スコアを平均して候補を評価します。";
  }
  if (riskType === "var") return "悪い側α割合の境界値で候補を評価します。";
  if (riskType === "cvar") return "悪い側α割合の平均値で候補を評価します。";
  return "すべての摂動サンプルの平均値で候補を評価します。";
}

/** Web controls for aggregating values expanded by InputPerturbation. */
export default function InputPerturbationRiskSettingsControl({
  disabled = false,
  acquisitionFamily
}: InputPerturbationRiskSettingsProps) {
  const [settings, setSettings] = useState<InputPerturbationRiskSettings>(
    loadInputPerturbationRiskSettings
  );

  function patch(patchValue: Partial<InputPerturbationRiskSettings>) {
    const next = { ...settings, ...patchValue };
    setSettings(next);
    saveInputPerturbationRiskSettings(next);
  }

  return (
    <>
      <label>
        Risk集約
        <select
          value={settings.riskType}
          disabled={disabled}
          onChange={(event) => patch({ riskType: event.target.value as InputPerturbationRiskType })}
        >
          <option value="none">平均（Expectation）</option>
          <option value="var">VaR</option>
          <option value="cvar">CVaR</option>
        </select>
      </label>
      {settings.riskType !== "none" && (
        <label>
          悪い側の割合 α
          <input
            type="number"
            min={0.01}
            max={1}
            step={0.05}
            value={settings.alpha}
            disabled={disabled}
            onChange={(event) => patch({ alpha: Number(event.target.value) })}
          />
        </label>
      )}
      <small className="settings-note">
        {disabled
          ? "VaR/CVaRは現在、ベイズ最適化とレベルセット推定で使用できます。"
          : riskDescription(settings.riskType, acquisitionFamily)}
      </small>
    </>
  );
}
