from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(name: str) -> Path:
    return ROOT / name


def replace_once(file_name: str, old: str, new: str) -> None:
    file_path = path(file_name)
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected snippet not found in {file_name}:\n{old[:500]}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_file(file_name: str, content: str) -> None:
    file_path = path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.rstrip() + "\n"
    if file_path.exists() and file_path.read_text(encoding="utf-8") == normalized:
        return
    file_path.write_text(normalized, encoding="utf-8")


# ---------------------------------------------------------------------------
# Frontend target schema / controls
# ---------------------------------------------------------------------------
replace_once(
    "web/src/types.ts",
    '''  /** One or more desired ordinal classes when goal is `target`. */
  target_values?: TargetClassValue[];
}''',
    '''  /** One or more desired ordinal classes when goal is `target`. */
  target_values?: TargetClassValue[];
  /** Relative contribution to a multi-output level-set acquisition. */
  level_set_weight?: number;
}''',
)

replace_once(
    "web/src/components/TargetProposalSettings.tsx",
    '''    if (isLevelSet) {
      return (
        <span className="muted-cell" title="レベルセット推定では最大化・最小化方向を使用しません。">
          境界推定
        </span>
      );
    }''',
    '''    if (isLevelSet) {
      return (
        <div className="table-field">
          <span className="muted-cell" title="レベルセット推定では最大化・最小化方向を使用しません。">
            境界推定
          </span>
          {targetColumns.length > 1 && (
            <label className="table-field">
              <span>境界重み</span>
              <input
                type="number"
                min={0}
                step={0.1}
                value={setting.level_set_weight ?? 1}
                aria-label={`${target}のレベルセット重み`}
                onChange={(event) => patchTargetSetting(target, {
                  level_set_weight: numberOrUndefined(event.target.value) ?? 0
                })}
              />
            </label>
          )}
        </div>
      );
    }''',
)

replace_once(
    "web/src/components/TargetProposalSettings.tsx",
    '''              <th>目的変数</th><th>最適化対象</th><th>{isLevelSet ? "探索" : "方向"}</th><th>制約</th>
              <th>{isLevelSet ? "境界しきい値／目標値" : "しきい値／目標値"}</th><th>対象クラス</th>''',
    '''              <th>目的変数</th><th>最適化対象</th><th>{isLevelSet ? "探索" : "方向"}</th>
              <th>{isLevelSet ? "境界条件 / 制約" : "制約"}</th>
              <th>{isLevelSet ? "境界しきい値／目標値" : "しきい値／目標値"}</th><th>対象クラス</th>''',
)

replace_once(
    "web/src/components/TargetProposalSettings.tsx",
    '''                      disabled={!isLevelSet && targetMode}
                      title={!isLevelSet && targetMode ? "目標値は方向で設定されています。" : undefined}
                      onChange={(event) => changeConstraintGoal(''',
    '''                      disabled={!isLevelSet && targetMode}
                      title={isLevelSet
                        ? setting.optimize
                          ? "最適化対象ではLSEの境界条件として使用し、候補のhard constraintにはしません。"
                          : "最適化対象外では候補の実行可能性制約として使用します。"
                        : targetMode
                          ? "目標値は方向で設定されています。"
                          : undefined}
                      onChange={(event) => changeConstraintGoal(''',
)

replace_once(
    "web/src/components/TargetProposalSettings.tsx",
    '''        {isLevelSet
          ? "レベルセット推定では最大化・最小化方向を設定しません。回帰・順序回帰の「目標値」は制約欄から選択できます。Multiclass は選択したクラス群の合計確率をしきい値と比較し、順序回帰の「以上／以下」はクラス順の順位を境界として扱います。"
          : "方向・制約・対象クラスは候補提案時にのみ使用します。ここを変更しても、互換性のある学習済みモデルは再利用できます。"}''',
    '''        {isLevelSet
          ? "レベルセット推定では最大化・最小化方向を設定しません。最適化対象行の「以上／以下／目標値」は探索する境界を定義し、hard constraintにはしません。チェックを外した行の「以上／以下」は実行可能性制約として使用します。Multiclass は選択クラス群の合計確率、順序回帰は設定済みクラス順を境界尺度として扱います。複数出力の境界重みは相対値として正規化されます。"
          : "方向・制約・対象クラスは候補提案時にのみ使用します。ここを変更しても、互換性のある学習済みモデルは再利用できます。"}''',
)

# ---------------------------------------------------------------------------
# LSE acquisition parameter UI
# ---------------------------------------------------------------------------
replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''function searchMethodFamilyFor(searchMethod: SearchMethod): SearchMethodFamily {
  return SEARCH_METHOD_OPTIONS.find((option) => option.value === searchMethod)?.family ?? "gradient";
}

/** Configures objectives, search space, constraints, acquisition, and candidate search. */''',
    '''function searchMethodFamilyFor(searchMethod: SearchMethod): SearchMethodFamily {
  return SEARCH_METHOD_OPTIONS.find((option) => option.value === searchMethod)?.family ?? "gradient";
}

function compactAcquisitionName(value: string): string {
  return value.replace(/[_\\-\\s]/g, "").toLowerCase();
}

function levelSetParameter(name: string): {
  label: string;
  min: number;
  defaultValue: number;
  help: string;
} {
  const key = compactAcquisitionName(name);
  if (key === "boundaryvariance") {
    return {
      label: "境界幅 τ",
      min: 1e-12,
      defaultValue: 1,
      help: "小さいほど境界しきい値の近傍を強く重視します。"
    };
  }
  if (key === "icu") {
    return {
      label: "Bandwidth",
      min: 0,
      defaultValue: 0,
      help: "0では予測標準偏差を自動的に帯域幅として使用します。"
    };
  }
  return {
    label: "Straddle β",
    min: 0,
    defaultValue: 1.96,
    help: "大きいほど不確実性を強く評価し、境界近傍から広めに探索します。"
  };
}

/** Configures objectives, search space, constraints, acquisition, and candidate search. */''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''  const searchMethodFamily = searchMethodFamilyFor(searchMethod);
  const availableSearchMethodFamilies = useMemo(''',
    '''  const searchMethodFamily = searchMethodFamilyFor(searchMethod);
  const lseParameter = levelSetParameter(acquisition);
  const availableSearchMethodFamilies = useMemo(''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''    } else {
      setAcquisition("straddle");
    }
  }

  function changeSearchMethod(nextMethod: SearchMethod) {''',
    '''    } else {
      setAcquisition("straddle");
      setBeta(levelSetParameter("straddle").defaultValue);
    }
  }

  function changeAcquisition(nextAcquisition: string) {
    setAcquisition(nextAcquisition);
    if (acquisitionFamily === "level_set_estimation") {
      setBeta(levelSetParameter(nextAcquisition).defaultValue);
    }
  }

  function changeSearchMethod(nextMethod: SearchMethod) {''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''    if (
      acquisitionFamily === "level_set_estimation" &&
      optimizedTargetSettings.some((setting) => setting.goal === "none")
    ) {
      errors.push("レベルセット推定では、最適化対象ごとに以上・以下・目標値のいずれかを設定してください。");
    }
    if (searchMethod === "nsgaii"''',
    '''    if (
      acquisitionFamily === "level_set_estimation" &&
      optimizedTargetSettings.some((setting) => setting.goal === "none")
    ) {
      errors.push("レベルセット推定では、最適化対象ごとに以上・以下・目標値のいずれかを設定してください。");
    }
    if (acquisitionFamily === "level_set_estimation") {
      const key = compactAcquisitionName(acquisition);
      if (!Number.isFinite(beta) || beta < 0) {
        errors.push("LSEの獲得関数パラメータは0以上の有限値にしてください。");
      } else if (key === "boundaryvariance" && beta <= 0) {
        errors.push("Boundary Varianceのτは0より大きくしてください。");
      }
      const levelSetWeights = optimizedTargetSettings.map(
        (setting) => Number(setting.level_set_weight ?? 1)
      );
      if (levelSetWeights.some((weight) => !Number.isFinite(weight) || weight < 0)) {
        errors.push("LSEの境界重みは0以上の有限値にしてください。");
      } else if (levelSetWeights.reduce((sum, weight) => sum + weight, 0) <= 0) {
        errors.push("LSEの境界重みは少なくとも1つを0より大きくしてください。");
      }
    }
    if (searchMethod === "nsgaii"''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''  }, [
    acquisitionFamily,
    candidateSettingsValid,''',
    '''  }, [
    acquisition,
    acquisitionFamily,
    beta,
    candidateSettingsValid,''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''            <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)} disabled={searchMethod === "nsgaii"}>''',
    '''            <select value={acquisition} onChange={(event) => changeAcquisition(event.target.value)} disabled={searchMethod === "nsgaii"}>''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''          {acquisition.toUpperCase().includes("UCB") && searchMethod !== "nsgaii" && (
            <label>Beta<input type="number" min={0} step="any" value={beta} onChange={(event) => setBeta(Number(event.target.value))} /></label>
          )}
          <p className="settings-note">''',
    '''          {acquisition.toUpperCase().includes("UCB") && searchMethod !== "nsgaii" && (
            <label>Beta<input type="number" min={0} step="any" value={beta} onChange={(event) => setBeta(Number(event.target.value))} /></label>
          )}
          {acquisitionFamily === "level_set_estimation" && (
            <>
              <label>
                {lseParameter.label}
                <input
                  type="number"
                  min={lseParameter.min}
                  step="any"
                  value={beta}
                  onChange={(event) => setBeta(Number(event.target.value))}
                />
              </label>
              <small className="settings-note">{lseParameter.help}</small>
            </>
          )}
          <p className="settings-note">''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''            <InputPerturbationRiskSettingsControl
              disabled={acquisitionFamily !== "bayesian_optimization"}
            />''',
    '''            <InputPerturbationRiskSettingsControl
              acquisitionFamily={acquisitionFamily}
              disabled={acquisitionFamily === "active_learning"}
            />''',
)

write_file(
    "web/src/components/InputPerturbationRiskSettings.tsx",
    r'''import { useState } from "react";
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
''',
)

# ---------------------------------------------------------------------------
# Frontend request / restore contract
# ---------------------------------------------------------------------------
replace_once(
    "web/src/api.ts",
    '''  if (setting.direction !== "maximize" && setting.direction !== "minimize") {
    return `${setting.target}: 最大化または最小化を選択してください。`;
  }
  if (setting.task_type === "regression") {''',
    '''  if (setting.direction !== "maximize" && setting.direction !== "minimize") {
    return `${setting.target}: 最大化または最小化を選択してください。`;
  }
  if (
    setting.level_set_weight !== undefined &&
    (!Number.isFinite(Number(setting.level_set_weight)) || Number(setting.level_set_weight) < 0)
  ) {
    return `${setting.target}: LSEの境界重みは0以上の有限値にしてください。`;
  }
  if (setting.task_type === "regression") {''',
)

replace_once(
    "web/src/api.ts",
    '''  const riskSupported = input.inputPerturbation && input.acquisitionFamily === "bayesian_optimization";''',
    '''  const riskSupported = input.inputPerturbation && (
    input.acquisitionFamily === "bayesian_optimization" ||
    input.acquisitionFamily === "level_set_estimation"
  );''',
)

replace_once(
    "web/src/api.ts",
    '''          web_family: input.acquisitionFamily,
          web_risk_type: effectiveRiskType,
          web_risk_alpha: perturbationRisk.alpha''',
    '''          web_family: input.acquisitionFamily,
          web_level_set_parameter: input.acquisitionFamily === "level_set_estimation" ? input.beta : null,
          web_risk_type: effectiveRiskType,
          web_risk_alpha: perturbationRisk.alpha''',
)

replace_once(
    "web/src/modelArtifactRestore.ts",
    '''  const acquisition = String(acquisitionSettings.name ?? "EI");
  const beta = finiteNumber(acquisitionSettings.beta, 2);
  const q = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.q, 3)));''',
    '''  const acquisition = String(acquisitionSettings.name ?? "EI");
  const acquisitionKey = acquisition.replace(/[_\\-\\s]/g, "").toLowerCase();
  const defaultLevelSetParameter = acquisitionKey === "boundaryvariance"
    ? 1
    : acquisitionKey === "icu"
      ? 0
      : 1.96;
  const savedLevelSetParameter = acquisitionKwargs.web_level_set_parameter;
  const beta = acquisitionFamily === "level_set_estimation"
    ? savedLevelSetParameter === null || savedLevelSetParameter === undefined
      ? defaultLevelSetParameter
      : finiteNumber(savedLevelSetParameter, defaultLevelSetParameter)
    : finiteNumber(acquisitionSettings.beta, 2);
  const q = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.q, 3)));''',
)

# ---------------------------------------------------------------------------
# Backend target / constraint contract
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/serving/webapp/target_settings.py",
    '''from __future__ import annotations

from typing import Any''',
    '''from __future__ import annotations

import math
from typing import Any''',
)

replace_once(
    "src/bochan/serving/webapp/target_settings.py",
    '''                "target_values": [],
                "legacy": True,''',
    '''                "target_values": [],
                "level_set_weight": 1.0,
                "legacy": True,''',
)

replace_once(
    "src/bochan/serving/webapp/target_settings.py",
    '''        target_values = _list_values(setting.get("target_values"))

        # Compatibility with the previous `goal=target, value=<class>` contract.''',
    '''        target_values = _list_values(setting.get("target_values"))
        try:
            level_set_weight = float(setting.get("level_set_weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{target}: level_set_weight must be a non-negative finite number."
            ) from exc
        if not math.isfinite(level_set_weight) or level_set_weight < 0.0:
            raise ValueError(
                f"{target}: level_set_weight must be a non-negative finite number."
            )

        # Compatibility with the previous `goal=target, value=<class>` contract.''',
)

replace_once(
    "src/bochan/serving/webapp/target_settings.py",
    '''                "target_values": target_values,
                "legacy": False,''',
    '''                "target_values": target_values,
                "level_set_weight": level_set_weight,
                "legacy": False,''',
)

replace_once(
    "src/bochan/serving/webapp/search_settings.py",
    '''    directions: dict[str, str],
    hybrid_model: bool,
) -> Any | None:
    """Build target constraints, using class probabilities for classification.

    Classification constraints are model-dependent so above/below is applied to
    the selected class probability for BO, active learning, and level-set methods.
    """

    if all(bool(setting.get("legacy")) for setting in target_settings):''',
    '''    directions: dict[str, str],
    hybrid_model: bool,
    exclude_optimized_boundaries: bool = False,
) -> Any | None:
    """Build target constraints, using class probabilities for classification.

    For level-set estimation, optimized above/below settings define the contour
    itself and must not also become feasibility constraints. Setting
    ``exclude_optimized_boundaries=True`` keeps constraint-only targets active
    while leaving optimized targets free to sample both sides of the boundary.
    """

    if all(bool(setting.get("legacy")) for setting in target_settings):
        if exclude_optimized_boundaries:
            return None''',
)

replace_once(
    "src/bochan/serving/webapp/search_settings.py",
    '''    specs: list[Any] = []
    for setting in target_settings:
        goal = str(setting["goal"])''',
    '''    specs: list[Any] = []
    for setting in target_settings:
        if exclude_optimized_boundaries and bool(setting.get("optimize", True)):
            continue
        goal = str(setting["goal"])''',
)

write_file(
    "src/bochan/serving/webapp/level_set_settings.py",
    '''"""Source-level Level-Set Estimation settings for the Web workbench."""

from __future__ import annotations

import math
from typing import Any

from .target_roles import level_set_thresholds

_WEB_LEVEL_SET_PARAMETER_KEY = "web_level_set_parameter"


def level_set_output_weights(
    *,
    target_columns: list[str],
    target_settings: list[dict[str, Any]],
    objective_targets: list[str],
) -> list[float]:
    """Return relative Web LSE weights aligned with all modeled outputs."""

    settings_by_target = {
        str(setting["target"]): setting for setting in target_settings
    }
    selected = set(objective_targets)
    weights: list[float] = []
    for target in target_columns:
        if target not in selected:
            weights.append(0.0)
            continue
        setting = settings_by_target.get(target)
        if setting is None:
            raise ValueError(f"Missing target setting for level-set target: {target}")
        try:
            weight = float(setting.get("level_set_weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{target}: level_set_weight must be a non-negative finite number."
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{target}: level_set_weight must be a non-negative finite number."
            )
        weights.append(weight)

    if sum(weights) <= 0.0:
        raise ValueError(
            "At least one optimized level-set target must have a positive weight."
        )
    return weights


def _configure_acquisition_parameter(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
) -> None:
    raw_value = acqf_kwargs.pop(_WEB_LEVEL_SET_PARAMETER_KEY, None)
    if raw_value is None:
        return
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Web LSE acquisition parameter must be numeric.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Web LSE acquisition parameter must be non-negative and finite.")

    if acq_key == "straddle":
        acqf_kwargs.setdefault("beta", value)
        return
    if acq_key == "boundaryvariance":
        if value <= 0.0:
            raise ValueError("Boundary Variance tau must be greater than zero.")
        acqf_kwargs.setdefault("tau", value)
        return
    if acq_key == "icu":
        # Zero is the Web sentinel for the class default: bandwidth = posterior std.
        if value > 0.0:
            acqf_kwargs.setdefault("bandwidth", value)
        return
    raise ValueError(f"Unsupported Web level-set acquisition: {acq_key!r}.")


def _risk_score_objective(
    *,
    multi_output: bool,
    n_w: int,
    risk_type: str,
    alpha: float,
) -> Any | None:
    normalized = str(risk_type).lower()
    if normalized in {"", "none"}:
        return None
    if normalized not in {"var", "cvar"}:
        raise ValueError("LSE input-perturbation risk_type must be none, var, or cvar.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("LSE input-perturbation risk alpha must be in (0, 1].")

    if multi_output:
        from bochan.acquisition.regression.levelset_estimation.multi_output import (
            MultiOutputRegressionLevelSetScoreObjective,
        )

        objective_cls = MultiOutputRegressionLevelSetScoreObjective
    else:
        from bochan.acquisition.regression.levelset_estimation.single_output import (
            RegressionLevelSetScoreObjective,
        )

        objective_cls = RegressionLevelSetScoreObjective
    return objective_cls(
        n_w=n_w,
        risk_type=normalized,
        alpha=float(alpha),
        maximize=True,
    )


def configure_level_set_acqf_kwargs(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
    train_x: Any,
    target_columns: list[str],
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
    objective_targets: list[str],
    input_perturbation: bool,
    n_w: int,
    risk_type: str = "none",
    risk_alpha: float = 0.2,
) -> dict[str, Any]:
    """Attach Web LSE thresholds, weights, duplicate references, and risk."""

    if acq_key not in {"straddle", "boundaryvariance", "icu"}:
        raise ValueError(f"Unsupported Web level-set acquisition: {acq_key!r}.")

    thresholds = level_set_thresholds(
        target_columns=target_columns,
        target_metadata=target_metadata,
        objective_targets=objective_targets,
    )
    acqf_kwargs.setdefault("thresholds", thresholds)
    acqf_kwargs.setdefault(
        "output_weights",
        level_set_output_weights(
            target_columns=target_columns,
            target_settings=target_settings,
            objective_targets=objective_targets,
        ),
    )
    acqf_kwargs.setdefault("output_reduction", "weighted_mean")
    acqf_kwargs.setdefault("X_observed", train_x)
    _configure_acquisition_parameter(acqf_kwargs, acq_key=acq_key)

    if input_perturbation:
        n_w = int(n_w)
        if n_w <= 0:
            raise ValueError("LSE InputPerturbation n_w must be positive.")
        acqf_kwargs.setdefault("n_w", n_w)
        objective = _risk_score_objective(
            multi_output=len(target_columns) > 1,
            n_w=n_w,
            risk_type=risk_type,
            alpha=risk_alpha,
        )
        if objective is not None:
            acqf_kwargs.setdefault("objective", objective)
    elif str(risk_type).lower() not in {"", "none"}:
        raise ValueError("LSE VaR/CVaR requires input_perturbation=true.")

    return acqf_kwargs


__all__ = [
    "configure_level_set_acqf_kwargs",
    "level_set_output_weights",
]
''',
)

# ---------------------------------------------------------------------------
# Source-level Web risk settings (remove runtime function replacement)
# ---------------------------------------------------------------------------
write_file(
    "src/bochan/serving/webapp/risk_settings.py",
    '''"""Request-local risk settings for the Web input-perturbation workflow."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Iterator

from .prediction_shapes import normalize_prediction_rows

_WEB_RISK_TYPE_KEY = "web_risk_type"
_WEB_RISK_ALPHA_KEY = "web_risk_alpha"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_input_perturbation_risk",
    default=None,
)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def resolve_web_risk_settings(request: Any) -> dict[str, Any]:
    """Normalize Web risk markers without modifying API or workflow functions."""

    acquisition = _mapping(getattr(request, "acquisition", None))
    kwargs = _mapping(acquisition.get("acqf_kwargs"))
    risk_type = str(kwargs.get(_WEB_RISK_TYPE_KEY, "none")).lower()
    if risk_type not in {"none", "var", "cvar"}:
        raise ValueError("Input perturbation risk_type must be none, var, or cvar.")

    try:
        alpha = float(kwargs.get(_WEB_RISK_ALPHA_KEY, 0.2))
    except (TypeError, ValueError) as exc:
        raise ValueError("Input perturbation risk alpha must be numeric.") from exc
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Input perturbation risk alpha must be in (0, 1].")

    input_perturbation = bool(getattr(request, "input_perturbation", False))
    family = str(kwargs.get("web_family", "bayesian_optimization")).lower()
    enabled = input_perturbation and risk_type in {"var", "cvar"}
    if not input_perturbation and risk_type != "none":
        raise ValueError("VaR/CVaR requires input_perturbation=true.")
    if enabled and family not in {"bayesian_optimization", "level_set_estimation"}:
        raise ValueError(
            "VaR/CVaR input perturbation risk is available for Bayesian optimization "
            "or level-set estimation in the Web workbench."
        )

    return {
        "input_perturbation": input_perturbation,
        "risk_type": risk_type if input_perturbation else "none",
        "risk_alpha": alpha,
        "risk_enabled": enabled,
        "acquisition_family": family,
    }


@contextmanager
def web_risk_run(request: Any) -> Iterator[dict[str, Any]]:
    """Activate one request's Web input-perturbation risk metadata."""

    state = resolve_web_risk_settings(request)
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_web_risk_report() -> dict[str, Any]:
    """Return the active request's normalized risk settings."""

    return dict(_STATE.get() or {})


def apply_web_risk_to_objective_config(
    objective_config: Any,
    report: dict[str, Any],
) -> Any:
    """Return a BO ObjectiveConfig carrying explicit Web VaR/CVaR settings."""

    if objective_config is None or not report.get("risk_enabled"):
        return objective_config
    return replace(
        objective_config,
        risk_type=str(report["risk_type"]),
        alpha=float(report["risk_alpha"]),
    )


def normalize_web_prediction_rows(
    value: Any,
    *,
    n_rows: int,
    report: dict[str, Any],
) -> Any:
    """Aggregate InputPerturbation-expanded baseline values explicitly."""

    risk_type = str(report.get("risk_type")) if report.get("risk_enabled") else None
    return normalize_prediction_rows(
        value,
        n_rows=n_rows,
        risk_type=risk_type,
        alpha=float(report.get("risk_alpha", 0.2)),
    )


def attach_web_risk_metadata(
    result: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Attach effective risk settings to a Web result payload."""

    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "input_perturbation_risk_type": report.get("risk_type", "none"),
            "input_perturbation_risk_alpha": float(report.get("risk_alpha", 0.2)),
            "input_perturbation_risk_enabled": bool(report.get("risk_enabled")),
        }
    )
    result["metadata"] = metadata
    return metadata


__all__ = [
    "apply_web_risk_to_objective_config",
    "attach_web_risk_metadata",
    "current_web_risk_report",
    "normalize_web_prediction_rows",
    "resolve_web_risk_settings",
    "web_risk_run",
]
''',
)

replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''from .risk_settings import (
    attach_web_risk_metadata,
    install_web_risk_adapters,
    web_risk_run,
)''',
    '''from .risk_settings import attach_web_risk_metadata, web_risk_run''',
)
replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''install_workflow_adapters(_workflows_tabular)
install_web_risk_adapters(_workflows_tabular)''',
    '''install_workflow_adapters(_workflows_tabular)''',
)
replace_once(
    "src/bochan/serving/webapp/workflows.py",
    '''_workflows_tabular = import_module(".workflows_tabular", package=__package__)
_workflows_tabular._as_2d = normalize_prediction_rows

install_workflow_adapters(_workflows_tabular)''',
    '''_workflows_tabular = import_module(".workflows_tabular", package=__package__)

install_workflow_adapters(_workflows_tabular)''',
)

# ---------------------------------------------------------------------------
# Main Web workflow wiring
# ---------------------------------------------------------------------------
replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''from .logging import current_request_id, get_logger, log_event
from .search_settings import (''',
    '''from .level_set_settings import configure_level_set_acqf_kwargs
from .logging import current_request_id, get_logger, log_event
from .risk_settings import (
    apply_web_risk_to_objective_config,
    normalize_web_prediction_rows,
    resolve_web_risk_settings,
)
from .search_settings import (''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''from .target_settings import (
    _as_2d,
    _build_outcome_constraint_config,''',
    '''from .target_settings import (
    _build_outcome_constraint_config,''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''def _acquisition_family(acqf_kwargs: dict[str, Any]) -> str:
    family = str(acqf_kwargs.pop("web_family", "bayesian_optimization")).lower()''',
    '''def _acquisition_family(acqf_kwargs: dict[str, Any]) -> str:
    family = str(acqf_kwargs.pop("web_family", "bayesian_optimization")).lower()
    acqf_kwargs.pop("web_risk_type", None)
    acqf_kwargs.pop("web_risk_alpha", None)''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''            "target_values": setting.get("target_values", []),
        }''',
    '''            "target_values": setting.get("target_values", []),
            "level_set_weight": float(setting.get("level_set_weight", 1.0)),
        }''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    acquisition_family = _acquisition_family(acqf_kwargs)''',
    '''    risk_settings = resolve_web_risk_settings(request)
    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    acquisition_family = _acquisition_family(acqf_kwargs)''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''    if hybrid_model:
        objective_values_full = _as_2d(
            optimizer.model.posterior(train_x, output_mode="objective").mean,
            n_rows=int(train_x.shape[0]),
        ).detach()''',
    '''    if hybrid_model:
        objective_values_full = normalize_web_prediction_rows(
            optimizer.model.posterior(train_x, output_mode="objective").mean,
            n_rows=int(train_x.shape[0]),
            report=risk_settings,
        ).detach()''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''        directions=directions,
        hybrid_model=hybrid_model,
    )''',
    '''        directions=directions,
        hybrid_model=hybrid_model,
        exclude_optimized_boundaries=acquisition_family == "level_set_estimation",
    )''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''    else:
        if acq_key not in {"straddle", "boundaryvariance", "icu"}:
            raise ValueError(f"Level-set estimation supports straddle, boundary_variance, or ICU, got {acq_name}.")
        objective_config = None
        acqf_kwargs.setdefault(
            "thresholds",
            level_set_thresholds(
                target_columns=target_columns,
                target_metadata=target_metadata,
                objective_targets=objective_targets,
            ),
        )
        acqf_kwargs.setdefault(
            "output_weights",
            objective_weights(
                target_columns=target_columns,
                objective_targets=objective_targets,
            ),
        )
        acqf_kwargs.setdefault("output_reduction", "weighted_mean")
        acqf_kwargs.setdefault("X_observed", train_x)
        data_context = DataContext(
            X_baseline=train_x,
            Y_baseline=objective_values,
        )

    acq_config = AcquisitionConfig(''',
    '''    else:
        if acq_key not in {"straddle", "boundaryvariance", "icu"}:
            raise ValueError(f"Level-set estimation supports straddle, boundary_variance, or ICU, got {acq_name}.")
        objective_config = None
        configure_level_set_acqf_kwargs(
            acqf_kwargs,
            acq_key=acq_key,
            train_x=train_x,
            target_columns=target_columns,
            target_settings=target_settings,
            target_metadata=target_metadata,
            objective_targets=objective_targets,
            input_perturbation=bool(request.input_perturbation),
            n_w=int(request.n_w),
            risk_type=str(risk_settings["risk_type"]),
            risk_alpha=float(risk_settings["risk_alpha"]),
        )
        data_context = DataContext(
            X_baseline=train_x,
            Y_baseline=objective_values,
        )

    if acquisition_family == "bayesian_optimization":
        objective_config = apply_web_risk_to_objective_config(
            objective_config,
            risk_settings,
        )

    acq_config = AcquisitionConfig(''',
)

# level_set_thresholds remains imported by older helper/tests only if referenced.
workflow_path = path("src/bochan/serving/webapp/workflows_tabular.py")
workflow_text = workflow_path.read_text(encoding="utf-8")
if workflow_text.count("level_set_thresholds") == 1:
    workflow_text = workflow_text.replace("    level_set_thresholds,\n", "", 1)
    workflow_path.write_text(workflow_text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
write_file(
    "tests/test_webapp_input_perturbation_risk.py",
    '''from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import ObjectiveConfig
from bochan.serving.webapp.risk_settings import (
    apply_web_risk_to_objective_config,
    attach_web_risk_metadata,
    current_web_risk_report,
    normalize_web_prediction_rows,
    resolve_web_risk_settings,
    web_risk_run,
)
from bochan.serving.webapp.workflows_tabular import _acquisition_family


def _request(
    *,
    risk_type: str = "cvar",
    alpha: float = 0.25,
    family: str = "bayesian_optimization",
    perturbation: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_perturbation=perturbation,
        acquisition=SimpleNamespace(
            acqf_kwargs={
                "web_family": family,
                "web_risk_type": risk_type,
                "web_risk_alpha": alpha,
            }
        ),
    )


def test_web_risk_context_normalizes_request_settings() -> None:
    with web_risk_run(_request(risk_type="var", alpha=0.4)) as report:
        assert report == {
            "input_perturbation": True,
            "risk_type": "var",
            "risk_alpha": 0.4,
            "risk_enabled": True,
            "acquisition_family": "bayesian_optimization",
        }
        assert current_web_risk_report() == report

    assert current_web_risk_report() == {}


def test_web_risk_allows_level_set_estimation() -> None:
    report = resolve_web_risk_settings(
        _request(family="level_set_estimation", risk_type="cvar")
    )
    assert report["risk_enabled"] is True
    assert report["acquisition_family"] == "level_set_estimation"


def test_web_risk_requires_input_perturbation() -> None:
    with pytest.raises(ValueError, match="requires input_perturbation"):
        resolve_web_risk_settings(_request(perturbation=False))


def test_web_risk_rejects_active_learning() -> None:
    with pytest.raises(ValueError, match="Bayesian optimization or level-set estimation"):
        resolve_web_risk_settings(_request(family="active_learning"))


def test_web_risk_keys_are_removed_before_acquisition_construction() -> None:
    kwargs = {
        "web_family": "level_set_estimation",
        "web_risk_type": "cvar",
        "web_risk_alpha": 0.25,
    }

    family = _acquisition_family(kwargs)

    assert family == "level_set_estimation"
    assert kwargs == {}


def test_web_risk_is_applied_to_objective_config_without_engine_patch() -> None:
    config = ObjectiveConfig(mode="scalar", output=0)
    report = resolve_web_risk_settings(_request(risk_type="cvar", alpha=0.25))

    resolved = apply_web_risk_to_objective_config(config, report)

    assert resolved is not config
    assert resolved.risk_type == "cvar"
    assert resolved.alpha == 0.25


def test_workflow_baseline_uses_same_cvar_aggregation() -> None:
    values = torch.tensor([[1.0], [2.0], [8.0], [9.0]], dtype=torch.double)
    report = resolve_web_risk_settings(_request(risk_type="cvar", alpha=0.5))

    actual = normalize_web_prediction_rows(values, n_rows=1, report=report)

    torch.testing.assert_close(
        actual,
        torch.tensor([[1.5]], dtype=torch.double),
    )


def test_web_risk_metadata_is_serialized() -> None:
    result = {"metadata": {"existing": True}}

    metadata = attach_web_risk_metadata(
        result,
        {
            "risk_type": "var",
            "risk_alpha": 0.2,
            "risk_enabled": True,
        },
    )

    assert metadata["existing"] is True
    assert metadata["input_perturbation_risk_type"] == "var"
    assert metadata["input_perturbation_risk_alpha"] == 0.2
    assert metadata["input_perturbation_risk_enabled"] is True
''',
)

write_file(
    "tests/test_webapp_lse_web_contract.py",
    '''from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.serving.webapp.level_set_settings import (
    configure_level_set_acqf_kwargs,
    level_set_output_weights,
)
from bochan.serving.webapp.search_settings import build_target_constraint_config


def _meta(target: str, value: float) -> dict[str, object]:
    return {
        "target": target,
        "internal_task": "regression",
        "goal": "above",
        "configured_value": value,
        "direction": "maximize",
        "class_index": None,
        "class_indices": [],
        "num_classes": None,
    }


def _setting(
    target: str,
    *,
    optimize: bool = True,
    weight: float = 1.0,
    goal: str = "above",
) -> dict[str, object]:
    return {
        "target": target,
        "task_type": "regression",
        "optimize": optimize,
        "direction": "maximize",
        "goal": goal,
        "value": 1.0,
        "level_set_weight": weight,
        "legacy": False,
    }


def test_level_set_output_weights_keep_constraint_only_outputs_at_zero() -> None:
    actual = level_set_output_weights(
        target_columns=["a", "guard", "b"],
        target_settings=[
            _setting("a", weight=2.0),
            _setting("guard", optimize=False, weight=9.0),
            _setting("b", weight=0.5),
        ],
        objective_targets=["a", "b"],
    )
    assert actual == pytest.approx([2.0, 0.0, 0.5])


def test_level_set_output_weights_require_positive_total() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        level_set_output_weights(
            target_columns=["a", "b"],
            target_settings=[_setting("a", weight=0.0), _setting("b", weight=0.0)],
            objective_targets=["a", "b"],
        )


@pytest.mark.parametrize(
    ("name", "parameter", "kwarg", "expected"),
    [
        ("straddle", 1.7, "beta", 1.7),
        ("boundaryvariance", 0.4, "tau", 0.4),
        ("icu", 0.3, "bandwidth", 0.3),
    ],
)
def test_level_set_parameter_is_routed_to_acquisition(
    name: str,
    parameter: float,
    kwarg: str,
    expected: float,
) -> None:
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {"web_level_set_parameter": parameter}

    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key=name,
        train_x=train_x,
        target_columns=["y"],
        target_settings=[_setting("y")],
        target_metadata={"y": _meta("y", 0.5)},
        objective_targets=["y"],
        input_perturbation=False,
        n_w=4,
    )

    assert "web_level_set_parameter" not in kwargs
    assert kwargs[kwarg] == pytest.approx(expected)


def test_icu_zero_parameter_keeps_automatic_bandwidth() -> None:
    kwargs: dict[str, object] = {"web_level_set_parameter": 0.0}
    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key="icu",
        train_x=torch.tensor([[0.0]], dtype=torch.double),
        target_columns=["y"],
        target_settings=[_setting("y")],
        target_metadata={"y": _meta("y", 0.5)},
        objective_targets=["y"],
        input_perturbation=False,
        n_w=4,
    )
    assert "bandwidth" not in kwargs


def test_boundary_variance_rejects_zero_tau() -> None:
    kwargs: dict[str, object] = {"web_level_set_parameter": 0.0}
    with pytest.raises(ValueError, match="tau must be greater than zero"):
        configure_level_set_acqf_kwargs(
            kwargs,
            acq_key="boundaryvariance",
            train_x=torch.tensor([[0.0]], dtype=torch.double),
            target_columns=["y"],
            target_settings=[_setting("y")],
            target_metadata={"y": _meta("y", 0.5)},
            objective_targets=["y"],
            input_perturbation=False,
            n_w=4,
        )


def test_level_set_input_perturbation_cvar_builds_score_objective() -> None:
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {"web_level_set_parameter": 1.96}

    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key="straddle",
        train_x=train_x,
        target_columns=["a", "b"],
        target_settings=[_setting("a", weight=2.0), _setting("b", weight=1.0)],
        target_metadata={"a": _meta("a", 0.2), "b": _meta("b", 0.8)},
        objective_targets=["a", "b"],
        input_perturbation=True,
        n_w=8,
        risk_type="cvar",
        risk_alpha=0.25,
    )

    assert kwargs["n_w"] == 8
    objective = kwargs["objective"]
    assert objective.__class__.__name__ == "MultiOutputRegressionLevelSetScoreObjective"
    assert objective.n_w == 8
    assert objective.risk_type == "cvar"
    assert objective.alpha == pytest.approx(0.25)
    assert kwargs["output_weights"] == pytest.approx([2.0, 1.0])


def test_optimized_lse_boundary_is_not_a_hard_outcome_constraint() -> None:
    settings = [
        _setting("boundary", optimize=True, goal="above"),
        _setting("guard", optimize=False, goal="below"),
    ]
    metadata = {
        "boundary": _meta("boundary", 0.5),
        "guard": {**_meta("guard", 0.2), "goal": "below"},
    }

    config = build_target_constraint_config(
        SimpleNamespace(outcome_constraints=[]),
        target_settings=settings,
        target_metadata=metadata,
        target_columns=["boundary", "guard"],
        directions={"boundary": "maximize", "guard": "maximize"},
        hybrid_model=True,
        exclude_optimized_boundaries=True,
    )

    assert config is not None
    assert config.constraints is not None
    assert len(config.constraints) == 1
    constraint = config.constraints[0]
    assert constraint.output == "guard"
    assert constraint.sense == "le"
    assert constraint.threshold == pytest.approx(0.2)
''',
)

# ---------------------------------------------------------------------------
# Documentation and focused CI
# ---------------------------------------------------------------------------
write_file(
    "docs/web_target_roles.md",
    '''# Web target roles and acquisition families

Each selected target is modeled, but its role in candidate generation is configured independently.

- **Optimization target**: included in the acquisition objective.
- **Constraint-only target**: modeled and evaluated for feasibility, but excluded from the acquisition objective.
- **Direction**: maximize or minimize for Bayesian optimization. Level-set estimation does not expose a maximize/minimize selector.
- **Target value**: treated as a distance objective for Bayesian optimization and as a zero-distance contour for level-set estimation.

At least one selected target must remain an optimization target.

The Optimize page exposes three acquisition families:

1. Bayesian optimization: EI, PI, UCB, EHVI, NEHVI, or NParEGO depending on objective count.
2. Active learning: posterior variance, predictive entropy, BALD, or NIPV.
3. Level-set estimation: Straddle, Boundary Variance, or ICU.

## Level-set estimation in the Web workbench

For an optimized LSE target, `above`, `below`, or `target` defines the contour to learn. It does **not** also become a hard feasibility constraint; sampling on both sides of the contour remains possible. An output whose optimization checkbox is cleared can still use `above` / `below` as a constraint-only feasibility rule.

The Web UI exposes the acquisition-specific scalar parameter:

- Straddle: `beta` (default 1.96)
- Boundary Variance: `tau` (default 1.0)
- ICU: `bandwidth`; Web value `0` keeps the class default and uses posterior standard deviation as the bandwidth

For multiple modeled outputs, optimized targets have non-negative relative `level_set_weight` values. Constraint-only outputs receive zero acquisition weight. The multi-output LSE implementation normalizes positive weights internally.

With InputPerturbation, mean aggregation is always supported. VaR / CVaR are available for Bayesian optimization and LSE. BO applies risk to the objective through `ObjectiveConfig`; LSE applies it to perturbation-expanded level-set scores through the regression LSE score objective. These paths are wired directly in the Web workflow and do not replace engine or acquisition functions at runtime.

The Web request stores integration-only markers such as `web_family`, `web_level_set_parameter`, `web_risk_type`, and `web_risk_alpha` inside `acquisition.acqf_kwargs` to keep the FastAPI request schema backward compatible. The workflow consumes these markers before constructing the acquisition function.
''',
)

write_file(
    ".github/workflows/web-lse-settings-smoke.yml",
    '''name: Web LSE settings smoke

on:
  pull_request:
    branches:
      - main
    paths:
      - "src/bochan/serving/webapp/level_set_settings.py"
      - "src/bochan/serving/webapp/risk_settings.py"
      - "src/bochan/serving/webapp/search_settings.py"
      - "src/bochan/serving/webapp/target_settings.py"
      - "src/bochan/serving/webapp/workflows.py"
      - "src/bochan/serving/webapp/workflows_tabular.py"
      - "web/src/api.ts"
      - "web/src/components/InputPerturbationRiskSettings.tsx"
      - "web/src/components/TargetProposalSettings.tsx"
      - "web/src/modelArtifactRestore.ts"
      - "web/src/pages/OptimizePage.tsx"
      - "web/src/types.ts"
      - "tests/test_webapp_input_perturbation_risk.py"
      - "tests/test_webapp_level_set_settings.py"
      - "tests/test_webapp_lse_web_contract.py"
      - ".github/workflows/web-lse-settings-smoke.yml"
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install Python dependencies
        run: python -m pip install -e ".[test,api,visualization]"
      - name: Run focused LSE Web tests
        run: |
          pytest -q --maxfail=1 \
            tests/test_webapp_level_set_settings.py \
            tests/test_webapp_lse_web_contract.py \
            tests/test_webapp_input_perturbation_risk.py
      - name: Run Ruff
        run: |
          python -m pip install ruff
          ruff check --ignore I001,SIM102,SIM108 \
            src/bochan/serving/webapp/level_set_settings.py \
            src/bochan/serving/webapp/risk_settings.py \
            src/bochan/serving/webapp/search_settings.py \
            src/bochan/serving/webapp/target_settings.py \
            src/bochan/serving/webapp/workflows.py \
            src/bochan/serving/webapp/workflows_tabular.py \
            tests/test_webapp_input_perturbation_risk.py \
            tests/test_webapp_lse_web_contract.py
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: web/package-lock.json
      - name: Build Web
        working-directory: web
        run: |
          npm ci
          npm run build
''',
)

print("Web LSE codemod applied successfully")
