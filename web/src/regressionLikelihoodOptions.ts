import type { ModelFamily } from "./modelOptions";

export type RegressionLikelihood =
  | "gaussian"
  | "gamma"
  | "beta"
  | "poisson"
  | "negative_binomial";

export type RegressionModelVariant =
  | "base"
  | "deepgp"
  | "deepkernel"
  | "saas"
  | "pca"
  | "rembo"
  | "rrp"
  | "hetero"
  | "random_forest"
  | "lightgbm_ensemble"
  | "ngboost_ensemble"
  | "multitask";

export const REGRESSION_LIKELIHOOD_OPTIONS: ReadonlyArray<{
  value: RegressionLikelihood;
  label: string;
  description: string;
}> = [
  {
    value: "gaussian",
    label: "Gaussian",
    description: "実数値を通常のガウス観測モデルで扱います。"
  },
  {
    value: "gamma",
    label: "Gamma",
    description: "0より大きい連続値をGamma分布で扱います。"
  },
  {
    value: "beta",
    label: "Beta",
    description: "0より大きく1未満の割合・比率をBeta分布で扱います。"
  },
  {
    value: "poisson",
    label: "Poisson",
    description: "0以上の整数カウントをPoisson分布で扱います。"
  },
  {
    value: "negative_binomial",
    label: "Negative Binomial",
    description: "分散が平均より大きい0以上の整数カウントを扱います。"
  }
];

const MODEL_VARIANT_LABELS: Record<RegressionModelVariant, string> = {
  base: "Base GP",
  deepgp: "Deep GP",
  deepkernel: "Deep Kernel",
  saas: "SAAS",
  pca: "PCA",
  rembo: "REMBO",
  rrp: "Robust (RRP)",
  hetero: "Heteroskedastic",
  random_forest: "Random Forest",
  lightgbm_ensemble: "LightGBM",
  ngboost_ensemble: "NGBoost",
  multitask: "Multitask GP"
};

/** Resolve the response likelihood encoded by an existing public model key. */
export function regressionLikelihoodFor(modelType: string): RegressionLikelihood {
  if (modelType.startsWith("negative_binomial_")) return "negative_binomial";
  if (modelType.startsWith("poisson_")) return "poisson";
  if (modelType.startsWith("gamma_")) return "gamma";
  if (modelType.startsWith("beta_")) return "beta";
  return "gaussian";
}

/** Resolve the architecture variant while ignoring the response likelihood prefix. */
export function regressionModelVariantFor(modelType: string): RegressionModelVariant {
  let variant = modelType;
  for (const prefix of ["negative_binomial_", "poisson_", "gamma_", "beta_"]) {
    if (variant.startsWith(prefix)) {
      variant = variant.slice(prefix.length);
      break;
    }
  }
  if (variant === "robust") return "rrp";
  if (
    variant === "base" ||
    variant === "deepgp" ||
    variant === "deepkernel" ||
    variant === "saas" ||
    variant === "pca" ||
    variant === "rembo" ||
    variant === "rrp" ||
    variant === "hetero" ||
    variant === "random_forest" ||
    variant === "lightgbm_ensemble" ||
    variant === "ngboost_ensemble" ||
    variant === "multitask"
  ) {
    return variant;
  }
  return "base";
}

/** Return a distribution-independent label for one model architecture. */
export function regressionModelVariantLabel(modelType: string): string {
  return MODEL_VARIANT_LABELS[regressionModelVariantFor(modelType)];
}

/** Pick the closest model after changing the response likelihood or family. */
export function selectRegressionModelType(
  options: ReadonlyArray<{ value: string; family: ModelFamily }>,
  likelihood: RegressionLikelihood,
  preferredVariant: RegressionModelVariant,
  preferredFamily: ModelFamily
): string | null {
  const likelihoodOptions = options.filter(
    (option) => regressionLikelihoodFor(option.value) === likelihood
  );
  return (
    likelihoodOptions.find(
      (option) =>
        option.family === preferredFamily &&
        regressionModelVariantFor(option.value) === preferredVariant
    )?.value ??
    likelihoodOptions.find((option) => option.family === preferredFamily)?.value ??
    likelihoodOptions.find(
      (option) => regressionModelVariantFor(option.value) === preferredVariant
    )?.value ??
    likelihoodOptions[0]?.value ??
    null
  );
}
