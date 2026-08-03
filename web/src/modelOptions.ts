export type ModelFamily =
  | "standard_gp"
  | "deep_representation"
  | "high_dimensional"
  | "robust_noise"
  | "multitask";

export const MODEL_FAMILY_OPTIONS: Array<{ value: ModelFamily; label: string }> = [
  { value: "standard_gp", label: "標準ガウス過程" },
  { value: "deep_representation", label: "深層・表現学習" },
  { value: "high_dimensional", label: "高次元・次元削減" },
  { value: "robust_noise", label: "ノイズ・頑健" },
  { value: "multitask", label: "マルチタスク" }
];

export const MODEL_OPTIONS = [
  { value: "base", label: "Base GP", family: "standard_gp" },
  { value: "deepgp", label: "Deep GP", family: "deep_representation" },
  { value: "deepkernel", label: "Deep Kernel", family: "deep_representation" },
  { value: "saas", label: "SAAS", family: "high_dimensional" },
  { value: "pca", label: "PCA", family: "high_dimensional" },
  { value: "rembo", label: "REMBO", family: "high_dimensional" },
  { value: "robust", label: "Robust (RRP)", family: "robust_noise" },
  { value: "hetero", label: "Heteroskedastic", family: "robust_noise" },
  { value: "multitask", label: "Multitask GP", family: "multitask" },

  { value: "gamma_base", label: "Gamma Base", family: "standard_gp" },
  { value: "gamma_deepgp", label: "Gamma Deep GP", family: "deep_representation" },
  { value: "gamma_deepkernel", label: "Gamma Deep Kernel", family: "deep_representation" },
  { value: "gamma_saas", label: "Gamma SAAS", family: "high_dimensional" },
  { value: "gamma_pca", label: "Gamma PCA", family: "high_dimensional" },
  { value: "gamma_rembo", label: "Gamma REMBO", family: "high_dimensional" },
  { value: "gamma_rrp", label: "Gamma RRP", family: "robust_noise" },
  { value: "gamma_hetero", label: "Gamma Heteroskedastic", family: "robust_noise" },
  { value: "gamma_multitask", label: "Gamma Multitask", family: "multitask" },

  { value: "beta_base", label: "Beta Base", family: "standard_gp" },
  { value: "beta_deepgp", label: "Beta Deep GP", family: "deep_representation" },
  { value: "beta_deepkernel", label: "Beta Deep Kernel", family: "deep_representation" },
  { value: "beta_saas", label: "Beta SAAS", family: "high_dimensional" },
  { value: "beta_pca", label: "Beta PCA", family: "high_dimensional" },
  { value: "beta_rembo", label: "Beta REMBO", family: "high_dimensional" },
  { value: "beta_rrp", label: "Beta RRP", family: "robust_noise" },
  { value: "beta_hetero", label: "Beta Heteroskedastic", family: "robust_noise" },
  { value: "beta_multitask", label: "Beta Multitask", family: "multitask" },

  { value: "poisson_base", label: "Poisson Base", family: "standard_gp" },
  { value: "poisson_deepgp", label: "Poisson Deep GP", family: "deep_representation" },
  { value: "poisson_deepkernel", label: "Poisson Deep Kernel", family: "deep_representation" },
  { value: "poisson_saas", label: "Poisson SAAS", family: "high_dimensional" },
  { value: "poisson_pca", label: "Poisson PCA", family: "high_dimensional" },
  { value: "poisson_rembo", label: "Poisson REMBO", family: "high_dimensional" },
  { value: "poisson_rrp", label: "Poisson RRP", family: "robust_noise" },
  { value: "poisson_hetero", label: "Poisson Heteroskedastic", family: "robust_noise" },
  { value: "poisson_multitask", label: "Poisson Multitask", family: "multitask" },

  { value: "negative_binomial_base", label: "Negative Binomial Base", family: "standard_gp" },
  { value: "negative_binomial_deepgp", label: "Negative Binomial Deep GP", family: "deep_representation" },
  { value: "negative_binomial_deepkernel", label: "Negative Binomial Deep Kernel", family: "deep_representation" },
  { value: "negative_binomial_saas", label: "Negative Binomial SAAS", family: "high_dimensional" },
  { value: "negative_binomial_pca", label: "Negative Binomial PCA", family: "high_dimensional" },
  { value: "negative_binomial_rembo", label: "Negative Binomial REMBO", family: "high_dimensional" },
  { value: "negative_binomial_rrp", label: "Negative Binomial RRP", family: "robust_noise" },
  { value: "negative_binomial_hetero", label: "Negative Binomial Heteroskedastic", family: "robust_noise" },
  { value: "negative_binomial_multitask", label: "Negative Binomial Multitask", family: "multitask" }
] as const satisfies ReadonlyArray<{
  value: string;
  label: string;
  family: ModelFamily;
}>;

export type WebModelType = (typeof MODEL_OPTIONS)[number]["value"];

export const MODEL_DESCRIPTIONS: Record<WebModelType, string> = {
  base: "標準的なガウス過程モデルです。",
  deepgp: "複数層のガウス過程で非線形な表現を学習します。",
  deepkernel: "ニューラルネットワークで特徴表現を学習し、ガウス過程へ接続します。",
  saas: "高次元入力のうち重要な少数次元を疎に選択します。",
  pca: "指定次元へPCA射影してモデル化します。",
  rembo: "指定次元の低次元空間から探索します。",
  robust: "内部ではRRPモデルを使用し、外れ値や頑健性を考慮します。",
  hetero: "入力位置によって異なる観測ノイズをモデル化します。",
  multitask: "回帰目的間の相関を学習して情報共有します。",

  gamma_base: "正値目的変数のGamma変分GPです。",
  gamma_deepgp: "Gamma尤度を用いるDeep GPです。",
  gamma_deepkernel: "学習特徴上のGamma変分GPです。",
  gamma_saas: "高次元Gamma回帰向けSAASモデルです。",
  gamma_pca: "raw入力をPCA射影するGamma回帰です。",
  gamma_rembo: "raw入力をREMBO射影するGamma回帰です。",
  gamma_rrp: "外れ値に頑健なGamma回帰です。",
  gamma_hetero: "入力依存分散を扱うGamma回帰です。",
  gamma_multitask: "正値の複数目的間の相関を学習する変分Gamma GPです。",

  beta_base: "0より大きく1未満の連続割合をBeta分布でモデル化します。",
  beta_deepgp: "Beta尤度を用いるDeep GP割合回帰です。",
  beta_deepkernel: "学習特徴上でBeta割合回帰を行います。",
  beta_saas: "高次元の割合目的変数向けBeta SAASモデルです。",
  beta_pca: "raw入力をPCA射影してBeta回帰を行います。",
  beta_rembo: "低次元REMBO空間でBeta回帰を行います。",
  beta_rrp: "外れ観測に頑健なBeta割合回帰です。",
  beta_hetero: "入力依存の追加分散を持つBeta割合回帰です。",
  beta_multitask: "0より大きく1未満の複数目的間の相関を学習するBeta GPです。",

  poisson_base: "非負整数の発生回数をPoisson分布でモデル化します。",
  poisson_deepgp: "Poisson尤度を用いるDeep GPカウント回帰です。",
  poisson_deepkernel: "学習特徴上でPoissonカウント回帰を行います。",
  poisson_saas: "高次元カウント目的変数向けPoisson SAASモデルです。",
  poisson_pca: "raw入力をPCA射影してPoisson回帰を行います。",
  poisson_rembo: "低次元REMBO空間でPoisson回帰を行います。",
  poisson_rrp: "外れ観測に頑健なPoisson回帰です。",
  poisson_hetero: "入力依存の追加分散を持つPoisson回帰です。",
  poisson_multitask: "複数の非負整数目的間の相関を学習するPoisson GPです。",

  negative_binomial_base: "過分散を持つ非負整数カウントをNegative Binomial分布でモデル化します。",
  negative_binomial_deepgp: "Negative Binomial尤度を用いるDeep GPカウント回帰です。",
  negative_binomial_deepkernel: "学習特徴上でNegative Binomial回帰を行います。",
  negative_binomial_saas: "高次元カウント目的変数向けNegative Binomial SAASモデルです。",
  negative_binomial_pca: "raw入力をPCA射影してNegative Binomial回帰を行います。",
  negative_binomial_rembo: "低次元REMBO空間でNegative Binomial回帰を行います。",
  negative_binomial_rrp: "外れ観測に頑健なNegative Binomial回帰です。",
  negative_binomial_hetero: "入力依存の追加分散を持つNegative Binomial回帰です。",
  negative_binomial_multitask: "複数の過分散カウント目的間の相関を学習するNegative Binomial GPです。"
};

export function modelFamilyFor(modelType: string): ModelFamily {
  return MODEL_OPTIONS.find((option) => option.value === modelType)?.family ?? "standard_gp";
}

export function isMultitaskModelType(modelType: string): boolean {
  return modelFamilyFor(modelType) === "multitask";
}

export function isProjectedModelType(modelType: string): boolean {
  return modelType === "pca" || modelType === "rembo" ||
    modelType.endsWith("_pca") || modelType.endsWith("_rembo");
}

export function isNonGaussianModelType(modelType: string): boolean {
  return modelType.startsWith("beta_") ||
    modelType.startsWith("gamma_") ||
    modelType.startsWith("poisson_") ||
    modelType.startsWith("negative_binomial_");
}
