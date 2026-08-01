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
  { value: "gamma_multitask", label: "Gamma Multitask", family: "multitask" }
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
  gamma_multitask: "正値の複数目的間の相関を学習する変分Gamma GPです。"
};

export function modelFamilyFor(modelType: string): ModelFamily {
  return MODEL_OPTIONS.find((option) => option.value === modelType)?.family ?? "standard_gp";
}
