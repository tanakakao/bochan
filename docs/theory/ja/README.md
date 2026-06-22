# `bochan` 理論リファレンス

このディレクトリは、`bochan`で用いる数理、統計的仮定、逐次意思決定、および実装上の契約を章立てで整理した日本語版リファレンスです。

単なる機能一覧ではなく、次の4層を一貫して理解できる参考書として構成しています。

1. 数学的な問題設定
2. 確率モデルと推論
3. 逐次的な意思決定基準
4. `bochan`およびBoTorch上の具体的な実装

各トピックには主担当章を定め、同じ導出を複数の章で繰り返さない構成としています。モデルや獲得関数を扱う章には、理論とクラス、メソッド、posterior、Tensor shape、ソースパスとの対応を記載しています。

---

## Part I. 基礎理論と逐次意思決定

| 章 | ファイル | 主な内容 |
|---:|---|---|
| 00 | `00_overview.md` | 全体構成、用語、4つの数学空間、表記、読書順、posterior契約 |
| 01 | `01_gaussian_process_models.md` | ガウス過程、カーネル、条件付き分布、周辺尤度、変分推論、数値安定性 |
| 02 | `02_bayesian_optimization.md` | ベイズ最適化、regret、ノイズ、q-batch、pending、look-ahead、評価 |
| 03 | `03_acquisition_functions.md` | PI、EI、LogEI、NEI、UCB、Thompson sampling、KG、獲得関数最適化 |
| 04 | `04_active_learning.md` | Entropy、BALD、IPV、NIPV、jointおよび異分散の能動学習 |
| 05 | `05_level_set_estimation.md` | レベル集合、境界、信頼集合、損失関数、停止条件、評価 |
| 06 | `06_classification_and_ordinal_bo.md` | 分類・順序出力の確率／効用目的、EI・PI・UCB、制約 |
| 07 | `07_multi_objective_and_constraints.md` | Pareto、hypervolume、EHVI、NEHVI、scalarization、chance constraint |
| 08 | `08_input_perturbation_and_risk.md` | 入力摂動、平均／最悪値、VaR、CVaR、chance constraint、`q * n_w` |
| 09 | `09_shape_conventions.md` | Tensor軸、t-batch、q-batch、sample、class、boundary、DeepGP、ensemble |

## Part II. モデル族と実装詳細

| 章 | ファイル | 主な内容 |
|---:|---|---|
| 10 | `10_regression_models_and_likelihoods.md` | Gaussian、Beta、Gamma、Poisson、Negative Binomial、混合入力、多出力回帰 |
| 11 | `11_classification_models.md` | 二値・多クラスGP分類、変分推論、確率化、較正、posterior契約 |
| 12 | `12_ordinal_models.md` | Ordered Logit、cutpoint、識別可能性、quadrature、順序確率、効用 |
| 13 | `13_heteroscedastic_and_robust_models.md` | 既知／学習型ノイズ、異分散、RRP、外れ値、robust model |
| 14 | `14_deep_and_high_dimensional_models.md` | DKL、DeepGP、SAAS、PCA、REMBO、VAE-GP |
| 15 | `15_heterogeneous_multi_output.md` | 異種出力、共有潜在モデル、`OutputSpec`、`HybridPosterior` |
| 16 | `16_level_set_mathematics_and_implementation.md` | 現在の回帰・二値・多クラス・順序LSEクラスが実装する具体式 |

---

## トピックの担当範囲

重複を避けるため、各内容は次の章を主担当とします。

- GPの確率論、条件付き分布、ELBO：01章
- BO獲得関数の数式：03章
- 能動学習の情報量・不確かさ基準：04章
- LSEの問題設定と外部損失：05章
- 分類・順序出力の意思決定目的：06章
- Pareto最適化と制約：07章
- リスク尺度と入力摂動：08章
- Tensor軸と削減順：09章
- 尤度、推論、posterior契約：10〜15章
- 現行LSEクラスの実装式：16章

---

## 推奨読書順

### 標準的なガウス過程ベイズ最適化

1. 00章、01章
2. 10章
3. 02章、03章
4. 09章

### 二値・多クラス分類

1. 00章、01章、11章
2. BOなら06章、能動学習なら04章
3. LSEなら05章、16章
4. 09章

### 順序モデル

1. 00章、01章、12章
2. BOなら06章、能動学習なら04章
3. LSEなら05章、16章
4. 09章

### 異分散・ロバスト実験

1. 01章、10章、13章
2. リスクを扱う場合は08章
3. 能動学習なら04章、境界探索なら16章
4. 09章

### 多目的・異種出力

1. 07章
2. 15章
3. 離散出力の効用変換は06章、12章
4. 09章

### 深層・高次元モデル

1. 01章と対象応答のモデル章
2. 14章
3. 02章、03章
4. 09章

---

## 主要なposterior契約

モデル族によって`posterior()`の意味は同一ではありません。

| モデル族 | 現在の主な契約 |
|---|---|
| Gaussian回帰 | 応答／潜在Gaussian posterior。必要に応じて観測ノイズを含む |
| 二値分類 | `posterior()`は確率空間、`latent_posterior()`は潜在GP |
| 多クラス分類 | `posterior()`はクラス確率、`latent_posterior()`はクラス別潜在GP |
| 順序 | `posterior()`はスカラー潜在GP、`class_probs()`は順序クラス確率 |
| Hybrid | `posterior(..., output_mode=...)`はモード別`HybridPosterior` |

分散として返される量も、潜在のepistemic variance、観測ノイズ込みのpredictive variance、Bernoulli／Categoricalの観測分散、確率関数のposterior variance、クラス効用分散、補助ノイズ、proxy varianceなどで意味が異なります。

また、共通の`[..., q, m]`形状を持つだけでは、出力間共分散が存在するとは限りません。独立ModelList、相関multitask、変換後に積み上げた異種出力を区別してください。

---

## 主要な削減軸

次の軸が同時に存在する場合があります。

```text
posterior samples
model / ensemble batch
BoTorch t-batch
q candidates
input perturbations
outputs / tasks
classes
ordinal boundaries
```

どの順番で削減するかは数式の一部です。09章を共通のshape仕様として参照してください。

---

## 文書化の標準

各章では、可能な限り次の項目を明示します。

1. 問題設定と確率変数
2. 仮定と尤度
3. 導出または中心となる数式
4. 不確かさの意味
5. 等価ではない代替手法
6. Tensor shapeへの影響
7. 実装クラスとソースパス
8. 現在の近似と限界
9. 検証指標
10. 参考文献

ICU、異分散分類、hybrid posteriorなど、一般名と現行実装の意味が一致しない可能性がある場合は、名称だけでなく実際の式とposterior契約を優先して記載します。
