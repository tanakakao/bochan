# Refactoring Specification for BoTorch-Compatible Bayesian Optimization Extension Library

## 0. Purpose

本仕様書は、ベイズ最適化拡張ライブラリ全体の public API・内部設計・命名規則・ファイル構成を統一し、
以下を達成することを目的とする。

- BoTorch / GPyTorch と整合的な API を提供する
- Gaussian regression だけでなく、Beta / Gamma / Count regression, classification, ordinal を統一的に扱う
- mixed input を各モデル族に対する標準バリエーションとして扱う
- robust / high-dimensional / deep 拡張を一貫した規則で整理する
- acquisition function をタスク × 手法ファミリーで整理する
- 将来の配布・保守・自動リファクタリングを容易にする
- 既存コードとの後方互換性を可能な限り維持する

本仕様書では、新規アルゴリズム追加よりも、既存実装の構造整理・命名統一・API 統一を優先する。


## 1. Scope

### 1.1 Included
以下をリファクタリング対象とする。

- regression models
  - gaussian
  - beta
  - gamma
  - count (poisson / negative binomial)
- classification models
- ordinal models
- mixed-input variants
- heteroscedastic variants
- robust relevance pursuit (RRP) variants
- high-dimensional variants
  - PCA
  - REMBO
  - ALEBO
  - SAAS
- deep GP / deep kernel variants
- acquisition functions
- fit helper functions
- utility / builder functions
- docstrings / type hints / usage examples
- tests

### 1.2 Excluded
以下は今回の対象外とする。

- 数理定義そのものの全面変更
- 既存獲得関数のアルゴリズム再設計
- 実験性能改善を主目的とした再実装
- UI / Streamlit 側の仕様変更
- 学習済みモデル保存フォーマットの策定


## 2. Design Principles

### 2.1 BoTorch-first
可能な限り BoTorch の設計思想に合わせる。

- `train_X`, `train_Y`, `train_Yvar`
- `input_transform`
- `outcome_transform`
- `posterior()`
- `construct_inputs()`
- `train_inputs`, `train_targets`
- `ModelList` / multi-output 互換性

を重視する。

### 2.2 Academic organization
モデルはまず **タスク** と **出力分布族** で整理し、その上で
- robust
- high-dimensional
- deep
などの拡張を加える。

### 2.3 Mixed input as a standard variant
mixed input は独立カテゴリではなく、各モデル族に対する標準バリエーションとして扱う。
そのため、通常版と mixed 版は原則として同一ファイルに併記する。

### 2.4 Minimal surprise
同種のモデル間で以下が揃っていることを優先する。

- constructor 引数名
- attribute 名
- posterior の意味
- fit API
- shape convention

### 2.5 Minimal-diff refactor
既存コードを壊しすぎない。既存 public API を変更する場合は、
deprecated alias または wrapper を残す。


## 3. Package Structure Standard

ライブラリ全体の標準構成は以下とする。

```text
models/
    __init__.py
    base.py
    utils.py

    components/
        __init__.py
        mixed_inputs.py
        deep_layers.py
        deep_kernel.py
        posteriors.py
        likelihood_utils.py
        robust_utils.py
        high_dim_utils.py

    regression/
        __init__.py

        gaussian/
            __init__.py
            base.py
            deep.py
            _builders.py
            robust/
                __init__.py
                hetero.py
                rrp.py
            high_dim/
                __init__.py
                pca.py
                rembo.py
                alebo.py
                saas.py

        beta/
            __init__.py
            base.py
            deep.py
            _builders.py
            robust/
                __init__.py
                hetero.py
                rrp.py
            high_dim/
                __init__.py
                pca.py
                rembo.py
                alebo.py
                saas.py

        gamma/
            __init__.py
            base.py
            deep.py
            _builders.py
            robust/
                __init__.py
                hetero.py
                rrp.py
            high_dim/
                __init__.py
                pca.py
                rembo.py
                alebo.py
                saas.py

        count/
            __init__.py
            base.py
            deep.py
            _builders.py
            robust/
                __init__.py
                hetero.py
                rrp.py
            high_dim/
                __init__.py
                pca.py
                rembo.py
                alebo.py
                saas.py

    classification/
        __init__.py
        base.py
        deep.py
        _builders.py
        robust/
            __init__.py
            hetero.py
            rrp.py
        high_dim/
            __init__.py
            pca.py
            rembo.py
            alebo.py
            saas.py

    ordinal/
        __init__.py
        base.py
        deep.py
        _builders.py
        robust/
            __init__.py
            hetero.py
            rrp.py
        high_dim/
            __init__.py
            pca.py
            rembo.py
            alebo.py
            saas.py
````

### 3.1 Structural meaning

* `regression` の下は出力分布族で分ける

  * `gaussian`
  * `beta`
  * `gamma`
  * `count`
* `count` には Poisson / Negative Binomial を含める
* `classification`, `ordinal` は独立タスクとして扱う
* `mixed` は独立ファイルを作らず、各ファイルで通常版と併記する
* builder 関数は `utils.py` に押し込まず、各モデル族の `_builders.py` に置く
* truly common な部品のみ `components/` へ置く

## 4. Model Family Definitions

## 4.1 Regression families

### Gaussian regression

通常の Gaussian process regression。

### Beta regression

目的変数が `(0, 1)` にある割合・比率データ向け。

### Gamma regression

目的変数が正の連続値を取る場合向け。

### Count regression

カウントデータ向け。

* `PoissonGPModel`
* `NegativeBinomialGPModel`

将来的に zero-inflated / hurdle などを追加してもよいが、
当面は count family に含める。

## 4.2 Classification

二値分類および多クラス分類。

## 4.3 Ordinal

順序付きカテゴリを対象とする。

## 5. Common Constructor Standard

すべての public model class は、可能な限り以下の基本形を守る。

```python
class SomeModel:
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        likelihood: Any | None = None,
        covar_module: Any | None = None,
        mean_module: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        ...
```

### 5.1 Rules

* `train_X`, `train_Y`, `train_Yvar` を先頭固定とする
* optional 引数は keyword-only とする
* BoTorch と近い引数名を優先する
* `config` dataclass は public API では原則使わない
* family-specific な引数は明示する

### 5.2 Mixed-input models

mixed 系は以下を追加する。

```python
cat_dims: Sequence[int]
category_counts: Mapping[int, int] | None = None
```

Rules:

* `cat_dims` は public API では必須
* `category_counts` は省略可
* `category_counts is None` の場合、自動推定を試みる
* mixed 版は独立ファイルにせず、通常版と同じモジュールに置く

### 5.3 Ordinal models

ordinal 系は以下を追加する。

```python
num_classes: int
fix_first_cutpoint: bool = True
init_gap: float = 1.0
```

### 5.4 Deep models

deep 系は以下の命名に統一する。

```python
hidden_dims: Sequence[int] | None = None
num_inducing: int = 128
learn_inducing_locations: bool = True
```

`list_hidden_dims` のような別名は deprecated alias とする。

## 6. Class Naming Standard

### 6.1 Regression

通常版 / mixed 版 / deep 版 / deep mixed 版は以下の規則で命名する。

#### Gaussian

* `GaussianGPModel`
* `GaussianMixedGPModel`
* `DeepGaussianGPModel`
* `DeepGaussianMixedGPModel`
* `DeepKernelGaussianGPModel`
* `DeepKernelGaussianMixedGPModel`

#### Beta

* `BetaGPModel`
* `BetaMixedGPModel`
* `DeepBetaGPModel`
* `DeepBetaMixedGPModel`
* `DeepKernelBetaGPModel`
* `DeepKernelBetaMixedGPModel`

#### Gamma

* `GammaGPModel`
* `GammaMixedGPModel`
* `DeepGammaGPModel`
* `DeepGammaMixedGPModel`
* `DeepKernelGammaGPModel`
* `DeepKernelGammaMixedGPModel`

#### Count

* `PoissonGPModel`
* `PoissonMixedGPModel`
* `NegativeBinomialGPModel`
* `NegativeBinomialMixedGPModel`
* `DeepPoissonGPModel`
* `DeepPoissonMixedGPModel`
* `DeepNegativeBinomialGPModel`
* `DeepNegativeBinomialMixedGPModel`

### 6.2 Robust

* `HeteroscedasticGaussianGPModel`
* `HeteroscedasticGaussianMixedGPModel`
* `RobustRelevancePursuitGaussianGPModel`
* `RobustRelevancePursuitGaussianMixedGPModel`

Beta / Gamma / Count / Classification / Ordinal でも同様の規則に従う。

### 6.3 High-dimensional

* `PCAGaussianGPModel`
* `PCAGaussianMixedGPModel`
* `REMBOGaussianGPModel`
* `REMBOGaussianMixedGPModel`
* `ALEBOGaussianGPModel`
* `ALEBOGaussianMixedGPModel`
* `SaasGaussianGPModel`
* `SaasGaussianMixedGPModel`

他 family でも同様の規則に従う。

### 6.4 Classification / Ordinal

同じ命名方針を適用する。

例:

* `ClassifierGP`

* `ClassifierMixedGP`

* `DeepClassifierGP`

* `DeepClassifierMixedGP`

* `OrdinalGPModel`

* `OrdinalMixedGPModel`

* `DeepOrdinalGPModel`

* `DeepOrdinalMixedGPModel`

## 7. File-Level Organization Rules

### 7.1 Mixed variants

`mixed.py` は作成しない。
通常版と mixed 版は同一ファイルに置く。

例:

* `gamma/base.py`

  * `GammaGPModel`
  * `GammaMixedGPModel`

* `gamma/deep.py`

  * `DeepGammaGPModel`
  * `DeepGammaMixedGPModel`
  * `DeepKernelGammaGPModel`
  * `DeepKernelGammaMixedGPModel`

### 7.2 Robust variants

robust は family 内でサブディレクトリ分割する。

* `robust/hetero.py`
* `robust/rrp.py`

### 7.3 High-dimensional variants

high-dimensional は family 内で手法別に分割する。

* `high_dim/pca.py`
* `high_dim/rembo.py`
* `high_dim/alebo.py`
* `high_dim/saas.py`

### 7.4 Builder functions

`_build_default_mixed_covar_module` のような builder は、
`utils.py` ではなく、各 family の `_builders.py` に置く。

例:

* `gaussian/_builders.py`
* `gamma/_builders.py`
* `count/_builders.py`

truly common な builder のみ `components/` に昇格してよい。

## 8. Posterior Standard

すべての public model は `posterior()` を実装し、可能な限り BoTorch の signature に近づける。

```python
posterior(
    X: Tensor,
    output_indices: list[int] | None = None,
    observation_noise: bool | Tensor = False,
    posterior_transform: Any | None = None,
)
```

### 8.1 Regression

* `posterior.mean` は予測平均
* `posterior.variance` は予測分散

### 8.2 Classification

* `posterior.mean` は確率スケールを基本とする
* latent 値が必要な場合は別メソッドを提供する

推奨:

* `latent_posterior()`
* `predict_proba()`
* `predict_label()`

### 8.3 Ordinal

* `posterior.mean` の意味は project 全体で統一する
* 推奨は expected score / expected utility のどちらかに固定
* カテゴリ確率は別メソッドで提供する

推奨:

* `predict_proba()`
* `predict_category()`
* `expected_score()`

## 9. Data and Shape Convention

### 9.1 Input

* `train_X`: `[n, d]`
* candidate `X`: `[..., q, d]`

### 9.2 Regression targets

* `train_Y`: `[n, 1]` を標準
* `[n]` が与えられた場合は内部で `[n, 1]` に正規化可

### 9.3 Classification targets

* binary / multiclass ともに project 内で一貫した shape に統一する

### 9.4 Ordinal targets

* class index は整数
* 値域は `[0, num_classes - 1]`
* shape ルールは全 ordinal model で統一する

### 9.5 Early shape validation

shape mismatch はできるだけ早い段階で明示的エラーを出す。

## 10. Transform Handling

## 10.1 Input transform

* `input_transform` は constructor に直接渡す
* raw training input は原則として以下に保存する

```python
self.train_X_original
```

### 10.1.1 Rules

* `train_inputs[0]` が transform 後でも許容
* raw 可視化・デバッグのため `train_X_original` を持つ
* `train_inputs_raw` などの別名は deprecated にする

## 10.2 Outcome transform

* regression family は BoTorch 互換を優先
* classification / ordinal / non-Gaussian regression では、未対応なら明示的に禁止する
* 対応する場合は「何に transform がかかるか」を docstring に明記する

## 11. Mixed Input Policy

### 11.1 cat_dims

* raw 入力空間での categorical column index
* 0-based
* 昇順で保持

### 11.2 category_counts

* key は raw 入力空間での categorical column index
* value はカテゴリ数

### 11.3 Auto inference

`category_counts is None` の場合、自動推定を試みる。

条件:

* 対象列が整数相当
* 最小値が 0
* 最大値が `K-1`

失敗時は明示的エラーを出す。

```python
ValueError(
    "Failed to infer category_counts from train_X. "
    "Please pass category_counts explicitly."
)
```

### 11.4 Common helpers

category handling の共通ロジックは `components/mixed_inputs.py` に置いてよい。

例:

* `infer_category_counts`
* `validate_cat_dims`
* `split_numeric_and_categorical`

## 12. Acquisition Function Structure Standard

獲得関数は **各モデル族ごと** に整理し、その中を **目的別** に以下の 3 分類で構成する。

- `bayesian_optimization`
- `active_learning`
- `level_set_estimation`

ここでの意味は次の通り。

- **bayesian_optimization**
  - 目的関数の良い点を探すことを主目的とする
  - 最大化 / 最小化 / target 追従 / expected utility 最大化を含む
- **active_learning**
  - 情報獲得、モデル改善、不確実性低減を主目的とする
- **level_set_estimation**
  - 閾値境界、decision boundary、level set、contour の推定を主目的とする

### 12.1 Directory structure

```text
acquisition/
    __init__.py
    base.py
    penalties.py
    utils.py

    regression/
        __init__.py

        gaussian/
            __init__.py
            bayesian_optimization.py
            active_learning.py
            level_set_estimation.py

        beta/
            __init__.py
            bayesian_optimization.py
            active_learning.py
            level_set_estimation.py

        gamma/
            __init__.py
            bayesian_optimization.py
            active_learning.py
            level_set_estimation.py

        count/
            __init__.py
            bayesian_optimization.py
            active_learning.py
            level_set_estimation.py

    classification/
        __init__.py
        bayesian_optimization.py
        active_learning.py
        level_set_estimation.py

    ordinal/
        __init__.py
        bayesian_optimization.py
        active_learning.py
        level_set_estimation.py
````

### 12.2 Structural meaning

* 獲得関数はまず **モデル族** に対応して整理する
* その中で、**何を目的に点を選ぶか** に応じて

  * `bayesian_optimization.py`
  * `active_learning.py`
  * `level_set_estimation.py`
    に分ける
* `single_output.py` や `multi_output.py` を package の主分類にはしない
* single-objective / q-batch / multi-objective は、同じ目的カテゴリ内で近くに配置する

### 12.3 In-file ordering

各 acquisition file では、原則として以下の順で class / function を並べる。

1. single-objective
2. q-batch single-objective
3. multi-output / multi-objective
4. q-batch multi-output / multi-objective
5. private helper functions

これにより、同じ目的・同じ理論に基づく single / q / multi 系を近くに保つ。

### 12.4 Naming policy

`normal.py` や `standard.py` は使わず、通常の最適化目的は必ず
`bayesian_optimization.py`
とする。

理由:

* `active_learning`
* `level_set_estimation`
  と並べたときに、目的が最も明確になるため。

---

## 13. Acquisition Family Definitions

## 13.1 Bayesian optimization

`bayesian_optimization.py` には、**良い目的値を持つ点を選ぶ** ための獲得関数を置く。

対象:

* maximization
* minimization
* target-seeking
* expected utility maximization
* scalarized optimization
* hypervolume / Pareto-based optimization
* knowledge-gradient / entropy search のうち BO 目的で使うもの

### Representative examples

#### Regression

* Expected Improvement
* Log Expected Improvement
* Probability of Improvement
* Upper Confidence Bound
* Knowledge Gradient
* Max-value entropy / predictive entropy / joint entropy search 系
* scalarized BO
* multi-objective BO

#### Classification

* positive class probability maximization
* class utility maximization
* target class probability optimization

#### Ordinal

* expected utility maximization
* expected score maximization
* expected utility improvement
* utility-based UCB
* multi-objective ordinal utility optimization

---

## 13.2 Active learning

`active_learning.py` には、**情報獲得・不確実性低減・モデル改善** を目的とする獲得関数を置く。

対象:

* entropy-based methods
* uncertainty sampling
* BALD
* variance reduction
* integrated posterior uncertainty reduction
* information gain 型 active learning
* NIPV 系

### Representative examples

#### Regression

* qNegIntegratedPosteriorVariance
* posterior variance sampling
* integrated variance reduction
* pure uncertainty reduction

#### Classification

* predictive entropy
* uncertainty sampling
* BALD
* qBALD
* joint BALD
* greedy joint BALD

#### Ordinal

* ordinal predictive entropy
* ordinal uncertainty reduction
* ordinal information gain

---

## 13.3 Level-set estimation

`level_set_estimation.py` には、**閾値境界・decision boundary・level set・contour** の推定を目的とする獲得関数を置く。

対象:

* probability of exceedance
* level set uncertainty
* contour uncertainty
* boundary uncertainty
* straddle methods
* threshold crossing / feasible region estimation
* boundary-focused multi-objective methods

### Important rule

`straddle` 系は原則として `level_set_estimation.py` に置く。

理由:

* straddle は本質的に threshold / boundary 近傍を狙うため
* level-set / contour / decision boundary estimation と理論的に近いため

### Representative examples

#### Regression

* qStraddle
* LogDetqStraddle
* qICUAcquisition
* qJointBoundaryVariance
* qProbabilityOfExceedance
* qLevelSetUncertainty

#### Classification

* StraddleClassifierAcquisition
* LatentStraddleClassifierAcquisition
* JointLatentStraddleClassifierAcquisition
* threshold exceedance methods
* boundary-focused classification acquisitions

#### Ordinal

* OrdinalExpectedUtilityProbabilityOfExceedance
* OrdinalExpectedUtilityLevelSetUncertainty
* OrdinalExpectedUtilityStraddle
* qOrdinalExpectedUtilityProbabilityOfExceedance
* qOrdinalExpectedUtilityLevelSetUncertainty
* qOrdinalExpectedUtilityStraddle
* MultiObjectiveOrdinalLevelSetProbabilityOfExceedance
* MultiObjectiveOrdinalLevelSetUncertainty
* MultiObjectiveOrdinalStraddle
* qMultiObjectiveOrdinalLevelSetProbabilityOfExceedance
* qMultiObjectiveOrdinalLevelSetUncertainty
* qMultiObjectiveOrdinalStraddle

---

## 14. Multi-output / Multi-objective Policy

### 14.1 Do not separate by top-level files

`multi_output.py` や `multi_objective.py` を package の主分類としては使わない。

理由:

* multi-output / multi-objective は、各理論 family の拡張として現れるため
* single と multi を別ファイルへ分けると、同じ理論の実装が離れて管理しづらくなるため

### 14.2 Placement rule

multi-output / multi-objective acquisition は、対応する目的ファイルの中に置く。

例:

* ordinal の expected-utility straddle 系の multi-objective 版

  * `ordinal/level_set_estimation.py`
* classification の BALD 系 multi-output 版

  * `classification/active_learning.py`
* regression の multi-objective BO

  * `regression/*/bayesian_optimization.py`

### 14.3 Naming examples

* `OrdinalExpectedUtilityStraddle`

* `qOrdinalExpectedUtilityStraddle`

* `MultiObjectiveOrdinalStraddle`

* `qMultiObjectiveOrdinalStraddle`

* `ExpectedUtilityImprovement`

* `qExpectedUtilityImprovement`

* `MultiObjectiveExpectedUtilityImprovement`

* `qMultiObjectiveExpectedUtilityImprovement`

single / q / multi-objective の区別は class 名で表現する。

---

## 15. Penalty and Utility Helper Policy

### 15.1 Penalties

pending penalty / observed penalty / repulsion penalty は `penalties.py` に共通化する。

例:

* `_apply_pending_penalty`
* `_apply_observed_penalty`
* `_apply_repulsion_penalty`
* `_pairwise_distance_penalty`

### 15.2 Utility helpers

entropy, logdet, shape validation, score reduction などの小関数は `utils.py` に置く。

例:

* entropy calculators
* logdet helpers
* shape validation helpers
* posterior mean / variance extraction helpers

### 15.3 Base classes

共通 base / mixin は `base.py` に置く。

例:

* penalty-aware mixin
* pending-aware acquisition base
* common Monte Carlo acquisition base

---

## 16. Representative Placement Examples

### 16.1 Regression / Gaussian

#### `regression/gaussian/bayesian_optimization.py`

* `qExpectedImprovement`
* `qLogExpectedImprovement`
* `qProbabilityOfImprovement`
* `qUpperConfidenceBound`
* `qKnowledgeGradient`
* multi-objective BO 系

#### `regression/gaussian/active_learning.py`

* `qNegIntegratedPosteriorVariance`
* variance reduction 系
* posterior uncertainty reduction 系

#### `regression/gaussian/level_set_estimation.py`

* `qStraddle`
* `LogDetqStraddle`
* `qICUAcquisition`
* `qJointBoundaryVariance`
* `qProbabilityOfExceedance`
* `qLevelSetUncertainty`

### 16.2 Classification

#### `classification/bayesian_optimization.py`

* class probability maximization
* class utility maximization
* target class optimization

#### `classification/active_learning.py`

* `EntropyClassifierAcquisition`
* `UncertaintySamplingClassifierAcquisition`
* `BALDAcquisition`
* `JointQBALDAcquisitionBinary`
* `GreedyJointQBALDAcquisitionBinary`

#### `classification/level_set_estimation.py`

* `StraddleClassifierAcquisition`
* `LatentStraddleClassifierAcquisition`
* `JointLatentStraddleClassifierAcquisition`
* threshold exceedance methods

### 16.3 Ordinal

#### `ordinal/bayesian_optimization.py`

* ordinal expected utility maximization
* ordinal expected score maximization
* ordinal expected utility improvement
* ordinal utility UCB
* multi-objective ordinal utility optimization

#### `ordinal/active_learning.py`

* `qOrdinalPredictiveEntropy`
* ordinal uncertainty reduction
* ordinal information gain 系

#### `ordinal/level_set_estimation.py`

* `OrdinalExpectedUtilityProbabilityOfExceedance`
* `OrdinalExpectedUtilityLevelSetUncertainty`
* `OrdinalExpectedUtilityStraddle`
* `qOrdinalExpectedUtilityProbabilityOfExceedance`
* `qOrdinalExpectedUtilityLevelSetUncertainty`
* `qOrdinalExpectedUtilityStraddle`
* `MultiObjectiveOrdinalLevelSetProbabilityOfExceedance`
* `MultiObjectiveOrdinalLevelSetUncertainty`
* `MultiObjectiveOrdinalStraddle`
* `qMultiObjectiveOrdinalLevelSetProbabilityOfExceedance`
* `qMultiObjectiveOrdinalLevelSetUncertainty`
* `qMultiObjectiveOrdinalStraddle`

---

## 17. Codex Refactor Instructions for Acquisition

Codex は acquisition の整理において、以下の規則を守ること。

1. まず各既存 acquisition を

   * `bayesian_optimization`
   * `active_learning`
   * `level_set_estimation`
     のいずれかへ分類する
2. 次に、各モデル族 / タスク配下へ再配置する
3. `single_output.py` / `multi_output.py` を主分類として新設しない
4. single / q / multi-objective は同一目的ファイル内で近くに置く
5. `straddle` は原則 `level_set_estimation.py` へ置く
6. penalty / utility helper は `penalties.py` / `utils.py` へ共通化する
7. class 名は可能な限り維持し、配置のみ整理する
8. `__init__.py` で浅い import path を維持する

### Forbidden

* single / multi の分離だけを目的とした過剰なファイル分割
* straddle の恣意的な移動
* multi-objective をトップレベル独立 package に移すこと
* class 名の無断変更

---

## 18. Acceptance Criteria for Acquisition Refactor

以下を満たしたら acquisition 整理は完了とする。

* acquisition が各モデル族 / タスク配下に整理されている
* 各配下に

  * `bayesian_optimization.py`
  * `active_learning.py`
  * `level_set_estimation.py`
    がある
* single / q / multi-objective が同一目的ファイル内で整理されている
* `straddle` 系が `level_set_estimation.py` に配置されている
* penalty と helper が共通モジュールへ整理されている
* `__init__.py` による浅い import path が維持されている


## 19. Codex Execution Instructions

Codex は以下の順で作業すること。

1. current public API の一覧を作る
2. package structure を標準構成へ合わせる
3. constructor signature を統一する
4. mixed variants を通常版と同一ファイルへまとめる
5. builder functions を `_builders.py` へ整理する
6. acquisition を task × method family で再配置する
7. deprecated alias を追加する
8. docstring / type hints / examples を整備する
9. smoke tests を追加する
10. migration note を出す

### Forbidden

* 数理ロジックを独断で変更しない
* posterior semantics を勝手に変えない
* mixed / non-mixed の public class 名を無断変更しない
* README / examples を削除しない

### Preferred

* 小さい patch 単位で進める
* representative modules でパターン確定後に横展開する

## 20. Acceptance Criteria

以下を満たしたら完了とする。

* package structure が本仕様と整合している
* regression families に gaussian / beta / gamma / count がある
* count family に Poisson / Negative Binomial が整理されている
* mixed variants が各 family ファイルに統合されている
* robust が `hetero.py` / `rrp.py` に分かれている
* high-dimensional が `pca.py` / `rembo.py` / `alebo.py` / `saas.py` に分かれている
* builder functions が `_builders.py` に整理されている
* acquisition が task × method family で整理されている
* docstring / examples / tests が最低限整備されている
* backward compatibility が説明されている
