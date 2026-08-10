# AGENTS.md

## Scope

この文書は `bochan` の model 実装と model family の整理規則を定義する。
モデルの追加・変更・レビューでは、特別な理由がない限り以下を優先する。

## 1. Design Principles

### 1.1 Public models own their behavior

public model class の挙動は、その class または明示的な基底 class / mixin に実装する。
import 時に別 module から class method を差し替える monkey patch は使わない。

BoTorch / GPyTorch の extension point として公式に提供される registration API は、
class method の差し替えとは区別して使用してよい。

### 1.2 One canonical public API

同じ意味の class 名・constructor 引数・method 名を複数公開しない。
古い名称を alias として残す compatibility layer は作らず、call site・registry・test・docs を
canonical 名へ同時に移行する。

### 1.3 Same family, same contract

同じ model family では、可能な限り constructor、属性、posterior、fit、shape convention を揃える。
分類・順序回帰・非 Gaussian likelihood など task / family 固有の意味を持つ引数は無理に共通化しない。

## 2. Package Organization

モデルは原則として task と distribution family で分ける。

```text
models/
    components/
    regression/
        gaussian/
            deep/
            robust/
            high_dim/
        non_gaussian/
            beta/
            gamma/
            poisson/
            negative_binomial/
    classification/
        binary/
        multiclass/
    ordinal/
    hybrid/
```

### 2.1 Structural meaning

* regression は出力分布族で分ける
* classification と ordinal は独立 task として扱う
* mixed variant は可能なら通常版と同じ module に置く
* truly common な実装のみ `components/` に置く
* runtime patch 用 module は作らない

## 3. Common Constructor Standard

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

Rules:

* `train_X`, `train_Y`, `train_Yvar` を先頭に置く
* shared optional 引数は keyword-only を基本とする
* BoTorch / GPyTorch と近い引数名を優先する
* 同義の deprecated alias は定義しない
* family-specific / task-specific 引数は明示する
* `config` dataclass は public constructor の必須経路にしない

### 3.1 Mixed-input models

mixed 系は必要に応じて以下を追加する。

```python
cat_dims: Sequence[int]
category_counts: Mapping[int, int] | None = None
```

* `cat_dims` は public API では明示する
* `category_counts` は省略可能な場合に自動推定する
* input transform は categorical column の意味を壊してはいけない

### 3.2 Ordinal models

ordinal 系は必要に応じて以下を追加する。

```python
num_classes: int | None = None
fix_first_cutpoint: bool = True
init_gap: float = 1.0
```

`num_classes=None` を許す model は、canonicalized training target から class 数を推定する。

### 3.3 Deep models

deep 系の共通引数名は以下を canonical とする。

```python
hidden_dims: Sequence[int] | None = None
num_inducing: int = 128
learn_inducing_locations: bool = True
```

`list_hidden_dims`、`inducing_points_num` などの同義 alias は公開しない。

## 4. Class Naming Standard

命名は「variant / method + task or distribution + GPModel」を基本とし、同じ概念の語順を family 間で揃える。

### 4.1 Gaussian regression

* `GaussianGPModel`
* `GaussianMixedGPModel`
* `DeepGaussianGPModel`
* `DeepGaussianMixedGPModel`
* `DeepKernelGaussianGPModel`
* `DeepKernelGaussianMixedGPModel`
* `DeepKernelDeepGaussianGPModel`
* `DeepKernelDeepGaussianMixedGPModel`
* `GaussianKroneckerMultiTaskGP`
* `GaussianMixedKroneckerMultiTaskGP`

### 4.2 Non-Gaussian regression

例:

* `BetaGPModel`
* `DeepBetaGPModel`
* `DeepBetaMixedGPModel`
* `GammaGPModel`
* `DeepGammaGPModel`
* `DeepGammaMixedGPModel`
* `PoissonGPModel`
* `DeepPoissonGPModel`
* `DeepPoissonMixedGPModel`
* `NegativeBinomialGPModel`
* `DeepNegativeBinomialGPModel`
* `DeepNegativeBinomialMixedGPModel`

### 4.3 Robust

`Outlier...` / `SafeRobust...` のような歴史的 naming variant は作らず、method 名を揃える。

例:

* `HeteroscedasticGaussianGPModel`
* `HeteroscedasticGaussianMixedGPModel`
* `RobustRelevancePursuitGaussianGPModel`
* `RobustRelevancePursuitGaussianMixedGPModel`
* `RobustRelevancePursuitBinaryClassificationGPModel`
* `RobustRelevancePursuitOrdinalGPModel`

### 4.4 High-dimensional

例:

* `PCAGaussianGPModel`
* `PCAGaussianMixedGPModel`
* `REMBOGaussianGPModel`
* `REMBOGaussianMixedGPModel`
* `SaasGaussianGPModel`
* `SaasGaussianMixedGPModel`

他 family でも同じ語順を使う。

### 4.5 Classification / Ordinal

例:

* `BinaryClassificationGPModel`
* `DeepBinaryClassificationGPModel`
* `DeepBinaryClassificationMixedGPModel`
* `MulticlassClassificationGPModel`
* `DeepMulticlassClassificationGPModel`
* `DeepMulticlassClassificationMixedGPModel`
* `OrdinalGPModel`
* `OrdinalMixedGPModel`
* `DeepOrdinalGPModel`
* `DeepOrdinalMixedGPModel`

## 5. Posterior Standard

すべての public model は可能な限り BoTorch に近い `posterior()` signature を持つ。

```python
posterior(
    X: Tensor,
    output_indices: list[int] | None = None,
    observation_noise: bool | Tensor = False,
    posterior_transform: Any | None = None,
)
```

### 5.1 Regression

* `posterior.mean`: predictive mean
* `posterior.variance`: predictive variance

### 5.2 Classification

* public `posterior.mean` は probability scale を基本とする
* latent distribution は canonical `latent_posterior()` で提供する
* probability / label helper は task 内で一つの naming に揃える
* `posterior_latent()` / `posterior_f()` のような同義 compatibility alias は新設しない

### 5.3 Ordinal

* `posterior.mean` の意味を family 内で統一する
* class probability / expected score / category prediction は役割を分ける
* ordinal 固有の `num_classes`, cutpoint, utility 引数は共通化のために削らない

## 6. Data and Shape Convention

### 6.1 Input

* training `train_X`: `[n, d]` または batch 付き equivalent
* candidate `X`: `[..., q, d]`
* InputPerturbation 等で内部 q-like axis が増える場合も、public raw-space と transformed-space を区別する

### 6.2 Targets

* regression: `[n, 1]` を基本とし、必要なら `[n]` を内部 canonicalize する
* binary / multiclass / ordinal: task 固有の target semantics を保持する

### 6.3 Stored training data

BoTorch / GPyTorch が要求する `train_inputs`, `train_targets` は compatibility alias ではなく framework contract として扱う。
追加の raw / transformed training attributes は明確な役割があるものだけ保持する。

## 7. Fit / MLL Contract

model 固有 MLL が必要な場合は `make_mll()` に実装する。
fit utility が class type ごとの private workaround を増やすより、model 自身が必要な training objective を明示する。

例:

```python
mll = model.make_mll(beta=1.0)
```

`beta` など family-specific な引数は、その family で意味がある場合に残す。

## 8. Refactoring Rules

model contract を変更する PR では以下を同時に行う。

1. class / method 本体へ runtime patch の実装を移す
2. patch installer と import-time side effect を削除する
3. canonical class / argument / method 名へ internal call site を移行する
4. compatibility alias を削除する
5. registry、tests、docs を同じ commit で更新する
6. source-level guard と behavior test の両方で regression を防ぐ

## 9. Review Checklist

* import 時に model class へ method assignment / `setattr` をしていないか
* BoTorch class 自体を書き換えていないか
* deprecated class / arg / method alias が残っていないか
* 同じ family の shared constructor args が同名か
* classification / ordinal 固有引数を無理に共通化していないか
* posterior shape / semantics が既存 acquisition と一致するか
* registry / docs / tests が canonical 名を参照しているか
* `python -m compileall` と relevant tests が通るか
