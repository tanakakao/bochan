# 03b. 情報理論・look-ahead型の獲得関数

本章では `bochan` の高位APIから利用できる MES、JES、HVKG を整理します。これらは EI / UCB のように候補点そのものの改善量だけを見るのではなく、最適値・最適点・将来のPareto frontに関する情報価値を評価します。

## 1. Max-value Entropy Search (MES)

MESは、未知の最大値 `f*` に関して候補点の観測がどれだけ情報を与えるかを評価します。

```math
\alpha_{MES}(x)=I(y_x; f^*\mid \mathcal D)
```

`bochan` では次のaliasを使います。

```python
AcquisitionConfig(name="mes")
AcquisitionConfig(name="qmes")
```

これらはBoTorchの `qMaxValueEntropy` に解決されます。

### 自動設定

`candidate_set` を明示しなければ、`DataContext.bounds` からBoTorchのinput constructorを使って候補集合を自動生成します。候補数は次で変更できます。

```python
DataContext(
    bounds=bounds,
    extra={"mes_candidate_size": 2000},
)
```

既定値は1000です。

`q > 1` ではMESは逐次的なbatch構築が必要になるため、高位APIは `OptimizeConfig.sequential=True` を自動設定します。

multi-output modelへMESを適用する場合は、何を最大化するscalar posteriorとするかを曖昧にしないため、明示的な `posterior_transform` が必要です。

## 2. Joint Entropy Search (JES)

JESは最適点と最適値の組 `(x*, f*)` に対する情報量を評価します。

```math
\alpha_{JES}(x)=I(y_x;(x^*,f^*)\mid\mathcal D)
```

```python
AcquisitionConfig(name="jes")
AcquisitionConfig(name="qjes")
```

はBoTorchの `qJointEntropySearch` に解決されます。

### optimal sampleの自動生成

`optimal_inputs` / `optimal_outputs` を指定しない場合、BoTorchのinput constructorを使ってposteriorから最適点・最適値sampleを生成します。

```python
DataContext(
    bounds=bounds,
    extra={"jes_num_optima": 64},
)
```

明示する場合は `optimal_inputs` と `optimal_outputs` を必ず組で指定します。

```python
AcquisitionConfig(
    name="jes",
    acqf_kwargs={
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
    },
)
```

multi-output modelではMESと同様に明示的な `posterior_transform` が必要です。

## 3. Hypervolume Knowledge Gradient (HVKG)

HVKGはmulti-objective BOにおいて、観測後に得られるPareto front / hypervolumeの将来価値を評価するKnowledge Gradientです。

```python
AcquisitionConfig(name="hvkg")
AcquisitionConfig(name="qhvkg")
```

はBoTorchの `qHypervolumeKnowledgeGradient` に解決されます。

HVKGは最低2出力を持つmulti-output regression-like modelを必要とします。

### ref point

優先順位は次です。

1. `AcquisitionConfig.acqf_kwargs["ref_point"]`
2. `DataContext.ref_point`
3. 観測済みmulti-objective値から自動生成

明示値が自動値やcontext値で上書きされることはありません。

### current value

`current_value` を指定しない場合は、BoTorchのinput constructorを利用して現在の最適hypervolume valueを計算します。

```python
AcquisitionConfig(
    name="hvkg",
    acqf_kwargs={
        "num_fantasies": 8,
        "num_pareto": 10,
    },
)
```

`current_value` を明示した場合は再計算しません。

HVKGはone-shot acquisitionなので、高位APIはsequential optimizationへ変換せずjoint optimizationを維持します。

## 4. Objectiveとの関係

MES/JESはposteriorの最適値・最適点に対する情報量を扱うため、通常のMC `objective` を暗黙に適用しません。multi-outputをscalar化する場合は `posterior_transform` を明示します。

HVKGはmulti-objective objectiveを保持します。`MultiObjectiveConfig.scalarization_weights`によるgeneric scalarizationはHVKGでは無効化され、目的次元を保ったままhypervolumeを評価します。

## 5. Task routing

short alias `mes` / `jes` / `hvkg` はregression系posteriorに限定します。binary / multiclass / ordinalへBoTorch標準実装を暗黙転送すると、確率空間・utility空間の意味が変わるためです。

classification向け情報理論acquisitionはBALD、predictive entropyなど既存のtask-specific実装を利用します。

## 6. 利用例

### MES

```python
acq = AcquisitionConfig(name="mes")
opt = OptimizeConfig(q=3)

X_next, value = optimizer.candidate(
    acq,
    opt,
    data_context=DataContext(bounds=bounds),
)
```

`q=3`ではsequentialが自動適用されます。

### JES

```python
acq = AcquisitionConfig(name="jes")
opt = OptimizeConfig(q=1)

X_next, value = optimizer.candidate(
    acq,
    opt,
    data_context=DataContext(
        bounds=bounds,
        extra={"jes_num_optima": 64},
    ),
)
```

### HVKG

```python
acq = AcquisitionConfig(name="hvkg")
context = DataContext(
    bounds=bounds,
    multi_objective=MultiObjectiveConfig(
        ref_point=ref_point,
    ),
)

X_next, value = optimizer.candidate(
    acq,
    OptimizeConfig(q=1),
    data_context=context,
)
```

## 7. 選択の目安

| 目的 | 手法 |
|---|---|
| 最適値そのものの不確実性を減らす | MES |
| 最適点と最適値を同時に特定する | JES |
| multi-objectiveの将来Pareto frontを改善する | HVKG |
| 直接的な改善量を重視する | LogEI / LogNEI |
| Pareto hypervolumeの即時改善を重視する | LogEHVI / LogNEHVI |

MES/JES/HVKGは一般にEI系より計算コストが高いため、1回の実験コストが高く情報価値を重視する材料探索で特に有効な候補です。
