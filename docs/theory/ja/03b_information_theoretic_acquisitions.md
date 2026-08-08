# 03b. 情報理論・look-ahead型の獲得関数

本章では `bochan` の高位APIから利用できる KG、MES、JES、MO-MES、MO-JES、HVKG を整理します。これらは EI / UCB のように候補点そのものの改善量だけを見るのではなく、最適値・最適点・Pareto front・将来の意思決定価値に関する情報価値を評価します。

KGの `current_value`、multi-output scalarization、pending point、one-shot optimizationの詳細は `03c_knowledge_gradient.md`、Pareto多目的Entropy Searchの詳細は `03d_multiobjective_entropy_search.md` を参照してください。

## 1. Max-value Entropy Search (MES)

MESは、未知の最大値 `f*` に関して候補点の観測がどれだけ情報を与えるかを評価します。

```math
\alpha_{MES}(x)=I(y_x; f^*\mid \mathcal D)
```

```python
AcquisitionConfig(name="mes")
AcquisitionConfig(name="qmes")
```

はBoTorchの `qMaxValueEntropy` に解決されます。

`candidate_set` を明示しなければ、`DataContext.bounds` からBoTorchのinput constructorを使って候補集合を自動生成します。

```python
DataContext(bounds=bounds, extra={"mes_candidate_size": 2000})
```

既定値は1000です。`q > 1` では高位APIが `OptimizeConfig.sequential=True` を自動設定します。

multi-output modelへMESを適用する場合は、何を最大化するscalar posteriorとするかを曖昧にしないため、明示的な `posterior_transform` が必要です。これはPareto多目的MESではなく、複数出力を1つのscalar utilityに落としたMESです。

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

`optimal_inputs` / `optimal_outputs` を指定しない場合、BoTorchのinput constructorを使ってposteriorから最適点・最適値sampleを生成します。

```python
DataContext(bounds=bounds, extra={"jes_num_optima": 64})
```

明示する場合は `optimal_inputs` と `optimal_outputs` を必ず組で指定します。multi-output modelではMESと同様に明示的な `posterior_transform` が必要です。

## 3. Pareto多目的 Entropy Search

### MO-MES / MESMO

```python
AcquisitionConfig(name="mo_mes")
AcquisitionConfig(name="mesmo")
```

はBoTorchの `qLowerBoundMultiObjectiveMaxValueEntropySearch` に解決されます。候補観測がPareto optimal outputsについて与える情報量を評価します。

### MO-JES

```python
AcquisitionConfig(name="mo_jes")
```

はBoTorchの `qLowerBoundMultiObjectiveJointEntropySearch` に解決されます。Pareto optimal input-output pairsについての情報量を評価します。

MO-MES / MO-JESはscalarizationを行わず、モデルの複数出力をPareto objectivesとして保持します。補助量を省略するとBoTorchの `sample_optimal_points` と `compute_sample_box_decomposition` からPareto samples / hypercell boundsを生成します。

既定の補助設定は次です。

```python
DataContext(
    bounds=bounds,
    extra={
        "mo_entropy_num_pareto_samples": 8,
        "mo_entropy_num_pareto_points": 8,
        "mo_entropy_num_samples": 64,
        "mo_entropy_estimation_type": "LB",
    },
)
```

`q > 1` ではBoTorch公式tutorialに合わせてsequential greedy candidate generationを自動使用します。正式な自動生成範囲は、連続入力・Gaussian GP compatible model・2目的以上・制約なしです。mixed/categorical inputでは補助量を明示してください。

詳細は `03d_multiobjective_entropy_search.md` を参照してください。

## 4. Hypervolume Knowledge Gradient (HVKG)

HVKGはmulti-objective BOにおいて、観測後に得られるPareto front / hypervolumeの将来価値を評価するKnowledge Gradientです。

```python
AcquisitionConfig(name="hvkg")
AcquisitionConfig(name="qhvkg")
```

はBoTorchの `qHypervolumeKnowledgeGradient` に解決され、最低2出力を必要とします。

ref pointの優先順位は次です。

1. `AcquisitionConfig.acqf_kwargs["ref_point"]`
2. `DataContext.ref_point`
3. 観測済みmulti-objective値から自動生成

`current_value` を指定しない場合はBoTorchのinput constructorで計算します。明示値は再計算しません。

HVKGはone-shot acquisitionなので、高位APIはsequential optimizationへ変換せずjoint optimizationを維持します。

## 5. Objectiveとの関係

- **MES / JES**: scalar posteriorを対象とします。multi-outputでは `posterior_transform` を明示します。
- **KG**: scalar terminal objectiveを扱います。multi-outputでは `objective` / `objective_config` / `objective_factory` または `posterior_transform` を明示します。
- **MO-MES / MO-JES**: モデルの複数出力そのものをPareto objectivesとして使います。scalar `objective` / `posterior_transform` は使いません。
- **HVKG**: multi-objective objectiveを保持し、hypervolumeの将来価値を評価します。

したがって、multi-outputだから自動的にMO-MES / MO-JESになるわけではありません。

```text
multi-output → scalar utility → MES / JES / KG
multi-output → Pareto objectives → MO-MES / MO-JES / HVKG
```

## 6. Task routing

short alias `kg` / `mes` / `jes` / `mo_mes` / `mo_jes` / `hvkg` はGaussian regression系posteriorを中心に扱います。binary / multiclass / ordinalへBoTorch標準実装を暗黙転送しません。

classification向け情報理論acquisitionはBALD、predictive entropyなど既存のtask-specific実装を利用します。

MO-MES / MO-JESの自動Pareto samplingはhomogeneous regression objectiveを前提とし、hybridやmixed categoricalの補助量自動生成は正式サポート外です。

## 7. 利用例

### KG

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="kg"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### MES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mes"),
    OptimizeConfig(q=3),
    data_context=DataContext(bounds=bounds),
)
```

### JES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### MO-MES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_mes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### MO-JES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### HVKG

```python
context = DataContext(
    bounds=bounds,
    multi_objective=MultiObjectiveConfig(ref_point=ref_point),
)

X_next, value = optimizer.candidate(
    AcquisitionConfig(name="hvkg"),
    OptimizeConfig(q=1),
    data_context=context,
)
```

## 8. 選択の目安

| 問題 | 情報獲得 / look-ahead | 改善量ベース |
|---|---|---|
| 単目的・scalar utility | KG / MES / JES | LogEI / LogNEI |
| Pareto多目的 | MO-MES / MO-JES / HVKG | LogEHVI / LogNEHVI |
| scalar化多目的 | KG / MES / JES | LogNParEGO |

目的別には次のように整理できます。

| 目的 | 手法 |
|---|---|
| 観測後の最終意思決定価値を高める | KG |
| scalar最適値の不確実性を減らす | MES |
| scalar最適点と最適値を同時に特定する | JES |
| Pareto optimal outputsについて学ぶ | MO-MES / MESMO |
| Pareto optimal inputs + outputsについて学ぶ | MO-JES |
| 将来のPareto hypervolumeを改善する | HVKG |

これらの情報獲得・look-ahead acquisitionは一般にimprovement系より計算コストが高いため、1回の物理実験コストが高く、次の実験から得られる情報価値を重視する材料探索で特に有力です。
