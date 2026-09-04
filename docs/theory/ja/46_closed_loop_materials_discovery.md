# 46. Closed-loop Materials Discovery

Closed-loop materials discoveryは、modelが候補を提案し、simulation/experimentを実行し、その結果でmodelを更新して次候補を決めるcycleです。Bayesian OptimizationやActive Learningはこのloopのdecision engineになります。

## 1. 基本loop

```text
Design space
   ↓
Representation
   ↓
Surrogate model
   ↓
Acquisition
   ↓
Candidate
   ↓
DFT / Experiment
   ↓
Data validation
   ↓
Model update
   └──────────────→ repeat
```

重要なのはcandidate generationだけでなく、evaluation resultを確実に次cycleへ戻すことです。

## 2. Design space

探索変数には

```text
composition
crystal structure
process conditions
measurement conditions
fidelity
```

などがあります。

各変数のconstraint、単位、domain、conditional dependencyをmachine-readableに定義する必要があります。

## 3. Representation layer

raw variablesをmodel入力へ変換します。

```text
composition -> CrabNet / Roost / descriptors
structure   -> graph / MLIP / ALIGNN representation
process     -> normalized tabular variables
```

representation versionも再現性の一部です。

## 4. Surrogate layer

目的に応じてGP、DKL、multi-task、residual GP、classification/ordinal model等を選びます。

posterior meanだけでなくuncertainty calibrationがcandidate selectionへ直接影響します。

## 5. Decision layer

目的がoptimizationならEI/UCB/KG/EHVI等、model improvementならvariance/entropy/BALD等を使います。

constraints、robustness、fidelity、batch sizeもこの層で扱います。

## 6. Execution layer

候補を実際のevaluationへ渡します。

```text
candidate
 -> simulation input generation
 -> scheduler / instrument
 -> execution
 -> result collection
```

bochan本体がDFT engineや装置driverを直接抱えるよりadapter/API境界を持つ方が保守しやすくなります。

## 7. Result validation

自動loopでは失敗dataをそのまま学習へ入れない仕組みが必要です。

- convergence check
- unit validation
- range validation
- missing output
- duplicated experiment
- instrument error
- provenance check

validation failureは単なる欠損ではなくfeasibility informationとして利用できる場合があります。

## 8. Provenance

各観測には少なくとも

```text
candidate specification
evaluation method
fidelity
model/workflow version
timestamp
raw result
processed result
failure state
```

を紐付けると再現性が高まります。

## 9. Human-in-the-loop

完全自動化だけがclosed loopではありません。

```text
model recommendation
 -> researcher review
 -> approve / modify / reject
 -> execution
```

というhuman approvalを含むloopも実務では有効です。特に危険な条件、設備制約、新規材料では重要です。

## 10. Batch operation

実験設備では1点ずつよりbatchで処理することが多いため、q-batch acquisition、diversity、装置capacity、campaign schedulingを考えます。

## 11. Asynchronous operation

DFT/実験の終了時間が異なる場合、完了を待たずに次candidateを選びたいことがあります。pending pointsをposterior/acquisitionへ反映し、重複提案を防ぎます。

## 12. Multi-fidelity loop

```text
candidate
 -> choose evaluation source
      MLIP / DFT / experiment
 -> cost-aware acquisition
```

とすれば、candidateとevaluation fidelityを同時に管理できます。

## 13. Robust closed loop

実験でinput variationがある場合、nominal optimumだけでなくVaR/CVaR等を用いたrobust objectiveをdecision layerへ入れます。

loopが進むにつれてprocess variability model自体を更新することも考えられます。

## 14. Stopping criteria

closed loopは無限に回すものではありません。

- target property達成
- Pareto front改善停止
- uncertainty低下
- budget消化
- expected improvement低下
- cost-benefit threshold

を停止条件にできます。

## 15. Monitoring

運用時には

```text
best observed value
hypervolume
posterior calibration
candidate diversity
failure rate
cost consumed
cycle time
```

などを監視します。

単にmodel lossを見るだけではclosed-loop performanceを評価できません。

## 16. Reproducibility

各iterationで

```text
data snapshot
model configuration
random seed
acquisition configuration
candidate
result
```

を保存すると、なぜその候補が選ばれたか追跡できます。

## 17. bochanのarchitecture

bochanをclosed-loop decision platformとして見ると

```text
Data / Representation
        ↓
Model Factory
        ↓
Posterior
        ↓
Objective / Constraints / Risk
        ↓
Acquisition Factory
        ↓
Candidate Optimization
        ↓
Workflow / API
        ↓
External evaluator
```

という構造になります。

Materials系ではさらに

```text
Composition models
Structure models
MLIP relaxation
Residual GP
Multi-fidelity
```

をこの共通decision loopへ接続します。

## 18. 段階的な自動化

推奨順序は

```text
1. recommendation only
2. human-approved execution
3. automatic result ingestion
4. automatic retraining
5. automatic candidate submission
6. multi-fidelity / scheduling optimization
```

です。

最初から完全自律化するより、各境界でvalidationとrollbackを可能にします。

## 19. 最終像

```text
composition / structure / process space
        ↓
physics + pretrained models
        ↓
probabilistic surrogate
        ↓
BO / AL / robust / multi-fidelity decision
        ↓
DFT / experiment
        ↓
validated data
        ↓
continuous learning
```

bochanの各機能は独立機能の集合ではなく、このclosed-loopを構成するcomponentとして整理できます。
