# 44. MLIP・DFT・実験をつなぐ階層的材料探索

材料探索では評価方法ごとにaccuracyとcostが大きく異なります。pretrained model、MLIP、DFT、実験を単一の予測器として扱うのではなく、役割を分けて階層的に利用することで探索効率を高められます。

## 1. Evaluation hierarchy

典型的には

```text
cheap screening
  -> ML / composition model
  -> MLIP relaxation
  -> DFT
  -> experiment
```

とcostが上昇します。

各段階の目的は「最終propertyを完全に当てる」ことではなく、次の高価な段階へ送る候補を効率よく絞ることです。

## 2. Composition screening

候補が非常に多い場合、CrabNet/Roost等でcomposition-only screeningを行えます。

```text
millions of compositions
 -> composition encoder/model
 -> shortlist
```

この段階ではcrystal structureを知らないため、structure-sensitive propertyには限界があります。

## 3. Structure generation

shortlistされたcompositionについて候補structureを生成します。

```text
composition
 -> prototype / database / generator
 -> structure candidates
```

1組成から複数polymorphが生成されることがあります。

## 4. MLIP relaxation

構造候補をMLIPでrelaxし、明らかに不安定な候補を除外します。

```text
structure candidates
 -> MACE / CHGNet / M3GNet / ALIGNN-FF
 -> relaxed structures
```

ここではenergy、force、stressを利用できます。

## 5. Residual GP

pretrained MLIPとtarget-domain high-fidelity dataにsystematic gapがあるなら

```math
y_H=f_{MLIP}(x)+\delta_{GP}(x)+\epsilon
```

として補正できます。

これはMLIPの大量事前学習知識を保持しながら、少量のDFT/experimentでtarget domainへcalibrateする方法です。

## 6. DFT selection

全候補をDFTへ送る代わりにacquisitionを使います。

```text
relaxed candidate bank
 -> posterior
 -> EI/UCB/KG/AL
 -> selected DFT calculations
```

optimizationならobjective improvement、model improvementならAL criterionを使います。

## 7. Experiment selection

DFT結果と実験結果にはdomain gapがあります。最終目的が実験propertyなら、DFT optimumだけを追うのではなくexperiment posteriorを更新する必要があります。

## 8. Residual vs Multi-fidelity

MLIP predictionを全候補へほぼ無料で計算できるならResidual GPが単純です。

一方、DFTと実験のどちらを次に評価するかまで決めたいなら

```math
(x,s)=\text{candidate + fidelity}
```

としてMulti-fidelity BOを考えます。

## 9. Cost-aware decision

各段階のcostを

```math
c_{ML}<c_{MLIP}<c_{DFT}<c_{EXP}
```

とすると、情報価値/costでevaluation sourceを選ぶ設計が可能です。

ただし実験costは単純な時間だけでなく材料、装置、人、失敗riskを含みます。

## 10. Domain shift

pretrained modelのtraining domainと探索domainが離れるとprediction biasやoverconfidenceが生じます。したがって階層探索では各stageで

- calibration
- residual error
- uncertainty
- out-of-domain indication

を監視する必要があります。

## 11. Closed-loop architecture

```text
candidate pool
  -> cheap model
  -> MLIP relaxation
  -> GP / residual GP
  -> acquisition
  -> DFT / experiment
  -> result ingestion
  -> model update
  -> next acquisition
```

これを繰り返します。

## 12. bochanの責務

bochanは全physics engineを内包するより、

```text
representation
surrogate
uncertainty
acquisition
workflow orchestration
```

を共通化する方が強みを出せます。

DFT codeやlaboratory automationはadapter/APIを介して接続する構造が望ましいです。

## 13. Failure handling

DFT convergence failure、relaxation failure、experiment failureも情報です。単純にmissingとして捨てるだけでなく、feasibility modelやfailure classifierとして学習する余地があります。

## 14. 推奨導入順

```text
1. pretrained screening
2. MLIP relaxation
3. high-fidelity residual GP
4. acquisition-based DFT selection
5. experiment feedback
6. cost-aware multi-fidelity decision
7. automated closed loop
```

一度に完全自動化するより、各stageのprediction qualityとselection gainを検証しながら進めます。

この階層化により、bochanのcomposition model、MLIP、Residual GP、Multi-fidelity、BO/ALを一つの材料探索workflowとして接続できます。
