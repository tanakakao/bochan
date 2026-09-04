# 30. Multi-fidelity Bayesian Optimization

Multi-fidelity Bayesian Optimization (MFBO) は、評価精度とcostの異なる複数の情報源を使いながら、最終的なhigh-fidelity objectiveを効率よく最適化する方法です。

## 1. 問題設定

design variableを `x`、fidelityを `s` とします。

```math
y=f(x,s)+\epsilon
```

最終的に興味があるtarget fidelityを `s_H` とすると、目的は

```math
x^*=\arg\max_x f(x,s_H)
```

です。

しかし毎回 `s_H` を評価すると高価なので、安価な `s_L` の観測も利用します。

---

## 2. Fidelityとは何か

fidelityは単なるcategoryではなく、通常はtarget evaluationとのaccuracy/cost trade-offを表します。

例:

| Low fidelity | High fidelity |
|---|---|
| ML surrogate | DFT |
| coarse DFT settings | converged DFT |
| small simulation | large simulation |
| short simulation time | long simulation time |
| DFT | experiment |
| simplified process model | pilot experiment |

ただしMLIP predictionが常にDFTの有効なlow fidelityになるとは限りません。target domainでcorrelationを確認する必要があります。

---

## 3. Discrete fidelity

情報源が明確に離散的なら

```math
s\in\{0,1,2\}
```

とします。

例えば

```text
0 = MLIP
1 = DFT
2 = experiment
```

です。

この場合、task-like covarianceを利用するmodelも考えられます。

---

## 4. Continuous fidelity

simulation resolutionなどでは

```math
s\in[0,1]
```

としてcontinuous fidelityを使えます。

例えば `s=1` がtarget accuracyで、低い `s` ほど計算が安いとします。

入力は

```math
\tilde x=[x,s]
```

となります。

通常のdesign dimensionとfidelity dimensionは意味が異なるため、kernelやacquisitionでもその役割を区別します。

---

## 5. Posteriorが学ぶもの

MF modelは

```math
p(f(x,s)\mid D)
```

を構築し、low-fidelity observationからtarget fidelity `s_H` のposteriorも更新します。

重要なのはcross-fidelity covarianceです。

```math
\operatorname{Cov}(f(x,s_L),f(x',s_H))
```

がほぼ0なら、low fidelityを何点追加してもhigh-fidelity optimum探索にはあまり役立ちません。

---

## 6. Candidateとfidelityを同時に選ぶ

standard BOでは

```math
x_{t+1}=\arg\max_x\alpha(x)
```

ですが、MFBOでは

```math
(x_{t+1},s_{t+1})
=\arg\max_{x,s}\alpha_{MF}(x,s)
```

となります。

この点が単なるmulti-source regressionとの大きな違いです。

---

## 7. Cost model

fidelityごとのcostを

```math
c(x,s)>0
```

とします。

単純な場合は `c(s)` だけでも構いません。

例えば

```text
MLIP       1
DFT       100
experiment 1000
```

のようなrelative costを置けます。

ただしwall-clock timeだけでなく、人的cost、装置占有時間、材料消費、失敗riskなど何をcostとするかを明示します。

---

## 8. Cost-aware utility

情報価値を `V(x,s)` とすると、概念的なcost-aware scoreは

```math
\alpha(x,s)=\frac{V(x,s)}{c(x,s)}
```

です。

実際のacquisitionはより複雑ですが、MFBOの直感は「安いからlow fidelity」ではなく、**target optimumについて得られる情報をcostと比較する**ことです。

---

## 9. Knowledge Gradientとの関係

Knowledge Gradientは観測後に最終decisionの価値がどれだけ改善するかを見るvalue-of-information criterionです。

MF settingでは、fidelity `s` で観測した結果がtarget fidelityでの最終decisionをどれだけ改善するかを評価できます。

概念的には

```math
KG(x,s)
=
\mathbb E\left[V(D\cup\{(x,s,y)\})-V(D)\right]
```

です。

これをcost-awareにすれば、安価だが情報量の少ないevaluationと、高価だがtargetへ直接近いevaluationを比較できます。

---

## 10. Entropy / information-based MFBO

別の考え方はtarget optimum `x^*` やtarget functionについてのentropy reductionです。

```math
I(y_{x,s};x^*\mid D)
```

を評価すれば、「このfidelityの観測が最適解の不確かさをどれだけ減らすか」を直接扱えます。

costでnormalizeすればinformation per costという解釈になります。

---

## 11. Residual GPとの実務上の違い

### Residual GP

```text
MLIP prediction
     +
target residual GP
     ↓
corrected target posterior
```

MLIP predictionを全candidateで安価に計算できるなら、この方式は非常に扱いやすいです。

### MFBO

```text
candidate x
   ↓
choose fidelity s
   ↓
MLIP / DFT / experiment
   ↓
joint posterior update
```

low-fidelity evaluation自体に無視できないcostがあり、どのfidelityを次に評価するか選びたい場合にMFBOの価値が大きくなります。

---

## 12. Materialsでの例

### MLIP → DFT

MLIP relaxation/energyをlow fidelity、DFTをhigh fidelityとして扱う構成です。

ただしpretrained MLIPを全候補へほぼ無料で評価できるなら、MLIPをbaseline featureとして使うResidual GPの方が単純な場合があります。

### DFT → Experiment

DFTをlow fidelity、実験をhigh fidelityとする構成は自然ですが、simulation-to-experiment gapが大きい場合はcorrelation modelが重要です。

### 計算条件

k-point density、cutoff energy、supercell sizeなどをfidelity parameterとして扱うことも考えられます。

---

## 13. 実装時のデータ表現

conceptualには

```text
X = [design variables..., fidelity]
y = observed property
cost = evaluation cost
```

です。

fidelityを単なる通常featureとして追加するだけでは、target fidelityへのprojectionやcost-aware decisionが自動的に実現するわけではありません。

model、acquisition、optimizerの全てがfidelity semanticsを共有する必要があります。

---

## 14. bochanへ追加する場合の責務

将来bochanでMFBOを共通化する場合、少なくとも次を分離すると拡張しやすくなります。

```text
FidelitySpec
  - fidelity dimensions / tasks
  - target fidelity
  - allowed fidelity values

CostModel
  - evaluation cost

MultiFidelityModel
  - cross-fidelity posterior

MultiFidelityAcquisition
  - target-value / information gain
  - cost awareness

Optimizer
  - optimize x and fidelity
  - optionally project to target fidelity
```

既存の`residual_gp` modeとは独立した概念として設計する方が、APIの意味が明確になります。

---

## 15. 導入判断

MFBOを追加する前に次を確認します。

1. low/high fidelityの両方に実測dataがあるか
2. fidelity間に十分なcorrelationがあるか
3. cost差が大きいか
4. low fidelityにも無視できないcostがあるか
5. 次のfidelityを選択する意思決定が必要か

5が不要で、low-fidelity predictionを全候補に簡単に計算できるなら、まずResidual GPやfeature augmentationをbaselineにする価値があります。

---

## 16. まとめ

```text
standard BO:
    choose x

multi-fidelity BO:
    choose x + fidelity

residual GP:
    baseline + learned correction
```

bochanにとってMFBOはResidual GPの置き換えではなく、**evaluation sourceそのものをsequential decisionへ含める新しい探索軸**として位置付けるのが自然です。
