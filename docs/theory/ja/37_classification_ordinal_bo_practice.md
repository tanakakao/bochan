# 37. Classification / Ordinal Bayesian Optimization 実践編

実験結果が連続値とは限りません。成功/失敗、品質class、順序ratingしか得られない場合でも、latent probabilistic modelを通してBOを構成できます。本章は06章を補完し、acquisitionへの接続を中心に整理します。

## 1. Binary response

latent function `f(x)` に対して

```math
P(y=1\mid x)=\sigma(f(x))
```

とします。観測は0/1でもposteriorはlatent functionとsuccess probabilityの不確かさを持ちます。

## 2. Multiclass

```math
P(y=k\mid x)=\pi_k(x),\qquad\sum_k\pi_k(x)=1
```

です。目的は特定classの確率最大化、class utility最大化、feasibility判定などに変換できます。

## 3. Ordinal

順序classでは

```math
P(y=k\mid f)=P(b_{k-1}<f+\epsilon\le b_k)
```

のようにthresholdを持つlatent modelを考えます。class番号をそのままGaussian regressionするより順序構造を適切に表現できます。

## 4. Probability of Feasibility

binary responseが合格/不合格なら

```math
\alpha(x)=P(y=\mathrm{feasible}\mid x,D)
```

自体が有用なcriterionです。制約付きBOのfeasibility modelにも使えます。

## 5. Expected utility

class `k` にutility `u_k` を与えれば

```math
U(x)=\sum_k u_k P(y=k\mid x,D)
```

としてscalar objectiveへ変換できます。ordinalでは単純なclass indexよりdomain utilityを定義する方が意味が明確です。

## 6. EI / PI / UCB拡張

continuous latent value、probability、expected utilityのどれをoptimization対象とするかを明示する必要があります。

bochanのbinary/multiclass/ordinal acquisitionは、このprobability/utility semanticsをstandard BO criterionへ接続する層として理解できます。

## 7. Multi-output classification

複数のclassification/ordinal outputsをobjective vectorへ変換すれば、EHVI/NEHVI/NParEGO型のmulti-objective探索へ拡張できます。

重要なのはposterior raw latentをそのままPareto objectiveにするのか、probability/utilityへ変換してから使うのかを固定することです。

## 8. Active Learningとの違い

classification ALはboundaryやmodel uncertaintyを学ぶことが目的です。一方classification BOは高utility class/probabilityを持つ領域を発見することが目的です。

```text
BO: best decisionを探す
AL: classifierを改善する
LSE: threshold boundaryを特定する
```

同じposteriorでもacquisitionの目的が異なります。

## 9. Calibration

classification BOではprobabilityがdecisionへ直接使われるため、accuracyだけでなくcalibrationが重要です。

例えばpredicted probability 0.9の点が実際には60%しか成功しないmodelではPoF-based decisionが過度に楽観的になります。

## 10. 材料・工程例

- 合成成功 / 失敗
- phase formation yes/no
- quality grade A/B/C
- crack severity ordinal class
- processability rating

classification/ordinal BOは、無理に連続scoreを作るより**観測dataの生成過程に合ったlikelihoodを使い、そのposteriorを意思決定へ変換する**ことが基本です。
