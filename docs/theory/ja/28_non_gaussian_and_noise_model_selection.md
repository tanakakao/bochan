# 28. 応答分布とNoiseモデルの選択

Gaussian regressionだけでは、確率、count、positive continuous、ordinal responseなどを自然に表現できない場合があります。本章では「目的変数のsupport」と「noise生成機構」からmodel familyを選ぶ考え方を整理します。

## 1. Gaussian regression

連続値 `y∈R` で、条件付き分布が概ね対称なら

```math
y=f(x)+ε,  ε~N(0,σ²)
```

が基本です。変換後にGaussian approximationが妥当なら、最も単純で扱いやすいbaselineになります。

## 2. Binary response

成功/失敗など `y∈{0,1}` ではlatent functionをlink functionで確率へ変換します。

```math
p(y=1|x)=g(f(x))
```

`g` はlogistic/probitなどです。BOではlatent valueそのものではなく、成功確率、feasibility probability、utilityなど何を最適化するかを明示します。

## 3. Multiclass response

unordered classでは

```math
p(y=k|x),  k=1,...,K
```

を扱います。target class probability、expected utility、entropyなど、posterior class probabilityからdecision scoreへの変換が必要です。

## 4. Ordinal response

順序カテゴリではclass間の順序を捨てないことが重要です。latent score `f(x)` とcutpoint `c_k` を用いて

```math
P(y≤k|x)=g(c_k-f(x))
```

のように表現できます。

単純multiclassより、順序情報を利用したexpected utilityやboundary explorationが自然です。

## 5. Count data

count responseにはPoissonやNegative Binomialが候補になります。

Poisson:

```math
y~Poisson(λ(x)),  log λ(x)=f(x)
```

Poissonでは平均と分散が同程度という仮定があります。overdispersionが強ければNegative Binomialなどを検討します。

## 6. [0,1]の連続値

割合・率のような連続値で0と1を除く場合、Beta regressionが候補です。

```math
y~Beta(α(x),β(x))
```

平均 `μ(x)` とprecision `φ(x)` を使えば

```math
α=μφ,  β=(1-μ)φ
```

と書けます。0/1を実際に含むデータではzero/one inflationや別modelを検討する必要があります。

## 7. 正の連続値

`y>0` で右裾が長く、varianceがmeanとともに増える場合、Gamma regressionが候補です。単純なlog transform + Gaussian GPとの比較も重要です。

## 8. HomoscedasticとHeteroscedastic

一定noiseでは

```math
Var(y|x)=σ²
```

ですが、入力依存noiseでは

```math
Var(y|x)=σ²(x)
```

です。

製造条件によってばらつきが変わる場合、heteroscedastic modelは平均性能だけでなく安定性を表現する上でも重要です。

## 9. EpistemicとAleatoric uncertainty

概念的にpredictive uncertaintyは

```math
predictive uncertainty
≈ epistemic uncertainty + aleatoric uncertainty
```

に分けて考えます。

データ追加で減らしたいのは主としてepistemic uncertaintyです。measurement noiseが大きい場所をposterior varianceだけで繰り返し観測すると、Active Learningとして非効率になる場合があります。

## 10. 変換するか専用likelihoodを使うか

例えばpositive responseなら

```text
log(y) + Gaussian GP
```

と

```text
Gamma likelihood + latent GP
```

の両方が候補です。

変換モデルは単純でBoTorch標準acquisitionと接続しやすい一方、元scaleでの期待値・variance解釈には逆変換の影響があります。専用likelihoodはdata-generating processを表現しやすい反面、variational inferenceやposterior transformationが複雑になります。

## 11. bochanのmodel/acquisition familyとの対応

bochanではresponse familyに応じてacquisition packageを分離しています。

| 応答 | acquisition family |
|---|---|
| Gaussian continuous | `regression` |
| Binary | `binary` |
| Multiclass | `multiclass` |
| Ordinal | `ordinal` |
| Poisson/Beta/Gamma/Negative Binomial等 | `non_gaussian` |

heteroscedastic variantでは、同じtask familyでもnoise modelを区別します。

## 12. 選択の実務フロー

```text
response supportを確認
  ↓
Gaussian approximationは妥当か
  ↓
専用likelihoodが必要か
  ↓
noiseは一定か入力依存か
  ↓
posterior calibrationを確認
  ↓
目的に合うobjective/acquisitionへ変換
```

最終的にはRMSEだけでなく、log score、calibration、coverage、classification probability calibration、そしてBO/ALでのsequential performanceを確認します。
