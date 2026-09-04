# 31. Robust Bayesian Optimization：Input Perturbation・VaR・CVaR

通常のBayesian Optimizationでは、指定した設計点 `x` がそのまま実現されることを暗黙に仮定します。しかし製造条件、組成比、温度、圧力、流量などには実行時ばらつきがあり、nominal optimumが実工程では不安定なことがあります。

Robust Bayesian Optimizationでは、**設定値そのものではなく、設定値の周辺で起こるばらつきを含めた性能**を最適化します。

---

## 1. Nominal optimizationとRobust optimization

通常の最適化は

```math
x^*=\arg\max_x f(x)
```

です。

実際の入力が

```math
\tilde x=x+\xi
```

となるなら、実現性能は

```math
f(x+\xi)
```

です。

ここで `\xi` はprocess variation、setting error、composition errorなどを表します。

Robust optimizationでは、例えば

```math
x^*=\arg\max_x \mathbb E_{\xi}[f(x+\xi)]
```

を考えます。

平均だけでなくworst-tail performanceを使うこともできます。

---

## 2. Input perturbation

bochanで使われる `n_w` は、1つのnominal candidateの周辺に複数のperturbed inputsを生成して評価する考え方と対応します。

nominal candidate `x` に対して

```math
x^{(j)}=x+\xi^{(j)},
\qquad j=1,\ldots,n_w
```

を生成します。

posterior sampleを含めればtensorは概念的に

```text
posterior sample × candidate q × perturbation n_w × output
```

のような構造を持ちます。

objective側では `n_w` axisをrisk measureでaggregateし、再びcandidate単位のutilityへ戻します。

---

## 3. Perturbation distribution

`\xi` の分布は工程知識を表します。

例:

```math
\xi\sim\mathcal N(0,\Sigma_w)
```

またはbounded uncertaintyとして

```math
\xi_j\sim\mathrm{Uniform}(-a_j,a_j)
```

などを使えます。

重要なのは、単純に全変数へ同じ標準偏差を入れることではありません。

- 温度 ±2 ℃
- 組成比 ±0.2 %
- 圧力 ±0.01 MPa

のように実際のcontrol accuracyや工程ばらつきを反映する方がrobust optimumの意味が明確になります。

---

## 4. Mean robustness

最も単純なのはexpected performanceです。

```math
R_{mean}(x)=\mathbb E_{\xi}[f(x+\xi)]
```

Monte Carloでは

```math
R_{mean}(x)
\approx
\frac{1}{n_w}\sum_{j=1}^{n_w}f(x+\xi^{(j)})
```

です。

平均性能が高い条件を選べますが、rare but severe failureを十分にpenalizeしない場合があります。

---

## 5. VaR

Value at Riskはdistributionのquantileを使います。

maximization problemでlower-tail riskを見るなら、performance random variable `Y_x=f(x+\xi)` に対して

```math
\mathrm{VaR}_{\alpha}(Y_x)
=Q_{1-\alpha}(Y_x)
```

のようにlower quantileをrobust scoreとして使えます。

実装では `alpha` の定義が「confidence level」なのか「tail probability」なのかを必ず確認します。

例えばlower 10% pointを重視するなら、「悪い10%の境界でも性能が高い条件」を選ぶ考え方になります。

---

## 6. CVaR

Conditional Value at Riskはtail内部の平均を見ます。

maximizationでlower tailをriskとする場合、概念的には

```math
\mathrm{CVaR}_{\alpha}(Y_x)
=\mathbb E[Y_x\mid Y_x\le \mathrm{VaR}_{\alpha}(Y_x)]
```

です。

VaRが1つのquantileだけを見るのに対し、CVaRは悪い領域全体を評価します。

製造条件探索では、平均性能が高いが一部で大きく品質低下する条件より、worst tailが安定した条件を優先できます。

---

## 7. Mean・VaR・CVaRの違い

| Risk measure | 見ているもの | 特徴 |
|---|---|---|
| Mean | 全体平均 | 効率重視、tail failureに弱い |
| VaR | 指定quantile | 解釈しやすいがquantile境界だけを見る |
| CVaR | worst tail平均 | severe failureを反映しやすい |

risk aversionを強めるほど、nominal maximumから離れても周辺安定性の高い領域を選ぶ傾向があります。

---

## 8. Posterior uncertaintyとの二重の不確かさ

Robust BOには少なくとも2種類のuncertaintyがあります。

```text
model uncertainty       : GP posterior uncertainty
input uncertainty       : process/input perturbation
```

これは同じものではありません。

GP posterior sampleを `f^{(s)}`、input perturbationを `\xi^{(j)}` とすると

```math
f^{(s)}(x+\xi^{(j)})
```

を評価します。

posterior sample axisとperturbation axisを混同しないことが重要です。

---

## 9. Observation noiseとの違い

input perturbationは

```math
x\rightarrow x+\xi
```

です。

observation noiseは

```math
y=f(x)+\epsilon
```

です。

工程条件がずれる問題と、同じ工程条件でも測定値がばらつく問題は別です。

Heteroscedastic GPは主に後者をmodel化し、Robust BOのinput perturbationは前者をdecision criterionへ入れます。

両方が同時に存在することもあります。

---

## 10. Multi-output robust objective

複数propertyがある場合、各perturbationでobjective変換してからrisk aggregationする設計が考えられます。

```math
u^{(j)}=U(\mathbf f(x+\xi^{(j)}))
```

その後

```math
R(x)=\rho(u^{(1)},\ldots,u^{(n_w)})
```

とします。

ここで `\rho` がmean、VaR、CVaRなどです。

重要なのは

```text
output scalarization
risk aggregation
```

の順序です。非線形utilityでは順序を交換できない場合があります。

---

## 11. Constraintsとrobustness

制約

```math
g(x)\le0
```

も入力ばらつきで破られる場合があります。

nominal feasibilityだけなら

```math
g(x)\le0
```

ですが、robust feasibilityでは

```math
P(g(x+\xi)\le0)\ge1-\delta
```

のようなchance constraintを考えられます。

製造では「平均品質は良いが工程ばらつきで規格外が頻発する条件」を避けるために重要です。

---

## 12. Composition optimization

組成比ではperturbation後にもclosure constraint

```math
\sum_j c_j=1
```

を満たす必要があります。

単純なindependent Gaussian perturbationではsimplexから外れる可能性があります。

そのため

- perturb後にrenormalize
- log-ratio spaceでperturb
- simplex上のdistributionを使う

などconstraint-aware perturbationが必要です。

CLR/ILR representationを使う場合、perturbationをどのspaceで定義するかも結果へ影響します。

---

## 13. Process optimization

工程条件では、各variableのばらつきが異なります。

例えば

```text
nominal:
T = 1000 ℃
P = 0.50 MPa

realized:
T = 1000 + ξ_T
P = 0.50 + ξ_P
```

とします。

`\xi_T` と `\xi_P` にcorrelationがあるなら、独立分布ではなくcovarianceを含むjoint distributionを使うべきです。

```math
\xi\sim\mathcal N(0,\Sigma_w)
```

工程データから `\Sigma_w` を推定できれば、実際のprocess variabilityをrobust searchへ反映できます。

---

## 14. bochanとの対応

bochanのobjective layerでは、概念的に

```text
candidate x
  ↓
input perturbation × n_w
  ↓
model posterior
  ↓
objective conversion
  ↓
risk aggregation
    mean / VaR / CVaR
  ↓
acquisition
```

という責務分離になります。

主要parameterは

```text
n_w       number of perturbation scenarios
risk_type mean / var / cvar
alpha     risk level
```

です。

`n_w` を増やすほどrisk estimateのMonte Carlo errorは減りやすい一方、posterior evaluation costは増えます。

---

## 15. Acquisitionとの関係

Robustnessはacquisition functionそのものと独立したobjective transformationとして扱える場合があります。

例えばEIなら

```text
posterior samples
  -> perturbation risk aggregation
  -> robust utility samples
  -> improvement
```

という構成が可能です。

同様にmulti-objective acquisitionではrobust objective vectorを作ってからEHVI/NEHVIへ渡す設計が考えられます。

ただしacquisition implementationが期待するsample shapeとobjective shapeを守る必要があります。

---

## 16. Robust BOを使うべき場面

有効なのは次のような場合です。

- nominal optimum付近の勾配が急で条件ずれに弱い
- process setting accuracyが有限
- composition preparation errorが無視できない
- 最良平均値より再現性を重視
- 規格外riskを下げたい

逆に、入力が高精度に制御できる場合やperturbation distributionを全く推定できない場合、複雑なrobust criterionを入れても意味が曖昧になります。

---

## 17. 実務的な導入順

まず通常BOをbaselineとし、次に

```text
1. input variationを推定
2. mean robustness
3. VaR / CVaR
4. robust constraints
5. multi-objective robust BO
```

と段階的に追加するのが安全です。

通常BOとのcandidate差、予測される平均性能、tail performance、実験再現性を比較します。

---

## 18. まとめ

Robust BOの中心は

```text
「最も良い設定値」
ではなく
「実際にばらついても良い設定値」
```

を探索することです。

bochanの `n_w`、`risk_type`、`alpha` はこの考え方をposterior objectiveへ接続するparameterとして理解できます。
