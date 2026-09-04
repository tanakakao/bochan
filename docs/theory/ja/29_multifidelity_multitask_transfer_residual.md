# 29. Multi-fidelity・Multi-task・Transfer Learning・Residual Learning

これらはすべて「別の情報源を利用して少ない高価なデータを補う」ために使われますが、同じ概念ではありません。特に材料探索では、pretrained MLIP + residual GP、DFT + experiment、複数計算条件、複数物性が同時に現れるため、問題設定を分離して考える必要があります。

## 1. まず何が違うのか

| 概念 | 主に表すもの | 典型的な問い |
|---|---|---|
| Residual learning | baselineからの誤差 | MLIP予測を実験値に合わせて補正できるか |
| Multi-fidelity | 精度・costの異なる評価源 | DFTを使うか実験するかも含めて次を選べるか |
| Multi-task | 関連するtask間の統計的共有 | 複数物性・条件・taskの情報を共有できるか |
| Transfer learning | sourceで得た表現・parameterをtargetへ移す | pretrained encoderを少量target dataへ利用できるか |

重要なのは、同じdatasetが複数の観点を同時に持つことがある点です。例えばDFTと実験をtaskとして符号化することもできますが、意思決定としてfidelity costを扱うならmulti-fidelity problemとして考える必要があります。

---

## 2. Residual learning

baseline `b(x)` があるとき、targetを

```math
y_H(x)=b(x)+\delta(x)+\epsilon
```

と分解します。

GP residualなら

```math
\delta\sim\mathcal{GP}(m_\delta,k_\delta)
```

です。

training targetは

```math
r_i=y_{H,i}-b(x_i)
```

となり、predictionは

```math
\mu_H(x)=b(x)+\mu_\delta(x)
```

です。

### MLIP + Residual GP

bochanの材料系では、例えば

```text
pretrained MACE energy
        +
GP residual learned from target-domain observations
```

という構造を取れます。

ここではMLIP predictionは**既知のbaseline function**として使われます。通常、この定式化だけでは「次にMLIPを評価するかDFTを評価するか」を選択しません。

したがってResidual GPそのものはmulti-fidelity BOではありません。

---

## 3. Multi-fidelity model

fidelity parameter `s` を明示して

```math
y=f(x,s)+\epsilon
```

とします。

例えば

```text
s = 0: inexpensive empirical / ML prediction
s = 1: DFT
s = 2: experiment
```

のように、同じdesign `x` を異なるaccuracy/costで評価できます。

continuous fidelityならmesh density、simulation time、number of iterationsなどを連続変数として持つ場合もあります。

### 重要な特徴

multi-fidelity BOではdecisionが

```math
(x_{t+1},s_{t+1})
```

になり得ます。

つまり「どのcandidateを調べるか」だけでなく、**どの精度・costで調べるか**も最適化対象です。

---

## 4. Fidelity correlation

low fidelityが役立つためにはhigh fidelityとの関係をmodel化する必要があります。

単純なautoregressive表現なら

```math
f_H(x)=\rho f_L(x)+\delta(x)
```

です。

これはresidual learningと似ていますが、`f_L`自体も観測される確率過程として扱い、low/high fidelity双方のdataをjointに利用できる点が重要です。

別の考え方では、fidelityを入力dimensionとしてkernel

```math
k((x,s),(x',s'))
```

を定義します。

このときfidelity間のsimilarityがposterior covarianceとして表現されます。

---

## 5. Cost-aware acquisition

multi-fidelityではhigh fidelityほど高価であることが一般的です。したがってinformation gainやutilityをcostで割る考え方が自然です。

概念的には

```math
\alpha_{\mathrm{cost}}(x,s)
=\frac{\text{expected value of information}(x,s)}{c(x,s)}
```

です。

low fidelityを大量に取ることが常に有利ではありません。high fidelityとのcorrelationが弱ければ、安価でも情報価値は小さくなります。

---

## 6. Multi-task GP

multi-task GPではtask index `t` を持たせ、

```math
f(x,t)
```

をmodel化します。

例えば

- 複数の測定装置
- 複数の材料property
- 複数のdomain
- simulationとexperiment

などをtaskとして表現できます。

代表的な共分散は

```math
k((x,t),(x',t'))=k_x(x,x')k_t(t,t')
```

です。

### Multi-fidelityとの違い

multi-taskではtaskに必ずしも「精度の序列」や「cost hierarchy」がありません。

```text
hardness vs conductivity
```

はmulti-taskになり得ますが、通常どちらかがもう一方のlow fidelityではありません。

逆に

```text
coarse simulation vs accurate simulation
```

はtaskとしてmodel化できても、decision problemとしてはmulti-fidelityと呼ぶ方が意味を明確にできます。

---

## 7. Transfer learning

Transfer learningではsource domainで学習したparameterやrepresentationをtarget domainへ利用します。

```text
source data
   -> pretrained encoder/model
   -> reuse / fine-tune
   -> target task
```

材料系では

```text
large materials database
   -> pretrained CrabNet / ALIGNN / MLIP
   -> target material family
```

のような利用が該当します。

### Frozen representation

encoderを固定し

```math
z=h_{\theta_0}(x)
```

として、その上にGPを置く方法があります。

### Fine-tuning

parameterをtarget dataで更新し

```math
\theta_0\rightarrow\theta_{target}
```

とします。

少量データではfine-tuningによるoverfitやcatastrophic forgettingに注意します。

---

## 8. Transfer learningとResidual GP

pretrained modelを使う点では似ていますが、補正する場所が違います。

### Fine-tuning

```text
pretrained model parameters
       ↓ update
target-domain model
```

### Residual GP

```text
pretrained prediction ──┐
                       + -> final prediction
GP correction ──────────┘
```

Residual GPはpretrained model本体を変更しないため、baselineとtarget-domain correctionの責務を分離しやすい特徴があります。

---

## 9. Delta learningとの関係

量子化学・材料分野で使われるdelta learningも

```math
\Delta(x)=y_{high}(x)-y_{low}(x)
```

を学習し、

```math
\hat y_{high}(x)=y_{low}(x)+\hat\Delta(x)
```

とするため、Residual learningと非常に近い考え方です。

ただし「delta learning」という名称だけではuncertainty modelやsequential decisionまで意味しません。GPでdeltaをmodel化すればposterior uncertaintyをBO/ALへ接続できます。

---

## 10. 材料探索での典型例

### A. MLIPを実験値へ補正

```text
MLIP -> baseline
experiment - MLIP -> GP residual
```

→ Residual GP

### B. DFTと実験を両方使い、次にどちらを実行するか選ぶ

```text
candidate x
  × fidelity {DFT, experiment}
  × evaluation cost
```

→ Multi-fidelity BO

### C. 8つの関連propertyを同時に学習

→ Multi-task / multi-output GP

### D. 大規模事前学習modelを特定材料系へ適用

→ Transfer learning

### E. pretrained encoderを固定してGPを学習

→ Transfer representation + GP / DKL-style workflow

---

## 11. 組み合わせることもできる

これらは排他的ではありません。

例えば

```text
pretrained structure encoder
        ↓
shared latent representation
        ↓
multi-fidelity GP
        ↓
cost-aware acquisition
```

や

```text
pretrained MLIP baseline
        ↓
fidelity-specific residual GPs
        ↓
DFT / experiment selection
```

のような構成も考えられます。

ただし複雑化するほど、どのcomponentがperformance向上へ寄与したか分かりにくくなります。baselineから段階的に比較することが重要です。

---

## 12. bochanでの整理

bochanの設計では、少なくとも次を別軸として扱うと明確です。

```text
representation / pretrained model
model family
output/task structure
fidelity structure
residual correction
acquisition
workflow
```

例えば現在のMLIP residual workflowは

```text
MACE
  -> energy baseline
  -> residual GP
  -> relaxation
  -> acquisition
```

です。

これをmulti-fidelityへ拡張する場合は、単に`residual_gp`というmode名を変更するのではなく、fidelity variable、fidelity-aware posterior、cost model、fidelity-aware acquisitionという別の責務を追加する方が設計上明確です。

---

## 13. 選択表

| 問題 | 選ぶ考え方 |
|---|---|
| 良いpretrained predictionがあり少量target dataで補正したい | Residual GP |
| low/high accuracy dataをjointに使いたい | Multi-fidelity model |
| 次にどのaccuracy/costで評価するかも決めたい | Multi-fidelity BO |
| 関連task間でdataを共有したい | Multi-task GP |
| pretrained representationをtargetへ再利用したい | Transfer learning |
| low/high method差分を学習したい | Delta / residual learning |

次に実装を検討するときは、「複数情報源がある」だけでmulti-fidelityと呼ばず、**fidelityを意思決定変数として扱うか**を明確にすることが重要です。
