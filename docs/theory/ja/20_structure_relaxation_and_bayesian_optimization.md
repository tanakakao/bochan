# 20. 構造緩和・結晶構造BO・Active Learning

本章では、MLIPによるstructure relaxationと、GPによるBayesian Optimization / Active Learningを区別したうえで接続します。

## 1. Structure relaxation

初期構造 `S_0` から局所的にenergyの低い構造を求めます。

原子座標のみなら

```math
R^*=\arg\min_R E(R;L,Z)
```

cellも緩和する場合は

```math
(R^*,L^*)=\arg\min_{R,L}E(R,L;Z)
```

です。

forceは

```math
F_i=-\nabla_{r_i}E
```

なので、optimizerはforceを利用してenergy surface上を移動します。

## 2. 収束条件 `fmax`

典型的には最大原子force

```math
F_{max}=\max_i\lVert F_i\rVert
```

がthresholdより小さくなれば収束とみなします。

```math
F_{max}<f_{max}
```

`fmax`を小さくすると厳密な緩和になりますが、計算stepは増える傾向があります。

## 3. FIRE / BFGS / LBFGS

bochanの共通relaxation APIではFIRE、BFGS、LBFGSを選べます。

### FIRE

forceに基づくdynamics型optimizerで、atomistic relaxationで広く使われます。初期構造がminimumから離れている場合にも扱いやすい選択肢です。

### BFGS

energy landscapeのcurvatureを逆Hessian近似として蓄積するquasi-Newton法です。

### LBFGS

BFGSの履歴を限定してmemory costを抑えます。自由度の大きい系で有利になる場合があります。

optimizerの優劣は構造・potential・初期条件に依存するため、名前だけで収束性能を保証することはできません。

## 4. RelaxationとBayesian Optimizationは別問題

ここが最も重要です。

Relaxation:

```math
S_0\rightarrow S_{relaxed}
```

は**1つの初期構造の局所最適化**です。

Bayesian Optimization:

```math
x_{next}=\arg\max_x\alpha(x)
```

は**複数候補の中から次に高価な評価を行う点を選択する逐次意思決定**です。

したがって「MLIPでrelaxした」だけではBOではありません。

## 5. Relax -> Rank

最も単純な構造探索は

```text
initial structures
    -> MLIP relaxation
    -> relaxed structure bank
    -> surrogate prediction
    -> rank
```

です。

posterior meanで順位付けするなら、最大化問題では

```math
S^*=\arg\max_{S_i}\mu(S_i)
```

となります。

不確かさも含めるならUCB型rankingとして

```math
score(S_i)=\mu(S_i)+\sqrt\beta\sigma(S_i)
```

を使えます。

## 6. Relax -> Acquisition

BO/ALでは単なる順位ではなくacquisitionを評価します。

```text
initial structures
    -> relax all candidates
    -> rebuild structure bank
    -> fit / bind probabilistic model
    -> acquisition value
    -> select next q structures
```

重要なのは、**relax後の構造順序とstructure indexを一致させること**です。bochanのgeneric relaxation selectorは最終relaxed bankをfactoryへ渡し、その順序を `0..n-1` のindex contractとして使います。

## 7. BOの選択基準

良い目的値を探す場合、例えばEIは

```math
\alpha_{EI}(x)=\mathbb E[(f(x)-f^*)_+]
```

で改善量を評価します。

UCBは

```math
\alpha_{UCB}(x)=\mu(x)+\sqrt\beta\sigma(x)
```

としてexploitationとexplorationを調整します。

noiseがある場合にはNEI系を選ぶなど、通常のBOと同じ判断が必要です。

## 8. Active Learningの選択基準

モデル改善が目的なら、最大variance

```math
x_{next}=\arg\max_x\sigma^2(x)
```

のようなuncertainty samplingが使えます。

bochanのstructure acquisition workflowではvarianceに加えてpredictive entropy、BALD、NIPVなどを利用できます。これらは「最も高性能そうな構造」とは異なる構造を選ぶ場合があります。

## 9. プロセス条件を含む探索

各構造にprocess variable `p` がある場合、候補を

```math
x=[i_{structure},p]
```

として表せます。

すると探索は

```math
(i^*,p^*)=\arg\max_{i,p}\alpha(i,p)
```

となり、「どの構造か」と「どの条件で評価するか」を同時に選択できます。

これは組成・構造・製造条件を接続する材料探索への入口です。

## 10. 探索loop

実験/DFTを含む完全なloopは次のようになります。

```text
candidate structures
      |
      v
MLIP relaxation
      |
      v
probabilistic surrogate
      |
      v
BO / AL acquisition
      |
      v
selected structures
      |
      v
DFT / experiment
      |
      v
new observations
      |
      +------> refit residual GP ------+
```

MLIPは高価な評価を完全に置き換える必要はありません。高速なphysics-informed baselineとして候補を整形・評価し、高価なground-truthをどこへ使うかをGPとacquisitionが決める、という分業ができます。

## 11. bochan workflowとの対応

4軸のidentityは

```text
backend -> quantity -> model_mode -> workflow_mode
```

です。

例:

```text
mace -> energy -> residual_gp -> relax_acquisition
```

これは「MACEで構造を扱い、energyを対象とし、MLIP baselineをResidual GPで補正し、relax後の候補をacquisitionで選択する」ことを意味します。

実行API・capability discovery・設定値は [Unified MLIP workflows](../../materials/mlip-workflows.md) を参照してください。

## 12. 実務上の注意

- pretrained MLIPの適用範囲を確認する
- relaxationの収束と物性予測精度を混同しない
- force residualでは固定原子数contractを守る
- uncertaintyがMLIP自身の完全な不確かさとは限らない
- BO目的とAL目的を明確に分ける
- DFT/実験値を追加したらresidual modelを更新する
- cell relaxationではstress conventionと単位を確認する

この分離を保つことで、MLIP・GP・BO/ALをそれぞれ交換可能な層として扱えます。
