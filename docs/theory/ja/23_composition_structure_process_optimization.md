# 23. 組成・構造・プロセス条件の統合最適化

実際の材料特性は材料組成だけで決まるとは限りません。焼成温度、圧力、保持時間、雰囲気、加工条件などによって、同じ組成でも異なる構造・組織・物性が得られます。

したがって材料開発の最適化問題は

```math
x=(c,s,p)
```

として、組成 `c`、構造 `s`、プロセス条件 `p` を区別して考えると整理しやすくなります。

## 1. 目的関数

最も単純には

```math
y=f(c,s,p)+\epsilon
```

です。しかし実際にはprocessがstructureを変化させるため、因果的な生成過程は

```text
composition + process
        ↓
realized structure / microstructure
        ↓
property
```

に近い場合があります。

この違いは重要です。単純な予測モデルで `(c,p)` から `y` を予測できても、それだけでprocessが物性へ直接作用すると結論付けることはできません。

## 2. 混合探索空間

組成・構造・processを同時に扱うと、探索空間はheterogeneousになります。

- 元素選択: discrete/combinatorial
- 組成比: constrained continuous
- structure index: discrete
- 温度・圧力・時間: continuous
- 装置・雰囲気・原料種: categorical

したがって単純なbox constraint

```math
\mathcal X=[l_1,u_1]\times\cdots\times[l_d,u_d]
```

だけでは表現できません。

## 3. 条件付き探索空間

変数間には条件付き制約もあります。例えば元素 `A` を選択していない候補では、その元素の組成比は必ず0です。

```math
z_j=0\Rightarrow a_j=0
```

ここで `z_j` は元素選択indicatorです。

また特定のprocess routeを選んだ場合だけ有効になる温度や時間もあります。このようなhierarchical/conditional spaceでは、候補生成後のpost-processingだけでなく、候補表現そのものに制約を組み込む方が効率的です。

## 4. encoder + process variables

材料表現 `z_m` とprocess変数 `p` を結合し、

```math
z=[z_m,p]
```

としてGP/DKLへ入力できます。

例えばcomposition encoderなら

```math
z_m=h_{\mathrm{CrabNet}}(c)
```

として

```math
f(c,p)=g([h_{\mathrm{CrabNet}}(c),p])
```

を学習できます。

structure encoderでも同様です。

この方式は、材料表現の高次元な特徴抽出をencoderへ任せながら、process条件を明示的な設計変数としてBOへ残せる点に利点があります。

## 5. structure index + process

有限個の候補構造がある場合は、構造bankのindexとprocess条件を組み合わせることもできます。

```text
X = [structure_index, temperature, pressure, time, ...]
```

structure indexは内部で実構造へ解決し、MLIPやstructure encoderが物理的baseline/representationを提供します。

これはPhase 15以降のbochanのrelaxation + acquisition設計と整合します。

## 6. 多目的化

材料開発では単一物性最大化より、複数目的を同時に扱う場合が多くあります。

```math
\mathbf y(x)=
(y_{\mathrm{performance}},
 y_{\mathrm{cost}},
 y_{\mathrm{stability}},
 y_{\mathrm{processability}})
```

すべてを単一scalarへ固定すると選好を早い段階で決めすぎるため、Pareto optimizationが有効です。

制約付きなら

```math
\max_x \mathbf f(x)
\quad\text{s.t.}\quad
c_j(x)\le0
```

として、安全性、製造可能性、元素使用量、コストなどをconstraintとして分離できます。

## 7. Robust optimization

製造条件にはばらつきがあります。nominal条件 `x` の性能だけでなく、摂動

```math
\tilde x=x+\xi
```

に対する期待性能や下側riskを最適化することがあります。

```math
\max_x \mathbb E_\xi[f(x+\xi)]
```

またはCVaRなどを使えば、工程ばらつきに弱い鋭いoptimumより、再現性の高い条件を選択できます。

## 8. 実験への接続

最終的な閉ループは

```text
candidate generation
  -> composition / structure / process candidate
  -> feasibility filtering
  -> surrogate posterior
  -> acquisition
  -> experiment or simulation
  -> observation
  -> model update
```

です。

bochanの役割は、この意思決定loopに必要なsurrogate、acquisition、constraints、candidate selectionを統一的に扱うことです。材料生成そのものや実験装置制御は別componentとして分離しておく方が、責務が明確になります。
