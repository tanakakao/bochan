# 21. 組成モデル：CrabNet・Roostと組成空間

材料探索では、結晶構造が未確定でも組成式だけが候補として与えられることがあります。この段階では、原子座標を要求するMLIPではなく、組成を直接表現するモデルが有効です。

## 1. 組成を入力とする問題

組成を元素集合と比率

```math
c=\{(e_j, a_j)\}_{j=1}^{m},\qquad a_j\ge0,\quad \sum_j a_j=1
```

として表します。目的物性を `y` とすれば、組成モデルは

```math
y=f_{\mathrm{comp}}(c)+\epsilon
```

を学習します。

組成式だけでは結晶多形、欠陥、原子配置を区別できないため、これは構造モデルより情報量の少ない表現です。一方、候補構造をまだ持たない初期スクリーニングには適しています。

## 2. CrabNet

CrabNetは元素をtokenとして扱い、元素埋め込みと組成比情報からattentionを用いて組成全体の表現を形成します。重要なのは、固定長の手作業descriptorだけに依存せず、元素間の関係をデータから学習できる点です。

概念的には

```text
composition
  -> element embeddings
  -> fractional representation
  -> self-attention
  -> composition embedding
  -> property prediction
```

となります。

bochanでCrabNet表現をGP/DKLへ接続する場合、encoderを特徴抽出器として利用し、その潜在表現 `z(c)` に対して

```math
f(c)=g(z(c)),\qquad g\sim\mathcal{GP}
```

と考えると整理できます。GPを後段に置く目的は、単なるpoint predictionだけでなくposterior uncertaintyをBO/ALへ供給することです。

## 3. Roost

Roostも組成のみから物性を予測しますが、組成を元素nodeからなるgraphとして表し、message passingによって材料表現を形成する考え方を取ります。

```text
elements + stoichiometry
  -> composition graph
  -> message passing
  -> material embedding
  -> property
```

CrabNetとRoostはどちらも「構造を必要としない」という意味で同じ探索段階に置けますが、内部表現の帰納バイアスが異なります。

## 4. 組成比の制約

組成は通常の独立な連続変数ではありません。

```math
\mathbf a=(a_1,\ldots,a_m),\qquad \sum_j a_j=1
```

なのでsimplex上に存在します。このclosure constraintを無視して各成分を独立に最適化すると、物理的に無効な候補が生成されます。

CLRやILRなどのlog-ratio座標は組成データ解析に有用ですが、ゼロ成分の扱いと逆変換後の制約維持が必要です。元素そのものを選ぶ問題では、連続比率最適化とは別に組合せ探索が必要です。

## 5. 元素選択は組合せ最適化

候補元素集合 `E` から `k` 元素を選ぶ場合、探索空間は

```math
\mathcal S_k=\{S\subseteq E:|S|=k\}
```

です。その後、選ばれた元素について組成比を探索できます。

```text
candidate elements
  -> subset selection
  -> composition-ratio optimization
  -> surrogate prediction
  -> acquisition
```

この階層を明示すると、「どの元素を使うか」と「何%使うか」を同じ連続変数として無理に扱う必要がありません。

## 6. 組成モデルを使う段階

| 利用可能な情報 | 適したモデル |
|---|---|
| 組成式のみ | CrabNet / Roostなど |
| 組成式 + プロセス条件 | composition encoder + process variables |
| 結晶構造あり | ALIGNNなどのstructure model |
| 原子構造 + energy/force/stress | MLIP |

組成モデルとMLIPは競合するものではなく、材料開発の異なる段階を担当します。

## 7. bochanでの位置づけ

bochanでは組成表現を予測器として使うだけでなく、GP/DKLと組み合わせてBO/ALへ接続することが重要です。また組成に温度、時間、圧力などのプロセス条件を連結すれば、材料そのものと製造条件を同時に探索できます。

次章では、組成しか分からない段階から結晶構造が得られた段階へ、モデルをどう切り替えるかを整理します。
