# 38. Level-set Estimation / Boundary Search 実践編

Level-set Estimation (LSE) は最大値を探すのではなく、functionがthresholdを超える領域を同定する問題です。本章は05章を補完し、工程窓・合否境界探索への使い方を整理します。

## 1. 問題設定

threshold `h` に対して

```math
S_+=\{x:f(x)\ge h\},\qquad S_-=\{x:f(x)<h\}
```

を識別します。

BOの

```math
\arg\max_x f(x)
```

とは目的が異なります。

## 2. Boundary uncertainty

posterior mean `mu(x)` がthresholdに近く、posterior std `sigma(x)` が大きい点ほどboundary同定に価値があります。

## 3. Straddle

代表的なscoreは

```math
\alpha(x)=\beta\sigma(x)-|\mu(x)-h|
```

です。thresholdへの近さとuncertaintyを同時に評価します。

## 4. Probability-based boundary search

```math
P(f(x)\ge h\mid D)
```

が0.5付近の点はclass membershipが曖昧です。binary entropyやmargin uncertaintyと同じ直感へつながります。

## 5. Integrated Classification Uncertainty

1点観測した後にdomain全体のclassification uncertaintyがどれだけ減るかを評価する考え方です。局所的に曖昧な点だけでなくglobal boundary reconstructionを目的にできます。

## 6. Boundary variance

threshold周辺へweightを置いてposterior varianceを評価することで、境界に関係しない高variance領域への探索を抑えます。

## 7. Classification ALとの違い

binary classifierのdecision boundaryを学ぶ場合とLSEは非常に近いですが、LSEではunderlying continuous latent/function thresholdが明示される場合があります。

## 8. Safe region

安全条件が `f(x)>=h` なら、posterior lower confidence boundを使ってconservative safe setを構築する考え方もあります。

ただし単なるLSEと、安全性を保証しながら探索するSafe BOは同一ではありません。

## 9. 工程窓

製造では最大性能点より

```text
temperature × pressure × time
```

のどの領域が規格を満たすか知りたいことがあります。LSEはこの「process window」を少ない実験で推定する目的に適します。

## 10. bochanでの位置付け

bochanでは

```text
Bayesian Optimization -> optimum
Active Learning       -> model uncertainty
Level-set Estimation  -> threshold boundary
```

を別task familyとして維持することが重要です。

LSEの成功指標もbest observed valueではなく、boundary error、set classification accuracy、uncertain volumeなどで評価する方が自然です。
