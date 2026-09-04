# 40. 高次元Bayesian Optimization

入力次元が増えると、GP fittingだけでなくacquisition optimizationも難しくなります。高次元BOでは「本当に全dimensionが同程度に重要か」を考えることが中心です。

## 1. Curse of dimensionality

探索空間 `[0,1]^d` ではdimension増加に伴いdataが極端に疎になります。少量dataからglobal functionを学ぶGPの仮定が弱くなり、acquisition optimizationもlocal optimumを持ちやすくなります。

## 2. ARD

dimension別lengthscale

```math
\ell_1,\ldots,\ell_d
```

を学習し、感度の低いdimensionを長いlengthscaleとして表現できます。

ただし `n << d` ではlengthscale自体の推定が不安定になります。

## 3. SAAS

SAASはinverse lengthscaleの多くを0付近へshrinkし、effective dimensionが低いというpriorを利用します。

高次元small-data BOの有力候補ですが、fully Bayesian inferenceの計算costがあります。

## 4. Trust region

global domain全体を一度に探索せず、有望領域周辺のlocal trust regionでBOを行います。

```text
successful improvements -> expand region
failures                 -> shrink region
```

という適応的local searchにより、高次元acquisition optimizationを扱いやすくします。

## 5. Dimensionality reduction

PCA等のunsupervised projectionは入力varianceを保存しますが、objectiveに重要なdirectionを保存する保証はありません。

supervised embedding、active subspace、learned encoderを使う場合も、projection uncertaintyやinverse mappingを考慮します。

## 6. DKL

```math
z=h_\theta(x),\qquad f(z)\sim GP
```

として高次元raw inputを低次元latent representationへ変換できます。

image、spectrum、composition descriptor、structure embeddingなどで有効ですが、small dataではencoder overfitに注意します。

## 7. Feature selection

domain knowledgeで不要variableを除くことは非常に強いbaselineです。BOの前にvariable数を減らせるなら、複雑な高次元methodより安定することがあります。

ただし過去dataで変化していないvariableを「影響なし」と断定しないよう注意します。

## 8. Structured high dimension

高次元でも構造がある場合は、それを利用します。

```text
composition block
process block
structure embedding
categorical block
```

ごとにkernel/encoderを分けるadditive・product構造も検討できます。

## 9. Acquisition optimization

高次元ではacquisition landscape自体のoptimizationが失敗することがあります。`raw_samples`、`num_restarts`、bounds、initialization、discrete candidate poolなどを診断します。

surrogateが良くてもacquisition optimizerが悪ければBO全体は失敗します。

## 10. 推奨導入順

```text
1. domain-based variable reduction
2. standard GP + ARD
3. SAAS
4. trust-region approach
5. pretrained/frozen representation + GP
6. DKL / learned representation
```

を比較するのが実務的です。

高次元BOでは「より複雑なmodel」より、**effective search dimensionをどう小さくするか**が主要な設計問題です。
