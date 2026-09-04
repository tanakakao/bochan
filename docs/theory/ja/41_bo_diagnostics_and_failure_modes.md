# 41. Bayesian Optimizationの診断と失敗モード

BOが提案するcandidateが不自然なとき、acquisitionだけを変更しても改善しないことがあります。BOは

```text
data -> surrogate -> posterior -> objective -> acquisition -> optimizer -> candidate
```

というpipelineなので、どの段階が失敗しているかを分離して診断します。

## 1. Data diagnosis

最初に確認するのは

- duplicate X
- missing / inf
- target scale
- input scale
- categorical encoding
- impossible combinations
- outliers
- leakage

です。model以前の問題をGP hyperparameterで吸収させないことが重要です。

## 2. Standardization / normalization

極端に異なるscaleはkernel fittingとacquisition optimizationを不安定にします。

```text
X -> normalized search domain
Y -> standardized output
```

を基本とし、prediction時のuntransformも確認します。

## 3. Lengthscale collapse

lengthscaleが極端に小さいと、training points間がほぼ無相関になりposteriorがspikyになります。極端に大きいとfunctionをほぼconstantとみなします。

確認項目は

- ARD lengthscales
- bounds / priors
- input scaling
- duplicate points
- noise estimate

です。

## 4. Noise collapse / inflation

noiseがほぼ0へcollapseするとtraining observationsへ過剰適合し、posterior uncertaintyが不自然になることがあります。逆にnoiseが大きすぎるとfunction structureを学べません。

repeated measurementsやknown `Yvar` があればnoise diagnosisに有用です。

## 5. Numerical conditioning

kernel matrixのconditionが悪い場合、Cholesky failureやjitter増加が起こります。

原因にはduplicate inputs、極端なlengthscale、scale mismatchなどがあります。

## 6. Posterior calibration

RMSEだけでなく、predictive interval coverageやstandardized residualを確認します。

```math
z_i=\frac{y_i-\mu_i}{\sigma_i}
```

posterior varianceが過小ならexploration不足、過大なら過剰探索につながります。

## 7. Extrapolation

GP uncertaintyが大きいboundaryへUCB等が集中することがあります。これは必ずしもbugではありません。

しかしbounds外挿、training support、physical feasibilityを確認し、不要ならsearch boundsやconstraintsを見直します。

## 8. Acquisition collapse

candidateが毎回同じ領域へ集中する場合、

- posterior varianceが小さすぎる
- exploration parameterが弱い
- best_f設定が不正
- objective directionが逆
- baseline/pending pointsが不正
- acquisition optimizerがlocal optimum

などを確認します。

## 9. Duplicate candidates

floating-point上は異なってもrounding後に同じcandidateになることがあります。

```text
optimize acquisition
 -> postprocess / round
 -> duplicate check
 -> refill / re-optimize
```

という処理が必要です。integer/categorical/composition探索では特に重要です。

## 10. Acquisition optimizer failure

surrogate posteriorが正常でも、`num_restarts`や`raw_samples`不足で悪いcandidateを返すことがあります。

有限poolなら全候補score比較がdiagnosticになります。continuous spaceでもrandom candidatesとのacquisition値比較が有効です。

## 11. Multi-objective failure

- objective directionの不一致
- reference pointがPareto frontより良い
- output scale差
- constraint transformの順序

を確認します。hypervolumeが増えない原因がmodelではなくreference pointということもあります。

## 12. Classification failure

accuracyだけでなくprobability calibrationを確認します。class imbalanceが強い場合、posterior probabilityが偏りPoFやentropy acquisitionが機能しないことがあります。

## 13. Robust BO failure

perturbation scaleが大きすぎると全candidateが似たrisk scoreになります。小さすぎるとnominal BOとほぼ同じになります。

`n_w`不足ではVaR/CVaR estimateがMonte Carlo noiseに支配される可能性があります。

## 14. Multi-fidelity failure

low fidelityとhigh fidelityのcorrelationが弱ければ、low fidelityを追加してもtarget posteriorが改善しません。costが安いという理由だけでlow fidelityを選ばないことが重要です。

## 15. Domain shift

pretrained MLIP/encoderを使う場合、target domainがpretraining domainから外れているとbaseline biasが大きくなります。

Residual GPで補正できる範囲か、baseline自体を変更すべきかをvalidationで判断します。

## 16. Sequential evaluation

BO modelの評価はrandom train/test splitだけでは不十分です。実際の目的はsequential regretやsample efficiencyだからです。

synthetic benchmarkやhistorical poolで

```text
initial data
 -> fit
 -> acquire
 -> reveal outcome
 -> refit
```

を再現し、best-so-far、hypervolume、boundary error等をiterationごとに比較します。

## 17. 診断順序

問題が起きたら次の順に確認します。

```text
1. data
2. transforms
3. model fit
4. posterior mean
5. posterior uncertainty
6. objective transform
7. acquisition values
8. acquisition optimizer
9. post-processing
10. sequential performance
```

acquisition名を変更するのはこの後です。

## 18. 最小baseline

複雑なmethodを評価するときも

```text
standard GP
+ simple EI/UCB
+ no exotic transform
```

を残します。SAAS、DKL、robust、lookahead、multi-fidelity等の追加componentが本当に改善へ寄与しているか比較できます。

BOの診断で最も重要なのは、**model accuracy・uncertainty・decision qualityを別々に確認すること**です。
