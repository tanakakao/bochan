# 11. Classification Models

This chapter distinguishes the latent Gaussian process, the likelihood, the
predictive class probabilities, and the uncertainty measures used by
classification acquisitions in `bochan`.

The central implementation warning is:

> A latent Gaussian posterior, a posterior distribution of class probability,
> and the categorical observation variance are different mathematical objects.

---

## 1. Binary Gaussian-process classification

### 1.1 Latent function and Bernoulli likelihood

Let

\[
f\sim\mathcal{GP}(m,k).
\]

For a binary label \(y\in\{0,1\}\), define

\[
p(y=1\mid f)=\pi(f),
\qquad
p(y=0\mid f)=1-\pi(f),
\]

where `pi` is a monotone inverse link.  Common choices are

\[
\pi(f)=\sigma(f)=\frac{1}{1+e^{-f}}
\]

for a logit model, or

\[
\pi(f)=\Phi(f)
\]

for a probit model.

GPyTorch's standard `BernoulliLikelihood` uses a probit-style construction.
Code and documentation should therefore avoid assuming that every binary model
uses a logistic sigmoid merely because the output is a probability.

The class probability conditioned on the dataset is

\[
p(y=1\mid x,\mathcal D)
=
\int \pi(f)\,p(f\mid x,\mathcal D)\,df.
\]

This integral is not generally equal to

\[
\pi\!\left(\mathbb E[f\mid x,\mathcal D]\right).
\]

The latter is a plug-in approximation that ignores latent uncertainty.

### 1.2 Variational posterior

The binary models in `bochan` use a sparse variational latent GP.  With inducing
variables \(u=f(Z)\), training maximizes

\[
\mathcal L
=
\sum_{i=1}^n
\mathbb E_{q(f_i)}[\log p(y_i\mid f_i)]
-
\operatorname{KL}[q(u)\|p(u)].
\]

The variational expectation is evaluated through GPyTorch's likelihood and
quadrature machinery.

### 1.3 Implementation contract

The relevant public models are

```text
BinaryClassificationGPModel
BinaryClassificationMixedGPModel
```

in

```text
src/bochan/models/classification/binary/base/models.py
```

Their contract is intentionally asymmetric:

```python
probability_posterior = model.posterior(X)
latent_posterior = model.latent_posterior(X)
```

`posterior(X)` applies the Bernoulli likelihood and returns a
`SimpleBernoulliPosterior`.  Its mean is the class-1 probability

\[
\mu_p(x)=p(y=1\mid x,\mathcal D).
\]

`latent_posterior(X)` bypasses the likelihood and returns the Gaussian posterior
of \(f(x)\).

This means the default binary boundary is written in two equivalent model
spaces only under a symmetric monotone link:

\[
f(x)=0
\quad\Longleftrightarrow\quad
p(y=1\mid x)=0.5.
\]

For a different probability threshold \(\tau_p\), the corresponding latent
threshold is

\[
\tau_f=\pi^{-1}(\tau_p).
\]

### 1.4 Bernoulli variance is not epistemic variance

For a Bernoulli observation with class probability \(p\),

\[
\operatorname{Var}(Y\mid p)=p(1-p).
\]

This is largest at `p=0.5`, even when the model knows `p` exactly.  It is an
observation variance caused by the discrete label distribution.

Epistemic uncertainty concerns uncertainty in the probability itself:

\[
\operatorname{Var}_{p(f\mid\mathcal D)}[\pi(f)].
\]

These quantities answer different questions:

| Quantity | Interpretation |
|---|---|
| \(p(1-p)\) | ambiguity of the next binary observation |
| \(\operatorname{Var}[\pi(f)]\) | uncertainty about the probability function |
| \(\operatorname{Var}[f]\) | uncertainty about the latent decision function |

`SimpleBernoulliPosterior.variance` is primarily a Bernoulli predictive
variance.  It should not automatically be used as a calibrated epistemic
uncertainty measure.

---

## 2. Binary active learning

### 2.1 Predictive entropy

For

\[
p=p(y=1\mid x,\mathcal D),
\]

the Bernoulli predictive entropy is

\[
H[Y\mid x,\mathcal D]
=-p\log p-(1-p)\log(1-p).
\]

Entropy sampling selects points where the predicted label is ambiguous.  It
cannot distinguish aleatoric ambiguity from model uncertainty.

### 2.2 BALD

Bayesian Active Learning by Disagreement uses mutual information between the
future label and model parameters or latent function:

\[
\operatorname{BALD}(x)
=
I(Y;f\mid x,\mathcal D).
\]

Using the entropy identity,

\[
I(Y;f\mid x,\mathcal D)
=
H[Y\mid x,\mathcal D]
-
\mathbb E_{p(f\mid x,\mathcal D)}
  [H[Y\mid f,x]].
\]

The first term is total predictive uncertainty.  The second is expected
conditional label noise.  Their difference emphasizes reducible model
uncertainty.

A Monte Carlo approximation is

\[
\widehat I
=
H\!\left(\frac1S\sum_{s=1}^S p_s\right)
-
\frac1S\sum_{s=1}^S H(p_s),
\]

where

\[
p_s=\pi(f^{(s)}),
\qquad
f^{(s)}\sim p(f\mid x,\mathcal D).
\]

### 2.3 Margin uncertainty

For binary probability,

\[
M(x)=1-|2p(x)-1|.
\]

For multiclass probabilities sorted as

\[
p_{(1)}\ge p_{(2)}\ge\cdots,
\]

a common margin score is

\[
M(x)=1-[p_{(1)}-p_{(2)}].
\]

Margin sampling is computationally cheap but does not explicitly represent
posterior epistemic uncertainty.

---

## 3. Multiclass GP classification

### 3.1 Class-wise latent functions

For `K` unordered classes, use one latent function per class:

\[
f_k\sim\mathcal{GP}(m_k,k_k),
\qquad k=1,\ldots,K.
\]

The categorical probabilities are obtained by a softmax:

\[
p(y=k\mid\mathbf f)
=
\frac{\exp(f_k/T)}
{\sum_{j=1}^{K}\exp(f_j/T)},
\]

where `T` is an optional temperature.

The model is invariant to adding the same constant to all logits:

\[
\operatorname{softmax}(\mathbf f+c\mathbf 1)
=
\operatorname{softmax}(\mathbf f).
\]

Consequently, absolute latent levels are not identifiable; differences between
class logits determine the prediction.

### 3.2 Variational multiclass model

`MulticlassClassificationGPModel` uses a class-batched sparse variational GP.
The latent batch shape is `[num_classes]`.  For an input tensor

```text
batch_shape x q x d
```

the implementation inserts a singleton class-batch axis before evaluating the
latent model so that GPyTorch can broadcast the class-wise inducing-point batch.

The public methods are

```python
latent = model.latent_posterior(X)
probability = model.posterior(X)
probs = model.class_probs(X)
predicted_class = model.predict_class(X)
```

`model.posterior(X)` wraps the latent posterior in
`MulticlassProbsPosterior`.  Its mean has shape

```text
batch_shape x q x K
```

and represents marginal class probabilities.

Main implementation:

```text
src/bochan/models/classification/multiclass/base/models.py
src/bochan/models/components/multiclass.py
```

### 3.3 Predictive entropy

For a categorical probability vector \(\mathbf p\),

\[
H(Y\mid x,\mathcal D)
=-\sum_{k=1}^Kp_k\log p_k.
\]

The maximum entropy is \(\log K\), attained by the uniform distribution.
Normalized entropy may be defined by

\[
H_{\mathrm{norm}}=\frac{H}{\log K}.
\]

### 3.4 Multiclass BALD

The multiclass mutual information is

\[
I(Y;\mathbf f\mid x,\mathcal D)
=
H\!\left(\mathbb E_{\mathbf f}[\mathbf p(\mathbf f)]\right)
-
\mathbb E_{\mathbf f}[H(\mathbf p(\mathbf f))].
\]

The same distinction between total ambiguity and reducible uncertainty applies
as in the binary case.

---

## 4. Mixed continuous and categorical inputs

For an input decomposed as

\[
x=(x_{\mathrm{cont}},x_{\mathrm{cat}}),
\]

the mixed kernels in the classification implementations follow the pattern

\[
k(x,x')
=
k_c+k_g+k_c'k_g'.
\]

This contains additive continuous and categorical similarity plus an interaction
term.  Category integers are identifiers, not quantities on an interval.
Therefore:

- do not normalize category codes as continuous variables;
- do not interpret a category difference of two as twice a difference of one;
- use fixed-feature enumeration or an optimizer designed for mixed spaces;
- preserve categorical columns under `input_transform`.

`bochan` explicitly checks that mixed-model transforms do not modify categorical
columns.

---

## 5. Classification Bayesian optimization

Classification BO converts class probabilities into a scalar objective.

### 5.1 Target-class probability

For target class `k*`,

\[
u(x)=p(y=k^*\mid x,\mathcal D).
\]

A probability-space EI-style acquisition uses improvement over a probability
reference `best_f`:

\[
I(x)=\max(u(x)-u_{\mathrm{best}},0).
\]

The reference and objective must both be in probability space.  A latent
`best_f` must not be mixed with probability samples.

### 5.2 Expected class utility

For utilities \(u_1,\ldots,u_K\),

\[
U(x)=\sum_{k=1}^K u_kp_k(x).
\]

This is useful when classes have known economic or engineering values.  It is
not ordinal modeling unless the likelihood itself represents the order; an
unordered multiclass model with monotone utilities still estimates the class
probabilities independently of order.

### 5.3 Probability of feasibility

A classification output can define a feasibility multiplier:

\[
\alpha_{\mathrm{constrained}}(x)
=
\alpha_{\mathrm{objective}}(x)
\prod_{j=1}^{m_c}p_j(\mathrm{feasible}\mid x).
\]

This is a constrained optimization role, not Level-set Estimation.  The same
classifier can be reused, but the acquisition objective differs.

---

## 6. Heteroscedastic classification caveat

A binary or multiclass response is already stochastic.  A heteroscedastic
classification extension must define what its second noise process represents.
Possible interpretations include:

1. label corruption probability;
2. annotator or measurement reliability;
3. local temperature or dispersion of the link;
4. uncertainty attached to an externally estimated class probability;
5. a heuristic penalty model used by an acquisition.

These interpretations are not equivalent.  Merely adding a predicted variance
to \(p(1-p)\) does not define a generative heteroscedastic Bernoulli model.

The current `bochan` robust classification wrappers expose a `noise_model` and
may add predicted noise to a probability-posterior variance.  Documentation and
acquisitions should call this an engineering noise convention unless the
likelihood explicitly contains the noise process.

Relevant locations:

```text
src/bochan/models/classification/binary/robust/
src/bochan/models/classification/multiclass/robust/
src/bochan/models/components/heteroscedastic.py
```

---

## 7. Acquisition-space selection

| Goal | Recommended model output |
|---|---|
| Find the latent decision boundary | `latent_posterior(X)` |
| Maximize probability of one class | probability posterior |
| Predict the next observed class | probability posterior |
| Entropy sampling | class probabilities |
| BALD | latent posterior samples transformed through likelihood |
| Feasibility weighting | probability posterior |
| Measure reducible model uncertainty | BALD or probability posterior variance, not only Bernoulli variance |

---

## 8. Source map

| Component | Implementation |
|---|---|
| Binary latent and probability wrapper | `src/bochan/models/classification/binary/base/models.py` |
| Binary posterior object | `src/bochan/posteriors/bernoulli.py` |
| Multiclass latent and probability wrapper | `src/bochan/models/classification/multiclass/base/models.py` |
| Multiclass posterior helper | `src/bochan/models/components/multiclass.py` |
| Binary acquisitions | `src/bochan/acquisition/binary/` |
| Multiclass acquisitions | `src/bochan/acquisition/multiclass/` |
| Classification posterior transforms | `src/bochan/models/transforms/posterior/classification.py` |

---

## 9. References

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- Houlsby et al., *Bayesian Active Learning for Classification and Preference Learning*, 2011.
