# 28. Response Distribution and Noise Model Selection

The likelihood should reflect the response support and observation mechanism.

Gaussian regression suits approximately continuous unconstrained responses. Bernoulli handles binary labels; categorical/softmax handles multiclass labels; ordered likelihoods handle ordinal categories. Poisson or Negative Binomial models suit counts. Beta distributions suit continuous responses in `(0,1)`, while Gamma models suit positive continuous responses.

Noise structure is a separate question. Homoscedastic noise assumes constant variance; heteroscedastic models allow input-dependent variance. Epistemic uncertainty from limited data should not be confused with aleatoric observation variability.

Transformed Gaussian models can be simpler than dedicated non-Gaussian likelihoods, but they encode different assumptions and may distort uncertainty.

`bochan` maps response families to matching BO/AL acquisition implementations so decision criteria are evaluated in the proper response space.