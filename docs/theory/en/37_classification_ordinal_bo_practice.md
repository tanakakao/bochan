# 37. Classification and Ordinal BO in Practice

Classification and ordinal BO must distinguish latent GP scores from decision-space probabilities or utilities.

Binary BO may optimize success probability, constrained improvement, or another utility derived from Bernoulli predictions. Multiclass BO may optimize the probability or utility of selected classes. Ordinal BO uses ordered class probabilities and can map categories to a scalar expected utility when appropriate.

Applying regression EI directly to latent logits is generally not equivalent to optimizing expected success or ordered utility.

Multi-objective EHVI/NEHVI/NParEGO extensions require every output to be transformed into a common objective representation first.

Calibration is especially important because acquisition values can be highly sensitive to overconfident class probabilities.

`bochan` maintains separate binary, multiclass, and ordinal acquisition families to preserve these contracts.