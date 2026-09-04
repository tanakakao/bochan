# 40. High-Dimensional Bayesian Optimization

GP-based BO becomes difficult as input dimension grows because data sparsity weakens length-scale estimation and acquisition optimization becomes harder.

ARD can identify relevant dimensions but requires enough data. SAAS priors encode the belief that only a small subset of dimensions matters. Projection methods such as PCA/REMBO reduce the search dimension under stronger structural assumptions. DKL learns nonlinear representations but introduces representation-training risk.

Trust-region methods can focus search locally when a global high-dimensional model is unreliable.

Feature selection should be evaluated through sequential BO performance, not only predictive cross-validation.

For materials/process problems, pretrained representations can reduce raw dimensionality, but uncertainty calibration in the resulting latent space remains essential.