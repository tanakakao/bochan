# 20. Structure Relaxation, Bayesian Optimization, and Active Learning

Structure relaxation and Bayesian optimization solve different problems. Relaxation performs local optimization of a supplied structure, typically minimizing energy until the force norm is sufficiently small. BO decides which candidate should receive an expensive evaluation.

## Relaxation

Given an initial structure `S_0`, optimizers such as FIRE, BFGS, or LBFGS update atomic coordinates and optionally the cell using MLIP energies and forces.

## Relax then rank

For a manageable finite bank, relax every candidate and rank the relaxed structures by predicted score.

## Relax then acquire

When DFT or experiment is expensive, relaxed structures can feed a probabilistic surrogate and acquisition function. Posterior-mean ranking is exploitation; EI/UCB and AL variance/entropy/BALD/NIPV explicitly use uncertainty.

## bochan perspective

The material workflow layer separates relaxation, posterior modeling, and candidate selection. This keeps a physics optimization step distinct from the sequential decision problem.