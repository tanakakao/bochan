# 36. Information-Theoretic Bayesian Optimization

Information-theoretic BO selects points that reduce uncertainty about an optimization target such as the optimum value, optimizer location, or Pareto set.

MES targets information about the maximum value. PES targets information about the optimizer. JES uses joint information about optimum value and location. Multi-objective entropy-search variants target uncertainty about the Pareto set/front.

These methods differ from ordinary Active Learning because they reduce uncertainty about the optimization solution, not necessarily the full surrogate function.

Compared with EI, information-theoretic acquisitions can favor measurements that are valuable for resolving where the optimum is even when immediate improvement is unlikely. Their cost and approximation complexity are correspondingly higher.

`bochan` keeps these acquisitions distinct from generic entropy/BALD AL criteria.