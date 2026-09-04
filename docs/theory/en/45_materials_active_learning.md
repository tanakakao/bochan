# 45. Materials Active Learning

Materials Active Learning allocates expensive DFT or experimental labels to improve a surrogate or potential efficiently.

For GP regression, posterior variance is a simple baseline. Classification tasks can use predictive entropy or boundary criteria. BALD emphasizes epistemic information gain. NIPV-like criteria value global reduction of posterior uncertainty.

MLIP Active Learning is a related but distinct task: structure/MD configurations are selected for high-fidelity labeling to expand the reliable domain of an interatomic potential. Force/stress uncertainty and configuration diversity may matter more than a property-optimization objective.

Residual-model AL can target regions where the correction to a pretrained baseline is uncertain.

A practical campaign may shift from exploration-heavy AL early to optimization-heavy BO later, or split a batch between both goals.

Stopping rules should consider uncertainty or marginal information gain, not only a fixed data count.