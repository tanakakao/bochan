# 34. Batch and Parallel Bayesian Optimization

When multiple experiments can run in parallel, BO chooses a batch `X` with `q>1` rather than a single point.

Joint q-acquisitions evaluate the utility of the entire batch and account for correlation between candidates. Sequential greedy batching selects one point at a time while conditioning on fantasies or pending points. The two strategies trade computational cost against joint optimality.

`X_pending` represents evaluations that have been submitted but are not yet observed. Ignoring pending evaluations can produce duplicates or highly redundant candidates.

Simple top-q ranking of a pointwise acquisition is generally not equivalent to optimizing a true q-acquisition.

In `bochan`, batch size, sequential versus joint optimization, pending-point handling, and duplicate/post-processing rules should be treated as explicit workflow configuration.