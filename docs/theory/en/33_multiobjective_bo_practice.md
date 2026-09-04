# 33. Multi-Objective Bayesian Optimization in Practice

Multi-objective BO searches for Pareto-efficient trade-offs rather than a single optimum. A point is Pareto dominated when another candidate is no worse in every objective and better in at least one.

Hypervolume measures the volume dominated by a Pareto set relative to a reference point. EHVI maximizes expected hypervolume improvement; NEHVI handles noisy observations more naturally. NParEGO converts the vector objective into randomized scalarized subproblems.

Reference points must live in the same transformed objective space as the acquisition. Objective scaling matters because poorly scaled outputs can distort scalarization and hypervolume geometry.

Constraints can be combined with multi-objective acquisitions by computing utility only for feasible outcomes.

For `bochan`, multi-output posterior construction, objective transformation, reference-point definition, and acquisition selection are separate responsibilities.