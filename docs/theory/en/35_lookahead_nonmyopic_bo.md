# 35. Lookahead and Non-Myopic Bayesian Optimization

Most common acquisitions are myopic: they value the immediate next evaluation. Non-myopic BO values how an observation changes future decisions.

Knowledge Gradient measures expected improvement in the best achievable posterior decision after observing a candidate. Multi-step lookahead extends this idea through future fantasy observations and nested optimization.

Lookahead is valuable when an exploratory observation may unlock better later decisions, but computational cost rises rapidly with horizon, fantasy count, batch size, and acquisition optimization complexity.

The practical default remains myopic EI/NEI/UCB unless the evaluation cost is sufficiently high to justify more expensive planning.

`bochan` exposes KG and lookahead-oriented components separately so users can opt into the additional decision depth rather than paying for it by default.