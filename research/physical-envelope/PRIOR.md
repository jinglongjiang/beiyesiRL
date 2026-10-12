# Prior and Scope Before Outcomes

Generic local neural repair is prior art, not our contribution. SARP official repository was inspected (MIT, examples and utility source); it uses expert policy fitting and predictive-property repair. REASSURE explicitly targets continuous piecewise-linear networks and localized patch guarantees, whereas this frozen GRU uses gates and normalization. Neither full baseline was reproduced. No code or unsupported guarantee is borrowed from either.

Sources: https://github.com/k1majd/SARP and https://github.com/BU-DEPEND-Lab/REASSURE . Repair with preservation is also explicit in https://doi.org/10.1017/cbp.2025.10003 . A generic minimal head update is therefore not being advanced as a new method.

QRL https://arxiv.org/html/2304.01203 Sections2--3 imposes goal-conditioned quasimetric structure and learns distances through constrained local/global relations; official code https://github.com/quasimetric-learning/quasimetric-rl . Its undiscounted nonpositive transition-cost formulation is not identical to this terminal-positive discounted task; this experiment does not claim that QRL cannot solve it or reproduce its baseline.

This candidate instead clips a frozen continuation value only when it exceeds an optimistic native-execution reachability envelope. Admissible bounds and clipping themselves are established ideas. The only proposed incremental capability is incorporating finite action support, previous executed velocity and exact smoothing into this envelope, versus the mandatory distance-only clipping control. A gain absorbed by that control or by pure geometric scoring is negative, not a renamed contribution.

Bound scope: it is a necessary free-motion projected-distance condition, hence an upper return bound only under the pinned nonpositive nonterminal reward and terminal-success convention. It ignores humans and makes no safety or optimality guarantee. Actual safety and task outcomes are measured in the unmodified environment. Future pedestrian tapes are never used in selection.
