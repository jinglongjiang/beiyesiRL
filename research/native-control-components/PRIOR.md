# Primary Source Check

CSC-MPPI (Park/Jang/Kim,2025), SectionIII: primal-dual adjustment of continuous trajectories and DBSCAN clustering, with per-cluster control estimates. Official source solver/jax_mppi.py dbscan() and optimization loop inspected at4fdabebb3f1f8d69cd2bfcf645a69fa486908a03. https://arxiv.org/html/2506.16386 https://github.com/RCILab/RCI_cscmppi

Exact-Safe MPPI (2026), composition.py/mppi.py and filter interfaces inspected atcc63232eea8f58ca0708fa374de108f651c36532: nonsmooth barrier composition and continuous sampled-command filtering. https://arxiv.org/html/2610.07369 https://github.com/lc-lab25/Exact-Safe-MPPI

These establish constrained/multimodal sampling as prior art. Neither generic safe sampling nor clustering is claimed original. Their executable code is not directly substituted because it uses continuous controls/other dynamics and would change the frozen native80 execution contract. No official benchmark reproduction is claimed.

The small reference tests component-balanced proposals on the actual finite safe-action lattice, compared to the same sequential-safe categorical sampler without balancing. Core planner/dynamics use pinned LemonPi pytorch_mppi; connected components use scipy.sparse.csgraph, not a new graph solver. If ordinary safe sampling explains the gain, or the graph is almost always connected, KILL. Even a successful comparison needs a credible further contribution review before any paper PASS.

Existing fixed8 failures have zero plan invalidations. This rejects a claim that they are simply caused by absence of any valid plan; it does not establish component undersampling. Forecast error and finite planning horizons remain competing explanations, not confirmed causes.
