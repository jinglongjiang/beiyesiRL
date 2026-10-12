# Local Action-Gap Retrieval: VERSION_NEGATIVE

The centered retrieval candidate failed its registered complete-navigation criterion:116/120 successes versus119/120 for both ordinary absolute retrieval and fixed8 MPPI, with no contact in these arms. It was30.22% slower than fixed8 on paired jointly successful tasks. Scientific verdict REFUTED for the specified independent-benefit proposition; research decision KILL. No method-level progress, no structural repair and no fresh confirmation.

## Frozen Implementation

Pre-outcome commits7c6330b3,8d865b2b,e6949bef. Original corrected frozen GRU seed61/RL2000; original80 actions, native50s, all-human safety, margin0.2/risk0.8/smoothing0.3. Source and checkpoint hashes pass before and after execution. Only the standalone retrieval/controller wrapper differs; no original project source or network weights changed, no RL training, no4090 use.

Both memories contain the same1531 nonterminal safe action rows from whole even-case roots in the existing counterfactual catalogue. Odd cases are heldout diagnostics; all development cases300--319 are disjoint from fitting cases. Keys have256 dimensions,8 neighbors, inverse-distance kernel and a fitting-only leave-one-case-out radius rule. All deployment queries use the actual CV candidate windows, not privileged actual future states. Labels are fixed-world original-policy realized returns, not exact legal-observation conditional expected Q.

Absolute retrieval predicts Q residuals using absolute frozen features. Relative retrieval subtracts the within-root safe feature mean and residual mean. Both memories are queried in both arms to match work. Both arms use the identical fixed8 MPPI fallback. Same masks, terminal precedence and executable commands apply. Six-scene complete duplicate controls produce identical records/candidate arrays. Control case301 also belongs to development; it was used only after freezing parameters, never for tuning. Development is not confirmation.

## Actual Closed Loops

| Arm | Success | Collision | Timeout | Swept contact | Successful time(s) | Mean decision(ms) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen GRU |105|0|15|0|15.764|9.049|
| Fixed8 MPPI |119|0|1|0|14.805|29.452|
| Ordinary absolute memory + same MPPI |119|0|1|0|17.061|100.698|
| Relative action-gap memory + same MPPI |116|0|4|0|19.241|112.353|

Each arm has120 episodes/20 numeric-case clusters. GRU/MPPI are source-hashed reused results, not reruns. Both memory arms actually executed all120 episodes. Runtime is co-run wall time and includes reused earlier baseline measurements; it is not a dedicated isolated speed benchmark.

Relative versus fixed8:0 rescues,3 damages (baseline_square308; dense_square302/304), net-2.50pp, case-cluster bootstrap interval[-5.00,0.00]pp. Paired success time+4.466s. Versus ordinary absolute memory: same0/3 rescue/damage, paired time+2.450s. Relative versus GRU:+9.17pp[+2.50,+17.50]pp does NOT establish a method benefit because the common ordinary MPPI already performs better.

## What Failed

CONFIRMED: the memory affects3963 executed non-MPPI decisions. This is not a disconnected mechanism. Relative correction produces409 fallback activations versus169 for unmodified fixed8; its harmful nominal ranking changes coexist with more frequent planning assistance. This measures actual activation, not a unique upstream cause.

SUPPORTED: subtracting the state/common feature offset is insufficient to transfer beneficial action preferences. It neither preserves all successful control cases nor improves efficiency beyond ordinary retrieval or planning. There is no independent benefit to attribute to the relative representation.

UNRESOLVED: whether loss comes primarily from cross-world feature aliasing, sparse catalogue support, or labels evaluated under the old policy rather than the changed policy. None was isolated causally; do not invent a root cause. Different-arm heldout residual MAEs are not directly comparable because the centered and absolute targets have different scales. No new diagnosis is required to kill this specified candidate.

NEC's episodic kernel lookup, advantage centering and RankQ's ranking objectives are established. PRIOR.md links the actual primary methods; no official NEC or RankQ benchmark was reproduced and no paper novelty has been established. The third-party EC source inspected was d528dde5131de679b30f4064ed01d8b12328b14c. The original frozen GRU training replay remains missing; this experiment cannot reconstruct its historical learning cause.

## Preserved Evidence and Next Decision

protocol.json, frozen-manifest.json, bank.json/bank.npz, controls.json, all240 newly executed episode records/candidate arrays/mechanism logs, reused baseline source hashes, and development-summary.json are preserved. Recompute with `python research/local-action-gap/run.py verify` and `python research/local-action-gap/run.py summarize` from the repository root in mamba_env. No invalid scientific run occurred. One exploratory sed used a nonexistent catalogue path and two candidate project URLs failed; these were source-discovery errors, not experiments.

Seal relative episodic action-gap retrieval without parameter rescue. Continue the user-requested loop with a distinct, pre-registered hypothesis; do not convert these negatives into a paper PASS.
