# Complementary Five-Human Action Witnesses: Negative Reference

## Verdict

REFUTED for the registered claim of independent task benefit; KILL this frozen-policy selector. No method-level progress. It changes execution but loses one success and is slower than the strongest ordinary planner. Do not retrain or tune its selection horizon/weights to rescue this version.

## Frozen Comparison

Pre-outcome specification/code: e4fee10f. Complete correctness controls: f3e399f5. Parent SHA256 is recorded in protocol.json and checked before/after every arm. Original source/checkpoint, reward,80 native actions,50s, all-human safety and fixed8 MPPI are unchanged. Only which five legal humans reach the existing value encoder changes. All three new arms calculate the same TTC/scalar/coverage selections for matched work. Nearest/fixed8 outcomes are reused from the exact original cohort; their historical timing is not a simultaneous compute benchmark.

Primary-source screening in PRIOR.md: risk-based human selection and greedy coverage are established. No independent novelty claim was made from their combination. No official ROS/graph baseline reproduction is claimed.

Controls passed: original nearest token parity on every query, complete five-human trajectory parity, exact duplicate, executable-velocity smoothing, synthetic complementary witness.5992/23976 eligible token queries differ from scalar selection, across five scenes. Six full smoke episodes all succeed without contact. These are correctness/activation findings, not task benefit.

## Complete Development Outcomes

Six native scenes, cases300--319,120 episodes per arm;20 numeric-case bootstrap clusters, not120 training seeds. Three new arms produce360 complete trajectories.

| Arm | Success | Collision | Timeout | Geometric contact | Successful time(s) | Mean decision(ms) |
|---|---:|---:|---:|---:|---:|---:|
| GRU |105|0|15|0|15.764|9.049|
| Fixed8 MPPI, original nearest5 |119|0|1|0|14.805|29.452|
| TTC5 |118|1|1|1|14.977|60.398|
| Scalar-risk5 |118|0|2|0|15.148|60.234|
| Complementary coverage5 |118|0|2|0|15.019|60.846|

Coverage versus fixed8:0rescues/1damage, SR difference-0.833pp,95% case-bootstrap interval[-2.50,0]pp; jointly successful time+0.2288s(+1.547%). Coverage versus scalar: identical success/contact outcomes and only0.853% faster paired time, below the registered5% threshold. Coverage versus TTC:1rescue/1damage, no net success difference; avoids TTC's single contact, but does not establish superiority over the contact-free fixed8/scalar controls.93/120 actual executed trajectories differ from fixed8. Correct activation therefore cannot explain away the negative result.

## Damage and Bounds on Causal Explanation

dense_square/302 is the added timeout: no fallback, no planner invalidations, minimum true clearance0.201907m, final distance1.42714m after50.25s. Its final nonterminal steps still contain11,52,65,54,10,48,67 safe candidates; failure is not explained by an empty safe set. Saved execution shows alternating negative/positive progress and ten interventions, not absent activation. This directly establishes a changed closed-loop path that still times out. It does not uniquely establish why the chosen five-human representation sustains that path. Nearest-trained feature distribution change and the inadequacy of geometric witness coverage as a value-context criterion remain UNRESOLVED. There is no further diagnostic/repair campaign for this version.

## Deliverables and Reproduction

protocol.json, frozen-manifest.json, controls.json, development-{nominal,hold8,ttc,scalar,cover}.json, development-summary.json, artifacts/development/{ttc,scalar,cover} contain complete outcomes, candidate scores, executed actions, selected human sets and planner status, including every failure. Controls and all full-arm processes completed. No GPU training/4090 use. No experiment traceback or failed scientific assertion occurred in this cycle.

Recompute aggregate results without navigation reruns:

```bash
docker exec -w /workspace/recovery-duration-20261011 mamba_env /root/miniconda3/envs/mamba/bin/python research/action-witness-cover/run.py summarize
docker exec -w /workspace/recovery-duration-20261011 mamba_env /root/miniconda3/envs/mamba/bin/python research/action-witness-cover/run.py verify
```

Scientific scope: this rejects the specified frozen-policy adaptation, not all trained selection mechanisms. One nominal GRU seed and adaptive development use do not establish population or cross-training generalization. Fresh340--379 remains unused. Next distinct hypothesis is comparison-context bias from candidate-dependent nearest-five truncation; unlike this selector it does not rank people by risk or coverage, and must face a fixed-root-support simple control before any method claim.
