# Matched Context Lookahead: Negative Reference

## Research Verdict

REFUTED for the registered independent navigation-benefit claim; KILL this reference. No method-level progress. Correcting candidate-dependent human-context offsets changes execution but performs worse than both the fixed-context control and the existing fixed8 MPPI baseline. No offset scaling, refitting, or fresh-confirmation rescue is authorized.

## Frozen Method and Controls

Specification/source frozen in 1ab805e5; completed correctness controls in af2d3328. Parent checkpoint, original source, legal history, reward,80 native actions,50s horizon, all-human safety and MPPI remain pinned. The candidate subtracts each successor context's current-time value offset relative to the root context. The ordinary control freezes root nearest-five identities across candidates. Both new arms perform the same additional value evaluations. This subtraction is a heuristic, not an unbiased estimator or a proven value bound. PRIOR.md records primary literature/source scope; novelty remains UNVERIFIED.

Controls pass: original uncorrected candidate-score parity, terminal scoring, complete five-human trajectory parity, deterministic duplicate, and legal current-time history. In six complete smoke episodes, support changes on117/338 decisions and rankings change on16 decisions across three scenes. All six succeed without contact. These checks establish implementation validity, not task benefit.

## Complete Native Closed Loop

Six scenes, cases300--319,120 episodes per arm. Nominal and fixed8 exact-cohort results are reused; two new arms produce240 complete trajectories. Statistical resampling clusters the20 numeric case IDs across scenes, not individual decisions or training seeds.

| Arm | Success | Collision | Timeout | Geometric contact | Successful time(s) | Decision time(ms) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen GRU |105|0|15|0|15.764|9.049|
| Original fixed8 MPPI |119|0|1|0|14.805|29.452|
| Fixed-root human context |119|0|1|0|15.038|32.101|
| Matched-context correction |117|1|2|1|15.128|31.187|

Against fixed8:0 rescues,2 damages, success difference-1.667pp with95% case-bootstrap interval[-4.167,0]pp. Jointly successful navigation is0.372s slower(+2.520%). Against fixed-context: likewise0 rescues and2 damages, paired time+0.088s(+0.582%). Against GRU,12 rescues do not establish a contribution because the stronger ordinary controls absorb the benefit and have no contact. Historical reused timing is not a simultaneous compute benchmark.

The candidate changes81 executed trajectories. Its actual rankings change on491 of2660 support-changing decisions, out of7512 total decisions. Thus lack of activation does not explain failure. Frozen success-nonregression, safety and independent-task-gain gates all fail; activation and compute gates pass. No fresh cases are used.

## Failure Evidence and Limits

Added damage dense_square/304 times out with minimum clearance0.215712m and final goal distance3.86696m. Its intervention activates176 steps, changes169 actions, and has no fallback or plan invalidation. Added damage large_square/314 collides at7.5s with minimum clearance-0.0224904m before any intervention trigger, with5 fallback steps. The correction therefore can damage nominal decisions before the planner activates; the timeout cannot be explained by absent intervention. The remaining dense_square/307 timeout also has active interventions and no fallback/invalidation.

These facts reject the tested correction, not all context-consistency methods. They do not uniquely prove whether subtracted offsets contain legitimate information or whether another upstream representation problem dominates: those explanations remain UNRESOLVED. One frozen GRU training seed and repeatedly used development cases do not establish cross-training or population generalization.

## Preserved Deliverables

protocol.json, frozen-manifest.json, controls.json, development-{nominal,hold8,fixed,matched}.json, development-summary.json and artifacts/development/{fixed,matched} preserve outcomes, execution, scores, context changes and planner status including failures. All full processes finish successfully and source verification passes. No experiment traceback/assertion failure or4090 computation occurs.

```bash
docker exec -w /workspace/recovery-duration-20261011 mamba_env /root/miniconda3/envs/mamba/bin/python research/matched-context-lookahead/run.py summarize
docker exec -w /workspace/recovery-duration-20261011 mamba_env /root/miniconda3/envs/mamba/bin/python research/matched-context-lookahead/run.py verify
```

Research decision for this candidate: KILL. Overall autonomous research continues with primary-source screening of a distinct mechanism; this report does not claim a second-paper PASS.
