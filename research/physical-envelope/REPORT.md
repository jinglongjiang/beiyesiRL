# Physical Critic Envelope: VERSION_NEGATIVE

The candidate did not meet its frozen navigation criterion. It reached107/120 with no contact, versus114/120 for the strongest fixed8 MPPI control. This cycle achieved no independent method-level progress. Scientific verdict REFUTED for the registered comparative-benefit proposition; research decision KILL this version. No structural repair, fresh confirmation or integration rescue is authorized by this result.

## Frozen Comparison

Pre-outcome commit deca5fbf. Corrected frozen GRU seed61 RL2000, SHA in protocol.json, original80 actions, native50s/dt.25, legal-prefix24, nearest5 value input, all-human safety, margin.2/risk.8/smoothing.3. No reward, environment, architecture, checkpoint or training change. All modifications are standalone research wrappers. Local3060/CPU only; no4090.

Six native scenes, cases100--119 each,120 episodes and20 numeric-case clusters. Existing development cases, not confirmation. Nominal and fixed8 results are reused with source hashes from the completed handoff comparison; the three value-channel conditions actually executed complete native closed loops. Each step verifies pedestrian tape and actual robot update. Future tapes/goals are not available to the selector.

## Actual Outcomes

| Arm | Success | Collision | Timeout | Geometry contact | Successful time(s) | Mean decision(ms) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen GRU |102|0|18|0|16.561|16.795|
| Fixed8 MPPI |114|0|6|0|15.340|39.477|
| Bound-only geometric control |4|21|95|21|35.938|13.909|
| Distance-only upper cap |103|0|17|0|19.087|14.973|
| Execution-consistent envelope |107|0|13|0|15.586|15.164|

All rows n=120. Runtime is co-run wall time, including reused prior measurements, not a dedicated isolated speed benchmark. Native collision and independent swept geometric contact are recorded separately even where counts agree.

Candidate versus GRU:6 rescues/1 damage, net+4.17pp, case-cluster95% interval[0,+10.00]pp. Versus distance cap:7 rescues/3 damages,+3.33pp[-1.67,+9.17]pp. Versus fixed8:1 rescue/8 damages,-5.83pp[-10.83,-0.83]pp. Paired successful-task time versus fixed8 is+0.281s, ratio1.0185, difference interval[-0.450,+1.148]s. Exact IDs and all outcomes remain in development-summary.json and per-episode artifacts.

## Mechanism and Failure

CONFIRMED: the optimistic envelope follows the original finite action support and smoothing recurrence. Six-scene controls verify the recurrence against explicit repeated commands; repeated complete runs have exactly equal trace and candidate arrays. Grid80, native50s, parent/import contracts and final source/checkpoint hashes pass. The upper-bound construction requires nonpositive nonterminal rewards, asserted from the actual environment; it is not a safety certificate.

SUPPORTED: capping impossible optimistic values can change useful rankings locally, but it does not solve the full navigation defect. Candidate value clipping is active on33.39% of its9284 decision steps. Its failure against fixed8 is therefore not an inactive-channel result. Actual old-policy training causes remain UNRESOLVED; this inference intervention cannot identify them.

The bound-only control fails severely and is NOT the strongest geometric planner. It replaces learned values with integer earliest-arrival bounds, producing top-score ties on92.33% of steps, median41 tied safe actions; executed speed median0.129m/s. These plateaus and the existing first-index tie break support a quantization/slow-selection explanation. A matched tie-break intervention was not run, so this is not a confirmed unique cause. Do not use its collapse to claim that learned values or the candidate beat all geometric methods. The already included fixed8 MPPI is the dangerous stronger geometric comparison and defeats the candidate. No need to rescue this weak auxiliary control to decide the candidate.

## Scope and Research Judgment

Admissible bounds, constrained value geometry and localized neural repair are existing ideas. SARP official notebook was inspected at commit beb3d42509a3d1dc9715ebd690db755fc74461d8: it combines imitation error and predictive collision/velocity constraints. REASSURE's stated piecewise-linear scope does not directly cover this GRU. QRL Sections2--3 learn quasimetric value geometry. None was benchmark-reproduced here; PRIOR.md is not a claim of state-of-the-art reproduction. See PRIOR.md for primary links.

The registered additional mechanism was executable finite-action/smoothing consistency beyond distance-only clipping. Its comparison has no stable confirmed outcome advantage, and the best existing simple method remains better. Kill this candidate, preserve its source/raw results, and do not claim novelty or a paper PASS. Continue the user-requested loop only with a genuinely different evidence-supported hypothesis.

## Reproduction

Use the container Python from the repository root:

    python research/physical-envelope/run.py verify
    python research/physical-envelope/run.py summarize
    python research/physical-envelope/analyze-mechanism.py

Do not overwrite original episode outputs. Evaluation entry exists in run.py; full new runs must use separately versioned output directories. No invalid scientific episode or discarded case occurred. One exploratory rg command referenced a nonexistent prior-source folder and exited2; it did not execute an experiment or alter scientific observations.
