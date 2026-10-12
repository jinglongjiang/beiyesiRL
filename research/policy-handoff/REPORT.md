# Policy-Specific Handoff: Complete Negative

Research decision: KILL this candidate. No method-level progress established. The frozen independent-benefit criterion fails; the overall autonomous search continues. This is not a claim that policy-aware handoff is impossible.

## Native Paired Outcomes

Six native scenarios, cases100--119, original50s,120 episodes per arm. Previously observed development cases; not fresh confirmation. Frozen corrected GRU and original80-action execution contract.

| Arm | Success | Collision | Timeout | Independent contact | Successful time, s | Mean decision, ms |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 102 | 0 | 18 | 0 | 16.561 | 16.795 |
| Fixed8 MPPI plan | 114 | 0 | 6 | 0 | 15.340 | 39.477 |
| Ordinary variable-prefix policy rollout | 115 | 0 | 5 | 0 | 15.415 | 38.204 |
| Safe/productive nominal-policy handoff | 116 | 0 | 4 | 0 | 15.554 | 37.685 |

Against ordinary rollout: one extra success (dense_square/105), no damages, paired SR difference0.833pp, whole-case bootstrap95% interval[0,2.5]pp. Among115 paired successes the candidate is0.074s slower, ratio1.00479. The gate required two extra successes or5% paired time reduction, without safety regression. It fails. No fresh cohort opened, no duration/threshold rescue.

## What Changed and Why It Did Not Pass

The re-entry constraint changes122 planning decisions, so the null task conclusion is not an inactive module. Both candidate and ordinary rollout query the same frozen GRU continuation, same four prefix lengths and same eight-step tail with correct GRU history and smoothing. Candidate additionally rejects predicted unsafe/nonproductive re-entry. It does not establish enough additional task benefit beyond the ordinary rollout baseline. The one rescue is preserved, not called stable generalization.

## Correctness and Scope

Six smoke cases121 repeated twice: exact actions, physical traces and raw values match. Candidate first nominal-tail action matches actual GRU; temporary history, previous action and environment references restore exactly. All executed actions obey the original current safety mask where nonempty. Human execution matches the cached native tape; no tape or hidden goals are passed to the controller. Full frozen source/weight SHA checks pass before and after outcomes.

Policies ran concurrently on local3060/CPU. Reported wall-clock latency is co-run latency, not a dedicated hardware speed comparison. Two failed ad-hoc status-print commands had Python SyntaxError; they never ran simulation or changed datasets and are not scientific observations. Outcome processes and verification all exited0.

## Prior Boundary

Actual reuse: pinned unchanged pytorch_mppi. Read official MPS code and primary truncated-rollout methods in PRIOR.md. Additional primary review during outcomes: DMPS, https://arxiv.org/html/2405.13863 , Eq2/Algorithm1 already uses task-aware recovery planning and learned terminal value. Reversible Simplex, https://arxiv.org/html/2601.19499 , Eqs22--26 already constructs learned-policy continuation and debt-gated re-entry sets. These are not exact executions of this crowd adapter, but substantially constrain claims. Generic safe/progress-aware handoff is NOT established novelty here. No claim of formal robustness under approximate CV forecasts.

## Reproduction and Decision

Pre-outcome code/protocol commit147bdc22. python research/policy-handoff/run.py verify; python research/policy-handoff/run.py summarize. development-summary.json holds exact paired outcomes and gate; artifacts/development retains every episode, executed action, raw80-value vector and handoff option diagnostic. Scientific verdict: REFUTED for this frozen candidate's specified development benefit. No METHOD_ENTRY_FOUND.
