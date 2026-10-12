# Progress-Deadline Candidate: Completed Negative

Scientific proposition: a fixed repayment deadline improves native closed-loop navigation beyond matched ordinary planning and a fixed plan prefix. Verdict: REFUTED for the frozen development success criterion. Research decision: KILL this candidate. No method-level contribution or paper PASS established. The autonomous method search continues with a different mechanism.

## Complete Native Results

Six native scenarios, cases 300--319, 120 paired episodes per arm. These are reused development cases, not confirmation. Frozen corrected GRU; original 80 actions, reward, 50-second limit and execution smoothing. No training or 4090 use.

| Arm | Success | Collision | Timeout | Independent contact | Successful time, s | Mean decision, ms |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 105 | 0 | 15 | 0 | 15.764 | 9.049 |
| Original fixed8 progress | 115 | 0 | 5 | 0 | 15.587 | 9.058 |
| Low-progress fixed8 | 116 | 1 | 3 | 1 | 14.933 | 10.344 |
| Ordinary MPPI | 118 | 0 | 2 | 0 | 14.676 | 27.079 |
| MPPI with fixed8 plan prefix | 119 | 0 | 1 | 0 | 14.805 | 29.452 |
| Fixed repayment deadline | 119 | 0 | 1 | 0 | 14.609 | 28.848 |

Deadline rescues 14 GRU timeouts and damages none, but rescues **zero** cases beyond fixed8 plan execution. Both fail dense_square/307. Paired time difference against fixed8 is -0.195 s; this does not satisfy the precommitted requirement of two extra successes. No fresh cohort was opened. Bootstrap resamples 20 numeric-case clusters jointly across scenarios; this is not independent training-seed confirmation.

## Mechanism and Limits

All 143 complete deadline windows actually achieve the prescribed 0.5 m repayment; ordinary MPPI does so in 139/164 windows and fixed8 in 136/157. This supports mechanism activation, not an independent task benefit. Endpoint CV certificates are not true-world safety/liveness guarantees. Recorded prediction-versus-realization comparisons alone do not uniquely identify prediction error or replanning as a cause.

The original no-terminal-value reference was stopped after 13 ordinary-MPPI episodes (7 successes, 6 timeouts, no contact). Native progress/time/standing rewards are zero, but discomfort penalties are NOT zero. One common terminal-value correction was frozen before rerunning all planning arms. Original partial observations remain in progress-contract and REPAIR-RECORD.md; they are not relabeled as invalid engineering observations. No further repair is permitted for this candidate.

## Reuse and Novelty Boundary

Actual source reuse: MIT pytorch_mppi, commit e04a569cd4215e705f9013145c496fc59cb25ed6, unchanged optimization engine. Generic MPPI, contractive/flexible-step MPC and retained backup trajectories are established techniques, not new contributions. Closest primary sources: https://arxiv.org/abs/2006.01229 ; https://arxiv.org/abs/2211.02780 ; https://arxiv.org/abs/2211.14361 . This application-level deadline did not establish an incremental navigation capability beyond the fixed-prefix baseline.

## Reproduction

From the worktree in the pinned container: python research/progress-contract-v2/run.py verify; python research/progress-contract-v2/run.py summarize; python research/progress-contract-v2/analyze-mechanism.py. Frozen protocol and SHA manifest precede outcomes in commit 3e30a4a9. Full episode records, actions, raw candidate values, independent geometry and diagnostics are retained under artifacts/development. development-summary.json contains exact paired statistics. controls.json contains deterministic repeat and projection/smoothing tests.
