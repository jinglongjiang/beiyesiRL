# Existing-method reuse: direction consistency in the GRU navigation loop

## Research verdict

**KILL this fixed TEB-inspired holonomic side-preference adaptation.** Its complete navigation success is 102/120, below both nominal GRU (105/120) and the existing fixed8 simple baseline (115/120). It rescues no nominal failures and damages three nominal successes. The intervention reaches execution, but does not solve sustained negative progress. **This round achieved no independent method-level progress.** No parameter sweep, structural rescue, new training, or fresh confirmation was started.

Scientific proposition: this frozen direction-consistency adaptation provides navigation improvement beyond fixed8 under the shared contract. **REFUTED on this development cohort.** This does not refute full TEB, topology-guided MPC, or every possible direction-consistency method.

## 1. GitHub reuse decision

The user requested existing implementations be checked before new algorithm development. Actual source was inspected, not just repository titles or reported success rates.

| Project | Verified existing mechanism | Fit to this task | Decision |
|---|---|---|---|
| [TEB](https://github.com/rst-tu-dortmund/teb_local_planner) | `FailureDetector`, retained preferred rotation after oscillation; one-sided rotation penalty; homotopy selection hysteresis and switching blocking period | Closest cheap reusable anti-switching mechanism. Full optimizer depends on ROS/g2o and differs from the host holonomic 80-action value scorer | Adapt only the direction preference, with license and interface differences disclosed; tested below |
| [Nav2](https://github.com/ros-navigation/navigation2) | Progress checker tests displacement from a stored pose within a time allowance; behavior tree supplies standard recovery actions | Suitable engineering reference, but displacement alone does not distinguish goal progress from retreat/looping. Full ROS2 stack is not a drop-in controlled module | Do not rebuild the navigation system around it; do not call ordinary progress/recovery logic novel |
| [T-MPC / T-MPC++](https://github.com/tud-amr/mpc_planner) | Parallel topology-conditioned solvers, feasible-solution objective selection, multiplicative preference for previously selected guidance | Strong related planning implementation, but importing it wholesale changes planner, action parameterization and constraints; requires solver generation and additional dependencies | Reference, not a quick replacement in the current controlled experiment |

Pinned TEB commit: `9e55af4724310c34bcfa8cbb3b2783478c1a246a`, BSD-3-Clause. The source mechanisms are [direction retention](https://github.com/rst-tu-dortmund/teb_local_planner/blob/9e55af4724310c34bcfa8cbb3b2783478c1a246a/src/teb_local_planner_ros.cpp#L980), [one-sided rotation penalty](https://github.com/rst-tu-dortmund/teb_local_planner/blob/9e55af4724310c34bcfa8cbb3b2783478c1a246a/include/teb_local_planner/g2o_types/edge_prefer_rotdir.h), and [trajectory switching hysteresis](https://github.com/rst-tu-dortmund/teb_local_planner/blob/9e55af4724310c34bcfa8cbb3b2783478c1a246a/src/homotopy_class_planner.cpp#L514).

Nav2 source: [simple_progress_checker.cpp](https://github.com/ros-navigation/navigation2/blob/main/nav2_controller/plugins/simple_progress_checker.cpp), [recovery behavior tree](https://github.com/ros-navigation/navigation2/blob/main/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml). Source was read; Nav2 was not executed here. The mutable `main` links are references, not a pinned reproduced baseline.

Pinned T-MPC source: commit `3002da6a5e43b3577f584381bbcad569c82a381b`, Apache-2.0; [guidance_constraints.cpp](https://github.com/tud-amr/mpc_planner/blob/3002da6a5e43b3577f584381bbcad569c82a381b/mpc_planner_modules/src/guidance_constraints.cpp#L315). Its already-existing consistency preference rules out claiming that merely retaining a passing choice is an original contribution. No claim is made that these three repositories exhaust all prior work.

## 2. Frozen hypothesis and exact adaptation

Hypothesis: keeping a bypass side during repeated nonpositive progress, while retaining value ranking and safety, can reduce failed reversals without forcing forward motion. The strongest simple explanation is the already-tested fixed8 intervention, not nominal GRU alone.

Code and protocol were committed before new navigation outcomes: `49738a53`; the positive-control fixture correction was committed as `ae154136` before any navigation run. The controller and navigation parameters were not changed after outcomes.

Same repaired seed61 GRU checkpoint as the previous study:

```
research/recovery-duration/artifacts/nominal.pth
SHA256 879c612fcea61a7e39bdef6506b4eb141b6eecdf0de46eda39ad8525acd8ee2c
```

Unchanged: IL/RL weights, legal T24 history, nearest5 value inputs, all-human safety geometry, 80 executable smoothed actions, margin .2, risk .8, smoothing .3, reward, task limit50s and dt .25. Native pre-step timeout semantics are retained, including possible201 executed steps. No human goals or future states enter the controller. Cached human futures drive only the environment.

Activation is the existing fixed8 trigger, not TEB's native detector:8 consecutive nominal nonpositive-progress decisions and no goal-distance reduction over the preceding8 executed steps. The intervention lasts8 steps. Empty-safe-set behavior is unchanged; an active interval is cancelled on an empty set.

Let e be the goal direction at activation and c the retained side. For each original executable candidate velocity v_a:

```
beta_a = asin(clip(cross(e, v_a) / max(norm(v_a), 1e-12), -1, 1))
w_t = std(original scores over the original safe set)
score'_a = score_a - w_t * max(0, -c * beta_a)^2
```

The side comes from the current actual lateral velocity, or the highest-value safe non-collinear candidate if lateral velocity is negligible. There is no progress bonus or new action support. This maps TEB's direction penalty to holonomic passing-side geometry; it is **not** equivalent to TEB's yaw optimization, full trajectory hysteresis, or full official planner. `side-reset` uses the same interval and formula but recomputes side/axis each step; `side-latched` retains them.

## 3. Correctness and preserved engineering failure

- Six-scene nominal replay: recorded actions, physical records and all candidate values are exactly equal to the previous run.
- Six-scene repeated side-latched smoke: all executed actions, physical records, candidate arrays and outcomes exactly equal; timings excluded.
- Unit checks: trigger timing, direction retention, native fallback, safe selection, excluding masked -1e9 from scale, rotation/translation equivariance and direction sensitivity pass.
- Runtime assertions:80 actions, legal import paths, same checkpoint, original source hashes, same physical human tape, selected/executed velocity and robot displacement agreement all pass.
- `verification.json` independently rechecks240 new complete trajectories /19,495 executed steps and240 reused complete baseline outcomes.
- `crowd_nav`, `crowd_sim` and all previous study sources have no difference from parent `5b096626`.

Preserved engineering run: the first synthetic positive control incorrectly required an action flip although its monotonically increasing scores had a near-collinear best action and insufficient penalty. It failed before any navigation. The test fixture was changed to two known near-tied opposite-side peaks; controller/protocol unchanged. Evidence: `engineering-log.json` and `engineering-controls-development-side-latched.json`. This is an invalid test fixture, not a scientific negative or an algorithm repair.

## 4. Actual paired closed-loop outcomes

Six native scenarios, cases300..319,120 complete episodes per arm. This is a previously used development cohort, not fresh confirmation. Nominal and fixed8 outcomes were reused, with source hashes checked and nominal execution parity established. Only the two side arms are newly simulated; no baseline retraining or whole-cohort rerun.

| Method | Success | Collision | Timeout | Independent contact | Success time(s) | Decision ms |
|---|---:|---:|---:|---:|---:|---:|
| Nominal GRU, reused |105/120 (87.50%)|0|15 (12.50%)|0|15.764|9.049|
| Fixed8 simple baseline, reused |115/120 (95.83%)|0|5 (4.17%)|0|15.587|9.058|
| Side-reset reference |105/120 (87.50%)|0|15 (12.50%)|0|15.636|10.459|
| Side-latched adaptation |102/120 (85.00%)|0|18 (15.00%)|0|15.429|10.490|

Success-only time is descriptive; it is affected by which episodes succeed. On both-success pairs, side-latched is .047s slower than nominal and .468s slower than fixed8. Historical baseline timing is reused rather than contemporaneously measured, so no precise causal percentage for runtime overhead is claimed. All arms use one original GRU candidate pass, with no additional solver/search budget.

| Scenario | GRU success/20 | Fixed8 | Side-reset | Side-latched |
|---|---:|---:|---:|---:|
| baseline_circle |20|20|20|20|
| baseline_square |19|19|19|19|
| dense_circle |20|20|20|20|
| dense_square |13|16|13|11|
| large_circle |17|20|17|17|
| large_square |16|20|16|15|

Pairing and bootstrap: numerical case is the cluster across six scenes;20 clusters,4000 draws. Candidate/step counts are not sample sizes. Only one nominal training seed is tested.

- Side-latched minus nominal SR: -2.50pp,95% case-cluster interval[-5.00,0.00]pp;0 rescues /3 damages.
- Side-latched minus fixed8: -10.83pp, interval[-17.50,-5.83]pp;1 rescue /14 damages.
- Side-latched minus side-reset: -2.50pp, interval[-5.00,0.00]pp;0 rescues /3 damages.
- Nominal damages: dense_square/302, dense_square/315, large_square/305. These are also the three damages relative to side-reset.
- The only rescue relative to fixed8 is dense_square/305; it was already successful under nominal, so it is not a rescued nominal failure.
- All damage/rescue IDs, geometric clearance, original return, path, terminal flags and complete executed steps are retained, not just successful examples.

Zero observed contacts is not a zero-risk certificate. The development interval is not an estimate of generalization across independently trained policies.

## 5. Why this adaptation failed

**CONFIRMED implementation fact:** the mechanism is connected. Side-latched activates in45 episodes,198 intervals /1540 steps, and changes114 executed choices. Side-reset activates in45 episodes,171 intervals /1343 steps, and changes115 choices. It is not a disconnected module or zero-activation negative.

**SUPPORTED mechanism limitation:** constraining lateral choice leaves the original GRU's longitudinal value preference largely intact. During side-latched active steps,84.29% of actual actions still have nonpositive goal progress; the same-state nominal choice is nonpositive83.51% of the time. A safe positive-progress candidate exists in99.68% of these steps. The intervention nevertheless selects nonpositive progress in1293 such steps.169/198 intervals have nonpositive net progress, with median interval progress -1.834m. This is post-outcome descriptive analysis, explicitly recorded as such in `mechanism.json`, not a preregistered new causal experiment.

The formula explains a concrete limitation: motion with zero lateral component gets zero side penalty regardless of whether it progresses or retreats. Thus side retention does not enforce a successful longitudinal sequence, nor correct longitudinal value errors. The existing success difference between fixed8 and this adaptation directly rejects the claim that this chosen lateral regularization is a sufficient replacement for its intervention. It does not establish the unique historical training cause of GRU's value preferences, or prove that every retreat here was unnecessary.

The three newly damaged episodes and all original failures remain available. No unsupported statement is made that an increased number of triggers alone caused the damage, that direction memory is universally harmful, or that full TEB would fail.

## 6. Decision and completed execution

- **SUPPORTED:** reusable public anti-switching mechanisms exist; this adaptation genuinely reaches execution under the unchanged host contract.
- **REFUTED on this cohort:** the frozen side-latched adaptation improves navigation beyond fixed8.
- **UNVERIFIED:** any new algorithmic contribution or improvement of the full official TEB/T-MPC planners in this environment; neither full planner was executed.
- **Research decision: KILL this adaptation.** Keep fixed8 as the strongest tested simple baseline, not as a new contribution.
- The frozen development gate failed, so the reserved cases340..359 were not run. No fresh data were consumed and no new parameters were selected.
- The single prioritized disposition is to retain the fixed8 baseline and seal this adaptation with its evidence; no automatic next research branch is opened.

## 7. Reproduction and artifacts

From the repository root in the existing container environment:

```
python research/teb-side-reference/run.py controls
python research/teb-side-reference/run.py evaluate --arm nominal
python research/teb-side-reference/run.py evaluate --arm commit8
python research/teb-side-reference/run.py evaluate --arm side-reset
python research/teb-side-reference/run.py evaluate --arm side-latched
python research/teb-side-reference/run.py summarize
python research/teb-side-reference/summarize-mechanism.py
python research/teb-side-reference/verify.py
```

Do not run `freeze` on changed code to claim reproduction of this version; use the committed frozen manifest. The evaluator reuses already complete per-episode outputs where present. A clean outcome rerun should use a separate copy of this study directory retaining protocol and manifest.

`protocol.json`, `frozen-manifest.json`, `controller.py`, `run.py`, `controls.json`, `development-*.json`, `development-summary.json`, `mechanism.json`, `verification.json`, and `artifacts/{controls,development}/...` provide the frozen specification, complete per-step candidates/actions, results and recomputation. No new weights were trained. The existing nominal checkpoint is unchanged and already part of the repository evidence.

Actual new-arm evaluation elapsed112.25s and116.43s, plus controls/statistics. Local RTX3060 inference only; no4090 connection, compute, file change or training was used.
