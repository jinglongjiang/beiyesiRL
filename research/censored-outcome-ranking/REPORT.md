# Censored Outcome Ranking: VERSION_NEGATIVE

Scientific verdict REFUTED for the registered claim of independent complete-navigation benefit. Research decision KILL this correctly activated reference, with no repair and no fresh confirmation. This round achieved no method-level progress. Continue the user's hypothesis/implementation/paired-outcome loop on a genuinely different mechanism.

## Complete Outcomes

Native50s, original80 actions, corrected frozen GRU seed61. Cases300--319 across six scenarios:120 episodes,20 numeric-case clusters, not120 training seeds. All learned arms have the identical fixed8 MPPI fallback and identical32897-parameter residual architecture, initialization,1000-update budget, legal features and safety settings.

| Arm | Success | Collision | Timeout | Geometric contact | Successful time(s) | Decision(ms) |
|---|---:|---:|---:|---:|---:|---:|
| Original GRU |105|0|15|0|15.7643|9.049|
| Fixed8 MPPI |119|0|1|0|14.8046|29.452|
| Ordinary pairwise |117|1|2|1|15.6068|83.942|
| Policy-weighted set KL |119|0|1|0|16.3634|94.535|
| Winner-only task-margin candidate |116|3|1|3|16.2909|86.064|

Original GRU/fixed8 rows reuse exact complete previously generated cases, with source hashes recorded in development JSONs; their timing was measured in the previous run, not concurrently with the three new arms. All new arms actually execute complete episodes and evaluate all three residual models for matched inference work. The displayed successful times have different success subsets; paired comparison below is the primary efficiency evidence. Concurrent local3060 load affects absolute timing; no4090 work.

Candidate versus fixed8:0 rescues/3 damages, netSR-2.50pp, whole-case bootstrap95%[-5.00,0.00]pp; paired jointly successful time+1.5259s,+10.334%. Versus set KL:0 rescues/3 damages, paired time-0.0496s, nowhere near the registered5% gain. Versus pairwise:1 rescue/2 damages. Candidate versus original GRU rescues14/damages3, but the established fixed8 control already absorbs that improvement without extra contacts. No inference of contribution from beating only GRU.

3895 actual non-intervention decisions change due to the candidate residual. Mechanism activation and compute gates pass; success, safety and independent-benefit gates fail. No structural repair: the intended feature, gradient and consumer paths function correctly.

## What Failed and Why

CONFIRMED: the proposed winner-only correction fails to preserve navigation safety and fails to outperform the ordinary controls. It is not a disconnected module. Learned ranking alone did not safely transfer fixed-world old-policy successful alternatives to the changed closed-loop policy.

Direct trace evidence for the three candidate damages:

| Scene/case | Last four safe-action counts | Last four MPPI-active flags | Final native/geometry outcome |
|---|---|---|---|
| dense_circle/308 |3,0,0,0|false,false,false,false|collision/contact|
| dense_square/300 |3,0,0,0|false,false,false,false|collision/contact|
| large_square/311 |10,0,0,0|false,false,false,false|collision/contact|

The existing safety mask is enforced whenever a safe candidate exists; when empty, the original unmodified fallback remains in force. The negative-progress-triggered planner is not active in these final steps. These facts explain the immediate execution channel; they do NOT prove whether feature generalization, stale old-policy labels or the winner-only constraint is the unique upstream learning cause. That separation remains UNRESOLVED and does not change KILL. No safety condition or action support was relaxed to obtain these outcomes.

All-failure/all-success roots intentionally do not invent an action preference. This avoids one invalid supervision assumption, but does not guarantee corrected-policy recoverability. Standard max-margin, set supervision and preservation penalties are known ingredients; no novelty claim survives this negative result.

## Frozen Evidence and Reproduction

Pre-fit specification/source freeze cbc187d0; fitting artifacts a8e2eaf1, weights67ed665e; passing complete correctness smoke6381da43. The fitting-artifact commit included partial smoke outputs already in progress, not a pre-smoke commit; the protocol itself predates all fitting and smoke outcomes. All weight files are tracked explicitly despite the repository's default pth ignore rule.

Files: protocol.json, frozen-manifest.json, data.json with raw-label hashes, training.json with fitted weight hashes, controls.json, development-{nominal,hold8,pair,set,regret}.json, development-summary.json, artifacts/development/{arm}/{scene}-{case}.{json,npz}, and corresponding mechanism JSONs. Parent SHA879c612fcea61a7e39bdef6506b4eb141b6eecdf0de46eda39ad8525acd8ee2c remains unchanged.

From the worktree inside mamba_env:

```bash
/root/miniconda3/envs/mamba/bin/python research/censored-outcome-ranking/run.py verify
/root/miniconda3/envs/mamba/bin/python research/censored-outcome-ranking/run.py summarize
```

Correctness controls: synthetic distinguishing gradients; zero-output initialization; unsafe/terminal corrections zero; GRID80; native50s; six complete case121 smoke episodes; one bitwise-exact repeat. Smoke success is not comparative scientific benefit.

Engineering-only failures: a one-line monitoring expression indexed an already extracted list as a dictionary (TypeError), then was corrected without touching running experiments. An exploratory sparse clone of the next-prior risk repository failed before producing a directory; a separate no-checkout source clone succeeded. An attempted grep for nonexistent graph files in the Social NCE repository returned no match and was replaced by inspecting the actual relational-graph repository. None entered training, labels, navigation or scientific counts.

## Next Decision

Do not tune this loss, add labels, relax safety, or run fresh confirmation. Preserve all negatives. The next distinct hypothesis concerns selecting complementary action-relevant pedestrian witnesses under the existing five-person value-input budget, compared against ordinary scalar risk selection and the unchanged all-human safety/planning control. Selection itself and greedy submodular coverage are not automatically novel; check actual closest source before freezing that reference.
