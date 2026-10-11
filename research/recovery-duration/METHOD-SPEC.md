# Recovery Duration V1

## Decision and frozen difference

The target is avoidable stalling without damaging useful robot waiting. All three
arms use one newly trained corrected GRU nominal policy. Its inputs, candidate
execution, native reward, dynamics and evaluation settings are shared. No old
checkpoint is used with corrected inputs. Human motion does not respond to the
invisible robot; this study does not model reciprocal negotiation.

At each eligible decision there are ten choices: continue nominal, or one of
three goal-relative recovery directions for 2/4/8 native steps. The two learned
controllers have identical inputs, ten outputs, parameters, initialization,
minibatches, optimizer, training steps, and execution/safety rules. Each training
root is evaluated to its original deadline under all choices, followed by the
same frozen nominal policy. These are actual native rollouts, not oracle values
available to the deployed controller.

Ordinary-Q minimizes MSE between outputs q_j and absolute discounted task returns
G_j. The candidate instead minimizes

    MSE(q_0, G_0) + MSE(q_j-q_0, G_j-G_0)
        + MSE(q_(m,d)-q_(m,d'), G_(m,d)-G_(m,d'))

for j>0 and all duration pairs of the same recovery direction m. Thus the proposed
local change is loss allocation to intervention opportunity and duration regret,
not a different reward, action set, safety margin or information source. It tests
whether explicitly preserving paired waiting/recovery contrasts helps the finite
capacity learner beyond ordinary value regression. This is a candidate incremental
method, not an established novel algorithm. Algebraically the contrasts contain
no additional information over the common complete return table.

## Closest prior and strongest explanation

- Recovery RL learns a safety critic and switches to a recovery controller based
  on constraint risk. Its official `recovery_rl/qrisk.py` fits safety targets and
  optimizes recovery against them. Here the target is task-return opportunity cost
  against waiting, including elapsed commitment duration, rather than a new safety
  shield. https://arxiv.org/abs/2010.15920
  https://github.com/abalakrishna123/recovery-rl/blob/master/recovery_rl/qrisk.py
- Option-Critic already learns option policies and termination using advantages.
  Initiation, advantage subtraction, and learned duration are NOT claimed novel.
  https://arxiv.org/abs/1609.05140
- Frozone tackles freezing with predicted crowd geometry and direction deviation.
  Our candidate must outperform ordinary recovery learning with the same recovery
  directions; escaping a freezing episode alone is not a contribution.
  https://arxiv.org/abs/2003.05395

The dangerous explanation is that absolute-Q learns exactly the same useful
choices. If it reproduces the benefit, stop this candidate. Do not rename the loss
or increase its budget. No broad novelty claim is licensed by this initial review.

## Evaluation and stop

See protocol.json for case lists, training budgets, signal thresholds and the one
permitted concrete repair. Report all failures and every rescue/damage case.
Only a development signal permits fresh confirmation. Inconclusive scientific
evidence does not prevent a KILL investment decision. No outcome from public
repairs is attributed to the recovery loss.
