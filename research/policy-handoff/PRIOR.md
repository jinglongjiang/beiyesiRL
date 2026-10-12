# Closest Existing Mechanisms, Before Outcomes

Actual reuse: unchanged MIT pytorch_mppi optimizer, e04a569cd4215e705f9013145c496fc59cb25ed6. Existing discrete-action projection and corrected Parent execution reused from the sealed progress-contract reference; no claim these are new algorithms.

Read primary MPS Section III and Algorithm 1, https://arxiv.org/html/1905.10691 . Inspected official code https://github.com/obastani/model-predictive-shielding , commit 8d74b38f809ea39ea54dfa028d9498767a6f8650, python/spire/control/shield.py: ShieldPolicy.act_shield and _is_recoverable. It tests backup-policy reachability of a safe invariant region. It does not in that code test whether the learned navigation policy itself makes finite-window goal progress after handback. We do NOT execute or reproduce its cartpole/bicycle benchmarks, and do not claim its safety theorem applies to CV crowd forecasts.

Read Bertsekas's primary unified MPC/DP article, Section 2.6 truncated rollout and Section 4 adaptive rollout, https://web.mit.edu/dimitrib/www/Bertsekas_NMPC_IFAC.pdf . Prefix optimization plus a nominal-policy tail is established. Therefore **ordinary variable-prefix rollout is a required same-information, same-budget baseline**, not Ours.

Gatekeeper (https://arxiv.org/abs/2211.14361) already retains valid committed backup trajectories. Contractive/flexible-step MPC already handles temporary increase followed by finite-step decrease. Neither retaining plans nor allowing negative progress is claimed new.

The only candidate difference tested here is an explicit safe/productive nominal-policy re-entry condition, rather than ranking trajectories by aggregate rollout task return alone or returning after a fixed prefix. Publication-level novelty remains UNVERIFIED; an empirical win would require a further exact closest-method check, not automatic METHOD_ENTRY_FOUND. If ordinary rollout absorbs its gain, KILL this mechanism, not modify durations or thresholds to rescue it.
