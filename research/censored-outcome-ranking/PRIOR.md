# Primary Prior Check

Official Social NCE source, commit61630607c56b3a5a5d4ef5320aab15d10307ac3a: crowd_nav/snce/contrastive.py:24--96 computes contrastive representation loss against sampled motion events. It does not supply fixed-root continuation outcomes. We inspected it; we did not reproduce its benchmark or change our frozen encoder.

CLIC official repository https://github.com/clic-webpage/CLIC, commit99e239d0f0d954601a737720c289d2096d2ed8e4, src/agents/CLIC_torch.py:479--482 implements target-to-policy KL plus a gradient penalty. The paper https://arxiv.org/html/2502.07645 section6 derives set-based policy shaping, including policy-weighted targets. Set supervision itself is already covered. Our finite80 set control uses that target/KL primitive with the same residual model and common regularizer as the other arms; it is an adaptation, not a complete CLIC reproduction. No manipulation environment restoration.

RankQ https://arxiv.org/html/2605.11151 uses ranked real/noisy/random action training. Ordinary softplus ranking is a necessary control, not our contribution. Its full model/benchmark has not been executed.

The tested difference is an outcome-censored, winner-preserving decision constraint rather than fitting absolute returns, forcing every good action above every bad action, or fitting a probability distribution over all good actions. This does NOT establish novelty: max-margin structured prediction and selective constraints are established. Any paper claim requires a demonstrably additional navigation capability beyond these controls, followed by fresh confirmation and precise closest-prior comparison. If ordinary controls absorb it, stop immediately.

Fit labels are realized returns under one hidden world and old continuation policy. Their use is offline supervision only. Online correction consumes frozen legal GRU features, not human goals, future paths or actual-next tokens. All-failure roots provide no claimed one-step rescue supervision.
