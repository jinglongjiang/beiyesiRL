# Primary-source boundary

NEC (Pritzel et al., ICML2017), method2, DND equations1/5: per-action episodic embedding lookup, inverse-distance weighted return values. https://arxiv.org/html/1703.01988
The paper-linked Kaixhin EC implementation was read at models.py (_knn_search and DND lookup); it is third-party, not official DeepMind source. https://github.com/Kaixhin/EC
Its old FAISS/CUDA binding is not needed: scipy cKDTree provides the same established exact nearest-neighbor primitive. No claim to invent episodic control, no upstream benchmark reproduced.

RankQ (Choi and Xu, 2026), method3 equations4-6: TD plus synthetic-action ordering losses, including proximity to successful actions. https://arxiv.org/html/2605.11151
This reference instead retains a frozen controller and uses paired realized returns for its actual legal candidate set. That distinguishes the implementation but does not by itself establish a contribution.

The tested extra capability is transferable, context-offset-invariant correction of local action gaps, compared directly with absolute residual retrieval on identical data. Dueling-style value/advantage decomposition is already established. If the centered retrieval does not add navigation benefit beyond the ordinary memory and the strong fixed8 MPPI control, seal it. If it succeeds, audit exact contextual episodic-advantage precedents before any paper PASS.

No privileged state enters deployment lookup; simulator continuation labels are offline realized-world supervision, not true legal-observation conditional Q. This is not a causal reconstruction of missing historical training replay.
