# Course Project Proposal

## *Marked System-2 Recommenders: Adding Engagement Signals to the Dual-Kernel Hawkes Process, with Identifiability*

**Grounded in:** Agarwal, Usunier, Lazaric, Nickel. *System-2 Recommenders: Disentangling Utility and Engagement in Recommendation Systems via Temporal Point-Processes.* FAccT 2024. [arXiv:2406.01611](https://arxiv.org/abs/2406.01611)

**Duration:** 4 weeks
**Weight:** 40% of course grade

---

## 1. The core idea

The paper throws away an enormous amount of information. In its model, each "click event" is reduced to a single timestamp — and the two psychological systems (impulsive S1, deliberate S2) are separated purely through the *decay rates* β¹ > β² of an exponential Hawkes kernel. But every modern platform also observes rich per-event **engagement signals**: watch-time, like, share, comment, save. These behave *systematically differently* between S1 and S2 consumption. A doom-scroll click has low dwell + no share; a deliberate click has long dwell + save/like.

**This project extends the dual-kernel Hawkes to a *marked* point process** where each event also emits an engagement vector, and **proves that the marks give a second, independent source of disentanglement** — one that still works even when the decay-rate gap collapses (the regime where the paper's method provably fails).

This is exactly the direction of "use engagement signals and show it's MLE-learnable."

---

## 2. Why this is novel (not a reapplication)

- Marked Hawkes processes exist in the statistics literature, but **nobody has used the mark channel to identify two latent psychological systems**.
- The resulting identifiability theorem is **strictly stronger** than the paper's: the marked model is identifiable even when β¹ = β² — a case where the original model is *provably not* identifiable.
- The paper's own discussion flags richer engagement modeling as an open direction; this project answers it head-on.

---

## 3. The model

### 3.1 Events
Each observed event is a tuple `(tₖ, iₖ, jₖ, mₖ)` where `mₖ ∈ ℝ^L` is an engagement mark vector (e.g., watch-time in seconds, like-indicator, share-indicator).

### 3.2 Arrival intensity — unchanged from the paper
```
λᵢⱼ(t) = μᵢⱼ + Σ_{t′<t} ( α¹ᵢⱼ · e^{−β¹(t−t′)} + α²ᵢⱼ · e^{−β²(t−t′)} )
```
with `α¹ᵢⱼ = φ(⟨v¹ᵢ, u¹ⱼ⟩)`, `α²ᵢⱼ = φ(⟨v²ᵢ, u²ⱼ⟩)`, `φ(x) = (x+1)/4`.

### 3.3 Mark generation — new
Introduce a latent system indicator `zₖ ∈ {0, 1, 2}` denoting {base, S1-triggered, S2-triggered}. By the Hawkes cluster representation,

```
P(zₖ = 0 | history) = μ / λ(tₖ)
P(zₖ = 1 | history) = (α¹ · K₁(tₖ − t*)) / λ(tₖ)
P(zₖ = 2 | history) = (α² · K₂(tₖ − t*)) / λ(tₖ)
```

where K_s(·) is the exponential kernel of component s. Given `zₖ = s`, the mark is drawn from a system-specific distribution:

```
mₖ | zₖ = s  ~  F_s(· ; θ_s)
```

**Mark model — mixed-type product.** Rather than a pure Gaussian, each mark is the product of three identifiable parametric families reflecting the realistic structure of engagement logs:

```
Lₖ  ~  NegBin(ηₛ,L)      (click/event count in session)
Dₖ  ~  LogNormal(ηₛ,D)   (session dwell / watch-time)
Cₖ  ~  Bernoulli(ηₛ,C)   (like / share / save indicator)
```

with system-specific natural-parameters `θ_s = (ηₛ,L, ηₛ,D, ηₛ,C)`. All three are identifiable parametric families, so the mixture-identifiability argument in §4 goes through unchanged. Compared to the Gaussian first-choice, this construction (i) matches the shape of real engagement signals (over-dispersed counts, heavy-tailed dwell, binary reactions), and (ii) gives weighted-MLE closed forms for the M-step in the EM algorithm below.

**Label-switching constraint.** To avoid label ambiguity (1↔2) during training, impose `β¹ > β²` by a hard clamp (as in the paper) *and* order the mark distributions — e.g. constrain `ηₛ,D` so that S1 sessions are stochastically shorter than S2 sessions (a direct operationalization of the "fast impulsive vs. slow deliberate" intuition).

### 3.4 The marginal log-likelihood
After marginalizing over `zₖ`, the per-event mark log-likelihood becomes a finite mixture:

```
log p(mₖ | history) = log Σ_s [ ω_s(tₖ) · f_s(mₖ; θ_s) ]
```

where `ω_s(tₖ) = (αˢ · Kₛ(tₖ − t*)) / λ(tₖ)` is the intensity-weight of component s at event tₖ. The full joint log-likelihood combines (i) the Hawkes arrival term (same as the paper) and (ii) this mark mixture term.

### 3.5 Inference: branching-EM
Direct gradient MLE on this mixture is known to be local-minima prone. Instead, we use the **branching-structure EM** of Veen & Schoenberg (2008), extended with the latent system label `zₖ`:

- **Latents per event k**: parent assignment `pₖ ∈ {0, 1, …, k−1}` (0 = immigrant) *and* label `zₖ ∈ {1, 2}`.
- **E-step** (closed form, exponential kernels): branching responsibilities
  ```
  qₖ,ⱼ^{(s)}  ∝  qₖ(s) · αˢᵢⱼ · βˢ · exp(−βˢ(tⱼ − tₖ))
  ```
  and label posteriors `qₖ(s)` via softmax over the log mark-likelihood plus expected offspring contribution.
- **M-step**:
  - `μ ← Σₖ qₖ,₀ / T`
  - Weighted MLE per system `s` for mark GLM parameters (`ηₛ,L`, `ηₛ,D`, `ηₛ,C`) using `qₖ(s)` as weights — closed form for NegBin/LogNormal/Bernoulli GLMs.
  - Gradient ascent on `(βˢ, u¹, u², v¹, v²)` using the expected complete log-likelihood; exponential kernels give O(N) state recursion per gradient step.

This is algorithmically standard for latent-mixture Hawkes models and gives posteriors `qₖ(zₖ)` as a free side-output, useful for downstream evaluation.

---

## 4. Theoretical contribution

**Theorem (new).** Under the paper's Assumptions 1–2, *plus* assuming the mark distributions `F₁, F₂` belong to an identifiable parametric family with `θ₁ ≠ θ₂`, the marked dual-kernel Hawkes process is identifiable from the joint log-likelihood — **even when β¹ = β²**.

**Proof sketch.**
- The paper's Theorem 1 separates α¹ from α² via Teicher's theorem on identifiability of mixtures of exponentials, which *requires* β¹ ≠ β².
- Adding marks gives an *independent* mixture-identifiability path. At each event, the conditional mark distribution is a 2-component finite mixture `(α¹/(α¹+α²)) · F₁ + (α²/(α¹+α²)) · F₂`, whose weights determine the α¹/α² ratio.
- Mixtures of identifiable parametric families are themselves identifiable (Teicher 1963), so the mixture weight — hence α¹/α² — is recovered.
- Combined with base-intensity identification from inactivity intervals (as in the original paper), all parameters follow.

**Corollary (MLE consistency).** Under the same regularity conditions as Theorem 2 of the paper, MLE of the marked dual-kernel Hawkes is consistent. The proof structure of the paper transfers directly, since the marked log-likelihood is a smooth function of the parameters and the Gao et al. (2018) consistency result applies to the joint model.

**Corollary (Failure modes — what the marks cannot rescue).** The marked model is *not* identifiable beyond label swapping in the following regimes:

1. **Joint collapse**: `β¹ = β²` *and* `F₁ ≡ F₂` (same mark distribution). Neither the timing signal nor the mark signal separates the two kernels; only the aggregate intensity is identifiable.
2. **Degenerate covariates**: if `v¹`, `v²` lie in a proper subspace of ℝᵈ, the item embeddings `u¹`, `u²` are only identified up to that subspace's orthogonal complement. This is the same spanning requirement as the paper.
3. **Adaptive-policy confounding**: if the session covariates `vₖ` depend on the realized history via a policy correlated with `zₖ`, the observational equivalence class expands — different `(u¹, u²)` pairs can explain the data by shifting dependence into the policy. Not resolvable without randomization or instrumental variation.

The two positive identifiability paths (β-gap and mark-separation) *together* cover all non-degenerate regimes. This is the sharper statement of the contribution: **marks and timing are complementary identifiability channels.**

The theoretical writeup is ~4–5 pages and is the main novel artifact.

---

## 5. Experiments (all synthetic, all tractable)

### Experiment 1 — Reproduce Figure 1 in marked form
Generate synthetic data from the marked model. Show MLE recovers all parameters `(μ, α¹, α², β¹, β², θ₁, θ₂, v¹, v², u¹, u²)` and error → 0 as sample size grows.

### Experiment 2 — Flagship: β-gap collapse
Reproduce the paper's Figure 2 (vary β² ∈ [0, 1], fix β¹ = 1), with and without marks. Plot estimation error for both models.
**Expected result:** the unmarked model's error spikes as β² → β¹ (replicating the paper). The marked model's error stays flat, because marks carry the disentanglement information when timing can't.

**This single plot is the whole story of the project.**

### Experiment 3 — Sample-complexity gain
Fix β² = 0.5 (moderate gap). Plot estimation error vs number of sessions for unmarked vs marked. Show marks strictly reduce the sample complexity required to hit a fixed error threshold.

### Experiment 4 — Mark-informativeness ablation
Vary the gap between mark-distribution means `‖θ₁ − θ₂‖`. When marks are uninformative (θ₁ ≈ θ₂), the marked model should degrade toward the unmarked model. When marks are highly informative, the β-gap requirement should disappear entirely.

---

## 6. Week-by-week plan

### Week 1 — Derivation + implementation
- Derive the marked log-likelihood for the mixed-type mark model (NegBin / LogNormal / Bernoulli).
- Implement **branching-EM** in PyTorch (closed-form E-step over parent + label latents; M-step via weighted MLE for mark GLMs and gradient ascent for `(βˢ, u¹, u², v¹, v²)`).
- Enforce label-switching constraints: `β¹ > β²` clamp + ordered mark distributions.
- Sanity check: run with uninformative marks — should match the unmarked case from the paper.

**Deliverable:** working implementation + reproduction sanity-check figure.

### Week 2 — The theorem
- Write out the identifiability proof in full.
- Write the MLE-consistency corollary, leveraging Gao et al. 2018.
- 3–4 page theoretical section of the report.

**Deliverable:** theory section draft.

### Week 3 — The three experiments
- Run Experiments 1, 2, 3. Experiment 2 is the flagship.
- All figures with confidence bands over random seeds.

**Deliverable:** three experiment figures, with Exp 2 as the headline plot.

### Week 4 — Ablation + write-up
- Run Experiment 4 (mark-informativeness ablation).
- 8-page report in the paper's style: problem setup, model, theorem, experiments, discussion.
- Clean codebase + README.

**Deliverable:** final PDF + GitHub repo.

---

## 7. Deliverables Summary

| # | Artifact | Due |
|---|---|---|
| 1 | PyTorch marked-Hawkes branching-EM implementation | End of week 1 |
| 2 | Identifiability theorem + consistency corollary | End of week 2 |
| 3 | Experiments 1–3 (flagship β-gap plot) | End of week 3 |
| 4 | Experiment 4 (mark-informativeness ablation) | Week 4 |
| 5 | Final report (8 pages) | Week 4 |
| 6 | Reproducible codebase + README | Week 4 |

---

## 8. Why this is feasible in a month

- **No datasets to ingest** — all experiments are synthetic, exactly as in the paper.
- **No LLM APIs, no Kuaishou, no ingestion engineering.**
- **Mathematical core is compact**: one new theorem + corollary, proof leverages Teicher 1963 and Gao 2018.
- **Code is incremental**: extend the paper's likelihood with one additional mixture term — the mark model.
- **Headline result is a single plot**: Figure 2 with two curves instead of one.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mixture + Hawkes optimization unstable | Medium | Use branching-EM (§3.5) instead of direct MLE — closed-form E-step, weighted-MLE M-step for mark GLMs; initialize θ_s via k-means on observed marks |
| Label switching (S1/S2 swap across seeds) | Medium | Hard clamp `β¹ > β²` + ordering constraint on mark dwell distributions; report results modulo label permutation |
| Proof gap when β¹ = β² exactly | Low | Mark-mixture identifiability path (Teicher 1963) handles this regime directly; β-gap is an independent channel |
| Mark model too rigid | Low | Mixed-type product (NegBin × LogNormal × Bernoulli) covers counts, durations, and binary signals — all identifiable families |
| Experiment 2 result isn't clean | Low | Even a partial improvement at β² → β¹ is a publishable result |

---

## 10. What to cut if running late (ordered)

1. Experiment 4 (mark-informativeness ablation) — nice-to-have.
2. Experiment 3 (sample-complexity gain) — supporting evidence.
3. Full MLE-consistency corollary proof — state the result and cite Gao 2018.

**The theorem + Experiment 2 must ship.** Those two alone are a defensible 40% project.

---

## 11. Why this is a defensible 40% project

- **New theorem** strictly stronger than the paper's — works where the original model provably fails.
- **Clean empirical story**: one headline plot shows the marked model succeeds exactly where the unmarked model breaks.
- **Directly addresses a limitation** the paper's discussion section flags as open.
- **Tight scope** — four weeks, synthetic-only, one core experiment, one core theorem.
- **Graceful failure modes**: if the theorem is subtly wrong, the experiments still reveal *where* marks help; if experiments are ambiguous, the theorem is still a contribution.

---

## 12. Key references

**Core paper**
- Agarwal, Usunier, Lazaric, Nickel. *System-2 Recommenders.* FAccT 2024. [arXiv:2406.01611](https://arxiv.org/abs/2406.01611)

**Theoretical background**
- Kleinberg, Mullainathan, Raghavan. *The Challenge of Understanding What Users Want: Inconsistent Preferences and Engagement Optimization.* 2022. [arXiv:2202.11776](https://arxiv.org/pdf/2202.11776)
- Gao, Zhu, Zha, Song. *Consistency of maximum likelihood estimator for Hawkes processes.* 2018.
- Teicher. *Identifiability of mixtures.* Ann. Math. Stat., 1961 / 1963.

**Marked / Neural Hawkes processes**
- Daley, Vere-Jones. *An Introduction to the Theory of Point Processes.* 2003.
- Zuo et al. *Transformer Hawkes Process.* ICML 2020.
- Mei, Eisner. *The Neural Hawkes Process.* NeurIPS 2017.

**Inference algorithms**
- Veen, Schoenberg. *Estimation of space–time branching process models in seismology using an EM-type algorithm.* JASA, 2008.
- Xu, Zha. *A Dirichlet Mixture Model of Hawkes Processes for Event Sequence Clustering.* NeurIPS 2017.

**Dual-process theory**
- Kahneman. *Thinking, Fast and Slow.* 2011.
