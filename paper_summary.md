# System-2 Recommenders: Disentangling Utility and Engagement in Recommendation Systems via Temporal Point-Processes

**Authors:** Arpit Agarwal, Nicolas Usunier, Alessandro Lazaric, Maximilian Nickel (FAIR at Meta, 2024)

---

## 1. The Core Problem

Modern recommender systems (news feeds, short-video platforms, social networks) are trained to maximize **engagement signals** — likes, shares, clicks, watch time. The paper's central worry: engagement is a *noisy proxy* for what a user actually values.

The authors draw directly on **dual-process psychology** (Kahneman's *Thinking, Fast and Slow*):

- **System-1** → fast, impulsive, habit/affect-driven. Clicking a clickbait headline, doom-scrolling.
- **System-2** → slow, deliberative, aligned with long-term preferences. Reading an in-depth article, saving a tutorial.

A typical engagement-maximizing model conflates both. That can lead to the well-documented phenomenon where a platform "optimizes" user time-on-app while users themselves report regret. The paper's two motivating questions:

1. Are engagement signals (likes, shares, watch-time) aligned with true user utility?
2. Can we use those same signals to *estimate* utility and recommend content based on it?

---

## 2. The Key Idea: Self-Exciting Hawkes Processes

The paper's technical contribution is modeling a user's interaction arrivals with an item as a **marked Hawkes process** whose intensity is the sum of two components, giving the two "systems" a precise mathematical identity.

### Background — Temporal Point Processes (TPPs)
A TPP is a stochastic process where realizations are sequences of event times. It's fully characterized by the conditional intensity λ(t), loosely the instantaneous rate of events given the history.

A **Hawkes process** is self-exciting: each past event increases the current intensity:

```
λ(t) = μ + Σ α · exp(−β(t − tₖ))
```

- α = intensity of the jump caused by a past event
- β = decay rate (how fast the effect fades)

### The Dual Model (core equation, page 5)
For user *i* interacting with item *j*, the intensity is:

```
λᵢⱼ(t) = μᵢⱼ + Σ (α¹ᵢⱼ · e^{−β¹(t − t′)} + α²ᵢⱼ · e^{−β²(t − t′)})
```

with the assumption `0 ≤ β² < β¹`. That single ordering is what carries all the psychology:

- **System-1 kernel** (α¹, β¹): fast decay → impulsive, short-term moreishness. Drives binge sessions.
- **System-2 kernel** (α², β²): slow decay → deliberate return based on long-term utility.

User–item engagement and utility are modeled as bilinear in embeddings:

- α¹ᵢⱼ = φ(v¹ᵢᵀ uⱼ¹)   (moreishness / System-1)
- α²ᵢⱼ = φ(v²ᵢᵀ uⱼ²)   (utility / System-2)

where φ maps to [0, 0.5] to keep the Hawkes process stationary/ergodic.

### The Goal
Learn u¹, u² (item embeddings) and v¹, v² (user embeddings) from observed interactions. Then **rank items using only the System-2 score** ⟨v²ᵢ, u²ⱼ⟩ via deterministic ranking — producing a "System-2 recommender" that ignores the impulsive component.

---

## 3. Identifiability & Consistency (the theory)

A legitimate concern: with two kernels summed, why can't the model attribute impulsive behavior to the utility term or vice-versa? The paper proves it can't, given some conditions.

- **Lemma 1** (Gao et al., 2018): Hawkes processes are identifiable up to their intensity.
- **Assumption 1**: link function φ(x) = (x+1)/4 and the normalization ‖u²‖₂ ≤ 1, making the mapping one-to-one.
- **Assumption 2** (Session independence): sessions Sₜ given t don't depend on the realization history 𝓗. This is what rules out random-bet explanations.
- **Theorem 1**: Under Assumptions 1 & 2, all parameters (μ, α¹, α², β¹, β²) are uniquely identifiable.
- **Theorem 2** (Consistency): MLE is consistent — with enough data the learned parameters converge to the true ones, so the two systems can be provably disentangled.

The proof decomposes the intensity on disjoint intervals (isolating pure-System-1 vs mixed behavior) and reduces to the identifiability of mixtures of exponentials (Teicher, 1961 — see Appendix Lemma 2, page 17).

---

## 4. Experiments (Section 4)

All experiments are **synthetic** — the authors generate data from a known ground truth and check that their MLE procedure recovers it and produces better recommendations.

### Setup
- **Number of items** *m* = 1000; **embedding dim** *d* = 10.
- **Link function** φ(x) = (x + 1)/4, mapping to [−0.5, 0.5].
- Ground-truth user/item embeddings v¹, u¹ drawn uniformly in [−1, 1].
- **Critical setup for the hypothesis:** v² = Q₁ v¹ and u² = Q₂ u¹ with Q₁, Q₂ random orthogonal rotations, so utility and engagement preferences are *correlated but distinct* — exactly the "long-lasting effect on return probability" scenario.
- β¹ = 1, β² = 0.1 (fast vs slow decay).
- **Session generation:** each "session" is a realization of a Hawkes process thinned by a probability *p*; *p* = 0.04. The session's duration is drawn uniformly from [1, 6]. So users show up, interact for a short window, leave, return.
- **Training:** jointly learn μ, α¹, α², β¹, β² plus all embeddings by log-likelihood maximization with Adam (lr 0.002, batch 16). They clamp β¹ ≥ 1 and β² ≤ 1 to impose the required ordering for identifiability.
- Real item embeddings assumed known (u¹, u²) — the discussion acknowledges this and notes matrix factorization / content features as extensions.

### Experiment 1 — Sample size vs estimation error (Figure 1, page 9)
They sweep the number of generated sessions from ~50 up to ~1000 and plot estimation error (L2 distance) for each parameter.

**Result:** all parameter errors (μ, α¹, α², β¹, β², and both embeddings) decay toward zero. This empirically confirms Theorem 2's consistency. The two systems are separable with just ~1000 sessions per user in a 10-dim, 1000-item regime.

### Experiment 2 — Effect of the gap |β¹ − β²| (Figure 2, page 9)
They fix β¹ = 1 and vary β² ∈ [−0.5, 1] (they also test the degenerate β² = β¹ case).

**Result:** as the gap shrinks, estimation error grows — when β¹ ≈ β² the two exponentials become nearly collinear and the MLE's signal to disentangle weakens. This pins down a practical requirement: the two systems must decay on measurably different timescales for the method to work.

Two utility plots then compare **engagement optimization** vs **(true) utility maximization** across the β²-gap axis, showing that as the gap grows, ranking by ⟨v², u²⟩ increasingly dominates ranking by engagement.

### Experiment 3 — Ranking quality by similarity to u¹ vs u² (Section 4.3, Figure 2)
They build a QR factorization of the ground-truth item matrices. For each target similarity score *s*, they construct an item whose System-1 similarity to the user is *s* and whose System-2 similarity is 1, and another with the roles swapped, then rank.

**Result:** the System-2 recommender ranks the truly-useful item (the one with high u² similarity) above the impulsive one. Engagement-maximization can't reliably distinguish the two because they look identical under a combined signal.

### Experiment 4 — Size of the item inventory (Section 4.4)
They scale *m* up while holding everything else fixed and check whether the utility gain from System-2 ranking persists.

**Result:** as inventory grows, the chance of finding "trap" items (high moreishness, low utility) grows, so the advantage of the System-2 recommender over the engagement-max baseline *widens*. Practically: the bigger the catalog (think TikTok), the more the disentanglement matters.

---

## 5. Discussion, Limitations, Related Work

- **Alternative objectives** (Section 5): instead of maximizing per-session utility one can maximize the *number of active sessions* or total time horizon — raising the same optimization question but keeping utility, not moreishness, as the driver of re-engagement.
- **Limitations** they flag (Section 6):
  1. Behavioral assumptions (dual systems) aren't validated on real platforms.
  2. Non-stationary arrivals (seasonality, content drift) aren't modeled — could be patched by adding a time-dependent μ(t).
  3. Assumes item embeddings are given; on real platforms these need to be learned from content features or past interactions jointly.
- **Related work** (Section 7): connects to misalignment literature (Milli 2021, 2023; Kleinberg 2022), fairness in rec systems, RL-based long-term engagement optimization (Zou 2019, Zhan 2019), and TPP-based user modeling (Jing & Smola 2017).

---

## Bottom Line
The paper gives a **principled way to pull apart "I clicked because it hooked me" from "I came back because I genuinely valued it"** using the different *temporal decay signatures* of the two behaviors, and proves that under modest assumptions this decomposition is unique. The experiments are synthetic-only, but they confirm: (a) MLE recovers the true parameters as data grows, (b) the ability to disentangle depends on the decay-rate gap, and (c) a System-2 recommender beats an engagement-max baseline more strongly as inventory scales — the regime that describes every real-world feed.
