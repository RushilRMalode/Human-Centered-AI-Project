# Marked Dual-Kernel Hawkes: Identifiability and Consistency

## 1 Setup

### 1.1 The paper's unmarked model

Agarwal et al. (2024) model a user's arrivals on an item as a Hawkes process with intensity

$$
\lambda(t) = \mu + \sum_{t_l < t} \left[\alpha^1 e^{-\beta^1(t-t_l)} + \alpha^2 e^{-\beta^2(t-t_l)}\right], \qquad 0 \le \beta^2 < \beta^1. \tag{1}
$$

The two components are interpreted as *System-1* (fast-decaying impulsive excitation) and *System-2* (slow-decaying deliberate utility). Their Theorem 1 shows that under a session-independence assumption and a strict inequality $\beta^1 > \beta^2$, the parameters $(\mu, \alpha^1, \alpha^2, \beta^1, \beta^2)$ are identifiable from the observation distribution. The proof reduces to Teicher's (1961) classical identifiability of finite mixtures of exponentials (paper's Appendix Lemma 2): the triggering function $\kappa(t) = \beta^1 \alpha^1 e^{-\beta^1 t} + \beta^2 \alpha^2 e^{-\beta^2 t}$, viewed as a probability density after normalising, is a mixture of two exponentials, which is identifiable iff the two rates are distinct. **When $\beta^1 = \beta^2$ the two components collapse and the unmarked model is provably not identifiable beyond their sum.**

### 1.2 The marked extension

We extend (1) with two pieces of information per event: a *label* $z_k \in \{1, 2\}$ and a *mark* $y_k \in \mathcal{Y}$. Formally, we use separate base rates $\mu_1, \mu_2$:

$$
\lambda(t) = (\mu_1 + \mu_2) + \sum_{t_l < t} \left[\alpha^1 \beta^1 e^{-\beta^1(t-t_l)} + \alpha^2 \beta^2 e^{-\beta^2(t-t_l)}\right]. \tag{2}
$$

(We use the normalised kernel $\alpha_j \beta_j e^{-\beta_j t}$ — integral $\alpha_j$ — so that $\alpha_j$ is the expected offspring count per event in the $j$-th channel.)

Each event $k$ has a latent label $z_k \in \{1, 2\}$ generated as follows:

- If $k$ is an **immigrant** (its intensity contribution comes from the base $\mu_1+\mu_2$), then $z_k = 1$ with probability $\mu_1/(\mu_1+\mu_2)$ and $z_k = 2$ otherwise.
- If $k$ is the **offspring** of some earlier event $l$ via kernel $j$ (intensity contribution $\alpha^j \beta^j e^{-\beta^j(t_k-t_l)}$), then $z_k = j$.

The mark is drawn from a system-specific parametric family:

$$
y_k \mid z_k = j \;\sim\; F_j(\,\cdot\,; \theta_j), \qquad j \in \{1, 2\}. \tag{3}
$$

A concrete instantiation used in the experiments is a mixed-type product $F_j = \mathrm{NegBin}(\lambda_j, r_j) \times \mathrm{LogNormal}(m_j, \sigma_j^2) \times \mathrm{Bernoulli}(p_j)$, representing session event count, dwell time, and binary like/share indicator respectively. Each factor is an identifiable parametric family (Teicher 1963), so product mixtures of $F_1, F_2$ are identifiable.

The full parameter vector is

$$
\theta = (\mu_1, \mu_2, \alpha^1, \alpha^2, \beta^1, \beta^2, \theta_1, \theta_2).
$$

## 2 Assumptions

**(A1) Positive rates.** $\mu_1, \mu_2 > 0$ and $\alpha^j, \beta^j > 0$ for $j = 1, 2$.

**(A2) Stability.** $\alpha^1 + \alpha^2 < 1$, so the expected branching ratio is less than one and the Hawkes process is stationary and ergodic (Guo et al. 2018).

**(A3) Label anchor.** One of the following holds:
- *(Timing anchor)* $\beta^1 > \beta^2$, or
- *(Mark anchor)* $\beta^1 = \beta^2$ and $F_1, F_2$ are ordered in a parameter-fixed way (e.g. $m_1 < m_2$, fixing the "System 1 has shorter mean dwell than System 2" convention).

This assumption removes the trivial label-swap symmetry $(1 \leftrightarrow 2)$.

**(A4) Mark identifiability.** $F_1, F_2$ belong to a parametric family whose finite mixtures are identifiable (Teicher 1961, 1963), and $\theta_1 \ne \theta_2$.

**(A5) Non-degeneracy.** Either $\beta^1 \ne \beta^2$, or the mixture weights satisfy $\alpha^1/(\alpha^1+\alpha^2) \ne \mu_1/(\mu_1+\mu_2)$.

(A5) is the one we add on top of the paper's assumptions. The paper implicitly lives in the $\beta^1 \ne \beta^2$ side; (A5) lets the mark channel cover the other side.

## 3 Main result

**Theorem 1 (Identifiability).** *Under Assumptions (A1)–(A5), the parameter $\theta$ is identifiable from the stationary distribution of the marked arrival process $(t_k, y_k)_{k \ge 1}$.*

### 3.1 Proof

The proof proceeds in three steps. Write $\lambda(t) = \mu + \kappa * dN(t)$ with $\mu = \mu_1 + \mu_2$ and triggering function $\kappa(t) = \alpha^1 \beta^1 e^{-\beta^1 t} + \alpha^2 \beta^2 e^{-\beta^2 t}$. Let $\mathbb{P}_\theta$ be the observation distribution induced by parameter $\theta$, and let $\mathbb{P}_\theta = \mathbb{P}_{\tilde\theta}$ for some alternate parameter $\tilde\theta$.

**Step 1 — timing identifies $\mu$, $\beta^1, \beta^2$, and the aggregates $\alpha^1 + \alpha^2$ (and $\alpha^j$ individually when $\beta^1 \ne \beta^2$).**

The unmarked arrival-only observation distribution is determined by $(\mu, \kappa)$. Matching intensity functions gives $\mu = \tilde\mu$. Then $\kappa = \tilde\kappa$ as functions on $\mathbb{R}_+$.

*Case (a): $\beta^1 \ne \beta^2$.* By the paper's Lemma 2 (Teicher 1961), the mixture of exponentials $\kappa(t)/\int \kappa$ is identifiable, so $(\beta^1, \beta^2)$ and the ratio $\alpha^1/(\alpha^1+\alpha^2)$ are identified. Combined with $\int \kappa = \alpha^1 + \alpha^2$ (also determined by $\kappa$), each $\alpha^j$ is identified individually.

*Case (b): $\beta^1 = \beta^2 \equiv \beta$.* Then $\kappa(t) = (\alpha^1 + \alpha^2) \beta e^{-\beta t}$ is a single exponential, which identifies only the sum $\alpha^1 + \alpha^2$ and $\beta$. The individual $\alpha^j$ are **not** identified from timing alone. This is where marks are required.

**Step 2 — marks identify $\theta_1, \theta_2$ and the long-gap weight $\mu_1/(\mu_1+\mu_2)$.**

Consider events separated from their predecessors by long inactivity: formally, along subsequences $(t_{k_n})$ with $t_{k_n} - t_{k_n-1} \to \infty$, the kernel contribution $\kappa \ast dN(t_{k_n}) \to 0$, so $\lambda(t_{k_n}^-) \to \mu$. Such events are asymptotically immigrants, and conditional on being immigrant, their labels have distribution $\mathrm{Cat}(\mu_1/\mu, \mu_2/\mu)$. The marginal mark distribution along this subsequence is therefore

$$
q_{\mathrm{long}}(y) \;=\; \frac{\mu_1}{\mu_1+\mu_2} F_1(y; \theta_1) \;+\; \frac{\mu_2}{\mu_1+\mu_2} F_2(y; \theta_2). \tag{4}
$$

Since (A4) guarantees that finite mixtures of the $F_j$-family are identifiable, (4) uniquely determines the pair $(F_1, F_2)$ up to a label permutation, together with the weight $\mu_1/(\mu_1 + \mu_2)$. Assumption (A3) fixes the permutation. Thus $\theta_1$, $\theta_2$, and $\mu_1/(\mu_1+\mu_2)$ are identified; combined with $\mu = \mu_1 + \mu_2$ from Step 1, we obtain $\mu_1$ and $\mu_2$ individually.

**Step 3 — marks also identify $\alpha^1, \alpha^2$ individually (needed only in Case (b)).**

Consider the complementary subsequence of events that arrive *during* a burst — formally, sequences with $t_{k_n} - t_{k_n-1} \to 0$ where many recent events are still exciting. Along these, the kernel contribution $\kappa \ast dN(t_{k_n})$ dominates, so $\lambda(t_{k_n}^-) \approx \kappa \ast dN(t_{k_n})$. The conditional probability that $z_{k_n} = j$ is proportional to the component-$j$ kernel contribution, which along this subsequence tends to $\alpha^j/(\alpha^1+\alpha^2)$. Hence the marginal mark distribution along the burst subsequence is

$$
q_{\mathrm{short}}(y) \;=\; \frac{\alpha^1}{\alpha^1+\alpha^2} F_1(y; \theta_1) \;+\; \frac{\alpha^2}{\alpha^1+\alpha^2} F_2(y; \theta_2). \tag{5}
$$

By (A4) applied to (5) — with $F_1, F_2$ already known from Step 2 — the weight $\alpha^1/(\alpha^1+\alpha^2)$ is identified. Combined with Step 1 ($\alpha^1 + \alpha^2$ known), we recover $\alpha^1$ and $\alpha^2$ individually.

The non-degeneracy assumption (A5) is exactly what ensures that $q_{\mathrm{short}} \ne q_{\mathrm{long}}$ in Case (b), so that Steps 2 and 3 give genuinely different equations to solve. If $\alpha^1/(\alpha^1+\alpha^2) = \mu_1/(\mu_1+\mu_2)$ and $\beta^1 = \beta^2$, the two equations coincide and the individual $\alpha^j$ cannot be separated.

Together, Steps 1–3 identify every coordinate of $\theta$. $\square$

### 3.2 Why this is stronger than the paper

The paper's Theorem 1 requires $\beta^1 \ne \beta^2$ (strict). Theorem 1 above requires **either** $\beta^1 \ne \beta^2$ **or** a generic non-degeneracy condition on the weights. The $\beta^1 = \beta^2$ regime — provably non-identifiable in the paper's setting — becomes identifiable as soon as marks $F_1 \ne F_2$ are observed. This is the formal content of the intuition that *marks provide a second, independent identifiability channel*.

## 4 Consistency

**Corollary 1 (MLE Consistency).** *Let $\hat\theta_T$ denote the maximum-likelihood estimator of $\theta$ over a compact parameter space $\Theta$ satisfying (A1)–(A5), computed from observations on $[0, T]$. Then $\hat\theta_T \xrightarrow{\;\mathbb{P}\;} \theta^\star$ as $T \to \infty$, where $\theta^\star$ is the true parameter.*

**Proof sketch.** Follows the pattern of the paper's Theorem 2 (Gao et al. 2018). The marked-Hawkes log-likelihood

$$
\ell_T(\theta) = \sum_{k: t_k \le T} \log\Big[\!\sum_{\text{source}} w_{\mathrm{source}}(t_k; \theta) \, f_{\mathrm{source}}(y_k; \theta)\Big] - \int_0^T \lambda(t; \theta) \, dt
$$

is a continuous function of $\theta \in \Theta$ (compactness). By ergodicity of the marked Hawkes (Guo et al. 2018, extended to marks via the marked-TPP martingale framework), $T^{-1} \ell_T(\theta) \to \ell_\infty(\theta)$ almost surely, uniformly on $\Theta$. The population log-likelihood $\ell_\infty$ is maximised uniquely at $\theta^\star$ by Theorem 1 (identifiability) and standard M-estimation arguments (van der Vaart 1998, Theorem 5.7) deliver consistency of the M-estimator $\hat\theta_T$. $\square$

## 5 Failure modes

The converse direction of Theorem 1 is just as informative. Identifiability is lost in precisely three regimes:

**(F1) Joint collapse.** $\beta^1 = \beta^2$ **and** $F_1 = F_2$. Neither timing nor marks separate the components; only the aggregates $(\mu_1+\mu_2, \alpha^1+\alpha^2, \beta^1)$ are identifiable.

**(F2) Degenerate mixture weights.** $\beta^1 = \beta^2$ and $\alpha^1/(\alpha^1+\alpha^2) = \mu_1/(\mu_1+\mu_2)$. Here $q_{\mathrm{short}} = q_{\mathrm{long}}$ and Steps 2 and 3 provide the same equation, so the individual $\alpha^j$ and $\mu_j$ cannot be separated (though $F_1, F_2$ and the common mixing ratio *are* identifiable). This is a measure-zero subset of the parameter space and, crucially, is *violated generically* — any small perturbation of parameters restores identifiability.

**(F3) Non-ergodicity.** $\alpha^1 + \alpha^2 \ge 1$: the branching ratio is not strictly less than one, and the stationary distribution does not exist. Neither Theorem 1 nor its consistency corollary apply.

The hierarchy of these failure modes is worth emphasising:
- (F1) is *fundamental* — no amount of data rescues identifiability when every observable distribution coincides.
- (F2) is *non-generic* — it's a single algebraic equation on parameters and fails only on a negligible set.
- (F3) is a modelling violation, not an identifiability failure per se.

## 6 Discussion

**Summary of the contribution.** Theorem 1 extends the paper's identifiability result by adding a mark channel, giving a *disjunctive* sufficient condition: identifiability holds whenever at least one of $\{\beta^1 \ne \beta^2, \;F_1 \ne F_2\}$ holds nontrivially. The two channels are *independent* in the sense that each alone is sufficient: timing anchors the proof when $\beta^1 \ne \beta^2$, marks anchor it otherwise. The experiments in §7 of the report give direct empirical content to this statement — Experiment 2 (β-gap sweep) shows that the marked model is flat across the full range of $\beta^2$ while the unmarked model degrades monotonically as $\beta^2 \to \beta^1$.

**Relation to prior mixture-Hawkes work.** Xu & Zha (NeurIPS 2017) give identifiability for Dirichlet mixtures of Hawkes processes aimed at event-sequence *clustering*. Their setup allocates entire sequences to latent clusters; our latents $z_k$ live at the *event* level and interact with the branching structure. The theorem and the branching-EM inference algorithm we use (§5 of the report) are tailored to this event-level mixture structure.

**What the theorem does not do.** The theorem is asymptotic and does not state finite-sample rates. It also assumes a fixed (known) parametric family for $F_1, F_2$ and fixed known kernel class (exponential). Relaxing to a nonparametric kernel or a nonparametric mark family are separate (and harder) problems.

## References

- Agarwal, Usunier, Lazaric, Nickel (2024). System-2 Recommenders. arXiv:2406.01611.
- Teicher (1961). Identifiability of mixtures. Ann. Math. Stat. 32, 244–248.
- Teicher (1963). Identifiability of finite mixtures. Ann. Math. Stat. 34, 1265–1269.
- Gao, Zhu, Zha, Song (2018). Consistency of maximum likelihood estimator for Hawkes processes. arXiv:1810.02955.
- Guo, Zhu, Zha, Song (2018). See above.
- Veen, Schoenberg (2008). Estimation of space-time branching process models in seismology using an EM-type algorithm. JASA.
- Xu, Zha (2017). A Dirichlet mixture model of Hawkes processes for event sequence clustering. NeurIPS.
- van der Vaart (1998). Asymptotic Statistics. Cambridge UP.
