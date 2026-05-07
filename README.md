# Marked System-2 Recommenders

Extends the System-2 Recommenders paper (Agarwal et al., FAccT 2024) by adding **engagement marks** to the dual-kernel Hawkes process. Proves that the marked model is identifiable **even when the two decay rates collide** — the regime in which the original paper is provably non-identifiable. Implementation is synthetic-data-only, numpy + scipy, branching-EM inference.

## TL;DR

The paper disentangles two user-behaviour components (System-1 impulsive, System-2 deliberate) from arrival-time timing alone. That requires $\beta^1 \ne \beta^2$. We add per-event engagement marks and prove identifiability holds as long as **either** $\beta^1 \ne \beta^2$ **or** $F_1 \ne F_2$. Flagship experiment: at $\beta^2 = \beta^1$ (paper's failure regime), the marked model is **5.2× more accurate** than the unmarked baseline.

## Repository layout

```
human ai project/
├── README.md                       (this file)
├── paper_summary.md                (Agarwal et al. 2024 summary)
├── project_proposal.md             (the 40% course project proposal)
├── mhawkes/                        (the model + inference)
│   ├── __init__.py
│   ├── marks.py                    (NegBin × LogNormal × Bernoulli mixed-type marks)
│   ├── simulate.py                 (Ogata thinning, with ground-truth z labels)
│   ├── em.py                       (branching-EM + multi-restart wrapper)
│   └── forecast.py                 (held-out predictive log-lik, calibration, baselines)
│
├── experiments/                    (all synthetic, fixed seeds)
│   ├── sanity_check.py             (parameter recovery gate)
│   ├── stress_beta_collapse.py    (demonstrates Corollary F1)
│   ├── exp1_consistency.py         (N → error, all parameter families)
│   ├── exp2_beta_gap.py            (flagship: marked vs unmarked over β²)
│   ├── exp3_sample_complexity.py   (moderate β-gap: marked dominates)
│   ├── exp4_mark_informativeness.py (mark-separation ablation at β-collapse)
│   └── exp5_forecasting.py         (held-out predictive log-lik, calibration, z-accuracy)
│
├── figures/                        (auto-generated)
│   ├── exp1_consistency.{png,npz}
│   ├── exp2_beta_gap.{png,npz}     (headline plot)
│   ├── exp3_sample_complexity.{png,npz}
│   ├── exp4_mark_informativeness.{png,npz}
│   └── exp5_forecasting.{png,npz}
│
├── writeup/                        (markdown)
│   ├── theorem.md                  (formal proofs; 150 lines)
│   └── report.md                   (8-section final report; 278 lines)
│
├── latex_report/                   (LaTeX version of the report)
│   ├── main.tex                    (9-page compiled PDF)
│   ├── main.pdf
│   ├── references.bib
│   └── README.md                   (compile instructions)
│
└── core idea/                      (focused explainer: model + notation + experiments)
    ├── main.tex                    (6-page compiled PDF)
    ├── main.pdf
    └── README.md
```

## Reproducing the experiments

Tested on **Python 3.14** with `numpy 2.4`, `scipy 1.17`, `matplotlib 3.10`. No PyTorch needed — the closed-form branching-EM runs in numpy.

From the repository root:

```bash
# Sanity check that parameters recover on a standard synthetic setup.
python -m experiments.sanity_check

# Stress test: does the model still work under β-collapse with distinct marks?
python -m experiments.stress_beta_collapse

# The four labelled experiments from the report.
python -m experiments.exp1_consistency
python -m experiments.exp2_beta_gap                 # flagship (~2 min)
python -m experiments.exp3_sample_complexity
python -m experiments.exp4_mark_informativeness     # uses multi-restart EM (~20 min)
python -m experiments.exp5_forecasting              # held-out forecasting (~1 min)
```

All scripts write `.png` (figure) and `.npz` (raw numerical results) into `figures/`.

Expected wall-clock on a laptop: about 30 minutes total, dominated by Exp 4's multi-restart sweep.

## Headline numbers

| Experiment | Claim | Result |
|---|---|---|
| Exp 1 | MLE consistency for the marked model | Every parameter's error decays to zero as $N \to \infty$; shrinkage factors 8.4–32.6× across 6 parameter families |
| Exp 2 (flagship) | Marks rescue identifiability at $\beta$-collapse | 5.2× improvement at $\beta^2 = \beta^1 = 1$ |
| Exp 3 | Marks dominate sample complexity at moderate $\beta$-gap | Unmarked plateaus at error ~0.32; marked < 0.1 by $N \approx 120$ |
| Exp 4 | Marks aren't magic (validates Corollary F1) | $z$-accuracy ramps from 54.7% (chance) at $s=0$ to 98.1% at $s=1$ |
| Exp 5 | Better recovery $\Rightarrow$ better held-out predictions | Marked beats unmarked-Hawkes-+-iid-marks by $\sim$0.6 nats / event in both regimes; held-out $z$-accuracy 98% (marked) vs 54% (unmarked, basically chance) at $\beta^1 = \beta^2$ |

## Key files

- **`mhawkes/em.py`** — branching-EM. `fit_em` is the single-init version (fast); `fit_em_restarts` runs several random inits and keeps the best log-likelihood (slower but more robust).
- **`writeup/theorem.md`** — the identifiability theorem with a three-step proof (timing → long-gap marks → short-burst marks).
- **`writeup/report.md`** / **`latex_report/main.tex`** — the 8-page final report in markdown and LaTeX form.

## References

Principal reference (the paper we extend):

- Agarwal, A., Usunier, N., Lazaric, A., Nickel, M. *System-2 Recommenders: Disentangling Utility and Engagement in Recommendation Systems via Temporal Point-Processes.* FAccT 2024. [arXiv:2406.01611](https://arxiv.org/abs/2406.01611)

Full reference list in `writeup/report.md` and `latex_report/references.bib`.
