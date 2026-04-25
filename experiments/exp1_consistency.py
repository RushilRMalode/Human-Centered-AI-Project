"""Experiment 1: sample size -> estimation error (paper Figure 1 in marked form).

Fix all parameters, sweep the horizon T (and therefore N = expected event
count) from small to large, and plot L2 parameter error vs N. The expected
behaviour, under Theorem 2 of the paper (consistency) and its marked-model
analogue, is monotone decay of every parameter's error to zero.

Outputs:
    figures/exp1_consistency.png
    figures/exp1_consistency.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mhawkes.em import fit_em  # noqa: E402
from mhawkes.marks import SystemMarkParams  # noqa: E402
from mhawkes.simulate import TrueParams, events_to_arrays, simulate  # noqa: E402


def ground_truth() -> TrueParams:
    return TrueParams(
        mu_1=0.05, mu_2=0.03,
        alpha_1=0.4, alpha_2=0.3,
        beta_1=1.0, beta_2=0.1,
        mark_1=SystemMarkParams(nb_mean=3.0,  nb_disp=5.0, ln_mu=0.0,  ln_sigma2=0.25, be_p=0.05),
        mark_2=SystemMarkParams(nb_mean=12.0, nb_disp=5.0, ln_mu=1.61, ln_sigma2=0.25, be_p=0.6),
    )


def unsigned_errors(truth: TrueParams, fit) -> dict:
    """Errors against label-swap-aligned truth. We reuse the alpha ordering
    chosen by beta-ordering (enforced in EM) -- no extra swap here because
    beta_1 != beta_2 by construction of this experiment."""
    e_mu = np.hypot(fit.mu_1 - truth.mu_1, fit.mu_2 - truth.mu_2)
    e_alpha = np.hypot(fit.alpha_1 - truth.alpha_1, fit.alpha_2 - truth.alpha_2)
    e_beta = np.hypot(fit.beta_1 - truth.beta_1, fit.beta_2 - truth.beta_2)
    e_lnmu = np.hypot(
        fit.mark_1.ln_mu - truth.mark_1.ln_mu,
        fit.mark_2.ln_mu - truth.mark_2.ln_mu,
    )
    e_bep = np.hypot(
        fit.mark_1.be_p - truth.mark_1.be_p,
        fit.mark_2.be_p - truth.mark_2.be_p,
    )
    e_nb = np.hypot(
        fit.mark_1.nb_mean - truth.mark_1.nb_mean,
        fit.mark_2.nb_mean - truth.mark_2.nb_mean,
    )
    return {
        "mu": float(e_mu), "alpha": float(e_alpha), "beta": float(e_beta),
        "ln_mu": float(e_lnmu), "be_p": float(e_bep), "nb_mean": float(e_nb),
    }


def run(T_values, n_seeds=4):
    truth = ground_truth()
    keys = ["mu", "alpha", "beta", "ln_mu", "be_p", "nb_mean"]
    data = {k: np.zeros((len(T_values), n_seeds)) for k in keys}
    Ns = np.zeros((len(T_values), n_seeds), dtype=int)

    for i, T in enumerate(T_values):
        for s in range(n_seeds):
            rng = np.random.default_rng(2000 * i + s)
            events = simulate(truth, T=T, rng=rng)
            t, L, D, C, _ = events_to_arrays(events)
            Ns[i, s] = len(t)

            t0 = time.time()
            fit, _ = fit_em(t, L, D, C, T=T, n_iter=150, tol=1e-7)
            elapsed = time.time() - t0

            errs = unsigned_errors(truth, fit)
            for k in keys:
                data[k][i, s] = errs[k]
            print(f"  T={T:6.0f} seed={s}  N={len(t):5d}  "
                  f"mu={errs['mu']:.3f} alpha={errs['alpha']:.3f} "
                  f"beta={errs['beta']:.3f} ln_mu={errs['ln_mu']:.3f}  "
                  f"({elapsed:.1f}s)")

    return data, Ns


def plot(data, Ns, out_path):
    N_mean = Ns.mean(axis=1)

    series = [
        ("mu",      r"$\mu$",           "#d62728"),
        ("alpha",   r"$\alpha$",        "#1f77b4"),
        ("beta",    r"$\beta$",         "#2ca02c"),
        ("ln_mu",   r"LogNormal $\mu$", "#9467bd"),
        ("be_p",    r"Bernoulli $p$",   "#ff7f0e"),
        ("nb_mean", r"NegBin mean",     "#8c564b"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label, color in series:
        mean = data[key].mean(axis=1)
        sem = data[key].std(axis=1) / np.sqrt(data[key].shape[1])
        ax.plot(N_mean, mean, "o-", color=color, label=label, linewidth=2)
        ax.fill_between(N_mean, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of events)")
    ax.set_ylabel("L2 parameter error")
    ax.set_title("Experiment 1: MLE consistency of the marked dual-kernel Hawkes")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9)
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


def main():
    T_values = [200.0, 400.0, 800.0, 1500.0, 3000.0, 6000.0]
    n_seeds = 4

    print(f"Experiment 1: sample-size sweep ({len(T_values)} T values x {n_seeds} seeds)")
    print()
    data, Ns = run(T_values, n_seeds=n_seeds)

    out_fig = PROJECT_ROOT / "figures" / "exp1_consistency.png"
    out_npz = PROJECT_ROOT / "figures" / "exp1_consistency.npz"
    plot(data, Ns, out_fig)

    np.savez(
        out_npz,
        T_values=np.array(T_values),
        n_events=Ns,
        **data,
    )
    print(f"Saved {out_npz}")

    # Headline: did every parameter's error drop monotonically?
    print("\nError at smallest vs largest T:")
    for k in ["mu", "alpha", "beta", "ln_mu", "be_p", "nb_mean"]:
        lo, hi = data[k][0].mean(), data[k][-1].mean()
        print(f"  {k:8s}  small-T={lo:.4f}  large-T={hi:.4f}  "
              f"shrink={lo / max(hi, 1e-6):.1f}x")


if __name__ == "__main__":
    main()
