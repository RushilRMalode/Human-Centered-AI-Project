"""Experiment 2 (flagship): beta-gap sweep for marked vs unmarked EM.

Fix beta_1 = 1.0 and sweep beta_2 in (0, 1]. For each setting:
  - simulate a marked dual-kernel Hawkes with known ground truth
  - fit the unmarked baseline (timing only)
  - fit the full marked model
  - record L2 error on alpha and L2 error on beta

The expectation, and the project's flagship claim: the unmarked model's
error on alpha spikes as beta_2 -> beta_1 (paper's failure regime), while
the marked model's error stays flat because marks carry the disentangling
signal when timing alone cannot.

Outputs:
  figures/exp2_beta_gap.png     -- the headline figure
  figures/exp2_beta_gap.npz     -- raw numerical results
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


def ground_truth(beta_2: float) -> TrueParams:
    return TrueParams(
        mu_1=0.05, mu_2=0.03,
        alpha_1=0.4, alpha_2=0.3,
        beta_1=1.0, beta_2=float(beta_2),
        mark_1=SystemMarkParams(nb_mean=3.0,  nb_disp=5.0, ln_mu=0.0,  ln_sigma2=0.25, be_p=0.05),
        mark_2=SystemMarkParams(nb_mean=12.0, nb_disp=5.0, ln_mu=1.61, ln_sigma2=0.25, be_p=0.6),
    )


def param_errors(truth: TrueParams, fit) -> dict:
    # Label-swap aware: compare against the assignment that minimises alpha error.
    fit_a = (fit.alpha_1, fit.alpha_2)
    fit_b = (fit.beta_1, fit.beta_2)
    true_a = (truth.alpha_1, truth.alpha_2)
    true_b = (truth.beta_1, truth.beta_2)

    d_same = np.hypot(fit_a[0] - true_a[0], fit_a[1] - true_a[1])
    d_swap = np.hypot(fit_a[1] - true_a[0], fit_a[0] - true_a[1])
    if d_swap < d_same:
        fit_a = (fit.alpha_2, fit.alpha_1)
        fit_b = (fit.beta_2, fit.beta_1)

    return {
        "alpha": float(np.hypot(fit_a[0] - true_a[0], fit_a[1] - true_a[1])),
        "beta": float(np.hypot(fit_b[0] - true_b[0], fit_b[1] - true_b[1])),
    }


def run(beta_2_values, n_seeds=5, T=1500.0):
    results = {
        "beta_2": list(beta_2_values),
        "unmarked_alpha": np.zeros((len(beta_2_values), n_seeds)),
        "unmarked_beta":  np.zeros((len(beta_2_values), n_seeds)),
        "marked_alpha":   np.zeros((len(beta_2_values), n_seeds)),
        "marked_beta":    np.zeros((len(beta_2_values), n_seeds)),
        "n_events":       np.zeros((len(beta_2_values), n_seeds), dtype=int),
    }

    for i, b2 in enumerate(beta_2_values):
        truth = ground_truth(b2)
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 * i + s)
            events = simulate(truth, T=T, rng=rng)
            t, L, D, C, _ = events_to_arrays(events)
            results["n_events"][i, s] = len(t)

            t0 = time.time()
            unmarked, _ = fit_em(t, L, D, C, T=T, n_iter=100, tol=1e-6, use_marks=False)
            marked, _   = fit_em(t, L, D, C, T=T, n_iter=100, tol=1e-6, use_marks=True)
            el = time.time() - t0

            eu = param_errors(truth, unmarked)
            em = param_errors(truth, marked)
            results["unmarked_alpha"][i, s] = eu["alpha"]
            results["unmarked_beta"][i, s]  = eu["beta"]
            results["marked_alpha"][i, s]   = em["alpha"]
            results["marked_beta"][i, s]    = em["beta"]

            print(f"  beta_2={b2:.2f} seed={s}  N={len(t):4d}  "
                  f"alpha_err unmarked={eu['alpha']:.3f} marked={em['alpha']:.3f}  "
                  f"({el:.1f}s)")

    return results


def plot(results, out_path):
    b2 = np.array(results["beta_2"])

    ua = results["unmarked_alpha"]
    ma = results["marked_alpha"]

    mean_u = ua.mean(axis=1)
    mean_m = ma.mean(axis=1)
    sem_u  = ua.std(axis=1) / np.sqrt(ua.shape[1])
    sem_m  = ma.std(axis=1) / np.sqrt(ma.shape[1])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(b2, mean_u, "o-",  color="#c43b3b", label="unmarked (paper)", linewidth=2)
    ax.fill_between(b2, mean_u - sem_u, mean_u + sem_u, color="#c43b3b", alpha=0.2)
    ax.plot(b2, mean_m, "s-",  color="#2e6cc1", label="marked (ours)",    linewidth=2)
    ax.fill_between(b2, mean_m - sem_m, mean_m + sem_m, color="#2e6cc1", alpha=0.2)

    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel(r"$\beta^2$  (fast decay $\beta^1 = 1$ fixed)")
    ax.set_ylabel(r"L2 error on $(\alpha^1, \alpha^2)$")
    ax.set_title(r"Flagship: marks rescue identifiability as $\beta^2 \to \beta^1$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"\nSaved {out_path}")


def main():
    beta_2_values = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    n_seeds = 4
    T = 1500.0

    print(f"Experiment 2: beta-gap sweep  ({len(beta_2_values)} betas x {n_seeds} seeds x 2 models)")
    print()
    results = run(beta_2_values, n_seeds=n_seeds, T=T)

    out_fig = PROJECT_ROOT / "figures" / "exp2_beta_gap.png"
    out_npz = PROJECT_ROOT / "figures" / "exp2_beta_gap.npz"
    plot(results, out_fig)
    np.savez(
        out_npz,
        beta_2=np.array(results["beta_2"]),
        unmarked_alpha=results["unmarked_alpha"],
        marked_alpha=results["marked_alpha"],
        unmarked_beta=results["unmarked_beta"],
        marked_beta=results["marked_beta"],
        n_events=results["n_events"],
    )
    print(f"Saved {out_npz}")

    # One-line headline stat.
    idx_near = np.argmin(np.abs(np.array(results["beta_2"]) - 1.0))
    u_near = results["unmarked_alpha"][idx_near].mean()
    m_near = results["marked_alpha"][idx_near].mean()
    print()
    print(f"Headline (beta_2 = {results['beta_2'][idx_near]}): "
          f"unmarked alpha-err = {u_near:.3f}, marked alpha-err = {m_near:.3f}  "
          f"(marked improves {u_near/max(m_near,1e-6):.1f}x)")


if __name__ == "__main__":
    main()
