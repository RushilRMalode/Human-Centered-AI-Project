"""Experiment 3: sample-complexity gain at moderate beta gap.

Fix beta_2 = 0.5 (so beta_1/beta_2 = 2.0, a modest gap where the unmarked
model is technically identifiable but information-inefficient). Sweep the
sample size N by varying horizon T, and plot L2 alpha error for marked vs
unmarked. The claim: marks strictly lower the sample size needed to reach
a fixed error threshold.

Outputs:
    figures/exp3_sample_complexity.png
    figures/exp3_sample_complexity.npz
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
        beta_1=1.0, beta_2=0.5,   # moderate gap (ratio 2.0)
        mark_1=SystemMarkParams(nb_mean=3.0,  nb_disp=5.0, ln_mu=0.0,  ln_sigma2=0.25, be_p=0.05),
        mark_2=SystemMarkParams(nb_mean=12.0, nb_disp=5.0, ln_mu=1.61, ln_sigma2=0.25, be_p=0.6),
    )


def alpha_error(truth: TrueParams, fit) -> float:
    # In this experiment beta_1 > beta_2 holds by EM constraint and truth,
    # so no extra label-swap check needed.
    return float(np.hypot(fit.alpha_1 - truth.alpha_1, fit.alpha_2 - truth.alpha_2))


def run(T_values, n_seeds=4):
    truth = ground_truth()
    umk = np.zeros((len(T_values), n_seeds))
    mk  = np.zeros((len(T_values), n_seeds))
    Ns  = np.zeros((len(T_values), n_seeds), dtype=int)

    for i, T in enumerate(T_values):
        for s in range(n_seeds):
            rng = np.random.default_rng(3000 * i + s)
            events = simulate(truth, T=T, rng=rng)
            t, L, D, C, _ = events_to_arrays(events)
            Ns[i, s] = len(t)

            t0 = time.time()
            fit_u, _ = fit_em(t, L, D, C, T=T, n_iter=150, tol=1e-7, use_marks=False)
            fit_m, _ = fit_em(t, L, D, C, T=T, n_iter=150, tol=1e-7, use_marks=True)
            el = time.time() - t0

            umk[i, s] = alpha_error(truth, fit_u)
            mk[i, s]  = alpha_error(truth, fit_m)
            print(f"  T={T:6.0f} seed={s}  N={len(t):5d}  "
                  f"alpha_err unmarked={umk[i,s]:.3f} marked={mk[i,s]:.3f}  ({el:.1f}s)")

    return umk, mk, Ns


def plot(umk, mk, Ns, out_path):
    N_mean = Ns.mean(axis=1)

    mean_u = umk.mean(axis=1); sem_u = umk.std(axis=1) / np.sqrt(umk.shape[1])
    mean_m = mk.mean(axis=1);  sem_m = mk.std(axis=1)  / np.sqrt(mk.shape[1])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(N_mean, mean_u, "o-", color="#c43b3b", label="unmarked (paper)", linewidth=2)
    ax.fill_between(N_mean, mean_u - sem_u, mean_u + sem_u, color="#c43b3b", alpha=0.2)
    ax.plot(N_mean, mean_m, "s-", color="#2e6cc1", label="marked (ours)", linewidth=2)
    ax.fill_between(N_mean, mean_m - sem_m, mean_m + sem_m, color="#2e6cc1", alpha=0.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of events)")
    ax.set_ylabel(r"L2 error on $(\alpha^1, \alpha^2)$")
    ax.set_title(r"Experiment 3: sample complexity at $\beta^2 = 0.5$ (moderate gap)")
    ax.legend(frameon=False)
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


def main():
    T_values = [200.0, 400.0, 800.0, 1500.0, 3000.0]
    n_seeds = 5

    print(f"Experiment 3: sample complexity sweep ({len(T_values)} T x {n_seeds} seeds x 2 models)")
    print()
    umk, mk, Ns = run(T_values, n_seeds=n_seeds)

    out_fig = PROJECT_ROOT / "figures" / "exp3_sample_complexity.png"
    out_npz = PROJECT_ROOT / "figures" / "exp3_sample_complexity.npz"
    plot(umk, mk, Ns, out_fig)
    np.savez(
        out_npz,
        T_values=np.array(T_values), n_events=Ns,
        unmarked_alpha=umk, marked_alpha=mk,
    )
    print(f"Saved {out_npz}")

    # Find the smallest N at which marked drops below error 0.1; same for unmarked.
    thresh = 0.1
    N_mean = Ns.mean(axis=1)
    def first_below(errs, thresh):
        mean = errs.mean(axis=1)
        idx = np.argmax(mean < thresh) if (mean < thresh).any() else -1
        return int(N_mean[idx]) if idx >= 0 else None
    N_u = first_below(umk, thresh)
    N_m = first_below(mk, thresh)
    print(f"\nSamples to hit alpha-err < {thresh}: unmarked ~{N_u}, marked ~{N_m}")
    if N_u and N_m:
        print(f"Sample-complexity ratio: {N_u / N_m:.1f}x")


if __name__ == "__main__":
    main()
