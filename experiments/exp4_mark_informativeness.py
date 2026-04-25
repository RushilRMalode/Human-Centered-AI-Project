"""Experiment 4: mark-informativeness ablation at beta collapse.

Fix beta_1 = beta_2 = 1.0 (the paper's provably non-identifiable regime).
Sweep the separation between the two systems' mark distributions along a
single scalar "informativeness" dial s in [0, 1]:

    F_1 = (short dwell, low like prob, low count)     scaled by s
    F_2 = (long dwell,  high like prob, high count)   scaled by s

When s = 0 the two mark distributions collapse to identical, and the
marked model must degrade to match the unmarked model (both fail under
beta collapse). When s = 1 the marks are fully informative and the marked
model should recover parameters cleanly.

This experiment directly tests the Failure-Modes Corollary: the marked
model is not magic -- it needs at least one channel (timing OR marks) to
carry disentangling signal.

Outputs:
    figures/exp4_mark_informativeness.png
    figures/exp4_mark_informativeness.npz
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

from mhawkes.em import fit_em, fit_em_restarts, predict_z_posterior  # noqa: E402
from mhawkes.marks import SystemMarkParams  # noqa: E402
from mhawkes.simulate import TrueParams, events_to_arrays, simulate  # noqa: E402


# Anchor mark distributions. At informativeness s = 1 each system uses its
# own end-point; at s = 0 both systems use the midpoint. Linear interpolation
# in the mark-parameter space.
ANCHOR_1 = dict(nb_mean=3.0,  ln_mu=0.0,   be_p=0.05)
ANCHOR_2 = dict(nb_mean=12.0, ln_mu=1.61,  be_p=0.60)
NB_DISP = 5.0
LN_SIGMA2 = 0.25


def mid(a: dict, b: dict) -> dict:
    return {k: 0.5 * (a[k] + b[k]) for k in a}


def interpolate(anchor: dict, midpoint: dict, s: float) -> SystemMarkParams:
    """s=0 -> midpoint (identical across systems), s=1 -> anchor."""
    return SystemMarkParams(
        nb_mean=midpoint["nb_mean"] + s * (anchor["nb_mean"] - midpoint["nb_mean"]),
        nb_disp=NB_DISP,
        ln_mu=midpoint["ln_mu"] + s * (anchor["ln_mu"] - midpoint["ln_mu"]),
        ln_sigma2=LN_SIGMA2,
        be_p=midpoint["be_p"] + s * (anchor["be_p"] - midpoint["be_p"]),
    )


def ground_truth(informativeness: float) -> TrueParams:
    mp = mid(ANCHOR_1, ANCHOR_2)
    return TrueParams(
        mu_1=0.05, mu_2=0.03,
        alpha_1=0.4, alpha_2=0.3,
        beta_1=1.0, beta_2=1.0,      # COLLAPSE: timing provides no signal
        mark_1=interpolate(ANCHOR_1, mp, informativeness),
        mark_2=interpolate(ANCHOR_2, mp, informativeness),
    )


def alpha_error_swap_aware(truth: TrueParams, fit) -> float:
    """Pick the alpha assignment (with or without system swap) that
    minimises L2 error -- since beta_1 == beta_2 the natural ordering is
    ambiguous and we must compare against the better of the two."""
    d_same = np.hypot(fit.alpha_1 - truth.alpha_1, fit.alpha_2 - truth.alpha_2)
    d_swap = np.hypot(fit.alpha_2 - truth.alpha_1, fit.alpha_1 - truth.alpha_2)
    return float(min(d_same, d_swap))


def z_accuracy_swap_aware(z_true, z_prob_1):
    z_pred = np.where(z_prob_1 > 0.5, 1, 2)
    acc1 = (z_pred == z_true).mean()
    acc2 = (z_pred == (3 - z_true)).mean()
    return float(max(acc1, acc2))


def run(informativeness_values, n_seeds=4, T=1500.0):
    unmk_a = np.zeros((len(informativeness_values), n_seeds))
    mk_a   = np.zeros((len(informativeness_values), n_seeds))
    mk_z   = np.zeros((len(informativeness_values), n_seeds))
    Ns     = np.zeros((len(informativeness_values), n_seeds), dtype=int)

    for i, s in enumerate(informativeness_values):
        truth = ground_truth(s)
        for seed in range(n_seeds):
            rng = np.random.default_rng(4000 * i + seed)
            events = simulate(truth, T=T, rng=rng)
            t, L, D, C, z_true = events_to_arrays(events)
            Ns[i, seed] = len(t)

            t0 = time.time()
            # Multi-restart (3 inits, keep best log-lik) smooths out the
            # seed-level variance observed in the single-init version of
            # this experiment.
            restart_rng = np.random.default_rng(40000 + 17 * i + seed)
            fit_u, _ = fit_em_restarts(
                t, L, D, C, T=T, n_restarts=3, n_iter=150, tol=1e-7,
                use_marks=False, rng=restart_rng,
            )
            fit_m, _ = fit_em_restarts(
                t, L, D, C, T=T, n_restarts=3, n_iter=150, tol=1e-7,
                use_marks=True, rng=restart_rng,
            )
            el = time.time() - t0

            unmk_a[i, seed] = alpha_error_swap_aware(truth, fit_u)
            mk_a[i, seed]   = alpha_error_swap_aware(truth, fit_m)
            z_prob_1 = predict_z_posterior(t, L, D, C, fit_m)
            mk_z[i, seed]  = z_accuracy_swap_aware(z_true, z_prob_1)

            print(f"  s={s:.2f} seed={seed}  N={len(t):4d}  "
                  f"alpha unmarked={unmk_a[i,seed]:.3f} marked={mk_a[i,seed]:.3f}  "
                  f"z-acc={mk_z[i,seed]:.3f}  ({el:.1f}s)")

    return unmk_a, mk_a, mk_z, Ns


def plot(s_values, unmk_a, mk_a, mk_z, out_path):
    s_arr = np.array(s_values)
    mu_u = unmk_a.mean(axis=1); su_u = unmk_a.std(axis=1) / np.sqrt(unmk_a.shape[1])
    mu_m = mk_a.mean(axis=1);   su_m = mk_a.std(axis=1)   / np.sqrt(mk_a.shape[1])
    mu_z = mk_z.mean(axis=1);   su_z = mk_z.std(axis=1)   / np.sqrt(mk_z.shape[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # left: alpha error
    ax1.plot(s_arr, mu_u, "o-", color="#c43b3b", label="unmarked", linewidth=2)
    ax1.fill_between(s_arr, mu_u - su_u, mu_u + su_u, color="#c43b3b", alpha=0.2)
    ax1.plot(s_arr, mu_m, "s-", color="#2e6cc1", label="marked",   linewidth=2)
    ax1.fill_between(s_arr, mu_m - su_m, mu_m + su_m, color="#2e6cc1", alpha=0.2)
    ax1.set_xlabel("mark informativeness s")
    ax1.set_ylabel(r"L2 error on $(\alpha^1, \alpha^2)$")
    ax1.set_title(r"Exp 4: mark-informativeness ablation at $\beta^1 = \beta^2$")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    # right: z-accuracy
    ax2.plot(s_arr, mu_z, "s-", color="#2e6cc1", linewidth=2, label="marked")
    ax2.fill_between(s_arr, mu_z - su_z, mu_z + su_z, color="#2e6cc1", alpha=0.2)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.8, label="chance")
    ax2.set_xlabel("mark informativeness s")
    ax2.set_ylabel("z-label classification accuracy")
    ax2.set_title("z-label accuracy (marked model)")
    ax2.set_ylim(0.4, 1.02)
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


def main():
    informativeness = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    n_seeds = 4
    T = 1500.0

    print(f"Experiment 4: mark-informativeness ablation ({len(informativeness)} s x {n_seeds} seeds)")
    print()
    unmk_a, mk_a, mk_z, Ns = run(informativeness, n_seeds=n_seeds, T=T)

    out_fig = PROJECT_ROOT / "figures" / "exp4_mark_informativeness.png"
    out_npz = PROJECT_ROOT / "figures" / "exp4_mark_informativeness.npz"
    plot(informativeness, unmk_a, mk_a, mk_z, out_fig)
    np.savez(
        out_npz,
        informativeness=np.array(informativeness), n_events=Ns,
        unmarked_alpha=unmk_a, marked_alpha=mk_a, marked_z_accuracy=mk_z,
    )
    print(f"Saved {out_npz}")

    print("\nHeadline:")
    print(f"  s=0.0 (uninformative marks): unmarked={unmk_a[0].mean():.3f}, marked={mk_a[0].mean():.3f}, "
          f"z-acc={mk_z[0].mean():.3f}")
    print(f"  s=1.0 (informative marks):   unmarked={unmk_a[-1].mean():.3f}, marked={mk_a[-1].mean():.3f}, "
          f"z-acc={mk_z[-1].mean():.3f}")


if __name__ == "__main__":
    main()
