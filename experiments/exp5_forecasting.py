"""Experiment 5: held-out forecasting on a test segment.

Question: parameter recovery is great, but does it translate into better
*predictions* on data the model hasn't seen? This is the canonical
generative-model test: train on the first slice of an event sequence,
evaluate the predictive log-likelihood and calibration on a held-out slice.

We compare three models:

    - Marked Hawkes (ours)
    - Unmarked Hawkes (paper baseline)
    - Homogeneous Poisson + iid marks (sanity-check baseline)

across two regimes:

    - Easy: beta_1 = 1.0, beta_2 = 0.1   (default ground truth, both channels)
    - Hard: beta_1 = beta_2 = 1.0        (timing channel collapsed)

Three metrics on the test segment:

    - Predictive log-likelihood per event (higher = better)
    - Time-rescaling KS distance to Uniform[0, 1] (lower = better calibrated)
    - z-classification accuracy on held-out events (higher = better)

Outputs:
    figures/exp5_forecasting.png
    figures/exp5_forecasting.npz
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

from mhawkes.em import fit_em_restarts, predict_z_posterior  # noqa: E402
from mhawkes.forecast import (  # noqa: E402
    fit_poisson_iid,
    ks_distance_to_uniform,
    pit_values_hawkes,
    pit_values_poisson,
    predictive_log_likelihood_hawkes,
    predictive_log_likelihood_poisson,
)
from mhawkes.marks import SystemMarkParams, joint_logpdf  # noqa: E402
from mhawkes.simulate import TrueParams, events_to_arrays, simulate  # noqa: E402


def truth(beta_1: float, beta_2: float) -> TrueParams:
    return TrueParams(
        mu_1=0.05, mu_2=0.03,
        alpha_1=0.4, alpha_2=0.3,
        beta_1=beta_1, beta_2=beta_2,
        mark_1=SystemMarkParams(nb_mean=3.0,  nb_disp=5.0,
                                ln_mu=0.0,  ln_sigma2=0.25, be_p=0.05),
        mark_2=SystemMarkParams(nb_mean=12.0, nb_disp=5.0,
                                ln_mu=1.61, ln_sigma2=0.25, be_p=0.60),
    )


REGIMES = {
    "easy": dict(beta_1=1.0, beta_2=0.1, label=r"$\beta^1=1,\;\beta^2=0.1$ (both channels)"),
    "hard": dict(beta_1=1.0, beta_2=1.0, label=r"$\beta^1=\beta^2=1$ (timing collapsed)"),
}


def z_accuracy_swap_aware(z_true: np.ndarray, z_prob_1: np.ndarray) -> float:
    if len(z_true) == 0:
        return float("nan")
    z_pred = np.where(z_prob_1 > 0.5, 1, 2)
    a1 = float((z_pred == z_true).mean())
    a2 = float((z_pred == (3 - z_true)).mean())
    return max(a1, a2)


def run_one(regime: str, seed: int, T_total: float, train_frac: float):
    cfg = REGIMES[regime]
    rng = np.random.default_rng(50_000 + 13 * seed + (0 if regime == "easy" else 7))
    events = simulate(truth(cfg["beta_1"], cfg["beta_2"]), T=T_total, rng=rng)
    t, L, D, C, z_true = events_to_arrays(events)
    if len(t) < 50:
        return None  # too few events to learn anything

    T_train = train_frac * T_total
    train_mask = t <= T_train
    test_mask = ~train_mask
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    if n_train < 30 or n_test < 10:
        return None

    t_tr = t[train_mask]; L_tr = L[train_mask]; D_tr = D[train_mask]; C_tr = C[train_mask]

    # --- fit three models on train ---
    restart_rng = np.random.default_rng(70_000 + 91 * seed)
    fit_m, _ = fit_em_restarts(t_tr, L_tr, D_tr, C_tr, T=T_train,
                               n_restarts=3, n_iter=120, tol=1e-7,
                               use_marks=True, rng=restart_rng)
    fit_u, _ = fit_em_restarts(t_tr, L_tr, D_tr, C_tr, T=T_train,
                               n_restarts=3, n_iter=120, tol=1e-7,
                               use_marks=False, rng=restart_rng)
    fit_p = fit_poisson_iid(t_tr, L_tr, D_tr, C_tr, T_train)

    # --- predictive log-likelihood per event ---
    # Apples-to-apples: every model predicts BOTH timing and marks.
    #   Marked Hawkes: per-system F_j combined with the dual-kernel Hawkes timing.
    #   Unmarked Hawkes + iid marks: timing from unmarked Hawkes; marks from
    #     the same pooled iid mark distribution as the Poisson baseline.
    #   Poisson + iid: constant-rate timing + same pooled mark distribution.
    L_te, D_te, C_te = L[test_mask], D[test_mask], C[test_mask]
    iid_mark_logL_test = float(joint_logpdf(L_te, D_te, C_te, fit_p.mark).sum())

    ll_m, _ = predictive_log_likelihood_hawkes(t, L, D, C, T_train, T_total, fit_m, use_marks=True)
    ll_u_timing, _ = predictive_log_likelihood_hawkes(t, L, D, C, T_train, T_total, fit_u, use_marks=False)
    ll_u = ll_u_timing + iid_mark_logL_test
    ll_p, _ = predictive_log_likelihood_poisson(t, L, D, C, T_train, T_total, fit_p)
    pe_m = ll_m / max(n_test, 1)
    pe_u = ll_u / max(n_test, 1)
    pe_p = ll_p / max(n_test, 1)

    # --- calibration (KS to uniform) ---
    pit_m = pit_values_hawkes(t, T_train, fit_m)
    pit_u = pit_values_hawkes(t, T_train, fit_u)
    pit_p = pit_values_poisson(t, T_train, fit_p)
    ks_m = ks_distance_to_uniform(pit_m)
    ks_u = ks_distance_to_uniform(pit_u)
    ks_p = ks_distance_to_uniform(pit_p)

    # --- z-accuracy on test events (uses full-sequence posterior) ---
    # Marked uses its learned F_j; unmarked uses timing-only posterior since
    # its mark params were never updated during training (use_marks=False).
    z_full_m = predict_z_posterior(t, L, D, C, fit_m, use_marks=True)
    z_full_u = predict_z_posterior(t, L, D, C, fit_u, use_marks=False)
    zacc_m = z_accuracy_swap_aware(z_true[test_mask], z_full_m[test_mask])
    zacc_u = z_accuracy_swap_aware(z_true[test_mask], z_full_u[test_mask])

    return dict(
        n_train=n_train, n_test=n_test,
        ll_per_event=dict(marked=pe_m, unmarked=pe_u, poisson=pe_p),
        ks=dict(marked=ks_m, unmarked=ks_u, poisson=ks_p),
        zacc=dict(marked=zacc_m, unmarked=zacc_u),
    )


def aggregate(rows):
    """rows is a list of result dicts; return per-model arrays."""
    out = {
        "ll_marked": [], "ll_unmarked": [], "ll_poisson": [],
        "ks_marked": [], "ks_unmarked": [], "ks_poisson": [],
        "zacc_marked": [], "zacc_unmarked": [],
        "n_test": [],
    }
    for r in rows:
        out["ll_marked"].append(r["ll_per_event"]["marked"])
        out["ll_unmarked"].append(r["ll_per_event"]["unmarked"])
        out["ll_poisson"].append(r["ll_per_event"]["poisson"])
        out["ks_marked"].append(r["ks"]["marked"])
        out["ks_unmarked"].append(r["ks"]["unmarked"])
        out["ks_poisson"].append(r["ks"]["poisson"])
        out["zacc_marked"].append(r["zacc"]["marked"])
        out["zacc_unmarked"].append(r["zacc"]["unmarked"])
        out["n_test"].append(r["n_test"])
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def mean_se(arr: np.ndarray) -> tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0) / np.sqrt(len(arr)))


def plot(results, out_path):
    regimes = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    width = 0.27
    x_base = np.arange(len(regimes))

    # ---- Panel 1: predictive log-likelihood per event ----
    ax = axes[0]
    all_ll_means = []
    for j, (model, color) in enumerate([("marked", "#2e6cc1"),
                                        ("unmarked", "#c43b3b"),
                                        ("poisson", "#888888")]):
        ms = [mean_se(results[r][f"ll_{model}"]) for r in regimes]
        ms_arr = np.array(ms)  # (n_regimes, 2)
        all_ll_means.extend(ms_arr[:, 0].tolist())
        offset = (j - 1) * width
        ax.bar(x_base + offset, ms_arr[:, 0], width=width, color=color,
               yerr=ms_arr[:, 1], capsize=4, label=model)
    ax.set_xticks(x_base)
    ax.set_xticklabels([REGIMES[r]["label"] for r in regimes], fontsize=8)
    ax.set_ylabel("predictive log-lik per event")
    ax.set_title("Held-out predictive log-likelihood\n(higher is better)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, axis="y")
    # Zoom y-axis to where the bars actually live
    lo, hi = min(all_ll_means), max(all_ll_means)
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    # ---- Panel 2: KS distance to uniform ----
    ax = axes[1]
    for j, (model, color) in enumerate([("marked", "#2e6cc1"),
                                        ("unmarked", "#c43b3b"),
                                        ("poisson", "#888888")]):
        ms = [mean_se(results[r][f"ks_{model}"]) for r in regimes]
        ms_arr = np.array(ms)
        offset = (j - 1) * width
        ax.bar(x_base + offset, ms_arr[:, 0], width=width, color=color,
               yerr=ms_arr[:, 1], capsize=4, label=model)
    ax.set_xticks(x_base)
    ax.set_xticklabels([REGIMES[r]["label"] for r in regimes], fontsize=8)
    ax.set_ylabel("KS distance to U[0,1]")
    ax.set_title("Time-rescaling calibration\n(lower is better)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # ---- Panel 3: held-out z-accuracy ----
    ax = axes[2]
    width2 = 0.36
    for j, (model, color) in enumerate([("marked", "#2e6cc1"),
                                        ("unmarked", "#c43b3b")]):
        ms = [mean_se(results[r][f"zacc_{model}"]) for r in regimes]
        ms_arr = np.array(ms)
        offset = (j - 0.5) * width2
        ax.bar(x_base + offset, ms_arr[:, 0], width=width2, color=color,
               yerr=ms_arr[:, 1], capsize=4, label=model)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="chance")
    ax.set_xticks(x_base)
    ax.set_xticklabels([REGIMES[r]["label"] for r in regimes], fontsize=8)
    ax.set_ylabel("held-out $z$-accuracy")
    ax.set_title("Held-out $z$-classification\n(higher is better)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


def main():
    n_seeds = 4
    T_total = 2000.0
    train_frac = 0.7

    print(f"Experiment 5: held-out forecasting "
          f"({len(REGIMES)} regimes x {n_seeds} seeds, T={T_total}, train_frac={train_frac})")
    print()

    results = {}
    for regime in REGIMES:
        print(f"--- regime: {regime} ({REGIMES[regime]['label']}) ---")
        rows = []
        for seed in range(n_seeds):
            t0 = time.time()
            r = run_one(regime, seed, T_total, train_frac)
            el = time.time() - t0
            if r is None:
                print(f"  seed {seed}: skipped (too few events)")
                continue
            rows.append(r)
            print(f"  seed {seed}  N_tr={r['n_train']:4d} N_te={r['n_test']:4d}  "
                  f"logL/event m={r['ll_per_event']['marked']:.3f} "
                  f"u={r['ll_per_event']['unmarked']:.3f} "
                  f"p={r['ll_per_event']['poisson']:.3f}  "
                  f"KS m={r['ks']['marked']:.3f} u={r['ks']['unmarked']:.3f} "
                  f"p={r['ks']['poisson']:.3f}  "
                  f"z m={r['zacc']['marked']:.3f} u={r['zacc']['unmarked']:.3f} "
                  f"({el:.1f}s)")
        if not rows:
            print(f"  (no usable rows for regime {regime})")
            continue
        results[regime] = aggregate(rows)
        agg = results[regime]
        print(f"  --> means: logL m={agg['ll_marked'].mean():+.3f} "
              f"u={agg['ll_unmarked'].mean():+.3f} "
              f"p={agg['ll_poisson'].mean():+.3f}  "
              f"KS m={agg['ks_marked'].mean():.3f} "
              f"u={agg['ks_unmarked'].mean():.3f}  "
              f"z m={agg['zacc_marked'].mean():.3f} "
              f"u={agg['zacc_unmarked'].mean():.3f}")
        print()

    out_fig = PROJECT_ROOT / "figures" / "exp5_forecasting.png"
    out_npz = PROJECT_ROOT / "figures" / "exp5_forecasting.npz"
    plot(results, out_fig)
    save_kwargs = {}
    for r, agg in results.items():
        for k, v in agg.items():
            save_kwargs[f"{r}__{k}"] = v
    np.savez(out_npz, **save_kwargs)
    print(f"Saved {out_npz}")

    print("\nHeadline (means):")
    for regime in results:
        agg = results[regime]
        print(f"  {regime:5s}: marked logL/event={agg['ll_marked'].mean():+.3f}, "
              f"unmarked={agg['ll_unmarked'].mean():+.3f} "
              f"(diff={agg['ll_marked'].mean()-agg['ll_unmarked'].mean():+.3f}); "
              f"marked z-acc={agg['zacc_marked'].mean():.3f}, "
              f"unmarked={agg['zacc_unmarked'].mean():.3f}")


if __name__ == "__main__":
    main()
