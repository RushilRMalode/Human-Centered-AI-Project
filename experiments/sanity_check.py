"""Week-1 sanity check: simulate a marked dual-kernel Hawkes from known
ground truth, fit with branching-EM, and verify parameter recovery plus
z-label classification accuracy.

Usage:  python -m experiments.sanity_check
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (python experiments/sanity_check.py) by
# adding the project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mhawkes.em import fit_em, predict_z_posterior  # noqa: E402
from mhawkes.marks import SystemMarkParams  # noqa: E402
from mhawkes.simulate import TrueParams, events_to_arrays, simulate  # noqa: E402


def make_ground_truth() -> TrueParams:
    # System 1 = impulsive (fast decay, short dwell, low like rate)
    mark_1 = SystemMarkParams(
        nb_mean=3.0,   # short bursts
        nb_disp=5.0,
        ln_mu=np.log(1.0),   # median dwell ~ 1 time unit
        ln_sigma2=0.25,
        be_p=0.05,     # rarely likes
    )
    # System 2 = deliberate (slow decay, long dwell, high like rate)
    mark_2 = SystemMarkParams(
        nb_mean=12.0,
        nb_disp=5.0,
        ln_mu=np.log(5.0),   # median dwell ~ 5
        ln_sigma2=0.25,
        be_p=0.6,
    )
    return TrueParams(
        mu_1=0.05,
        mu_2=0.03,
        alpha_1=0.4,
        beta_1=1.0,
        alpha_2=0.3,
        beta_2=0.1,
        mark_1=mark_1,
        mark_2=mark_2,
    )


def report_recovery(true: TrueParams, fit, prefix=""):
    print(f"{prefix}mu_1     true={true.mu_1:.4f}   fit={fit.mu_1:.4f}")
    print(f"{prefix}mu_2     true={true.mu_2:.4f}   fit={fit.mu_2:.4f}")
    print(f"{prefix}alpha_1  true={true.alpha_1:.4f}  fit={fit.alpha_1:.4f}")
    print(f"{prefix}alpha_2  true={true.alpha_2:.4f}  fit={fit.alpha_2:.4f}")
    print(f"{prefix}beta_1   true={true.beta_1:.4f}  fit={fit.beta_1:.4f}")
    print(f"{prefix}beta_2   true={true.beta_2:.4f}  fit={fit.beta_2:.4f}")
    print(f"{prefix}S1 dwell mu   true={true.mark_1.ln_mu:.3f}  fit={fit.mark_1.ln_mu:.3f}")
    print(f"{prefix}S2 dwell mu   true={true.mark_2.ln_mu:.3f}  fit={fit.mark_2.ln_mu:.3f}")
    print(f"{prefix}S1 like p     true={true.mark_1.be_p:.3f}   fit={fit.mark_1.be_p:.3f}")
    print(f"{prefix}S2 like p     true={true.mark_2.be_p:.3f}   fit={fit.mark_2.be_p:.3f}")
    print(f"{prefix}S1 NB mean    true={true.mark_1.nb_mean:.3f}  fit={fit.mark_1.nb_mean:.3f}")
    print(f"{prefix}S2 NB mean    true={true.mark_2.nb_mean:.3f}  fit={fit.mark_2.nb_mean:.3f}")


def main(seed=42, T=1500.0):
    rng = np.random.default_rng(seed)
    truth = make_ground_truth()

    events = simulate(truth, T=T, rng=rng)
    t, L, D, C, z_true = events_to_arrays(events)
    N = len(t)
    print(f"Simulated {N} events on [0, {T}].")
    print(f"  expected N ~ (mu_1+mu_2)*T / (1 - alpha_1 - alpha_2) = "
          f"{(truth.mu_1 + truth.mu_2) * T / max(1e-3, 1 - truth.alpha_1 - truth.alpha_2):.1f}")
    print(f"  true z=1 fraction: {(z_true == 1).mean():.3f}")
    print()

    params, history = fit_em(t, L, D, C, T=T, n_iter=80, tol=1e-6, verbose=False)
    print(f"Fit converged: logL from {history[0]:.2f} -> {history[-1]:.2f} in {len(history)-1} iters.")
    print()
    print("Parameter recovery:")
    report_recovery(truth, params, prefix="  ")
    print()

    # z-posterior accuracy
    z_prob_1 = predict_z_posterior(t, L, D, C, params)
    z_pred = np.where(z_prob_1 > 0.5, 1, 2)
    acc = (z_pred == z_true).mean()
    print(f"Latent label (z) classification accuracy: {acc:.3f}  (chance = {(z_true==1).mean():.3f})")

    # Error summary
    err_mu = np.hypot(params.mu_1 - truth.mu_1, params.mu_2 - truth.mu_2)
    err_alpha = np.hypot(params.alpha_1 - truth.alpha_1, params.alpha_2 - truth.alpha_2)
    err_beta = np.hypot(params.beta_1 - truth.beta_1, params.beta_2 - truth.beta_2)
    print()
    print("L2 parameter errors:")
    print(f"  ||mu_hat - mu||     = {err_mu:.4f}")
    print(f"  ||alpha_hat - alpha|| = {err_alpha:.4f}")
    print(f"  ||beta_hat - beta||   = {err_beta:.4f}")

    # Pass/fail gates for this sanity check
    passed = (
        err_mu < 0.05
        and err_alpha < 0.1
        and err_beta < 0.2
        and acc > 0.8
    )
    print()
    print("SANITY CHECK:", "PASS" if passed else "REVIEW")
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
