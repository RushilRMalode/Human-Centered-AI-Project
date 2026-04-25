"""Stress test the flagship claim: with beta_1 == beta_2, the unmarked model
is provably non-identifiable but the marked model should still recover
(alpha_1, alpha_2) and the mark parameters from the mark channel alone.

Runs two scenarios:
    (a) beta_1 == beta_2 == 1.0, marks clearly distinct  -> should recover
    (b) beta_1 == beta_2 == 1.0, marks nearly identical  -> should fail

Usage:  python -m experiments.stress_beta_collapse
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mhawkes.em import fit_em, predict_z_posterior  # noqa: E402
from mhawkes.marks import SystemMarkParams  # noqa: E402
from mhawkes.simulate import TrueParams, events_to_arrays, simulate  # noqa: E402


def build(marks_distinct: bool) -> TrueParams:
    if marks_distinct:
        m1 = SystemMarkParams(nb_mean=3.0, nb_disp=5.0, ln_mu=0.0,  ln_sigma2=0.25, be_p=0.05)
        m2 = SystemMarkParams(nb_mean=12.0, nb_disp=5.0, ln_mu=1.61, ln_sigma2=0.25, be_p=0.6)
    else:
        # Identical marks up to tiny noise
        m1 = SystemMarkParams(nb_mean=7.0, nb_disp=5.0, ln_mu=0.8, ln_sigma2=0.25, be_p=0.3)
        m2 = SystemMarkParams(nb_mean=7.1, nb_disp=5.0, ln_mu=0.82, ln_sigma2=0.25, be_p=0.32)
    return TrueParams(
        mu_1=0.05, mu_2=0.03,
        alpha_1=0.4, alpha_2=0.3,
        beta_1=1.0, beta_2=1.0,       # DEGENERATE: original paper's failure regime
        mark_1=m1, mark_2=m2,
    )


def run_scenario(name: str, truth: TrueParams, T: float, seed: int):
    rng = np.random.default_rng(seed)
    events = simulate(truth, T=T, rng=rng)
    t, L, D, C, z_true = events_to_arrays(events)
    print(f"\n== {name}  (N = {len(t)} events) ==")
    params, history = fit_em(t, L, D, C, T=T, n_iter=120, tol=1e-7)
    print(f"  logL: {history[0]:.2f} -> {history[-1]:.2f}")
    print(f"  alpha  true=({truth.alpha_1:.3f},{truth.alpha_2:.3f})  "
          f"fit=({params.alpha_1:.3f},{params.alpha_2:.3f})")
    print(f"  beta   true=({truth.beta_1:.3f},{truth.beta_2:.3f})  "
          f"fit=({params.beta_1:.3f},{params.beta_2:.3f})")
    print(f"  S1 dwell mu  true={truth.mark_1.ln_mu:.3f}  fit={params.mark_1.ln_mu:.3f}")
    print(f"  S2 dwell mu  true={truth.mark_2.ln_mu:.3f}  fit={params.mark_2.ln_mu:.3f}")
    print(f"  S1 like p    true={truth.mark_1.be_p:.3f}   fit={params.mark_1.be_p:.3f}")
    print(f"  S2 like p    true={truth.mark_2.be_p:.3f}   fit={params.mark_2.be_p:.3f}")
    z_prob_1 = predict_z_posterior(t, L, D, C, params)
    z_pred = np.where(z_prob_1 > 0.5, 1, 2)
    acc = (z_pred == z_true).mean()
    print(f"  z-label accuracy: {acc:.3f}  (chance = {max((z_true==1).mean(), (z_true==2).mean()):.3f})")
    # Error on alpha and marks (not beta - we can't separate betas when they collide)
    err_alpha = np.hypot(params.alpha_1 - truth.alpha_1, params.alpha_2 - truth.alpha_2)
    err_marks = (
        abs(params.mark_1.ln_mu - truth.mark_1.ln_mu)
        + abs(params.mark_2.ln_mu - truth.mark_2.ln_mu)
        + abs(params.mark_1.be_p - truth.mark_1.be_p)
        + abs(params.mark_2.be_p - truth.mark_2.be_p)
    )
    return {"err_alpha": err_alpha, "err_marks": err_marks, "z_acc": acc}


def main():
    T = 1500.0
    r_ok = run_scenario("A) beta collapse, distinct marks", build(True),  T=T, seed=7)
    r_bad = run_scenario("B) beta collapse, identical marks (should fail)",
                          build(False), T=T, seed=7)
    print("\n-- summary --")
    print(f"  distinct marks:  alpha_err={r_ok['err_alpha']:.3f}  z_acc={r_ok['z_acc']:.3f}")
    print(f"  identical marks: alpha_err={r_bad['err_alpha']:.3f}  z_acc={r_bad['z_acc']:.3f}")
    # Expect: distinct marks recover alpha despite beta_1=beta_2; identical do not
    verdict = (
        r_ok["err_alpha"] < 0.15 and r_ok["z_acc"] > 0.8
        and (r_bad["err_alpha"] > 0.1 or r_bad["z_acc"] < 0.7)
    )
    print("FLAGSHIP CLAIM:", "SUPPORTED" if verdict else "NEEDS REVIEW")
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
