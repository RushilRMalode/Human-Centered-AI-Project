"""Branching-EM for the marked dual-kernel Hawkes process.

Latents per event k:
  - parent assignment (0 = immigrant from mu_j, or event l < k via kernel j)
  - system label z_k in {1, 2}

The latent indexing is actually a single variable per event: one of the
"sources" that could have produced event k. The sources are:
  - immigrant, label 1:   weight = mu_1 * f_1(y_k)
  - immigrant, label 2:   weight = mu_2 * f_2(y_k)
  - offspring of l via kernel j (for l < k, j in {1,2}):
      weight = alpha_j * beta_j * exp(-beta_j (t_k - t_l)) * f_j(y_k)

E-step: normalise to get responsibilities r_k0(j), r_kl(j).
M-step (closed form under large-T approximation):
  mu_j    = sum_k r_k0(j) / T
  alpha_j = sum_k sum_l r_kl(j) / N
  beta_j  = sum_k sum_l r_kl(j) / sum_k sum_l r_kl(j) * (t_k - t_l)
  mark_j  = weighted MLE with weights r_k(j) = r_k0(j) + sum_l r_kl(j)

Label-switching is handled by enforcing beta_1 > beta_2 after each M-step.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from .marks import (
    SystemMarkParams,
    joint_logpdf,
    weighted_update,
)


@dataclass
class FittedParams:
    mu_1: float
    mu_2: float
    alpha_1: float
    alpha_2: float
    beta_1: float
    beta_2: float
    mark_1: SystemMarkParams
    mark_2: SystemMarkParams


def _default_init(L, D, C) -> FittedParams:
    """A simple asymmetric init that breaks label symmetry up front."""
    # Split events roughly by dwell: short-dwell -> S1, long-dwell -> S2.
    # This ensures the returned beta_1 > beta_2 is aligned with
    # "S1 = impulsive short dwell, S2 = deliberate long dwell".
    med = np.median(D)
    w1 = (D <= med).astype(float)
    w2 = 1.0 - w1
    m1 = weighted_update(L, D, C, w1)
    m2 = weighted_update(L, D, C, w2)
    return FittedParams(
        mu_1=0.05,
        mu_2=0.05,
        alpha_1=0.3,
        alpha_2=0.2,
        beta_1=1.0,
        beta_2=0.1,
        mark_1=m1,
        mark_2=m2,
    )


def _random_init(L, D, C, rng: np.random.Generator) -> FittedParams:
    """Random asymmetric init: soft-partition events by a random dwell
    quantile. Distinct from `_default_init` so multi-restart EM explores
    different local optima."""
    # Random quantile in [0.3, 0.7] for the split; random logistic slope
    # for how soft/hard the soft-assignment is.
    q = rng.uniform(0.3, 0.7)
    threshold = np.quantile(D, q)
    scale = np.std(np.log(np.maximum(D, 1e-9))) + 1e-3
    slope = rng.uniform(2.0, 6.0) / scale
    w1 = 1.0 / (1.0 + np.exp(slope * (np.log(np.maximum(D, 1e-9))
                                      - np.log(max(threshold, 1e-9)))))
    w2 = 1.0 - w1
    m1 = weighted_update(L, D, C, w1)
    m2 = weighted_update(L, D, C, w2)
    # Mild random jitter on scalar params
    return FittedParams(
        mu_1=float(rng.uniform(0.02, 0.10)),
        mu_2=float(rng.uniform(0.02, 0.10)),
        alpha_1=float(rng.uniform(0.15, 0.45)),
        alpha_2=float(rng.uniform(0.15, 0.35)),
        beta_1=float(rng.uniform(0.6, 1.4)),
        beta_2=float(rng.uniform(0.05, 0.5)),
        mark_1=m1,
        mark_2=m2,
    )


def _log_mark(L, D, C, params: SystemMarkParams) -> np.ndarray:
    return joint_logpdf(L, D, C, params)


def log_likelihood(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    T: float, params: FittedParams,
    use_marks: bool = True,
) -> float:
    """Observed-data marked-Hawkes log-likelihood.

        log L = sum_k log [sum_source w_source_k * f_source(y_k)] - integral lambda dt

    When ``use_marks`` is False the mark contribution is dropped, reducing to
    the paper's unmarked dual-kernel Hawkes likelihood.
    """
    N = len(t)
    if N == 0:
        return -(params.mu_1 + params.mu_2) * T

    if use_marks:
        log_f1 = _log_mark(L, D, C, params.mark_1)
        log_f2 = _log_mark(L, D, C, params.mark_2)
    else:
        log_f1 = np.zeros(N)
        log_f2 = np.zeros(N)

    # Pre-compute pairwise dt matrix (k, l), masked to l < k.
    dt = t[:, None] - t[None, :]  # (N, N)
    mask = dt > 0

    # log kernel_j[k, l] = log(alpha_j * beta_j) - beta_j * dt[k, l]
    log_a1b1 = np.log(max(params.alpha_1 * params.beta_1, 1e-300))
    log_a2b2 = np.log(max(params.alpha_2 * params.beta_2, 1e-300))
    log_K1 = np.where(mask, log_a1b1 - params.beta_1 * dt, -np.inf)
    log_K2 = np.where(mask, log_a2b2 - params.beta_2 * dt, -np.inf)

    # Add mark log-likelihood per row: offs_j_kl adds log_f_j(y_k).
    log_offs_1 = log_K1 + log_f1[:, None]
    log_offs_2 = log_K2 + log_f2[:, None]

    # Immigrant contributions per event.
    log_base_1 = np.log(max(params.mu_1, 1e-300)) + log_f1
    log_base_2 = np.log(max(params.mu_2, 1e-300)) + log_f2

    # Z_k = logsumexp over {base_1, base_2, offs_1[k,:], offs_2[k,:]}.
    # Assemble per row and take logsumexp.
    log_rows = np.concatenate(
        [
            log_base_1[:, None],
            log_base_2[:, None],
            log_offs_1,
            log_offs_2,
        ],
        axis=1,
    )
    log_Z = logsumexp(log_rows, axis=1)

    # Integral of intensity over [0, T].
    integral = (params.mu_1 + params.mu_2) * T
    # Offspring integrals: sum over past events of alpha_j (1 - exp(-beta_j (T - t_l)))
    tail_1 = params.alpha_1 * (1.0 - np.exp(-params.beta_1 * (T - t))).sum()
    tail_2 = params.alpha_2 * (1.0 - np.exp(-params.beta_2 * (T - t))).sum()
    integral += tail_1 + tail_2

    return float(log_Z.sum() - integral)


def fit_em(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    T: float,
    init: FittedParams | None = None,
    n_iter: int = 50,
    tol: float = 1e-5,
    verbose: bool = False,
    use_marks: bool = True,
) -> tuple[FittedParams, list[float]]:
    """Run branching-EM for `n_iter` iterations (or until log-lik converges).

    Set ``use_marks=False`` to reproduce the paper's unmarked-baseline EM
    (timing-only disentanglement). Under that setting, label assignment
    relies entirely on the beta-gap.

    Returns the fitted parameters and the trace of log-likelihoods.
    """
    N = len(t)
    if init is None:
        init = _default_init(L, D, C)
    params = init
    history = [log_likelihood(t, L, D, C, T, params, use_marks=use_marks)]

    # Pairwise dt (N, N) computed once
    dt = t[:, None] - t[None, :]
    mask = dt > 0  # l < k

    for it in range(n_iter):
        # ---- E-step ----
        if use_marks:
            log_f1 = _log_mark(L, D, C, params.mark_1)
            log_f2 = _log_mark(L, D, C, params.mark_2)
        else:
            log_f1 = np.zeros(N)
            log_f2 = np.zeros(N)

        log_a1b1 = np.log(max(params.alpha_1 * params.beta_1, 1e-300))
        log_a2b2 = np.log(max(params.alpha_2 * params.beta_2, 1e-300))
        log_K1 = np.where(mask, log_a1b1 - params.beta_1 * dt, -np.inf)
        log_K2 = np.where(mask, log_a2b2 - params.beta_2 * dt, -np.inf)

        log_offs_1 = log_K1 + log_f1[:, None]
        log_offs_2 = log_K2 + log_f2[:, None]
        log_base_1 = np.log(max(params.mu_1, 1e-300)) + log_f1
        log_base_2 = np.log(max(params.mu_2, 1e-300)) + log_f2

        log_rows = np.concatenate(
            [log_base_1[:, None], log_base_2[:, None], log_offs_1, log_offs_2],
            axis=1,
        )
        log_Z = logsumexp(log_rows, axis=1)

        # Responsibilities (normalised).
        r_base_1 = np.exp(log_base_1 - log_Z)
        r_base_2 = np.exp(log_base_2 - log_Z)
        r_offs_1 = np.exp(log_offs_1 - log_Z[:, None])
        r_offs_2 = np.exp(log_offs_2 - log_Z[:, None])

        # Per-event label posterior r_k(j) = r_kb(j) + sum_l r_kl(j)
        r_1 = r_base_1 + r_offs_1.sum(axis=1)
        r_2 = r_base_2 + r_offs_2.sum(axis=1)

        # Numerical safety: force r_1 + r_2 == 1 exactly
        s = r_1 + r_2
        r_1 /= np.maximum(s, 1e-300)
        r_2 /= np.maximum(s, 1e-300)

        # ---- M-step ----
        # Base rates
        mu_1 = float(r_base_1.sum() / T)
        mu_2 = float(r_base_2.sum() / T)

        # Offspring sums for alpha, beta
        A1 = float(r_offs_1.sum())
        A2 = float(r_offs_2.sum())
        # Weighted dt for beta update
        B1 = float(np.sum(r_offs_1 * dt))
        B2 = float(np.sum(r_offs_2 * dt))

        alpha_1 = np.clip(A1 / max(N, 1), 1e-4, 0.95)
        alpha_2 = np.clip(A2 / max(N, 1), 1e-4, 0.95)
        beta_1 = np.clip(A1 / max(B1, 1e-9), 1e-3, 1e3)
        beta_2 = np.clip(A2 / max(B2, 1e-9), 1e-3, 1e3)

        # Enforce beta_1 > beta_2 to avoid label switching.
        if beta_1 < beta_2:
            beta_1, beta_2 = beta_2, beta_1
            alpha_1, alpha_2 = alpha_2, alpha_1
            mu_1, mu_2 = mu_2, mu_1
            r_1, r_2 = r_2, r_1

        # Mark updates (skip if running the unmarked baseline)
        if use_marks:
            mark_1 = weighted_update(L, D, C, r_1)
            mark_2 = weighted_update(L, D, C, r_2)
        else:
            mark_1 = params.mark_1
            mark_2 = params.mark_2

        params = FittedParams(
            mu_1=float(mu_1), mu_2=float(mu_2),
            alpha_1=float(alpha_1), alpha_2=float(alpha_2),
            beta_1=float(beta_1), beta_2=float(beta_2),
            mark_1=mark_1, mark_2=mark_2,
        )
        ll = log_likelihood(t, L, D, C, T, params, use_marks=use_marks)
        history.append(ll)
        if verbose:
            print(f"[EM] iter {it+1:3d}  logL = {ll:10.3f}  "
                  f"mu=({params.mu_1:.3f},{params.mu_2:.3f}) "
                  f"alpha=({params.alpha_1:.3f},{params.alpha_2:.3f}) "
                  f"beta=({params.beta_1:.3f},{params.beta_2:.3f})")
        if abs(history[-1] - history[-2]) < tol * max(1.0, abs(history[-1])):
            break

    return params, history


def predict_z_posterior(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    params: FittedParams,
) -> np.ndarray:
    """Return P(z_k = 1 | all data) under the fitted model."""
    dt = t[:, None] - t[None, :]
    mask = dt > 0
    log_f1 = _log_mark(L, D, C, params.mark_1)
    log_f2 = _log_mark(L, D, C, params.mark_2)
    log_K1 = np.where(mask, np.log(max(params.alpha_1 * params.beta_1, 1e-300)) - params.beta_1 * dt, -np.inf)
    log_K2 = np.where(mask, np.log(max(params.alpha_2 * params.beta_2, 1e-300)) - params.beta_2 * dt, -np.inf)
    log_offs_1 = log_K1 + log_f1[:, None]
    log_offs_2 = log_K2 + log_f2[:, None]
    log_base_1 = np.log(max(params.mu_1, 1e-300)) + log_f1
    log_base_2 = np.log(max(params.mu_2, 1e-300)) + log_f2

    # Log numerator for z=1: logsumexp(base_1, offs_1)
    log_num_1 = logsumexp(
        np.concatenate([log_base_1[:, None], log_offs_1], axis=1), axis=1
    )
    log_num_2 = logsumexp(
        np.concatenate([log_base_2[:, None], log_offs_2], axis=1), axis=1
    )
    log_den = np.logaddexp(log_num_1, log_num_2)
    return np.exp(log_num_1 - log_den)


def fit_em_restarts(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    T: float,
    n_restarts: int = 3,
    n_iter: int = 100,
    tol: float = 1e-6,
    use_marks: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[FittedParams, list[float]]:
    """Multi-restart EM. Runs ``fit_em`` from several initialisations and
    keeps the fit with the highest final log-likelihood. Restart 0 uses
    the deterministic dwell-split init; restarts 1..n-1 use random inits.

    Returns (best_params, best_history). The returned history is the
    log-likelihood trace of the winning restart only.
    """
    rng = rng or np.random.default_rng()
    best_params, best_history = None, None
    best_ll = -np.inf

    for r in range(max(1, n_restarts)):
        if r == 0:
            init = _default_init(L, D, C)
        else:
            init = _random_init(L, D, C, rng)
        params, history = fit_em(
            t, L, D, C, T=T, init=init,
            n_iter=n_iter, tol=tol, use_marks=use_marks,
        )
        ll = history[-1]
        if ll > best_ll:
            best_ll = ll
            best_params = params
            best_history = history

    return best_params, best_history
