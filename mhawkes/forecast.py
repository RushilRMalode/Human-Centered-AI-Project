"""Held-out forecasting evaluation for the marked dual-kernel Hawkes process.

Three quantities, all computed on the test segment of an event sequence:

1. **Per-event predictive log-likelihood.** The standard generative-model
   metric. Higher = the model assigns more probability mass to the events
   that actually occurred. We compute it as

       L_test = sum_{k: t_k > T_train} log lambda_marked(t_k | history)
                 - integral_{T_train}^{T_total} lambda(t) dt

   where the offspring contributions for each test event use the *full*
   prior history (train events + earlier test events), and the integral
   includes kernel mass deposited by both train and test events that
   bleeds into the test window.

2. **Time-rescaling PIT.** By the time-rescaling theorem, if events are
   drawn from a process with intensity lambda, then the rescaled gaps
   Lambda(t_k) - Lambda(t_{k-1}) are iid Exp(1) under the true model.
   So PIT_k = 1 - exp(-gap_k) is Uniform[0,1] under correct calibration.
   The KS distance from the empirical PIT distribution to U[0,1] is a
   single-number calibration metric (lower = better).

3. **Held-out z-accuracy.** Same swap-aware accuracy as Exp 4, but on
   events the model never saw during training.

A simple Poisson + iid-marks baseline is provided for comparison, with
the same predictive interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .em import FittedParams
from .marks import SystemMarkParams, joint_logpdf, weighted_update


# ---------------------------------------------------------------------------
# Marked / unmarked Hawkes evaluation
# ---------------------------------------------------------------------------

def _per_event_log_intensity(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    params: FittedParams, use_marks: bool,
) -> np.ndarray:
    """log of (intensity * mark density) at each event, given the full
    history. Returns an array of shape (N,).

    For event k, the four candidate sources are: immigrant-1, immigrant-2,
    offspring of any earlier event l via kernel 1, offspring of l via
    kernel 2. The log of the marked intensity at t_k is the logsumexp
    over those candidates.
    """
    N = len(t)
    if N == 0:
        return np.zeros(0)

    if use_marks:
        log_f1 = joint_logpdf(L, D, C, params.mark_1)
        log_f2 = joint_logpdf(L, D, C, params.mark_2)
    else:
        log_f1 = np.zeros(N)
        log_f2 = np.zeros(N)

    dt = t[:, None] - t[None, :]
    mask = dt > 0
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
    return logsumexp(log_rows, axis=1)


def _integral_window(
    t_hist: np.ndarray, T_lo: float, T_hi: float, params: FittedParams,
) -> float:
    """Integral of the dual-kernel Hawkes intensity over [T_lo, T_hi],
    given that events at times in `t_hist` (any with t < T_hi) contribute
    offspring mass during the window."""
    base = (params.mu_1 + params.mu_2) * (T_hi - T_lo)
    if len(t_hist) == 0:
        return base
    # For each event l with t_l < T_hi:
    #   lower limit of kernel mass during window is max(t_l, T_lo).
    lower = np.maximum(t_hist, T_lo)
    # Only events with t_l < T_hi contribute; clip via mask.
    in_window = t_hist < T_hi
    # Kernel mass during window for event l, kernel j:
    #   alpha_j * [exp(-beta_j (T_lo' - t_l)) - exp(-beta_j (T_hi - t_l))]
    # where T_lo' = max(t_l, T_lo).
    if not np.any(in_window):
        return float(base)
    tl = t_hist[in_window]
    lo = lower[in_window]
    K1 = params.alpha_1 * (np.exp(-params.beta_1 * (lo - tl))
                           - np.exp(-params.beta_1 * (T_hi - tl)))
    K2 = params.alpha_2 * (np.exp(-params.beta_2 * (lo - tl))
                           - np.exp(-params.beta_2 * (T_hi - tl)))
    return float(base + K1.sum() + K2.sum())


def predictive_log_likelihood_hawkes(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    T_train: float, T_total: float, params: FittedParams,
    use_marks: bool = True,
) -> tuple[float, int]:
    """Held-out marked-Hawkes log-likelihood on events in (T_train, T_total].

    Returns (logL_total, n_test_events). Per-event logL is logL_total / n_test.
    """
    test_mask = t > T_train
    n_test = int(test_mask.sum())
    if n_test == 0:
        return 0.0, 0

    # log marked-intensity at every event; pick out the test events.
    log_intens = _per_event_log_intensity(t, L, D, C, params, use_marks=use_marks)
    sum_log_intens_test = float(log_intens[test_mask].sum())

    integral = _integral_window(t, T_train, T_total, params)
    return sum_log_intens_test - integral, n_test


def compensator_at(
    t_eval: np.ndarray, t_hist: np.ndarray, params: FittedParams,
) -> np.ndarray:
    """Compute Lambda(t) = integral_0^t lambda(s) ds at each time in
    `t_eval`, using the offspring contributions of all events in `t_hist`
    that satisfy t_l < t.
    """
    out = np.zeros(len(t_eval))
    for i, t in enumerate(t_eval):
        out[i] = (params.mu_1 + params.mu_2) * t
        if len(t_hist) == 0:
            continue
        prior = t_hist[t_hist < t]
        if len(prior) == 0:
            continue
        K1 = params.alpha_1 * (1.0 - np.exp(-params.beta_1 * (t - prior)))
        K2 = params.alpha_2 * (1.0 - np.exp(-params.beta_2 * (t - prior)))
        out[i] += K1.sum() + K2.sum()
    return out


def pit_values_hawkes(
    t: np.ndarray, T_train: float, params: FittedParams,
) -> np.ndarray:
    """PIT values for the inter-arrival gaps that *end* in a test event.

    Under the true generative model, these are Uniform[0, 1] (time-rescaling
    theorem). Returns one PIT value per test event.
    """
    test_idx = np.where(t > T_train)[0]
    if len(test_idx) == 0:
        return np.zeros(0)

    eval_times = np.concatenate([[T_train], t])
    Lambda = compensator_at(eval_times, t, params)
    Lam_train = Lambda[0]
    Lam_events = Lambda[1:]

    # gap for first test event uses Lambda(T_train) as the lower anchor;
    # gap for subsequent test events uses the previous event's Lambda.
    gaps = np.empty(len(test_idx))
    for i, k in enumerate(test_idx):
        lo = Lam_events[k - 1] if k > 0 else Lam_train
        # If the previous event is in the train segment, lo is its compensator
        # value (still valid). If k == 0 there are no prior events at all and
        # lo is Lambda(T_train), which uses no history.
        gaps[i] = max(Lam_events[k] - lo, 0.0)
    return 1.0 - np.exp(-gaps)


def ks_distance_to_uniform(pit: np.ndarray) -> float:
    """One-sample KS distance from empirical CDF of PIT values to U[0, 1]."""
    if len(pit) == 0:
        return float("nan")
    p = np.sort(pit)
    n = len(p)
    cdf_emp = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf_emp - p)))


# ---------------------------------------------------------------------------
# Poisson + iid-marks baseline
# ---------------------------------------------------------------------------

@dataclass
class PoissonIIDFit:
    rate: float
    mark: SystemMarkParams


def fit_poisson_iid(
    t_train: np.ndarray, L_train: np.ndarray, D_train: np.ndarray,
    C_train: np.ndarray, T_train: float,
) -> PoissonIIDFit:
    """Maximum-likelihood fit of a homogeneous Poisson process with iid
    marks drawn from the same NegBin x LogNormal x Bernoulli family
    (single distribution, no system labels)."""
    rate = float(len(t_train) / max(T_train, 1e-9))
    weights = np.ones(len(t_train), dtype=float)
    mark = weighted_update(L_train, D_train, C_train, weights)
    return PoissonIIDFit(rate=rate, mark=mark)


def predictive_log_likelihood_poisson(
    t: np.ndarray, L: np.ndarray, D: np.ndarray, C: np.ndarray,
    T_train: float, T_total: float, fit: PoissonIIDFit,
) -> tuple[float, int]:
    """Held-out predictive log-likelihood under the Poisson+iid baseline."""
    test_mask = t > T_train
    n_test = int(test_mask.sum())
    if n_test == 0:
        return 0.0, 0
    L_te, D_te, C_te = L[test_mask], D[test_mask], C[test_mask]
    log_marks = joint_logpdf(L_te, D_te, C_te, fit.mark).sum()
    log_rate_term = n_test * np.log(max(fit.rate, 1e-300))
    integral = fit.rate * (T_total - T_train)
    return float(log_marks + log_rate_term - integral), n_test


def pit_values_poisson(t: np.ndarray, T_train: float, fit: PoissonIIDFit) -> np.ndarray:
    """PIT values under a homogeneous Poisson rate (no history needed)."""
    test_idx = np.where(t > T_train)[0]
    if len(test_idx) == 0:
        return np.zeros(0)
    eval_times = np.concatenate([[T_train], t])  # last train point or T_train
    # For Poisson, gap = rate * delta_t.
    out = np.empty(len(test_idx))
    for i, k in enumerate(test_idx):
        lo = t[k - 1] if k > 0 else T_train
        out[i] = fit.rate * (t[k] - lo)
    return 1.0 - np.exp(-np.maximum(out, 0.0))
