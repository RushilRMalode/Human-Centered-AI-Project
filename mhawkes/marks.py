"""Mark distributions and weighted-MLE updates for the mixed-type mark model.

Per system s in {1, 2}:
    L_k  ~  NegBin(mean=lam_s, dispersion=k_s)     # session event count
    D_k  ~  LogNormal(mu_s, sigma2_s)              # session dwell / watch-time
    C_k  ~  Bernoulli(p_s)                         # like / share indicator

Weighted-MLE updates:
  - LogNormal, Bernoulli: closed form.
  - NegBin: method-of-moments on (mean, variance). Collapses to Poisson
    (dispersion -> inf) when var <= mean. This avoids the Newton solve
    on the dispersion parameter while staying inside an identifiable family.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

EPS = 1e-9


@dataclass
class SystemMarkParams:
    # NegBin
    nb_mean: float
    nb_disp: float
    # LogNormal
    ln_mu: float
    ln_sigma2: float
    # Bernoulli
    be_p: float


def negbin_logpmf(x: np.ndarray, mean: float, disp: float) -> np.ndarray:
    mean = max(float(mean), EPS)
    disp = max(float(disp), EPS)
    return (
        gammaln(x + disp) - gammaln(disp) - gammaln(x + 1.0)
        + disp * (np.log(disp) - np.log(disp + mean))
        + x * (np.log(mean) - np.log(disp + mean))
    )


def lognormal_logpdf(x: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
    sigma2 = max(float(sigma2), EPS)
    log_x = np.log(np.maximum(x, EPS))
    return (
        -0.5 * np.log(2.0 * np.pi * sigma2)
        - log_x
        - 0.5 * (log_x - mu) ** 2 / sigma2
    )


def bernoulli_logpmf(x: np.ndarray, p: float) -> np.ndarray:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return x * np.log(p) + (1.0 - x) * np.log(1.0 - p)


def joint_logpdf(L, D, C, params: SystemMarkParams) -> np.ndarray:
    return (
        negbin_logpmf(L, params.nb_mean, params.nb_disp)
        + lognormal_logpdf(D, params.ln_mu, params.ln_sigma2)
        + bernoulli_logpmf(C, params.be_p)
    )


# ---------------------------------------------------------------------------
# Weighted-MLE M-step updates
# ---------------------------------------------------------------------------

def _safe_weights(w: np.ndarray) -> float:
    W = float(np.sum(w))
    return W if W > EPS else EPS


def negbin_weighted_mme(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    W = _safe_weights(w)
    mean = float(np.sum(w * x) / W)
    mean = max(mean, EPS)
    var = float(np.sum(w * (x - mean) ** 2) / W)
    if var <= mean + EPS:
        disp = 1e6  # Poisson limit
    else:
        disp = mean ** 2 / (var - mean)
        disp = float(np.clip(disp, 1e-3, 1e6))
    return mean, disp


def lognormal_weighted_mle(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    W = _safe_weights(w)
    log_x = np.log(np.maximum(x, EPS))
    mu = float(np.sum(w * log_x) / W)
    sigma2 = float(np.sum(w * (log_x - mu) ** 2) / W)
    sigma2 = max(sigma2, 1e-6)
    return mu, sigma2


def bernoulli_weighted_mle(x: np.ndarray, w: np.ndarray) -> float:
    W = _safe_weights(w)
    p = float(np.sum(w * x) / W)
    return float(np.clip(p, EPS, 1.0 - EPS))


def weighted_update(L, D, C, w) -> SystemMarkParams:
    nb_mean, nb_disp = negbin_weighted_mme(L, w)
    ln_mu, ln_sigma2 = lognormal_weighted_mle(D, w)
    be_p = bernoulli_weighted_mle(C, w)
    return SystemMarkParams(nb_mean, nb_disp, ln_mu, ln_sigma2, be_p)


def sample(params: SystemMarkParams, rng: np.random.Generator) -> np.ndarray:
    # NegBin sampling via Gamma-Poisson mixture: X ~ Poisson(Gamma(r, p))
    r = params.nb_disp
    mean = params.nb_mean
    p = r / (r + mean)
    # numpy uses P(success)=p; mean = r*(1-p)/p
    L = rng.negative_binomial(r, p)
    D = rng.lognormal(mean=params.ln_mu, sigma=np.sqrt(params.ln_sigma2))
    C = rng.binomial(1, params.be_p)
    return np.array([L, D, C], dtype=float)
