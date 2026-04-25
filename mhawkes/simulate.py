"""Ogata thinning simulator for the dual-kernel marked Hawkes process.

Intensity:
    lambda(t) = (mu_1 + mu_2)
              + sum_{t_l < t} [ a1 * b1 * exp(-b1 (t-t_l))
                              + a2 * b2 * exp(-b2 (t-t_l)) ]

Both kernels are triggered by every past event (dual self-excitation).
We record the ground-truth latent label z_k in {1, 2} for validation:
  - immigrants are labelled 1 with probability mu_1/(mu_1+mu_2), else 2.
  - offspring via kernel j receive label j.

The per-event state (s1, s2) is updated incrementally to avoid an O(N)
recomputation of the sum at each step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .marks import SystemMarkParams, sample as sample_mark


@dataclass
class TrueParams:
    mu_1: float
    mu_2: float
    alpha_1: float
    alpha_2: float
    beta_1: float
    beta_2: float
    mark_1: SystemMarkParams
    mark_2: SystemMarkParams


@dataclass
class Event:
    t: float
    L: int
    D: float
    C: int
    z: int  # ground-truth system label


def simulate(params: TrueParams, T: float, rng: np.random.Generator) -> list[Event]:
    mu = params.mu_1 + params.mu_2
    p_immigrant_1 = params.mu_1 / max(mu, 1e-12)

    a1, b1 = params.alpha_1, params.beta_1
    a2, b2 = params.alpha_2, params.beta_2

    s1 = 0.0
    s2 = 0.0
    t = 0.0
    events: list[Event] = []

    while True:
        lam_upper = mu + s1 + s2
        if lam_upper <= 0:
            break
        dt = rng.exponential(1.0 / lam_upper)
        t_cand = t + dt
        if t_cand >= T:
            break

        # Decay states to t_cand
        s1_c = s1 * np.exp(-b1 * dt)
        s2_c = s2 * np.exp(-b2 * dt)
        lam_actual = mu + s1_c + s2_c

        if rng.random() * lam_upper <= lam_actual:
            # Accept. Determine triggering component.
            p_imm = mu / lam_actual
            p_s1 = s1_c / lam_actual
            u = rng.random()
            if u < p_imm:
                z = 1 if rng.random() < p_immigrant_1 else 2
            elif u < p_imm + p_s1:
                z = 1
            else:
                z = 2
            mark_params = params.mark_1 if z == 1 else params.mark_2
            L, D, C = sample_mark(mark_params, rng)
            events.append(Event(t=float(t_cand), L=int(L), D=float(D), C=int(C), z=z))
            # Both kernels get a jump of alpha_j * beta_j from this event
            s1 = s1_c + a1 * b1
            s2 = s2_c + a2 * b2
        else:
            s1 = s1_c
            s2 = s2_c

        t = t_cand

    return events


def events_to_arrays(events: list[Event]):
    t = np.array([e.t for e in events], dtype=float)
    L = np.array([e.L for e in events], dtype=float)
    D = np.array([e.D for e in events], dtype=float)
    C = np.array([e.C for e in events], dtype=float)
    z = np.array([e.z for e in events], dtype=int)
    return t, L, D, C, z
