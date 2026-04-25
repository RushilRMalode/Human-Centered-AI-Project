"""Marked dual-kernel Hawkes process for System-1 / System-2 disentanglement.

Implements the model from the project proposal:

    lambda(t) = mu_1 + mu_2
              + sum_{t_l < t} [ alpha_1 * beta_1 * exp(-beta_1 (t - t_l))
                              + alpha_2 * beta_2 * exp(-beta_2 (t - t_l)) ]

Each event has a latent label z_k in {1, 2} (system identity) and a mark
vector y_k = (L_k, D_k, C_k) drawn from a system-specific mixed-type
product distribution: NegBin count x LogNormal dwell x Bernoulli indicator.
"""
