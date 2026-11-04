#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Sandved-Smith et al. (2021) computational phenomenology model.
Mathematical operations for active inference with hierarchical precision control.
"""

from typing import Tuple, Union

import numpy as np


def softmax(X: Union[np.ndarray, list], axis: int = -1) -> np.ndarray:
    """
    Convert log probabilities to normalized probabilities with numerical stability.
    """
    X = np.asarray(X, dtype=float)
    X_max = np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X - X_max) + 1e-12
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


def softmax_dim2(X: np.ndarray) -> np.ndarray:
    """
    Convert matrix of log probabilities to matrix of normalized probabilities.
    Normalizes along axis 0 (columns).
    """
    return softmax(X, axis=0)


def normalise(X: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a matrix of probabilities along columns (axis 0).
    """
    X = np.asarray(X, dtype=float)
    s = np.sum(X, axis=axis, keepdims=True)
    s = np.where(s == 0, eps, s)
    return X / s


def precision_weighted_likelihood(A: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply precision weighting to likelihood matrix.
    """
    A = np.asarray(A, dtype=float)
    return softmax_dim2(np.log(np.clip(A, 1e-16, 1.0)) * gamma)


def bayesian_model_average(
    beta_values: np.ndarray,
    state_beliefs: np.ndarray,
    likelihood_matrix: np.ndarray | None = None,
) -> float:
    """
    Compute Bayesian model average for precision beliefs.
    """
    beta_values = np.asarray(beta_values, dtype=float)
    state_beliefs = np.asarray(state_beliefs, dtype=float)
    if likelihood_matrix is not None:
        mapped = np.dot(likelihood_matrix.T, state_beliefs)
        mapped = mapped / np.sum(mapped)
        return float(np.sum(beta_values * mapped))
    else:
        weights = state_beliefs / np.sum(state_beliefs)
        return float(np.sum(beta_values * weights))


def compute_attentional_charge(
    O_bar: np.ndarray, A_bar: np.ndarray, X_bar: np.ndarray, A: np.ndarray
) -> float:
    """
    Compute 'attentional charge' - the inverse precision updating term.
    Based on "Uncertainty, epistemics and active inference" (Parr & Friston).
    """
    charge = 0.0
    n_obs, n_states = A.shape

    for i in range(n_obs):  # Loop over outcomes
        for j in range(n_states):  # Loop over states
            charge += (
                (O_bar[i] - A_bar[i, j])
                * X_bar[j]
                * np.log(np.clip(A[i, j], 1e-16, 1.0))
            )

    return float(charge)


def expected_free_energy(
    O_pred: np.ndarray, C: np.ndarray, X_pred: np.ndarray, H: np.ndarray
) -> float:
    """
    Compute expected free energy for a policy.
    G = E[o*(ln(o) - C) - x*H]
    """
    O_pred = np.asarray(O_pred, dtype=float)
    C = np.asarray(C, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    H = np.asarray(H, dtype=float)
    epistemic_term = np.sum(O_pred * (np.log(np.clip(O_pred, 1e-16, 1.0)) - C))
    pragmatic_term = -np.sum(X_pred * H)
    return float(epistemic_term + pragmatic_term)


def variational_free_energy(
    X_bar: np.ndarray, X_pred: np.ndarray, A: np.ndarray, obs_idx: int
) -> float:
    """
    Compute variational free energy term for policy evaluation.
    """
    X_bar = np.asarray(X_bar, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    A = np.asarray(A, dtype=float)
    return float(
        np.sum(
            X_bar
            * (
                np.log(np.clip(X_bar, 1e-16, 1.0))
                - np.log(np.clip(A[obs_idx, :], 1e-16, 1.0))
                - np.log(np.clip(X_pred, 1e-16, 1.0))
            )
        )
    )


def update_precision_beliefs(
    beta_prior: float, charge: float, beta_bounds: Tuple[float, float]
) -> float:
    """
    Update inverse precision beliefs based on prediction error.
    """
    min_beta, max_beta = beta_bounds

    if charge > min_beta:
        charge = min_beta - 1e-5

    beta_posterior = beta_prior - charge
    beta_posterior = np.clip(beta_posterior, min_beta, max_beta)

    return float(beta_posterior)


def policy_posterior(
    log_prior: np.ndarray,
    expected_free_energy_vals: np.ndarray,
    variational_free_energy_vals: np.ndarray | None = None,
    gamma_G: float = 1.0,
) -> np.ndarray:
    """
    Compute posterior beliefs over policies.
    """
    log_prior = np.asarray(log_prior, dtype=float)
    expected_free_energy_vals = np.asarray(expected_free_energy_vals, dtype=float)
    log_post = log_prior - gamma_G * expected_free_energy_vals

    if variational_free_energy_vals is not None:
        log_post -= np.asarray(variational_free_energy_vals, dtype=float)

    return softmax(log_post)


def discrete_choice(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """
    Make discrete choice based on probability distribution.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 0:
        return 0
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def generate_oddball_sequence(T: int, oddball_times: list | None = None) -> np.ndarray:
    """
    Generate oddball stimulus sequence.
    """
    sequence = np.zeros(T, dtype=int)

    if oddball_times is None:
        oddball_times = [int(T / 5), int(2 * T / 5), int(3 * T / 5), int(4 * T / 5)]

    for t in oddball_times:
        if 0 <= t < T:
            sequence[t] = 1

    return sequence


def setup_transition_matrices() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Set up transition matrices for the three-level model.
    """
    B1 = np.array([[0.8, 0.2], [0.2, 0.8]])
    B2a = np.array(
        [
            [0.8, 0.0],  # Stay policy
            [0.2, 1.0],
        ]
    )
    B2b = np.array(
        [
            [0.0, 1.0],  # Switch policy
            [1.0, 0.0],
        ]
    )
    B3 = np.array([[0.9, 0.1], [0.1, 0.9]])

    return B1, B2a, B2b, B3


def setup_likelihood_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up likelihood matrices for the three-level model.
    """
    A1 = np.array([[0.75, 0.25], [0.25, 0.75]])
    A2 = np.array([[0.65, 0.35], [0.35, 0.65]])
    A3 = np.array([[0.9, 0.1], [0.1, 0.9]])

    return A1, A2, A3


def compute_entropy_terms(A: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute entropy terms for expected free energy calculation.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        H = np.zeros(A.shape[1])
        for j in range(A.shape[1]):
            col = np.clip(A[:, j], 1e-16, 1.0)
            H[j] = np.sum(col * np.log(col))
        return H
    P = np.clip(A, 1e-16, 1.0)
    return -np.sum(P * np.log(P), axis=axis)
