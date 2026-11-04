#!/usr/bin/env python3
"""
Identity Mixture Model (iMM) - Object identity classification module.

Implements the iMM from the AXIOM architecture, assigning discrete type labels
to object slots based on their color and shape features.
"""

import logging
from typing import Any, Dict

import numpy as np
from scipy.stats import invwishart, multivariate_normal

logger = logging.getLogger(__name__)


class IdentityMixtureModel:
    """
    Identity Mixture Model for object identity classification.

    Assigns discrete type labels to object slots based on their appearance
    features using Normal-Inverse-Wishart priors.
    """

    def __init__(self, K_slots: int, V_identities: int, alpha_imm: float = 1.0):
        """
        Initialize Identity Mixture Model.

        Args:
            K_slots: Number of object slots
            V_identities: Number of identity types
            alpha_imm: Stick-breaking concentration parameter
        """
        self.K_slots = K_slots
        self.V_identities = V_identities
        self.V_active = V_identities
        self.alpha_imm = alpha_imm

        # Identity assignments [K_slots, V_identities]
        self.z_identity = np.zeros((K_slots, V_identities))

        # Identity type parameters (NIW priors)
        self.theta_imm_mu = np.random.normal(
            loc=[0.5, 0.5, 0.5, 0.1, 0.1],  # Default gray, small object
            scale=0.1,
            size=(V_identities, 5),  # Color(3) + shape(2)
        )

        self.theta_imm_Sigma = np.array([0.1 * np.eye(5) for _ in range(V_identities)])

        # Mixing weights
        self.theta_imm_pi = self._initialize_stick_breaking_weights()

        # NIW hyperparameters
        self.m_identity = np.tile([0.5, 0.5, 0.5, 0.1, 0.1], (V_identities, 1))
        self.kappa_identity = np.ones(V_identities)
        self.U_identity = np.array([0.1 * np.eye(5) for _ in range(V_identities)])
        self.nu_identity = np.full(V_identities, 6.0)  # > dimensionality

        # Sufficient statistics for learning
        self.N_v = np.zeros(V_identities)
        self.x_bar_v = np.zeros((V_identities, 5))
        self.S_v = np.array([np.zeros((5, 5)) for _ in range(V_identities)])

        # Performance tracking
        self.inference_count = 0

    def _initialize_stick_breaking_weights(self) -> np.ndarray:
        """Initialize stick-breaking weights for identity types."""
        betas = np.random.beta(1, self.alpha_imm, self.V_identities)
        weights = np.zeros(self.V_identities)

        stick_remaining = 1.0
        for v in range(self.V_identities - 1):
            weights[v] = betas[v] * stick_remaining
            stick_remaining *= 1 - betas[v]
        weights[-1] = stick_remaining

        return weights

    def inference(self, s_appearance: np.ndarray) -> np.ndarray:
        """
        Perform inference to assign slots to identity types.

        Args:
            s_appearance: Slot appearance features [K_slots, 5] (color + shape)

        Returns:
            Identity assignments [K_slots, V_identities]
        """
        self.inference_count += 1

        # E-step: Update identity assignments
        self._update_identity_assignments(s_appearance)

        # M-step: Update NIW parameters
        self._update_niw_parameters(s_appearance)

        return self.z_identity

    def _update_identity_assignments(self, s_appearance: np.ndarray):
        """Update posterior over identity assignments."""
        K, _ = s_appearance.shape
        log_resp = np.zeros((K, self.V_active))

        for k in range(K):
            for v in range(self.V_active):
                # Log prior
                log_prior = np.log(self.theta_imm_pi[v] + 1e-8)

                # Log likelihood
                try:
                    log_lik = multivariate_normal.logpdf(
                        s_appearance[k], self.theta_imm_mu[v], self.theta_imm_Sigma[v]
                    )
                    log_resp[k, v] = log_prior + log_lik
                except np.linalg.LinAlgError:
                    log_resp[k, v] = -np.inf

        # Normalize responsibilities
        for k in range(K):
            log_sum = np.logaddexp.reduce(log_resp[k])
            if not np.isfinite(log_sum):
                # Uniform assignment if all likelihoods are -inf
                self.z_identity[k] = 1.0 / self.V_active
            else:
                self.z_identity[k, : self.V_active] = np.exp(log_resp[k] - log_sum)

    def _update_niw_parameters(self, s_appearance: np.ndarray):
        """Update Normal-Inverse-Wishart parameters."""

        for v in range(self.V_active):
            # Sufficient statistics
            assignments = self.z_identity[:, v]
            self.N_v[v] = np.sum(assignments)

            if self.N_v[v] < 1e-6:
                continue

            # Weighted mean
            self.x_bar_v[v] = np.average(s_appearance, axis=0, weights=assignments)

            # Weighted scatter matrix
            centered = s_appearance - self.x_bar_v[v]
            self.S_v[v] = np.sum(
                assignments[:, np.newaxis, np.newaxis]
                * centered[:, :, np.newaxis]
                @ centered[:, np.newaxis, :],
                axis=0,
            )

            # Update NIW hyperparameters
            kappa_new = self.kappa_identity[v] + self.N_v[v]
            m_new = (
                self.kappa_identity[v] * self.m_identity[v]
                + self.N_v[v] * self.x_bar_v[v]
            ) / kappa_new

            nu_new = self.nu_identity[v] + self.N_v[v]

            U_new = (
                self.U_identity[v]
                + self.S_v[v]
                + (self.kappa_identity[v] * self.N_v[v] / kappa_new)
                * np.outer(
                    self.x_bar_v[v] - self.m_identity[v],
                    self.x_bar_v[v] - self.m_identity[v],
                )
            )

            # Sample new parameters from posterior NIW
            try:
                Sigma_new = invwishart.rvs(df=nu_new, scale=U_new)
                mu_new = np.random.multivariate_normal(m_new, Sigma_new / kappa_new)

                self.theta_imm_mu[v] = mu_new
                self.theta_imm_Sigma[v] = Sigma_new

            except (np.linalg.LinAlgError, ValueError):
                # Keep previous parameters if sampling fails
                pass

        # Update mixing weights
        self._update_mixing_weights()

    def _update_mixing_weights(self):
        """Update stick-breaking mixing weights."""
        counts = np.sum(self.z_identity, axis=0)
        total_count = np.sum(counts) + 1e-6

        # Empirical weights with Dirichlet smoothing
        self.theta_imm_pi = (counts + self.alpha_imm / self.V_active) / (
            total_count + self.alpha_imm
        )

    def expand_identities(self, new_V: int):
        """Expand the number of identity types."""
        if new_V <= self.V_identities:
            return

        old_V = self.V_identities
        self.V_identities = new_V

        # Expand parameters
        new_mu = np.random.normal(
            loc=[0.5, 0.5, 0.5, 0.1, 0.1], scale=0.1, size=(new_V - old_V, 5)
        )
        self.theta_imm_mu = np.vstack([self.theta_imm_mu, new_mu])

        new_Sigma = np.array([0.1 * np.eye(5) for _ in range(new_V - old_V)])
        self.theta_imm_Sigma = np.concatenate([self.theta_imm_Sigma, new_Sigma])

        # Expand hyperparameters
        new_m = np.tile([0.5, 0.5, 0.5, 0.1, 0.1], (new_V - old_V, 1))
        self.m_identity = np.vstack([self.m_identity, new_m])

        new_kappa = np.ones(new_V - old_V)
        self.kappa_identity = np.concatenate([self.kappa_identity, new_kappa])

        new_U = np.array([0.1 * np.eye(5) for _ in range(new_V - old_V)])
        self.U_identity = np.concatenate([self.U_identity, new_U])

        new_nu = np.full(new_V - old_V, 6.0)
        self.nu_identity = np.concatenate([self.nu_identity, new_nu])

        # Expand assignments
        new_assignments = np.zeros((self.K_slots, new_V - old_V))
        self.z_identity = np.hstack([self.z_identity, new_assignments])

        # Expand sufficient statistics
        self.N_v = np.concatenate([self.N_v, np.zeros(new_V - old_V)])
        new_x_bar = np.zeros((new_V - old_V, 5))
        self.x_bar_v = np.vstack([self.x_bar_v, new_x_bar])
        new_S = np.array([np.zeros((5, 5)) for _ in range(new_V - old_V)])
        self.S_v = np.concatenate([self.S_v, new_S])

        # Reinitialize mixing weights
        self.theta_imm_pi = self._initialize_stick_breaking_weights()

        logger.info(f"iMM expanded from {old_V} to {new_V} identity types")

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        # Mu parameters + Sigma parameters + mixing weights
        return self.V_identities * 5 + self.V_identities * 25 + self.V_identities

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state for saving/loading."""
        return {
            "K_slots": self.K_slots,
            "V_identities": self.V_identities,
            "V_active": self.V_active,
            "alpha_imm": self.alpha_imm,
            "z_identity": self.z_identity,
            "theta_imm_mu": self.theta_imm_mu,
            "theta_imm_Sigma": self.theta_imm_Sigma,
            "theta_imm_pi": self.theta_imm_pi,
            "m_identity": self.m_identity,
            "kappa_identity": self.kappa_identity,
            "U_identity": self.U_identity,
            "nu_identity": self.nu_identity,
            "N_v": self.N_v,
            "x_bar_v": self.x_bar_v,
            "S_v": self.S_v,
            "inference_count": self.inference_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state."""
        for key, value in state_dict.items():
            setattr(self, key, value)

    def get_complexity_metrics(self) -> Dict[str, float]:
        """Get model complexity metrics for structure learning."""
        return {
            "effective_identities": np.sum(self.theta_imm_pi > 0.01),
            "total_parameters": self.count_parameters(),
            "entropy": -np.sum(self.theta_imm_pi * np.log(self.theta_imm_pi + 1e-8)),
            "assignment_concentration": np.max(np.sum(self.z_identity, axis=1)),
            "mean_assignment_entropy": np.mean(
                [
                    -np.sum(self.z_identity[k] * np.log(self.z_identity[k] + 1e-8))
                    for k in range(self.K_slots)
                ]
            ),
        }
