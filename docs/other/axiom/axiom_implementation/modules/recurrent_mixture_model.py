#!/usr/bin/env python3
"""
Recurrent Mixture Model (rMM) - Interaction and control module.

Implements the rMM from the AXIOM architecture, modeling dependencies
between objects, actions, and rewards for sparse interaction modeling.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class RecurrentMixtureModel:
    """
    Recurrent Mixture Model for interaction and control.

    Models dependencies between objects, actions, and rewards using
    mixed continuous-discrete context features.
    """

    def __init__(
        self,
        K_slots: int,
        M_contexts: int,
        F_continuous: int,
        F_discrete: int,
        alpha_rmm: float = 1.0,
    ):
        """
        Initialize Recurrent Mixture Model.

        Args:
            K_slots: Number of object slots
            M_contexts: Number of context modes
            F_continuous: Number of continuous features
            F_discrete: Number of discrete features
            alpha_rmm: Stick-breaking concentration parameter
        """
        self.K_slots = K_slots
        self.M_contexts = M_contexts
        self.M_active = M_contexts
        self.F_continuous = F_continuous
        self.F_discrete = F_discrete
        self.alpha_rmm = alpha_rmm

        # Context assignments [K_slots, M_contexts]
        self.s_rmm_context = np.zeros((K_slots, M_contexts))

        # Continuous feature parameters (NIW)
        self.theta_rmm_mu = np.random.normal(0, 0.1, size=(M_contexts, F_continuous))
        self.theta_rmm_Sigma = np.array(
            [0.1 * np.eye(F_continuous) for _ in range(M_contexts)]
        )

        # Discrete feature parameters (Dirichlet)
        self.theta_rmm_alpha = np.random.gamma(
            1, 1, size=(M_contexts, F_discrete, 5)
        )  # Max 5 discrete values

        # Mixing weights
        self.theta_rmm_pi = self._initialize_stick_breaking_weights()

        # Prediction parameters
        self.theta_rmm_tmm = np.random.dirichlet(
            np.ones(20),
            size=M_contexts,  # Assume max 20 dynamics modes
        )
        self.theta_rmm_reward = np.random.normal(0, 1, size=M_contexts)

        # NIW hyperparameters
        self.m_rmm = np.zeros(F_continuous)
        self.kappa_rmm = 1.0
        self.U_rmm = 0.1 * np.eye(F_continuous)
        self.nu_rmm = F_continuous + 2

        # Performance tracking
        self.inference_count = 0

    def _initialize_stick_breaking_weights(self) -> np.ndarray:
        """Initialize stick-breaking weights for context modes."""
        betas = np.random.beta(1, self.alpha_rmm, self.M_contexts)
        weights = np.zeros(self.M_contexts)

        stick_remaining = 1.0
        for m in range(self.M_contexts - 1):
            weights[m] = betas[m] * stick_remaining
            stick_remaining *= 1 - betas[m]
        weights[-1] = stick_remaining

        return weights

    def inference(
        self,
        f_continuous: np.ndarray,
        d_discrete: np.ndarray,
        u_action: int,
        r_reward: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform context inference and make predictions.

        Args:
            f_continuous: Continuous context features [K_slots, F_continuous]
            d_discrete: Discrete context features [K_slots, F_discrete]
            u_action: Current action
            r_reward: Current reward

        Returns:
            Tuple of (context_assignments, dynamics_predictions, reward_prediction)
        """
        self.inference_count += 1

        # Update context assignments
        self._update_context_assignments(f_continuous, d_discrete)

        # Make predictions
        dynamics_predictions = self._predict_dynamics()
        reward_prediction = self._predict_reward()

        # Update parameters
        self._update_parameters(f_continuous, d_discrete, u_action, r_reward)

        return self.s_rmm_context, dynamics_predictions, reward_prediction

    def _update_context_assignments(
        self, f_continuous: np.ndarray, d_discrete: np.ndarray
    ):
        """Update posterior over context assignments."""
        K, _ = f_continuous.shape
        log_resp = np.zeros((K, self.M_active))

        for k in range(K):
            for m in range(self.M_active):
                # Log prior
                log_prior = np.log(self.theta_rmm_pi[m] + 1e-8)

                # Continuous feature likelihood
                try:
                    log_lik_cont = multivariate_normal.logpdf(
                        f_continuous[k], self.theta_rmm_mu[m], self.theta_rmm_Sigma[m]
                    )
                except np.linalg.LinAlgError:
                    log_lik_cont = -np.inf

                # Discrete feature likelihood
                log_lik_disc = 0.0
                for d in range(self.F_discrete):
                    if d_discrete[k, d] < self.theta_rmm_alpha.shape[2]:
                        disc_probs = self.theta_rmm_alpha[m, d]
                        disc_probs = disc_probs / (np.sum(disc_probs) + 1e-8)
                        log_lik_disc += np.log(disc_probs[d_discrete[k, d]] + 1e-8)

                log_resp[k, m] = log_prior + log_lik_cont + log_lik_disc

        # Normalize responsibilities
        for k in range(K):
            log_sum = np.logaddexp.reduce(log_resp[k])
            if np.isfinite(log_sum):
                self.s_rmm_context[k, : self.M_active] = np.exp(log_resp[k] - log_sum)
            else:
                # Uniform assignment if all likelihoods are -inf
                self.s_rmm_context[k, : self.M_active] = 1.0 / self.M_active

    def _predict_dynamics(self) -> np.ndarray:
        """Predict next dynamics modes for each slot."""
        K = self.K_slots
        L_max = self.theta_rmm_tmm.shape[1]
        dynamics_predictions = np.zeros((K, L_max))

        for k in range(K):
            # Weighted combination of context mode predictions
            for m in range(self.M_active):
                context_weight = self.s_rmm_context[k, m]
                dynamics_predictions[k] += context_weight * self.theta_rmm_tmm[m]

        return dynamics_predictions

    def _predict_reward(self) -> float:
        """Predict next reward based on current context."""
        reward_prediction = 0.0

        for k in range(self.K_slots):
            for m in range(self.M_active):
                context_weight = self.s_rmm_context[k, m]
                reward_prediction += context_weight * self.theta_rmm_reward[m]

        # Average over slots
        reward_prediction /= self.K_slots

        return reward_prediction

    def _update_parameters(
        self,
        f_continuous: np.ndarray,
        d_discrete: np.ndarray,
        u_action: int,
        r_reward: float,
    ):
        """Update model parameters based on current data."""

        for m in range(self.M_active):
            # Update continuous feature parameters (NIW)
            assignments = self.s_rmm_context[:, m]
            N_m = np.sum(assignments)

            if N_m > 1e-6:
                # Sufficient statistics
                x_bar = np.average(f_continuous, axis=0, weights=assignments)

                # Update NIW parameters
                kappa_new = self.kappa_rmm + N_m
                m_new = (self.kappa_rmm * self.m_rmm + N_m * x_bar) / kappa_new
                nu_new = self.nu_rmm + N_m

                # Scatter matrix
                centered = f_continuous - x_bar
                S = np.sum(
                    assignments[:, np.newaxis, np.newaxis]
                    * centered[:, :, np.newaxis]
                    @ centered[:, np.newaxis, :],
                    axis=0,
                )

                U_new = (
                    self.U_rmm
                    + S
                    + (self.kappa_rmm * N_m / kappa_new)
                    * np.outer(x_bar - self.m_rmm, x_bar - self.m_rmm)
                )

                # Sample new parameters
                try:
                    from scipy.stats import invwishart

                    Sigma_new = invwishart.rvs(df=nu_new, scale=U_new)
                    mu_new = np.random.multivariate_normal(m_new, Sigma_new / kappa_new)

                    self.theta_rmm_mu[m] = mu_new
                    self.theta_rmm_Sigma[m] = Sigma_new

                except (np.linalg.LinAlgError, ValueError):
                    # Keep previous parameters if sampling fails
                    pass

                # Update discrete feature parameters
                for d in range(self.F_discrete):
                    # Count discrete values weighted by assignments
                    counts = np.zeros(self.theta_rmm_alpha.shape[2])
                    for k in range(self.K_slots):
                        if d_discrete[k, d] < len(counts):
                            counts[d_discrete[k, d]] += assignments[k]

                    # Update Dirichlet parameters
                    self.theta_rmm_alpha[m, d] = counts + 1.0  # Add-one smoothing

                # Update reward prediction parameter
                weighted_reward = np.sum(assignments) * r_reward / (N_m + 1e-6)
                self.theta_rmm_reward[m] = (
                    0.9 * self.theta_rmm_reward[m] + 0.1 * weighted_reward
                )

        # Update mixing weights
        self._update_mixing_weights()

    def _update_mixing_weights(self):
        """Update stick-breaking mixing weights."""
        counts = np.sum(self.s_rmm_context, axis=0)
        total_count = np.sum(counts) + 1e-6

        # Empirical weights with Dirichlet smoothing
        self.theta_rmm_pi = (counts + self.alpha_rmm / self.M_active) / (
            total_count + self.alpha_rmm
        )

    def expand_contexts(self, new_M: int):
        """Expand the number of context modes."""
        if new_M <= self.M_contexts:
            return

        old_M = self.M_contexts
        self.M_contexts = new_M

        # Expand parameters
        new_mu = np.random.normal(0, 0.1, size=(new_M - old_M, self.F_continuous))
        self.theta_rmm_mu = np.vstack([self.theta_rmm_mu, new_mu])

        new_Sigma = np.array(
            [0.1 * np.eye(self.F_continuous) for _ in range(new_M - old_M)]
        )
        self.theta_rmm_Sigma = np.concatenate([self.theta_rmm_Sigma, new_Sigma])

        new_alpha = np.random.gamma(1, 1, size=(new_M - old_M, self.F_discrete, 5))
        self.theta_rmm_alpha = np.concatenate([self.theta_rmm_alpha, new_alpha])

        # Expand prediction parameters
        new_tmm = np.random.dirichlet(
            np.ones(self.theta_rmm_tmm.shape[1]), size=new_M - old_M
        )
        self.theta_rmm_tmm = np.vstack([self.theta_rmm_tmm, new_tmm])

        new_reward = np.random.normal(0, 1, size=new_M - old_M)
        self.theta_rmm_reward = np.concatenate([self.theta_rmm_reward, new_reward])

        # Expand assignments
        new_assignments = np.zeros((self.K_slots, new_M - old_M))
        self.s_rmm_context = np.hstack([self.s_rmm_context, new_assignments])

        # Reinitialize mixing weights
        self.theta_rmm_pi = self._initialize_stick_breaking_weights()

        logger.info(f"rMM expanded from {old_M} to {new_M} context modes")

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        # Mu + Sigma + alpha + tmm + reward + mixing weights
        return (
            self.M_contexts * self.F_continuous
            + self.M_contexts * self.F_continuous * self.F_continuous
            + self.M_contexts * self.F_discrete * 5
            + self.M_contexts * 20  # Assume max 20 dynamics modes
            + self.M_contexts
            + self.M_contexts
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state for saving/loading."""
        return {
            "K_slots": self.K_slots,
            "M_contexts": self.M_contexts,
            "M_active": self.M_active,
            "F_continuous": self.F_continuous,
            "F_discrete": self.F_discrete,
            "alpha_rmm": self.alpha_rmm,
            "s_rmm_context": self.s_rmm_context,
            "theta_rmm_mu": self.theta_rmm_mu,
            "theta_rmm_Sigma": self.theta_rmm_Sigma,
            "theta_rmm_alpha": self.theta_rmm_alpha,
            "theta_rmm_pi": self.theta_rmm_pi,
            "theta_rmm_tmm": self.theta_rmm_tmm,
            "theta_rmm_reward": self.theta_rmm_reward,
            "m_rmm": self.m_rmm,
            "kappa_rmm": self.kappa_rmm,
            "U_rmm": self.U_rmm,
            "nu_rmm": self.nu_rmm,
            "inference_count": self.inference_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state."""
        for key, value in state_dict.items():
            setattr(self, key, value)

    def get_complexity_metrics(self) -> Dict[str, float]:
        """Get model complexity metrics for structure learning."""
        return {
            "effective_contexts": np.sum(self.theta_rmm_pi > 0.01),
            "total_parameters": self.count_parameters(),
            "entropy": -np.sum(self.theta_rmm_pi * np.log(self.theta_rmm_pi + 1e-8)),
            "context_utilization": np.mean(np.max(self.s_rmm_context, axis=1)),
            "mean_reward_prediction": np.mean(self.theta_rmm_reward),
        }
