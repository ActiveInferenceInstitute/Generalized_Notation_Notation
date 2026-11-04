#!/usr/bin/env python3
"""
Transition Mixture Model (tMM) - Object dynamics module.

Implements the tMM from the AXIOM architecture, modeling object dynamics
as piecewise linear trajectories with switching linear dynamical systems.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TransitionMixtureModel:
    """
    Transition Mixture Model for object dynamics.

    Models each slot's evolution as piecewise linear trajectories using
    a switching linear dynamical system (SLDS) approach.
    """

    def __init__(self, K_slots: int, L_dynamics: int, alpha_tmm: float = 1.0):
        """
        Initialize Transition Mixture Model.

        Args:
            K_slots: Number of object slots
            L_dynamics: Number of dynamics modes
            alpha_tmm: Stick-breaking concentration parameter
        """
        self.K_slots = K_slots
        self.L_dynamics = L_dynamics
        self.L_active = L_dynamics
        self.alpha_tmm = alpha_tmm

        # Dynamics mode assignments [K_slots, L_dynamics]
        self.s_tmm_mode = np.zeros((K_slots, L_dynamics))

        # Linear dynamics parameters
        self.theta_tmm_D = np.array(
            [np.eye(7) + 0.1 * np.random.randn(7, 7) for _ in range(L_dynamics)]
        )

        self.theta_tmm_b = np.random.normal(0, 0.1, size=(L_dynamics, 7))

        self.theta_tmm_Sigma = np.array([0.01 * np.eye(7) for _ in range(L_dynamics)])

        # Mixing weights
        self.theta_tmm_pi = self._initialize_stick_breaking_weights()

        # Stickiness parameters
        self.z_tmm_sticky = np.random.binomial(1, 0.8, L_dynamics).astype(bool)
        self.gamma_tmm_stick = np.random.gamma(2.0, 1.0, L_dynamics)

        # Previous mode assignments for stickiness
        self.s_tmm_mode_prev = np.zeros((K_slots, L_dynamics))

        # Performance tracking
        self.inference_count = 0

        # Sufficient statistics for learning
        self.regression_data = [[] for _ in range(L_dynamics)]

    def _initialize_stick_breaking_weights(self) -> np.ndarray:
        """Initialize stick-breaking weights for dynamics modes."""
        betas = np.random.beta(1, self.alpha_tmm, self.L_dynamics)
        weights = np.zeros(self.L_dynamics)

        stick_remaining = 1.0
        for l in range(self.L_dynamics - 1):
            weights[l] = betas[l] * stick_remaining
            stick_remaining *= 1 - betas[l]
        weights[-1] = stick_remaining

        return weights

    def inference(
        self, s_slot: np.ndarray, dynamics_predictions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Perform dynamics inference and update slot states.

        Args:
            s_slot: Current slot states [K_slots, 7]
            dynamics_predictions: Optional predictions from rMM [K_slots, L_dynamics]

        Returns:
            Updated slot states [K_slots, 7]
        """
        self.inference_count += 1

        # Update mode assignments
        self._update_mode_assignments(s_slot, dynamics_predictions)

        # Apply dynamics to update slot states
        s_slot_new = self._apply_dynamics(s_slot)

        # Update dynamics parameters
        self._update_dynamics_parameters(s_slot, s_slot_new)

        # Store previous assignments for stickiness
        self.s_tmm_mode_prev = self.s_tmm_mode.copy()

        return s_slot_new

    def _update_mode_assignments(
        self, s_slot: np.ndarray, dynamics_predictions: Optional[np.ndarray] = None
    ):
        """Update posterior over dynamics mode assignments."""
        K, _ = s_slot.shape
        log_resp = np.zeros((K, self.L_active))

        for k in range(K):
            for l in range(self.L_active):
                # Base prior probability
                if dynamics_predictions is not None:
                    # Use rMM predictions if available
                    log_prior = np.log(dynamics_predictions[k, l] + 1e-8)
                else:
                    log_prior = np.log(self.theta_tmm_pi[l] + 1e-8)

                # Add stickiness bonus
                if self.z_tmm_sticky[l]:
                    sticky_bonus = self.gamma_tmm_stick[l] * self.s_tmm_mode_prev[k, l]
                    log_prior += sticky_bonus

                log_resp[k, l] = log_prior

        # Normalize responsibilities
        for k in range(K):
            log_sum = np.logaddexp.reduce(log_resp[k])
            if np.isfinite(log_sum):
                self.s_tmm_mode[k, : self.L_active] = np.exp(log_resp[k] - log_sum)
            else:
                # Uniform assignment if all priors are -inf
                self.s_tmm_mode[k, : self.L_active] = 1.0 / self.L_active

    def _apply_dynamics(self, s_slot: np.ndarray) -> np.ndarray:
        """Apply linear dynamics to update slot states."""
        K, _ = s_slot.shape
        s_slot_new = np.zeros_like(s_slot)

        for k in range(K):
            # Weighted combination of dynamics modes
            slot_update = np.zeros(7)

            for l in range(self.L_active):
                # Linear dynamics: s' = D * s + b + noise
                dynamics_update = self.theta_tmm_D[l] @ s_slot[k] + self.theta_tmm_b[l]

                # Add process noise
                noise = np.random.multivariate_normal(
                    np.zeros(7), self.theta_tmm_Sigma[l]
                )
                dynamics_update += noise

                # Weight by mode assignment
                slot_update += self.s_tmm_mode[k, l] * dynamics_update

            s_slot_new[k] = slot_update

            # Apply physical constraints
            s_slot_new[k] = self._apply_constraints(s_slot_new[k])

        return s_slot_new

    def _apply_constraints(self, slot_state: np.ndarray) -> np.ndarray:
        """Apply physical constraints to slot states."""
        # Position constraints (stay within [0, 1] x [0, 1])
        slot_state[0] = np.clip(slot_state[0], 0.0, 1.0)
        slot_state[1] = np.clip(slot_state[1], 0.0, 1.0)

        # Color constraints (stay within [0, 1]^3)
        slot_state[2:5] = np.clip(slot_state[2:5], 0.0, 1.0)

        # Shape constraints (positive size)
        slot_state[5:7] = np.clip(slot_state[5:7], 0.01, 0.5)

        return slot_state

    def _update_dynamics_parameters(self, s_slot: np.ndarray, s_slot_new: np.ndarray):
        """Update dynamics parameters using weighted linear regression."""

        # Collect regression data for each mode
        for l in range(self.L_active):
            mode_data = []
            weights = []

            for k in range(self.K_slots):
                assignment_weight = self.s_tmm_mode[k, l]
                if assignment_weight > 0.01:  # Only include significant assignments
                    # Input: current state, Output: next state
                    X = np.concatenate([s_slot[k], [1.0]])  # Add bias term
                    y = s_slot_new[k]

                    mode_data.append((X, y))
                    weights.append(assignment_weight)

            # Update parameters if we have sufficient data
            if len(mode_data) >= 3:  # Need at least 3 data points
                self._update_single_mode_parameters(l, mode_data, weights)

        # Update mixing weights
        self._update_mixing_weights()

    def _update_single_mode_parameters(self, mode_idx: int, data: list, weights: list):
        """Update parameters for a single dynamics mode."""
        try:
            # Prepare weighted regression
            X_list, y_list = zip(*data)
            X = np.array(X_list)  # [N, 8] (7 state + 1 bias)
            y = np.array(y_list)  # [N, 7]
            w = np.array(weights)

            # Weighted least squares: (X^T W X)^{-1} X^T W y
            W = np.diag(w)
            XTW = X.T @ W
            XTWX = XTW @ X
            XTWy = XTW @ y

            # Solve for parameters
            if np.linalg.det(XTWX) > 1e-6:
                params = np.linalg.solve(XTWX, XTWy)

                # Update D matrix and bias vector
                self.theta_tmm_D[mode_idx] = params[:-1].T  # [7, 7]
                self.theta_tmm_b[mode_idx] = params[-1]  # [7]

                # Update noise covariance
                residuals = y - X @ params
                weighted_residuals = w[:, np.newaxis] * residuals
                self.theta_tmm_Sigma[mode_idx] = np.cov(
                    weighted_residuals.T
                ) + 1e-6 * np.eye(7)

        except (np.linalg.LinAlgError, ValueError):
            # Keep previous parameters if update fails
            pass

    def _update_mixing_weights(self):
        """Update stick-breaking mixing weights."""
        counts = np.sum(self.s_tmm_mode, axis=0)
        total_count = np.sum(counts) + 1e-6

        # Empirical weights with Dirichlet smoothing
        self.theta_tmm_pi = (counts + self.alpha_tmm / self.L_active) / (
            total_count + self.alpha_tmm
        )

    def expand_dynamics(self, new_L: int):
        """Expand the number of dynamics modes."""
        if new_L <= self.L_dynamics:
            return

        old_L = self.L_dynamics
        self.L_dynamics = new_L

        # Expand parameters
        new_D = np.array(
            [np.eye(7) + 0.1 * np.random.randn(7, 7) for _ in range(new_L - old_L)]
        )
        self.theta_tmm_D = np.concatenate([self.theta_tmm_D, new_D])

        new_b = np.random.normal(0, 0.1, size=(new_L - old_L, 7))
        self.theta_tmm_b = np.vstack([self.theta_tmm_b, new_b])

        new_Sigma = np.array([0.01 * np.eye(7) for _ in range(new_L - old_L)])
        self.theta_tmm_Sigma = np.concatenate([self.theta_tmm_Sigma, new_Sigma])

        # Expand stickiness parameters
        new_sticky = np.random.binomial(1, 0.8, new_L - old_L).astype(bool)
        self.z_tmm_sticky = np.concatenate([self.z_tmm_sticky, new_sticky])

        new_gamma = np.random.gamma(2.0, 1.0, new_L - old_L)
        self.gamma_tmm_stick = np.concatenate([self.gamma_tmm_stick, new_gamma])

        # Expand assignments
        new_assignments = np.zeros((self.K_slots, new_L - old_L))
        self.s_tmm_mode = np.hstack([self.s_tmm_mode, new_assignments])
        self.s_tmm_mode_prev = np.hstack([self.s_tmm_mode_prev, new_assignments])

        # Expand regression data
        self.regression_data.extend([[] for _ in range(new_L - old_L)])

        # Reinitialize mixing weights
        self.theta_tmm_pi = self._initialize_stick_breaking_weights()

        logger.info(f"tMM expanded from {old_L} to {new_L} dynamics modes")

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        # D matrices + b vectors + Sigma matrices + mixing weights + stickiness params
        return (
            self.L_dynamics * 49
            + self.L_dynamics * 7
            + self.L_dynamics * 49
            + self.L_dynamics
            + self.L_dynamics * 2
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state for saving/loading."""
        return {
            "K_slots": self.K_slots,
            "L_dynamics": self.L_dynamics,
            "L_active": self.L_active,
            "alpha_tmm": self.alpha_tmm,
            "s_tmm_mode": self.s_tmm_mode,
            "theta_tmm_D": self.theta_tmm_D,
            "theta_tmm_b": self.theta_tmm_b,
            "theta_tmm_Sigma": self.theta_tmm_Sigma,
            "theta_tmm_pi": self.theta_tmm_pi,
            "z_tmm_sticky": self.z_tmm_sticky,
            "gamma_tmm_stick": self.gamma_tmm_stick,
            "s_tmm_mode_prev": self.s_tmm_mode_prev,
            "regression_data": self.regression_data,
            "inference_count": self.inference_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state."""
        for key, value in state_dict.items():
            setattr(self, key, value)

    def get_complexity_metrics(self) -> Dict[str, float]:
        """Get model complexity metrics for structure learning."""
        return {
            "effective_dynamics": np.sum(self.theta_tmm_pi > 0.01),
            "total_parameters": self.count_parameters(),
            "entropy": -np.sum(self.theta_tmm_pi * np.log(self.theta_tmm_pi + 1e-8)),
            "stickiness_ratio": np.mean(self.z_tmm_sticky),
            "mean_stickiness": np.mean(self.gamma_tmm_stick),
            "mode_utilization": np.mean(np.max(self.s_tmm_mode, axis=1)),
        }
