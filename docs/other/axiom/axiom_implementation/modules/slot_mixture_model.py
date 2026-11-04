#!/usr/bin/env python3
"""
Slot Mixture Model (sMM) - Object-centric visual perception module.

Implements the sMM from the AXIOM architecture, performing object-centric
decomposition of pixel observations into K competing object slots.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class SlotMixtureModel:
    """
    Slot Mixture Model for object-centric visual perception.

    Decomposes pixel observations into K competing object slots using
    Gaussian mixture modeling with stick-breaking priors.
    """

    def __init__(self, K_slots: int, N_pixels: int, alpha_smm: float = 1.0):
        """
        Initialize Slot Mixture Model.

        Args:
            K_slots: Number of object slots
            N_pixels: Number of pixels in observation
            alpha_smm: Stick-breaking concentration parameter
        """
        self.K_slots = K_slots
        self.K_active = K_slots  # Start with all slots active
        self.N_pixels = N_pixels
        self.alpha_smm = alpha_smm

        # Slot representations (position(2) + color(3) + shape(2))
        self.s_slot = np.random.normal(
            loc=[0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1], scale=0.1, size=(K_slots, 7)
        )

        # Mixing weights (stick-breaking)
        self.theta_smm_pi = self._initialize_stick_breaking_weights()

        # Projection matrices (fixed as per GNN spec)
        self.theta_smm_A = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],  # X position
                [0, 1, 0, 0, 0, 0, 0],  # Y position
                [0, 0, 1, 0, 0, 0, 0],  # R color
                [0, 0, 0, 1, 0, 0, 0],  # G color
                [0, 0, 0, 0, 1, 0, 0],  # B color
            ]
        )

        self.theta_smm_B = np.array(
            [
                [0, 0, 0, 0, 0, 1, 0],  # X shape extent
                [0, 0, 0, 0, 0, 0, 1],  # Y shape extent
            ]
        )

        # Color variances per slot
        self.theta_smm_sigma = np.ones((K_slots, 3)) * 0.1

        # Pixel assignment probabilities
        self.z_slot_assign = np.zeros((N_pixels, K_slots))

        # Performance tracking
        self.inference_count = 0

    def _initialize_stick_breaking_weights(self) -> np.ndarray:
        """Initialize stick-breaking weights for slots."""
        # Sample stick-breaking weights
        betas = np.random.beta(1, self.alpha_smm, self.K_slots)
        weights = np.zeros(self.K_slots)

        stick_remaining = 1.0
        for k in range(self.K_slots - 1):
            weights[k] = betas[k] * stick_remaining
            stick_remaining *= 1 - betas[k]
        weights[-1] = stick_remaining

        return weights

    def inference(self, o_pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference to assign pixels to slots and update slot features.

        Args:
            o_pixels: Pixel observations [N_pixels, 5] (RGB + XY)

        Returns:
            Tuple of (slot_assignments, slot_features)
        """
        self.inference_count += 1

        # E-step: Update pixel-to-slot assignments
        log_responsibilities = self._compute_log_responsibilities(o_pixels)
        self.z_slot_assign = np.exp(log_responsibilities)

        # M-step: Update slot parameters
        self._update_slot_parameters(o_pixels)

        return self.z_slot_assign, self.s_slot

    def _compute_log_responsibilities(self, o_pixels: np.ndarray) -> np.ndarray:
        """Compute log responsibilities for pixel-to-slot assignments."""
        N, _ = o_pixels.shape
        log_resp = np.zeros((N, self.K_active))

        for k in range(self.K_active):
            # Project slot features to pixel space
            pixel_mean = self.theta_smm_A @ self.s_slot[k]
            shape_extent = self.theta_smm_B @ self.s_slot[k]

            # Covariance matrix with shape-dependent spatial variance
            spatial_var = np.maximum(shape_extent, 0.01)  # Ensure positive
            color_var = np.maximum(self.theta_smm_sigma[k], 0.01)  # Ensure positive
            cov = np.diag(
                np.concatenate(
                    [
                        spatial_var,  # Spatial covariance from shape
                        color_var,  # Color covariance
                    ]
                )
            )

            # Compute log likelihood
            try:
                log_lik = multivariate_normal.logpdf(o_pixels, pixel_mean, cov)
                log_resp[:, k] = np.log(self.theta_smm_pi[k] + 1e-8) + log_lik
            except np.linalg.LinAlgError:
                # Handle singular covariance matrices
                log_resp[:, k] = -np.inf

        # Normalize log responsibilities
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)

        return log_resp

    def _update_slot_parameters(self, o_pixels: np.ndarray):
        """Update slot parameters based on current assignments."""

        for k in range(self.K_active):
            # Effective number of pixels assigned to this slot
            N_k = np.sum(self.z_slot_assign[:, k])

            if N_k < 1e-6:  # Avoid division by zero
                continue

            # Weighted pixel features
            weighted_pixels = (self.z_slot_assign[:, k, np.newaxis] * o_pixels).sum(
                axis=0
            )
            mean_pixel = weighted_pixels / N_k

            # Update slot position and color from mean pixel
            self.s_slot[k, 0:2] = mean_pixel[3:5]  # XY position
            self.s_slot[k, 2:5] = mean_pixel[0:3]  # RGB color

            # Update shape based on spatial spread
            spatial_variance = np.var(
                o_pixels[self.z_slot_assign[:, k] > 0.1, 3:5], axis=0
            )
            self.s_slot[k, 5:7] = np.sqrt(spatial_variance + 1e-6)

            # Update color variance
            color_residuals = o_pixels[:, 0:3] - self.s_slot[k, 2:5]
            self.theta_smm_sigma[k] = (
                np.var(color_residuals[self.z_slot_assign[:, k] > 0.1], axis=0) + 1e-6
            )

        # Update mixing weights
        self._update_mixing_weights()

    def _update_mixing_weights(self):
        """Update stick-breaking mixing weights."""
        counts = np.sum(self.z_slot_assign, axis=0)
        total_count = np.sum(counts) + 1e-6

        # Empirical weights with Dirichlet smoothing
        self.theta_smm_pi = (counts + self.alpha_smm / self.K_active) / (
            total_count + self.alpha_smm
        )

    def expand_slots(self, new_K: int):
        """Expand the number of slots."""
        if new_K <= self.K_slots:
            return

        old_K = self.K_slots
        self.K_slots = new_K

        # Expand slot features
        new_slots = np.random.normal(
            loc=[0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1], scale=0.1, size=(new_K - old_K, 7)
        )
        self.s_slot = np.vstack([self.s_slot, new_slots])

        # Expand other parameters
        new_sigma = np.ones((new_K - old_K, 3)) * 0.1
        self.theta_smm_sigma = np.vstack([self.theta_smm_sigma, new_sigma])

        # Reinitialize mixing weights
        self.theta_smm_pi = self._initialize_stick_breaking_weights()

        logger.info(f"sMM expanded from {old_K} to {new_K} slots")

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        # Slot features + mixing weights + color variances
        return self.K_slots * 7 + self.K_slots + self.K_slots * 3

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state for saving/loading."""
        return {
            "K_slots": self.K_slots,
            "K_active": self.K_active,
            "N_pixels": self.N_pixels,
            "alpha_smm": self.alpha_smm,
            "s_slot": self.s_slot,
            "theta_smm_pi": self.theta_smm_pi,
            "theta_smm_sigma": self.theta_smm_sigma,
            "z_slot_assign": self.z_slot_assign,
            "inference_count": self.inference_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state."""
        self.K_slots = state_dict["K_slots"]
        self.K_active = state_dict["K_active"]
        self.N_pixels = state_dict["N_pixels"]
        self.alpha_smm = state_dict["alpha_smm"]
        self.s_slot = state_dict["s_slot"]
        self.theta_smm_pi = state_dict["theta_smm_pi"]
        self.theta_smm_sigma = state_dict["theta_smm_sigma"]
        self.z_slot_assign = state_dict["z_slot_assign"]
        self.inference_count = state_dict["inference_count"]

    def get_complexity_metrics(self) -> Dict[str, float]:
        """Get model complexity metrics for structure learning."""
        return {
            "effective_slots": np.sum(self.theta_smm_pi > 0.01),
            "total_parameters": self.count_parameters(),
            "entropy": -np.sum(self.theta_smm_pi * np.log(self.theta_smm_pi + 1e-8)),
            "max_weight": np.max(self.theta_smm_pi),
            "min_weight": np.min(self.theta_smm_pi),
        }
