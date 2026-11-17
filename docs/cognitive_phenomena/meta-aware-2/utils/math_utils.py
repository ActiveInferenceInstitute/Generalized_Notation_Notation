#!/usr/bin/env python3
"""
Mathematical Utilities for Meta-Awareness Active Inference Model

Generic, dimensionally-flexible mathematical operations for hierarchical active inference.
Supports arbitrary state space dimensions while maintaining numerical stability.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MathUtils:
    """
    Mathematical utility functions for active inference computations.

    All functions are designed to work with arbitrary state space dimensions
    and maintain numerical stability across different model configurations.
    """

    @staticmethod
    def softmax(
        X: Union[np.ndarray, list], axis: int = 0, temperature: float = 1.0
    ) -> np.ndarray:
        """
        Compute softmax with numerical stability and temperature scaling.

        Args:
            X: Input array (log probabilities)
            axis: Axis along which to compute softmax
            temperature: Temperature parameter for scaling

        Returns:
            Normalized probability distribution
        """
        X = np.asarray(X, dtype=float)

        # Temperature scaling
        if temperature != 1.0:
            X = X / temperature

        # Numerical stability: subtract max
        X_shifted = X - np.max(X, axis=axis, keepdims=True)

        # Compute exponentials with small epsilon
        exp_X = np.exp(X_shifted) + 1e-16

        # Normalize
        return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

    @staticmethod
    def log_softmax(X: Union[np.ndarray, list], axis: int = 0) -> np.ndarray:
        """
        Compute log softmax with numerical stability.

        Args:
            X: Input array
            axis: Axis along which to compute log softmax

        Returns:
            Log normalized probabilities
        """
        X = np.asarray(X, dtype=float)
        X_shifted = X - np.max(X, axis=axis, keepdims=True)
        return X_shifted - np.log(np.sum(np.exp(X_shifted), axis=axis, keepdims=True))

    @staticmethod
    def normalize(X: np.ndarray, axis: int = 0, epsilon: float = 1e-16) -> np.ndarray:
        """
        Normalize array along specified axis with numerical stability.

        Args:
            X: Input array
            axis: Axis along which to normalize
            epsilon: Small value to prevent division by zero

        Returns:
            Normalized array
        """
        X = np.asarray(X, dtype=float)
        X_positive = np.maximum(X, epsilon)
        return X_positive / np.sum(X_positive, axis=axis, keepdims=True)

    @staticmethod
    def precision_weighted_likelihood(
        A: np.ndarray, gamma: float, epsilon: float = 1e-16
    ) -> np.ndarray:
        """
        Apply precision weighting to likelihood matrix.

        Args:
            A: Likelihood matrix of arbitrary dimensions
            gamma: Precision parameter (inverse variance)
            epsilon: Numerical stability parameter

        Returns:
            Precision-weighted likelihood matrix
        """
        log_A = np.log(np.maximum(A, epsilon))
        weighted_log_A = log_A * gamma
        return MathUtils.softmax(weighted_log_A, axis=0)

    @staticmethod
    def bayesian_model_average(
        values: np.ndarray, weights: np.ndarray, A_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Bayesian model average with arbitrary dimensions.

        Args:
            values: Array of values to average
            weights: Belief weights for averaging
            A_matrix: Optional mapping matrix for hierarchical averaging

        Returns:
            Model-averaged value
        """
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if A_matrix is not None:
            # Hierarchical averaging through mapping matrix
            mapped_weights = np.dot(A_matrix.T, weights)
            return np.sum(values * mapped_weights)
        else:
            # Direct averaging
            return np.sum(values * weights)

    @staticmethod
    def compute_entropy(
        p: np.ndarray, axis: int = 0, epsilon: float = 1e-16
    ) -> np.ndarray:
        """
        Compute Shannon entropy with numerical stability.

        Args:
            p: Probability distribution
            axis: Axis along which to compute entropy
            epsilon: Small value to prevent log(0)

        Returns:
            Entropy values
        """
        p = np.asarray(p, dtype=float)
        p_safe = np.maximum(p, epsilon)
        return -np.sum(p_safe * np.log(p_safe), axis=axis)

    @staticmethod
    def compute_kl_divergence(
        p: np.ndarray, q: np.ndarray, axis: int = 0, epsilon: float = 1e-16
    ) -> np.ndarray:
        """
        Compute KL divergence D(p||q) with numerical stability.

        Args:
            p: First probability distribution
            q: Second probability distribution
            axis: Axis along which to compute KL divergence
            epsilon: Small value to prevent log(0)

        Returns:
            KL divergence values
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        p_safe = np.maximum(p, epsilon)
        q_safe = np.maximum(q, epsilon)

        return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=axis)

    @staticmethod
    def compute_attentional_charge(
        O_bar: np.ndarray,
        A_bar: np.ndarray,
        X_bar: np.ndarray,
        A_orig: np.ndarray,
        epsilon: float = 1e-16,
    ) -> float:
        """
        Compute attentional charge for precision updating (generic dimensions).

        Based on Parr & Friston "Uncertainty, epistemics and active inference".

        Args:
            O_bar: Observation posterior beliefs
            A_bar: Precision-weighted likelihood matrix
            X_bar: State posterior beliefs
            A_orig: Original likelihood matrix
            epsilon: Numerical stability parameter

        Returns:
            Attentional charge value
        """
        O_bar = np.asarray(O_bar, dtype=float)
        A_bar = np.asarray(A_bar, dtype=float)
        X_bar = np.asarray(X_bar, dtype=float)
        A_orig = np.asarray(A_orig, dtype=float)

        charge = 0.0
        n_obs, n_states = A_orig.shape

        for i in range(n_obs):
            for j in range(n_states):
                log_term = np.log(np.maximum(A_orig[i, j], epsilon))
                charge += (O_bar[i] - A_bar[i, j]) * X_bar[j] * log_term

        return charge

    @staticmethod
    def expected_free_energy(
        O_pred: np.ndarray,
        C: np.ndarray,
        X_pred: np.ndarray,
        H: np.ndarray,
        epsilon: float = 1e-16,
    ) -> float:
        """
        Compute expected free energy for policy evaluation (generic dimensions).

        G = E[o*(ln(o) - C) - x*H]

        Args:
            O_pred: Predicted observations under policy
            C: Prior preferences over observations
            X_pred: Predicted states under policy
            H: Entropy terms for each state
            epsilon: Numerical stability parameter

        Returns:
            Expected free energy value
        """
        O_pred = np.asarray(O_pred, dtype=float)
        C = np.asarray(C, dtype=float)
        X_pred = np.asarray(X_pred, dtype=float)
        H = np.asarray(H, dtype=float)

        # Epistemic term: expected surprise about observations
        O_safe = np.maximum(O_pred, epsilon)
        epistemic_term = np.sum(O_safe * (np.log(O_safe) - C))

        # Pragmatic term: expected information gain
        pragmatic_term = -np.sum(X_pred * H)

        return epistemic_term + pragmatic_term

    @staticmethod
    def variational_free_energy(
        X_bar: np.ndarray,
        X_pred: np.ndarray,
        A: np.ndarray,
        obs_idx: int,
        epsilon: float = 1e-16,
    ) -> float:
        """
        Compute variational free energy for belief updating (generic dimensions).

        Args:
            X_bar: Posterior state beliefs
            X_pred: Predicted state beliefs
            A: Likelihood matrix
            obs_idx: Index of observed outcome
            epsilon: Numerical stability parameter

        Returns:
            Variational free energy value
        """
        X_bar = np.asarray(X_bar, dtype=float)
        X_pred = np.asarray(X_pred, dtype=float)
        A = np.asarray(A, dtype=float)

        # Ensure valid observation index
        if obs_idx >= A.shape[0] or obs_idx < 0:
            logger.warning(
                f"Invalid observation index {obs_idx} for matrix shape {A.shape}"
            )
            return 0.0

        X_bar_safe = np.maximum(X_bar, epsilon)
        A_safe = np.maximum(A[obs_idx, :], epsilon)
        X_pred_safe = np.maximum(X_pred, epsilon)

        return np.sum(
            X_bar_safe * (np.log(X_bar_safe) - np.log(A_safe) - np.log(X_pred_safe))
        )

    @staticmethod
    def policy_posterior(
        log_prior: np.ndarray,
        expected_free_energy: np.ndarray,
        variational_free_energy: Optional[np.ndarray] = None,
        gamma_G: float = 1.0,
    ) -> np.ndarray:
        """
        Compute posterior beliefs over policies (generic number of policies).

        Args:
            log_prior: Log prior beliefs over policies
            expected_free_energy: Expected free energy for each policy
            variational_free_energy: Variational free energy terms (optional)
            gamma_G: Precision parameter for policy selection

        Returns:
            Posterior policy distribution
        """
        log_prior = np.asarray(log_prior, dtype=float)
        expected_free_energy = np.asarray(expected_free_energy, dtype=float)

        log_posterior = log_prior - gamma_G * expected_free_energy

        if variational_free_energy is not None:
            variational_free_energy = np.asarray(variational_free_energy, dtype=float)
            log_posterior = log_posterior - variational_free_energy

        return MathUtils.softmax(log_posterior)

    @staticmethod
    def discrete_choice(
        probabilities: np.ndarray, method: str = "deterministic"
    ) -> int:
        """
        Make discrete choice based on probability distribution.

        Args:
            probabilities: Probability distribution over choices
            method: 'deterministic' (argmax) or 'stochastic' (sampling)

        Returns:
            Selected choice index
        """
        probabilities = np.asarray(probabilities, dtype=float)

        if method == "deterministic":
            return np.argmax(probabilities)
        elif method == "stochastic":
            return np.random.choice(len(probabilities), p=probabilities)
        else:
            raise ValueError(f"Unknown choice method: {method}")

    @staticmethod
    def update_precision_beliefs(
        beta_prior: float, charge: float, bounds: Tuple[float, float]
    ) -> float:
        """
        Update inverse precision beliefs based on prediction error.

        Args:
            beta_prior: Prior inverse precision
            charge: Computed charge (prediction error signal)
            bounds: (min_beta, max_beta) bounds for stability

        Returns:
            Updated inverse precision
        """
        min_beta, max_beta = bounds

        # Clamp charge to prevent numerical instability
        if charge > min_beta:
            charge = min_beta - 1e-5

        beta_posterior = beta_prior - charge

        # Ensure bounds
        beta_posterior = np.clip(beta_posterior, min_beta, max_beta)

        return beta_posterior

    @staticmethod
    def validate_probability_matrix(
        matrix: np.ndarray,
        matrix_type: str = "unknown",
        axis: int = 0,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Validate that a matrix represents valid probabilities.

        Args:
            matrix: Matrix to validate
            matrix_type: Type of matrix for logging
            axis: Axis along which probabilities should sum to 1
            tolerance: Numerical tolerance for validation

        Returns:
            True if valid, False otherwise
        """
        matrix = np.asarray(matrix, dtype=float)

        # Check non-negativity
        if np.any(matrix < 0):
            logger.warning(f"{matrix_type} matrix contains negative values")
            return False

        # Check normalization
        sums = np.sum(matrix, axis=axis)
        if not np.allclose(sums, 1.0, atol=tolerance):
            logger.warning(
                f"{matrix_type} matrix not properly normalized along axis {axis}: {sums}"
            )
            return False

        return True

    @staticmethod
    def check_matrix_dimensions(
        matrix: np.ndarray,
        expected_shape: Tuple[int, ...],
        matrix_name: str = "unknown",
    ) -> bool:
        """
        Check that matrix has expected dimensions.

        Args:
            matrix: Matrix to check
            expected_shape: Expected shape tuple
            matrix_name: Name for logging

        Returns:
            True if dimensions match, False otherwise
        """
        if matrix.shape != expected_shape:
            logger.warning(
                f"{matrix_name} has shape {matrix.shape}, expected {expected_shape}"
            )
            return False
        return True

    @staticmethod
    def create_identity_matrix(dim: int) -> np.ndarray:
        """Create identity matrix of specified dimension."""
        return np.eye(dim, dtype=float)

    @staticmethod
    def create_uniform_distribution(dim: int) -> np.ndarray:
        """Create uniform probability distribution of specified dimension."""
        return np.ones(dim, dtype=float) / dim

    @staticmethod
    def create_transition_matrix(dim: int, persistence: float = 0.8) -> np.ndarray:
        """
        Create a generic transition matrix with specified persistence.

        Args:
            dim: Dimension of state space
            persistence: Probability of staying in same state

        Returns:
            Transition matrix with high diagonal values
        """
        if not 0 <= persistence <= 1:
            raise ValueError("Persistence must be between 0 and 1")

        matrix = np.eye(dim) * persistence
        off_diagonal = (1 - persistence) / (dim - 1) if dim > 1 else 0
        matrix = matrix + off_diagonal * (1 - np.eye(dim))

        return matrix

    @staticmethod
    def create_likelihood_matrix(
        obs_dim: int, state_dim: int, accuracy: float = 0.8
    ) -> np.ndarray:
        """
        Create a generic likelihood matrix with specified accuracy.

        Args:
            obs_dim: Number of possible observations
            state_dim: Number of hidden states
            accuracy: Probability of correct observation given state

        Returns:
            Likelihood matrix A[obs, state]
        """
        if not 0 <= accuracy <= 1:
            raise ValueError("Accuracy must be between 0 and 1")

        if obs_dim != state_dim:
            # For mismatched dimensions, create random normalized matrix
            matrix = np.random.dirichlet(np.ones(obs_dim), size=state_dim).T
        else:
            # For matched dimensions, create accurate diagonal mapping
            matrix = np.eye(obs_dim) * accuracy
            off_diagonal = (1 - accuracy) / (obs_dim - 1) if obs_dim > 1 else 0
            matrix = matrix + off_diagonal * (1 - np.eye(obs_dim))

        return matrix

    @staticmethod
    def generate_oddball_sequence(
        T: int, oddball_times: List[int] = None
    ) -> np.ndarray:
        """
        Generate oddball stimulus sequence matching original implementation.

        Args:
            T: Total time steps
            oddball_times: List of time points for oddball stimuli

        Returns:
            Binary sequence (0=standard, 1=oddball/deviant)
        """
        sequence = np.zeros(T, dtype=int)

        if oddball_times is None:
            # Default pattern: at 1/5, 2/5, 3/5, 4/5 of trial
            oddball_times = [int(T / 5), int(2 * T / 5), int(3 * T / 5), int(4 * T / 5)]

        for t in oddball_times:
            if 0 <= t < T:
                sequence[t] = 1

        return sequence

    @staticmethod
    def setup_transition_matrices() -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Set up standard transition matrices matching original implementation.

        Returns:
            Tuple of (B_perception, B_attention_stay, B_attention_switch, B_meta)
        """
        # Perception transitions (2x2) - persistent
        B_perception = np.array([[0.9, 0.1], [0.1, 0.9]])

        # Attention "stay" policy (2x2) - high persistence
        B_attention_stay = np.array([[0.9, 0.1], [0.1, 0.9]])

        # Attention "switch" policy (2x2) - promotes switching
        B_attention_switch = np.array([[0.3, 0.7], [0.7, 0.3]])

        # Meta-awareness transitions (2x2) - very persistent
        B_meta = np.array([[0.95, 0.05], [0.05, 0.95]])

        return B_perception, B_attention_stay, B_attention_switch, B_meta

    @staticmethod
    def setup_likelihood_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Set up standard likelihood matrices matching original implementation.

        Returns:
            Tuple of (A_perception, A_attention, A_meta)
        """
        # Perception likelihood (2x2) - moderate accuracy
        A_perception = np.array([[0.8, 0.2], [0.2, 0.8]])

        # Attention likelihood (2x2) - good accuracy
        A_attention = np.array([[0.85, 0.15], [0.15, 0.85]])

        # Meta-awareness likelihood (2x2) - high accuracy
        A_meta = np.array([[0.9, 0.1], [0.1, 0.9]])

        return A_perception, A_attention, A_meta

    @staticmethod
    def compute_entropy_terms(A: np.ndarray) -> np.ndarray:
        """
        Compute entropy terms for each column of likelihood matrix.

        Args:
            A: Likelihood matrix

        Returns:
            Entropy values for each state
        """
        entropies = np.zeros(A.shape[1])
        for j in range(A.shape[1]):
            entropies[j] = MathUtils.compute_entropy(A[:, j])
        return entropies

    @staticmethod
    def softmax_dim2(X: np.ndarray) -> np.ndarray:
        """
        Convert matrix of log probabilities to matrix of normalized probabilities.
        Normalizes along axis 0 (columns) to match original implementation.

        Args:
            X: Matrix of log probabilities

        Returns:
            Matrix of normalized probabilities
        """
        return MathUtils.softmax(X, axis=0)

    @staticmethod
    def normalise(X: np.ndarray) -> np.ndarray:
        """
        Normalize a matrix of probabilities along columns (axis 0).
        Alternative spelling to match original implementation.

        Args:
            X: Matrix of probabilities

        Returns:
            Column-normalized matrix
        """
        return MathUtils.normalize(X, axis=0)


# Convenience functions for common operations
def softmax(x, axis=0, temperature=1.0):
    """Convenience wrapper for softmax."""
    return MathUtils.softmax(x, axis, temperature)


def normalize(x, axis=0):
    """Convenience wrapper for normalization."""
    return MathUtils.normalize(x, axis)


def entropy(p, axis=0):
    """Convenience wrapper for entropy computation."""
    return MathUtils.compute_entropy(p, axis)


def kl_div(p, q, axis=0):
    """Convenience wrapper for KL divergence."""
    return MathUtils.compute_kl_divergence(p, q, axis)


def softmax_dim2(X):
    """Convenience wrapper for matrix softmax (matches original naming)."""
    return MathUtils.softmax_dim2(X)


def normalise(X):
    """Convenience wrapper for normalization (matches original naming)."""
    return MathUtils.normalise(X)


def precision_weighted_likelihood(A, gamma):
    """Convenience wrapper for precision-weighted likelihood."""
    return MathUtils.precision_weighted_likelihood(A, gamma)


def bayesian_model_average(values, weights, A_matrix=None):
    """Convenience wrapper for Bayesian model averaging."""
    return MathUtils.bayesian_model_average(values, weights, A_matrix)


def compute_attentional_charge(O_bar, A_bar, X_bar, A_orig):
    """Convenience wrapper for attentional charge computation."""
    return MathUtils.compute_attentional_charge(O_bar, A_bar, X_bar, A_orig)


def expected_free_energy(O_pred, C, X_pred, H):
    """Convenience wrapper for expected free energy."""
    return MathUtils.expected_free_energy(O_pred, C, X_pred, H)


def variational_free_energy(X_bar, X_pred, A, obs_idx):
    """Convenience wrapper for variational free energy."""
    return MathUtils.variational_free_energy(X_bar, X_pred, A, obs_idx)


def update_precision_beliefs(beta_prior, charge, bounds):
    """Convenience wrapper for precision belief updates."""
    return MathUtils.update_precision_beliefs(beta_prior, charge, bounds)


def policy_posterior(
    log_prior, expected_free_energy, variational_free_energy=None, gamma_G=1.0
):
    """Convenience wrapper for policy posterior computation."""
    return MathUtils.policy_posterior(
        log_prior, expected_free_energy, variational_free_energy, gamma_G
    )


def discrete_choice(probabilities, method="deterministic"):
    """Convenience wrapper for discrete choice."""
    return MathUtils.discrete_choice(probabilities, method)


def generate_oddball_sequence(T, oddball_times=None):
    """Convenience wrapper for oddball sequence generation."""
    return MathUtils.generate_oddball_sequence(T, oddball_times)


def setup_transition_matrices():
    """Convenience wrapper for transition matrix setup."""
    return MathUtils.setup_transition_matrices()


def setup_likelihood_matrices():
    """Convenience wrapper for likelihood matrix setup."""
    return MathUtils.setup_likelihood_matrices()


def compute_entropy_terms(A):
    """Convenience wrapper for entropy terms computation."""
    return MathUtils.compute_entropy_terms(A)


# Test functions
def test_math_utils():
    """Test suite for mathematical utilities."""
    print("Testing mathematical utilities...")

    # Test softmax
    x = np.array([1, 2, 3])
    probs = softmax(x)
    assert np.isclose(np.sum(probs), 1.0), "Softmax should sum to 1"
    print("✓ Softmax test passed")

    # Test normalization
    y = np.array([2, 4, 6])
    norm_y = normalize(y)
    assert np.isclose(np.sum(norm_y), 1.0), "Normalization should sum to 1"
    print("✓ Normalization test passed")

    # Test entropy
    uniform = np.ones(4) / 4
    ent = entropy(uniform)
    expected_ent = np.log(4)
    assert np.isclose(ent, expected_ent), f"Uniform entropy should be {expected_ent}"
    print("✓ Entropy test passed")

    # Test matrix creation
    trans_mat = MathUtils.create_transition_matrix(3, 0.8)
    assert trans_mat.shape == (3, 3), "Transition matrix should be 3x3"
    assert MathUtils.validate_probability_matrix(trans_mat, axis=1), (
        "Transition matrix should be valid"
    )
    print("✓ Matrix creation test passed")

    print("All mathematical utility tests passed!")


if __name__ == "__main__":
    test_math_utils()
