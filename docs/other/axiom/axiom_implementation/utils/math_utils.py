"""
Mathematical Utilities for AXIOM Implementation
==============================================

Implements core mathematical functions for Bayesian mixture models,
variational inference, and Active Inference calculations following
GNN specifications.

Authors: AXIOM Research Team
Institution: VERSES AI / Active Inference Institute
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.special as special
import scipy.stats as stats
from numpy.typing import NDArray

# Type aliases for clarity
Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]
BinaryMatrix = NDArray[np.bool_]


class BayesianUtils:
    """Utilities for Bayesian inference and mixture models."""

    @staticmethod
    def log_categorical(x: Vector, theta: Vector) -> float:
        """Log probability of categorical distribution."""
        return np.sum(x * np.log(theta + 1e-10))

    @staticmethod
    def log_multivariate_normal(x: Vector, mu: Vector, sigma: Matrix) -> float:
        """Log probability of multivariate normal distribution."""
        try:
            diff = x - mu
            sigma_inv = np.linalg.inv(sigma + 1e-6 * np.eye(len(sigma)))
            log_det = np.linalg.slogdet(sigma)[1]
            k = len(x)

            log_prob = -0.5 * (
                k * np.log(2 * np.pi) + log_det + diff.T @ sigma_inv @ diff
            )
            return log_prob
        except np.linalg.LinAlgError:
            return -np.inf

    @staticmethod
    def log_normal_inverse_wishart(
        mu: Vector, sigma: Matrix, m: Vector, kappa: float, nu: float, psi: Matrix
    ) -> float:
        """Log probability of Normal-Inverse-Wishart distribution."""

        len(mu)

        # NIW log probability
        log_prob = 0.0

        # Wishart component for precision matrix
        sigma_inv = np.linalg.inv(sigma)
        log_prob += stats.wishart.logpdf(sigma_inv, df=nu, scale=np.linalg.inv(psi))

        # Normal component for mean given precision
        mean_sigma = sigma / kappa
        log_prob += BayesianUtils.log_multivariate_normal(mu, m, mean_sigma)

        return log_prob

    @staticmethod
    def stick_breaking_weights(alpha: float, K: int) -> Vector:
        """Generate stick-breaking weights for Dirichlet process."""

        betas = np.random.beta(1, alpha, K - 1)
        weights = np.zeros(K)

        weights[0] = betas[0]
        for k in range(1, K - 1):
            weights[k] = betas[k] * np.prod(1 - betas[:k])
        weights[-1] = np.prod(1 - betas)

        return weights

    @staticmethod
    def dirichlet_expectation(alpha: Vector) -> Vector:
        """Compute expectation of Dirichlet distribution."""
        return alpha / np.sum(alpha)

    @staticmethod
    def dirichlet_entropy(alpha: Vector) -> float:
        """Compute entropy of Dirichlet distribution."""
        alpha_sum = np.sum(alpha)
        entropy = (
            special.gammaln(alpha_sum)
            - np.sum(special.gammaln(alpha))
            + np.sum(
                (alpha - 1) * (special.digamma(alpha) - special.digamma(alpha_sum))
            )
        )
        return entropy


class VariationalInference:
    """Coordinate ascent variational inference for mixture models."""

    @staticmethod
    def update_assignment_probabilities(
        log_likelihoods: Matrix, mixing_weights: Vector
    ) -> Matrix:
        """Update assignment probabilities (E-step)."""

        len(mixing_weights)
        log_likelihoods.shape[0]

        # Add log mixing weights
        log_responsibilities = log_likelihoods + np.log(mixing_weights + 1e-10)

        # Normalize (log-sum-exp trick)
        max_log_resp = np.max(log_responsibilities, axis=1, keepdims=True)
        log_responsibilities -= max_log_resp
        responsibilities = np.exp(log_responsibilities)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        return responsibilities

    @staticmethod
    def update_mixing_weights(responsibilities: Matrix, alpha: float) -> Vector:
        """Update mixing weights with Dirichlet prior."""

        N_k = np.sum(responsibilities, axis=0)
        weights = (N_k + alpha - 1) / (np.sum(N_k) + len(N_k) * alpha - len(N_k))

        return weights

    @staticmethod
    def update_niw_parameters(
        data: Matrix,
        responsibilities: Vector,
        m_prior: Vector,
        kappa_prior: float,
        nu_prior: float,
        psi_prior: Matrix,
    ) -> Tuple[Vector, float, float, Matrix]:
        """Update Normal-Inverse-Wishart parameters."""

        N = np.sum(responsibilities)

        if N < 1e-6:
            return m_prior, kappa_prior, nu_prior, psi_prior

        # Weighted sample mean
        x_bar = np.sum(responsibilities[:, np.newaxis] * data, axis=0) / N

        # Updated parameters
        kappa_new = kappa_prior + N
        m_new = (kappa_prior * m_prior + N * x_bar) / kappa_new
        nu_new = nu_prior + N

        # Weighted sample covariance
        diff_data = data - x_bar
        S = np.sum(
            responsibilities[:, np.newaxis, np.newaxis]
            * diff_data[:, :, np.newaxis]
            * diff_data[:, np.newaxis, :],
            axis=0,
        )

        diff_mean = x_bar - m_prior
        psi_new = (
            psi_prior
            + S
            + (kappa_prior * N / kappa_new) * np.outer(diff_mean, diff_mean)
        )

        return m_new, kappa_new, nu_new, psi_new

    @staticmethod
    def compute_variational_lower_bound(
        data: Matrix,
        responsibilities: Matrix,
        log_likelihoods: Matrix,
        mixing_weights: Vector,
    ) -> float:
        """Compute variational lower bound (ELBO)."""

        # Expected log likelihood
        expected_log_likelihood = np.sum(responsibilities * log_likelihoods)

        # Entropy of assignments
        entropy_assignments = -np.sum(
            responsibilities * np.log(responsibilities + 1e-10)
        )

        # Prior on mixing weights (simplified)
        log_prior_weights = np.sum(np.log(mixing_weights + 1e-10))

        elbo = expected_log_likelihood + entropy_assignments + log_prior_weights

        return elbo


class LinearDynamics:
    """Utilities for linear dynamical systems and transitions."""

    @staticmethod
    def fit_linear_dynamics(
        states_t: Matrix, states_t1: Matrix, responsibilities: Vector
    ) -> Tuple[Matrix, Vector, Matrix]:
        """Fit linear dynamics s_{t+1} = D s_t + b + noise."""

        if np.sum(responsibilities) < 1e-6:
            d = states_t.shape[1]
            return np.eye(d), np.zeros(d), 0.01 * np.eye(d)

        # Weighted least squares
        W = np.diag(responsibilities)
        X = np.column_stack([states_t, np.ones(len(states_t))])  # Add bias column

        try:
            # Solve weighted least squares: (X^T W X)^{-1} X^T W Y
            XTW = X.T @ W
            XTWX_inv = np.linalg.inv(XTW @ X + 1e-6 * np.eye(X.shape[1]))
            params = XTWX_inv @ XTW @ states_t1

            # Extract dynamics matrix and bias
            D = params[:-1, :].T  # Dynamics matrix
            b = params[-1, :]  # Bias vector

            # Compute noise covariance
            predictions = X @ params
            residuals = states_t1 - predictions
            weighted_residuals = residuals * np.sqrt(responsibilities)[:, np.newaxis]
            noise_cov = (weighted_residuals.T @ weighted_residuals) / np.sum(
                responsibilities
            )

            # Regularize noise covariance
            noise_cov += 1e-3 * np.eye(noise_cov.shape[0])

        except np.linalg.LinAlgError:
            # Fallback to identity dynamics
            d = states_t.shape[1]
            D = np.eye(d)
            b = np.zeros(d)
            noise_cov = 0.01 * np.eye(d)

        return D, b, noise_cov

    @staticmethod
    def predict_linear_dynamics(state: Vector, D: Matrix, b: Vector) -> Vector:
        """Predict next state using linear dynamics."""
        return D @ state + b


class ActiveInferenceUtils:
    """Utilities for Active Inference planning and free energy computation."""

    @staticmethod
    def expected_free_energy(
        predicted_obs: Vector,
        predicted_reward: float,
        model_uncertainty: float,
        gamma: float = 1.0,
    ) -> float:
        """Compute expected free energy for Active Inference planning."""

        # Pragmatic value (expected utility)
        pragmatic_value = predicted_reward

        # Epistemic value (information gain)
        epistemic_value = gamma * model_uncertainty

        # Expected free energy (negative because we minimize)
        G = -pragmatic_value - epistemic_value

        return G

    @staticmethod
    def softmax_policy(Q_values: Vector, precision: float) -> Vector:
        """Convert Q-values to policy using softmax with precision parameter."""

        # Apply precision scaling
        scaled_Q = precision * Q_values

        # Softmax with numerical stability
        max_Q = np.max(scaled_Q)
        exp_Q = np.exp(scaled_Q - max_Q)
        policy = exp_Q / np.sum(exp_Q)

        return policy

    @staticmethod
    def kl_divergence(p: Vector, q: Vector) -> float:
        """Compute KL divergence D_KL(p || q)."""

        # Add small epsilon for numerical stability
        p_safe = p + 1e-10
        q_safe = q + 1e-10

        kl = np.sum(p_safe * np.log(p_safe / q_safe))

        return kl

    @staticmethod
    def entropy(p: Vector) -> float:
        """Compute entropy H(p)."""

        p_safe = p + 1e-10
        entropy = -np.sum(p_safe * np.log(p_safe))

        return entropy


class StructureLearningUtils:
    """Utilities for online structure learning and Bayesian Model Reduction."""

    @staticmethod
    def posterior_predictive_likelihood(
        data_new: Vector, sufficient_stats: Dict, prior_params: Dict
    ) -> float:
        """Compute posterior predictive likelihood for new data."""

        # Extract sufficient statistics
        n = sufficient_stats.get("n", 0)
        mean = sufficient_stats.get("mean", prior_params["m"])
        cov = sufficient_stats.get("cov", prior_params["psi"])

        # Compute predictive distribution parameters
        kappa_n = prior_params["kappa"] + n
        nu_n = prior_params["nu"] + n

        # Student-t parameters for predictive distribution
        predictive_mean = mean
        predictive_scale = ((kappa_n + 1) / (kappa_n * (nu_n - len(mean) + 1))) * cov
        predictive_df = nu_n - len(mean) + 1

        # Log probability under multivariate t-distribution
        try:
            log_prob = stats.multivariate_t.logpdf(
                data_new, loc=predictive_mean, shape=predictive_scale, df=predictive_df
            )
        except Exception:
            log_prob = -np.inf

        return log_prob

    @staticmethod
    def expansion_criterion(
        log_likelihoods: Vector, threshold: float, concentration: float
    ) -> bool:
        """Check if new component should be added."""

        max_log_likelihood = np.max(log_likelihoods)
        criterion_value = threshold + np.log(concentration)

        return max_log_likelihood < criterion_value

    @staticmethod
    def bmr_merge_score(
        component1_stats: Dict,
        component2_stats: Dict,
        merged_stats: Dict,
        prior_params: Dict,
    ) -> float:
        """Compute Bayesian Model Reduction merge score."""

        # Compute free energies for separate and merged components
        F_separate = StructureLearningUtils._component_free_energy(
            component1_stats, prior_params
        ) + StructureLearningUtils._component_free_energy(
            component2_stats, prior_params
        )

        F_merged = StructureLearningUtils._component_free_energy(
            merged_stats, prior_params
        )

        # Merge score (positive = beneficial merge)
        merge_score = F_separate - F_merged

        return merge_score

    @staticmethod
    def _component_free_energy(stats: Dict, prior_params: Dict) -> float:
        """Compute variational free energy for a component."""

        # Simplified free energy computation
        stats.get("n", 0)
        log_likelihood = stats.get("log_likelihood", 0)

        # Prior contribution (simplified)
        log_prior = 0.0  # Would depend on specific prior

        # Entropy contribution (simplified)
        entropy = 0.0  # Would depend on variational distribution

        free_energy = -log_likelihood - log_prior + entropy

        return free_energy


class NumericalUtils:
    """Numerical utilities for stable computation."""

    @staticmethod
    def log_sum_exp(x: Vector, axis: Optional[int] = None) -> Union[float, Vector]:
        """Numerically stable log-sum-exp."""

        x_max = np.max(x, axis=axis, keepdims=True)
        return x_max.squeeze() + np.log(np.sum(np.exp(x - x_max), axis=axis))

    @staticmethod
    def normalize_log_probabilities(log_probs: Vector) -> Vector:
        """Normalize log probabilities to sum to 1."""

        max_log_prob = np.max(log_probs)
        normalized = np.exp(log_probs - max_log_prob)
        normalized /= np.sum(normalized)

        return normalized

    @staticmethod
    def safe_cholesky(matrix: Matrix) -> Matrix:
        """Compute Cholesky decomposition with regularization."""

        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            # Add regularization and try again
            regularized = matrix + 1e-6 * np.eye(matrix.shape[0])
            return np.linalg.cholesky(regularized)

    @staticmethod
    def ensure_positive_definite(
        matrix: Matrix, min_eigenvalue: float = 1e-6
    ) -> Matrix:
        """Ensure matrix is positive definite by eigenvalue thresholding."""

        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, min_eigenvalue)

        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T


# Example usage and testing functions
def test_mathematical_utilities():
    """Test mathematical utilities with example data."""

    print("Testing AXIOM Mathematical Utilities...")

    # Test Bayesian utilities
    print("Testing Bayesian utilities...")
    x = np.array([1, 2, 3])
    mu = np.array([1.1, 2.1, 2.9])
    sigma = np.eye(3) * 0.1

    log_prob = BayesianUtils.log_multivariate_normal(x, mu, sigma)
    print(f"Log MVN probability: {log_prob:.4f}")

    # Test variational inference
    print("Testing variational inference...")
    np.random.seed(42)
    np.random.randn(100, 3)
    log_likelihoods = np.random.randn(100, 5)
    mixing_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])

    responsibilities = VariationalInference.update_assignment_probabilities(
        log_likelihoods, mixing_weights
    )
    print(f"Responsibilities shape: {responsibilities.shape}")
    print(f"Responsibilities sum: {np.sum(responsibilities, axis=1)[:5]}")

    # Test linear dynamics
    print("Testing linear dynamics...")
    states_t = np.random.randn(50, 4)
    states_t1 = states_t + 0.1 * np.random.randn(50, 4)  # Small dynamics
    responsibilities = np.ones(50)

    D, b, noise_cov = LinearDynamics.fit_linear_dynamics(
        states_t, states_t1, responsibilities
    )
    print(f"Dynamics matrix shape: {D.shape}")
    print(f"Bias vector shape: {b.shape}")

    # Test Active Inference utilities
    print("Testing Active Inference utilities...")
    Q_values = np.array([1.0, 2.0, 0.5, -1.0, 3.0])
    policy = ActiveInferenceUtils.softmax_policy(Q_values, precision=2.0)
    print(f"Policy: {policy}")
    print(f"Policy sum: {np.sum(policy):.4f}")

    print("All tests completed successfully!")


if __name__ == "__main__":
    test_mathematical_utilities()

# Module-level exports for backward compatibility
dirichlet_entropy = BayesianUtils.dirichlet_entropy
dirichlet_expectation = BayesianUtils.dirichlet_expectation
entropy = ActiveInferenceUtils.entropy
expected_free_energy = ActiveInferenceUtils.expected_free_energy
kl_divergence = ActiveInferenceUtils.kl_divergence
log_categorical = BayesianUtils.log_categorical
log_multivariate_normal = BayesianUtils.log_multivariate_normal
log_sum_exp = NumericalUtils.log_sum_exp
normalize_log_probabilities = NumericalUtils.normalize_log_probabilities
softmax_policy = ActiveInferenceUtils.softmax_policy
stick_breaking_weights = BayesianUtils.stick_breaking_weights
ensure_positive_definite = NumericalUtils.ensure_positive_definite
fit_linear_dynamics = LinearDynamics.fit_linear_dynamics
compute_variational_lower_bound = VariationalInference.compute_variational_lower_bound
