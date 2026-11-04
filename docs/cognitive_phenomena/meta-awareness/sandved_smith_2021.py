#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sandved-Smith et al. (2021) Computational Phenomenology of Mental Action

Implementation of the hierarchical active inference model for meta-awareness
and attentional control from:

"Towards a computational phenomenology of mental action: modelling meta-awareness
and attentional control with deep parametric active inference"

Neuroscience of Consciousness, 2021(1), niab018
https://doi.org/10.1093/nc/niab018

This module implements both the two-level (attention) and three-level (meta-awareness)
models described in the paper, with exact replication of the computational methods.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

warnings.filterwarnings("ignore")

# Import our utility functions
import pathlib as _p
import sys

_here = _p.Path(__file__).parent
sys.path.insert(0, str(_here))
from utils import (
    bayesian_model_average,
    compute_attentional_charge,
    compute_entropy_terms,
    discrete_choice,
    expected_free_energy,
    generate_oddball_sequence,
    policy_posterior,
    precision_weighted_likelihood,
    setup_likelihood_matrices,
    setup_transition_matrices,
    softmax,
    variational_free_energy,
)
from visualizations import (
    display_results_summary,
    save_all_figures,
)


class SandvedSmithModel:
    """
    Implementation of the Sandved-Smith et al. (2021) computational phenomenology model.

    This class implements hierarchical active inference with precision control for
    modeling meta-awareness and attentional control during mind-wandering.
    """

    def __init__(
        self, T: int = 100, three_level: bool = False, random_seed: Optional[int] = None
    ):
        """
        Initialize the model.

        Args:
            T: Number of time steps
            three_level: If True, use three-level model with meta-awareness
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.T = T
        self.three_level = three_level
        self.random_seed = random_seed

        # Initialize all state and parameter arrays
        self._setup_parameters()
        self._setup_transition_matrices()
        self._setup_likelihood_matrices()
        self._initialize_states()

    def _setup_parameters(self):
        """Set up model parameters and hyperparameters."""
        # Policy parameters
        self.E2 = np.array(
            [0.99, 0.99]
        )  # Prior over attentional policies (stay, switch)
        self.gamma_G2 = (
            4.0 if not self.three_level else 2.0
        )  # Policy selection precision
        self.C2 = np.array([2, -2])  # Preferences over attentional outcomes

        # Precision parameters
        self.beta_A1m = np.array([0.5, 2.0])  # Inverse precision bounds for level 1

        if self.three_level:
            self.beta_A2m = np.array([0.5, 2.0])  # Inverse precision bounds for level 2

        # Stimulus sequence
        self.O = generate_oddball_sequence(self.T)

    def _setup_transition_matrices(self):
        """Set up transition matrices for all levels."""
        mats = setup_transition_matrices()
        if mats is None:
            # Fallback defaults if util returns None
            B1 = np.array([[0.8, 0.2], [0.2, 0.8]])
            B2a = np.array([[0.8, 0.0], [0.2, 1.0]])
            B2b = np.array([[0.0, 1.0], [1.0, 0.0]])
            B3 = np.array([[0.9, 0.1], [0.1, 0.9]])
            mats = (B1, B2a, B2b, B3)
        self.B1, self.B2a, self.B2b, self.B3 = mats

        # Combined transition matrix tensor
        self.B2t = np.zeros((2, 2, 2))
        self.B2t[:, :, 0] = self.B2a  # Stay policy
        self.B2t[:, :, 1] = self.B2b  # Switch policy

    def _setup_likelihood_matrices(self):
        """Set up likelihood matrices for all levels."""
        mats = setup_likelihood_matrices()
        if mats is None:
            A1 = np.array([[0.75, 0.25], [0.25, 0.75]])
            A2 = np.array([[0.65, 0.35], [0.35, 0.65]])
            A3 = np.array([[0.9, 0.1], [0.1, 0.9]])
            mats = (A1, A2, A3)
        self.A1, self.A2, self.A3 = mats

        # Compute entropy terms for expected free energy
        self.H2 = compute_entropy_terms(self.A2)

        # Initialize precision-weighted likelihood
        self.gamma_A2_fixed = 1.0
        self.A2bar_fixed = precision_weighted_likelihood(self.A2, self.gamma_A2_fixed)

    def _initialize_states(self):
        """Initialize state arrays and beliefs."""
        # Policy arrays
        self.Pi2 = np.zeros((2, self.T))  # Prior attentional actions
        self.Pi2_bar = np.zeros((2, self.T))  # Posterior attentional actions

        # State arrays - Level 1 (Perception)
        self.X1 = np.zeros((2, self.T))  # Perceptual state prior
        self.X1_bar = np.zeros((2, self.T))  # Perceptual state posterior

        # State arrays - Level 2 (Attention)
        self.X2 = np.zeros((2, self.T))  # Attentional state prior
        self.X2_bar = np.zeros((2, self.T))  # Attentional state posterior
        self.x2 = np.zeros(self.T, dtype=int)  # True attentional states
        self.u2 = np.zeros(self.T, dtype=int)  # True attentional actions

        # State arrays - Level 3 (Meta-awareness, if three_level=True)
        if self.three_level:
            self.X3 = np.zeros((2, self.T))  # Meta-awareness state prior
            self.X3_bar = np.zeros((2, self.T))  # Meta-awareness state posterior
            self.x3 = np.zeros(self.T, dtype=int)  # True meta-awareness states
            self.gamma_A2 = np.zeros(self.T)  # Attentional precision

        # Observation arrays
        self.O1 = np.zeros((2, self.T))  # Observation prior
        self.O1_bar = np.zeros((2, self.T))  # Observation posterior

        if self.three_level:
            self.O2_bar = np.zeros((2, self.T))  # Level 2 observation posterior

        # Precision arrays
        self.gamma_A1 = np.zeros(self.T)  # Perceptual precision

        # Free energy arrays
        self.G2 = np.zeros((2, self.T))  # Expected free energy
        self.F2 = np.zeros((2, self.T))  # Variational free energy

        # Initial state beliefs
        if self.three_level:
            self.X3[:, 0] = [1.0, 0.0]  # Start in high meta-awareness
            self.x3[0] = 0

        self.X2[:, 0] = [0.5, 0.5]  # Uncertain attentional state
        self.X1[:, 0] = [0.5, 0.5]  # Uncertain perceptual state
        self.x2[0] = 0  # Start focused

        # Set up observation posterior
        for t in range(self.T):
            self.O1_bar[int(self.O[t]), t] = 1

    def run_simulation(self, figure_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete simulation.

        Args:
            figure_mode: 'fig7', 'fig10', 'fig11', or None for normal simulation

        Returns:
            Dictionary containing all simulation results
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._initialize_states()
        print(f"Running {'3-level' if self.three_level else '2-level'} simulation...")
        print(f"Time steps: {self.T}")

        if figure_mode:
            print(f"Figure mode: {figure_mode}")

        for t in range(self.T):
            self._update_beliefs(t, figure_mode)

            if t < self.T - 1:
                self._policy_selection(t)
                self._update_generative_process(t, figure_mode)

        self._compute_final_free_energies()

        print("Simulation completed!")

        return self._collect_results()

    def _update_beliefs(self, t: int, figure_mode: Optional[str] = None):
        """Update beliefs at time step t."""
        if self.three_level:
            self._update_three_level_beliefs(t, figure_mode)
        else:
            self._update_two_level_beliefs(t, figure_mode)

    def _update_two_level_beliefs(self, t: int, figure_mode: Optional[str] = None):
        """Update beliefs for two-level model."""
        # Compute precision using Bayesian model average
        try:
            beta_A1 = bayesian_model_average(
                self.beta_A1m, self.X2[:, t], self.A2bar_fixed
            )
        except TypeError:
            beta_A1 = bayesian_model_average(self.beta_A1m, self.X2[:, t])

        # True precision based on generative process
        self.gamma_A1[t] = self.beta_A1m[int(self.x2[t])] ** -1

        # Precision-weighted likelihood mapping
        A1_bar = precision_weighted_likelihood(self.A1, self.gamma_A1[t])
        if A1_bar is None:
            A1_bar = self.A1.astype(float)
        else:
            A1_bar = np.asarray(A1_bar, dtype=float)

        # Observation prior (matrix-vector product)
        self.O1[:, t] = np.dot(A1_bar, self.X1[:, t])

        # Perceptual state posterior
        log_likelihood = self.gamma_A1[t] * np.log(self.A1[int(self.O[t]), :])
        self.X1_bar[:, t] = softmax(np.log(self.X1[:, t]) + log_likelihood)

        # Compute attentional charge
        AtC = compute_attentional_charge(
            self.O1_bar[:, t], A1_bar, self.X1_bar[:, t], self.A1
        )

        # Clamp charge for numerical stability
        if AtC > self.beta_A1m[0]:
            AtC = self.beta_A1m[0] - 1e-5

        beta_A1_bar = beta_A1 - AtC

        # Attentional state posterior
        precision_ratio = (self.beta_A1m - AtC) / self.beta_A1m * beta_A1 / beta_A1_bar
        precision_ratio = np.maximum(precision_ratio, 1e-16)
        log_precision_evidence = -1.0 * np.log(precision_ratio)
        self.X2_bar[:, t] = softmax(np.log(self.X2[:, t]) + log_precision_evidence)

    def _update_three_level_beliefs(self, t: int, figure_mode: Optional[str] = None):
        """Update beliefs for three-level model."""
        # Meta-awareness level (Level 3)
        beta_A2 = bayesian_model_average(self.beta_A2m, self.X3[:, t], self.A3)
        self.gamma_A2[t] = self.beta_A2m[int(self.x3[t])] ** -1
        A2_bar = precision_weighted_likelihood(self.A2, self.gamma_A2[t])
        if A2_bar is None:
            A2_bar = self.A2.astype(float)
        else:
            A2_bar = np.asarray(A2_bar, dtype=float)

        # Set true lower-level attentional states as observations
        self.O2_bar[int(self.x2[t]), t] = 1

        # Attentional level (Level 2)
        beta_A1 = bayesian_model_average(self.beta_A1m, self.X2[:, t], A2_bar)
        self.gamma_A1[t] = self.beta_A1m[int(self.x2[t])] ** -1
        A1_bar = precision_weighted_likelihood(self.A1, self.gamma_A1[t])
        if A1_bar is None:
            A1_bar = self.A1.astype(float)
        else:
            A1_bar = np.asarray(A1_bar, dtype=float)

        # Perceptual level (Level 1)
        self.O1[:, t] = np.dot(A1_bar, self.X1[:, t])
        log_likelihood = self.gamma_A1[t] * np.log(self.A1[int(self.O[t]), :])
        self.X1_bar[:, t] = softmax(np.log(self.X1[:, t]) + log_likelihood)

        # Compute attentional charge (Level 1 -> Level 2)
        AtC = compute_attentional_charge(
            self.O1_bar[:, t], A1_bar, self.X1_bar[:, t], self.A1
        )

        if AtC > self.beta_A1m[0]:
            AtC = self.beta_A1m[0] - 1e-5

        beta_A1_bar = beta_A1 - AtC

        # Attentional state posterior with both ascending evidence and direct observation
        precision_ratio = (self.beta_A1m - AtC) / self.beta_A1m * beta_A1 / beta_A1_bar
        log_precision_evidence = -1.0 * np.log(precision_ratio)
        direct_observation = self.gamma_A2[t] * np.log(self.A2[int(self.x2[t]), :])

        self.X2_bar[:, t] = softmax(
            np.log(self.X2[:, t]) + direct_observation + log_precision_evidence
        )

        # Compute meta-awareness charge (Level 2 -> Level 3)
        MaC = compute_attentional_charge(
            self.O2_bar[:, t], A2_bar, self.X2_bar[:, t], self.A2
        )

        if MaC > self.beta_A2m[0]:
            MaC = self.beta_A2m[0] - 1e-5

        beta_A2_bar = beta_A2 - MaC

        # Meta-awareness state posterior
        precision_ratio = (self.beta_A2m - MaC) / self.beta_A2m * beta_A2 / beta_A2_bar
        log_precision_evidence = -1.0 * np.log(precision_ratio)
        direct_observation = 0.1 * np.log(
            np.clip(self.A3[int(self.x3[t]), :], 1e-16, 1.0)
        )  # Weaker direct observation

        self.X3_bar[:, t] = softmax(
            np.log(np.clip(self.X3[:, t], 1e-16, 1.0))
            + direct_observation
            + log_precision_evidence
        )

    def _policy_selection(self, t: int):
        """Perform policy selection for attentional control."""
        # Predict states under each policy
        X2a = np.inner(self.B2a, self.X2_bar[:, t])  # Stay policy
        X2b = np.inner(self.B2b, self.X2_bar[:, t])  # Switch policy

        # Predict observations under each policy
        A2_current = (
            self.A2bar_fixed
            if not self.three_level
            else precision_weighted_likelihood(self.A2, self.gamma_A2[t])
        )
        if A2_current is None:
            A2_current = self.A2.astype(float)

        O2a = np.dot(A2_current, X2a)
        O2b = np.dot(A2_current, X2b)

        # Compute expected free energy for each policy
        self.G2[0, t] = expected_free_energy(O2a, self.C2, X2a, self.H2)
        self.G2[1, t] = expected_free_energy(O2b, self.C2, X2b, self.H2)

        # Policy posterior
        self.Pi2[:, t] = policy_posterior(
            np.log(self.E2), self.G2[:, t], gamma_G=self.gamma_G2
        )

    def _update_generative_process(self, t: int, figure_mode: Optional[str] = None):
        """Update the generative process (true states and transitions)."""
        # Evolve expected states
        B2 = self.B2a * self.Pi2[0, t] + self.B2b * self.Pi2[1, t]

        if self.three_level:
            self.X3[:, t + 1] = np.inner(self.B3, self.X3_bar[:, t])

        self.X2[:, t + 1] = np.inner(B2, self.X2_bar[:, t])
        self.X1[:, t + 1] = np.inner(self.B1, self.X1_bar[:, t])

        # Deterministic action selection for reproducibility
        self.u2[t] = int(np.argmax(self.Pi2[:, t]))

        # Update true states based on figure mode or policy
        if figure_mode == "fig7":
            # Figure 7: Fixed attentional state schedule
            self.x2[t + 1] = 0 if (t + 1) < self.T // 2 else 1
        else:
            # Normal simulation: transition according to policy
            if self.u2[t] == 0:  # Stay policy
                self.x2[t + 1] = discrete_choice(self.B2a[:, int(self.x2[t])])
            else:  # Switch policy
                self.x2[t + 1] = discrete_choice(self.B2b[:, int(self.x2[t])])

        # Update meta-awareness states (if three-level)
        if self.three_level:
            if figure_mode == "fig11":
                # Figure 11: Fixed meta-awareness schedule
                self.x3[t + 1] = 0 if (t + 1) < self.T // 2 else 1
            else:
                # Normal simulation: stochastic transitions
                self.x3[t + 1] = discrete_choice(self.B3[:, int(self.x3[t])])

    def _compute_final_free_energies(self):
        """Compute final variational free energies for policy posteriors."""
        for t in range(1, self.T):
            if t < self.T - 1:
                # Predict states under each policy
                X2a = np.inner(self.B2a, self.X2_bar[:, t])
                X2b = np.inner(self.B2b, self.X2_bar[:, t])

                # Compute charges for precision updating
                A1_bar = precision_weighted_likelihood(self.A1, self.gamma_A1[t])
                AtC = compute_attentional_charge(
                    self.O1_bar[:, t], A1_bar, self.X1_bar[:, t], self.A1
                )

                if AtC > self.beta_A1m[0]:
                    AtC = self.beta_A1m[0] - 1e-5

                try:
                    beta_A1 = bayesian_model_average(
                        self.beta_A1m,
                        self.X2[:, t],
                        self.A2bar_fixed
                        if not self.three_level
                        else precision_weighted_likelihood(self.A2, self.gamma_A2[t]),
                    )
                except TypeError:
                    beta_A1 = bayesian_model_average(self.beta_A1m, self.X2[:, t])
                beta_A1_bar = beta_A1 - AtC

                # Update state predictions with precision evidence
                precision_ratio = (
                    (self.beta_A1m - AtC) / self.beta_A1m * beta_A1 / beta_A1_bar
                )
                precision_ratio = np.maximum(precision_ratio, 1e-16)
                log_precision_evidence = -1.0 * np.log(precision_ratio)

                X2a_bar = softmax(
                    np.log(np.maximum(X2a, 1e-16)) + log_precision_evidence
                )
                X2b_bar = softmax(
                    np.log(np.maximum(X2b, 1e-16)) + log_precision_evidence
                )

                # Compute variational free energy
                self.F2[0, t - 1] = variational_free_energy(
                    X2a_bar, X2a, self.A2, int(self.x2[t])
                )
                self.F2[1, t - 1] = variational_free_energy(
                    X2b_bar, X2b, self.A2, int(self.x2[t])
                )

                # Final policy posterior
                self.Pi2_bar[:, t - 1] = policy_posterior(
                    np.log(self.E2), self.G2[:, t - 1], self.F2[:, t - 1], self.gamma_G2
                )

    def _collect_results(self) -> Dict[str, Any]:
        """Collect all simulation results into a dictionary."""
        results = {
            "T": self.T,
            "three_level": self.three_level,
            # Observations and stimuli
            "O": self.O,
            "O1_bar": self.O1_bar,
            # Level 1: Perceptual states
            "X1": self.X1,
            "X1_bar": self.X1_bar,
            # Level 2: Attentional states and policies
            "X2": self.X2,
            "X2_bar": self.X2_bar,
            "x2": self.x2,
            "u2": self.u2,
            "Pi2": self.Pi2,
            "Pi2_bar": self.Pi2_bar,
            # Precision
            "gamma_A1": self.gamma_A1,
            # Free energies
            "G2": self.G2,
            "F2": self.F2,
        }

        # Add three-level specific results
        if self.three_level:
            results.update(
                {
                    "X3": self.X3,
                    "X3_bar": self.X3_bar,
                    "x3": self.x3,
                    "gamma_A2": self.gamma_A2,
                    "O2_bar": self.O2_bar,
                }
            )

        return results


def run_figure_7_simulation() -> Dict[str, Any]:
    """Run simulation for Figure 7 (two-level with fixed attentional schedule)."""
    print("\n" + "=" * 60)
    print("FIGURE 7: Influence of attentional state on perception")
    print("=" * 60)

    model = SandvedSmithModel(T=100, three_level=False, random_seed=42)
    results = model.run_simulation(figure_mode="fig7")

    return results


def run_figure_10_simulation() -> Dict[str, Any]:
    """Run simulation for Figure 10 (two-level with attentional cycles)."""
    print("\n" + "=" * 60)
    print("FIGURE 10: Two-level model with attentional cycles")
    print("=" * 60)

    model = SandvedSmithModel(T=100, three_level=False, random_seed=42)
    results = model.run_simulation(figure_mode="fig10")

    return results


def run_figure_11_simulation() -> Dict[str, Any]:
    """Run simulation for Figure 11 (three-level with meta-awareness)."""
    print("\n" + "=" * 60)
    print("FIGURE 11: Three-level model with meta-awareness")
    print("=" * 60)

    model = SandvedSmithModel(T=100, three_level=True, random_seed=42)
    results = model.run_simulation(figure_mode="fig11")

    return results


def main():
    """Main function to run all simulations and generate figures."""
    print("SANDVED-SMITH ET AL. (2021) - COMPUTATIONAL PHENOMENOLOGY OF MENTAL ACTION")
    print("=" * 80)

    # Run all simulations
    results_fig7 = run_figure_7_simulation()
    results_fig10 = run_figure_10_simulation()
    results_fig11 = run_figure_11_simulation()

    # Generate and save all figures
    print("\nGenerating figures...")

    save_all_figures(results_fig7, "figures_fig7")
    save_all_figures(results_fig10, "figures_fig10")
    save_all_figures(results_fig11, "figures_fig11")

    # Display result summaries
    display_results_summary(results_fig7)
    display_results_summary(results_fig10)
    display_results_summary(results_fig11)

    print("\n" + "=" * 80)
    print("All simulations completed successfully!")
    print(
        "Figures saved in figures_fig7/, figures_fig10/, and figures_fig11/ directories"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
