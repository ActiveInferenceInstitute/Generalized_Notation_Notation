#!/usr/bin/env python3
"""
Active Inference Planning - Planning and control module.

Implements active inference planning using expected free energy
minimization for AXIOM agent control.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ActiveInferencePlanning:
    """
    Active Inference Planning module for AXIOM agent.

    Implements planning via expected free energy minimization with
    rollout-based policy evaluation.
    """

    def __init__(self, config, models: Dict[str, Any]):
        """
        Initialize Active Inference Planning.

        Args:
            config: AXIOM configuration
            models: Dictionary of mixture models
        """
        self.config = config
        self.models = models

        # Planning parameters
        self.H_planning = config.H_planning
        self.N_rollouts = config.N_rollouts
        self.gamma_precision = config.gamma_precision

        # Action space
        self.action_space_size = config.action_space_size

        # Policy and free energy tracking
        self.pi_policy = (
            np.ones((self.H_planning, self.action_space_size)) / self.action_space_size
        )
        self.G_expected_free_energy = np.zeros(self.H_planning)

        # Planning history
        self.planning_history = {
            "policies": [],
            "free_energies": [],
            "selected_actions": [],
            "rollout_rewards": [],
        }

        # Performance tracking
        self.planning_count = 0

    def plan(self, s_slot: np.ndarray, o_pixels: np.ndarray, r_reward: float) -> int:
        """
        Perform active inference planning to select next action.

        Args:
            s_slot: Current slot states [K_slots, 7]
            o_pixels: Current pixel observations [N_pixels, 5]
            r_reward: Current reward

        Returns:
            Selected action (discrete)
        """
        self.planning_count += 1

        # Generate policy candidates
        policy_candidates = self._generate_policy_candidates()

        # Evaluate each policy using rollouts
        policy_scores = []
        for policy in policy_candidates:
            score = self._evaluate_policy(policy, s_slot, o_pixels, r_reward)
            policy_scores.append(score)

        # Select best policy using softmax selection
        action = self._select_action(policy_candidates, policy_scores)

        # Update planning history
        self._update_planning_history(policy_candidates, policy_scores, action)

        return action

    def _generate_policy_candidates(self) -> List[np.ndarray]:
        """Generate candidate policies for evaluation."""
        candidates = []

        # Random policy candidates
        for _ in range(self.N_rollouts // 4):
            policy = np.random.dirichlet(
                np.ones(self.action_space_size), size=self.H_planning
            )
            candidates.append(policy)

        # Biased towards specific actions
        for action in range(self.action_space_size):
            policy = np.ones((self.H_planning, self.action_space_size)) * 0.1
            policy[:, action] = 0.7  # Bias towards this action
            policy = policy / policy.sum(axis=1, keepdims=True)
            candidates.append(policy)

        # Previous policy with noise
        if len(self.planning_history["policies"]) > 0:
            prev_policy = self.planning_history["policies"][-1]
            for _ in range(self.N_rollouts // 4):
                noise = np.random.dirichlet(
                    np.ones(self.action_space_size) * 0.1, self.H_planning
                )
                noisy_policy = 0.8 * prev_policy + 0.2 * noise
                noisy_policy = noisy_policy / noisy_policy.sum(axis=1, keepdims=True)
                candidates.append(noisy_policy)

        # Fill remaining slots with uniform random
        while len(candidates) < self.N_rollouts:
            policy = np.random.dirichlet(
                np.ones(self.action_space_size), size=self.H_planning
            )
            candidates.append(policy)

        return candidates

    def _evaluate_policy(
        self,
        policy: np.ndarray,
        s_slot: np.ndarray,
        o_pixels: np.ndarray,
        r_reward: float,
    ) -> float:
        """
        Evaluate a policy using forward rollouts.

        Args:
            policy: Policy to evaluate [H_planning, action_space_size]
            s_slot: Initial slot states [K_slots, 7]
            o_pixels: Initial observations [N_pixels, 5]
            r_reward: Initial reward

        Returns:
            Policy score (negative expected free energy)
        """

        total_score = 0.0
        rollout_rewards = []

        # Perform multiple rollouts
        for rollout in range(
            max(1, self.N_rollouts // len(policy) if hasattr(policy, "__len__") else 10)
        ):
            # Initialize rollout state
            s_slot_rollout = s_slot.copy()
            rollout_reward = 0.0

            # Forward simulate using the policy
            for t in range(self.H_planning):
                # Sample action from policy
                action_probs = policy[t]
                action = np.random.choice(self.action_space_size, p=action_probs)

                # Simulate one step forward
                reward_step, s_slot_rollout = self._simulate_step(
                    s_slot_rollout, action, t
                )

                rollout_reward += reward_step * (0.95**t)  # Discount factor

            rollout_rewards.append(rollout_reward)
            total_score += rollout_reward

        # Average score across rollouts
        if rollout_rewards:
            avg_score = total_score / len(rollout_rewards)
            # Add exploration bonus (entropy)
            entropy_bonus = -np.sum(policy * np.log(policy + 1e-8))
            total_score = avg_score + 0.1 * entropy_bonus
        else:
            total_score = 0.0

        return total_score

    def _simulate_step(
        self, s_slot: np.ndarray, action: int, timestep: int
    ) -> Tuple[float, np.ndarray]:
        """
        Simulate one step forward using the models.

        Args:
            s_slot: Current slot states [K_slots, 7]
            action: Action to take
            timestep: Current timestep in planning horizon

        Returns:
            Tuple of (predicted_reward, next_slot_states)
        """

        # Construct context features for rMM
        f_continuous, d_discrete = self._construct_rollout_context_features(
            s_slot, action
        )

        # Get rMM predictions
        try:
            _, dynamics_predictions, reward_prediction = self.models["rmm"].inference(
                f_continuous, d_discrete, action, 0.0
            )
        except Exception:
            # Fallback if rMM inference fails
            dynamics_predictions = (
                np.ones((s_slot.shape[0], self.models["tmm"].L_active))
                / self.models["tmm"].L_active
            )
            reward_prediction = np.random.normal(0, 0.1)

        # Apply dynamics using tMM
        try:
            s_slot_next = self.models["tmm"].inference(s_slot, dynamics_predictions)
        except Exception:
            # Fallback dynamics
            s_slot_next = s_slot + np.random.normal(0, 0.01, s_slot.shape)
            s_slot_next = np.clip(s_slot_next, 0, 1)  # Keep in bounds

        # Predict reward using simple heuristics if rMM fails
        if np.isnan(reward_prediction) or not np.isfinite(reward_prediction):
            # Simple reward based on slot positions (encourage spreading out)
            distances = []
            for i in range(s_slot.shape[0]):
                for j in range(i + 1, s_slot.shape[0]):
                    dist = np.linalg.norm(s_slot[i, :2] - s_slot[j, :2])
                    distances.append(dist)

            avg_distance = np.mean(distances) if distances else 0.5
            reward_prediction = (avg_distance - 0.3) * 0.1  # Reward for spreading

            # Add small penalty for extreme positions
            edge_penalty = 0.0
            for i in range(s_slot.shape[0]):
                pos = s_slot[i, :2]
                if np.any(pos < 0.1) or np.any(pos > 0.9):
                    edge_penalty += 0.01

            reward_prediction -= edge_penalty

        return reward_prediction, s_slot_next

    def _construct_rollout_context_features(
        self, s_slot: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct context features for rollout simulation."""
        K_slots = s_slot.shape[0]

        # Continuous features
        f_continuous = np.zeros((K_slots, self.config.F_continuous))

        for k in range(K_slots):
            # Basic features
            f_continuous[k, 0:2] = s_slot[k, 0:2]  # Position

            # Distances to other slots
            distances = []
            for j in range(K_slots):
                if j != k:
                    dist = np.linalg.norm(s_slot[k, 0:2] - s_slot[j, 0:2])
                    distances.append(dist)

            if distances:
                f_continuous[k, 2] = np.mean(distances)  # Average distance
                f_continuous[k, 3] = np.min(distances)  # Minimum distance

            # Center of mass relative position
            center_of_mass = np.mean(s_slot[:, 0:2], axis=0)
            f_continuous[k, 4:6] = s_slot[k, 0:2] - center_of_mass

            # Fill remaining features with slot properties
            if self.config.F_continuous > 6:
                remaining = min(self.config.F_continuous - 6, 4)
                f_continuous[k, 6 : 6 + remaining] = s_slot[
                    k, 2 : 2 + remaining
                ]  # Color/shape

        # Discrete features
        d_discrete = np.zeros((K_slots, self.config.F_discrete), dtype=int)

        for k in range(K_slots):
            d_discrete[k, 0] = k % 5  # Slot identity
            d_discrete[k, 1] = action  # Current action
            d_discrete[k, 2] = 1  # Neutral reward sign

            # Add more discrete features if needed
            if self.config.F_discrete > 3:
                d_discrete[k, 3] = int(s_slot[k, 0] > 0.5)  # Position quadrant

            if self.config.F_discrete > 4:
                d_discrete[k, 4] = int(s_slot[k, 1] > 0.5)  # Position quadrant

        return f_continuous, d_discrete

    def _select_action(self, policies: List[np.ndarray], scores: List[float]) -> int:
        """Select action using softmax over policy scores."""

        if not scores:
            return np.random.randint(self.action_space_size)

        # Convert scores to probabilities
        scores = np.array(scores)
        scores = scores - np.max(scores)  # Numerical stability
        exp_scores = np.exp(scores * self.gamma_precision)
        policy_probs = exp_scores / (np.sum(exp_scores) + 1e-8)

        # Select policy
        selected_policy_idx = np.random.choice(len(policies), p=policy_probs)
        selected_policy = policies[selected_policy_idx]

        # Sample action from first timestep of selected policy
        action_probs = selected_policy[0]
        action = np.random.choice(self.action_space_size, p=action_probs)

        # Store selected policy for next iteration
        self.pi_policy = selected_policy
        self.G_expected_free_energy = np.array(scores)

        return action

    def _update_planning_history(
        self, policies: List[np.ndarray], scores: List[float], action: int
    ):
        """Update planning history for analysis."""

        if policies:
            # Store best policy
            best_idx = np.argmax(scores) if scores else 0
            self.planning_history["policies"].append(policies[best_idx].copy())
            self.planning_history["free_energies"].append(
                scores[best_idx] if scores else 0.0
            )

        self.planning_history["selected_actions"].append(action)

        # Limit history length
        max_history = 1000
        for key in self.planning_history:
            if len(self.planning_history[key]) > max_history:
                self.planning_history[key] = self.planning_history[key][-max_history:]

    def get_state_dict(self) -> Dict[str, Any]:
        """Get planning state for saving/loading."""
        return {
            "pi_policy": self.pi_policy,
            "G_expected_free_energy": self.G_expected_free_energy,
            "planning_history": self.planning_history,
            "planning_count": self.planning_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load planning state."""
        self.pi_policy = state_dict["pi_policy"]
        self.G_expected_free_energy = state_dict["G_expected_free_energy"]
        self.planning_history = state_dict["planning_history"]
        self.planning_count = state_dict["planning_count"]

    def get_planning_metrics(self) -> Dict[str, float]:
        """Get planning performance metrics."""

        if not self.planning_history["free_energies"]:
            return {
                "mean_free_energy": 0.0,
                "planning_entropy": 0.0,
                "action_diversity": 0.0,
                "planning_count": self.planning_count,
            }

        # Mean free energy over recent history
        recent_fe = self.planning_history["free_energies"][-100:]
        mean_free_energy = np.mean(recent_fe)

        # Policy entropy
        if len(self.planning_history["policies"]) > 0:
            recent_policy = self.planning_history["policies"][-1]
            policy_entropy = -np.sum(recent_policy * np.log(recent_policy + 1e-8))
        else:
            policy_entropy = 0.0

        # Action diversity
        recent_actions = self.planning_history["selected_actions"][-100:]
        if recent_actions:
            action_counts = np.bincount(
                recent_actions, minlength=self.action_space_size
            )
            action_probs = action_counts / (np.sum(action_counts) + 1e-8)
            action_diversity = -np.sum(action_probs * np.log(action_probs + 1e-8))
        else:
            action_diversity = 0.0

        return {
            "mean_free_energy": mean_free_energy,
            "planning_entropy": policy_entropy,
            "action_diversity": action_diversity,
            "planning_count": self.planning_count,
        }

    def reset_planning(self):
        """Reset planning state for new episode."""
        self.pi_policy = (
            np.ones((self.H_planning, self.action_space_size)) / self.action_space_size
        )
        self.G_expected_free_energy = np.zeros(self.H_planning)

        # Clear short-term history but keep some for learning
        if len(self.planning_history["policies"]) > 100:
            for key in self.planning_history:
                self.planning_history[key] = self.planning_history[key][-50:]
