#!/usr/bin/env python3
"""
AXIOM: Active eXpanding Inference with Object-centric Models
============================================================

Main orchestration file for the complete AXIOM implementation based on GNN specifications.
This module coordinates all four mixture models (sMM, iMM, tMM, rMM) plus structure learning
and active inference planning.

Authors: AXIOM Research Team
Institution: VERSES AI / Active Inference Institute
Based on: Heins et al. (2025) - arXiv:2505.24784
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from modules.identity_mixture_model import IdentityMixtureModel
from modules.planning import ActiveInferencePlanning
from modules.recurrent_mixture_model import RecurrentMixtureModel

# Import AXIOM components
from modules.slot_mixture_model import SlotMixtureModel
from modules.structure_learning import StructureLearning
from modules.transition_mixture_model import TransitionMixtureModel
from utils.performance_utils import (
    PerformanceTracker,
)
from utils.visualization_utils import (
    plot_model_complexity,
    plot_performance_metrics,
    plot_reward_history,
    visualize_slots,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AxiomConfig:
    """Configuration for AXIOM system based on GNN specifications."""

    # Core dimensions
    K_slots: int = 8  # Number of object slots
    V_identities: int = 5  # Number of identity types
    L_dynamics: int = 10  # Number of dynamics modes
    M_contexts: int = 20  # Number of context modes

    # Maximum limits
    K_max: int = 16  # Maximum slots
    V_max: int = 10  # Maximum identities
    L_max: int = 20  # Maximum dynamics modes
    M_max: int = 100  # Maximum contexts

    # Environment parameters
    screen_height: int = 210  # Gameworld screen height
    screen_width: int = 160  # Gameworld screen width
    n_pixels: int = 33600  # Total pixels (210 * 160)
    action_space_size: int = 5  # Number of actions

    # Structure learning parameters
    tau_smm: float = -2.0  # sMM expansion threshold
    tau_imm: float = -1.5  # iMM expansion threshold
    tau_tmm: float = -1.0  # tMM expansion threshold
    tau_rmm: float = -0.5  # rMM expansion threshold
    T_bmr: int = 500  # BMR interval

    # Planning parameters
    H_planning: int = 16  # Planning horizon
    N_rollouts: int = 512  # Number of rollout samples
    gamma_precision: float = 16.0  # Policy precision

    # Learning parameters
    alpha_smm: float = 1.0  # sMM stick-breaking concentration
    alpha_imm: float = 1.0  # iMM stick-breaking concentration
    alpha_tmm: float = 1.0  # tMM stick-breaking concentration
    alpha_rmm: float = 1.0  # rMM stick-breaking concentration

    # Feature dimensions
    F_continuous: int = 10  # Continuous context features
    F_discrete: int = 5  # Discrete context features

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("./axiom_output"))
    save_visualizations: bool = True
    save_models: bool = True
    log_performance: bool = True


class AxiomAgent:
    """
    Complete AXIOM agent implementing all four mixture models with structure learning
    and active inference planning. Follows GNN specifications for mathematical rigor.
    """

    def __init__(self, config: AxiomConfig):
        self.config = config
        self.timestep = 0
        self.episode = 0
        self.total_reward = 0.0

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker()

        # Initialize mixture models (following GNN StateSpaceBlock specifications)
        logger.info("Initializing AXIOM mixture models...")
        self._initialize_mixture_models()

        # Initialize structure learning
        logger.info("Initializing structure learning...")
        self.structure_learning = StructureLearning(
            config=config,
            models={"smm": self.smm, "imm": self.imm, "tmm": self.tmm, "rmm": self.rmm},
        )

        # Initialize planning
        logger.info("Initializing active inference planning...")
        self.planning = ActiveInferencePlanning(
            config=config,
            models={"smm": self.smm, "imm": self.imm, "tmm": self.tmm, "rmm": self.rmm},
        )

        # Initialize state variables
        self._initialize_state_variables()

        logger.info(
            f"AXIOM agent initialized with {self.count_parameters()} parameters"
        )

    def _initialize_mixture_models(self):
        """Initialize all four mixture models following GNN InitialParameterization."""

        # Slot Mixture Model (sMM) - Object-centric perception
        self.smm = SlotMixtureModel(
            K_slots=self.config.K_slots,
            N_pixels=self.config.n_pixels,
            alpha_smm=self.config.alpha_smm,
        )

        # Identity Mixture Model (iMM) - Object categorization
        self.imm = IdentityMixtureModel(
            K_slots=self.config.K_slots,
            V_identities=self.config.V_identities,
            alpha_imm=self.config.alpha_imm,
        )

        # Transition Mixture Model (tMM) - Object dynamics
        self.tmm = TransitionMixtureModel(
            K_slots=self.config.K_slots,
            L_dynamics=self.config.L_dynamics,
            alpha_tmm=self.config.alpha_tmm,
        )

        # Recurrent Mixture Model (rMM) - Object interactions
        self.rmm = RecurrentMixtureModel(
            K_slots=self.config.K_slots,
            M_contexts=self.config.M_contexts,
            F_continuous=self.config.F_continuous,
            F_discrete=self.config.F_discrete,
            alpha_rmm=self.config.alpha_rmm,
        )

    def _initialize_state_variables(self):
        """Initialize state variables following GNN StateSpaceBlock."""

        # Object slots (K x 7: position(2) + color(3) + shape(2))
        self.s_slot = np.random.normal(
            loc=[0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1],  # Center, gray, small
            scale=0.1,
            size=(self.config.K_slots, 7),
        )

        # Slot presence and movement indicators
        self.z_slot_present = np.ones(self.config.K_slots, dtype=bool)
        self.z_slot_moving = np.zeros(self.config.K_slots, dtype=bool)

        # Current observations and actions
        self.o_pixels = np.zeros((self.config.n_pixels, 5))  # RGB + XY
        self.u_action = 0  # No action initially
        self.r_reward = 0.0

        # Planning variables
        self.pi_policy = (
            np.ones((self.config.H_planning, self.config.action_space_size))
            / self.config.action_space_size
        )
        self.G_expected_free_energy = np.zeros(self.config.H_planning)

        # History for analysis
        self.history = {
            "rewards": [],
            "actions": [],
            "free_energy": [],
            "model_complexity": [],
            "slot_trajectories": [],
            "performance_metrics": [],
        }

    def step(self, observation: np.ndarray, reward: float) -> int:
        """
        Single timestep of AXIOM agent following GNN Equations and Connections.

        Args:
            observation: Visual observation (H x W x 3 RGB image)
            reward: Scalar reward signal

        Returns:
            action: Selected action (discrete)
        """

        self.timestep += 1
        self.r_reward = reward
        self.total_reward += reward

        with self.performance_tracker.track_operation("total_step_time"):
            # 1. Perception: Convert observation to pixel features (sMM input)
            with self.performance_tracker.track_operation("perception"):
                self.o_pixels = self._observation_to_pixels(observation)

            # 2. Object-centric parsing via sMM
            with self.performance_tracker.track_operation("slot_mixture_model"):
                slot_assignments, slot_features = self.smm.inference(self.o_pixels)
                self.s_slot = slot_features

            # 3. Identity classification via iMM
            with self.performance_tracker.track_operation("identity_mixture_model"):
                self.imm.inference(self.s_slot[:, 2:7])  # Color + shape features

            # 4. Context feature construction for rMM
            with self.performance_tracker.track_operation("context_features"):
                f_continuous, d_discrete = self._construct_context_features()

            # 5. Context classification and dynamics prediction via rMM
            with self.performance_tracker.track_operation("recurrent_mixture_model"):
                context_assignments, dynamics_predictions, reward_predictions = (
                    self.rmm.inference(
                        f_continuous, d_discrete, self.u_action, self.r_reward
                    )
                )

            # 6. Dynamics update via tMM
            with self.performance_tracker.track_operation("transition_mixture_model"):
                self.s_slot = self.tmm.inference(self.s_slot, dynamics_predictions)

            # 7. Structure learning (expansion and BMR)
            with self.performance_tracker.track_operation("structure_learning"):
                if self.timestep % 10 == 0:  # Check expansion every 10 steps
                    self.structure_learning.check_expansion()

                if self.timestep % self.config.T_bmr == 0:  # Apply BMR periodically
                    self.structure_learning.apply_bmr()

            # 8. Active inference planning
            with self.performance_tracker.track_operation("planning"):
                action = self.planning.plan(
                    s_slot=self.s_slot, o_pixels=self.o_pixels, r_reward=self.r_reward
                )

            # 9. Update history and visualizations
            self._update_history()

            if self.config.save_visualizations and self.timestep % 100 == 0:
                self._save_visualizations()

        self.u_action = action
        return action

    def _observation_to_pixels(self, observation: np.ndarray) -> np.ndarray:
        """Convert RGB image to pixel features with coordinates."""

        H, W, C = observation.shape
        pixels = np.zeros((H * W, 5))  # RGB + XY coordinates

        idx = 0
        for y in range(H):
            for x in range(W):
                pixels[idx, 0:3] = observation[y, x] / 255.0  # Normalize RGB
                pixels[idx, 3] = x / W  # Normalized X coordinate
                pixels[idx, 4] = y / H  # Normalized Y coordinate
                idx += 1

        return pixels

    def _construct_context_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Construct continuous and discrete context features for rMM."""

        f_continuous = np.zeros((self.config.K_slots, self.config.F_continuous))
        d_discrete = np.zeros((self.config.K_slots, self.config.F_discrete), dtype=int)

        for k in range(self.config.K_slots):
            if not self.z_slot_present[k]:
                continue

            # Continuous features (following GNN equations)
            f_continuous[k, 0:2] = self.s_slot[k, 0:2]  # Position

            # Distances to other objects
            distances = []
            for j in range(self.config.K_slots):
                if j != k and self.z_slot_present[j]:
                    dist = np.linalg.norm(self.s_slot[k, 0:2] - self.s_slot[j, 0:2])
                    distances.append(dist)

            if distances:
                f_continuous[k, 2] = np.mean(distances)  # Average distance
                f_continuous[k, 3] = np.min(distances)  # Minimum distance

            # Velocity (if we have previous state)
            if hasattr(self, "s_slot_prev"):
                f_continuous[k, 4:6] = self.s_slot[k, 0:2] - self.s_slot_prev[k, 0:2]

            # Center of mass relative position
            center_of_mass = np.mean(self.s_slot[self.z_slot_present, 0:2], axis=0)
            f_continuous[k, 6:8] = self.s_slot[k, 0:2] - center_of_mass

            # Recent reward history
            if len(self.history["rewards"]) >= 2:
                f_continuous[k, 8:10] = self.history["rewards"][-2:]

            # Discrete features
            d_discrete[k, 0] = k  # Slot identity for iMM output
            d_discrete[k, 1] = self.u_action  # Previous action
            d_discrete[k, 2] = np.sign(self.r_reward) + 1  # Reward sign (0, 1, 2)
            d_discrete[k, 3] = int(self.z_slot_moving[k])  # Movement indicator
            d_discrete[k, 4] = int(self._check_boundary_contact(k))  # Boundary contact

        return f_continuous, d_discrete

    def _check_boundary_contact(self, slot_idx: int) -> bool:
        """Check if slot is in contact with environment boundary."""
        pos = self.s_slot[slot_idx, 0:2]
        shape = self.s_slot[slot_idx, 5:7]

        # Check boundaries with shape extent
        return (
            pos[0] - shape[0] <= 0.01
            or pos[0] + shape[0] >= 0.99
            or pos[1] - shape[1] <= 0.01
            or pos[1] + shape[1] >= 0.99
        )

    def _update_history(self):
        """Update history for analysis and visualization."""

        self.history["rewards"].append(self.r_reward)
        self.history["actions"].append(self.u_action)
        self.history["free_energy"].append(self.G_expected_free_energy.sum())

        # Model complexity
        complexity = {
            "K_slots": self.smm.K_active,
            "V_identities": self.imm.V_active,
            "L_dynamics": self.tmm.L_active,
            "M_contexts": self.rmm.M_active,
            "total_parameters": self.count_parameters(),
        }
        self.history["model_complexity"].append(complexity)

        # Slot trajectories
        self.history["slot_trajectories"].append(self.s_slot.copy())

        # Performance metrics
        if self.timestep % 100 == 0:
            metrics = self.performance_tracker.get_summary()
            self.history["performance_metrics"].append(metrics)

        # Save previous state for velocity calculation
        self.s_slot_prev = self.s_slot.copy()

    def _save_visualizations(self):
        """Save visualizations of current state."""

        if not self.config.save_visualizations:
            return

        datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = (
            self.config.output_dir / "visualizations" / f"step_{self.timestep:06d}"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Plot slot positions and features
        visualize_slots(
            self.s_slot, self.z_slot_present, save_path=viz_dir / "slots.png"
        )

        # Plot reward history
        plot_reward_history(self.history["rewards"], save_path=viz_dir / "rewards.png")

        # Plot model complexity evolution
        plot_model_complexity(
            self.history["model_complexity"], save_path=viz_dir / "complexity.png"
        )

        # Plot performance metrics
        if self.history["performance_metrics"]:
            plot_performance_metrics(
                self.history["performance_metrics"],
                save_path=viz_dir / "performance.png",
            )

    def count_parameters(self) -> int:
        """Count total number of parameters in all models."""

        total = 0
        total += self.smm.count_parameters()
        total += self.imm.count_parameters()
        total += self.tmm.count_parameters()
        total += self.rmm.count_parameters()

        return total

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state dictionary for saving/loading."""

        return {
            "config": self.config,
            "timestep": self.timestep,
            "episode": self.episode,
            "total_reward": self.total_reward,
            "s_slot": self.s_slot,
            "z_slot_present": self.z_slot_present,
            "z_slot_moving": self.z_slot_moving,
            "smm_state": self.smm.get_state_dict(),
            "imm_state": self.imm.get_state_dict(),
            "tmm_state": self.tmm.get_state_dict(),
            "rmm_state": self.rmm.get_state_dict(),
            "structure_learning_state": self.structure_learning.get_state_dict(),
            "planning_state": self.planning.get_state_dict(),
            "history": self.history,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load complete state from dictionary."""

        self.timestep = state_dict["timestep"]
        self.episode = state_dict["episode"]
        self.total_reward = state_dict["total_reward"]
        self.s_slot = state_dict["s_slot"]
        self.z_slot_present = state_dict["z_slot_present"]
        self.z_slot_moving = state_dict["z_slot_moving"]

        self.smm.load_state_dict(state_dict["smm_state"])
        self.imm.load_state_dict(state_dict["imm_state"])
        self.tmm.load_state_dict(state_dict["tmm_state"])
        self.rmm.load_state_dict(state_dict["rmm_state"])
        self.structure_learning.load_state_dict(state_dict["structure_learning_state"])
        self.planning.load_state_dict(state_dict["planning_state"])

        self.history = state_dict["history"]

    def save(self, filepath: Path):
        """Save complete agent state."""

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.get_state_dict(), f)

        logger.info(f"AXIOM agent saved to {filepath}")

    def load(self, filepath: Path):
        """Load complete agent state."""

        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)

        self.load_state_dict(state_dict)
        logger.info(f"AXIOM agent loaded from {filepath}")

    def reset_episode(self):
        """Reset for new episode."""

        self.episode += 1
        self.timestep = 0
        self.total_reward = 0.0

        # Reset state variables but keep learned parameters
        self._initialize_state_variables()

        logger.info(f"Starting episode {self.episode}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the agent."""

        return {
            "timestep": self.timestep,
            "episode": self.episode,
            "total_reward": self.total_reward,
            "total_parameters": self.count_parameters(),
            "model_complexity": {
                "K_slots": self.smm.K_active
                if hasattr(self.smm, "K_active")
                else self.config.K_slots,
                "V_identities": self.imm.V_active
                if hasattr(self.imm, "V_active")
                else self.config.V_identities,
                "L_dynamics": self.tmm.L_active
                if hasattr(self.tmm, "L_active")
                else self.config.L_dynamics,
                "M_contexts": self.rmm.M_active
                if hasattr(self.rmm, "M_active")
                else self.config.M_contexts,
            },
            "performance_summary": self.performance_tracker.get_summary()
            if hasattr(self, "performance_tracker")
            else {},
            "average_reward": np.mean(self.history["rewards"])
            if self.history["rewards"]
            else 0.0,
            "recent_reward": np.mean(self.history["rewards"][-100:])
            if len(self.history["rewards"]) >= 100
            else 0.0,
        }


def create_axiom_agent(
    config: Optional[AxiomConfig] = None, output_dir: Optional[Path] = None
) -> AxiomAgent:
    """
    Factory function to create AXIOM agent with sensible defaults.

    Args:
        config: Optional configuration override
        output_dir: Optional output directory override

    Returns:
        Configured AXIOM agent
    """

    if config is None:
        config = AxiomConfig()

    if output_dir is not None:
        config.output_dir = output_dir

    return AxiomAgent(config)


def run_axiom_experiment(
    agent: AxiomAgent,
    environment,
    n_episodes: int = 10,
    max_steps_per_episode: int = 10000,
    save_interval: int = 1000,
) -> Dict[str, Any]:
    """
    Run complete AXIOM experiment with environment interaction.

    Args:
        agent: AXIOM agent instance
        environment: Environment with step() and reset() methods
        n_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        save_interval: Save agent every N steps

    Returns:
        Experiment results dictionary
    """

    logger.info(
        f"Starting AXIOM experiment: {n_episodes} episodes, max {max_steps_per_episode} steps each"
    )

    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "learning_curves": [],
        "model_growth": [],
        "performance_metrics": [],
    }

    total_steps = 0

    for episode in range(n_episodes):
        observation = environment.reset()
        agent.reset_episode()

        episode_reward = 0.0
        episode_steps = 0

        for step in range(max_steps_per_episode):
            # Agent step
            action = agent.step(observation, reward=0.0 if step == 0 else reward)

            # Environment step
            observation, reward, done, info = environment.step(action)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Save agent periodically
            if total_steps % save_interval == 0:
                save_path = (
                    agent.config.output_dir / f"agent_step_{total_steps:06d}.pkl"
                )
                agent.save(save_path)

            if done:
                break

        # Record episode results
        results["episode_rewards"].append(episode_reward)
        results["episode_lengths"].append(episode_steps)
        results["learning_curves"].append(agent.history["rewards"].copy())
        results["model_growth"].append(agent.history["model_complexity"].copy())

        # Performance metrics
        summary = agent.get_summary()
        results["performance_metrics"].append(summary)

        logger.info(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Reward = {episode_reward:.2f}, "
            f"Steps = {episode_steps}, "
            f"Params = {summary['total_parameters']}"
        )

    # Save final results
    results_path = agent.config.output_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], np.ndarray)
            ):
                json_results[key] = [v.tolist() for v in value]
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    # Save final agent
    final_agent_path = agent.config.output_dir / "final_agent.pkl"
    agent.save(final_agent_path)

    logger.info(f"Experiment completed. Results saved to {agent.config.output_dir}")

    return results


if __name__ == "__main__":
    """
    Example usage and testing of AXIOM implementation.
    """

    # Create configuration
    config = AxiomConfig(
        K_slots=8,
        V_identities=5,
        L_dynamics=10,
        M_contexts=20,
        output_dir=Path("./axiom_test_output"),
    )

    # Create agent
    agent = create_axiom_agent(config)

    # Test with sample environment
    class DummyEnvironment:
        def __init__(self):
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return np.random.rand(210, 160, 3)  # Random RGB image

        def step(self, action):
            self.step_count += 1
            observation = np.random.rand(210, 160, 3)
            reward = np.random.randn() * 0.1  # Small random reward
            done = self.step_count >= 1000
            info = {}
            return observation, reward, done, info

    # Run experiment
    env = DummyEnvironment()
    results = run_axiom_experiment(
        agent=agent,
        environment=env,
        n_episodes=2,
        max_steps_per_episode=1000,
        save_interval=500,
    )

    print("AXIOM experiment completed successfully!")
    print(f"Final agent parameters: {agent.count_parameters()}")
    print(f"Average episode reward: {np.mean(results['episode_rewards']):.3f}")
