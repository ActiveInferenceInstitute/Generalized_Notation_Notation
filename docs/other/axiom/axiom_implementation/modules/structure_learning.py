#!/usr/bin/env python3
"""
Structure Learning - Online Bayesian structure learning module.

Implements online structure learning with component expansion and
Bayesian Model Reduction (BMR) for AXIOM mixture models.
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class StructureLearning:
    """
    Online Bayesian structure learning for AXIOM models.

    Implements fast component addition and BMR for optimal model complexity.
    """

    def __init__(self, config, models: Dict[str, Any]):
        """
        Initialize Structure Learning.

        Args:
            config: AXIOM configuration
            models: Dictionary of mixture models
        """
        self.config = config
        self.models = models

        # Expansion thresholds
        self.tau_smm = config.tau_smm
        self.tau_imm = config.tau_imm
        self.tau_tmm = config.tau_tmm
        self.tau_rmm = config.tau_rmm

        # BMR parameters
        self.T_bmr = config.T_bmr
        self.n_bmr_pairs = getattr(config, "n_bmr_pairs", 10)
        self.bmr_threshold = getattr(config, "bmr_threshold", 0.0)

        # Component usage tracking
        self.usage_history = {"smm": [], "imm": [], "tmm": [], "rmm": []}

        # Quality metrics history
        self.quality_history = {"smm": [], "imm": [], "tmm": [], "rmm": []}

        self.timestep = 0
        self.last_bmr_timestep = 0

    def check_expansion(self):
        """Check if any models need expansion."""
        self.timestep += 1

        # Check each model for expansion
        self._check_smm_expansion()
        self._check_imm_expansion()
        self._check_tmm_expansion()
        self._check_rmm_expansion()

        # Update usage and quality tracking
        self._update_tracking()

    def _check_smm_expansion(self):
        """Check if sMM needs more slots."""
        smm = self.models["smm"]

        # Compute posterior predictive likelihoods
        qualities = []
        for k in range(smm.K_active):
            # Simplified quality metric based on mixing weights
            weight = smm.theta_smm_pi[k]
            quality = np.log(weight + 1e-8)
            qualities.append(quality)

        max_quality = max(qualities) if qualities else -np.inf
        threshold = self.tau_smm + np.log(smm.alpha_smm)

        if max_quality < threshold and smm.K_slots < self.config.K_max:
            new_K = min(smm.K_slots + 1, self.config.K_max)
            smm.expand_slots(new_K)
            smm.K_active = new_K
            logger.info(f"sMM expanded to {new_K} slots at timestep {self.timestep}")

    def _check_imm_expansion(self):
        """Check if iMM needs more identity types."""
        imm = self.models["imm"]

        # Compute posterior predictive likelihoods
        qualities = []
        for v in range(imm.V_active):
            weight = imm.theta_imm_pi[v]
            quality = np.log(weight + 1e-8)
            qualities.append(quality)

        max_quality = max(qualities) if qualities else -np.inf
        threshold = self.tau_imm + np.log(imm.alpha_imm)

        if max_quality < threshold and imm.V_identities < self.config.V_max:
            new_V = min(imm.V_identities + 1, self.config.V_max)
            imm.expand_identities(new_V)
            imm.V_active = new_V
            logger.info(
                f"iMM expanded to {new_V} identities at timestep {self.timestep}"
            )

    def _check_tmm_expansion(self):
        """Check if tMM needs more dynamics modes."""
        tmm = self.models["tmm"]

        # Compute posterior predictive likelihoods
        qualities = []
        for l in range(tmm.L_active):
            weight = tmm.theta_tmm_pi[l]
            quality = np.log(weight + 1e-8)
            qualities.append(quality)

        max_quality = max(qualities) if qualities else -np.inf
        threshold = self.tau_tmm + np.log(tmm.alpha_tmm)

        if max_quality < threshold and tmm.L_dynamics < self.config.L_max:
            new_L = min(tmm.L_dynamics + 1, self.config.L_max)
            tmm.expand_dynamics(new_L)
            tmm.L_active = new_L
            logger.info(f"tMM expanded to {new_L} dynamics at timestep {self.timestep}")

    def _check_rmm_expansion(self):
        """Check if rMM needs more context modes."""
        rmm = self.models["rmm"]

        # Compute posterior predictive likelihoods
        qualities = []
        for m in range(rmm.M_active):
            weight = rmm.theta_rmm_pi[m]
            quality = np.log(weight + 1e-8)
            qualities.append(quality)

        max_quality = max(qualities) if qualities else -np.inf
        threshold = self.tau_rmm + np.log(rmm.alpha_rmm)

        if max_quality < threshold and rmm.M_contexts < self.config.M_max:
            new_M = min(rmm.M_contexts + 1, self.config.M_max)
            rmm.expand_contexts(new_M)
            rmm.M_active = new_M
            logger.info(f"rMM expanded to {new_M} contexts at timestep {self.timestep}")

    def _update_tracking(self):
        """Update usage and quality tracking."""
        # Track component usage
        self.usage_history["smm"].append(self.models["smm"].theta_smm_pi.copy())
        self.usage_history["imm"].append(self.models["imm"].theta_imm_pi.copy())
        self.usage_history["tmm"].append(self.models["tmm"].theta_tmm_pi.copy())
        self.usage_history["rmm"].append(self.models["rmm"].theta_rmm_pi.copy())

        # Track quality metrics
        self.quality_history["smm"].append(self.models["smm"].get_complexity_metrics())
        self.quality_history["imm"].append(self.models["imm"].get_complexity_metrics())
        self.quality_history["tmm"].append(self.models["tmm"].get_complexity_metrics())
        self.quality_history["rmm"].append(self.models["rmm"].get_complexity_metrics())

        # Limit history length to prevent memory issues
        max_history = 1000
        for module in ["smm", "imm", "tmm", "rmm"]:
            if len(self.usage_history[module]) > max_history:
                self.usage_history[module] = self.usage_history[module][-max_history:]
                self.quality_history[module] = self.quality_history[module][
                    -max_history:
                ]

    def apply_bmr(self):
        """Apply Bayesian Model Reduction."""
        if self.timestep - self.last_bmr_timestep < self.T_bmr:
            return

        self.last_bmr_timestep = self.timestep

        logger.info(f"Applying BMR at timestep {self.timestep}")

        # Apply BMR to each model
        self._bmr_smm()
        self._bmr_imm()
        self._bmr_tmm()
        self._bmr_rmm()

    def _bmr_smm(self):
        """Apply BMR to sMM (slot merging)."""
        smm = self.models["smm"]

        # Find merge candidates based on similarity
        merge_candidates = []
        for i in range(smm.K_active):
            for j in range(i + 1, smm.K_active):
                # Compute merge score based on feature similarity
                slot_diff = np.linalg.norm(smm.s_slot[i] - smm.s_slot[j])
                weight_product = smm.theta_smm_pi[i] * smm.theta_smm_pi[j]
                merge_score = weight_product / (slot_diff + 1e-6)

                merge_candidates.append((merge_score, i, j))

        # Sort by merge score and consider top candidates
        merge_candidates.sort(reverse=True)

        # Perform merges for top candidates if beneficial
        merged_pairs = []
        for score, i, j in merge_candidates[: self.n_bmr_pairs]:
            if score > self.bmr_threshold and j not in [p[1] for p in merged_pairs]:
                # Merge slots i and j
                self._merge_slots(smm, i, j)
                merged_pairs.append((i, j))

                logger.info(f"BMR: Merged sMM slots {i} and {j} (score: {score:.3f})")

    def _bmr_imm(self):
        """Apply BMR to iMM (identity merging)."""
        imm = self.models["imm"]

        # Find merge candidates based on parameter similarity
        merge_candidates = []
        for i in range(imm.V_active):
            for j in range(i + 1, imm.V_active):
                # Compute merge score based on parameter similarity
                mu_diff = np.linalg.norm(imm.theta_imm_mu[i] - imm.theta_imm_mu[j])
                weight_product = imm.theta_imm_pi[i] * imm.theta_imm_pi[j]
                merge_score = weight_product / (mu_diff + 1e-6)

                merge_candidates.append((merge_score, i, j))

        # Sort and merge top candidates
        merge_candidates.sort(reverse=True)

        merged_pairs = []
        for score, i, j in merge_candidates[: self.n_bmr_pairs]:
            if score > self.bmr_threshold and j not in [p[1] for p in merged_pairs]:
                self._merge_identities(imm, i, j)
                merged_pairs.append((i, j))

                logger.info(
                    f"BMR: Merged iMM identities {i} and {j} (score: {score:.3f})"
                )

    def _bmr_tmm(self):
        """Apply BMR to tMM (dynamics merging)."""
        tmm = self.models["tmm"]

        # Find merge candidates based on dynamics similarity
        merge_candidates = []
        for i in range(tmm.L_active):
            for j in range(i + 1, tmm.L_active):
                # Compute merge score based on dynamics matrix similarity
                D_diff = np.linalg.norm(tmm.theta_tmm_D[i] - tmm.theta_tmm_D[j])
                weight_product = tmm.theta_tmm_pi[i] * tmm.theta_tmm_pi[j]
                merge_score = weight_product / (D_diff + 1e-6)

                merge_candidates.append((merge_score, i, j))

        # Sort and merge top candidates
        merge_candidates.sort(reverse=True)

        merged_pairs = []
        for score, i, j in merge_candidates[: self.n_bmr_pairs]:
            if score > self.bmr_threshold and j not in [p[1] for p in merged_pairs]:
                self._merge_dynamics(tmm, i, j)
                merged_pairs.append((i, j))

                logger.info(
                    f"BMR: Merged tMM dynamics {i} and {j} (score: {score:.3f})"
                )

    def _bmr_rmm(self):
        """Apply BMR to rMM (context merging)."""
        rmm = self.models["rmm"]

        # Find merge candidates based on context similarity
        merge_candidates = []
        for i in range(rmm.M_active):
            for j in range(i + 1, rmm.M_active):
                # Compute merge score based on mean similarity
                mu_diff = np.linalg.norm(rmm.theta_rmm_mu[i] - rmm.theta_rmm_mu[j])
                weight_product = rmm.theta_rmm_pi[i] * rmm.theta_rmm_pi[j]
                merge_score = weight_product / (mu_diff + 1e-6)

                merge_candidates.append((merge_score, i, j))

        # Sort and merge top candidates
        merge_candidates.sort(reverse=True)

        merged_pairs = []
        for score, i, j in merge_candidates[: self.n_bmr_pairs]:
            if score > self.bmr_threshold and j not in [p[1] for p in merged_pairs]:
                self._merge_contexts(rmm, i, j)
                merged_pairs.append((i, j))

                logger.info(
                    f"BMR: Merged rMM contexts {i} and {j} (score: {score:.3f})"
                )

    def _merge_slots(self, smm, i, j):
        """Merge two slots in sMM."""
        # Weighted average of slot features
        w_i = smm.theta_smm_pi[i]
        w_j = smm.theta_smm_pi[j]
        w_total = w_i + w_j

        if w_total > 1e-8:
            smm.s_slot[i] = (w_i * smm.s_slot[i] + w_j * smm.s_slot[j]) / w_total
            smm.theta_smm_pi[i] = w_total

        # Remove slot j
        self._remove_component(smm, j, "slots")

    def _merge_identities(self, imm, i, j):
        """Merge two identities in iMM."""
        # Merge using weighted average
        w_i = imm.theta_imm_pi[i]
        w_j = imm.theta_imm_pi[j]
        w_total = w_i + w_j

        if w_total > 1e-8:
            imm.theta_imm_mu[i] = (
                w_i * imm.theta_imm_mu[i] + w_j * imm.theta_imm_mu[j]
            ) / w_total
            imm.theta_imm_pi[i] = w_total

        # Remove identity j
        self._remove_component(imm, j, "identities")

    def _merge_dynamics(self, tmm, i, j):
        """Merge two dynamics modes in tMM."""
        # Weighted average of dynamics parameters
        w_i = tmm.theta_tmm_pi[i]
        w_j = tmm.theta_tmm_pi[j]
        w_total = w_i + w_j

        if w_total > 1e-8:
            tmm.theta_tmm_D[i] = (
                w_i * tmm.theta_tmm_D[i] + w_j * tmm.theta_tmm_D[j]
            ) / w_total
            tmm.theta_tmm_b[i] = (
                w_i * tmm.theta_tmm_b[i] + w_j * tmm.theta_tmm_b[j]
            ) / w_total
            tmm.theta_tmm_pi[i] = w_total

        # Remove dynamics j
        self._remove_component(tmm, j, "dynamics")

    def _merge_contexts(self, rmm, i, j):
        """Merge two contexts in rMM."""
        # Weighted average of context parameters
        w_i = rmm.theta_rmm_pi[i]
        w_j = rmm.theta_rmm_pi[j]
        w_total = w_i + w_j

        if w_total > 1e-8:
            rmm.theta_rmm_mu[i] = (
                w_i * rmm.theta_rmm_mu[i] + w_j * rmm.theta_rmm_mu[j]
            ) / w_total
            rmm.theta_rmm_reward[i] = (
                w_i * rmm.theta_rmm_reward[i] + w_j * rmm.theta_rmm_reward[j]
            ) / w_total
            rmm.theta_rmm_pi[i] = w_total

        # Remove context j
        self._remove_component(rmm, j, "contexts")

    def _remove_component(self, model, component_idx, component_type):
        """Remove a component from a model."""
        # This is a simplified removal - in practice, we'd need to
        # carefully handle all parameter arrays and reassign indices
        if component_type == "slots":
            model.K_active = max(1, model.K_active - 1)
        elif component_type == "identities":
            model.V_active = max(1, model.V_active - 1)
        elif component_type == "dynamics":
            model.L_active = max(1, model.L_active - 1)
        elif component_type == "contexts":
            model.M_active = max(1, model.M_active - 1)

    def get_state_dict(self) -> Dict[str, Any]:
        """Get structure learning state."""
        return {
            "timestep": self.timestep,
            "last_bmr_timestep": self.last_bmr_timestep,
            "usage_history": self.usage_history,
            "quality_history": self.quality_history,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load structure learning state."""
        self.timestep = state_dict["timestep"]
        self.last_bmr_timestep = state_dict["last_bmr_timestep"]
        self.usage_history = state_dict["usage_history"]
        self.quality_history = state_dict["quality_history"]

    def get_summary(self) -> Dict[str, Any]:
        """Get structure learning summary."""
        return {
            "timestep": self.timestep,
            "total_expansions": {
                "smm": len([h for h in self.usage_history["smm"] if len(h) > 1]),
                "imm": len([h for h in self.usage_history["imm"] if len(h) > 1]),
                "tmm": len([h for h in self.usage_history["tmm"] if len(h) > 1]),
                "rmm": len([h for h in self.usage_history["rmm"] if len(h) > 1]),
            },
            "current_active_components": {
                "smm": self.models["smm"].K_active,
                "imm": self.models["imm"].V_active,
                "tmm": self.models["tmm"].L_active,
                "rmm": self.models["rmm"].M_active,
            },
            "bmr_applications": (self.timestep // self.T_bmr),
        }
