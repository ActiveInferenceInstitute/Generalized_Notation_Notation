#!/usr/bin/env python3
"""Shared multi-agent detection helpers for framework renderers.

Both the RxInfer.jl and ActiveInference.jl renderers historically treated
multi-agent GNN specs as the POMDP extractor's *composed joint* model (one
joint state space of size ``prod(factor_sizes)``). This module provides the
shared detection layer for the native multi-agent (stigmergic) compilation
path: it recovers the per-agent generative models (``A_agentN``,
``B_agentN``, ``C_agentN``, ``D_agentN`` from ``structured_pomdp.matrices``)
and the shared environmental affordance (``env_signal`` initialisation plus
``signal_decay``) directly from the parsed GNN spec, so renderers can emit
per-agent scripts and an auditable environment trace without expanding the
joint state space.

The exemplar this targets is ``input/gnn_files/multiagent/stigmergic_swarm.md``:
three homogeneous agents navigating a shared 3x3 grid whose cells carry a
stigmergic signal. The backends reconstruct ``env_signal`` after independent
per-agent inference (deposition at occupied cells, decay per timestep). When
the spec additionally declares an env-conditioned observation likelihood
(``env_obs_likelihood``) and latent signal prior (``env_signal_prior``), each
agent also infers the local signal level as a latent from its observations and
conditions its action selection on that latent (signal-seeking). See
:func:`detect_env_conditioned` / :func:`has_env_conditioned_action_selection`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "AGENT_MATRIX_RE",
    "detect_agent_groups",
    "detect_env_coupling",
    "detect_env_conditioned",
    "multi_agent_structure",
    "has_native_multi_agent_structure",
    "has_env_conditioned_action_selection",
    "canonicalise_b",
    "validate_native_agent_groups",
]

# Matches per-agent generative matrices: A_agent1, B_agent2, ... The letters
# are the canonical POMDP matrices (A likelihood, B transition, C preference,
# D initial prior). E (habit prior) is optional everywhere and defaulted.
AGENT_MATRIX_RE = re.compile(r"^([ABCDE])_agent(\d+)$")


def _as_flat_list(value: Any) -> List[float]:
    """Flatten a nested GNN matrix (tuple-of-lists) into ``float``s."""
    return [float(x) for x in value]


def canonicalise_b(
    value: Any,
    num_actions: int,
    *,
    model_parameters: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[List[List[float]]]:
    """Use the common declared-axis contract for a native agent transition.

    Earlier agent literals use action/next/previous, unlike the historical
    generic action/previous/next default. An explicit declaration or recorded
    source orientation takes precedence, including equal-sized tensor axes.
    """
    from gnn.render.pomdp_contract import canonicalise_b_matrix

    params = dict(model_parameters or {})
    source_order = (provenance or {}).get("source_order")
    if (
        not any(
            params.get(k)
            for k in ("b_tensor_order", "B_tensor_order", "transition_tensor_order")
        )
        and source_order
        and source_order != "inferred"
    ):
        params["b_tensor_order"] = source_order
    raw = np.asarray(value)
    if (
        not any(
            params.get(k)
            for k in ("b_tensor_order", "B_tensor_order", "transition_tensor_order")
        )
        and raw.ndim == 3
        and raw.shape[0] in {1, num_actions}
        and raw.shape[1] == raw.shape[2]
    ):
        params["b_tensor_order"] = "action_next_state_previous_state"
    states = (
        raw.shape[1]
        if raw.ndim == 3 and params.get("b_tensor_order", "").startswith("action_")
        else raw.shape[0]
    )
    return canonicalise_b_matrix(
        value, num_states=states, num_actions=num_actions, model_parameters=params
    )[0]


def validate_native_agent_groups(gnn_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Validate source factors, returning canonical values plus explicit custody.

    Native action selection currently has no habit-prior term. Declared E_agentN
    is rejected rather than omitted or silently replaced by a uniform habit.
    """
    from copy import deepcopy

    from gnn.render.pomdp_contract import (
        canonicalise_b_matrix,
        normalise_matrix_columns,
        normalise_vector,
    )

    agents = detect_agent_groups(gnn_spec)
    structured = gnn_spec.get("structured_pomdp") or {}
    matrices = structured.get("matrices") or {}
    declared = {
        f"agent{match[2]}"
        for key in matrices
        if (match := AGENT_MATRIX_RE.fullmatch(str(key)))
    }
    if set(agents) != declared or len(agents) < 2:
        raise ValueError(
            "Native agents require complete A/B/C/D declarations for every agent"
        )
    if any(key.startswith("E_agent") for key in matrices):
        raise ValueError(
            "Native agent habit priors E_agentN are unsupported; declared values cannot be omitted"
        )
    params = gnn_spec.get("model_parameters") or {}
    count = params.get("nr_agents", params.get("num_agents", len(agents)))
    if (
        isinstance(count, bool)
        or not np.isfinite(float(count))
        or float(count) != len(agents)
    ):
        raise ValueError("Declared agent count does not match source matrix identities")
    provenance = (
        gnn_spec.get("matrix_provenance") or structured.get("matrix_provenance") or {}
    )
    canonical_groups = {}
    for name, group in agents.items():
        a = np.asarray(normalise_matrix_columns(group["A"], name=f"{name}/A"))
        d = normalise_vector(group["D"], name=f"{name}/D")
        c = np.asarray(group["C"], dtype=float).reshape(-1)
        if a.shape[1] != len(d) or a.shape[0] != len(c) or not np.isfinite(c).all():
            raise ValueError(f"{name}: A/C/D dimensions or preferences are invalid")
        agent_params = dict(params)
        per_agent_order = params.get(f"b_tensor_order_{name}")
        b_provenance = provenance.get(f"B_{name}") or {}
        source_order = (
            per_agent_order
            or params.get("b_tensor_order")
            or params.get("B_tensor_order")
            or params.get("transition_tensor_order")
            or b_provenance.get("source_order")
        )
        raw_b = np.asarray(group["B"])
        if not source_order or source_order == "inferred":
            source_order = (
                "action_next_state_previous_state"
                if raw_b.ndim == 3 and raw_b.shape[1:] == (len(d), len(d))
                else "next_state_previous_state_action"
            )
        agent_params["b_tensor_order"] = source_order
        inferred_actions = (
            raw_b.shape[0]
            if str(source_order).startswith("action_")
            else raw_b.shape[-1]
            if raw_b.ndim == 3
            else 1
        )
        actions_raw = params.get("num_actions", inferred_actions)
        if (
            actions_raw is None
            or isinstance(actions_raw, bool)
            or float(actions_raw) < 1
            or float(actions_raw) != int(actions_raw)
        ):
            raise ValueError(f"{name}: num_actions must be a positive integer")
        b, b_meta = canonicalise_b_matrix(
            group["B"],
            num_states=len(d),
            num_actions=int(actions_raw),
            model_parameters=agent_params,
        )
        custody = {
            "B": {
                **deepcopy(b_provenance),
                **b_meta,
                "source_values": deepcopy(group["B"]),
                "derived": b_meta["normalized"],
            }
        }
        for key, value in (("A", a.tolist()), ("D", d)):
            changed = not np.array_equal(np.asarray(value), np.asarray(group[key]))
            custody[key] = {
                **deepcopy(provenance.get(f"{key}_{name}") or {}),
                "source_values": deepcopy(group[key]),
                "derived": changed,
                "normalized": changed,
                "normalization": "declared_precision_mass_correction"
                if changed
                else None,
            }
        canonical_groups[name] = {
            "A": a.tolist(),
            "B": b,
            "C": c.tolist(),
            "D": d,
            "matrix_provenance": custody,
        }
    return canonical_groups


def detect_agent_groups(gnn_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return per-agent canonical matrices keyed by agent name.

    Reads ``structured_pomdp.matrices`` from the parsed GNN spec and groups
    ``A_agentN`` / ``B_agentN`` / ``C_agentN`` / ``D_agentN`` entries into
    ``{"agentN": {"A": ..., "B": ..., "C": ..., "D": ...}}``. An agent group
    is only returned when all four matrices are present (the minimum for a
    generative model). Returns ``{}`` for flat / non-multi-agent specs.
    """
    matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in matrices.items():
        match = AGENT_MATRIX_RE.match(str(key))
        if not match:
            continue
        letter, agent_index = match.group(1), match.group(2)
        agent_name = f"agent{agent_index}"
        groups.setdefault(agent_name, {})[letter] = value
    return {
        name: dict(m) for name, m in groups.items() if {"A", "B", "C", "D"} <= set(m)
    }


def detect_env_coupling(
    gnn_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the stigmergic environment coupling when declared.

    Looks for the shared affordance declarations in
    ``initialparameterization``: an ``env_signal`` vector (per-cell signal
    intensities, all-zero initialisation in the swarm exemplar) and a scalar
    ``signal_decay`` (per-timestep retention factor). Returns
    ``{"variable": "env_signal", "initial": [...], "decay": float}`` or
    ``None`` when either piece is missing (non-stigmergic model).
    """
    initial = gnn_spec.get("initialparameterization") or {}
    env_initial = initial.get("env_signal")
    decay = initial.get("signal_decay")
    if env_initial is None or decay is None:
        return None
    try:
        initial_vector = _as_flat_list(env_initial)
        decay_values = _as_flat_list(decay)
    except (TypeError, ValueError):
        return None
    if not initial_vector or not decay_values:
        return None
    return {
        "variable": "env_signal",
        "initial": initial_vector,
        "decay": float(decay_values[0]),
    }


def _as_float_matrix(value: Any) -> List[List[float]]:
    """Flatten a nested GNN matrix literal into row lists of ``float``s."""
    return [[float(x) for x in row] for row in value]


def detect_env_conditioned(
    gnn_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the env-conditioned observation likelihood when declared.

    Reads the MAJ-03 declarations from ``initialparameterization``:
    ``env_obs_likelihood`` (P(observation category | local signal level), with
    columns over signal levels none/low/high), ``env_signal_prior`` (prior over
    those signal levels), and an optional ``signal_seek`` scalar gain. Returns
    ``{"obs_likelihood": [...], "signal_prior": [...], "seek": float}`` or
    ``None`` when the likelihood or prior is absent (non-conditioned model).
    """
    initial = gnn_spec.get("initialparameterization") or {}
    obs_likelihood = initial.get("env_obs_likelihood")
    signal_prior = initial.get("env_signal_prior")
    if obs_likelihood is None or signal_prior is None:
        return None
    try:
        likelihood_matrix = _as_float_matrix(obs_likelihood)
        prior_vector = _as_flat_list(signal_prior)
    except (TypeError, ValueError):
        return None
    if not likelihood_matrix or not prior_vector:
        return None
    seek = initial.get("signal_seek")
    seek_value = float(seek[0]) if seek else 1.0
    return {
        "obs_likelihood": likelihood_matrix,
        "signal_prior": prior_vector,
        "seek": seek_value,
    }


def has_env_conditioned_action_selection(gnn_spec: Dict[str, Any]) -> bool:
    """Return True when the spec declares env-conditioned action selection.

    Requires both the env-coupling (``env_signal``/``signal_decay``) and the
    env-conditioned observation likelihood / latent signal prior.
    """
    return (
        detect_env_coupling(gnn_spec) is not None
        and detect_env_conditioned(gnn_spec) is not None
    )


def multi_agent_structure(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the full native multi-agent structure for a GNN spec.

    Combines :func:`detect_agent_groups` and :func:`detect_env_coupling`.
    The result is used by renderers to decide between the composed-joint
    path and the native per-agent (stigmergic) path:

    .. code-block:: python

        {"agents": {"agent1": {...}, ...}, "env": {...} | None}
    """
    return {
        "agents": detect_agent_groups(gnn_spec),
        "env": detect_env_coupling(gnn_spec),
    }


def has_native_multi_agent_structure(gnn_spec: Dict[str, Any]) -> bool:
    """Return True when the spec declares >= 2 complete agent groups.

    Two or more agents with full ``A/B/C/D`` matrices is the threshold for
    the native multi-agent path; a single agent group renders through the
    canonical flat renderer.
    """
    return len(detect_agent_groups(gnn_spec)) >= 2
