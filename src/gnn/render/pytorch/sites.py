"""GNN variable to traced PyTorch site mapping for the discrete renderer.

The discrete PyTorch renderer emits ``torch.nn.Module`` components whose
buffer and submodule addresses are fixed (``SITE_DESCRIPTIONS``). This module
maps the variables a GNN specification names in ``## Connections`` onto those
addresses, so a TorchLens trace of the emitted rollout can be compared with
the declared graph by :mod:`gnn.analysis.pytorch.torchlens_fidelity`.

The mapping is deliberately conservative. Matrices map by their canonical
letters only; hidden-state, observation and control variables map through the
extractor's factor roles only when the flat single-factor emission makes the
correspondence unambiguous. Every other variable is reported as unmapped with
a reason rather than guessed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

#: Address -> meaning for every module and buffer of ``ActiveInferenceRollout``.
#: The generative model (``agent``) and generative process (``process``) hold
#: separate buffers so a trace distinguishes inference from environment flow.
SITE_DESCRIPTIONS: Dict[str, str] = {
    "agent.A": "likelihood used by the agent for inference and EFE",
    "agent.B": "transitions used by the agent for prediction and EFE",
    "agent.C": "observation preferences (log-preferences before softmax)",
    "agent.D": "prior over initial hidden states",
    "agent.E": "habit prior over one-step actions",
    "agent.transition": "predicted next-state prior B[:, :, u] s (s_prime)",
    "agent.state_inference": "posterior over hidden states (s)",
    "agent.efe": "expected free energy per one-step action (G)",
    "agent.efe.ambiguity": "expected observation entropy (epistemic term of G)",
    "agent.efe.risk": "KL of predicted observations from softmax(C) (risk term)",
    "agent.policy": "action distribution softmax(log E - G) (pi)",
    "agent.action": "sampled one-hot action (u)",
    "process.A": "likelihood used by the environment to emit observations",
    "process.B": "transitions used by the environment to move the true state",
    "process.D": "initial true-state distribution of the environment",
    "process.initial_state": "sampled initial true hidden state",
    "process.observe": "sampled one-hot observation (o)",
    "process.transition": "sampled next true hidden state",
}

_MATRIX_SITES: Dict[str, Tuple[str, ...]] = {
    "A": ("agent.A", "process.A"),
    "B": ("agent.B", "process.B"),
    "C": ("agent.C",),
    "D": ("agent.D", "process.D"),
    "E": ("agent.E",),
}

_EFE_SITES: Dict[str, Tuple[str, ...]] = {
    "G": ("agent.efe",),
    "EFE": ("agent.efe",),
    "G_epi": ("agent.efe.ambiguity",),
    "G_epistemic": ("agent.efe.ambiguity",),
    "G_ins": ("agent.efe.risk",),
    "G_instrumental": ("agent.efe.risk",),
    "G_pragmatic": ("agent.efe.risk",),
}

_POLICY_NAMES = frozenset({"π", "pi", "policy"})

_NOT_COMPUTED: Dict[str, str] = {
    "F": "variational free energy is not computed by the PyTorch renderer",
    "VFE": "variational free energy is not computed by the PyTorch renderer",
    "t": "the time index is a Python loop counter, not a traced tensor",
    "γ": "policy precision is fixed at one by the PyTorch renderer",
    "gamma": "policy precision is fixed at one by the PyTorch renderer",
}

_GROUP = re.compile(r"^\((?P<members>.*)\)$")
_TYPED_RELATIONS = {
    "directed": ">",
    "undirected": "-",
    "conditional": "|",
    "bidirectional": "<->",
}


def connection_endpoints(name: str) -> List[str]:
    """Return the variable names of one connection endpoint.

    Grouped endpoints such as ``(s_f0, u)`` expand to their members; a plain
    name is returned as a single-item list. Empty members are dropped.
    """
    text = str(name).strip()
    match = _GROUP.match(text)
    if match is None:
        return [text] if text else []
    return [part.strip() for part in match.group("members").split(",") if part.strip()]


def _typed_endpoint(values: Sequence[str]) -> str:
    members = ", ".join(values)
    return f"({members})" if len(values) > 1 else members


def declared_connections(gnn_spec: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Return the specification's ``## Connections`` as source/relation/target rows.

    Extracted rows keep the authored endpoint text and relation symbol (``>``, ``<`` or
    ``-``); grouped endpoints are expanded later by the comparison.
    """
    rows: List[Dict[str, str]] = []
    for connection in gnn_spec.get("connections") or []:
        if not isinstance(connection, Mapping):
            continue
        source = str(connection.get("source") or "").strip()
        target = str(connection.get("target") or "").strip()
        relation = str(connection.get("relation", "-")).strip() or "-"
        if not source and not target and "connection_type" in connection:
            source = _typed_endpoint(connection.get("source_variables") or [])
            target = _typed_endpoint(connection.get("target_variables") or [])
            kind = connection["connection_type"]
            relation = _TYPED_RELATIONS.get(kind, str(kind))
        if source and target:
            rows.append({"source": source, "relation": relation, "target": target})
    return rows


def _factor_names(
    model_parameters: Mapping[str, Any], key: str
) -> List[Mapping[str, Any]]:
    factors = model_parameters.get(key) or []
    return [
        factor
        for factor in factors
        if isinstance(factor, Mapping) and str(factor.get("name", "")).strip()
    ]


def _role_sites(
    gnn_spec: Mapping[str, Any],
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, str]]:
    """Map state/observation/control variables through the extractor roles."""
    model_parameters = gnn_spec.get("model_parameters") or {}
    if not isinstance(model_parameters, Mapping):
        return {}, {}
    mapped: Dict[str, Tuple[str, ...]] = {}
    unmapped: Dict[str, str] = {}

    def assign(names: Sequence[str], sites: Tuple[str, ...], kind: str) -> None:
        if len(names) == 1:
            mapped[names[0]] = sites
            return
        for name in names:
            unmapped[name] = (
                f"{len(names)} {kind} variables are flattened into one emitted "
                "site by the PyTorch renderer; no per-variable site exists"
            )

    states = _factor_names(model_parameters, "state_factors")
    assign(
        [str(f["name"]) for f in states if f.get("role") == "factor"],
        ("agent.state_inference",),
        "hidden-state factor",
    )
    assign(
        [str(f["name"]) for f in states if f.get("role") == "bookkeeping"],
        ("agent.transition",),
        "predicted-state",
    )
    observations = _factor_names(model_parameters, "observation_modalities")
    assign(
        [str(f["name"]) for f in observations if f.get("role") == "factor"],
        ("process.observe",),
        "observation modality",
    )
    controls = _factor_names(model_parameters, "control_factors")
    policies = [
        str(f["name"])
        for f in controls
        if str(f["name"]) in _POLICY_NAMES or f.get("role") == "bookkeeping"
    ]
    actions = [
        str(f["name"])
        for f in controls
        if str(f["name"]) not in policies and f.get("role") == "factor"
    ]
    assign(policies, ("agent.policy",), "policy")
    assign(actions, ("agent.action",), "control factor")
    return mapped, unmapped


def gnn_variable_sites(
    gnn_spec: Mapping[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Map every connection endpoint variable to emitted PyTorch sites.

    Returns ``(variable_sites, unmapped_reasons)``. ``variable_sites`` maps a
    GNN variable name to the module/buffer addresses that compute it;
    ``unmapped_reasons`` explains each variable without a traced site.
    """
    role_mapped, role_unmapped = _role_sites(gnn_spec)
    names: List[str] = []
    for row in declared_connections(gnn_spec):
        for endpoint in (row["source"], row["target"]):
            for name in connection_endpoints(endpoint):
                if name not in names:
                    names.append(name)

    variable_sites: Dict[str, List[str]] = {}
    unmapped: Dict[str, str] = {}
    for name in names:
        sites = _MATRIX_SITES.get(name) or _EFE_SITES.get(name) or role_mapped.get(name)
        if sites:
            variable_sites[name] = list(sites)
        elif name in role_unmapped:
            unmapped[name] = role_unmapped[name]
        elif name in _NOT_COMPUTED:
            unmapped[name] = _NOT_COMPUTED[name]
        else:
            unmapped[name] = (
                "no emitted PyTorch module or buffer computes this variable "
                "under the flat single-factor rendering"
            )
    return variable_sites, unmapped


__all__ = [
    "SITE_DESCRIPTIONS",
    "connection_endpoints",
    "declared_connections",
    "gnn_variable_sites",
]
