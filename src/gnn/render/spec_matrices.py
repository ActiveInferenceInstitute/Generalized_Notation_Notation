#!/usr/bin/env python3
"""Shared discrete A/B/C/D extraction for lightweight framework renderers.

Single source of truth for the parameter-source fallback chain and matrix
parsing used identically by the PyTorch and NumPyro renderers (previously a
verbatim ~50-line duplicate in each). The default extraction requires complete, valid scientific matrices. The
explicit ``allow_adapters=True`` mode provides neutral defaults and weight
normalization only for callers deliberately constructing demo models.

Parameter-source precedence (first non-empty wins):
    1. ``stateSpace.parameters``
    2. ``initialparameterization``
    3. ``parameters``

Array literals embedded as strings are parsed with
``gnn.utils.runtime_safety.safe_eval.safe_literal_eval`` (bounded), never ``eval``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from gnn.render.matrix_utils import validate_abcd_shapes
from gnn.render.pomdp_contract import (
    CANONICAL_B_ORDER,
    _declared_b_order,
    canonicalise_b_matrix,
    normalise_matrix_columns,
    normalise_vector,
)


def parse_gnn_matrix_value(raw: Any, default: Any) -> Any:
    """Parse one raw matrix/vector value from a GNN spec.

    ``None``/unparseable values return *default* unchanged; lists and
    arrays convert to ``float`` ndarrays; strings go through the safe
    literal evaluator (falling back to *default* on failure).
    """
    if raw is None:
        return default
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=float)
    if isinstance(raw, str):
        try:
            from gnn.utils.runtime_safety.safe_eval import (
                MATRIX_MAX_LEN,
                safe_literal_eval,
            )

            parsed = safe_literal_eval(raw, max_len=MATRIX_MAX_LEN)
            return np.array(parsed, dtype=float)
        except Exception:
            return default
    return default


def _declared_dimension(*values: Any, name: str) -> int | None:
    """Require consistent positive integral declarations when provided."""
    dimensions = []
    for value in values:
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
        dimensions.append(int(value))
    if len(set(dimensions)) > 1:
        raise ValueError(f"Conflicting declarations for {name}")
    return dimensions[0] if dimensions else None


def extract_abcd_matrices(
    gnn_spec: dict[str, Any],
    *,
    allow_adapters: bool = False,
    transformation_provenance: dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract A, B, C, D matrices from a discrete GNN spec.

    Validate finite nonnegative probability mass and declared shapes before
    emission. Only small declared decimal mass errors (absolute 1e-4) may be
    corrected. Missing matrices and arbitrary weights require the explicit
    demo adapter mode.

    Returns:
        Tuple ``(A, B, C, D)`` as numpy arrays.
    """
    params = gnn_spec.get("stateSpace", {}).get("parameters", {})
    if not params:
        params = gnn_spec.get("initialparameterization", {})
    if not params:
        params = gnn_spec.get("parameters", {})

    declared_states = _declared_dimension(
        gnn_spec.get("stateSpace", {}).get("size"),
        gnn_spec.get("model_parameters", {}).get("num_hidden_states"),
        name="num_hidden_states",
    )
    declared_obs = _declared_dimension(
        gnn_spec.get("observationSpace", {}).get("size"),
        gnn_spec.get("model_parameters", {}).get("num_obs"),
        name="num_obs",
    )
    num_states = declared_states if declared_states is not None else 2
    num_obs = declared_obs if declared_obs is not None else num_states

    defaults: dict[str, Any] = {}
    if allow_adapters:
        default_c = np.zeros(num_obs)
        default_c[0] = 1.0
        defaults = {
            "A": np.eye(num_obs, num_states),
            "B": np.eye(num_states),
            "C": default_c,
            "D": np.ones(num_states) / num_states,
        }

    if not allow_adapters:
        missing = [key for key in ("A", "B", "C", "D") if key not in params]
        if missing:
            raise ValueError(f"Missing declared matrices: {', '.join(missing)}")
    values = []
    adapter_defaults: set[str] = set()
    for name in ("A", "B", "C", "D"):
        value = parse_gnn_matrix_value(params.get(name), None)
        if value is None:
            if not allow_adapters:
                raise ValueError(f"{name} is not a valid numeric matrix")
            value = defaults[name]
            adapter_defaults.add(name)
        values.append(np.asarray(value, dtype=float))
    a, b, c, d = values
    if a.ndim == 2:
        if declared_states is not None and a.shape[1] != declared_states:
            raise ValueError("A columns do not match declared num_hidden_states")
        if declared_obs is not None and a.shape[0] != declared_obs:
            raise ValueError("A rows do not match declared num_obs")
    originals = {"A": a.copy(), "B": b.copy(), "C": c.copy(), "D": d.copy()}
    b_metadata: dict[str, Any] = {}
    if b.ndim == 3 and a.ndim == 2 and not allow_adapters:
        model_parameters = dict(gnn_spec.get("model_parameters", {}))
        # This direct API accepts canonical B by default; explicit source
        # declarations override that contract and must be honored before
        # validating dimensions and conditional probability mass.
        order = _declared_b_order(model_parameters)
        if not order:
            model_parameters["b_tensor_order"] = CANONICAL_B_ORDER
            order = CANONICAL_B_ORDER
        actions = model_parameters.get(
            "num_actions", b.shape[0] if order.startswith("action") else b.shape[2]
        )
        if isinstance(actions, bool) or not isinstance(actions, (int, np.integer)):
            raise ValueError("B action count must be a positive integer")
        if actions <= 0:
            raise ValueError("B action count must be a positive integer")
        canonical, b_metadata = canonicalise_b_matrix(
            b,
            num_states=a.shape[1],
            num_actions=int(actions),
            model_parameters=model_parameters,
        )
        b = np.asarray(canonical)
    valid, message = validate_abcd_shapes(a, b, c, d)
    if not valid:
        raise ValueError(message)
    if not np.isfinite(c).all():
        raise ValueError("C must contain finite preferences")
    a = np.asarray(normalise_matrix_columns(a, name="A", allow_weights=allow_adapters))
    if b.ndim == 2:
        b = np.asarray(
            normalise_matrix_columns(b, name="B", allow_weights=allow_adapters)
        )
    else:
        b = np.stack(
            [
                normalise_matrix_columns(
                    b[:, :, action],
                    name=f"B action {action}",
                    allow_weights=allow_adapters,
                )
                for action in range(b.shape[2])
            ],
            axis=2,
        )
    d = np.asarray(normalise_vector(d, name="D", allow_weights=allow_adapters))

    if transformation_provenance is not None:
        for name, emitted in (("A", a), ("B", b), ("C", c), ("D", d)):
            source = originals[name]
            transformed = name == "B" and b_metadata.get("orientation_transformed")
            if (
                name in adapter_defaults
                or transformed
                or not np.array_equal(source, emitted)
            ):
                transformation_provenance[name] = {
                    **transformation_provenance.get(name, {}),
                    **(b_metadata if name == "B" else {}),
                    "derived": True,
                    "source_values": None
                    if name in adapter_defaults
                    else source.tolist(),
                    "canonical_values": emitted.tolist(),
                    "default_supplied": name in adapter_defaults,
                    "policy": "explicit_demo_adapter"
                    if allow_adapters
                    else "declared_orientation_and_decimal_mass_tolerance",
                }

    return a, b, c, d


def format_array_literal(
    arr: np.ndarray, *, prefix: str, suffix: str = "", indent: int = 4
) -> str:
    """Format a numpy array as a ``<prefix>(...)`` code literal.

    Floats use round-trippable precision, with matching indentation;
    higher-rank arrays fall back to ``arr.tolist()``. *suffix* is appended
    inside the parentheses (e.g. ``", dtype=torch.float64"``).
    """
    body_prefix = " " * indent
    if not np.isfinite(arr).all():
        raise ValueError("Array literals must contain finite values")
    if arr.ndim == 1:
        vals = ", ".join(repr(float(v)) for v in arr)
        return f"{prefix}([{vals}]{suffix})"
    if arr.ndim == 2:
        rows = []
        for row in arr:
            vals = ", ".join(repr(float(v)) for v in row)
            rows.append(f"{body_prefix}    [{vals}]")
        inner = ",\n".join(rows)
        return f"{prefix}([\n{inner}\n{body_prefix}]{suffix})"
    return f"{prefix}({arr.tolist()}{suffix})"


__all__ = ["extract_abcd_matrices", "format_array_literal", "parse_gnn_matrix_value"]
