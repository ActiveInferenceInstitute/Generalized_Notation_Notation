#!/usr/bin/env python3
"""
PyMDP Renderer (pymdp 1.0.0 / JAX-first).

Generates executable Python scripts that run real pymdp 1.0.0 simulations
from GNN specifications. Two shapes of output are emitted (see
``pymdp_templates.py``):

1. **Pipeline runner** — delegates to ``src.execute.pymdp.run_pymdp_simulation``
   so the script is a thin wrapper around the pipeline's tested rollout.
2. **Standalone runner** — builds a pymdp 1.0.0 ``Agent`` directly and
   performs a rollout inline. Useful for sharing a self-contained script
   with external users or running without the GNN pipeline on PYTHONPATH.

Upstream reference: https://github.com/infer-actively/pymdp
Breaking-change notes: docs/pymdp/pymdp_1_0_0_alignment_matrix.md
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gnn.parsers.common import (  # noqa: F401
    GNNInternalRepresentation,
    ParseResult,
)
from gnn.parsers.markdown_parser import MarkdownGNNParser

from .pymdp_templates import (
    generate_pipeline_runner_script,
    generate_standalone_runner_script,
)

logger = logging.getLogger(__name__)


def _ensure_initialparameterization_from_parameters(gnn_spec: Dict[str, Any]) -> None:
    """
    Markdown ``to_dict()`` stores matrices under ``parameters`` as a list of
    ``{name, value}``. PyMDP generation expects ``initialparameterization``
    (dict). Merge when the latter is absent.
    """
    existing = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if existing:
        if "initialparameterization" not in gnn_spec and isinstance(existing, dict):
            gnn_spec["initialparameterization"] = dict(existing)
        return

    raw = gnn_spec.get("parameters") or []
    if not isinstance(raw, list) or not raw:
        return

    merged: Dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        merged[str(name).strip()] = item.get("value")

    if merged:
        gnn_spec["initialparameterization"] = merged


def parse_gnn_markdown(content: str, file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse GNN markdown into a dict, or return None on failure."""
    try:
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        if getattr(result, "success", False):
            gnn_spec = result.model.to_dict()
            _ensure_initialparameterization_from_parameters(gnn_spec)
            return gnn_spec
        logger.error(
            "Failed to parse GNN file %s: %s", file_path, getattr(result, "errors", "")
        )
        return None
    except Exception as e:  # noqa: BLE001
        logger.exception("Exception parsing GNN file %s: %s", file_path, e)
        return None


def _to_clean_nested(obj: Any) -> Any:
    """Recursively coerce numpy arrays into plain JSON-safe nested lists."""
    if obj is None:
        return None
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_clean_nested(x) for x in obj]
    if isinstance(obj, (int, float, bool, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_clean_nested(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return _to_clean_nested(obj.tolist())
    return obj


def _gnn_spec_for_json_embedding(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON-serializable spec (str keys; no Path keys from parsers)."""
    raw = _to_clean_nested(gnn_spec)
    return raw if isinstance(raw, dict) else dict(gnn_spec)


def _json_to_python_literal(json_str: str) -> str:
    """Embed JSON-clean values without rewriting words inside authored strings."""
    return repr(_json.loads(json_str))


def _extract_dimensions(
    gnn_spec: Dict[str, Any], init_params: Dict[str, Any]
) -> Tuple[int, int, int]:
    """Figure out (num_obs, num_states, num_actions) from spec or matrices."""
    variables = gnn_spec.get("variables") or []
    state_vars = {
        var.get("name"): var
        for var in variables
        if isinstance(var, dict) and var.get("name") in {"A", "B", "C", "D", "E"}
    }

    num_obs = 3
    num_states = 3
    num_actions = 3

    if "A" in state_vars:
        dims = state_vars["A"].get("dimensions") or []
        if len(dims) >= 2:
            num_obs = int(dims[0])
            num_states = int(dims[1])

    if "B" in state_vars:
        dims = state_vars["B"].get("dimensions") or []
        if len(dims) >= 3:
            num_actions = int(dims[2])

    model_params = gnn_spec.get("model_parameters") or {}

    # Fallback to initial_params where ModelParameters are merged by MarkdownGNNParser
    num_states = int(
        model_params.get(
            "num_hidden_states", init_params.get("num_hidden_states", num_states)
        )
    )
    num_obs = int(model_params.get("num_obs", init_params.get("num_obs", num_obs)))
    num_actions = int(
        model_params.get("num_actions", init_params.get("num_actions", num_actions))
    )

    A_raw = init_params.get("A")
    if A_raw is not None:
        arr = np.asarray(A_raw)
        if arr.ndim == 2:
            num_obs, num_states = int(arr.shape[0]), int(arr.shape[1])

    B_raw = init_params.get("B")
    if B_raw is not None:
        arr = np.asarray(B_raw)
        if arr.ndim == 3:
            # Canonical order is (next_state, previous_state, action) =
            # pymdp 1.0.0 B[s',s,a] (see docs/pymdp/pymdp_1_0_0_alignment_matrix.md).
            # Prefer matrix_provenance canonical_order when present: it marks
            # a tensor already in canonical order, so the action axis is
            # last. The shape match below remains the documented fallback
            # for specs without provenance.
            provenance_b = (gnn_spec.get("matrix_provenance") or {}).get("B") or {}
            if provenance_b.get("canonical_order") or provenance_b.get("source_order"):
                num_actions = int(arr.shape[-1])
            elif arr.shape[0] == num_states and arr.shape[1] == num_states:
                num_actions = int(arr.shape[2])
            else:
                num_actions = int(arr.shape[0])
        elif arr.ndim == 2:
            num_actions = 1
    return int(num_obs), int(num_states), int(max(num_actions, 1))


class PyMDPRenderer:
    """
    GNN → pymdp 1.0.0 (JAX-first) code generator.

    Parameters
    ----------
    options
        * ``mode`` — ``"pipeline"`` (default) or ``"standalone"``.
          ``"pipeline"`` emits a thin runner that calls
          ``src.execute.pymdp.run_pymdp_simulation``.
          ``"standalone"`` emits a self-contained pymdp 1.0.0 script.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the instance."""
        self.options = options or {}
        self.mode = str(self.options.get("mode", "pipeline")).lower()
        if self.mode not in {"pipeline", "standalone"}:
            self.mode = "pipeline"
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """Render file."""
        try:
            with open(gnn_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            gnn_spec = parse_gnn_markdown(content, gnn_file_path)
            if not gnn_spec:
                return False, f"Failed to parse GNN file: {gnn_file_path}"

            code = self._generate_code(gnn_spec, gnn_file_path.stem)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")
            self.logger.info("Generated pymdp 1.0.0 runner: %s", output_path)
            return True, "Successfully generated pymdp 1.0.0 simulation script"
        except Exception as e:  # noqa: BLE001
            msg = f"Error rendering {gnn_file_path}: {e}"
            self.logger.exception(msg)
            return False, msg

    def render_spec(
        self, gnn_spec: Dict[str, Any], output_path: Path
    ) -> Tuple[bool, str, List[str]]:
        """Render spec."""
        try:
            model_name = gnn_spec.get("model_name", "GNN_Model")
            code = self._generate_code(gnn_spec, model_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding="utf-8")

            warnings: List[str] = []
            if not gnn_spec.get("model_parameters"):
                warnings.append("No model parameters found — using inferred dimensions")
            return True, f"Generated pymdp 1.0.0 runner: {output_path}", warnings
        except Exception as e:  # noqa: BLE001
            msg = f"Error rendering spec to {output_path}: {e}"
            self.logger.exception(msg)
            return False, msg, []

    def render_directory(
        self, output_dir: Path, input_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Render directory."""
        results: Dict[str, Any] = {
            "rendered_files": [],
            "failed_files": [],
            "total_files": 0,
            "successful_renders": 0,
        }
        if input_dir:
            gnn_files = list(input_dir.glob("*.md")) + list(input_dir.glob("*.gnn"))
        else:
            gnn_files = list(Path("input/gnn_files").glob("*.md"))
        results["total_files"] = len(gnn_files)

        for gnn_file in gnn_files:
            out = output_dir / f"{gnn_file.stem}_pymdp_simulation.py"
            ok, msg = self.render_file(gnn_file, out)
            if ok:
                results["rendered_files"].append(
                    {
                        "input_file": str(gnn_file),
                        "output_file": str(out),
                        "message": msg,
                    }
                )
                results["successful_renders"] += 1
            else:
                results["failed_files"].append(
                    {"input_file": str(gnn_file), "error": msg}
                )
        return results

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    def _generate_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate code."""
        from gnn.render.execution_contracts import unsupported_contract

        reason = unsupported_contract(gnn_spec, "pymdp")
        if reason:
            raise ValueError(reason)
        model_display_name = gnn_spec.get("model_name", model_name)
        model_annotation = gnn_spec.get("annotation", "")
        init_params = gnn_spec.get("initialparameterization") or gnn_spec.get(
            "initial_parameterization", {}
        )
        nonstationary_keys = [
            str(key) for key in init_params if str(key).startswith(("B_t", "B_regime"))
        ]
        is_nonstationary = bool(nonstationary_keys)
        if is_nonstationary and self.mode == "standalone":
            raise ValueError(
                "unsupported-nonstationary: the standalone pymdp runner builds "
                "a single static Agent and cannot express time-indexed (B_t) "
                "or regime-switched (B_regime) transitions; render with the "
                "default pipeline mode, whose runner delegates to "
                "run_pymdp_simulation's per-step switching rollout"
            )

        num_obs, num_states, num_actions = _extract_dimensions(gnn_spec, init_params)

        A_matrix = _to_clean_nested(init_params.get("A"))
        B_matrix = _to_clean_nested(init_params.get("B"))
        C_vector = _to_clean_nested(init_params.get("C"))
        D_vector = _to_clean_nested(init_params.get("D"))
        E_vector = _to_clean_nested(init_params.get("E"))

        missing = [
            label
            for label, val in (
                ("A", A_matrix),
                ("B", B_matrix),
                ("C", C_vector),
                ("D", D_vector),
            )
            if val is None and not (label == "B" and is_nonstationary)
        ]
        if missing:
            raise ValueError(
                f"Missing required PyMDP initial parameterization entries: {missing}"
            )

        context: Dict[str, Any] = {
            "model_name": model_name,
            "model_display_name": model_display_name,
            "model_annotation": model_annotation,
            "num_obs": num_obs,
            "num_states": num_states,
            "num_actions": num_actions,
            "A_literal": _json.dumps(A_matrix) if A_matrix is not None else "None",
            "B_literal": _json.dumps(B_matrix) if B_matrix is not None else "None",
            "C_literal": _json.dumps(C_vector) if C_vector is not None else "None",
            "D_literal": _json.dumps(D_vector) if D_vector is not None else "None",
            "E_literal": _json.dumps(E_vector) if E_vector is not None else "None",
            "gnn_spec_literal": _json_to_python_literal(
                _json.dumps(
                    _gnn_spec_for_json_embedding(gnn_spec),
                    indent=4,
                    default=str,
                    allow_nan=False,
                )
            ),
            "num_timesteps": int(
                (gnn_spec.get("model_parameters") or {}).get(
                    "num_timesteps", init_params.get("num_timesteps", 20)
                )
            ),
        }

        if self.mode == "standalone":
            return generate_standalone_runner_script(context)
        return generate_pipeline_runner_script(context)


def render_gnn_to_pymdp(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Public entry point for the render pipeline (Step 11)."""
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "pymdp")
    if refusal:
        return False, refusal, []

    try:
        renderer = PyMDPRenderer(options)
        return renderer.render_spec(gnn_spec, output_path)
    except Exception as e:  # noqa: BLE001
        return False, f"Error generating pymdp 1.0.0 script: {e}", []
