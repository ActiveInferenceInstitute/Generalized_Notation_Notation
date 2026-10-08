"""
Code-generation templates for pymdp 1.0.0 runner scripts.

pymdp 1.0.0 is the JAX-first rewrite
(https://github.com/infer-actively/pymdp). Any script produced here uses the
new API exclusively:

    from jax import numpy as jnp, random as jr
    from pymdp.agent import Agent
    # list[jax.Array] models, Agent.infer_states(empirical_prior=...),
    # Agent.infer_policies(qs), Agent.sample_action(q_pi, rng_key=...)

There are two generators:

* ``generate_pipeline_runner_script`` — emits a thin runner that delegates to
  ``execute.pymdp.run_pymdp_simulation`` so the generated script
  shares the pipeline's tested rollout code.
* ``generate_standalone_runner_script`` — emits a fully self-contained
  pymdp 1.0.0 script usable outside the GNN pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

_PIPELINE_RUNNER_DOCSTRING = """
pymdp 1.0.0 runner for {model_display_name}

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        {model_display_name}
Description:  {model_annotation}

State Space:
  - Hidden States: {num_states}
  - Observations:  {num_obs}
  - Actions:       {num_actions}

Initial matrices present in GNN spec:
  - A (likelihood):   {A_present}
  - B (transitions):  {B_present}
  - C (preferences):  {C_present}
  - D (state prior):  {D_present}
  - E (policy prior): {E_present}
"""

_PIPELINE_RUNNER_TEMPLATE = '''#!/usr/bin/env python3
{module_docstring_literal}
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Script directory name 'pymdp' would shadow the installed library — drop it
# ---------------------------------------------------------------------------
if sys.path and sys.path[0] and sys.path[0].endswith("pymdp"):
    sys.path.pop(0)

# ---------------------------------------------------------------------------
# Repository root resolution (prefer GNN_PROJECT_ROOT; else walk upwards)
# ---------------------------------------------------------------------------
_gnn_root = os.environ.get("GNN_PROJECT_ROOT")
if _gnn_root:
    _repo = Path(_gnn_root).resolve()
    sys.path.insert(0, str(_repo / "src"))
else:
    _cur = Path(__file__).resolve().parent
    _found = None
    for _ in range(24):
        if (_cur / "pyproject.toml").is_file() and (_cur / "src").is_dir():
            _found = _cur
            break
        if _cur.parent == _cur:
            break
        _cur = _cur.parent
    if _found is None:
        print(
            "ERROR: Cannot locate GNN repository root. Run via the pipeline "
            "execute step, or set GNN_PROJECT_ROOT to the checkout root.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.path.insert(0, str(_found / "src"))

# ---------------------------------------------------------------------------
# pymdp 1.0.0 presence check (hard requirement)
# ---------------------------------------------------------------------------
try:
    import pymdp  # noqa: F401
    from pymdp.agent import Agent  # noqa: F401
    if not hasattr(Agent, "update_empirical_prior"):
        raise ImportError("unsupported pymdp (<1.0.0) detected")
    print("PyMDP 1.0.0+ detected (JAX-first Agent).")
except ImportError as e:
    print(
        "ERROR: pymdp 1.0.0 required. Install with: "
        "uv pip install 'inferactively-pymdp>=1.0.0' (original error: "
        + str(e) + ")",
        file=sys.stderr,
    )
    sys.exit(1)

from gnn.execute.pymdp import execute_pymdp_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run a pymdp 1.0.0 simulation for the GNN model embedded in this file."""
    # Matrices embedded verbatim from the GNN spec.
    A_data = {A_literal}
    B_data = {B_literal}
    C_data = {C_literal}
    D_data = {D_literal}
    E_data = {E_literal}

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {gnn_spec_literal}
    gnn_spec.setdefault("initialparameterization", {{}})
    if A_data is not None: gnn_spec["initialparameterization"]["A"] = A_data
    if B_data is not None: gnn_spec["initialparameterization"]["B"] = B_data
    if C_data is not None: gnn_spec["initialparameterization"]["C"] = C_data
    if D_data is not None: gnn_spec["initialparameterization"]["D"] = D_data
    if E_data is not None: gnn_spec["initialparameterization"]["E"] = E_data
    gnn_spec.setdefault("model_parameters", {{}})
    gnn_spec["model_parameters"].setdefault("num_timesteps", {num_timesteps})

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", {output_path_literal}))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info({rollout_message_literal})
    logger.info("Output directory: %s", output_dir)

    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            correlation_id="render_generated_script",
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        logger.error("Unexpected error: %s", exc)
        traceback.print_exc()
        return 1

    if success:
        logger.info("Simulation completed successfully")
        logger.info("  framework:    %s", results.get("framework"))
        logger.info("  pymdp ver:    %s", results.get("pymdp_version"))
        logger.info("  backend:      %s", results.get("backend"))
        logger.info("  num_timesteps:%s", results.get("num_timesteps"))
        return 0

    logger.error("Simulation failed: %s", results.get("error", "Unknown error"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


_STANDALONE_RUNNER_DOCSTRING = """
Self-contained pymdp 1.0.0 rollout for {model_display_name}

Generated from a GNN specification. This script is fully standalone: it does
not depend on the GNN pipeline code, only on ``inferactively-pymdp>=1.0.0``
and its JAX/equinox runtime.

Model:        {model_display_name}
Description:  {model_annotation}
"""

_STANDALONE_RUNNER_TEMPLATE = """#!/usr/bin/env python3
{module_docstring_literal}
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp.agent import Agent
    if not hasattr(Agent, "update_empirical_prior"):
        raise ImportError("unsupported pymdp (<1.0.0) detected")
except ImportError as e:
    print(
        "ERROR: pymdp 1.0.0 required. Install with: "
        "uv pip install 'inferactively-pymdp>=1.0.0' (original error: "
        + str(e) + ")",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numpy-based normalisation helpers (GNN → pymdp 1.0.0 input contract)
# ---------------------------------------------------------------------------
def _norm_vec(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).flatten()
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        n = max(len(arr), 1)
        return np.ones(n, dtype=np.float64) / n
    return arr / total


def _norm_cols(mat: np.ndarray) -> np.ndarray:
    out = np.asarray(mat, dtype=np.float64).copy()
    cs = out.sum(axis=0, keepdims=True)
    zero = cs <= 0
    cs = np.where(zero, 1.0, cs)
    out = out / cs
    if zero.any():
        rows = out.shape[0]
        for j in np.where(zero.flatten())[0]:
            out[:, j] = 1.0 / rows
    return out


def _canonical_B(raw: np.ndarray, num_states: int, num_actions: int) -> np.ndarray:
    if raw.ndim == 2:
        tensor = raw[:, :, None]
    elif raw.ndim == 3:
        if raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
            tensor = raw.transpose(2, 1, 0)
        else:
            tensor = raw
    else:
        raise ValueError("B matrix must be 2D or 3D")
    tensor = tensor.copy()
    for a in range(tensor.shape[2]):
        tensor[:, :, a] = _norm_cols(tensor[:, :, a])
    return tensor


def _to_batched_jax(mat: np.ndarray, batch: int = 1):
    arr = jnp.asarray(mat, dtype=jnp.float32)
    if batch == 1:
        return arr[None, ...]
    return jnp.broadcast_to(arr[None, ...], (batch, *arr.shape))


def main() -> int:
    # -----------------------------------------------------------------------
    # Model matrices embedded from GNN
    # -----------------------------------------------------------------------
    A_data = {A_literal}
    B_data = {B_literal}
    C_data = {C_literal}
    D_data = {D_literal}
    E_data = {E_literal}

    num_actions_default = {num_actions}
    num_timesteps = {num_timesteps}

    # Build canonical numpy matrices
    if A_data is None:
        raise ValueError("A matrix is required for pymdp execution")
    A_np = np.asarray(A_data, dtype=np.float64)
    A_np = _norm_cols(A_np)
    num_obs, num_states = A_np.shape

    if B_data is None:
        raise ValueError("B matrix is required for pymdp execution")
    B_np = _canonical_B(np.asarray(B_data, dtype=np.float64), num_states, num_actions_default)
    num_actions = B_np.shape[2]

    if C_data is None:
        raise ValueError("C vector is required for pymdp execution")
    c = np.asarray(C_data, dtype=np.float64).flatten()
    C_np = np.zeros(num_obs, dtype=np.float64)
    C_np[: min(num_obs, len(c))] = c[: min(num_obs, len(c))]

    if D_data is None:
        raise ValueError("D vector is required for pymdp execution")
    D_np = _norm_vec(np.asarray(D_data, dtype=np.float64))
    if D_np.shape[0] != num_states:
        padded = np.ones(num_states, dtype=np.float64) / num_states
        k = min(num_states, D_np.shape[0])
        padded[:k] = D_np[:k]
        D_np = _norm_vec(padded)

    E_np: Optional[np.ndarray] = None
    if E_data is not None:
        E_np = _norm_vec(np.asarray(E_data, dtype=np.float64))

    logger.info("Dims: No=%d Ns=%d Nu=%d T=%d", num_obs, num_states, num_actions, num_timesteps)

    # -----------------------------------------------------------------------
    # Build pymdp 1.0.0 Agent
    # -----------------------------------------------------------------------
    A_list = [_to_batched_jax(A_np)]
    B_list = [_to_batched_jax(B_np)]
    C_list = [_to_batched_jax(C_np)]
    D_list = [_to_batched_jax(D_np)]
    kwargs: Dict[str, Any] = dict(
        A=A_list,
        B=B_list,
        C=C_list,
        D=D_list,
        num_controls=[num_actions],
        policy_len=1,
        batch_size=1,
    )
    if num_actions > 1:
        kwargs["control_fac_idx"] = [0]
    if E_np is not None:
        kwargs["E"] = _to_batched_jax(E_np)
    agent = Agent(**kwargs)
    logger.info("pymdp 1.0.0 Agent built")

    # -----------------------------------------------------------------------
    # Rollout
    # -----------------------------------------------------------------------
    rng_np = np.random.default_rng(0)
    jax_key = jr.PRNGKey(0)
    true_state = int(rng_np.choice(num_states, p=D_np))
    empirical_prior = agent.D

    observations: List[int] = []
    actions: List[int] = []
    beliefs: List[List[float]] = []
    true_states: List[int] = [true_state]

    for t in range(num_timesteps):
        obs_idx = int(rng_np.choice(num_obs, p=_norm_vec(A_np[:, true_state])))
        observations.append(obs_idx)

        obs_jax = [jnp.array([obs_idx], dtype=jnp.int32)]
        qs, info = agent.infer_states(obs_jax, empirical_prior=empirical_prior, return_info=True)
        belief_vec = np.asarray(qs[0][0, -1], dtype=np.float64).flatten()
        beliefs.append(belief_vec.tolist())

        q_pi, neg_efe = agent.infer_policies(qs)
        jax_key, subkey = jr.split(jax_key)
        action_keys = jr.split(subkey, 2)
        action = agent.sample_action(q_pi, rng_key=action_keys[1:])
        a_idx = int(np.asarray(action)[0, 0])
        actions.append(a_idx)

        true_state = int(rng_np.choice(num_states, p=_norm_vec(B_np[:, true_state, a_idx])))
        true_states.append(true_state)
        empirical_prior = agent.update_empirical_prior(action, qs)
        logger.info("t=%02d obs=%d action=%d belief=%s", t, obs_idx, a_idx, np.round(belief_vec, 3).tolist())

    results = {{
        "schema_version": "pymdp_simulation_v1",
        "framework": "PyMDP",
        "pymdp_version": "1.0.0+",
        "backend": "jax",
        "model_name": {model_display_name_literal},
        "num_timesteps": num_timesteps,
        "observations": observations,
        "actions": actions,
        "beliefs": beliefs,
        "true_states": true_states,
        "simulation_trace": {{
            "observations": observations,
            "actions": actions,
            "beliefs": beliefs,
            "true_states": true_states,
        }},
        "success": True,
    }}

    out_dir = Path({output_path_literal})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "simulation_results.json").write_text(json.dumps(results, indent=2))
    logger.info("Done — wrote %s/simulation_results.json", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def _flag(label: str, value: Any) -> str:
    """Handle flag for internal callers."""
    return "Present" if value is not None and value != "None" else "Missing"


def generate_pipeline_runner_script(ctx: Dict[str, Any]) -> str:
    """Render the pipeline runner template with the given GNN context."""
    context = dict(ctx)
    context.update(
        A_present=_flag("A", ctx.get("A_literal")),
        B_present=_flag("B", ctx.get("B_literal")),
        C_present=_flag("C", ctx.get("C_literal")),
        D_present=_flag("D", ctx.get("D_literal")),
        E_present=_flag("E", ctx.get("E_literal")),
    )
    context["module_docstring_literal"] = repr(
        _PIPELINE_RUNNER_DOCSTRING.format(**context)
    )
    context["output_path_literal"] = repr(
        f"output/pymdp_simulations/{ctx['model_name']}"
    )
    context["rollout_message_literal"] = repr(
        f"Running pymdp 1.0.0 rollout for {ctx['model_display_name']}"
    )
    return _PIPELINE_RUNNER_TEMPLATE.format(**context)


def generate_standalone_runner_script(ctx: Dict[str, Any]) -> str:
    """Render the standalone runner template with the given GNN context."""
    context = dict(ctx)
    context["module_docstring_literal"] = repr(
        _STANDALONE_RUNNER_DOCSTRING.format(**context)
    )
    context["output_path_literal"] = repr(
        f"output/pymdp_simulations/{ctx['model_name']}"
    )
    context["model_display_name_literal"] = repr(str(ctx["model_display_name"]))
    return _STANDALONE_RUNNER_TEMPLATE.format(**context)


# ---------------------------------------------------------------------------
# Helpers that build string fragments for the pymdp 1.0.0 JAX API.
# ---------------------------------------------------------------------------
def generate_file_header(model_name: str) -> str:
    """Return a short module docstring header for a generated script."""
    return (
        "#!/usr/bin/env python3\n"
        f'"""pymdp 1.0.0 agent script — {model_name} (generated by GNN renderer)."""\n'
        "from __future__ import annotations\n"
        "import jax.numpy as jnp\n"
        "import jax.random as jr\n"
        "from pymdp.agent import Agent\n"
    )


def generate_conversion_summary(log_entries: List[str]) -> str:
    """Generate conversion summary."""
    lines = "\n".join(f"# {entry}" for entry in log_entries)
    return f"\n# --- GNN to pymdp 1.0.0 conversion summary ---\n{lines}\n# --- end summary ---\n"


def generate_example_usage_template(
    model_name: str,
    num_modalities: int,
    num_factors: int,
    control_factor_indices: List[int],
    sim_timesteps: int = 5,
    **_: Any,
) -> List[str]:
    """
    Produce a minimal pymdp 1.0.0 rollout snippet as a list of code lines.

    The lines below use the canonical JAX-first pattern with rng_key and
    empirical_prior arguments.
    """
    lines: List[str] = [
        "",
        "# --- pymdp 1.0.0 rollout ---",
        "if __name__ == '__main__':",
        "    import numpy as np",
        "    import jax.numpy as jnp",
        "    import jax.random as jr",
        "",
        f"    num_modalities = {num_modalities}",
        f"    num_factors = {num_factors}",
        f"    control_fac_idx = {control_factor_indices}",
        f"    T = {sim_timesteps}",
        "",
        "    rng_np = np.random.default_rng(0)",
        "    jax_key = jr.PRNGKey(0)",
        "    empirical_prior = agent.D",
        "",
        "    for t in range(T):",
        "        # Single-modality observation sample for generated examples",
        "        obs = [jnp.array([0], dtype=jnp.int32) for _ in range(num_modalities)]",
        "        qs, info = agent.infer_states(obs, empirical_prior=empirical_prior, return_info=True)",
        "        q_pi, neg_efe = agent.infer_policies(qs)",
        "        jax_key, subkey = jr.split(jax_key)",
        "        action_keys = jr.split(subkey, agent.batch_size + 1)",
        "        action = agent.sample_action(q_pi, rng_key=action_keys[1:])",
        "        print(f\"t={t} action={action} vfe={float(info['vfe'].mean()):.3f}\")",
        "        empirical_prior = agent.update_empirical_prior(action, qs)",
        "",
        f"    print('Rollout finished: {model_name}')",
    ]
    return lines
