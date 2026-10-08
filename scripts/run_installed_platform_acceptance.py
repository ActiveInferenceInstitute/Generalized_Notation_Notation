#!/usr/bin/env python3
"""Accept an ordinary installed wheel from an isolated, declared data root.

This is an external acceptance driver, not pipeline domain implementation.
Run with ``python -I`` after installing the frozen exported runtime lock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def accept(
    checkout: Path, root: Path, model: Path, *, api: bool = True
) -> dict[str, Any]:
    """Exercise installed imports, native execution and negative outcomes."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    import gnn
    from gnn.execute.subprocess_envelope import CancelToken, run_subprocess_envelope
    from gnn.render.jax import render_gnn_to_jax
    from gnn.utils.runtime_safety.jax_stack_validation import verify_jax_pymdp_stack

    installed = Path(gnn.__file__).resolve()
    _require(not installed.is_relative_to(checkout), "checkout shadows installed wheel")
    _require(
        not Path.cwd().resolve().is_relative_to(checkout),
        "working directory is in checkout",
    )
    modules = (
        "numpy",
        "matplotlib",
        "pandas",
        "h5py",
        "numpyro",
        "flax",
        "equinox",
        "optax",
        "pymdp",
        "discopy",
        "fastapi",
        "thrml",
    )
    versions: dict[str, str] = {}
    for module in modules:
        importlib.import_module(module)
        versions[module] = importlib.metadata.version(
            "inferactively-pymdp" if module == "pymdp" else module
        )
    versions["jax"] = importlib.metadata.version("jax")
    versions["jaxlib"] = importlib.metadata.version("jaxlib")
    verify_jax_pymdp_stack()
    np.testing.assert_array_equal(jax.jit(lambda x: x @ x)(jnp.eye(2)), np.eye(2))

    source = {
        "model_name": "installed asymmetric categorical acceptance",
        "initialparameterization": {
            "A": [[0.8, 0.3], [0.2, 0.7]],
            "B": [[[0.8], [0.3]], [[0.2], [0.7]]],
            "C": [0.2, -0.1],
            "D": [0.6, 0.4],
        },
        "model_parameters": {
            "num_states": 2,
            "num_obs": 2,
            "num_actions": 1,
            "num_timesteps": 3,
            "b_tensor_order": "next_state_previous_state_action",
        },
    }
    script = root / "native_jax.py"
    success, reason, _ = render_gnn_to_jax(source, script, {"num_timesteps": 3})
    _require(success, reason)
    saved_script = script.read_bytes()
    _require(
        "\u221d" in saved_script.decode("utf-8"),
        "saved JAX scientific source did not preserve its Unicode in UTF-8",
    )
    native = run_subprocess_envelope(
        [sys.executable, "-I", str(script)],
        cwd=root,
        timeout=90,
        env={"GNN_OUTPUT_DIR": str(root / "native-jax")},
    )
    _require(native["success"], repr(native))
    result_path = root / "native-jax/simulation_results.json"
    result = json.loads(result_path.read_text())
    beliefs = np.asarray(result["beliefs"])
    _require(
        result["success"] and beliefs.shape == (3, 2),
        "native generated-run result rejected",
    )
    _require(
        bool(np.isfinite(beliefs).all() and (beliefs >= 0).all()),
        "invalid native probabilities",
    )
    np.testing.assert_allclose(beliefs.sum(axis=1), 1, atol=1e-5)

    # Real CLI commands run in isolated mode; package import never uses PYTHONPATH.
    cli: dict[str, int] = {}
    for command in (("validate", str(model)), ("extract", str(model), "--compact")):
        child = subprocess.run(
            [sys.executable, "-I", "-m", "gnn.cli", *command],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=45,
            check=False,
        )
        (root / f"cli-{command[0]}.log").write_text(
            child.stdout + child.stderr, encoding="utf-8"
        )
        _require(child.returncode == 0, f"CLI {command[0]} failed: {child.stderr}")
        cli[command[0]] = child.returncode
    missing = subprocess.run(
        [sys.executable, "-I", "-m", "gnn.cli", "validate", str(root / "absent.md")],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    _require(missing.returncode != 0, "missing CLI input accepted")

    from gnn.cli.templates import list_templates, pull_template, show_template

    templates = list_templates()
    _require(bool(templates), "installed template registry empty")
    template = show_template("pomdp-gridworld-3x3")
    pulled = pull_template(template["name"], root / "templates")
    _require(
        pulled["copied"] and _digest(Path(pulled["destination"])) == template["sha256"],
        "packaged template copy hash mismatch",
    )
    for command in (("templates", "list"), ("templates", "show", template["name"])):
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-W",
                "error::DeprecationWarning",
                "-m",
                "gnn.cli",
                *command,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        _require(
            child.returncode == 0, f"installed template CLI failed: {child.stderr}"
        )

    # Three concurrent allocating workers witness measured RSS, direct-worker
    # cancellation and timeout. This does not claim hostile descendant confinement.
    def worker(mode: str) -> dict[str, Any]:
        token = CancelToken()
        timer = threading.Timer(1, token.cancel, args=("installed-platform-load",))
        if mode == "cancel":
            timer.start()
        started = time.monotonic()
        try:
            outcome = run_subprocess_envelope(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import time; x=bytearray(16*1024*1024); print('allocated',flush=True); time.sleep(20)",
                ],
                cwd=root,
                timeout=2 if mode == "timeout" else 8,
                cancel_token=token,
            )
        finally:
            timer.cancel()
            if mode == "cancel":
                timer.join(timeout=2)
        _require(
            "allocated" in outcome["stdout"],
            "allocating native worker was not observed",
        )
        expected = "Cancelled" if mode == "cancel" else "TimeoutExpired"
        _require(
            not outcome["success"] and outcome.get("error_type") == expected,
            repr(outcome),
        )
        _require(
            outcome.get("cleanup_verified") is True
            and outcome.get("streams_drained") is True,
            "worker cleanup not verified",
        )
        _require(
            time.monotonic() - started < 10,
            "worker cancellation exceeded bounded allowance",
        )
        _require(
            outcome.get("rss_samples_count", 0) > 0
            and outcome.get("child_peak_rss_mb", 0) > 0,
            "worker RSS was unmeasured",
        )
        return {
            key: outcome.get(key)
            for key in (
                "error_type",
                "duration_seconds",
                "cleanup_verified",
                "streams_drained",
                "child_peak_rss_mb",
                "rss_samples_count",
                "containment",
            )
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        workers = list(pool.map(worker, ("cancel", "timeout", "cancel")))

    api_result: dict[str, Any] = {
        "accepted": False,
        "reason": "explicitly omitted local pre-integration API witness",
    }
    if api:
        os.environ["GNN_API_ROOT"] = str(root)
        from fastapi.testclient import TestClient

        import gnn.api.path_utils
        from gnn.api.app import create_app
        from gnn.api.path_utils import PathValidationError, get_repo_root

        implicit_data_root = Path(gnn.api.path_utils.__file__).resolve().parents[3]
        before_implicit = sorted(path.name for path in implicit_data_root.iterdir())
        configured_root = os.environ.pop("GNN_API_ROOT")
        try:
            try:
                get_repo_root()
            except PathValidationError:
                pass
            else:
                raise RuntimeError(
                    "unconfigured installed API accepted an implicit code directory"
                )
        finally:
            os.environ["GNN_API_ROOT"] = configured_root
        _require(
            sorted(path.name for path in implicit_data_root.iterdir())
            == before_implicit,
            "unconfigured API created storage alongside installed code",
        )

        _require(
            get_repo_root() == root,
            "installed API does not honor declared scratch root",
        )
        import gnn.model_registry

        original = gnn.model_registry.process_model_registry
        scratch_paths: list[Path] = []

        # Observe the real call without replacing its behavior or outcome.
        def observe(**kwargs: Any) -> Any:
            scratch_paths.append(Path(kwargs["output_dir"]))
            return original(**kwargs)

        gnn.model_registry.process_model_registry = observe
        try:
            with TestClient(create_app(runs_store={})) as client:

                def query(term: str) -> Any:
                    return client.get(
                        "/api/v1/models",
                        params={
                            "target_dir": str(model.parent),
                            "query_ontology": term,
                        },
                    )

                with ThreadPoolExecutor(max_workers=4) as pool:
                    responses = list(
                        pool.map(
                            query, ("", "missing-ontology-a", "", "missing-ontology-b")
                        )
                    )
                _require(
                    all(response.status_code == 200 for response in responses),
                    "concurrent HTTP registry request failed",
                )
                _require(
                    responses[0].json()["data"]["total_models"] > 0,
                    "registry snapshot empty",
                )
                _require(
                    responses[1].json()["data"]["matching_models"] == [],
                    "ontology negative case failed",
                )
                rejected = client.get(
                    "/api/v1/models", params={"target_dir": str(root.parent)}
                )
                _require(rejected.status_code == 400, "API root escape was accepted")
        finally:
            gnn.model_registry.process_model_registry = original
        _require(
            len(scratch_paths) == 4 and len(set(scratch_paths)) == 4,
            "registry requests reused scratch",
        )
        _require(
            all(not path.exists() for path in scratch_paths),
            "registry scratch retained",
        )
        api_result = {
            "accepted": True,
            "unconfigured_installed_api_rejected": True,
            "implicit_code_directory_unchanged": True,
            "concurrent_requests": 4,
            "distinct_scratch_paths": 4,
            "scratch_removed": True,
            "escape_status": rejected.status_code,
        }

    _require(
        all(
            not Path(module.__file__).resolve().is_relative_to(checkout)
            for name, module in sys.modules.items()
            if name == "gnn" or name.startswith("gnn.")
            if getattr(module, "__file__", None)
        ),
        "an installed facade imported checkout source",
    )
    return {
        "schema": "gnn_installed_platform_acceptance_v1",
        "complete_acceptance": bool(api),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "gnn_version": importlib.metadata.version("generalized-notation-notation"),
        "installed_package": str(installed),
        "ordinary_install_outside_checkout": True,
        "versions": versions,
        "native_jax": {
            "source_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest(),
            "script_sha256": _digest(script),
            "saved_source_encoding": "utf-8",
            "unicode_source_preserved": True,
            "result_sha256": _digest(result_path),
            "belief_shape": list(beliefs.shape),
            "cleanup_verified": native["cleanup_verified"],
        },
        "cli": cli,
        "templates": {
            "count": len(templates),
            "copied_template": template["name"],
            "copied_sha256": template["sha256"],
            "deprecation_errors_enabled": True,
        },
        "missing_input_rejected": True,
        "concurrent_worker_load": workers,
        "api": api_result,
        "limits": [
            "RSS is sampled resident footprint, not an exact process memory upper bound.",
            "Direct-worker load cases do not certify containment of hostile detached descendants.",
            "GUI, network providers, audio and external toolchains require separate provisioning.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--omit-api",
        action="store_true",
        help="Local pre-integration witness only; never used by the required hosted lane",
    )
    args = parser.parse_args()
    receipt = accept(
        args.checkout.resolve(),
        args.root.resolve(),
        args.model.resolve(),
        api=not args.omit_api,
    )
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "python": receipt["python"],
                "platform": receipt["platform"],
                "api": receipt["api"],
                "accepted": receipt["complete_acceptance"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
