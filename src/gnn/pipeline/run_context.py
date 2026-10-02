"""Immutable, byte-bound model selection shared by every pipeline executor.

The input view copies exact source bytes into a private per-step directory.
This supports shallow standalone consumers without widening a matrix selection or
changing scientific input. Original paths and hashes remain in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from gnn.pipeline._io import atomic_write_text

CONTEXT_ENV = "GNN_RUN_CONTEXT_FILE"


def _source_bytes(path: Path, deadline: float | None) -> bytes:
    """Read a source with deadline checks between bounded chunks."""
    chunks = []
    with path.open("rb") as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Pipeline total deadline expired")
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    data = b"".join(chunks)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Pipeline total deadline expired")
    return data


@dataclass(frozen=True)
class ModelRef:
    """Original source identity and collision-safe artifact stem."""

    source_path: str
    relative_path: str
    model_id: str
    sha256: str
    artifact_stem: str
    steps: tuple[int, ...]


@dataclass(frozen=True)
class RunContext:
    """Resolved invocation configuration, selected inputs, and absolute deadline."""

    run_id: str
    input_root: str
    output_root: str
    config_json: str
    frameworks: tuple[str, ...]
    selected_steps: tuple[int, ...]
    models: tuple[ModelRef, ...]
    deadline_monotonic: float | None = None
    exclusions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.run_id):
            raise ValueError("run_id must be a filesystem-safe invocation identifier")
        if self.deadline_monotonic is not None and (
            isinstance(self.deadline_monotonic, bool)
            or not isinstance(self.deadline_monotonic, (int, float))
            or not math.isfinite(self.deadline_monotonic)
        ):
            raise ValueError("Run deadline must be finite")
        if len({model.model_id for model in self.models}) != len(self.models):
            raise ValueError("Duplicate model identities in run context")
        if len({model.artifact_stem for model in self.models}) != len(self.models):
            raise ValueError("Duplicate artifact stems in run context")
        for model in self.models:
            relative = Path(model.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid model relative path")
            if Path(model.artifact_stem).name != model.artifact_stem:
                raise ValueError("Invalid artifact stem")
            if (
                Path(model.source_path).resolve()
                != (Path(self.input_root) / relative).resolve()
            ):
                raise ValueError(
                    "Model source does not match its original relative path"
                )
            if (
                not Path(model.source_path)
                .resolve()
                .is_relative_to(Path(self.input_root).resolve())
            ):
                raise ValueError("Model source escapes input root")

    @property
    def input_config(self) -> dict[str, Any]:
        """Return a fresh configuration copy; callers cannot mutate the snapshot."""
        return dict(json.loads(self.config_json))

    def selected_models(self, step_number: int) -> tuple[ModelRef, ...]:
        """Return the exact model set authorized for a step."""
        return tuple(model for model in self.models if step_number in model.steps)

    def remaining_seconds(self) -> float | None:
        """Return remaining monotonic wall-clock budget, or None when unbounded."""
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def bounded_timeout(self, seconds: float) -> float:
        """Intersect a local budget with the invocation deadline."""
        remaining = self.remaining_seconds()
        return seconds if remaining is None else min(seconds, remaining)

    def process_deadlines(
        self, local_timeout: float | None, cleanup_seconds: float = 1.0
    ) -> tuple[float | None, float, float | None]:
        """Reserve supervised cleanup inside this invocation's absolute budget."""
        from gnn.utils.runtime_safety.process_budget import resolve_process_deadlines

        return resolve_process_deadlines(
            local_timeout,
            deadline_monotonic=self.deadline_monotonic,
            cleanup_seconds=cleanup_seconds,
        )

    def verify_sources(self) -> None:
        """Fail closed when the original source set changes after planning."""
        for model in self.models:
            self.raise_if_expired()
            if (
                hashlib.sha256(
                    _source_bytes(Path(model.source_path), self.deadline_monotonic)
                ).hexdigest()
                != model.sha256
            ):
                raise ValueError(
                    f"Model source changed after run planning: {model.relative_path}"
                )

    def raise_if_expired(self) -> None:
        """Stop planning, copying, and finalization work after the shared deadline."""
        if self.remaining_seconds() == 0:
            raise TimeoutError("Pipeline total deadline expired")

    def write(self, path: Path) -> Path:
        """Save the immutable snapshot atomically."""
        atomic_write_text(path, json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def read(cls, path: Path) -> RunContext:
        """Read and validate an invocation snapshot."""
        value = json.loads(path.read_text(encoding="utf-8"))
        value["models"] = tuple(
            ModelRef(**{**model, "steps": tuple(model["steps"])})
            for model in value["models"]
        )
        value["frameworks"] = tuple(value["frameworks"])
        value["selected_steps"] = tuple(value["selected_steps"])
        value["exclusions"] = tuple(
            tuple(record) for record in value.get("exclusions", ())
        )
        context = cls(**value)
        if len({m.model_id for m in context.models}) != len(context.models):
            raise ValueError("Duplicate model identities in run context")
        return context

    def input_view(self, step_number: int) -> Path:
        """Materialize verified bytes for one step, preserving a fixed selection."""
        selected = self.selected_models(step_number)
        fingerprint = hashlib.sha256(
            json.dumps([asdict(model) for model in selected], sort_keys=True).encode()
        ).hexdigest()[:24]
        view = (
            Path(self.output_root)
            / "00_pipeline_summary"
            / "selection"
            / self.run_id
            / fingerprint
            / str(step_number)
        )
        view.mkdir(parents=True, exist_ok=True)
        expected = {
            f"{model.artifact_stem}{Path(model.source_path).suffix}"
            for model in selected
        }
        if any(
            path.name not in expected or not path.is_file() for path in view.iterdir()
        ):
            raise ValueError("Unexpected file in immutable model input view")
        for model in selected:
            self.raise_if_expired()
            source = Path(model.source_path)
            data = _source_bytes(source, self.deadline_monotonic)
            if hashlib.sha256(data).hexdigest() != model.sha256:
                raise ValueError(
                    f"Model source changed after run planning: {model.relative_path}"
                )
            destination = view / f"{model.artifact_stem}{source.suffix}"
            # No generated model text: this is an exact copy of the bound bytes.
            destination.write_bytes(data)
        return view


def build_run_context(
    target_dir: Path,
    output_dir: Path,
    run_id: str,
    selected_steps: Iterable[int],
    config: dict[str, Any],
    *,
    frameworks: Iterable[str] = (),
    recursive: bool = True,
) -> RunContext:
    """Resolve source discovery and matrix selection once at invocation startup."""
    from gnn.parsers.common import get_supported_gnn_extensions
    from gnn.pipeline.step_registry import STEPS
    from gnn.processing.discovery import is_model_source_path

    started = time.monotonic()
    timeout = (config.get("pipeline") or {}).get("timeout", {}).get("total")
    deadline = None
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("pipeline.timeout.total must be a positive number or null")
        deadline = started + float(timeout)
    root = target_dir.resolve()
    extensions = set(get_supported_gnn_extensions())
    if not root.is_dir():
        raise ValueError(f"Pipeline input must be a directory: {root}")
    discovered = root.rglob("*") if recursive else root.glob("*")
    candidates = sorted(
        p for p in discovered if p.is_file() and p.suffix.lower() in extensions
    )
    files = []
    exclusions = (
        []
        if recursive
        else [
            (path.relative_to(root).as_posix(), "non_recursive_directory")
            for path in sorted(root.iterdir())
            if path.is_dir()
        ]
    )
    for source in candidates:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Pipeline total deadline expired")
        relative_source = source.relative_to(root)
        reason = (
            "documentation"
            if not is_model_source_path(source)
            else "archived"
            if "archived_gnn_files" in relative_source.parts
            else None
        )
        if reason:
            exclusions.append((relative_source.as_posix(), reason))
        elif not source.resolve().is_relative_to(root):
            raise ValueError(f"Model source escapes input root: {relative_source}")
        else:
            files.append(source)
    steps = tuple(sorted(set(selected_steps)))
    matrix = config.get("testing_matrix") or {}
    folder_rules = matrix.get("folders") or {}
    defaults = tuple(matrix.get("default_steps") or ())
    scopes = {
        int(step.script_stem.split("_")[0]): step.execution_scope for step in STEPS
    }
    stems: dict[str, int] = {}
    for source in files:
        stems[source.stem] = stems.get(source.stem, 0) + 1
    models = []
    for source in files:
        relative = source.relative_to(root).as_posix()
        digest = hashlib.sha256(relative.encode()).hexdigest()[:12]
        model_id = f"{re.sub(r'[^A-Za-z0-9_-]', '_', source.stem)}-{digest}"
        allowed = steps
        if matrix.get("enabled") and len(source.relative_to(root).parts) > 1:
            configured = folder_rules.get(source.relative_to(root).parts[0], defaults)
            # Environment and run-wide consumers do not acquire folder routing.
            allowed = tuple(
                step
                for step in steps
                if step in configured or scopes[step] == "run" or step in {0, 13, 22}
            )
            if allowed and all(scopes[step] == "run" for step in allowed):
                allowed = ()
                exclusions.append((relative, "matrix_selection"))
        models.append(
            ModelRef(
                str(source),
                relative,
                model_id,
                hashlib.sha256(_source_bytes(source, deadline)).hexdigest(),
                source.stem if stems[source.stem] == 1 else model_id,
                allowed,
            )
        )
    if timeout is None:
        from gnn.pipeline.step_timeouts import get_step_timeout

        budgets = {
            int(step.script_stem.split("_")[0]): get_step_timeout(
                f"{step.script_stem}.py"
            )
            for step in STEPS
        }
        budgets[13] = 600 * max(1, sum(13 in model.steps for model in models))
        deadline = started + sum(budgets[step] for step in steps)
    return RunContext(
        run_id,
        str(root),
        str(output_dir.resolve()),
        json.dumps(config, sort_keys=True, default=str),
        tuple(frameworks),
        steps,
        tuple(models),
        deadline,
        tuple(exclusions),
    )


def model_provenance(path: Path, step_number: int) -> dict[str, str]:
    """Bind a staged source or artifact path to its original selected model."""
    context = current_run_context()
    if context is None:
        return {}
    for model in context.selected_models(step_number):
        if (
            path.resolve() == Path(model.source_path).resolve()
            or path.stem == model.artifact_stem
            or model.artifact_stem in path.parts
        ):
            return {
                "model_id": model.model_id,
                "source_path": model.source_path,
                "source_relative_path": model.relative_path,
                "source_sha256": model.sha256,
            }
    return {}


def current_run_context() -> RunContext | None:
    """Read the explicit run context; standalone calls retain their own discovery."""
    path = os.environ.get(CONTEXT_ENV)
    if not path:
        return None
    context = RunContext.read(Path(path))
    if context.run_id != os.environ.get("GNN_RUN_ID"):
        raise ValueError("Run context identity does not match GNN_RUN_ID")
    return context


def effective_input_config() -> dict[str, Any] | None:
    """Return the top-level configuration snapshot for child processors."""
    context = current_run_context()
    return context.input_config if context else None


def selected_model_sources(
    target_dir: Path,
    step_number: int,
    *,
    recursive: bool = True,
    extensions: Iterable[str] | None = None,
) -> list[Path]:
    """Discover standalone inputs or project the current immutable selection."""
    from gnn.processing.discovery import is_model_source_path

    context = current_run_context()
    suffixes = set(extensions or (".md",))
    if context:
        view = context.input_view(step_number)
        return [
            view / f"{model.artifact_stem}{Path(model.source_path).suffix}"
            for model in context.selected_models(step_number)
            if Path(model.source_path).suffix.lower() in suffixes
        ]
    candidates = target_dir.rglob("*") if recursive else target_dir.glob("*")
    return sorted(
        p
        for p in candidates
        if p.is_file()
        and p.suffix.lower() in suffixes
        and is_model_source_path(p)
        and "archived_gnn_files" not in p.parts
    )
