"""Bounded, source-bound measurements of the real Step 8 production route.

Developer measurements are separate from runtime admission and complexity
estimates. Each repetition has fresh output and a byte-frozen source corpus.
The existing subprocess envelope owns deadlines, process cleanup and sampled
process-tree RSS; the child reports process CPU separately from wall time.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import runpy
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gnn.execute.subprocess_envelope import run_subprocess_envelope
from gnn.visualization.core.process import discover_visualization_files


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: int, name: str, maximum: int | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(
            f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else "")
        )


def _artifact_census(output: Path) -> list[dict[str, Any]]:
    """Record actual byte identities and decoded PNG pixels after the timed phase."""
    from PIL import Image

    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        row: dict[str, Any] = {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                pixels = image.convert("RGBA")
                row.update(
                    width=image.width,
                    height=image.height,
                    rgba_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                )
        artifacts.append(row)
    return artifacts


def _worker(source: Path, output: Path, receipt: Path, profile: bool) -> int:
    """Execute the numbered CLI unchanged; report only completed measurements."""
    cli = Path(__file__).resolve().parents[1] / "8_visualization.py"
    sys.argv = [
        str(cli),
        "--target-dir",
        str(source),
        "--output-dir",
        str(output),
        "--recursive",
    ]
    wall, cpu = time.perf_counter(), time.process_time()
    profiler = cProfile.Profile() if profile else None
    code = 1
    try:
        if profiler:
            profiler.enable()
        runpy.run_path(str(cli), run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(receipt.with_suffix(".prof")))
        receipt.write_text(
            json.dumps(
                {
                    "phase": "step_8_cli",
                    "returncode": code,
                    "wall_seconds": time.perf_counter() - wall,
                    "process_cpu_seconds": time.process_time() - cpu,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return code


def run_visualization_benchmark(
    target_dir: Path,
    output_dir: Path,
    *,
    repetitions: int = 3,
    concurrency: int = 1,
    timeout_seconds: float = 900,
    max_source_files: int = 256,
    max_input_bytes: int = 64 * 1024 * 1024,
    profile: bool = False,
) -> dict[str, Any]:
    """Measure complete selected Step 8 runs under explicit bounded workload limits.

    The input/output limits admit this developer workload. They do not certify
    total RSS, JIT allocations, backend convergence or distributed performance.
    Existing output is refused so a cache cannot impersonate fresh rendering.
    """
    _positive_integer(repetitions, "repetitions", 5)
    _positive_integer(concurrency, "concurrency", 2)
    _positive_integer(max_source_files, "max_source_files")
    _positive_integer(max_input_bytes, "max_input_bytes")
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 1800
    ):
        raise ValueError("timeout_seconds must be finite and within (0, 1800]")
    target, output = Path(target_dir).resolve(), Path(output_dir).resolve()
    if not target.is_dir():
        raise ValueError("benchmark target_dir must be a directory")
    sources = discover_visualization_files(target, recursive=True)
    if not sources:
        raise ValueError("benchmark requires at least one selected source")
    if len(sources) > max_source_files:
        raise ValueError("selected source count exceeds max_source_files")
    if any(
        path.is_symlink() or not path.resolve().is_relative_to(target)
        for path in sources
    ):
        raise ValueError(
            "benchmark sources must be ordinary files confined to target_dir"
        )
    corpus: list[dict[str, Any]] = [
        {
            "path": path.relative_to(target).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in sources
    ]
    total_bytes = sum(row["bytes"] for row in corpus)
    if total_bytes > max_input_bytes:
        raise ValueError("selected source bytes exceed max_input_bytes")
    for path, identity in zip(sources, corpus):
        identity["sha256"] = _sha(path)
    if output.exists():
        raise FileExistsError(f"benchmark output must be fresh: {output}")
    if output.is_relative_to(target):
        raise ValueError("benchmark output must be outside the selected source tree")
    output.mkdir(parents=True)
    frozen = output / "corpus"
    for path, identity in zip(sources, corpus):
        destination = frozen / identity["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if _sha(destination) != identity["sha256"]:
            raise RuntimeError(
                f"source changed while freezing corpus: {identity['path']}"
            )
    package = Path(__file__).resolve().parents[1]
    code_files = {
        path.relative_to(package).as_posix(): _sha(path)
        for path in sorted(package.rglob("*.py"))
    }
    config = {
        "steps": [8],
        "recursive": True,
        "backend": "matplotlib/Agg",
        "python_hash_seed": 0,
        "llm_mode": "disabled",
        "repetitions": repetitions,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "profile": profile,
        "max_source_files": max_source_files,
        "max_input_bytes": max_input_bytes,
    }
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("matplotlib", "numpy", "pillow", "psutil")
        },
    }
    lock = package.parents[1] / "uv.lock"
    environment["uv_lock_sha256"] = _sha(lock) if lock.exists() else None

    def measure(index: int) -> dict[str, Any]:
        run = output / f"run_{index:02d}"
        run.mkdir()
        worker_receipt = run / "phase.json"
        command = [
            sys.executable,
            "-m",
            "gnn.analysis.performance_benchmark",
            "--worker",
            str(frozen),
            str(run / "output"),
            str(worker_receipt),
        ]
        if profile:
            command.append("--profile")
        envelope = run_subprocess_envelope(
            command,
            timeout=timeout_seconds,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
        )
        (run / "stdout.log").write_text(envelope.get("stdout", ""), encoding="utf-8")
        (run / "stderr.log").write_text(envelope.get("stderr", ""), encoding="utf-8")
        phase = (
            json.loads(worker_receipt.read_text()) if worker_receipt.exists() else None
        )
        artifacts = _artifact_census(run / "output")
        return {
            "repetition": index,
            "success": envelope["success"]
            and phase is not None
            and phase["returncode"] == 0,
            "phase": phase,
            "envelope": {
                key: value
                for key, value in envelope.items()
                if key not in {"stdout", "stderr"}
            },
            "artifact_count": len(artifacts),
            "png_artifact_count": sum(
                artifact["path"].endswith(".png") for artifact in artifacts
            ),
            "artifact_bytes": sum(row["bytes"] for row in artifacts),
            "artifacts": artifacts,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(executor.map(measure, range(repetitions)))
    times = [row["phase"]["wall_seconds"] for row in rows if row["success"]]
    cpu_times = [row["phase"]["process_cpu_seconds"] for row in rows if row["success"]]
    code_files_after = {
        path.relative_to(package).as_posix(): _sha(path)
        for path in sorted(package.rglob("*.py"))
    }
    code_identity_unchanged = code_files == code_files_after
    receipt = {
        "schema_version": "gnn.visualization_benchmark/v1",
        "configuration": config,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "corpus": corpus,
        "corpus_sha256": hashlib.sha256(
            json.dumps(corpus, sort_keys=True).encode()
        ).hexdigest(),
        "selected_source_count": len(corpus),
        "selected_input_bytes": total_bytes,
        "code_files": code_files,
        "code_identity_unchanged": code_identity_unchanged,
        "code_files_after": code_files_after,
        "environment": environment,
        "success": code_identity_unchanged and all(row["success"] for row in rows),
        "rows": rows,
        "summary": {
            "completed_repetitions": len(times),
            "wall_median_seconds": statistics.median(times) if times else None,
            "wall_min_seconds": min(times) if times else None,
            "wall_max_seconds": max(times) if times else None,
            "wall_stdev_seconds": statistics.stdev(times) if len(times) > 1 else None,
            "cpu_median_seconds": statistics.median(cpu_times) if cpu_times else None,
            "cpu_min_seconds": min(cpu_times) if cpu_times else None,
            "cpu_max_seconds": max(cpu_times) if cpu_times else None,
            "cpu_stdev_seconds": statistics.stdev(cpu_times)
            if len(cpu_times) > 1
            else None,
            "batch_wall_seconds_including_census": time.perf_counter() - started,
        },
        "measurement_limits": [
            "RSS is a sampled summed resident footprint from the subprocess envelope, not exact peak RSS or JIT/allocation certification.",
            "Child process CPU covers its threads; separate descendant CPU is not included.",
            "Fresh processes retain operating-system caches; these runs do not establish physical cold-disk behavior.",
            "Package source identities are checked before and after the batch. Source drift refuses overall acceptance while retaining each measured outcome.",
            "Artifact hashing/PNG decoding is outside the timed phase but included in batch wall time.",
            "Only the complete selected Step 8 corpus and declared backend/LLM mode are measured; other backends and distributed transfer require separate witnesses.",
        ],
    }
    (output / "benchmark.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Developer CLI; fixed private worker mode never accepts arbitrary commands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--worker":
        if len(args) not in {4, 5} or (len(args) == 5 and args[4] != "--profile"):
            raise ValueError("invalid benchmark worker arguments")
        return _worker(Path(args[1]), Path(args[2]), Path(args[3]), len(args) == 5)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--max-source-files", type=int, default=256)
    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--profile", action="store_true")
    options = parser.parse_args(args)
    receipt = run_visualization_benchmark(**vars(options))
    print(json.dumps({"success": receipt["success"], "summary": receipt["summary"]}))
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
