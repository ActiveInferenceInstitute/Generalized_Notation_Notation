#!/usr/bin/env python3
"""
Audio processor module for GNN Processing Pipeline.

This module provides the main audio processing functionality.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

# Optional numpy import with recovery
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = cast(Any, None)
    NUMPY_AVAILABLE = False

from gnn.utils.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
)

from .generator import (
    generate_ambient_representation,
    generate_rhythmic_representation,
    generate_sonification_audio,
    generate_tonal_representation,
)
from .streaming import (
    chunks_from_frames,
    frames_from_execution_trace,
    write_stream_summary,
)


def process_audio(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process GNN files with audio generation and sonification.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("audio")

    try:
        log_step_start(logger, "Processing audio")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "audio_files_generated": [],
            "sonification_results": [],
            "audio_analysis": [],
            "audio_streaming": {},
        }

        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for audio processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Generate audio from GNN model
                    audio_result = generate_audio_from_gnn(
                        gnn_file, results_dir, verbose
                    )
                    results["audio_files_generated"].append(audio_result)

                    # Create sonification
                    sonification = create_sonification(gnn_file, results_dir, verbose)
                    results["sonification_results"].append(sonification)

                    # Analyze audio characteristics
                    analysis = analyze_audio_characteristics(audio_result, verbose)
                    results["audio_analysis"].append(analysis)

                except Exception as e:
                    results["success"] = False
                    error_info: dict[str, Any] = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")

        stream_summary = _process_audio_streaming(kwargs, results_dir, logger)
        if stream_summary:
            results["audio_streaming"] = stream_summary

        # Save detailed results
        results_file = results_dir / "audio_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Optionally emit a durable_streams StreamManifest sidecar for audio
        try:
            from gnn.pipeline.durable_streams import (
                StreamManifest,
                write_stream_manifest,
            )

            audio_array = np.array(
                results.get("audio_files_generated", []), dtype=object
            )
            if len(audio_array) > 0:
                audio_manifest = StreamManifest.from_array(
                    stream_id=f"audio_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    array=np.zeros((len(audio_array),), dtype=np.float32),
                    source=str(results_file),
                    chunk_size=int(kwargs.get("audio_chunk_size", 32)),
                    created_by="15_audio",
                )
                write_stream_manifest(
                    audio_manifest, results_dir / "audio_stream_manifest.json"
                )
        except Exception as manifest_err:
            logger.debug(
                "Optional durable stream manifest generation skipped: %s", manifest_err
            )

        # Generate summary report
        summary = generate_audio_summary(results)
        summary_file = results_dir / "audio_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary)

        if results["success"]:
            log_step_success(logger, "Audio processing completed successfully")
        else:
            log_step_error(logger, "Audio processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(logger, "Audio processing failed", context={"error": str(e)})
        return False


def _process_audio_streaming(
    kwargs: Dict[str, Any], output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """Generate streaming-safe audio chunk metadata from optional telemetry."""
    telemetries: List[tuple[str, Dict[str, Any]]] = []
    telemetry = kwargs.get("telemetry")
    telemetry_file = kwargs.get("telemetry_file")
    if isinstance(telemetry, dict):
        telemetries.append(("inline", telemetry))
    telemetry_files = list(kwargs.get("telemetry_files") or [])
    if telemetry_file:
        telemetry_files.append(telemetry_file)
    for path_value in telemetry_files:
        telemetry_path = Path(path_value)
        loaded = _load_telemetry_json(telemetry_path, logger)
        if loaded:
            telemetries.append((str(telemetry_path), loaded))
    execution_output_dir = kwargs.get("execution_output_dir") or kwargs.get(
        "execution_results_dir"
    )
    if not execution_output_dir:
        sibling_execution_dir = output_dir.parent / "12_execute_output"
        if sibling_execution_dir.exists():
            execution_output_dir = sibling_execution_dir
    if execution_output_dir:
        telemetries.extend(
            _load_execution_telemetry_dir(Path(execution_output_dir), logger)
        )
    if not telemetries:
        return {}
    frames = []
    provenance = []
    for source, item in telemetries:
        provenance.append(source)
        frames.extend(frames_from_execution_trace(item))
    if not frames:
        summary = write_stream_summary([], output_dir / "audio_stream_chunks.json")
        summary["telemetry_source_count"] = len(telemetries)
        summary["telemetry_provenance"] = provenance
        (output_dir / "audio_stream_chunks.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    chunks = chunks_from_frames(
        frames, chunk_size=int(kwargs.get("audio_chunk_size", 32))
    )
    summary = write_stream_summary(chunks, output_dir / "audio_stream_chunks.json")
    summary["telemetry_source_count"] = len(telemetries)
    summary["telemetry_provenance"] = provenance
    (output_dir / "audio_stream_chunks.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _load_telemetry_json(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load one telemetry JSON file if it contains an object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read audio telemetry file %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_execution_telemetry_dir(
    execution_output_dir: Path, logger: logging.Logger
) -> List[tuple[str, Dict[str, Any]]]:
    """Load Step 12 result JSON files that contain telemetry-like payloads."""
    if not execution_output_dir.exists():
        logger.warning("Execution output directory not found: %s", execution_output_dir)
        return []
    telemetries: List[tuple[str, Dict[str, Any]]] = []
    for path in sorted(execution_output_dir.rglob("*.json")):
        payload = _load_telemetry_json(path, logger)
        if not payload:
            continue
        if path.name == "execution_summary.json":
            telemetries.extend(
                _load_execution_summary_telemetries(
                    payload, path.parent, execution_output_dir, logger
                )
            )
        if any(
            key in payload
            for key in (
                "simulation_data",
                "telemetry",
                "free_energy",
                "expected_free_energy",
                "beliefs",
                "actions",
            )
        ):
            telemetries.append((str(path), payload))
    return _dedupe_telemetries(telemetries)


def _load_execution_summary_telemetries(
    summary: Dict[str, Any],
    summary_dir: Path,
    execution_output_dir: Path,
    logger: logging.Logger,
) -> List[tuple[str, Dict[str, Any]]]:
    """Follow Step 12 slim-summary pointers to structured result JSON payloads."""
    telemetries: List[tuple[str, Dict[str, Any]]] = []
    details = summary.get("execution_details")
    if not isinstance(details, list):
        return telemetries
    for detail in details:
        if not isinstance(detail, dict):
            continue
        simulation_data = detail.get("simulation_data")
        if isinstance(simulation_data, dict):
            telemetries.append(
                ("execution_summary.inline", {"simulation_data": simulation_data})
            )
            continue
        structured_result_file = detail.get("structured_result_file")
        if not isinstance(structured_result_file, str) or not structured_result_file:
            continue
        structured_path = _resolve_execution_summary_artifact(
            structured_result_file, summary_dir, execution_output_dir
        )
        try:
            structured_path.relative_to(execution_output_dir.resolve())
        except ValueError:
            logger.warning(
                "Ignoring Step 12 telemetry outside execution output: %s",
                structured_path,
            )
            continue
        payload = _load_telemetry_json(structured_path, logger)
        simulation_data = payload.get("simulation_data") if payload else None
        if isinstance(simulation_data, dict):
            telemetries.append(
                (str(structured_path), {"simulation_data": simulation_data})
            )
    return telemetries


def _resolve_execution_summary_artifact(
    path_value: str, summary_dir: Path, execution_output_dir: Path
) -> Path:
    """Resolve Step 12 summary artifact paths relative to the execution root.

    Step 12 summaries may contain absolute paths, paths rooted at
    ``12_execute_output``, paths rooted at the pipeline output directory, or
    paths relative to ``summaries/``.  Prefer candidates that exist under the
    execution output root so Step 15 does not synthesize duplicated paths like
    ``summaries/output/12_execute_output/...``.
    """
    raw_path = Path(path_value)
    execution_root = execution_output_dir.resolve()
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates: list[Path] = []
    parts = raw_path.parts
    if execution_output_dir.name in parts:
        index = parts.index(execution_output_dir.name)
        suffix = Path(*parts[index + 1 :]) if index + 1 < len(parts) else Path()
        candidates.append(execution_root / suffix)
        candidates.append(execution_root.parent / raw_path)

    candidates.extend(
        [
            execution_root / raw_path,
            execution_root.parent / raw_path,
            summary_dir / raw_path,
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(execution_root)
        except ValueError:
            continue
        if resolved.exists():
            return resolved

    return (execution_root / raw_path).resolve()


def _dedupe_telemetries(
    telemetries: List[tuple[str, Dict[str, Any]]],
) -> List[tuple[str, Dict[str, Any]]]:
    """Remove duplicate telemetry payloads found through summary and file scans."""
    unique: List[tuple[str, Dict[str, Any]]] = []
    seen: set[str] = set()
    for source, telemetry in telemetries:
        key = json.dumps(telemetry, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, telemetry))
    return unique


def generate_audio_from_gnn(
    file_path_or_content: Any, output_dir: Path | None = None, verbose: bool = False
) -> Dict[str, Any]:
    """
    Generate audio from a GNN model.

    Args:
        file_path: Path to the GNN file
        output_dir: Directory to save audio files
        verbose: Enable verbose output

    Returns:
        Dictionary containing audio generation results
    """
    try:
        # Accept either a path or raw content per tests
        if isinstance(file_path_or_content, (str, bytes)) and (
            "\n" in str(file_path_or_content)
            or len(str(file_path_or_content)) < 256
            and not Path(str(file_path_or_content)).exists()
        ):
            content = str(file_path_or_content)
            file_path = Path("gnn_input.md")
        else:
            file_path = Path(file_path_or_content)
            with open(file_path) as f:
                content = f.read()

        # Extract model structure for audio generation
        variables = extract_variables_for_audio(content)
        connections = extract_connections_for_audio(content)

        # Generate different types of audio
        audio_files: dict[Any, Any] = {}

        # 1. Generate tonal representation
        tonal_audio = generate_tonal_representation(variables, connections)
        if output_dir is None:
            raise ValueError("output_dir must be provided")
        output_dir.mkdir(parents=True, exist_ok=True)
        tonal_path = output_dir / f"{file_path.stem}_tonal.wav"
        save_audio_file(tonal_audio, tonal_path, sample_rate=44100)
        audio_files["tonal"] = str(tonal_path)

        # 2. Generate rhythmic representation
        rhythmic_audio = generate_rhythmic_representation(variables, connections)
        rhythmic_path = output_dir / f"{file_path.stem}_rhythmic.wav"
        save_audio_file(rhythmic_audio, rhythmic_path, sample_rate=44100)
        audio_files["rhythmic"] = str(rhythmic_path)

        # 3. Generate ambient representation
        ambient_audio = generate_ambient_representation(variables, connections)
        ambient_path = output_dir / f"{file_path.stem}_ambient.wav"
        save_audio_file(ambient_audio, ambient_path, sample_rate=44100)
        audio_files["ambient"] = str(ambient_path)

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "audio_files": audio_files,
            "variables_count": len(variables),
            "connections_count": len(connections),
            "generation_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise RuntimeError(f"Failed to generate audio from {file_path}: {e}") from e


def extract_variables_for_audio(content: str) -> List[Dict[str, Any]]:
    """Extract unique declarations from the canonical StateSpaceBlock."""
    section_match = re.search(
        r"(?ms)^##[ \t]+StateSpaceBlock[ \t]*\r?\n(.*?)(?=^##[ \t]+|\Z)",
        str(content or ""),
    )
    search_text = section_match.group(1) if section_match else str(content or "")
    declaration_re = re.compile(r"^\s*([^\W\d]\w*)\s*\[([^\]]+)\]", re.UNICODE)
    variables: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in search_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = declaration_re.match(line)
        if match is None or match.group(1) in seen:
            continue
        dimensions = match.group(2)
        type_match = re.search(r"(?:^|,)\s*type\s*=\s*([^,\]]+)", dimensions)
        name = match.group(1)
        seen.add(name)
        variables.append(
            {
                "name": name,
                "type": type_match.group(1).strip() if type_match else "unknown",
                "dimensions": dimensions,
                "definition": match.group(0),
            }
        )

    # Preserve support for the earlier ``name: type`` shorthand when no
    # StateSpaceBlock-style declaration is present.
    if not variables:
        for match in re.finditer(r"(?m)^\s*([^\W\d]\w*)\s*:\s*(\w+)\s*$", search_text):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                variables.append(
                    {
                        "name": name,
                        "type": match.group(2),
                        "definition": match.group(0).strip(),
                    }
                )
    return variables


def extract_connections_for_audio(content: str) -> List[Dict[str, Any]]:
    """Extract canonical and earlier connection operators without duplicates."""
    section_match = re.search(
        r"(?ms)^##[ \t]+Connections[ \t]*\r?\n(.*?)(?=^##[ \t]+|\Z)",
        str(content or ""),
    )
    search_text = section_match.group(1) if section_match else str(content or "")
    connection_re = re.compile(
        r"^\s*([^\W\d]\w*)\s*(->|→|>|-|\*|~)\s*([^\W\d]\w*)\s*$",
        re.UNICODE,
    )
    legacy_re = re.compile(
        r"^\s*([^\W\d]\w*)\s+connects\s+([^\W\d]\w*)\s*$",
        re.IGNORECASE | re.UNICODE,
    )
    connections: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_line in search_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = connection_re.match(line)
        if match:
            source, operator, target = match.groups()
        else:
            legacy_match = legacy_re.match(line)
            if legacy_match is None:
                continue
            source, target = legacy_match.groups()
            operator = "connects"
        key = (source, operator, target)
        if key in seen:
            continue
        seen.add(key)
        connections.append(
            {
                "source": source,
                "target": target,
                "operator": operator,
                "directed": operator in {"->", "→", ">"},
                "definition": line,
            }
        )
    return connections


def _clean_audio_array(audio: Any) -> np.ndarray:
    """Validate mono/stereo sample layout and replace non-finite samples."""
    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError("audio samples must be real numbers")
    try:
        samples = np.asarray(audio, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("audio samples must be numeric") from exc
    if samples.ndim not in (1, 2):
        raise ValueError("audio must be a mono or frames-by-channels array")
    if samples.ndim == 2 and samples.shape[1] < 1:
        raise ValueError("audio must contain at least one channel")
    return np.asarray(np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0))


def _validate_sample_rate(sample_rate: Any) -> int:
    if isinstance(sample_rate, bool):
        raise ValueError("sample_rate must be a positive integer")
    try:
        parsed = int(sample_rate)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be a positive integer") from exc
    if parsed <= 0 or parsed != sample_rate:
        raise ValueError("sample_rate must be a positive integer")
    return parsed


def save_audio_file(
    audio: np.ndarray, file_path: Path, sample_rate: int = 44100
) -> None:
    """Save audio data to file."""
    clean_audio = _clean_audio_array(audio)
    validated_sample_rate = _validate_sample_rate(sample_rate)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(file_path), clean_audio, validated_sample_rate)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        if file_path.suffix.lower() != ".wav":
            raise
        # A WAV target has a dependency-free, format-correct recovery path.
        write_basic_wav(clean_audio, file_path, validated_sample_rate)


def write_basic_wav(audio: np.ndarray, file_path: Path, sample_rate: int) -> Any:
    """Write basic WAV file without external dependencies."""
    import wave

    validated_sample_rate = _validate_sample_rate(sample_rate)
    samples = _clean_audio_array(audio)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(validated_sample_rate)
        wav_file.writeframes(pcm.tobytes(order="C"))


def create_sonification(
    file_path: Path | str, output_dir: Path, verbose: bool = False
) -> Dict[str, Any]:
    """Create sonification of the GNN model."""
    try:
        file_path = Path(file_path)
        with open(file_path) as f:
            content = f.read()

        # Extract model dynamics
        dynamics = extract_model_dynamics(content)

        # Generate sonification
        sonification_audio = generate_sonification_audio(dynamics)
        sonification_path = output_dir / f"{file_path.stem}_sonification.wav"
        save_audio_file(sonification_audio, sonification_path, sample_rate=44100)

        return {
            "file_path": str(file_path),
            "sonification_file": str(sonification_path),
            "dynamics_analyzed": len(dynamics),
            "sonification_type": "model_dynamics",
            "generation_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise RuntimeError(f"Failed to create sonification for {file_path}: {e}") from e


def extract_model_dynamics(content: str) -> List[Dict[str, Any]]:
    """Extract model dynamics for sonification."""
    dynamics: list[Any] = []

    # Look for dynamic elements
    dynamic_patterns: list[Any] = [
        r"(\w+)\s*evolves",  # variable evolves
        r"(\w+)\s*changes",  # variable changes
        r"(\w+)\s*updates",  # variable updates
        r"(\w+)\s*transitions",  # state transitions
    ]

    for pattern in dynamic_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            dynamics.append(
                {
                    "element": match.group(1),
                    "dynamic_type": pattern.split()[0],
                    "description": match.group(0),
                }
            )

    return dynamics


def analyze_audio_characteristics(
    audio_result: Dict[str, Any], verbose: bool = False
) -> Dict[str, Any]:
    """Analyze characteristics of generated audio."""
    analysis: dict[str, Any] = {
        "file_path": audio_result["file_path"],
        "audio_characteristics": {},
        "spectral_analysis": {},
        "temporal_analysis": {},
    }

    # Analyze each audio file
    for audio_type, audio_path in audio_result["audio_files"].items():
        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(audio_path)

            # Basic characteristics
            analysis["audio_characteristics"][audio_type] = {
                "duration": len(audio_data) / sample_rate,
                "sample_rate": sample_rate,
                "channels": 1 if audio_data.ndim == 1 else audio_data.shape[1],
                "max_amplitude": np.max(np.abs(audio_data)),
                "rms_amplitude": np.sqrt(np.mean(audio_data**2)),
            }

            # Spectral analysis
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Take first channel

            # FFT for spectral analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)

            # Find dominant frequencies
            magnitude = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude[: len(magnitude) // 2])
            dominant_freq = freqs[dominant_freq_idx]

            # Calculate spectral metrics with safe division
            magnitude_sum = np.sum(magnitude[: len(magnitude) // 2])
            if magnitude_sum > 0:
                spectral_centroid = (
                    np.sum(freqs[: len(freqs) // 2] * magnitude[: len(magnitude) // 2])
                    / magnitude_sum
                )
                spectral_bandwidth = np.sqrt(
                    np.sum(
                        (freqs[: len(freqs) // 2] - dominant_freq) ** 2
                        * magnitude[: len(magnitude) // 2]
                    )
                    / magnitude_sum
                )
            else:
                spectral_centroid = 0.0
                spectral_bandwidth = 0.0

            analysis["spectral_analysis"][audio_type] = {
                "dominant_frequency": dominant_freq,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
            }

        except Exception as e:
            analysis["audio_characteristics"][audio_type] = {"error": str(e)}

    return analysis


def generate_audio_summary(results: Dict[str, Any]) -> str:
    """Generate a markdown summary of audio processing results."""
    summary = f"""# Audio Processing Summary

Generated on: {results["timestamp"]}

## Overview
- **Files Processed**: {results["processed_files"]}
- **Success**: {results["success"]}
- **Errors**: {len(results["errors"])}

## Audio Files Generated
"""

    for audio_result in results["audio_files_generated"]:
        summary += f"""
### {audio_result["file_name"]}
- **Variables**: {audio_result["variables_count"]}
- **Connections**: {audio_result["connections_count"]}
- **Audio Files**: {len(audio_result["audio_files"])}
"""
        for audio_type, audio_path in audio_result["audio_files"].items():
            summary += f"  - {audio_type}: {Path(audio_path).name}\n"

    if results["errors"]:
        summary += "\n## Errors\n"
        for error in results["errors"]:
            summary += f"- {error}\n"

    return summary
