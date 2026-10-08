"""Public audio consumers verified through independently reopened local WAVs."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from gnn.audio import (
    AudioGenerator,
    SAPFGNNProcessor,
    convert_gnn_to_sapf,
    generate_audio_from_sapf,
    process_gnn_to_audio,
)

GNN_SOURCE = """# Local sensor sonification
## ModelName
NativeSensor
## StateSpaceBlock
s[2,type=float]
o[2,type=float]
## Connections
s > o
## Time
Dynamic
DiscreteTime = t
"""


def read_pcm(path: Path) -> tuple[dict[str, int], np.ndarray]:
    """Use the standard WAV reader rather than the producer's analysis routine."""
    with wave.open(str(path), "rb") as stream:
        header = {
            "channels": stream.getnchannels(),
            "sample_rate": stream.getframerate(),
            "frames": stream.getnframes(),
            "sample_width": stream.getsampwidth(),
        }
        raw = stream.readframes(header["frames"])
    assert header["channels"] == 1
    assert header["sample_width"] == 2
    assert header["sample_rate"] == 44100
    assert len(raw) == header["frames"] * header["channels"] * header["sample_width"]
    return header, np.frombuffer(raw, dtype="<i2")


def test_generator_wavs_have_real_frames_and_matching_public_metadata(
    tmp_path: Path,
) -> None:
    generator = AudioGenerator()
    result = generator.generate_audio(
        {
            "variables": [{"name": "s"}, {"name": "o"}],
            "connections": [{"source": "s", "target": "o"}],
            "output_dir": str(tmp_path / "waveforms"),
        }
    )
    assert result["success"] is True
    assert result["errors"] == []
    files = {Path(filename).stem: Path(filename) for filename in result["audio_files"]}
    assert set(files) == {"tonal", "rhythmic", "ambient", "sonification"}
    for name, duration in {
        "tonal": 5,
        "rhythmic": 5,
        "ambient": 10,
        "sonification": 8,
    }.items():
        header, pcm = read_pcm(files[name])
        assert header["frames"] == duration * header["sample_rate"]
        metadata = generator.analyze_audio(str(files[name]))
        assert metadata["success"] is True
        assert metadata["duration"] == duration
        assert metadata["sample_rate"] == header["sample_rate"]
        assert metadata["channels"] == header["channels"]
        if name != "sonification":
            assert np.any(pcm != 0)


def test_saved_sapf_configuration_drives_native_wave_generation(tmp_path: Path) -> None:
    source = tmp_path / "sensor.md"
    source.write_text(GNN_SOURCE, encoding="utf-8")
    conversion = convert_gnn_to_sapf(source.read_text(), tmp_path / "configuration")
    assert conversion["success"] is True
    config_file = Path(conversion["sapf_config"])
    config = json.loads(config_file.read_text())
    assert {variable["name"] for variable in config["variables"]} == {"s", "o"}
    assert config["connections"]
    assert config["model_type"] == "gnn"
    generated = generate_audio_from_sapf(config, tmp_path / "replayed-audio")
    assert generated["success"] is True
    assert generated["sapf_config"] == config
    assert set(generated["audio_files"]) == {"tonal", "rhythmic", "ambient"}
    for filename in generated["audio_files"].values():
        header, pcm = read_pcm(Path(filename))
        assert header["frames"] > 0
        assert np.any(pcm != 0)
    assert source.read_text() == GNN_SOURCE
    assert json.loads(config_file.read_text()) == config


def test_public_audio_facades_refuse_a_file_used_as_output_directory(
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "caller-owned-file"
    blocked.write_bytes(b"preserve the caller's artifact")
    for result in (
        AudioGenerator().generate_audio(
            {"variables": [{"name": "s"}], "output_dir": str(blocked)}
        ),
        process_gnn_to_audio(GNN_SOURCE, output_dir=str(blocked)),
        convert_gnn_to_sapf(GNN_SOURCE, blocked),
        generate_audio_from_sapf({"variables": [{"name": "s"}]}, blocked),
        SAPFGNNProcessor().generate_audio({"variables": [{"name": "s"}]}, blocked),
    ):
        assert result["success"] is False
        assert result["error"]
        assert blocked.read_bytes() == b"preserve the caller's artifact"
    assert not list(tmp_path.rglob("*.wav"))


def test_empty_gnn_content_is_rejected_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "uncreated"
    result = process_gnn_to_audio(" \n\t", output_dir=str(output))
    assert result["success"] is False
    assert "Empty GNN" in result["error"]
    assert not output.exists()
