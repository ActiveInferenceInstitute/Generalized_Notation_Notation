"""Real saved configuration files preserve options and reject invalid resources.

These are the documented saved-file utility contracts, not pipeline execution
or a claim that legacy configuration utilities replace shared run admission.
"""

from pathlib import Path

import pytest
import yaml

from gnn.utils.config_io.config_loader import load_config, save_config


def authored_configuration(tmp_path: Path) -> dict:
    models = tmp_path / "caller-models"
    models.mkdir()
    ontology = tmp_path / "caller-ontology.json"
    ontology.write_text('{"caller_term": "retained"}\n')
    return {
        "pipeline": {
            "target_dir": str(models),
            "output_dir": str(tmp_path / "caller-output"),
            "pipeline_summary_file": str(tmp_path / "caller-summary.json"),
            "recursive": False,
            "verbose": False,
            "enable_round_trip": False,
            "enable_cross_format": True,
            "skip_steps": [13, 14],
            "only_steps": [3, 5],
            "fast_only": False,
            "comprehensive": True,
        },
        "type_checker": {"strict": True, "estimate_resources": False},
        "ontology": {"terms_file": str(ontology)},
        "llm": {"tasks": "validation", "timeout": 47},
        "setup": {"recreate_venv": False, "dev": True, "install_all_extras": False},
        "sapf": {"duration": 2.5},
        "models": {
            "global": {"seed": 0},
            "model_overrides": {"caller-model": {"active": False}},
        },
    }


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_saved_configuration_reaches_public_argument_projection(tmp_path, suffix):
    authored = authored_configuration(tmp_path)
    path = tmp_path / f"authored{suffix}"
    save_config(authored, path)
    before = path.read_bytes()
    assert yaml.safe_load(before) == authored
    loaded = load_config(path)
    projected = loaded.to_pipeline_arguments()
    for name in ("target_dir", "output_dir", "pipeline_summary_file"):
        assert projected[name] == Path(authored["pipeline"][name])
    assert projected["recursive"] is False
    assert projected["verbose"] is False
    assert projected["enable_round_trip"] is False
    assert projected["enable_cross_format"] is True
    assert projected["skip_steps"] == "13,14"
    assert projected["only_steps"] == "3,5"
    assert projected["fast_only"] is False
    assert projected["comprehensive"] is True
    assert projected["strict"] is True
    assert projected["estimate_resources"] is False
    assert projected["ontology_terms_file"] == Path(authored["ontology"]["terms_file"])
    assert projected["llm_tasks"] == "validation"
    assert projected["llm_timeout"] == 47
    assert projected["recreate_venv"] is False
    assert projected["dev"] is True
    assert projected["install_all_extras"] is False
    assert projected["duration"] == 2.5
    assert loaded.models.global_settings == {"seed": 0}
    assert loaded.models.model_overrides == {"caller-model": {"active": False}}
    assert loaded.validate() == []
    assert path.read_bytes() == before
    assert not Path(authored["pipeline"]["output_dir"]).exists()
    assert not Path(authored["pipeline"]["pipeline_summary_file"]).exists()


@pytest.mark.parametrize(
    "invalid", ["target", "ontology", "zero_timeout", "negative_duration"]
)
def test_saved_configuration_resource_refusal_preserves_authored_file(
    tmp_path, caplog, invalid
):
    authored = authored_configuration(tmp_path)
    if invalid == "target":
        authored["pipeline"]["target_dir"] = str(tmp_path / "missing-models")
        diagnosis = "Target directory does not exist"
    elif invalid == "ontology":
        authored["ontology"]["terms_file"] = str(tmp_path / "missing-ontology.json")
        diagnosis = "Ontology terms file does not exist"
    elif invalid == "zero_timeout":
        authored["llm"]["timeout"] = 0
        diagnosis = "LLM timeout must be positive"
    else:
        authored["sapf"]["duration"] = -1
        diagnosis = "SAPF duration must be positive"
    path = tmp_path / "invalid.yaml"
    save_config(authored, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Configuration validation failed"):
        load_config(path)
    assert diagnosis in caplog.text
    assert path.read_bytes() == before
    assert not Path(authored["pipeline"]["output_dir"]).exists()


def test_malformed_saved_yaml_preserves_original_parser_diagnosis(tmp_path, caplog):
    path = tmp_path / "malformed.yaml"
    path.write_bytes(b"pipeline: [unterminated\n")
    before = path.read_bytes()
    with pytest.raises(yaml.YAMLError):
        load_config(path)
    assert "Failed to load configuration" in caplog.text
    assert str(path) in caplog.text
    assert path.read_bytes() == before


def test_unknown_saved_suffix_refuses_before_overwriting_existing_file(tmp_path):
    path = tmp_path / "caller.config"
    before = b"caller-owned previous configuration\n"
    path.write_bytes(before)
    with pytest.raises(ValueError, match="Unsupported config file extension"):
        save_config({"pipeline": {"recursive": False}}, path)
    assert path.read_bytes() == before
