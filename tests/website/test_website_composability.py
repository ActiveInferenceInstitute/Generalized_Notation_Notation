"""Composability pins for the dict-driven website API.

Pins the seams introduced when the generator↔collection coupling was
broken and the dict-driven API became first-class:

- the cycle direction: ``collection`` imports the ``steps`` leaf, never
  ``generator``; ``steps`` imports no website sibling; ``generator``
  holds the single module-level ``collection`` import;
- ``website_data_from_dict``: pure dict → data dict with ZERO filesystem
  access and the collectors' exact empty defaults;
- ``generate_website(..., filesystem=False)``: renders a complete site
  from a prebuilt dict with no collector invoked;
- ``SUPPORTED_FILE_TYPES``: one definition (``renderer``), one package
  re-export, derived flat/dot inventories.
"""

import inspect
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCycleBreak:
    """The generator↔collection import cycle is broken via the steps leaf."""

    def test_collection_no_longer_imports_generator(self) -> None:
        import gnn.website.collection as collection

        source = inspect.getsource(collection)
        assert "from .generator import" not in source, (
            "collection must not import generator — the steps leaf is the "
            "single PIPELINE_STEPS source; importing generator from "
            "collection re-creates the import cycle"
        )
        assert "from .steps import PIPELINE_STEPS" in source

    def test_steps_leaf_imports_no_website_sibling(self) -> None:
        import gnn.website.steps as steps

        source = inspect.getsource(steps)
        assert "website.generator" not in source
        assert "website.collection" not in source
        assert ".generator" not in source
        assert ".collection" not in source

    def test_generator_holds_the_collection_import(self) -> None:
        import gnn.website.generator as generator

        source = inspect.getsource(generator)
        assert "from gnn.website.collection import collect_website_data" in source, (
            "generator is the module-level collection importer (chosen direction)"
        )

    def test_single_pipeline_steps_object_across_the_package(self) -> None:
        import gnn.website as website
        import gnn.website.collection as collection
        import gnn.website.generator as generator
        import gnn.website.steps as steps

        assert collection.PIPELINE_STEPS is steps.PIPELINE_STEPS
        assert generator.PIPELINE_STEPS is steps.PIPELINE_STEPS
        assert website.PIPELINE_STEPS is steps.PIPELINE_STEPS
        assert generator.get_pipeline_steps() is steps.PIPELINE_STEPS


class TestWebsiteDataFromDict:
    """``website_data_from_dict`` is pure: caller dict in, data dict out."""

    def test_empty_dict_takes_the_collectors_empty_defaults(self) -> None:
        from gnn.website.collection import website_data_from_dict

        data = website_data_from_dict({})
        assert data["p_root"] is None
        assert data["processed_files"] == 0
        assert data["gui_navigation"] is False
        assert data["mcp_summary"] == {}
        assert data["pipeline_summary"] == {}
        for key in (
            "gnn_files",
            "models",
            "analysis",
            "warnings",
            "visualizations",
            "reports",
            "mcp_tools",
        ):
            assert data[key] == []
        assert len(data["step_statuses"]) == 25
        assert set(data["step_statuses"].values()) == {"pending"}

    def test_caller_keys_win_and_reserved_keys_are_excluded(self) -> None:
        from gnn.website.collection import website_data_from_dict

        data = website_data_from_dict(
            {
                "models": [{"name": "m", "slug": "m"}],
                "step_statuses": {3: "ok"},
                "search_data": {"pages": []},
                "output_dir": "/caller/out",
                "input_dir": "/caller/in",
                "pipeline_output_root": "/caller/root",
            },
            output_dir="/kwarg/out",
        )
        assert data["models"] == [{"name": "m", "slug": "m"}]
        assert data["step_statuses"] == {3: "ok"}
        assert data["search_data"] == {"pages": []}
        # Reserved keys are excluded from the user_data merge; output_dir
        # comes only from the explicit kwarg, input_dir never enters data.
        assert "input_dir" not in data
        assert data["p_root"] == Path("/caller/root")
        assert data["output_dir"] == Path("/kwarg/out")

    def test_output_dir_kwarg_normalizes_and_defaults_to_none(self) -> None:
        from gnn.website.collection import website_data_from_dict

        assert website_data_from_dict({})["output_dir"] is None
        assert website_data_from_dict({}, output_dir=str("/tmp/x"))[
            "output_dir"
        ] == Path("/tmp/x")

    def test_no_filesystem_access(self, monkeypatch: Any) -> None:
        calls = {"n": 0}
        orig_exists = Path.exists

        def counting(p: Path) -> bool:
            calls["n"] += 1
            return orig_exists(p)

        monkeypatch.setattr(Path, "exists", counting)
        from gnn.website.collection import website_data_from_dict

        website_data_from_dict({"models": [{"name": "m"}]})
        assert calls["n"] == 0, "website_data_from_dict must not touch the filesystem"


class TestPureDictGeneration:
    """``filesystem=False`` renders a complete site from a prebuilt dict."""

    def _pure_dict(self, tmp_path: Path) -> dict[str, Any]:
        return {
            "output_dir": str(tmp_path / "site"),
            "models": [
                {
                    "name": "Demo Model",
                    "source_name": "demo.md",
                    "variables": [],
                    "edges": [],
                }
            ],
            "step_statuses": {3: "ok"},
            "processed_files": 1,
        }

    def test_pure_dict_site_with_no_collector_invoked(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        import gnn.website.collection as collection
        import gnn.website.generator as generator
        from gnn.website import WebsiteGenerator

        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("collector invoked in filesystem=False mode")

        for name in (
            "_collect_gnn_files",
            "_collect_parsed_models",
            "_collect_step_statuses",
            "_collect_analysis_results",
            "_collect_visualizations",
            "_collect_reports",
            "_load_mcp_summary",
            "_load_registered_tools",
            "collect_website_data",
        ):
            if hasattr(collection, name):
                monkeypatch.setattr(collection, name, _forbidden)
        monkeypatch.setattr(generator, "collect_website_data", _forbidden)

        calls = {"n": 0}
        orig_exists = Path.exists

        def counting(p: Path) -> bool:
            calls["n"] += 1
            return orig_exists(p)

        monkeypatch.setattr(Path, "exists", counting)

        result = WebsiteGenerator().generate_website(
            self._pure_dict(tmp_path), filesystem=False
        )
        assert result["success"] is True, result["errors"]
        assert result["pages_created"] == 7
        assert result["model_pages_created"] == 1
        assert "model/demo-model.html" in result["model_pages"]
        assert calls["n"] == 0, "generation must not probe the filesystem"
        model_page = tmp_path / "site" / "model" / "demo-model.html"
        assert model_page.exists()
        search_index = (tmp_path / "site" / "search-index.json").read_text(
            encoding="utf-8"
        )
        assert "model/demo-model.html" in search_index
        listing = (tmp_path / "site" / "gnn_files.html").read_text(encoding="utf-8")
        assert 'href="model/demo-model.html"' in listing

    def test_pure_mode_requires_explicit_output_dir(self) -> None:
        from gnn.website import WebsiteGenerator

        result = WebsiteGenerator().generate_website({"models": []}, filesystem=False)
        assert result["success"] is False
        assert result["errors"]
        assert "requires an explicit" in result["errors"][0]

    def test_default_filesystem_path_unchanged(self, tmp_path: Any) -> None:
        from gnn.website import WebsiteGenerator

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = WebsiteGenerator().generate_website(
            {
                "input_dir": str(input_dir),
                "output_dir": str(tmp_path / "site"),
                "pipeline_output_root": str(tmp_path),
            }
        )
        assert result["success"] is True, result["errors"]
        assert result["pages_created"] == 7


class TestSupportedFileTypesSingleSource:
    """One definition, one re-export, derived inventories."""

    def test_package_reexport_is_the_renderer_object(self) -> None:
        import gnn.website as website
        import gnn.website.renderer as renderer

        assert website.SUPPORTED_FILE_TYPES is renderer.SUPPORTED_FILE_TYPES

    def test_flat_inventory_derived_and_deduplicated(self) -> None:
        from gnn.website import get_supported_file_types

        types = get_supported_file_types()
        assert isinstance(types, list)
        assert len(types) == len(set(types))
        assert {"html", "css", "js", "json", "md", "png"} <= set(types)

    def test_module_info_dots_derived(self) -> None:
        from gnn.website import get_module_info

        dots = get_module_info()["supported_file_types"]
        assert isinstance(dots, list)
        assert dots
        assert all(str(entry).startswith(".") for entry in dots)
