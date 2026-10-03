"""Contract tests for per-model website pages, breadcrumbs, and search.

Pins the Step-20 contract for the per-model page extension:

- One HTML page per parsed model under ``model/<slug>.html`` with the
  documented slug rules (lowercase, non-``[a-z0-9]`` → ``-``, collapsed,
  stripped) and ``-2``/``-3`` collision suffixes for duplicate slugs.
- Model pages render the model's FULL source — the 3000-char cap stays
  only on the aggregate ``gnn_files`` listing rows.
- A breadcrumb nav on every generated page (including ``index.html``)
  with depth-correct relative hrefs and escaped labels.
- A standalone ``search-index.json`` plus an inline copy on the listing
  page (``fetch()`` fails on ``file://``; the page uses the inline copy).
- Additive bookkeeping: ``model_pages_created``/``model_pages`` on the
  result and manifest, and an additive ``model_pages`` key on
  ``inspect_website`` — while ``pages``/``pages_created`` stay exactly
  the seven site pages.

Deterministic and filesystem-only — no network, tmp_path only.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#: Unique tail marker placed past the 3000-char truncation boundary of the
#: long fixture model. Must appear on the model page (full source) and be
#: absent from the aggregate listing page (3000-char cap kept).
_SENTINEL_TAIL = "SENTINEL-TAIL-3f9a-model-source-continues-beyond-3000-chars"

_MODEL_SOURCES: dict[str, str] = {
    "model_alpha.md": (
        "# Alpha Model\n\n"
        "## ModelName\nAlpha Model\n\n"
        "## StateSpaceBlock\n"
        "Xstate[2,2,type=float]\n"
        "Yobs[2,1,type=float]\n\n"
        "## Connections\n"
        "Xstate -> Yobs\n"
    ),
    # Same model name as model_alpha.md — claims the same slug.
    "model_dup.md": (
        "# Alpha Model\n\n"
        "## ModelName\nAlpha Model\n\n"
        "## StateSpaceBlock\n"
        "Xstate[2,2,type=float]\n\n"
        "## Connections\n"
        "Xstate -> Yobs\n"
    ),
    # Normalizes to the same slug ("Alpha  Model" → "alpha-model").
    "model_space.md": (
        "# Alpha  Model\n\n"
        "## ModelName\nAlpha  Model\n\n"
        "## StateSpaceBlock\n"
        "Yobs[2,1,type=float]\n\n"
        "## Connections\n"
        "Xstate -> Yobs\n"
    ),
    # No Connections section — the edges table must show an explicit None row.
    "model_beta.md": (
        "# Beta Model\n\n"
        "## ModelName\nBeta Model\n\n"
        "## StateSpaceBlock\n"
        "Bstate[3,3,type=float]\n"
    ),
    # Markup-sensitive model name: escaping conventions must apply.
    "model_amp.md": (
        "# R&D Model\n\n"
        "## ModelName\nR&D Model\n\n"
        "## StateSpaceBlock\n"
        "Xstate[2,2,type=float]\n\n"
        "## Connections\n"
        "Xstate -> Yobs\n"
    ),
    # Source longer than 3000 chars with the sentinel in the tail.
    "model_long.md": (
        "# Delta Model\n\n"
        "## ModelName\nDelta Model\n\n"
        "## StateSpaceBlock\n"
        "Xstate[2,2,type=float]\n"
        "Yobs[2,1,type=float]\n\n"
        "## Connections\n"
        "Xstate -> Yobs\n\n"
        "## ModelAnnotation\n"
        + ("-- filler source padding for the full-source pin --\n" * 80)
        + _SENTINEL_TAIL
        + "\n"
    ),
}

_SITE_PAGE_FILENAMES: tuple[str, ...] = (
    "index.html",
    "pipeline.html",
    "gnn_files.html",
    "analysis.html",
    "visualization.html",
    "reports.html",
    "mcp.html",
)

#: Site-root-relative model page filenames for the fixture above.
_EXPECTED_MODEL_PAGES: set[str] = {
    "model/alpha-model.html",
    "model/alpha-model-2.html",
    "model/alpha-model-3.html",
    "model/beta-model.html",
    "model/r-d-model.html",
    "model/delta-model.html",
}

_BREADCRUMB_NAV_RE = re.compile(
    r'<nav class="breadcrumbs"[^>]*>\s*<ol>(.*?)</ol>', re.DOTALL
)
_CRUMB_LI_RE = re.compile(r"<li(?:\s[^>]*)?>(.*?)</li>", re.DOTALL)
_CRUMB_LINK_RE = re.compile(r'<a href="([^"]*)">(.*?)</a>', re.DOTALL)
_SEARCH_DATA_RE = re.compile(
    r'<script type="application/json" id="gnn-search-data">(.*?)</script>',
    re.DOTALL,
)


def _build_multi_model_site(tmp_path: Any) -> tuple[Path, dict[str, Any]]:
    """Build a real site from a multi-model input dir (2+ files, duplicates)."""
    from gnn.website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name, source in _MODEL_SOURCES.items():
        (input_dir / name).write_text(source, encoding="utf-8")
    out = tmp_path / "site"
    result = WebsiteGenerator().generate_website(
        {
            "input_dir": str(input_dir),
            "output_dir": str(out),
            "pipeline_output_root": str(tmp_path),
        }
    )
    assert result["success"] is True, result["errors"]
    return out, result


def _build_empty_site(tmp_path: Any) -> tuple[Path, dict[str, Any]]:
    """Build a real site from an empty input dir."""
    from gnn.website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out = tmp_path / "site"
    result = WebsiteGenerator().generate_website(
        {
            "input_dir": str(input_dir),
            "output_dir": str(out),
            "pipeline_output_root": str(tmp_path),
        }
    )
    assert result["success"] is True, result["errors"]
    return out, result


def _breadcrumb_crumbs(page_html: str) -> list[tuple[str | None, str]]:
    """Parse the breadcrumb nav into (href-or-None, label) crumbs."""
    m = _BREADCRUMB_NAV_RE.search(page_html)
    assert m is not None, "breadcrumbs nav missing"
    crumbs: list[tuple[str | None, str]] = []
    for raw in _CRUMB_LI_RE.findall(m.group(1)):
        link = _CRUMB_LINK_RE.match(raw)
        if link is not None:
            crumbs.append((link.group(1), link.group(2)))
        else:
            crumbs.append((None, raw))
    assert crumbs, "breadcrumbs nav has no crumbs"
    return crumbs


def _resolve(page_path: Path, href: str) -> Path:
    """Resolve a depth-correct relative href against its page directory."""
    return Path(os.path.normpath(os.path.join(str(page_path.parent), href)))


def _h1_text_window(page_html: str, span: int = 120) -> str:
    """Text window right after the first <h1>, tolerating icon decorations."""
    idx = page_html.find("<h1")
    assert idx != -1, "model page has no <h1>"
    return page_html[idx : idx + span]


def _inline_scripts(page_html: str) -> list[str]:
    """Complete attribute-less script bodies, preserving their raw JavaScript."""

    class InlineScriptParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.scripts: list[str] = []
            self.body: list[str] | None = None

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag == "script":
                self.body = [] if not attrs else None

        def handle_startendtag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            # An XML-style empty tag supplies no complete HTML script body.
            pass

        def handle_data(self, data: str) -> None:
            if self.body is not None:
                self.body.append(data)

        def handle_endtag(self, tag: str) -> None:
            if tag == "script" and self.body is not None:
                self.scripts.append("".join(self.body))
                self.body = None

    parser = InlineScriptParser()
    parser.feed(page_html)
    parser.close()
    return parser.scripts


class TestInlineScriptExtraction:
    """The search-script checks consume complete HTML raw-text elements."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("page_html", "expected"),
        [
            ("<SCRIPT>const upper = true;</SCRIPT>", ["const upper = true;"]),
            ("<ScRiPt >mixed();</sCrIpT \n>", ["mixed();"]),
            (
                "<script>first();</script><script>second();</script>",
                ["first();", "second();"],
            ),
            ("<script></script>", [""]),
            ("<script>closed();</script><script>unfinished();", ["closed();"]),
            ("<script>unfinished();</script", []),
            ("<script/>not_a_complete_body();", []),
            ("<!-- <script>commented();</script> -->", []),
            ("<script src='external.js'>external();</script>", []),
            ("<script TYPE='application/json'>{\"value\":1}</script>", []),
            ("<script nonce='abc'>attributed();</script>", []),
            (
                "<script src='external.js'></script><SCRIPT>local();</SCRIPT>",
                ["local();"],
            ),
        ],
    )
    def test_extracts_only_complete_attribute_less_scripts(
        self, page_html: str, expected: list[str]
    ) -> None:
        assert _inline_scripts(page_html) == expected

    @pytest.mark.unit
    def test_preserves_raw_javascript_without_decoding_or_parsing_markup(self) -> None:
        body = (
            '\nconst markup = "<div>&lt; &#38; &amp;</div>";\n'
            'const nested = "<script>literal</scriptx>";\n'
            "// <!-- a raw JavaScript comment -->\n"
            "if (left < right && right > 0) { use(markup); }\n"
        )
        assert _inline_scripts(f"<script>{body}</script>") == [body]


class TestModelPageGeneration:
    """C2: one page per parsed model under ``model/`` with collision slugs."""

    @pytest.mark.unit
    def test_model_pages_created_with_collision_suffixes(self, tmp_path: Any) -> None:
        site, result = _build_multi_model_site(tmp_path)

        assert result["model_pages_created"] == 6
        assert isinstance(result["model_pages_created"], int)
        assert set(result["model_pages"]) == _EXPECTED_MODEL_PAGES
        for rel in result["model_pages"]:
            assert rel.startswith("model/") and rel.endswith(".html")
            assert "\\" not in rel, "model_pages entries must be POSIX paths"
            assert (site / rel).is_file()

    @pytest.mark.unit
    def test_model_page_content_h1_source_link_and_tables(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        page = (site / "model/alpha-model.html").read_text(encoding="utf-8")

        assert "Alpha Model" in _h1_text_window(page)
        # Source line linking back to the listing page, depth-correct.
        assert 'href="../gnn_files.html"' in page
        # Variables table + edges table from the parsed model data.
        assert page.count("<table") >= 2
        assert "Xstate" in page
        assert "Yobs" in page

    @pytest.mark.unit
    def test_model_page_without_edges_has_explicit_none_row(
        self, tmp_path: Any
    ) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        page = (site / "model/beta-model.html").read_text(encoding="utf-8")

        assert "Beta Model" in _h1_text_window(page)
        assert page.count("<table") >= 2
        assert "None" in page, "empty edges table must render an explicit None row"

    @pytest.mark.unit
    def test_model_page_renders_full_source_past_3000_chars(
        self, tmp_path: Any
    ) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        page = (site / "model/delta-model.html").read_text(encoding="utf-8")

        assert _SENTINEL_TAIL in page, "model page must render the full source"
        assert "… [truncated]" not in page, "no truncation marker on model pages"

    @pytest.mark.unit
    def test_listing_page_keeps_3000_char_truncation(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        listing = (site / "gnn_files.html").read_text(encoding="utf-8")

        assert _SENTINEL_TAIL not in listing, "listing rows stay capped at 3000"
        assert "… [truncated]" in listing, "aggregate listing keeps its cap"

    @pytest.mark.unit
    def test_model_page_escapes_model_name_markup(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        page = (site / "model/r-d-model.html").read_text(encoding="utf-8")

        assert "R&amp;D Model" in _h1_text_window(page)
        assert "R&D Model" not in page


class TestBreadcrumbs:
    """C1: breadcrumb nav on every generated page, depth-correct + escaped."""

    @pytest.mark.unit
    def test_breadcrumbs_present_on_every_generated_page(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        pages = [site / f for f in (*_SITE_PAGE_FILENAMES, *_EXPECTED_MODEL_PAGES)]
        for page_path in pages:
            html = page_path.read_text(encoding="utf-8")
            assert 'class="breadcrumbs"' in html, page_path.name
            assert 'aria-label="Breadcrumb"' in html, page_path.name
            assert 'aria-current="page"' in html, page_path.name

    @pytest.mark.unit
    def test_index_breadcrumb_is_single_home_crumb(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        html = (site / "index.html").read_text(encoding="utf-8")
        crumbs = _breadcrumb_crumbs(html)

        assert crumbs == [(None, "Home")], crumbs

    @pytest.mark.unit
    def test_site_page_breadcrumbs_home_then_page(self, tmp_path: Any) -> None:
        from gnn.website.pages import SITE_PAGES

        titles = {page.name: page.title for page in SITE_PAGES}
        site, _ = _build_multi_model_site(tmp_path)
        for name in ("pipeline", "gnn_files", "analysis", "reports", "mcp"):
            html = (site / f"{name}.html").read_text(encoding="utf-8")
            crumbs = _breadcrumb_crumbs(html)
            hrefs = [href for href, _ in crumbs]
            labels = [label for _, label in crumbs]
            assert hrefs == ["index.html", None], (name, crumbs)
            assert labels[0] == "Home"
            assert titles[name] in labels[1], (name, labels)

    @pytest.mark.unit
    def test_model_page_breadcrumbs_home_listing_then_model(
        self, tmp_path: Any
    ) -> None:
        from gnn.website.pages import SITE_PAGES

        listing_title = next(p.title for p in SITE_PAGES if p.name == "gnn_files")
        site, _ = _build_multi_model_site(tmp_path)
        html = (site / "model/beta-model.html").read_text(encoding="utf-8")
        crumbs = _breadcrumb_crumbs(html)

        assert len(crumbs) == 3, crumbs
        hrefs = [href for href, _ in crumbs]
        assert hrefs == ["../index.html", "../gnn_files.html", None], crumbs
        labels = [label for _, label in crumbs]
        assert labels[0] == "Home"
        assert listing_title in labels[1]
        assert "Beta Model" in labels[2]

    @pytest.mark.unit
    def test_breadcrumb_links_resolve_depth_correct(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        for rel in (*_SITE_PAGE_FILENAMES, *_EXPECTED_MODEL_PAGES):
            page_path = site / rel
            for href, _label in _breadcrumb_crumbs(
                page_path.read_text(encoding="utf-8")
            ):
                if href is None:
                    continue
                assert _resolve(page_path, href).is_file(), (rel, href)

    @pytest.mark.unit
    def test_breadcrumbs_css_rule_inlined(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        for rel in ("index.html", "model/beta-model.html"):
            html = (site / rel).read_text(encoding="utf-8")
            assert ".breadcrumbs" in html, rel


class TestSearchIndex:
    """C3: standalone search-index.json plus the inline copy on the listing page."""

    @pytest.mark.unit
    def test_search_index_exists_and_matches_schema(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        index = json.loads((site / "search-index.json").read_text(encoding="utf-8"))

        assert set(index) == {"generated", "pages"}
        assert isinstance(index["pages"], list) and index["pages"]
        for entry in index["pages"]:
            assert set(entry) == {"title", "url", "snippet"}, entry

    @pytest.mark.unit
    def test_search_index_covers_every_emitted_page(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        index = json.loads((site / "search-index.json").read_text(encoding="utf-8"))
        urls = [entry["url"] for entry in index["pages"]]

        assert len(index["pages"]) == 7 + 6
        assert set(urls) == set(_SITE_PAGE_FILENAMES) | _EXPECTED_MODEL_PAGES
        for url in urls:
            assert not url.startswith("/"), url
            assert "\\" not in url, url
            assert (site / url).is_file(), url

    @pytest.mark.unit
    def test_search_snippets_are_plain_and_bounded(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        index = json.loads((site / "search-index.json").read_text(encoding="utf-8"))

        for entry in index["pages"]:
            snippet = entry["snippet"]
            assert isinstance(snippet, str)
            assert len(snippet) <= 200, entry["url"]
            assert "<" not in snippet, entry["url"]

    @pytest.mark.unit
    def test_generated_field_is_iso_8601(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        index = json.loads((site / "search-index.json").read_text(encoding="utf-8"))

        raw = index["generated"]
        assert isinstance(raw, str) and raw
        datetime.fromisoformat(raw.replace("Z", "+00:00"))

    @pytest.mark.unit
    def test_listing_page_has_search_markup_and_inline_payload(
        self, tmp_path: Any
    ) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        listing = (site / "gnn_files.html").read_text(encoding="utf-8")

        assert '<input id="gnn-site-search" type="search">' in listing
        assert re.search(r'<ul id="gnn-search-results"[^>]*\bhidden', listing)
        assert '<script type="application/json" id="gnn-search-data">' in listing

        payload_match = _SEARCH_DATA_RE.search(listing)
        assert payload_match is not None, "inline search data payload missing"
        assert "</" not in payload_match.group(1), "payload must escape </ as <\\/ "
        json.loads(payload_match.group(1))  # parses despite the <\/ escapes

    @pytest.mark.unit
    def test_inline_search_payload_matches_standalone_index(
        self, tmp_path: Any
    ) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        standalone = json.loads((site / "search-index.json").read_text("utf-8"))
        listing = (site / "gnn_files.html").read_text(encoding="utf-8")
        payload_match = _SEARCH_DATA_RE.search(listing)
        assert payload_match is not None, "inline search data payload missing"
        inline = json.loads(payload_match.group(1))

        assert inline == standalone

    @pytest.mark.unit
    def test_inline_search_script_is_self_contained(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        listing = (site / "gnn_files.html").read_text(encoding="utf-8")

        assert re.search(r"<script[^>]*\ssrc=", listing) is None
        scripts = _inline_scripts(listing)
        assert scripts, "inline vanilla JS search filter missing"
        joined = "\n".join(scripts)
        assert "gnn-site-search" in joined
        assert "gnn-search-results" in joined
        for script in scripts:
            assert len(script.splitlines()) <= 60

    @pytest.mark.unit
    def test_empty_site_search_index_lists_seven_pages(self, tmp_path: Any) -> None:
        site, result = _build_empty_site(tmp_path)
        index = json.loads((site / "search-index.json").read_text(encoding="utf-8"))

        assert result["model_pages_created"] == 0
        assert result["model_pages"] == []
        assert len(index["pages"]) == 7
        assert {entry["url"] for entry in index["pages"]} == set(_SITE_PAGE_FILENAMES)


class TestInspectionAndManifest:
    """C4/C2 bookkeeping: additive keys, seven-page pins untouched."""

    @pytest.mark.unit
    def test_inspect_website_reports_model_pages_additively(
        self, tmp_path: Any
    ) -> None:
        from gnn.website import inspect_website
        from gnn.website.pages import page_filenames

        site, _ = _build_multi_model_site(tmp_path)
        inspection = inspect_website(site)

        assert inspection["success"] is True
        assert "model_pages" in inspection
        assert set(inspection["model_pages"]) == _EXPECTED_MODEL_PAGES
        # Root-only semantics unchanged.
        assert inspection["pages_count"] == 7
        assert set(inspection["pages"]) == set(page_filenames())

    @pytest.mark.unit
    def test_inspect_website_model_pages_empty_without_models(
        self, tmp_path: Any
    ) -> None:
        from gnn.website import inspect_website
        from gnn.website.pages import page_filenames

        site, _ = _build_empty_site(tmp_path)
        inspection = inspect_website(site)

        assert inspection["success"] is True
        assert inspection["model_pages"] == []
        assert inspection["pages_count"] == 7
        assert set(inspection["pages"]) == set(page_filenames())

    @pytest.mark.unit
    def test_result_bookkeeping_keeps_seven_page_pins(self, tmp_path: Any) -> None:
        site, result = _build_multi_model_site(tmp_path)

        assert result["pages_created"] == 7
        assert len(result["pages"]) == 7
        assert set(result["pages"]) == set(_SITE_PAGE_FILENAMES)
        assert result["model_pages_created"] == 6
        assert set(result["model_pages"]) == _EXPECTED_MODEL_PAGES

    @pytest.mark.unit
    def test_website_results_manifest_records_model_bookkeeping(
        self, tmp_path: Any
    ) -> None:
        from gnn.website import process_website

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name, source in _MODEL_SOURCES.items():
            (input_dir / name).write_text(source, encoding="utf-8")
        out = tmp_path / "out"

        assert (
            process_website(
                target_dir=input_dir,
                output_dir=out,
                verbose=False,
                logger=logging.getLogger("t"),
                recursive=False,
            )
            is True
        )

        manifest = json.loads((out / "website_results.json").read_text())
        assert manifest["success"] is True
        assert manifest["pages_created"] == 7
        assert set(manifest["pages"]) == set(_SITE_PAGE_FILENAMES)
        assert manifest["model_pages_created"] == 6
        assert set(manifest["model_pages"]) == _EXPECTED_MODEL_PAGES


class TestDeepLinksAndHygiene:
    """C5/C6: deep links resolve mechanically; no external references."""

    @pytest.mark.unit
    def test_listing_page_model_links_resolve_to_files(self, tmp_path: Any) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        listing = (site / "gnn_files.html").read_text(encoding="utf-8")

        hrefs = re.findall(r'href="(model/[^"]+\.html)"', listing)
        assert hrefs, "listing page must deep-link every model row"
        assert set(hrefs) == _EXPECTED_MODEL_PAGES
        for href in hrefs:
            assert os.path.exists(os.path.join(str(site), href)), href

    @pytest.mark.unit
    def test_new_artifacts_carry_no_external_resource_references(
        self, tmp_path: Any
    ) -> None:
        """Offline-true: no page fetches any external resource.

        Checked as resource references (src/href pointing at http(s), CSS
        ``@import``/``url()``) across EVERY emitted page — the seven site
        pages, all model pages, and ``search-index.json``. The schema.org
        context IRI inside the per-page JSON-LD is a vocabulary identifier,
        never dereferenced, so it must not trip this pin.
        """
        site, _ = _build_multi_model_site(tmp_path)

        pages = sorted(site.rglob("*.html"))
        assert {p.name for p in pages} >= set(_SITE_PAGE_FILENAMES)
        for page in pages:
            html = page.read_text(encoding="utf-8")
            for attribute in ("src", "href"):
                for match in re.findall(rf'{attribute}\s*=\s*"[^"]*"', html):
                    assert not match.lower().startswith(f'{attribute}="http'), (
                        page.name,
                        match,
                    )
            assert "@import" not in html, page.name
            assert not re.search(r"url\(\s*[\"']?https?://", html), page.name

        index_text = (site / "search-index.json").read_text(encoding="utf-8")
        assert "http://" not in index_text and "https://" not in index_text

        # Only the *new* markup on the listing page: payload + inline JS.
        listing = (site / "gnn_files.html").read_text(encoding="utf-8")
        payload_match = _SEARCH_DATA_RE.search(listing)
        assert payload_match is not None, "inline search data payload missing"
        payload = payload_match.group(1)
        scripts = "\n".join(_inline_scripts(listing))
        for snippet in (payload, scripts):
            assert "http://" not in snippet and "https://" not in snippet

    @pytest.mark.unit
    def test_every_page_carries_meta_description_and_jsonld(
        self, tmp_path: Any
    ) -> None:
        site, _ = _build_multi_model_site(tmp_path)
        pages = sorted(site.rglob("*.html"))
        assert {p.name for p in pages} >= set(_SITE_PAGE_FILENAMES)
        for page in pages:
            html = page.read_text(encoding="utf-8")
            assert '<meta name="description" content="GNN Pipeline Results' in (html), (
                page.name
            )
            match = re.search(
                r'<script type="application/ld\+json">(.*?)</script>',
                html,
                re.DOTALL,
            )
            assert match, page.name
            payload = json.loads(match.group(1))
            assert payload["@context"] == "https://schema.org"
            assert payload["@type"] == "WebPage"
            assert payload["name"].endswith("— GNN Pipeline")
            assert payload["description"].startswith("GNN Pipeline Results")
            assert payload["isPartOf"] == {
                "@type": "WebSite",
                "name": "GNN Pipeline Results",
            }
