# Specification: Website

## Design Requirements

The `src/gnn/website/` module generates static HTML websites from pipeline artifacts (Step 20).

## Interface Mapping

- `20_website.py`: Thin orchestrator binding `website.processor.process_website()`
- `processor.py`: Thin facade re-exporting `renderer.process_website`
- `collection.py`: Pipeline-artifact collectors behind `collect_website_data` (GNN files, step statuses, analysis, visualization assets, reports, MCP page data)
- `generator.py`: Site orchestration and dataset-specific page content producing the 7-page site, per-model detail pages and `search-index.json`; existing public entrypoints and re-exports are preserved
- `templates.py`: Offline page shell, styles, search script and HTML escaping; depends only on the live page catalogue, never on collection or filesystem publication
- `renderer.py`: `process_website`, embedding helpers, and module info
- `mcp.py`: MCP tool registration for website generation operations

## Functional Requirements

- **Static Site Generation**: Produce self-contained HTML websites from pipeline output artifacts
- **Inline Templating**: Pages are built with inline CSS/HTML (no Jinja2 dependency)
- **Visualization Embedding**: Embed graphs and images from Steps 8–9
- **Cross-Referencing**: Link between model pages, execution results, and analysis reports
- **Breadcrumbs**: Every generated page emits a breadcrumb nav in the page shell — `Home › <section>` on the six non-index site pages, a single `Home` crumb on the index, and `Home › GNN Files › <Model Name>` on model pages — with relative, depth-correct, `file://`-safe hrefs and escaped labels
- **Per-Model Pages**: One detail page per parsed model at `model/<slug>.html` per run (slug = lowercased name, every character outside `[a-z0-9]` collapsed to `-` and stripped; empty → `model`; first claimant keeps the slug, later duplicates get `-2`, `-3`, …). Content: model-name `h1`, a source link back to the model's entry on the GNN Files listing, variables/edges tables, embedded visualization assets, and the model's FULL GNN source — no truncation (the 3000-character cap applies only to the aggregate GNN Files listing rows). These pages are generated per model, not site furniture: the `SITE_PAGES` catalogue stays exactly 7 pages, and bookkeeping uses new manifest keys (`model_pages_created`, `model_pages`) instead of `pages`/`pages_created`
- **Client-Side Search**: The generator emits `search-index.json` (`{"generated", "pages": [{"title", "url", "snippet"}]}`, ≤200-char plain-text snippets) covering every emitted page — the 7 site pages plus all model pages — and ships the same payload inline on the GNN Files listing page with a search box and a small vanilla-JS filter. The payload is inlined because `fetch()` fails on `file://`; the standalone JSON exists for same-origin serving via the website server
- **Offline-True + Structured Data**: Generated pages carry zero external resource references (inline CSS/JS; system font stack; no CDN) and every page — site pages and per-model pages — embeds schema.org JSON-LD (`WebPage` with name/description/isPartOf; the `https://schema.org` context IRI is a vocabulary identifier, never dereferenced at runtime) plus a `<meta name="description">`. `website_results.json` is written atomically (temp file + rename, same helper as the pages), and its `warnings` list is live: collection failures (e.g. a visualization artifact whose copy failed is skipped) surface there instead of being linked with dangling hrefs.

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `WebsiteGenerator` | Class | Core site generation engine |
| `WebsiteRenderer` | Class | Template rendering and HTML output |
| `process_website()` | Function | Top-level entry point called by orchestrator |

## Standards

- Generated sites are self-contained (inline CSS/JS; no external CDN)
- HTML5 semantic markup with responsive CSS layouts
- No Jinja2/Markdown/Bleach dependency — stdlib only
- Every generated page emits a breadcrumb nav (`Home › <section>`; model pages: `Home › GNN Files › <Model Name>`; index: a single `Home` crumb) and cross-references
