"""Stable model navigation, artifact attachment, and search-index assembly."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from gnn.website.pages import SITE_PAGES


def _model_slug(name: str) -> str:
    """Deterministic file slug for a model name (``model`` when empty)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return slug or "model"


def _ensure_model_slugs(models: list[dict[str, Any]]) -> None:
    """Assign unique ``model/<slug>.html`` slugs to parsed models, in order.

    One slug map shared by the per-model pages, the listing deep links, and
    the search index: the first claimant keeps the bare slug, later
    duplicates get ``-2``, ``-3``, … Claim order is the deterministic
    collection order (sorted GNN filenames); pre-assigned slugs win.
    """
    used: set[str] = {m["slug"] for m in models if m.get("slug")}
    for model in models:
        if model.get("slug"):
            continue
        base = _model_slug(str(model.get("model_id") or model.get("name") or ""))
        slug = base
        counter = 2
        while slug in used:
            slug = f"{base}-{counter}"
            counter += 1
        used.add(slug)
        model["slug"] = slug


def _artifact_matches_model(artifact_title: str, model_slug: str) -> bool:
    """True when a collected visualization artifact belongs to a model.

    Step-8/9 artifacts are named ``{model}_{what}``, so the artifact stem
    slugifies to a string starting with the model's slug (equal when the
    artifact carries no suffix, e.g. ``graph.png`` for a single-model run).
    """
    stem_slug = _model_slug(artifact_title)
    return stem_slug == model_slug or stem_slug.startswith(model_slug + "-")


def _model_snippet(model: dict[str, Any]) -> str:
    """Plain-text snippet (≤200 chars, angle-bracket free) for a model page."""
    text = re.sub(r"[<>]", "", " ".join(str(model.get("annotation") or "").split()))
    if not text:
        text = (
            f"Model page — {len(model.get('variables') or [])} variables, "
            f"{len(model.get('edges') or [])} edges — source "
            f"{model.get('source_name') or 'unknown'}"
        )
    if len(text) > 200:
        text = text[:199] + "…"
    return text


def _attach_model_viz_assets(data: dict) -> None:
    """Attach matching collected image assets to each parsed model."""
    visuals = data.get("visualizations") or []
    for model in data.get("models") or []:
        slug = model.get("slug") or _model_slug(str(model.get("name") or ""))
        model["images"] = [
            v["path"]
            for v in visuals
            if v.get("type") == "image"
            and (
                v.get("model_id") == model.get("model_id")
                if model.get("model_id")
                else _artifact_matches_model(str(v.get("title") or ""), str(slug))
            )
        ]


def _build_search_pages(data: dict) -> list[dict[str, str]]:
    """Search-index entries covering the site pages plus every model page."""
    pages = [
        {"title": page.title, "url": page.filename, "snippet": page.description}
        for page in SITE_PAGES
    ]
    for model in data.get("models") or []:
        slug = model.get("slug") or _model_slug(str(model.get("name") or ""))
        pages.append(
            {
                "title": str(model.get("name") or model.get("source_name") or "Model"),
                "url": f"model/{slug}.html",
                "snippet": _model_snippet(model),
            }
        )
    return pages


def _build_search_data(data: dict) -> dict[str, Any]:
    """Full search-index payload: ``generated`` date plus page entries.

    The standalone ``search-index.json`` serves consumers that can fetch it
    (an HTTP server); on ``file://`` fetch() fails, so the gnn_files page
    also inlines this same payload and uses the inline copy at runtime.
    """
    return {
        "generated": datetime.now().date().isoformat(),
        "pages": _build_search_pages(data),
    }
