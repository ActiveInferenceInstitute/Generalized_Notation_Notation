"""Offline page shell, styles, search script and escaping for generated sites.

Presentation depends on the page catalogue, never on collectors or filesystem
publication. Site generators compose their content through this shell.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from typing import Any, Optional

from gnn.website.pages import SITE_PAGES

_CSS = """
/* ── GNN Pipeline Premium Design System ── */

:root {
  --bg-gradient: radial-gradient(circle at top right, #1b1e32, #07090f 80%);
  --bg-surface:  rgba(22, 27, 34, 0.45);
  --bg-card:     rgba(28, 34, 48, 0.6);
  --bg-hover:    rgba(33, 39, 58, 0.85);
  --border:      rgba(255, 255, 255, 0.08);
  --accent:      #a881fb;
  --accent-2:    #00f0ff;
  --success:     #3fb950;
  --warning:     #e3b341;
  --error:       #fc6c65;
  --text-1:      #ffffff;
  --text-2:      #aeb6c2;
  --text-3:      #606a78;
  --radius:      12px;
  --radius-lg:   16px;
  --shadow:      0 8px 32px rgba(0, 0, 0, 0.5);
  --glow:        0 0 24px rgba(168, 129, 251, 0.4);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #07090f;
  background-image: var(--bg-gradient);
  background-attachment: fixed;
  color: var(--text-1);
  min-height: 100vh;
  display: flex;
}

/* ── Sidebar (Glass) ── */
.sidebar {
  width: 260px;
  flex-shrink: 0;
  background: var(--bg-surface);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 0;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  box-shadow: 4px 0 24px rgba(0,0,0,0.2);
}
.sidebar-logo {
  padding: 24px 20px 20px;
  border-bottom: 1px solid var(--border);
}
.sidebar-logo h2 {
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.02em;
}
.sidebar-logo p { color: var(--text-2); font-size: 12px; margin-top: 4px; font-weight: 300; }
.sidebar-nav { padding: 16px 12px; flex: 1; }
.nav-section { margin-bottom: 12px; }
.nav-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-3);
  padding: 4px 12px;
  margin-bottom: 4px;
}
.nav-link {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: var(--radius);
  color: var(--text-2);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.nav-link:hover, .nav-link.active {
  background: rgba(255,255,255,0.05);
  color: var(--text-1);
  transform: translateX(4px);
}
.nav-link.active {
  background: linear-gradient(90deg, rgba(168,129,251,0.15), transparent);
  border-left: 3px solid var(--accent);
  color: var(--text-1);
}
.nav-link .icon { font-size: 16px; width: 22px; text-align: center; }

/* ── Main content ── */
.main {
  flex: 1;
  min-width: 0;
  padding: 40px 48px;
  overflow-x: auto;
  animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Page header ── */
.page-header {
  margin-bottom: 36px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}
.page-header h1 {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.02em;
}
.page-header .subtitle {
  color: var(--text-2);
  font-size: 15px;
  margin-top: 8px;
  font-weight: 300;
}

/* ── Stat cards (Glass) ── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 36px;
}
.stat-card {
  background: var(--bg-card);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: transparent;
  transition: background 0.3s ease;
}
.stat-card:hover { 
  transform: translateY(-6px) scale(1.02);
  box-shadow: var(--glow); 
  border-color: rgba(255,255,255,0.15);
}
.stat-card.success:hover::before { background: var(--success); }
.stat-card.accent:hover::before { background: var(--accent); }
.stat-card.accent2:hover::before { background: var(--accent-2); }

.stat-card .label { font-size: 11px; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.stat-card .value { font-size: 36px; font-weight: 700; margin-top: 8px; letter-spacing: -0.03em; }
.stat-card .sub   { font-size: 12px; color: var(--text-2); margin-top: 4px; font-weight: 300; }

.stat-card.success .value { color: var(--success); text-shadow: 0 0 16px rgba(63,185,80,0.4); }
.stat-card.accent  .value { color: var(--accent); text-shadow: 0 0 16px rgba(168,129,251,0.4); }
.stat-card.accent2 .value { color: var(--accent-2); text-shadow: 0 0 16px rgba(0,240,255,0.4); }

/* ── Step grid ── */
.step-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 40px;
}
.step-card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
  transition: all 0.3s ease;
  cursor: default;
  display: flex;
  flex-direction: column;
}
.step-card:hover {
  border-color: rgba(255,255,255,0.2);
  box-shadow: 0 12px 32px rgba(0,0,0,0.6), var(--glow);
  transform: translateY(-4px);
  background: var(--bg-hover);
}
.step-card .step-num {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-3);
  letter-spacing: 0.15em;
}
.step-card .step-name {
  font-size: 15px;
  font-weight: 600;
  margin: 6px 0 10px;
  color: var(--text-1);
}
.step-card .step-desc {
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.6;
  font-weight: 300;
  flex-grow: 1;
}
.step-card .step-badge {
  display: inline-flex;
  align-items: center;
  margin-top: 16px;
  font-size: 10px;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 20px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  align-self: flex-start;
}

/* Animations for badges */
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}
.badge-ok      { background: rgba(63,185,80,0.15);    color: var(--success); border: 1px solid rgba(63,185,80,0.3); }
.badge-skip    { background: rgba(227,179,65,0.15);   color: var(--warning); border: 1px solid rgba(227,179,65,0.3); }
.badge-error   { background: rgba(252,108,101,0.15);  color: var(--error); border: 1px solid rgba(252,108,101,0.3); }
.badge-pending { 
  background: rgba(255,255,255,0.05); color: var(--text-2); border: 1px solid var(--border);
  animation: pulse 2s infinite ease-in-out; 
}
.badge-warning { background: rgba(227,179,65,0.15); color: var(--warning); border: 1px solid rgba(227,179,65,0.3); }

/* ── Dashboard index (pipeline run) ── */
.dash-meta {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 40px;
}
.dash-meta__item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.dash-meta__label {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.dash-meta__value {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-1);
  font-family: ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace;
}
.dash-art {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 12px;
}
.dash-art__head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.dash-art__name { font-size: 14px; font-weight: 600; color: var(--text-1); }
.dash-art__count { font-size: 11px; color: var(--text-3); margin-left: auto; }
.dash-art__body { margin-top: 10px; }
.dash-files { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.dash-files li {
  font-size: 12px;
  color: var(--text-2);
  font-family: ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace;
}
.dash-files li::before { content: '📄 '; }
.dash-more { font-size: 11px; color: var(--text-3); margin: 6px 0 0; }
.dash-empty { font-size: 12px; color: var(--text-3); font-style: italic; margin: 0; }

/* ── Table (Glass) ── */
.table-wrap { 
  overflow-x: auto; 
  border-radius: var(--radius-lg); 
  border: 1px solid var(--border); 
  background: var(--bg-card);
  backdrop-filter: blur(12px);
}
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th {
  background: rgba(0,0,0,0.2);
  color: var(--text-2);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 14px 18px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
td {
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  color: var(--text-1);
  vertical-align: top;
  font-weight: 300;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.03); }

/* ── Code ── */
pre, code {
  font-family: ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace;
  font-size: 13px;
}
pre {
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  overflow-x: auto;
  line-height: 1.6;
  color: var(--text-2);
  box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
}
code { color: var(--accent-2); }

/* ── Section ── */
.section      { margin-bottom: 40px; }
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
  color: var(--text-1);
  display: flex;
  align-items: center;
  gap: 12px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Cards ── */
.card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  margin-bottom: 16px;
  transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow);
  border-color: rgba(255,255,255,0.15);
}
.card h3 { font-size: 16px; font-weight: 600; margin-bottom: 10px; }
.card p  { font-size: 14px; color: var(--text-2); line-height: 1.6; font-weight: 300; }

/* ── Visualization gallery ── */
.viz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}
.viz-card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
}
.viz-card:hover { 
  box-shadow: 0 16px 40px rgba(0,0,0,0.6), var(--glow); 
  transform: translateY(-6px);
  border-color: rgba(255,255,255,0.2);
}
.viz-card img  { width: 100%; display: block; border-bottom: 1px solid var(--border); }
.viz-card .viz-info { padding: 16px 20px; background: rgba(0,0,0,0.2); flex-grow: 1; }
.viz-card .viz-title { font-size: 14px; font-weight: 600; }
.viz-card .viz-desc  { font-size: 12px; color: var(--text-2); margin-top: 4px; font-weight: 300; }

/* ── MCP tool card ── */
.tool-card {
  background: rgba(0,0,0,0.2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-2);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 12px;
  transition: background 0.2s;
}
.tool-card:hover { background: rgba(255,255,255,0.05); }
.tool-card .tool-name { font-family: ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace; font-size: 14px; color: var(--accent-2); font-weight: 600; }
.tool-card .tool-mod  { font-size: 11px; color: var(--text-3); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.1em; }
.tool-card .tool-desc { font-size: 13px; color: var(--text-2); margin-top: 8px; line-height: 1.6; font-weight: 300; }

/* ── Pill badge ── */
.pill {
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Collapsible (Glass) ── */
details { margin-bottom: 12px; }
summary {
  cursor: pointer;
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  font-size: 14px;
  font-weight: 600;
  list-style: none;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background 0.2s, border-color 0.2s;
}
summary:hover { background: var(--bg-hover); border-color: rgba(255,255,255,0.15); }
summary::-webkit-details-marker { display: none; }
summary::after { content: '▸'; color: var(--text-3); font-size: 16px; transition: transform 0.2s; }
details[open] summary::after { transform: rotate(90deg); }
details[open] summary { border-radius: var(--radius) var(--radius) 0 0; background: rgba(0,0,0,0.3); border-bottom: none; }
.details-body {
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 var(--radius) var(--radius);
  padding: 16px 18px;
  background: rgba(0,0,0,0.2);
  backdrop-filter: blur(12px);
}

/* ── Breadcrumbs ── */
.breadcrumbs { margin-bottom: 24px; }
.breadcrumbs ol {
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0;
  padding: 0;
  font-size: 12px;
}
.breadcrumbs li { color: var(--text-3); }
.breadcrumbs a { color: var(--text-2); text-decoration: none; }
.breadcrumbs a:hover { color: var(--accent-2); text-decoration: underline; }
.breadcrumbs li[aria-current='page'] { color: var(--text-1); font-weight: 600; }

/* ── Site search (client-side filter over the inline search index) ── */
.site-search { margin-bottom: 32px; }
#gnn-site-search {
  width: 100%;
  max-width: 420px;
  padding: 10px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-1);
  font-size: 14px;
}
#gnn-site-search::placeholder { color: var(--text-3); }
#gnn-search-results { margin: 8px 0 0; padding: 0 0 0 4px; list-style: none; }
#gnn-search-results li { padding: 4px 0; font-size: 13px; color: var(--text-2); }
#gnn-search-results a { color: var(--accent-2); text-decoration: none; }
#gnn-search-results a:hover { text-decoration: underline; }
.gnn-search-snippet { color: var(--text-3); font-size: 12px; margin-left: 6px; }

/* ── Responsive ── */
@media (max-width: 768px) {
  .sidebar { position: fixed; transform: translateX(-100%); z-index: 100; transition: transform 0.3s; }
  .sidebar.open { transform: translateX(0); }
  .main { padding: 20px; }
  .page-header h1 { font-size: 26px; }
  .stat-card .value { font-size: 28px; }
}
"""


_SEARCH_JS = """(function () {
  'use strict';
  var dataEl = document.getElementById('gnn-search-data');
  var input = document.getElementById('gnn-site-search');
  var list = document.getElementById('gnn-search-results');
  if (!dataEl || !input || !list) { return; }
  var pages = [];
  try { pages = (JSON.parse(dataEl.textContent) || {}).pages || []; }
  catch (err) { return; }
  function render(matches) {
    list.textContent = '';
    if (matches.length === 0) {
      var empty = document.createElement('li');
      empty.className = 'gnn-search-empty';
      empty.textContent = 'No matching pages.';
      list.appendChild(empty);
      return;
    }
    matches.forEach(function (page) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = page.url;
      a.textContent = page.title;
      li.appendChild(a);
      var snippet = document.createElement('span');
      snippet.className = 'gnn-search-snippet';
      snippet.textContent = page.snippet;
      li.appendChild(snippet);
      list.appendChild(li);
    });
  }
  input.addEventListener('input', function () {
    var query = input.value.trim().toLowerCase();
    if (!query) {
      list.hidden = true;
      list.textContent = '';
      return;
    }
    var matches = [];
    for (var i = 0; i < pages.length; i++) {
      var page = pages[i];
      var haystack = (page.title + ' ' + page.url + ' ' + page.snippet).toLowerCase();
      if (haystack.indexOf(query) !== -1) { matches.push(page); }
    }
    list.hidden = false;
    render(matches);
  });
})();"""


def _page(
    title: str,
    active: str,
    body: str,
    *,
    depth: int = 0,
    breadcrumbs: Optional[list[tuple[Optional[str], str]]] = None,
) -> str:
    """Wrap body in the shared page shell with sidebar, nav, and breadcrumbs.

    ``depth`` is the page's directory depth below the site root (0 for the
    seven root pages, 1 for ``model/<slug>.html`` pages) so every emitted
    href stays relative and file://-safe. ``breadcrumbs`` is an ordered
    list of ``(site-root-relative href or None, label)`` crumbs — ``None``
    marks the current page; when omitted it derives from ``title``.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    root = "../" * depth
    nav_items: list[tuple[str, str, str, str]] = [
        (page.icon, page.title, page.filename, page.name) for page in SITE_PAGES
    ]
    nav_html = ""
    for icon, label, href, key in nav_items:
        cls = "nav-link active" if key == active else "nav-link"
        nav_html += (
            f'<a href="{root}{href}" class="{cls}">'
            f'<span class="icon">{icon}</span>{label}</a>\n'
        )

    if breadcrumbs is None:
        breadcrumbs = (
            [(None, "Home")]
            if active == "index"
            else [("index.html", "Home"), (None, title)]
        )
    crumb_html = ""
    for crumb_href, crumb_label in breadcrumbs:
        if crumb_href is None:
            crumb_html += f'<li aria-current="page">{_esc(crumb_label)}</li>'
        else:
            crumb_html += (
                f'<li><a href="{root}{_esc(crumb_href)}">{_esc(crumb_label)}</a></li>'
            )
    breadcrumb_nav = (
        f'<nav class="breadcrumbs" aria-label="Breadcrumb"><ol>{crumb_html}</ol></nav>'
    )

    description = f"GNN Pipeline Results — {title}"
    # Offline-true structured data: the schema.org context IRI is a
    # vocabulary identifier, never dereferenced at runtime — no external
    # resource is fetched. ``<`` is escaped so payload strings can never
    # close the script element.
    jsonld = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "WebPage",
            "name": f"{title} — GNN Pipeline",
            "description": description,
            "isPartOf": {"@type": "WebSite", "name": "GNN Pipeline Results"},
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{description}">
  <title>{title} — GNN Pipeline</title>
  <script type="application/ld+json">{jsonld}</script>
  <style>{_CSS}</style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-logo">
      <h2>GNN Pipeline</h2>
      <p>Generated {ts}</p>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">
        <div class="nav-label">Navigation</div>
        {nav_html}
      </div>
    </nav>
  </aside>
  <main class="main">
    {breadcrumb_nav}
    {body}
  </main>
</body>
</html>"""


def _esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation into page markup."""
    return escape(str(value))
