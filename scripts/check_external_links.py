#!/usr/bin/env python3
"""Check external (http/https) URLs referenced in maintained docs.

The in-repo audit (docs/development/docs_audit.py) validates *relative* links
only. This script extends coverage to external URLs so moved/retired resources
can be caught before they rot.

Run from repository root:

  uv run python scripts/check_external_links.py            # report only
  uv run python scripts/check_external_links.py --fail-on-dead
  uv run python scripts/check_external_links.py --strict   # fail on every unexempted error
  uv run python scripts/check_external_links.py --concurrency 8 --timeout 15

Design notes:

- **Informational by default.** External sites rate-limit or bot-block
  automated clients, so transient or blocked responses remain inconclusive.
  The docs-audit workflow uses ``--fail-on-dead`` on its weekly schedule,
  version tags, and opt-in manual runs. The host allowlist applies to
  bot-blocked responses in ``--strict`` mode.
- Local/template targets (localhost, 127.0.0.1, example.com, ``{host}``
  templates, ``server:port``) are skipped by design — they are configuration
  examples, not links.
- Findings are grouped by status code; 4xx/000 are the interesting ones.
- ``--fail-on-dead`` fails on HTTP 404/410 while retaining blocked, transient
  and timed-out requests as inconclusive diagnostics. It never certifies those
  inconclusive URLs. ``--strict`` retains its stronger all-findings contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_ROOTS = [
    ROOT / ".agent_rules",
    ROOT / ".github",
    ROOT / "docs",
    ROOT / "input",
    ROOT / "manuscript",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "AGENTS.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CLAUDE.md",
    ROOT / "CODE_OF_CONDUCT.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "DOCS.md",
    ROOT / "README.md",
    ROOT / "SECURITY.md",
    ROOT / "SETUP_GUIDE.md",
    ROOT / "SPEC.md",
    ROOT / "SUPPORT.md",
    ROOT / "TO-DO.md",
]

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "other",
    "archive",
    "build",
    "dist",
    "node_modules",
    "output",
    ".benchmarks",
    ".aii",
}

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")

# Hosts/paths that are configuration examples or local services, never links.
SKIP_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "example.com",
    "www.example.com",
    "docs.example.com",
    "gnn.api.example.com",
    "mcp-server.example.com",
    "yourdomain.com",
    "gnn_backends",
    "server",
    "colbert-server",
    "<server-ip",
}
TEMPLATE_MARKERS = ("{", "}", "server:port", "yourdomain")


def _is_generated(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    parts = rel.parts
    if any(p in SKIP_PARTS for p in parts):
        return True
    if any(p.startswith("activeinference_outputs_") for p in parts):
        return True
    if any(p.endswith("_outputs") or "_outputs_" in p for p in parts):
        return True
    return "pomdp_gridworld_outputs" in parts


def _should_skip_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return True
    if host in SKIP_HOSTS:
        return True
    if host == "example.com" or host.endswith(".example.com"):
        return True
    # Angle-bracket templates (<your-username>, <server-ip>) never appear in
    # real URLs; the regex character class stops at '>' so captures may be
    # truncated mid-template.
    if "<" in url or any(m in url for m in TEMPLATE_MARKERS):
        return True
    return False


def _normalize_url(raw: str) -> str:
    """Trim trailing punctuation/backticks and re-balance a cut-off paren.

    The URL regex character class stops at ``)``, truncating links such as
    ``https://learn.microsoft.com/.../ms256108(v=vs.85)`` before the closing
    paren; append it back when the capture is unbalanced.
    """
    url = raw.rstrip(".,;:!?`")
    if url.count("(") > url.count(")"):
        url += ")"
    return url


def collect() -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {}
    for root in MAINTAINED_ROOTS:
        if not root.exists() or _is_generated(root):
            continue
        candidates = (
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file())
        )
        for path in candidates:
            if path.suffix not in {".md", ".txt", ".yaml", ".yml", ".toml", ".cfg"}:
                continue
            if _is_generated(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # Fenced code blocks hold configuration examples (e.g.
            # OPENAI_BASE_URL=https://api.openai.com/v1), not navigable links —
            # strip them before URL extraction to avoid false 404s.
            if path.suffix == ".md":
                text = re.sub(
                    r"^(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$",
                    "",
                    text,
                    flags=re.MULTILINE | re.DOTALL,
                )
            for m in URL_RE.finditer(text):
                url = _normalize_url(m.group(0))
                if _should_skip_url(url):
                    continue
                urls.setdefault(url, []).append(str(path.relative_to(ROOT)))
    return urls


def check(url: str, timeout: int) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return str(resp.status), url
    except urllib.error.HTTPError as exc:
        return str(exc.code), url
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{type(exc).__name__}", url


# 403/429/999 are bot-blocking / rate-limiting classes: the URL is usually
# fine in a real browser (verified: crates.io, medium.com, paperswithcode).
BOT_BLOCKED = {"401", "403", "429", "999"}


def _host_allowed(url: str, allow_hosts: list[str]) -> bool:
    """Whether ``url``'s host matches an allow-listed host or its subdomain."""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return any(
        host == h.lower() or host.endswith("." + h.lower()) for h in allow_hosts if h
    )


def _bot_blocked_codes(
    by_code: dict[str, list[str]], allow_hosts: list[str]
) -> set[str]:
    """Status codes classified as bot-blocking that ``--strict`` must fail on.

    A code is exempt from ``--strict`` only when *every* URL carrying it
    belongs to an allow-listed host (exact or subdomain match) — a mixed
    group (one allow-listed host + one unverified host) stays flagged.
    """
    return {
        code
        for code in by_code
        if code in BOT_BLOCKED
        and not all(_host_allowed(url, allow_hosts) for url in by_code[code])
    }


def _failure_codes(
    by_code: dict[str, list[str]],
    strict: bool,
    fail_on_dead: bool,
    allow_hosts: list[str],
) -> set[str]:
    """Keep observed dead links separate from inconclusive access failures."""
    if strict:
        failures = {
            code
            for code in by_code
            if code.startswith("ERR") or (code.isdigit() and int(code) >= 400)
        }
        return (failures - BOT_BLOCKED) | _bot_blocked_codes(by_code, allow_hosts)
    return set(by_code) & {"404", "410"} if fail_on_dead else set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Fail on every unexempted failed request."
    )
    parser.add_argument(
        "--fail-on-dead",
        action="store_true",
        help="Fail on HTTP 404/410; report other failures as inconclusive.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Parallel checks (default: 16)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-URL timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--min-status",
        type=int,
        default=400,
        help="Report statuses >= this value (default: 400).",
    )
    parser.add_argument(
        "--allow-hosts",
        nargs="*",
        default=[],
        help="Hosts whose bot-blocked (401/403/429/999) responses are exempt "
        "from --strict (e.g. doi.org zenodo.org — verified live, bot-gated).",
    )
    args = parser.parse_args()

    urls = collect()
    print(
        f"check_external_links: {len(urls)} unique external URLs across maintained docs",
        file=sys.stderr,
    )

    results: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(check, url, args.timeout) for url in sorted(urls)]
        for fut in cf.as_completed(futures):
            results.append(fut.result())

    by_code: dict[str, list[str]] = defaultdict(list)
    for code, url in results:
        by_code[code].append(url)

    counts = Counter(code for code, _ in results)
    print("summary:", dict(sorted(counts.items())))
    print()

    bot_blocked = _bot_blocked_codes(by_code, args.allow_hosts)
    if bot_blocked:
        n = sum(len(by_code[c]) for c in bot_blocked)
        codes = ", ".join(sorted(bot_blocked))
        print(
            f"{n} bot-blocked/rate-limited response(s) ({codes}) — verify in a "
            f"browser before acting (use --strict to list them)"
        )
        print()

    interesting = {
        c
        for c in by_code
        if c.startswith("ERR") or (c.isdigit() and int(c) >= args.min_status)
    }
    interesting -= BOT_BLOCKED
    if args.fail_on_dead:
        interesting |= set(by_code) & BOT_BLOCKED
    elif args.strict:
        interesting |= bot_blocked
    dead = 0
    for code in sorted(
        interesting, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0)
    ):
        for url in sorted(by_code[code]):
            refs = ", ".join(sorted(set(urls[url]))[:4])
            extra = (
                f" (+{len(set(urls[url])) - 4} more)" if len(set(urls[url])) > 4 else ""
            )
            print(f"  {code} {url}")
            print(f"      <- {refs}{extra}")
            dead += 1
    if not dead:
        print("no findings above the configured threshold")

    failures = _failure_codes(by_code, args.strict, args.fail_on_dead, args.allow_hosts)
    failed = sum(len(by_code[code]) for code in failures)
    if failed:
        print(f"\ncheck_external_links: {failed} failing finding(s); exiting 1.")
        return 1
    if args.fail_on_dead:
        print("\nNo HTTP 404/410 responses. Other failed requests remain inconclusive;")
        print("this result does not establish that every external URL is accessible.")
        return 0
    print(
        "\n(non-strict: exit 0; use --strict to fail — triage bot-blocked hosts first)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
