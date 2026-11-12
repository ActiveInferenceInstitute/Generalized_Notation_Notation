"""Render custody manifest for GeneralizedNotationNotation.

The committed PDF evidence (``output/pdf/_combined_manuscript.log``/``.tex``/
``.md``) and the hydrated sections it was rendered from (
``output/manuscript/``) are four artifacts of one render invocation, but until
now nothing bound them together: every gate compared counts, so a log, a tex
and a prose tree from three different renders could pass together as long as
no count moved (the same collectively-stale failure mode SC-3 fixed for the
token map).

The custody point is a small digest manifest,
``output/data/manuscript_render_manifest.json``, recorded AFTER a render by
``scripts/z_record_manuscript_render_manifest.py``:

* ``variables_sha256`` — the commit-stable checksum of the committed token
  map the render substituted. The strict token gate already pins that map to
  ``HEAD``; this pins the render to the map, so the chain is
  ``HEAD → token map → hydrated prose → committed PDF evidence``.
* ``render_artifacts_sha256`` — sha256 of the committed ``.log``/``.tex``/
  ``.md``. A re-render committed without re-recording the manifest fails;
  so does a manifest re-recorded against artifacts that are not committed.
* ``publication_artifacts`` — exact SHA-256 and byte size of the shipped PDF.
  These identify the publication bytes, not byte-reproducible fresh-render
  evidence. The counts/source commit precedes the eventual artifact commit.
* ``render_inputs_sha256`` — sha256 of every committed hydrated file under
  ``output/manuscript/``. A prose edit that never reached a render shows up
  here as a disagreement with the tree on disk.
* ``counts_describe_commit`` / ``receipt_counts_describe_commit`` — the
  commit short-hash the token map describes and the one the producer's
  receipt (written by the pre-render producer run) describes.
* ``declared_environment`` — the environment the chain describes: the
  template revision the render cloned, and the fully-named render profile
  the tools made. ``recording_profile`` names the profile the render ran
  under; ``permitted_profiles`` holds every profile this chain accepts,
  each with every value spelled out. Declared so a fresh render can be
  asked about the *environment* before it is asked about bytes.

A profile mismatch narrows the comparison; it never widens a tolerance. On
the recording profile every artifact is compared. On a permitted but
different profile the one axis a permitted profile may differ in — the TeX
Live installation — takes its artifact out of the comparison (the ``.log``
is what TeX Live writes), and the run names what it compared and what it
did not. A profile outside the permitted set fails without comparing bytes:
the two axes that could have made the hosted runner differ, pandoc and
pandoc-crossref, are pinned to the recording versions in
``.github/workflows/custody-re-render.yml``, so a difference there is a pin
that drifted, not an environment to tolerate. There is no
artifact-comparison warning level: bytes from a permitted profile that
disagree with the committed chain are drift, on every host, because a
tolerance wide enough to hide a change is not a check.

Ordering honesty: the manifest is recorded after the render, from the files
on disk, so it cannot by itself prove a render ran. That ordering — regen,
rebuild figures, render, record, commit — is the manual half of the SC-22
ritual, stated in ``scripts/z_generate_manuscript_variables.py``'s docstring;
everything downstream of the record step is enforced here and by the strict
token gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

if str(Path(__file__).resolve().parents[3]) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gnn.manuscript.substitution import (  # noqa: E402
    hydrate_text,
    rebase_output_links_for_destination,
)
from gnn.manuscript.variables import token_checksum  # noqa: E402

__all__ = [
    "DECLARED_ENVIRONMENT_KEY",
    "ENGINE_DEPENDENT_ARTIFACTS",
    "HYDRATION_FIX",
    "MANIFEST_VERSION",
    "PERMITTED_PROFILES_KEY",
    "PERMITTED_RENDER_PROFILES",
    "PROFILE_KEYS",
    "RECORDING_PROFILE_KEY",
    "RECORD_COMMAND",
    "RENDER_INPUT_ROOT",
    "RENDERED_ARTIFACTS",
    "STALE_CHAIN",
    "TEMPLATE_REVISION_ENV",
    "UNMEASURED",
    "comparison_scope",
    "custody_issues",
    "environment_report",
    "hydrate_text",
    "hydration_issues",
    "load_render_manifest",
    "manifest_path",
    "mask_log_timestamp",
    "observed_render_environment",
    "record_render_manifest",
    "strip_volatile_tokens",
    "verify_fresh_render",
]

MANIFEST_VERSION = "gnn_render_manifest_v1"
RECORD_COMMAND = "uv run python scripts/z_record_manuscript_render_manifest.py"
#: A chain that describes a different render environment than the fresh
#: one gets its own status, not a generic failure: the render is not
#: necessarily wrong, the *chain* is out of date. Exit code 3 in
#: ``scripts/z_verify_fresh_render.py``.
STALE_CHAIN = "[STALE-CHAIN]"
#: Where a recording or verifying run reports the template checkout it
#: rendered with. The committed artifacts carry no template identity, so
#: this is the only carrier of that axis (rules §36: declare it, or the
#: comparison measures the host).
TEMPLATE_REVISION_ENV = "GNN_TEMPLATE_REVISION"
#: A chain whose environment cannot be measured gets its own status: a gate
#: that cannot fire must not report success. Exit code 4 in
#: ``scripts/z_verify_fresh_render.py``.
UNMEASURED = "[UNMEASURED]"
#: Manifest key holding the environment the chain describes.
DECLARED_ENVIRONMENT_KEY = "declared_environment"
#: Manifest keys naming the profile the chain was recorded under and the
#: profiles it permits, inside :data:`DECLARED_ENVIRONMENT_KEY`.
RECORDING_PROFILE_KEY = "recording_profile"
PERMITTED_PROFILES_KEY = "permitted_profiles"
#: The values one profile is made of. Membership is exact equality on every
#: one of them; there is no family, series or suffix rule, because a rule
#: that admits "3.x" admits a toolchain nobody measured.
PROFILE_KEYS = ("pandoc", "pandoc_crossref", "tex_engine", "tex_roots")
#: The artifacts TeX Live itself writes, out of :data:`RENDERED_ARTIFACTS`.
#: Excluded from the comparison on a permitted profile that is not the
#: recording one, and named in that run's report: TeX Live is the one axis a
#: permitted profile may differ in. The PDF is recorded separately as exact
#: publication bytes, not compared across renders (it embeds build ids).
ENGINE_DEPENDENT_ARTIFACTS = (Path("output") / "pdf" / "_combined_manuscript.log",)
#: The non-recording profile this chain permits, fully named and fully
#: specified: the hosted runner's TeX Live 2023/Debian (values measured from
#: the 2026-11-04 scheduled run's own log), with pandoc and pandoc-crossref
#: pinned to the recording versions by ``custody-re-render.yml``. It exists
#: because the runner cannot install TeX Live 2026, and it is the *only*
#: permitted difference: a profile whose pandoc or crossref differ is refused
#: rather than tolerated. Adding another one is a visible edit here that
#: names every value.
PERMITTED_RENDER_PROFILES: dict[str, dict[str, object]] = {
    "tex-live-2023-debian-pandoc-3.13-crossref-0.3.26": {
        "pandoc": "3.13",
        "pandoc_crossref": "0.3.26",
        "tex_engine": (
            "This is XeTeX, Version 3.141592653-2.6-0.999995 (TeX Live 2023/Debian)"
        ),
        "tex_roots": ["/usr/share/texlive/texmf-dist"],
    },
}

_MANIFEST_REL = Path("output") / "data" / "manuscript_render_manifest.json"
_VARIABLES_REL = Path("output") / "data" / "manuscript_variables.json"
RENDER_INPUT_ROOT = Path("output") / "manuscript"
_RECEIPT_REL = Path("output") / "data" / "manuscript_variables_receipt.json"
_PUBLICATION_REL = Path("output") / "pdf" / "GeneralizedNotationNotation_combined.pdf"
_SOURCE_MANUSCRIPT_REL = Path("manuscript")
# Hydration mirror of ``infrastructure.rendering.manuscript_injection``
# (docxology/template): ``_TOKEN_RE`` is its token pattern, and these are
# the auxiliary files ``write_resolved_manuscript_tree`` copies verbatim
# (``preamble.md`` is substituted first and then overwritten by the copy,
# so verbatim is what ships). Every ``*.bib`` is copied verbatim too.
_VERBATIM_AUX = ("config.yaml", "preamble.md")
HYDRATION_FIX = (
    "rerun the full SC-22 ritual (python -m scripts.manuscript_build_figures, "
    "then the template's stage_03_render, then "
    f"{RECORD_COMMAND}) and commit output/ — regenerating the token map and "
    "re-recording the manifest alone refreshes neither the hydrated prose "
    "nor the PDF evidence"
)
# The committed evidence tests/test_manuscript_latex_log.py reads, plus the
# pandoc markdown source the render converted. aux/bbl/toc are also shipped,
# but no gate reads them as evidence, so they stay out of the custody set.
RENDERED_ARTIFACTS = (
    Path("output") / "pdf" / "_combined_manuscript.log",
    Path("output") / "pdf" / "_combined_manuscript.md",
    Path("output") / "pdf" / "_combined_manuscript.tex",
)


_HOME_PREFIX_RES = (
    re.compile(rb"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(rb"/home/[A-Za-z0-9_.-]+/"),
)


def _sanitize_personal_paths(project_root: Path) -> list[str]:
    """Rewrite personal home prefixes out of the committed render evidence.

    LaTeX/pandoc logs and sources embed absolute machine paths (e.g.
    ``/Users/<user>/Library/texmf/...``) from the render host; the evidence
    is tracked, so sanitize in place before the manifest digests it — the
    certified bytes are then always the shipped bytes. Idempotent.
    """
    changed: list[str] = []
    for rel in RENDERED_ARTIFACTS:
        path = project_root / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        clean = data
        for pattern in _HOME_PREFIX_RES:
            clean = pattern.sub(b"~/", clean)
        if clean != data:
            path.write_bytes(clean)
            changed.append(rel.as_posix())
    return changed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_volatile_tokens(variables: dict[str, str]) -> dict[str, str]:
    """Copy of a token map without commit-varying tokens (checksum input).

    Mirrors ``check_manuscript_tokens._strip_volatile_tokens``: a committed
    map can never record the hash of the commit that carries it.
    """
    return {k: v for k, v in variables.items() if k != "GNN_GIT_COMMIT"}


def manifest_path(project_root: Path) -> Path:
    return project_root / _MANIFEST_REL


def load_render_manifest(project_root: Path) -> dict:
    """Read the custody manifest, or raise with the regen instruction."""
    path = manifest_path(project_root)
    if not path.is_file():
        raise RuntimeError(
            f"{_MANIFEST_REL.as_posix()} is missing — record it after a "
            f"render with: {RECORD_COMMAND}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{_MANIFEST_REL.as_posix()} is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{_MANIFEST_REL.as_posix()} is not a JSON object")
    return manifest


def _render_inputs(project_root: Path) -> dict[str, str]:
    """``{relpath: sha256}`` for every committed file the render hydrated."""
    root = project_root / RENDER_INPUT_ROOT
    return {
        p.relative_to(project_root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def record_render_manifest(project_root: Path) -> dict:
    """Digest the committed render artifacts + inputs and write the manifest.

    Run AFTER the template's ``stage_03_render`` has produced the committed
    PDF evidence and the producer has written the token map, as the last
    step of the SC-22 re-render ritual before committing ``output/``.
    Raises (without writing) when any input is missing or the committed
    receipt disagrees with the token map about which commit the counts
    describe — recording a manifest over those states would bless a
    half-committed render.
    """
    variables_path = project_root / _VARIABLES_REL
    if not variables_path.is_file():
        raise RuntimeError(
            f"{_VARIABLES_REL.as_posix()} is missing — run "
            "scripts/z_generate_manuscript_variables.py"
        )
    variables = json.loads(variables_path.read_text(encoding="utf-8"))
    counts_describe_commit = variables.get("GNN_GIT_COMMIT")
    if not counts_describe_commit:
        raise RuntimeError(
            f"{_VARIABLES_REL.as_posix()} carries no GNN_GIT_COMMIT — the "
            "strict token gate treats that as missing provenance; rerun the "
            "producer from a checkout with git available"
        )

    missing = [
        rel.as_posix()
        for rel in (*RENDERED_ARTIFACTS, _PUBLICATION_REL, RENDER_INPUT_ROOT)
        if not (rel if rel.is_absolute() else project_root / rel).exists()
    ]
    if missing:
        raise RuntimeError(
            "committed render artifacts missing: "
            f"{sorted(set(missing))} — run the template's stage_03_render "
            "before recording the custody manifest"
        )
    sanitized = _sanitize_personal_paths(project_root)

    receipt_path = project_root / _RECEIPT_REL
    receipt_commit: str | None = None
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_commit = receipt.get("counts_describe_commit")
        if receipt_commit != counts_describe_commit:
            raise RuntimeError(
                f"{_RECEIPT_REL.as_posix()} says the counts describe "
                f"{receipt_commit}, but the token map says "
                f"{counts_describe_commit} — rerun "
                "scripts/z_generate_manuscript_variables.py so the receipt "
                "and the map describe the same commit"
            )
    if sanitized:
        print(
            "[render-custody] sanitized personal home prefixes out of: "
            f"{', '.join(sanitized)}"
        )

    declared = observed_render_environment(project_root)
    declared_values = {key: declared[key] for key in PROFILE_KEYS}
    profile_name = _profile_name(declared_values)
    permitted_profiles: dict[str, dict[str, object]] = dict(PERMITTED_RENDER_PROFILES)
    for known, values in sorted(permitted_profiles.items()):
        if all(values.get(key) == declared_values[key] for key in PROFILE_KEYS):
            profile_name = known
            break
    permitted_profiles[profile_name] = declared_values
    for key in PROFILE_KEYS:
        if not declared_values[key]:
            print(
                f"[render-custody] note: {key} is unavailable on this host - "
                "recorded as unavailable and compared by value, so a host "
                "that measures it cannot drift past this chain"
            )
    if not declared["template_revision"]:
        print(
            f"[render-custody] note: {TEMPLATE_REVISION_ENV} is unset - this "
            "chain records no template revision, and the fresh-render gate "
            f"will report {UNMEASURED} until it is re-recorded with that "
            "variable exported (the workflow exports it: "
            "git -C template rev-parse HEAD)"
        )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "recorded_by": RECORD_COMMAND,
        "counts_describe_commit": counts_describe_commit,
        "receipt_counts_describe_commit": receipt_commit,
        # The axes this chain declares. Recorded rather than inferred at
        # compare time: a comparison that normalized an undeclared axis
        # would measure the render host, not the chain (rules §36).
        DECLARED_ENVIRONMENT_KEY: {
            "template_revision": declared["template_revision"],
            RECORDING_PROFILE_KEY: profile_name,
            PERMITTED_PROFILES_KEY: permitted_profiles,
        },
        "variables_sha256": token_checksum(strip_volatile_tokens(variables)),
        "render_artifacts_sha256": {
            rel.as_posix(): _sha256(project_root / rel) for rel in RENDERED_ARTIFACTS
        },
        "publication_artifacts": {
            _PUBLICATION_REL.as_posix(): {
                "sha256": _sha256(project_root / _PUBLICATION_REL),
                "size_bytes": (project_root / _PUBLICATION_REL).stat().st_size,
            }
        },
        "render_inputs_sha256": _render_inputs(project_root),
        # Preserve a trustworthy normalized baseline outside Git as well.
        # Never substitute the fresh file for an unavailable committed blob.
        "render_artifacts_normalized_sha256": {
            rel.as_posix(): _artifact_digest_from_bytes(
                rel.as_posix(),
                (project_root / rel).read_bytes(),
                (counts_describe_commit,),
            )
            for rel in RENDERED_ARTIFACTS
        },
        "render_inputs_normalized_sha256": {
            rel: _normalized_text_digest(project_root / rel, (counts_describe_commit,))
            for rel in _render_inputs(project_root)
        },
    }
    path = manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


# The stable part of the engine banner: the engine, its version and the TeX
# Live installation identifier. The rest of the line — the preloaded format's
# build date and the run's start time — varies per run, so a profile value
# that included it could never match twice and would have to be normalized,
# which is exactly the tolerance this design refuses.
_TEX_ENGINE_RE = re.compile(
    r"^This is (?:XeTeX|LuaTeX|pdfTeX)\b[^\n(]*(?:\([^\n)]*\))?", re.MULTILINE
)
# The TeX installation roots the log names: /usr/local/texlive/<year> for a
# from-source install, /usr/share/texlive/<tree> for Debian's packages (the
# hosted runner), .../Library/TeX/... for MacTeX.
_TEX_ROOT_RE = re.compile(
    r"/(?:usr/(?:local/|share/)?texlive|Library/TeX)/[0-9A-Za-z._-]+"
)


def _commit_equal(a: str, b: str) -> bool:
    """Whether two revisions name the same commit at any abbreviation."""
    return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))


def observed_render_environment(project_root: Path) -> dict[str, object]:
    """The environment axes the render ran in, as far as they are observable.

    Measured against the committed log rather than assumed: the engine
    banner is TeX's own ``This is XeTeX, Version ...`` line, and a
    banner this parser does not recognize stays empty and is reported,
    never guessed.

    ``template_revision`` comes from ``$GNN_TEMPLATE_REVISION`` because the
    committed artifacts carry no template identity: nothing in the manifest,
    the ``.tex``, the ``.log`` or the ``.md`` names ``docxology/template`` or
    any of its revisions (measured 2026-11-04), while the render stage clones
    the template at its **default-branch head** on purpose — that clone is
    the drift detector. ``tex_engine``/``tex_roots`` come from the committed
    log's engine banner, which the render host wrote.

    An unobservable axis stays empty here and is *reported* by
    :func:`environment_report`; it is never assumed equal.
    """
    log = project_root / RENDERED_ARTIFACTS[0]
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    engine = _TEX_ENGINE_RE.search(text)
    observed: dict[str, object] = {
        "template_revision": os.environ.get(TEMPLATE_REVISION_ENV, "").strip(),
        "tex_engine": engine.group(0).strip() if engine else "",
        "tex_roots": sorted(set(_TEX_ROOT_RE.findall(text))),
    }
    observed.update(_probe_toolchain())
    return observed


def _probe_toolchain() -> dict[str, str]:
    """The pandoc and pandoc-crossref versions this host has, ``""`` when absent.

    Probed, not declared: these are the two versions ``custody-re-render.yml``
    pins to the recording ones, so a difference here is a pin that drifted and
    the comparison has to be able to say so. An absent binary reports the
    empty string — reported as unavailable, and equal only to another empty
    string, so it can never agree with a measured version.
    """
    probes = (
        ("pandoc", ["pandoc", "--version"], re.compile(r"^pandoc\s+([0-9][^\s]*)")),
        (
            "pandoc_crossref",
            ["pandoc-crossref", "--version"],
            re.compile(r"^pandoc-crossref\s+v?([0-9][^\s]*)"),
        ),
    )
    found: dict[str, str] = {}
    for key, argv, pattern in probes:
        try:
            result = subprocess.run(argv, capture_output=True, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
            found[key] = ""
            continue
        match = (
            pattern.search(result.stdout.decode("utf-8", "replace"))
            if result.returncode == 0
            else None
        )
        found[key] = match.group(1) if match else ""
    return found


def _profile_name(values: dict[str, object]) -> str:
    """A readable name derived from a profile's own values.

    Derived, never assigned, so the name cannot disagree with the values it
    names: the TeX Live identifier, the pandoc version, the pandoc-crossref
    version.
    """
    engine = str(values.get("tex_engine") or "")
    texlive = re.search(r"\((TeX Live [^)]*)\)", engine)
    leaf = texlive.group(1) if texlive else "texlive-unavailable"
    slug = re.sub(r"[^0-9A-Za-z.]+", "-", leaf).strip("-").lower()
    return (
        f"{slug}-pandoc-{values.get('pandoc') or 'unavailable'}"
        f"-crossref-{values.get('pandoc_crossref') or 'unavailable'}"
    )


def _declared(manifest: dict) -> dict:
    """The chain's declared environment, or an empty dict when it has none."""
    declared = manifest.get(DECLARED_ENVIRONMENT_KEY)
    return declared if isinstance(declared, dict) else {}


def _profile_axes(
    manifest: dict,
) -> tuple[str, dict[str, object], dict[str, dict[str, object]]]:
    """``(recording_name, recording_values, permitted_profiles)``.

    Empty name and values when the chain declares no usable profile — a bare
    v1 manifest, or a declared name that is not one of its own permitted
    profiles. A chain whose name and values disagree cannot be asked about a
    fresh render, and :func:`verify_fresh_render` reports that as unmeasured
    rather than guessing which side to trust.
    """
    declared = _declared(manifest)
    raw = declared.get(PERMITTED_PROFILES_KEY)
    if not isinstance(raw, dict):
        return "", {}, {}
    permitted: dict[str, dict[str, object]] = {
        str(name): dict(values)
        for name, values in raw.items()
        if isinstance(values, dict)
    }
    name = str(declared.get(RECORDING_PROFILE_KEY) or "")
    values = permitted.get(name)
    if not isinstance(values, dict):
        return "", {}, permitted
    return name, values, permitted


def _matching_profile(
    observed: dict[str, object], permitted: dict[str, dict[str, object]]
) -> str | None:
    """The name of the permitted profile ``observed`` equals on every value."""
    for name, values in sorted(permitted.items()):
        if all(observed.get(key) == values.get(key) for key in PROFILE_KEYS):
            return name
    return None


def comparison_scope(project_root: Path, manifest: dict) -> str:
    """One line naming what a comparison on this host covers, and why.

    The narrowing a non-recording profile buys is the one thing a green run
    must not hide: the line names the artifacts compared, counts the render
    inputs, and names the artifact left out and the axis that put it out.
    """
    recorded_name, recorded, permitted = _profile_axes(manifest)
    observed = observed_render_environment(project_root)
    observed_values = {key: observed[key] for key in PROFILE_KEYS}
    inputs = len(manifest.get("render_inputs_sha256", {}) or {})
    matched = _matching_profile(observed_values, permitted) if recorded else None
    names = ", ".join(rel.name for rel in RENDERED_ARTIFACTS)
    if matched is not None and observed_values != recorded:
        compared = [
            rel for rel in RENDERED_ARTIFACTS if rel not in ENGINE_DEPENDENT_ARTIFACTS
        ]
        excluded = ", ".join(rel.name for rel in ENGINE_DEPENDENT_ARTIFACTS)
        return (
            f"profile {_profile_name(observed_values)!r} is permitted but is not "
            f"the recording profile {recorded_name!r}: comparing "
            f"{len(compared)} of {len(RENDERED_ARTIFACTS)} rendered artifacts "
            f"({', '.join(rel.name for rel in compared)}) and {inputs} render "
            f"inputs; {excluded} not compared on "
            "this host (the TeX Live engine is the one axis a permitted profile "
            "may differ in, and it writes that file)"
        )
    return (
        f"comparing {len(RENDERED_ARTIFACTS)} rendered artifacts ({names}) and "
        f"{inputs} render inputs"
    )


def _template_gate(manifest: dict, project_root: Path) -> str | None:
    """The ``[STALE-CHAIN]`` finding when the chain describes another template.

    Runs BEFORE any byte comparison, and its finding replaces the comparison:
    once the template has moved, every figure link in the hydrated prose is
    expected to move with it, so a byte-level verdict would report the
    artifact and bury the cause — the 2026-11-03 red run said only "your chain
    is stale for HEAD" and named no axis. The template revision is the signal,
    so it is never normalized (rules §36, second clause); when either side
    cannot observe it, the gate reports that through
    :func:`environment_report` and leaves the verdict to the byte comparison.
    """
    declared = manifest.get(DECLARED_ENVIRONMENT_KEY)
    if not isinstance(declared, dict):
        return None
    recorded = str(declared.get("template_revision") or "")
    observed = str(observed_render_environment(project_root)["template_revision"])
    if not recorded or not observed or _commit_equal(recorded, observed):
        return None
    return (
        f"{STALE_CHAIN} template moved {recorded} to {observed}; the chain "
        f"describes {recorded}; re-record is required. The figure links in "
        "the hydrated prose are expected to differ, so the artifact "
        f"comparison is skipped: rerun the SC-22 ritual, then {RECORD_COMMAND}"
    )


def environment_report(project_root: Path, manifest: dict) -> list[str]:
    """What this comparison can and cannot say about the render environment.

    Printed on every run, pass or fail — a chain that declares no environment
    is *reported* as unmeasured, never silently normalized into agreement, and
    a permitted-but-different profile is *reported* as a narrowed comparison,
    so a green run is never read as a fuller verification than it was. A
    v1 manifest hits the first branch, which is why the declaration is
    additive rather than a new manifest version.
    """

    def _show(value: object) -> str:
        if isinstance(value, list):
            return ",".join(str(item) for item in value) or "none"
        return str(value) if value else "unavailable"

    observed = observed_render_environment(project_root)
    declared = _declared(manifest)
    recorded_name, recorded, permitted = _profile_axes(manifest)
    lines = [
        "observed: "
        + ", ".join(f"{key}={_show(observed[key])}" for key in PROFILE_KEYS),
        "normalized on both sides by the byte comparison: commit stamps "
        "(abbreviation length only) and the TeX banner's run-time timestamp",
    ]
    if not recorded:
        lines.append(
            "the chain declares no environment (v1 manifest): the "
            "template-revision gate and the profile comparison are "
            f"UNAVAILABLE here — re-record ({RECORD_COMMAND}) declares them"
        )
    else:
        lines.append(
            f"recording profile {recorded_name!r}: "
            + ", ".join(f"{key}={_show(recorded[key])}" for key in PROFILE_KEYS)
        )
        lines.append(
            f"permitted profiles ({len(permitted)}): " + ", ".join(sorted(permitted))
        )
    if not observed["template_revision"]:
        lines.append(
            f"{TEMPLATE_REVISION_ENV} is unset: the template revision this "
            "render used is unmeasured, so the gate cannot fire"
        )
    elif recorded and not _commit_equal(
        str(declared.get("template_revision") or ""),
        str(observed["template_revision"]),
    ):
        lines.append(
            "declared and observed template_revision differ - see the "
            "gate's [STALE-CHAIN] finding"
        )
    lines.append(comparison_scope(project_root, manifest))
    return lines


def _input_tree_issues(
    project_root: Path, manifest: dict, stamps: tuple[str, ...] = ()
) -> list[str]:
    """One message per disagreement between the input tree and the manifest.

    Shared by :func:`custody_issues` (which reports the messages verbatim)
    and :func:`verify_fresh_render` (which reads an empty list as
    "``render_inputs_sha256`` fully matches the tree on disk"): recorded
    inputs that are gone or changed since the render the manifest
    describes, and files in the tree the manifest does not record.

    With ``stamps`` (the fresh-comparison mode), a recorded input is
    "changed" only when its PROSE changed: both sides are digested with
    commit-relative stamps masked, so a re-render at a new HEAD that
    stamps the identical prose with the new commit is not drift. The
    committed side is re-derived from ``git show`` at HEAD and masked;
    without git (the unit fixtures) the stored raw digest stands in, and
    fixture text carries no hex runs, so raw and masked agree. With
    ``stamps`` empty (the committed-chain mode used by
    :func:`custody_issues`) the comparison stays raw: the committed chain
    must be byte-identical to the render the manifest recorded.
    """
    issues: list[str] = []
    recorded_inputs: dict[str, str] = manifest.get("render_inputs_sha256", {})
    for rel, digest in sorted(recorded_inputs.items()):
        path = project_root / rel
        if not path.is_file():
            issues.append(f"{rel} is gone but the custody manifest records it")
        elif _committed_input_digest(
            project_root,
            rel,
            manifest.get("render_inputs_normalized_sha256", {}).get(rel, digest)
            if stamps
            else digest,
            stamps,
        ) != (_normalized_text_digest(path, stamps) if stamps else _sha256(path)):
            issues.append(
                f"{rel} changed after the render the custody manifest "
                "describes — the committed PDF evidence is stale for this "
                "prose; rerun the SC-22 ritual"
            )
    for path in sorted((project_root / RENDER_INPUT_ROOT).rglob("*")):
        if (
            path.is_file()
            and path.relative_to(project_root).as_posix() not in recorded_inputs
        ):
            issues.append(
                f"{path.relative_to(project_root).as_posix()} is not recorded "
                "in the custody manifest — re-record with: "
                f"{RECORD_COMMAND} (and rerun the ritual if it is new prose)"
            )
    return issues


def custody_issues(project_root: Path) -> list[str]:
    """One message per disagreement between the committed chain and the manifest.

    Returns (does not raise) so a test can collect every stale artifact at
    once; a missing manifest is the first issue and carries the regen
    command.
    """
    try:
        manifest = load_render_manifest(project_root)
    except RuntimeError as exc:
        return [str(exc)]

    issues: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        issues.append(
            f"manifest_version {manifest.get('manifest_version')!r} != "
            f"{MANIFEST_VERSION!r} — re-record with: {RECORD_COMMAND}"
        )

    variables_path = project_root / _VARIABLES_REL
    if not variables_path.is_file():
        issues.append(
            f"{_VARIABLES_REL.as_posix()} is missing — run "
            "scripts/z_generate_manuscript_variables.py"
        )
    else:
        variables = json.loads(variables_path.read_text(encoding="utf-8"))
        commit = variables.get("GNN_GIT_COMMIT")
        if manifest.get("counts_describe_commit") != commit:
            issues.append(
                "custody manifest says the render described commit "
                f"{manifest.get('counts_describe_commit')!r}, but the "
                f"committed token map describes {commit!r} — the token map "
                "was regenerated after the render; rerun the SC-22 ritual "
                f"({RECORD_COMMAND} is only the last step)"
            )
        live_sum = token_checksum(strip_volatile_tokens(variables))
        if manifest.get("variables_sha256") != live_sum:
            issues.append(
                "custody manifest variables_sha256 "
                f"{manifest.get('variables_sha256', '')[:12]} != committed "
                f"token map {live_sum[:12]} — the token map moved past the "
                f"render; rerun the SC-22 ritual (re-record alone with "
                f"{RECORD_COMMAND} cannot fix this)"
            )

    receipt_path = project_root / _RECEIPT_REL
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        recorded = manifest.get("receipt_counts_describe_commit")
        if recorded is not None and receipt.get("counts_describe_commit") != recorded:
            issues.append(
                f"{_RECEIPT_REL.as_posix()} was rewritten after the custody "
                f"manifest: receipt commit {receipt.get('counts_describe_commit')!r} "
                f"!= recorded {recorded!r} — re-record with: {RECORD_COMMAND}"
            )

    for rel, digest in sorted(manifest.get("render_artifacts_sha256", {}).items()):
        path = project_root / rel
        if not path.is_file():
            issues.append(f"{rel} is gone but the custody manifest records it")
        elif _sha256(path) != digest:
            issues.append(
                f"{rel} changed after the custody manifest was recorded — "
                f"re-render and re-record: {RECORD_COMMAND}"
            )
    for rel in RENDERED_ARTIFACTS:
        if rel.as_posix() not in manifest.get("render_artifacts_sha256", {}):
            issues.append(f"{rel.as_posix()} is not recorded in the custody manifest")

    issues.extend(_input_tree_issues(project_root, manifest))
    return issues


def _expected_hydrated_tree(
    project_root: Path, variables: dict[str, str], excluded: frozenset[str]
) -> dict[str, bytes]:
    """``{relpath: bytes}`` the injector would write from this token map.

    Substitution alone is not the injector's output: the injector's write path
    also re-expresses ``../output/<path>`` link targets for the copy's location
    under ``output/manuscript/`` (one level below the project root), leaving the
    source files' own form alone. Comparing against unrebased text reported the
    four figure-bearing manuscripts as drift on 2026-11-04 while the render
    itself was correct, so the expectation mirrors that step too.
    """
    source = project_root / _SOURCE_MANUSCRIPT_REL
    out: dict[str, bytes] = {}
    for md in sorted(source.glob("*.md")):
        if md.name in excluded or md.name in _VERBATIM_AUX:
            continue
        text = rebase_output_links_for_destination(
            hydrate_text(md.read_text(encoding="utf-8"), variables)
        )
        out[(RENDER_INPUT_ROOT / md.name).as_posix()] = text.encode("utf-8")
    verbatim = [source / name for name in _VERBATIM_AUX] + sorted(source.glob("*.bib"))
    for path in verbatim:
        if path.is_file():
            out[(RENDER_INPUT_ROOT / path.name).as_posix()] = path.read_bytes()
    return out


def hydration_issues(project_root: Path, excluded: frozenset[str]) -> list[str]:
    """One message per hydrated file that the committed token map would not produce.

    The PR-time half of the SC-22 custody check, and it needs no LaTeX. A
    count-changing PR regenerates the token map and re-records the custody
    manifest, and :func:`custody_issues` then passes. The hydrated prose
    under ``output/manuscript/`` and the PDF evidence rendered from it are
    only refreshed by the full ritual, so the scheduled
    ``custody-re-render.yml`` goes red after the merge. The stale prose is
    visible without rendering: it is not what the injector would write from
    the committed map. This re-substitutes ``manuscript/`` from that map and
    compares byte for byte, commit stamp included, because the cron's fresh
    comparison masks only the manifest's stamp and HEAD's. Prose carrying an
    older commit is drift there too.

    Both rendered Markdown and TeX must carry that same commit stamp, closing
    the prose-only hydration bypass. This is a necessary custody condition;
    stamps alone do not certify that the renderer actually ran.

    ``excluded`` is the injector's ``EXCLUDED_DOC_FILENAMES`` (the caller
    passes the shared active contract from ``substitution.py``).
    Returns (does not raise) so every stale file is reported at once; the
    remedy for all of them is :data:`HYDRATION_FIX`.
    """
    variables_path = project_root / _VARIABLES_REL
    if not variables_path.is_file():
        return [
            f"{_VARIABLES_REL.as_posix()} is missing — run "
            "scripts/z_generate_manuscript_variables.py"
        ]
    try:
        variables = json.loads(variables_path.read_text(encoding="utf-8"))
        if not isinstance(variables, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in variables.items()
        ):
            raise ValueError("expected an object containing string token values")
    except (OSError, ValueError) as exc:
        return [f"{_VARIABLES_REL.as_posix()} is unreadable: {exc}"]
    expected = _expected_hydrated_tree(project_root, variables, excluded)
    commit = variables.get("GNN_GIT_COMMIT")
    issues: list[str] = []
    for rel, data in sorted(expected.items()):
        path = project_root / rel
        if not path.is_file():
            issues.append(f"{rel} is missing from the hydrated tree")
        elif path.read_bytes() != data:
            issues.append(
                f"{rel} is not what the committed token map (commit {commit}) "
                "hydrates to"
            )
    root = project_root / RENDER_INPUT_ROOT
    for path in sorted(root.rglob("*")) if root.is_dir() else ():
        rel = path.relative_to(project_root).as_posix()
        if path.is_file() and rel not in expected:
            issues.append(
                f"{rel} has no source under {_SOURCE_MANUSCRIPT_REL.as_posix()}/ "
                "(the injector deletes stale copies)"
            )
    if not commit or re.fullmatch(r"[0-9a-f]{7,40}", commit) is None:
        issues.append(f"{_VARIABLES_REL.as_posix()} carries no valid GNN_GIT_COMMIT")
    else:
        for rendered_rel in RENDERED_ARTIFACTS[1:]:
            path = project_root / rendered_rel
            if not path.is_file():
                issues.append(
                    f"{rendered_rel.as_posix()} is missing rendered commit evidence"
                )
            elif not _text_contains_stamp(path.read_text(encoding="utf-8"), commit):
                issues.append(
                    f"{rendered_rel.as_posix()} does not carry token map commit {commit} — prose hydration alone is not a render"
                )
    return issues


def verify_fresh_render(project_root: Path) -> list[str]:
    """Classify the fresh render on disk against the committed custody manifest.

    The SC-22 cron (``.github/workflows/custody-re-render.yml``) re-renders
    at HEAD and then records a fresh manifest, but recording alone would
    bless whatever it finds: this check runs BEFORE the record step and
    compares the fresh bytes to the manifest as committed, so a red run
    certifies the committed chain, not just render success. Per rendered
    artifact:

    * ``[STALE-CHAIN]`` — the chain declares a different template
      revision than this render observed; the finding is returned alone
      and the artifact comparison is skipped, because the template *is*
      the signal and every figure link is expected to move with it.
    * ``[FAIL]`` — the artifact drifted. With the render inputs matching
      this chain, the same prose rendered to different bytes under a
      *permitted* profile: that is drift, not environment, and it has no
      warning level. With the inputs drifted too, the committed chain is
      stale for HEAD.
    * ``[UNMEASURED]`` — the chain declares no usable render profile (a
      v1 manifest), or the environment does not supply the template
      revision this render used, so the gate cannot fire. Exit code 4: a
      switch that is off must not report success.

    A profile mismatch narrows the comparison instead of widening a
    tolerance. On the recording profile every artifact is compared; on a
    permitted but different profile the artifacts that depend on the one
    axis a permitted profile may differ in — the TeX Live installation,
    which writes the ``.log`` — are excluded, named and counted by
    :func:`comparison_scope`; and a profile outside the permitted set fails
    without comparing bytes at all, because bytes from an undeclared
    toolchain are not evidence about this chain.

    Commit-relative stamps are normalized on both sides before digesting,
    and since 2026-11-04 the producer stamps a fixed-width abbreviation
    (:mod:`gnn.manuscript.snapshot`), so no host can reflow an artifact it
    already laid out by stamping a different length. The committed side is
    re-derived from ``git show`` at HEAD with the manifest's stamp masked;
    outside a git repo (the unit fixtures) the stored raw digest is used
    when the text carries no stamp. The ``.log`` also has its TeX engine
    banner's start timestamp masked (:func:`mask_log_timestamp`): it is the
    one line that differs between two renders of identical inputs, and left
    raw it turned every clean re-render into drift.

    Read-only. Returns (does not raise) so the caller can print every
    finding at once; a missing manifest is the first ``[FAIL]``.
    """
    try:
        manifest = load_render_manifest(project_root)
    except RuntimeError as exc:
        return [f"[FAIL] {exc}"]

    gate = _template_gate(manifest, project_root)
    if gate is not None:
        return [gate]

    stamp = manifest.get("counts_describe_commit") or ""
    stamps = tuple(dict.fromkeys(s for s in (stamp, _head_stamp(project_root)) if s))

    def _normalized(rel: Path) -> str | None:
        path = project_root / rel
        if not path.is_file():
            return None
        return _artifact_digest_from_bytes(rel.as_posix(), path.read_bytes(), stamps)

    def _committed_normalized(rel_posix: str, recorded: str | None) -> str | None:
        """Digest of the artifact AS COMMITTED, at the fresh side's normalization.

        With git, HEAD's bytes are re-digested at the fresh side's
        normalization. Without git (the unit fixtures) the committed bytes
        are unknowable once the disk file is mutated, so the STORED digest —
        which is exactly what was committed — is the committed side; the
        fresh side's stamp masking is a no-op there (fixture text carries no
        hex runs), so raw-vs-normalized agree.
        """
        committed = _git_show(project_root, rel_posix)
        if committed is not None:
            return _artifact_digest_from_bytes(rel_posix, committed, stamps)
        normalized: str | None = manifest.get(
            "render_artifacts_normalized_sha256", {}
        ).get(rel_posix, recorded)
        return normalized

    input_issues = _input_tree_issues(project_root, manifest, stamps)
    inputs_match = not input_issues
    recorded_artifacts: dict[str, str] = manifest.get("render_artifacts_sha256", {})
    issues: list[str] = [f"[FAIL] {issue}" for issue in input_issues]
    for rel in RENDERED_ARTIFACTS:
        rel_posix = rel.as_posix()
        if not (project_root / rel).is_file():
            issues.append(
                f"[FAIL] {rel_posix} is missing — the custody manifest "
                "records it but the fresh render did not produce it"
            )
        elif rel_posix not in recorded_artifacts:
            issues.append(
                f"[FAIL] {rel_posix} is not recorded in the custody "
                f"manifest — re-record with: {RECORD_COMMAND}"
            )

    observed = observed_render_environment(project_root)
    declared = _declared(manifest)
    recorded_name, recorded, permitted = _profile_axes(manifest)
    excluded: tuple[Path, ...] = ()
    compare = True
    if not recorded:
        issues.append(
            f"{UNMEASURED} the committed chain declares no usable render profile "
            "(v1 manifest, or its recording profile missing from its own "
            "permitted set): the template-revision gate and the profile "
            "comparison are unavailable, and a chain that cannot describe its "
            f"environment cannot be asked whether a render matched it — "
            f"re-record with: {RECORD_COMMAND}"
        )
    elif not observed["template_revision"]:
        declared_revision = str(declared.get("template_revision") or "")
        issues.append(
            f"{UNMEASURED} {TEMPLATE_REVISION_ENV} is unset: the template "
            "revision this render used is unmeasured, so the gate cannot "
            "fire. Export it (the workflow does: git -C template rev-parse "
            "HEAD) and re-run: "
            + (
                f"the chain declares {declared_revision!r}, so a template move "
                "would otherwise pass unremarked."
                if declared_revision
                else "the chain declares no template revision either, so a "
                "template move could not be detected at all."
            )
        )
    elif all((project_root / rel).is_file() for rel in ENGINE_DEPENDENT_ARTIFACTS):
        observed_values = {key: observed[key] for key in PROFILE_KEYS}
        matched = _matching_profile(observed_values, permitted)
        if matched is None:
            differences = "; ".join(
                f"{key}={observed_values[key]!r}"
                for key in PROFILE_KEYS
                if all(
                    observed_values[key] != values.get(key)
                    for values in permitted.values()
                )
            )
            issues.append(
                f"[FAIL] render profile {_profile_name(observed_values)!r} is "
                f"not permitted: {differences or 'no permitted profile matches'}"
                f". Permitted: {sorted(permitted)}. pandoc and pandoc-crossref "
                "are pinned to the recording versions by "
                "custody-re-render.yml and the TeX Live engine is the one axis "
                "a permitted profile may differ in, so a difference anywhere "
                "else is a pin that drifted: pin the tool back, or declare the "
                "new profile as a visible edit that names every value. The "
                "artifact comparison is skipped — bytes from an undeclared "
                "toolchain are not evidence about this chain."
            )
            compare = False
        elif observed_values != recorded:
            excluded = ENGINE_DEPENDENT_ARTIFACTS

    if compare:
        for rel in RENDERED_ARTIFACTS:
            rel_posix = rel.as_posix()
            recorded_digest = recorded_artifacts.get(rel_posix)
            if not (project_root / rel).is_file() or recorded_digest is None:
                continue
            if rel in excluded:
                continue
            if _committed_normalized(rel_posix, recorded_digest) == _normalized(rel):
                continue
            scope = (
                "every compared artifact" if not excluded else "the narrowed comparison"
            )
            issues.append(
                f"[FAIL] {rel_posix} differs from the committed custody "
                f"manifest under profile {recorded_name!r} — "
                + (
                    "the render inputs match this chain, so the same prose "
                    "produced different bytes, and a mismatch between two "
                    "recorded sides is drift, not environment"
                    if inputs_match
                    else "the render inputs drifted too, so the committed "
                    "chain is stale for HEAD"
                )
                + f" ({scope}); rerun the SC-22 ritual "
                f"({RECORD_COMMAND} alone is only the last step)"
            )
    return issues


def _text_contains_stamp(text: str, stamp: str) -> bool:
    """Whether any hex run in ``text`` matches ``stamp`` at any abbreviation."""
    return any(
        token == stamp or token.startswith(stamp) or stamp.startswith(token)
        for token in _COMMIT_HASH_RE.findall(text)
    )


def _head_stamp(project_root: Path) -> str:
    """The current HEAD's full sha, or ``""`` outside git.

    Abbreviation-tolerant masking key: the fresh render stamps prose with
    whatever length ``git rev-parse --short`` produced in THIS checkout
    (7 in a shallow CI clone, 9 locally), and any hex prefix of the full
    sha masks via the replacer's prefix rule.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", "replace").strip()


def _committed_input_digest(
    project_root: Path, rel_posix: str, recorded: str, stamps: tuple[str, ...]
) -> str:
    """Digest of a render input AS COMMITTED, at the fresh normalization.

    Mirrors the artifact-side rule: with git, HEAD's bytes are re-digested
    with commit-relative stamps masked, so a re-render at a new HEAD that
    stamps identical prose with the new commit is not input drift. Without
    git (the unit fixtures) the stored raw digest stands in; fixture text
    carries no hex runs, so raw and masked agree there.

    Only meaningful with non-empty ``stamps`` (the fresh-comparison mode);
    the committed-chain mode short-circuits on the recorded digest before
    this helper is consulted.
    """
    if not stamps:
        return recorded
    committed = _git_show(project_root, rel_posix)
    if committed is not None:
        return _normalized_text_digest_from_bytes(committed, stamps)
    return recorded


def _git_show(project_root: Path, rel_posix: str) -> bytes | None:
    """The file's bytes as committed at HEAD, or ``None`` outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "show", f"HEAD:{rel_posix}"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _normalized_text_digest(path: Path, stamps: tuple[str, ...]) -> str:
    """sha256 of a text file with commit-relative stamps masked out.

    The producer stamps ``GNN_GIT_COMMIT`` (a ``git rev-parse --short``
    abbreviation) into the hydrated sections, and abbreviation length varies
    with git's repo-size heuristic: the same commit abbreviates to 7 chars
    in a shallow CI clone and 9 locally. Those stamps are the ONLY
    commit-relative content in the hydrated inputs, so the fresh-vs-committed
    comparison normalizes them before digesting — otherwise every cron run
    at a HEAD abbreviating differently from the recording checkout is a
    ``[FAIL]``. Any hex run matching one of ``stamps``
    (or a stamp's prefix/suffix — same abbreviation, different length)
    is replaced by a fixed sentinel token; all other content digests
    verbatim.
    """
    return _normalized_text_digest_from_bytes(path.read_bytes(), stamps)


def _normalized_text_digest_from_bytes(data: bytes, stamps: tuple[str, ...]) -> str:
    """:func:`_normalized_text_digest` over in-memory bytes (git content)."""
    if not stamps:
        return hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8", "replace")
    normalized = _COMMIT_HASH_RE.sub(_commit_replacer(stamps), text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _commit_replacer(stamps: tuple[str, ...]) -> Callable[[re.Match[str]], str]:
    """Build the replacer for :data:`_COMMIT_HASH_RE` over ``stamps``."""

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        for stamp in stamps:
            if stamp and (
                token == stamp or token.startswith(stamp) or stamp.startswith(token)
            ):
                return "<commit>"
        return token

    return _repl


_COMMIT_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b")


# The first line of a TeX engine log: ``This is XeTeX, Version ... (preloaded
# format=xelatex 2026.9.18)  28 SEP 2026 21:42``. Only the trailing run
# start time is masked; the engine version and format date stay in the
# digest, because a change there is real toolchain variance.
_LOG_TIMESTAMP_RE = re.compile(
    r"^(This is \S*TeX, .*?)\s+\d{1,2} [A-Z]{3} \d{4} \d{1,2}:\d{2}[ \t]*$",
    re.MULTILINE,
)
_LOG_SUFFIX = ".log"


def mask_log_timestamp(text: str) -> str:
    """``text`` with the TeX log banner's run start timestamp replaced by a sentinel."""
    return _LOG_TIMESTAMP_RE.sub(r"\1  <timestamp>", text, count=1)


def _artifact_digest_from_bytes(
    rel_posix: str, data: bytes, stamps: tuple[str, ...]
) -> str:
    """Fresh-comparison digest of one rendered artifact.

    Commit stamps are masked for every artifact; the ``.log`` additionally
    has its banner timestamp masked, so two renders of the same inputs
    digest the same.
    """
    if not rel_posix.endswith(_LOG_SUFFIX):
        return _normalized_text_digest_from_bytes(data, stamps)
    masked = mask_log_timestamp(data.decode("utf-8", "replace"))
    return _normalized_text_digest_from_bytes(masked.encode("utf-8"), stamps)
