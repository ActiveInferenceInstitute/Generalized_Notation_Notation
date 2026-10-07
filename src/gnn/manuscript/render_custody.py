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
* ``render_inputs_sha256`` — sha256 of every committed hydrated file under
  ``output/manuscript/``. A prose edit that never reached a render shows up
  here as a disagreement with the tree on disk.
* ``counts_describe_commit`` / ``receipt_counts_describe_commit`` — the
  commit short-hash the token map describes and the one the producer's
  receipt (written by the pre-render producer run) describes.

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
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

if str(Path(__file__).resolve().parents[3]) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gnn.manuscript.substitution import hydrate_text  # noqa: E402
from gnn.manuscript.variables import token_checksum  # noqa: E402

__all__ = [
    "HYDRATION_FIX",
    "MANIFEST_VERSION",
    "RECORD_COMMAND",
    "RENDER_INPUT_ROOT",
    "RENDERED_ARTIFACTS",
    "custody_issues",
    "hydrate_text",
    "hydration_issues",
    "load_render_manifest",
    "manifest_path",
    "mask_log_timestamp",
    "record_render_manifest",
    "strip_volatile_tokens",
    "verify_fresh_render",
]

MANIFEST_VERSION = "gnn_render_manifest_v1"
RECORD_COMMAND = "uv run python scripts/z_record_manuscript_render_manifest.py"

_MANIFEST_REL = Path("output") / "data" / "manuscript_render_manifest.json"
_VARIABLES_REL = Path("output") / "data" / "manuscript_variables.json"
RENDER_INPUT_ROOT = Path("output") / "manuscript"
_RECEIPT_REL = Path("output") / "data" / "manuscript_variables_receipt.json"
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
        for rel in (*RENDERED_ARTIFACTS, RENDER_INPUT_ROOT)
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

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "recorded_by": RECORD_COMMAND,
        "counts_describe_commit": counts_describe_commit,
        "receipt_counts_describe_commit": receipt_commit,
        "variables_sha256": token_checksum(strip_volatile_tokens(variables)),
        "render_artifacts_sha256": {
            rel.as_posix(): _sha256(project_root / rel) for rel in RENDERED_ARTIFACTS
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
    """``{relpath: bytes}`` the injector would write from this token map."""
    source = project_root / _SOURCE_MANUSCRIPT_REL
    out: dict[str, bytes] = {}
    for md in sorted(source.glob("*.md")):
        if md.name in excluded or md.name in _VERBATIM_AUX:
            continue
        text = hydrate_text(md.read_text(encoding="utf-8"), variables)
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

    * ``[FAIL]`` — the artifact is missing, or it drifted from the manifest
      AND the recorded render inputs drifted too: the fresh render
      disagrees with the committed chain in prose as well as bytes, so the
      committed chain is stale for HEAD.
    * ``[WARN]`` — the artifact drifted but the committed render inputs
      still describe the tree: same inputs, different bytes, most likely
      toolchain variance. The v1 manifest records no tool versions, so
      input-match is the only v1-compatible discriminator; the committed
      chain still describes this prose.

    Commit-relative stamps are normalized on both sides before digesting:
    the producer abbreviates ``GNN_GIT_COMMIT`` with git's repo-size
    heuristic, so a fresh CI clone stamps 7 chars where the recording
    checkout stamped 9 — those stamps are content-equal, and only their
    length differs. The committed side is re-derived from ``git show`` at
    HEAD with the manifest's stamp masked; outside a git repo (the unit
    fixtures) the stored raw digest is used when the text carries no stamp.
    The ``.log`` also has its TeX engine banner's start timestamp masked
    (:func:`mask_log_timestamp`): it is the one line that differs between
    two renders of identical inputs, and left raw it turned every clean
    re-render into a ``[WARN]``.

    Read-only. Returns (does not raise) so the caller can print every
    finding at once; a missing manifest is the first ``[FAIL]``.
    """
    try:
        manifest = load_render_manifest(project_root)
    except RuntimeError as exc:
        return [f"[FAIL] {exc}"]

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

    inputs_match = not _input_tree_issues(project_root, manifest, stamps)
    recorded_artifacts: dict[str, str] = manifest.get("render_artifacts_sha256", {})
    issues: list[str] = []
    for rel in RENDERED_ARTIFACTS:
        rel_posix = rel.as_posix()
        path = project_root / rel
        recorded = recorded_artifacts.get(rel_posix)
        if not path.is_file():
            issues.append(
                f"[FAIL] {rel_posix} is missing — the custody manifest "
                "records it but the fresh render did not produce it"
            )
        elif recorded is None:
            issues.append(
                f"[FAIL] {rel_posix} is not recorded in the custody "
                f"manifest — re-record with: {RECORD_COMMAND}"
            )
        elif _committed_normalized(rel_posix, recorded) == _normalized(rel):
            continue
        elif inputs_match:
            issues.append(
                f"[WARN] {rel_posix} differs from the committed custody "
                "manifest but the render inputs match it exactly — same "
                "inputs, different bytes, most likely toolchain variance "
                "(the v1 manifest records no tool versions); the committed "
                "chain still describes this prose"
            )
        else:
            issues.append(
                f"[FAIL] {rel_posix} drifted from the committed custody "
                "manifest and the render inputs drifted too — the "
                "committed chain is stale for HEAD; rerun the SC-22 ritual "
                f"(re-record alone with {RECORD_COMMAND} cannot fix this)"
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
