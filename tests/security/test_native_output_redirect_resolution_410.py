"""Public output ownership admission on native redirect/error semantics."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from gnn.pipeline.output_lease import OutputLeaseError, validate_output_tree

pytestmark = pytest.mark.needs_posix


def _tree_bytes(root: Path) -> dict:
    """Retain authored bytes/link texts without following redirect targets."""
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                result[relative] = ("symlink", os.readlink(path))
            elif path.is_file():
                result[relative] = ("file", path.read_bytes())
            elif path.is_dir():
                result[relative] = ("directory",)
            else:
                result[relative] = ("nonregular", path.lstat().st_mode)
    return result


@pytest.mark.parametrize("self_cycle", [False, True])
def test_native_output_link_cycle_is_refused_without_tree_changes(
    tmp_path: Path, self_cycle: bool
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "retained.txt").write_bytes(b"retain prior output bytes")
    (output / "first").symlink_to("first" if self_cycle else "second")
    if not self_cycle:
        (output / "second").symlink_to("first")
    before = _tree_bytes(tmp_path)
    with pytest.raises(OutputLeaseError, match="cyclic symlink") as caught:
        validate_output_tree(output)
    cause = caught.value.__cause__
    assert isinstance(cause, RuntimeError) or (
        isinstance(cause, OSError) and cause.errno == errno.ELOOP
    )
    assert _tree_bytes(tmp_path) == before


def test_contained_ordinary_file_and_directory_links_remain_admitted(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    data = output / "data"
    data.mkdir(parents=True)
    (data / "model.json").write_bytes(b'{"retained": true}\n')
    (output / "file-alias").symlink_to("data/model.json")
    (output / "directory-alias").symlink_to("data", target_is_directory=True)
    before = _tree_bytes(tmp_path)
    assert validate_output_tree(output) is None
    assert (output / "file-alias").read_bytes() == b'{"retained": true}\n'
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize("target", ["missing/item", "missing/../data/model.json"])
def test_contained_dangling_and_partial_missing_links_keep_prior_admission(
    tmp_path: Path, target: str
) -> None:
    output = tmp_path / "output"
    data = output / "data"
    data.mkdir(parents=True)
    (data / "model.json").write_bytes(b'{"retained": true}\n')
    link = output / "alias"
    link.symlink_to(target)
    assert not link.exists()
    with pytest.raises(FileNotFoundError):
        link.resolve(strict=True)
    assert link.resolve().is_relative_to(output)
    before = _tree_bytes(tmp_path)
    assert validate_output_tree(output) is None
    assert _tree_bytes(tmp_path) == before
    assert not (output / "missing").exists()


def test_contained_dangling_link_chain_keeps_prior_admission(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "first").symlink_to("second")
    (output / "second").symlink_to("missing/item")
    before = _tree_bytes(tmp_path)
    assert validate_output_tree(output) is None
    assert _tree_bytes(tmp_path) == before
    assert not (output / "missing").exists()


@pytest.mark.parametrize("external_cycle", [False, True])
def test_partial_missing_normalization_cannot_hide_a_native_cycle(
    tmp_path: Path, external_cycle: bool
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    cycle_root = tmp_path / "external" if external_cycle else output
    cycle_root.mkdir(exist_ok=True)
    alias = output / "000-alias"
    alias.symlink_to(
        "missing/../../external/first" if external_cycle else "missing/../first"
    )
    (cycle_root / "first").symlink_to("second")
    (cycle_root / "second").symlink_to("first")
    # The first strict lookup stops at the actual missing prefix. Native
    # non-strict normalization can expose a remaining real loop instead.
    with pytest.raises(FileNotFoundError):
        alias.resolve(strict=True)
    with pytest.raises((RuntimeError, OSError)) as native:
        alias.resolve().resolve(strict=True)
    assert isinstance(native.value, RuntimeError) or native.value.errno == errno.ELOOP
    before = _tree_bytes(tmp_path)
    with pytest.raises(OutputLeaseError, match="cyclic symlink"):
        validate_output_tree(output)
    assert _tree_bytes(tmp_path) == before
    assert not (output / "missing").exists()


@pytest.mark.parametrize(
    "target", ["../external/missing", "missing/../../external/retained.txt"]
)
def test_dangling_and_partial_missing_escape_are_refused(
    tmp_path: Path, target: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "retained.txt").write_bytes(b"retain external bytes")
    (output / "alias").symlink_to(target)
    before = _tree_bytes(tmp_path)
    with pytest.raises(OutputLeaseError, match="escapes its output root"):
        validate_output_tree(output)
    assert _tree_bytes(tmp_path) == before
    assert not (external / "missing").exists()


def test_native_not_a_directory_error_is_not_treated_as_dangling(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "file").write_bytes(b"regular file cannot contain a child")
    link = output / "alias"
    link.symlink_to("file/child")
    with pytest.raises(NotADirectoryError):
        link.resolve(strict=True)
    before = _tree_bytes(tmp_path)
    with pytest.raises(OutputLeaseError, match="Cannot resolve") as caught:
        validate_output_tree(output)
    assert isinstance(caught.value.__cause__, NotADirectoryError)
    assert caught.value.__cause__.errno == errno.ENOTDIR
    assert _tree_bytes(tmp_path) == before


def test_external_hardlink_and_nonregular_entry_still_refuse_admission(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.txt"
    external.write_bytes(b"retain hardlinked external bytes")
    hardlinked = tmp_path / "hardlinked"
    hardlinked.mkdir()
    os.link(external, hardlinked / "alias.txt")
    nonregular = tmp_path / "nonregular"
    nonregular.mkdir()
    os.mkfifo(nonregular / "fifo")
    before = _tree_bytes(tmp_path)
    with pytest.raises(OutputLeaseError, match="single link"):
        validate_output_tree(hardlinked)
    with pytest.raises(OutputLeaseError, match="regular file or directory"):
        validate_output_tree(nonregular)
    assert _tree_bytes(tmp_path) == before
