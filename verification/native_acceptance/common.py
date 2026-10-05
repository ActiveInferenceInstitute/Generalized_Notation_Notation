"""Exact peer-reviewed private admission helpers; no native jobs at import."""
from pathlib import Path
import base64, csv, hashlib, importlib.metadata as md, io, json, os, shutil, sys, time, zipfile
from gnn.execute.subprocess_envelope import CancelToken
CODE = Path(__file__).resolve().parent
BASE = Path(os.environ["GNN_NATIVE_VERIFY_WORKSPACE"]).absolute()
def verify_wheel(base, split, expected_wheel, expected_artifact):
    """Bind the ordinary wheel archive, its RECORD, and the installed distribution."""
    import stat

    def no_symlink_path(path):
        absolute = path.absolute()
        assert all(not part.is_symlink() for part in (absolute, *absolute.parents)), (
            "Symlink in ordinary archive/installed path", str(path))
        return Path(os.path.abspath(absolute))

    def regular_file(path):
        checked = no_symlink_path(path)
        assert stat.S_ISREG(checked.lstat().st_mode), (
            "Ordinary archive/installed member is not a regular file", str(path))
        return checked

    binding = json.loads(regular_file(CODE / "source-binding.json").read_text())
    assert binding["content_commit"] == "88f24cb7cb3026ab6da66eed0d754037f763865e", binding["content_commit"]
    assert binding["artifact_commit"] == expected_artifact, binding["artifact_commit"]
    assert binding["wheel_sha256"] == expected_wheel, binding["wheel_sha256"]
    candidates = [regular_file(path) for path in (base / "wheels").glob("*.whl")
                  if hashlib.sha256(regular_file(path).read_bytes()).hexdigest() == expected_wheel]
    assert len(candidates) == 1, candidates
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        assert len(members) == len({member.filename for member in members}), "Duplicate wheel members"
        for member in members:
            kind = stat.S_IFMT(member.external_attr >> 16)
            permitted = {0, stat.S_IFDIR} if member.is_dir() else {0, stat.S_IFREG}
            assert kind in permitted, ("Non-regular wheel archive member", member.filename)
        names = [member.filename for member in members if not member.is_dir()]
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        blobs = {name: archive.read(name) for name in names}
    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    assert len(record_names) == 1, record_names
    record_name = record_names[0]
    dist_prefix = record_name.rsplit("/", 1)[0]
    wheel_rows = list(csv.reader(io.StringIO(blobs[record_name].decode())))
    assert all(len(row) == 3 for row in wheel_rows), "Malformed wheel RECORD"
    assert len(wheel_rows) == len({row[0] for row in wheel_rows}), "Duplicate wheel RECORD row"
    assert {row[0] for row in wheel_rows} == set(names), "Wheel RECORD/member set mismatch"
    for name, digest, size in wheel_rows:
        if name == record_name:
            assert not digest and not size
            continue
        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(blobs[name]).digest()).decode().rstrip("=")
        assert digest == expected and size == str(len(blobs[name])), name
    archive_python = {name: hashlib.sha256(blob).hexdigest()
                      for name, blob in blobs.items() if name.startswith("gnn/") and name.endswith(".py")}
    assert archive_python == binding["python_inventory"], "Wheel/source Python inventory mismatch"
    assert len(archive_python) == binding["python_source_files_bound"]

    import gnn
    distribution = md.distribution("generalized-notation-notation")
    site = no_symlink_path(Path(distribution.locate_file("")))
    env_root = no_symlink_path(base / ("py" + split))
    assert Path(sys.prefix).resolve() == env_root, (sys.prefix, env_root)
    assert regular_file(Path(gnn.__file__)) == site / "gnn/__init__.py"
    assert site.is_relative_to(env_root), site
    namespace_root = no_symlink_path(site / "gnn")
    installed_namespace = {}
    for path in namespace_root.rglob("*"):
        checked_path = no_symlink_path(path)
        if checked_path.is_dir():
            continue
        regular_file(checked_path)
        if "__pycache__" not in checked_path.parts and checked_path.suffix not in {".pyc", ".pyo"}:
            relative = "gnn/" + checked_path.relative_to(namespace_root).as_posix()
            installed_namespace[relative] = hashlib.sha256(checked_path.read_bytes()).hexdigest()
    archive_namespace = {name: hashlib.sha256(blob).hexdigest()
                         for name, blob in blobs.items() if name.startswith("gnn/")}
    assert installed_namespace == archive_namespace, "Installed namespace/archive byte mismatch"
    for name, blob in blobs.items():
        if name == record_name:
            continue
        installed_file = regular_file(Path(distribution.locate_file(name)))
        assert installed_file.is_relative_to(env_root) and installed_file.read_bytes() == blob, name

    installed_record = regular_file(Path(distribution.locate_file(record_name)))
    installed_rows = list(csv.reader(io.StringIO(installed_record.read_text())))
    assert all(len(row) == 3 for row in installed_rows), "Malformed installed RECORD"
    installed_paths = {row[0] for row in installed_rows}
    assert len(installed_rows) == len(installed_paths), "Duplicate installed RECORD row"
    assert set(blobs).issubset(installed_paths), "Installed RECORD omits archive member entries"
    checked = 0
    for name, digest, size in installed_rows:
        path = regular_file(Path(distribution.locate_file(name)))
        assert path.is_relative_to(env_root), ("Installed RECORD escapes environment", name)
        if name == record_name:
            assert not digest and not size, "Installed RECORD self row must be unhashed and unsized"
            continue
        if not digest:
            assert name.endswith(".pyc") and not size, name
            continue
        blob = path.read_bytes()
        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
        assert digest == expected and size == str(len(blob)), ("Installed RECORD mismatch", name)
        checked += 1
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        assert not json.loads(direct_url).get("dir_info", {}).get("editable"), "Editable install"
    versions = {name: md.version(name) for name in
                ("generalized-notation-notation", "jax", "jaxlib", "numpy", "matplotlib", "thrml", "cpomdp")}
    assert versions["generalized-notation-notation"] == "4.0.0"
    assert versions["thrml"] == "0.1.4" and versions["cpomdp"] == "0.4.4"
    expected_jax = {"311": "0.9.2", "312": "0.11.2"}[split]
    assert versions["jax"] == expected_jax and versions["jaxlib"] == expected_jax, versions
    assert tuple(sys.version_info[:3]) == {"311": (3, 11, 15), "312": (3, 12, 13)}[split]
    return binding, {
        "wheel_archive": str(wheel),
        "wheel_sha256": expected_wheel,
        "wheel_record_sha256": hashlib.sha256(blobs[record_name]).hexdigest(),
        "wheel_record_files_verified": len(wheel_rows),
        "installed_record_sha256": hashlib.sha256(installed_record.read_bytes()).hexdigest(),
        "installed_record_hashes_verified": checked,
        "wheel_metadata_sha256": hashlib.sha256(blobs[dist_prefix + "/WHEEL"]).hexdigest(),
        "installed_namespace_files_verified": len(installed_namespace),
        "python_source_files_bound": len(archive_python),
        "source_bytes_match": True,
        "installed_outside_checkout": True,
        "ordinary_wheel_install": True,
        "regular_archive_and_installed_files": True,
        "complete_unique_wheel_and_installed_records": True,
        "versions": versions,
        "python": sys.version,
    }

class StorageToken(CancelToken):
    """Sample storage through the public envelope's cooperative cancellation polls."""

    def __init__(self):
        super().__init__()
        self.minimum = None
        self.samples = 0
        self.last_sample_monotonic = None
        self.resource_failed = False
        self.probe_errors = []

    @property
    def cancelled(self):
        self.samples += 1
        self.last_sample_monotonic = time.monotonic()
        try:
            free = shutil.disk_usage(BASE).free
            if not isinstance(free, int) or isinstance(free, bool) or free < 0:
                raise ValueError("Invalid free-storage measurement")
            self.minimum = free if self.minimum is None else min(self.minimum, free)
            if self.minimum <= 4 * 1024**3:
                self.resource_failed = True
                self.cancel("private native acceptance storage floor exhausted")
        except (OSError, ValueError) as exc:
            self.resource_failed = True
            self.probe_errors.append(type(exc).__name__ + ": " + str(exc))
            self.cancel("private native acceptance storage measurement failed")
        return super().cancelled

    def snapshot(self):
        return {"storage_floor_bytes": 4 * 1024**3,
                "minimum_free_storage_bytes": self.minimum,
                "storage_samples": self.samples,
                "last_sample_monotonic": self.last_sample_monotonic,
                "resource_failed": self.resource_failed,
                "probe_errors": list(self.probe_errors),
                "sampling_boundary": "Start/end and monotonic public-envelope polls; no guarantee between samples"}

def validate_pipeline_process(envelope, summary, canonical_steps):
    """Admit the public warning exit only alongside complete verified final evidence."""
    code = envelope.get("return_code")
    assert isinstance(code, int) and not isinstance(code, bool) and code in {0, 2}, envelope
    assert envelope.get("cancelled") is False and not envelope.get("error_type"), envelope
    assert envelope.get("cleanup_verified") is True and envelope.get("streams_drained") is True, envelope
    status = summary.get("overall_status") or summary.get("status")
    assert status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}, status
    assert code == {"SUCCESS": 0, "SUCCESS_WITH_WARNINGS": 2}[status], (code, status)
    assert summary.get("run_id") and summary.get("end_time"), "Not a finalized current-run summary"
    assert summary.get("unfinished_steps") == [], "Required work remains unfinished"
    assert summary.get("evidence_integrity", {}).get("status") == "verified", "Final evidence is unverified"
    assert isinstance(summary.get("run_session_sha256"), str) and len(summary["run_session_sha256"]) == 64
    rows = summary.get("steps", [])
    scripts = [row["script_name"] for row in rows]
    assert len(scripts) == len(set(scripts)) == 7, "Missing or duplicate required step"
    assert [canonical_steps[name] for name in scripts] == [3, 7, 8, 11, 12, 16, 20], scripts
    assert summary.get("planned_steps") == scripts, "Final receipts differ from planned work"
    assert all(row.get("status") in {"SUCCESS", "SUCCESS_WITH_WARNINGS"} for row in rows), rows
    assert all(row.get("run_id") == summary["run_id"] for row in rows), "Step receipts belong to another run"
    return status
