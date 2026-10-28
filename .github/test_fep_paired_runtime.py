"""Real subprocess/file controls for the workflow's runtime custody wrapper.

The tiny fixture CLI exercises custody, not the scientific bridge contract;
the workflow separately executes the actual pinned FEP bridge.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path

HELPER = Path(__file__).with_name("fep_paired_runtime.py")
CLI = """import json, sys
from pathlib import Path
def main(argv):
    root = Path(argv[argv.index("--project-root") + 1])
    if (root / "fail").exists():
        print('{"status": "error"}')
        return 7
    if (root / "mutate").exists():
        (root / "uv.lock").write_text("changed during command")
    if "status" in argv:
        result = {"status": "ok", "native_claim_ready": False}
    else:
        result = {"status": "ok", "read_only": True, "model": argv[argv.index("--model") + 1]}
    print(json.dumps(result))
    return 0
"""


class WorkflowConfigurationTests(unittest.TestCase):
    def test_active_pair_uses_current_source_and_archived_policy_is_immutable(self):
        """Current CI cannot admit the historical release-baseline exception."""
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/fep-lean-paired-revision.yml").read_text()
        self.assertEqual(workflow.count("uses: actions/checkout@"), 2)
        self.assertNotIn("gnn-baseline", workflow)
        self.assertNotIn("--gnn-release-", workflow)
        self.assertNotIn(".github/fep-release-baseline.json", workflow)
        self.assertEqual(workflow.count('--gnn-root "$GITHUB_WORKSPACE/gnn"'), 2)
        self.assertEqual(
            workflow.count(
                '--gnn-revision "${{ github.event.pull_request.head.sha || github.sha }}"'
            ),
            2,
        )
        self.assertIn("for stage in installed status finite continuous final", workflow)
        self.assertIn("fep_paired_runtime.py q7-source", workflow)
        self.assertIn("if: matrix.python == '3.14'", workflow)
        self.assertIn('python: ["3.11", "3.12", "3.14"]', workflow)
        self.assertIn("--frozen --no-sync --python", workflow)
        self.assertIn("if: always()", workflow)
        self.assertFalse((root / ".github/fep-release-baseline.json").exists())
        archive = root / "docs/releases/fep_4_3_0_baseline.json"
        self.assertEqual(
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "b2de77c03736ff8113640c9becc8e389ca06e9e48f65eb582bd1994d0c39ec34",
        )
        historical = json.loads(archive.read_text())
        self.assertEqual(historical["release_version"], "4.3.0")
        self.assertEqual(historical["mode"], "fixed-release-baseline")

        freshness = (root / ".github/workflows/pair-pin-freshness.yml").read_text()
        self.assertIn("repository: ActiveInferenceInstitute/fep_formal", freshness)
        self.assertNotIn("repository: ActiveInferenceInstitute/fep_lean", freshness)


class RuntimeCustodyTests(unittest.TestCase):
    package_name = "fep_lean"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fep-runtime-control-"))
        print(f"retained control: {self.root}", flush=True)
        self.environment = self.root / "environment"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.environment)
        self.python = self.environment / "bin/python"
        self.version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.gnn = self.root / "gnn"
        self.fep = self.root / "fep"
        for root in (self.gnn, self.fep):
            root.mkdir()
            self.git(root, "init", "-q")
            self.git(root, "config", "core.fsmonitor", "false")
            self.git(root, "config", "user.name", "Runtime test")
            self.git(root, "config", "user.email", "runtime@example.invalid")
            (root / "uv.lock").write_text("fixture lock\n")
        package = self.fep / "src" / self.package_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text(CLI)
        self.fep_revision = self.commit(self.fep)
        pin = self.gnn / ".github/fep-lean-pair.json"
        pin.parent.mkdir()
        pin.write_text(
            json.dumps(
                {
                    "repository": "ActiveInferenceInstitute/fep_formal",
                    "revision": self.fep_revision,
                }
            )
        )
        self.gnn_revision = self.commit(self.gnn)
        site = subprocess.check_output(
            [str(self.python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
        Path(site, "fixture.pth").write_text(str(self.fep / "src"))
        self.receipts = self.root / "receipts"
        self.receipts.mkdir()
        self.release_flags = []

    def git(self, root, *args):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args], text=True
        ).strip()

    def commit(self, root):
        self.git(root, "add", ".")
        self.git(root, "commit", "-qm", "fixture")
        return self.git(root, "rev-parse", "HEAD")

    def run_stage(
        self, stage, *, version=None, python=None, environment=None, success=True
    ):
        result = subprocess.run(
            [
                str(python or self.python),
                str(HELPER),
                stage,
                "--expected-python",
                version or self.version,
                "--environment",
                str(environment or self.environment),
                "--gnn-root",
                str(self.gnn),
                "--gnn-revision",
                self.gnn_revision,
                "--fep-root",
                str(self.fep),
                "--fep-revision",
                self.fep_revision,
                "--receipts",
                str(self.receipts),
                *self.release_flags,
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode == 0, success, result.stderr)
        if not success:
            self.assertTrue((self.receipts / f"runtime-{stage}-failed.json").is_file())
        return result

    def fixed_pair(self):
        """Real Git checkouts retain separate baseline and current identities."""
        (self.gnn / "pyproject.toml").write_text('[project]\nversion = "4.2.0"\n')
        package = self.gnn / "src/gnn"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('__version__ = "4.2.0"\n')
        (package / "science.py").write_text("VALUE = 42\n")
        self.gnn_revision = self.commit(self.gnn)
        owners = [
            "pyproject.toml",
            "uv.lock",
            "src/gnn/__init__.py",
            "src/gnn/science.py",
        ]
        pin = self.fep / "specs/gnn-bridge-w2-source-custody/source-pin.json"
        pin.parent.mkdir(parents=True)
        pin.write_text(
            json.dumps(
                {
                    "gnn": {
                        "commit": self.gnn_revision,
                        "owners": {
                            name: hashlib.sha256(
                                (self.gnn / name).read_bytes()
                            ).hexdigest()
                            for name in owners
                        },
                    }
                }
            )
        )
        package = self.fep / "src" / self.package_name / "bridge"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "operations.py").write_text(
            "def owner_roster(root, repository):\n"
            '    return sorted(["pyproject.toml", "uv.lock"] + '
            '[p.relative_to(root).as_posix() for p in root.glob("src/gnn/**/*.py")])\n'
        )
        self.fep_revision = self.commit(self.fep)
        self.release = self.root / "release"
        subprocess.run(
            ["git", "clone", "-q", str(self.gnn), str(self.release)], check=True
        )
        self.git(self.release, "config", "user.name", "Runtime test")
        self.git(self.release, "config", "user.email", "runtime@example.invalid")
        (self.release / "pyproject.toml").write_text('[project]\nversion = "4.2.1"\n')
        (self.release / "src/gnn/__init__.py").write_text('__version__ = "4.2.1"\n')
        (self.release / "uv.lock").write_text("reviewed refreshed lock\n")
        (self.release / ".github/fep-lean-pair.json").write_text(
            json.dumps(
                {
                    "repository": "ActiveInferenceInstitute/fep_formal",
                    "revision": self.fep_revision,
                }
            )
        )
        (self.release / ".github/fep-release-baseline.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": "fixed-release-baseline",
                    "fep_revision": self.fep_revision,
                    "gnn_revision": self.gnn_revision,
                    "release_version": "4.2.1",
                    "approved_lock_sha256": hashlib.sha256(
                        (self.release / "uv.lock").read_bytes()
                    ).hexdigest(),
                    "resynchronize": "next-major-release",
                }
            )
        )
        self.release_flags = [
            "--gnn-release-root",
            str(self.release),
            "--gnn-release-revision",
            self.commit(self.release),
        ]

    def test_fixed_pair_preserves_baseline_and_current_bookends(self):
        self.fixed_pair()
        for stage in ("installed", "status", "finite", "continuous", "final"):
            self.run_stage(stage)
        result = json.loads((self.receipts / "runtime-final.json").read_text())[
            "before"
        ]
        self.assertEqual(result["gnn"]["revision"], self.gnn_revision)
        binding = result["release_binding"]
        self.assertEqual(binding["current_release"]["revision"], self.release_flags[-1])
        self.assertFalse(binding["current_pair_synchronized"])
        self.assertEqual(
            set(binding["changed_owners"]),
            {"pyproject.toml", "src/gnn/__init__.py", "uv.lock"},
        )

    def test_fixed_pair_rejects_scientific_change(self):
        self.fixed_pair()
        (self.release / "src/gnn/science.py").write_text("VALUE = 43\n")
        self.release_flags[-1] = self.commit(self.release)
        self.run_stage("installed", success=False)

    def test_fixed_pair_rejects_owner_addition(self):
        self.fixed_pair()
        (self.release / "src/gnn/new.py").write_text("VALUE = 42\n")
        self.release_flags[-1] = self.commit(self.release)
        self.run_stage("installed", success=False)

    def test_fixed_pair_rejects_metadata_beyond_version(self):
        self.fixed_pair()
        with (self.release / "pyproject.toml").open("a") as stream:
            stream.write('dependencies = ["unreviewed"]\n')
        self.release_flags[-1] = self.commit(self.release)
        self.run_stage("installed", success=False)

    def test_fixed_pair_rejects_unreviewed_lock(self):
        self.fixed_pair()
        (self.release / "uv.lock").write_text("unreviewed lock\n")
        self.release_flags[-1] = self.commit(self.release)
        self.run_stage("installed", success=False)

    def test_fixed_pair_rejects_initializer_code_change(self):
        self.fixed_pair()
        with (self.release / "src/gnn/__init__.py").open("a") as stream:
            stream.write("VALUE = 42\n")
        self.release_flags[-1] = self.commit(self.release)
        self.run_stage("installed", success=False)

    def test_actual_cli_execution_and_unchanged_environment(self):
        for stage in ("installed", "status", "finite", "continuous", "final"):
            self.run_stage(stage)
        receipt = json.loads((self.receipts / "runtime-final.json").read_text())
        self.assertEqual(receipt["before"], receipt["after"])
        self.assertEqual(receipt["before"]["runtime"]["actual_python"], self.version)
        self.assertEqual(receipt["before"]["runtime"]["prefix"], str(self.environment))
        self.assertFalse(receipt["native_claim_ready"])
        # Successful bridge commands alone cannot establish current Q7 freshness.
        self.run_stage("q7-source", success=False)
        rejected = json.loads(
            (self.receipts / "runtime-q7-source-failed.json").read_text()
        )
        self.assertIn(
            "verify_native.py" if self.version == "3.14" else "CPython 3.14",
            rejected["error"],
        )

    def test_wrong_requested_interpreter_rejected(self):
        self.run_stage(
            "installed",
            version="3.12" if self.version == "3.11" else "3.11",
            success=False,
        )

    def test_wrong_environment_rejected(self):
        self.run_stage("installed", environment=self.root / "wrong", success=False)

    def test_changed_environment_configuration_rejected(self):
        self.run_stage("installed")
        cfg = self.environment / "pyvenv.cfg"
        cfg.write_text(cfg.read_text() + "\nfixture = changed\n")
        self.run_stage("status", success=False)

    def test_changed_environment_inode_rejected(self):
        self.run_stage("installed")
        retained = self.root / "retained_environment"
        self.environment.rename(retained)
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.environment)
        self.run_stage("status", success=False)

    def test_source_revision_drift_rejected(self):
        self.run_stage("installed")
        (self.gnn / "new").write_text("different source")
        self.commit(self.gnn)
        self.run_stage("status", success=False)

    def test_source_byte_drift_rejected(self):
        self.run_stage("installed")
        (self.fep / "uv.lock").write_text("uncommitted drift")
        self.run_stage("status", success=False)

    def test_mutation_during_cli_rejected(self):
        self.run_stage("installed")
        (self.fep / "mutate").touch()
        self.run_stage("status", success=False)

    def test_failed_cli_retains_output_and_exit_code(self):
        self.run_stage("installed")
        (self.fep / "fail").touch()
        self.run_stage("status", success=False)
        receipt = json.loads((self.receipts / "runtime-status-failed.json").read_text())
        self.assertEqual(receipt["exit_code"], 7)
        self.assertEqual(
            json.loads((self.receipts / "bridge-status.json").read_text())["status"],
            "error",
        )

    def test_missing_receipt_rejected(self):
        self.run_stage("status", success=False)

    def test_wrong_receipt_stage_rejected(self):
        self.run_stage("installed")
        path = self.receipts / "runtime-installed.json"
        receipt = json.loads(path.read_text())
        receipt["stage"] = "final"
        path.write_text(json.dumps(receipt))
        self.run_stage("status", success=False)

    def test_changed_output_receipt_rejected(self):
        self.run_stage("installed")
        self.run_stage("status")
        (self.receipts / "bridge-status.json").write_text(
            '{"status": "ok", "native_claim_ready": false}'
        )
        self.run_stage("finite", success=False)

    def test_wrong_command_receipt_rejected(self):
        self.run_stage("installed")
        self.run_stage("status")
        path = self.receipts / "runtime-status.json"
        receipt = json.loads(path.read_text())
        receipt["command"] = ["catalogue"]
        path.write_text(json.dumps(receipt))
        self.run_stage("finite", success=False)

    def test_final_requires_every_stage(self):
        self.run_stage("installed")
        self.run_stage("final", success=False)

    def test_existing_receipt_not_overwritten(self):
        self.run_stage("installed")
        path = self.receipts / "runtime-installed.json"
        original = path.read_bytes()
        self.run_stage("installed", success=False)
        self.assertEqual(path.read_bytes(), original)

    def test_wrong_namespace_receipt_origin_rejected(self):
        self.run_stage("installed")
        self.run_stage("status")
        path = self.receipts / "runtime-status.json"
        receipt = json.loads(path.read_text())
        other = "fep_lean" if self.package_name == "fep_formal" else "fep_formal"
        receipt["cli_source"] = str(self.fep / "src" / other / "cli.py")
        path.write_text(json.dumps(receipt))
        self.run_stage("finite", success=False)

    def test_external_import_origin_rejected(self):
        package = self.root / "shadow" / self.package_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text(CLI)
        site = subprocess.check_output(
            [str(self.python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
        Path(site, "shadow.pth").write_text(
            f"import sys; sys.path.insert(0, {str(package.parent)!r})\n"
        )
        self.run_stage("installed")
        result = self.run_stage("status", success=False)
        self.assertIn("module source mismatch", result.stderr)

    def test_symlinked_cli_origin_rejected(self):
        cli = self.fep / "src" / self.package_name / "cli.py"
        retained = self.root / "outside-cli.py"
        cli.rename(retained)
        cli.symlink_to(retained)
        self.fep_revision = self.commit(self.fep)
        pin = self.gnn / ".github/fep-lean-pair.json"
        pin.write_text(
            json.dumps(
                {
                    "repository": "ActiveInferenceInstitute/fep_formal",
                    "revision": self.fep_revision,
                }
            )
        )
        self.gnn_revision = self.commit(self.gnn)
        self.run_stage("installed")
        result = self.run_stage("status", success=False)
        self.assertIn("invalid FEP module source", result.stderr)


class CanonicalRuntimeCustodyTests(RuntimeCustodyTests):
    package_name = "fep_formal"

    def test_mixed_legacy_module_alias_rejected(self):
        legacy = self.fep / "src/fep_lean"
        legacy.mkdir()
        (legacy / "__init__.py").write_text("")
        (legacy / "cli.py").write_text(CLI)
        (self.fep / "src/fep_formal/__init__.py").write_text(
            'import sys\nfrom fep_lean import cli\nsys.modules["fep_formal.cli"] = cli\n'
        )
        self.fep_revision = self.commit(self.fep)
        pin = self.gnn / ".github/fep-lean-pair.json"
        pin.write_text(
            json.dumps(
                {
                    "repository": "ActiveInferenceInstitute/fep_formal",
                    "revision": self.fep_revision,
                }
            )
        )
        self.gnn_revision = self.commit(self.gnn)
        self.run_stage("installed")
        result = self.run_stage("status", success=False)
        self.assertIn("module source mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
