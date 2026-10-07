# Filesystem ownership and native cleanup boundaries

GNN assumes the operator controls its API workspace, selected source directories,
output root and their parent directory entries throughout a run. An API caller
may provide hostile path text. A concurrent filesystem adversary may rename a
directory, replace an entry with a symbolic link or Windows junction, hardlink a
lock file, or race creation and deletion. The controls below address specific
operations under that adversary model; they do not turn generated code or the
pipeline into an operating-system sandbox.

## Workspace admission and output creation

The API retains repository-local defaults for checkout use. Installed deployments
set the operator environment variable `GNN_API_ROOT` to an **absolute, existing
directory with no symlink or reparse components**, before creating the service.
Requests cannot override that root. The configured root is a data workspace,
not the installed package's code directory. Empty, relative, missing or redirected
configured roots fail with `PathValidationError`; REST surfaces retain their
existing bad-request envelopes. Paths containing `..` are refused even when
normalization would finish inside the workspace.

On POSIX, output creation opens each component with `O_DIRECTORY | O_NOFOLLOW`
and uses `mkdir(..., dir_fd=parent)` for missing directories. The same parent
descriptor owns creation and reopening. Replacing the parent's pathname with a
symlink cannot redirect that operation to the symlink target. A final inode check
detects replacement of the requested destination during creation and refuses
admission. A failed creation can leave a newly created empty directory in the
opened, renamed parent; it does not roll back another writer's entries.

An open descriptor pins an **object**, not its permanent pathname ancestry. A
writer able to rename an opened directory can move that object; GNN cannot make
that rename impossible. Returning a `Path` does not protect subsequent pathname
reads or writes. Source snapshots, digests and run identity detect specified
changes but do not replace the trusted-directory precondition. No confinement
claim applies to every producer write in an attacker-writable output tree.

Windows stdlib directory creation has no equivalent descriptor-relative API.
Admission rejects reparse points, including junctions, but concurrent directory
replacement is outside the supported Windows creation guarantee. Keep Windows
workspace and ancestor directory entries inaccessible to untrusted writers.

## Output leases

`OutputLease` retains its regular, single-link `.gnn_run.lock` file, run identifier
and nonblocking cross-process exclusion. POSIX acquires the lock relative to the
opened output directory and holds its directory descriptors until lease release.
It checks the named lock inode and output directory identity before and after
the preexisting-tree inspection. Symlink, hardlink, special-file and replacement
failures remain `OutputLeaseError`. Rejected acquisitions do not overwrite a
replacement link target. Closing the lease releases native `flock` or Windows
`msvcrt` byte-range locking; the lock file remains for the next invocation.

The lease is advisory: cooperating GNN runs exclude each other. An uncooperative
writer can unlink or rename an active lock and create another inode. The lock
cannot prevent that writer from modifying outputs after acquisition. It is not
a tenant boundary or protection against hostile filesystem permissions.

## Run artifact deletion

Deleting an API run retains the workspace root, its ancestors and the shared
workspace `output` tree. A requested
symlink or junction directory is retained rather than resolved to its target.
On POSIX, the parent is opened without following symlinks and the native fd-based
`shutil.rmtree(..., dir_fd=parent)` handles recursive symlink races. A replaced
ancestor cannot redirect deletion through a newly planted symlink. This is
deletion of the directory entry in the opened parent, not proof that a pathname
still names the original run's directory after arbitrary real-directory swaps.
Failed or unavailable removal is reported in `artifacts_removed` and
`artifacts_note`, preserving the deletion response contract.

Windows rejects existing reparse components and uses native `shutil.rmtree`,
which does not descend directory junction targets. It still requires trusted
directory entries against concurrent mutation. Removing the run record does not
assert that artifacts were removed when the removal result is false.

## Process cleanup and resource observation

| Native environment | Declared boundary | Supported verification | Refused or excluded guarantee |
| --- | --- | --- | --- |
| macOS / Linux with descendant observation | `observed_descendants` | Owned process group plus retained birth-time-bound observed descendants; shared cleanup ceiling and pipe draining | Hostile detachment before observation; exact process-tree peak RSS |
| POSIX without a native child query or observer | `process_group_only` | Ordinary process-group cleanup | Required descendant observation or accounting |
| Windows | `direct_worker_only` | Direct-worker termination and bounded pipe draining | Descendant containment or descendant resource accounting |

Windows `CREATE_NEW_PROCESS_GROUP` is not a descendant tree-kill primitive. The
cleanup receipt now names its actual direct-worker scope. `cleanup_verified`
applies only to the receipt's declared `containment` boundary; it never certifies
unobserved children. A detached child retaining inherited output handles causes
bounded stream-drain failure rather than a successful cleanup receipt.

`run_subprocess_envelope` accepts `require_descendant_containment=True` and
`require_descendant_resource_accounting=True`. Unsupported native platforms
return `error_type="UnsupportedContainment"`, `containment="not_started"`
before launching. If a supported native observer is unavailable at runtime,
cleanup is attempted and the requested guarantee fails explicitly. Even native
descendant RSS remains sampled, includes shared-page double counting and can miss
short-lived or unobserved workers; it is advisory rather than a hard memory limit.
No implementation enumerates unrelated host processes to approximate ownership.

The API job and run executors share `api/process_supervision.py`: they observe
while the worker is alive, request native stop for cancellation, and
verify worker exit, observed descendants and pipe draining within a shared
one-second cleanup allowance. POSIX cancellation preserves a short graceful
SIGTERM drain (at most 150 ms) before escalating to SIGKILL inside that ceiling.
Normal leader exit also triggers cleanup without waiting indefinitely for
inherited pipes. Their `process_cleanup` receipt declares the same
native scope. A cancellation request with unverified cleanup now ends `failed`
with an explicit cleanup error instead of asserting successful cancellation.
This corrects the prior API behavior that could preserve `cancelled` after an
unverified communication error or wait indefinitely for a SIGTERM-ignoring worker.
Deleting a run whose cleanup is unverified returns the existing runtime-conflict
error and retains its record and artifacts; a failed status alone is not proof
that its workers have stopped.

## Acceptance

The maintained [filesystem tests](../../tests/security/test_filesystem_boundaries.py)
exercise real rename/symlink races with coordinated threads, outside-target
sentinels, independent-process lease exclusion, configured scratch workspaces,
native link or junction refusal, bounded cancellation and platform-specific
required-containment outcomes. POSIX tests use real descriptor operations;
Windows assertions must run on a native Windows runner. Passing macOS tests or
changing `sys.platform` in a unit test is not Windows acceptance evidence.

```bash
uv sync --frozen --extra dev --extra api
uv run --frozen --no-sync python -m pytest tests/security/test_filesystem_boundaries.py tests/api/test_path_utils_symlink.py tests/api/test_runs_delete_cancel.py tests/utils/test_process_tree.py tests/pipeline/test_current_run_contract.py tests/execute/test_subprocess_envelope.py -q
```

Installed-platform acceptance selects the portable tests from this module without
the `needs_posix` marker. A native Windows run is required to accept the junction,
lease and direct-worker guarantees. Native descendant execution is deliberately
unsupported on Windows; installation and direct-worker acceptance do not imply
full Windows backend execution support.

References: [Python `rmtree` semantics](https://docs.python.org/3/library/shutil.html#shutil.rmtree),
[Windows byte-range locking](https://docs.python.org/3/library/msvcrt.html#msvcrt.locking),
[Windows process-group semantics](https://docs.python.org/3/library/subprocess.html#subprocess.CREATE_NEW_PROCESS_GROUP).
