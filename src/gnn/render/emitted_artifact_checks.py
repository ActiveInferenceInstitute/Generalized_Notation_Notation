"""Static hygiene checks for emitted render artifacts.

Two conservative, dependency-free checks:

- ``undefined_names`` reports Load-context names that are never bound anywhere
  in the module (imports, defs, classes, assignments, args, except handlers,
  global/nonlocal) and are not builtins or module-context names. Binding
  anywhere counts, so the check is imprecise about scoping but has near-zero
  false positives - the right trade for a deterministic benchmark gate.
- ``first_party_unresolvable_imports`` reports ``gnn.*`` imports whose full
  dotted path does not resolve from the repository. Third-party import
  availability is an environment concern (optional extras are documented);
  first-party imports, however, must always resolve - a stale template
  referencing a renamed module compiles clean and passes every name-based
  check, yet ImportErrors the moment the emitted script runs.

- ``julia_syntax_findings`` and ``stan_syntax_findings`` accept an emitted
  ``.jl`` / ``.stan`` artifact only when the real target-language front-end
  accepts it. A regex or AST check on the artifact text cannot establish that;
  a renderer that returns success has only established that it returned.
  When no front-end is installed the checks report ``available=False`` instead
  of guessing, so a caller can never mistake "not checked" for "valid".

Consumers: ``scripts/bench_render_backends.py`` (corpus x framework
conformance benchmark) and ``tests/render/test_render_contracts.py``.
"""

from __future__ import annotations

# Module-context names that are never bound by statements but always exist.
MODULE_CONTEXT_NAMES = frozenset(
    {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__"}
)


def undefined_names(code: str) -> tuple[list[tuple[str, int]], bool]:
    """Scan one Python artifact for statically-undefined names.

    Returns ``((name, line) findings, star_import_present)``. When the module
    uses a star import the scan is skipped entirely (names cannot be resolved
    without executing the source module), reported via the second element.
    """
    import ast
    import builtins

    tree = ast.parse(code)
    bound: set[str] = set(MODULE_CONTEXT_NAMES)
    star_import = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    star_import = True
                    continue
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    if star_import:
        return [], True
    findings = sorted(
        {
            (node.id, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in bound
            and not hasattr(builtins, node.id)
        },
        key=lambda item: (item[1], item[0]),
    )
    return findings, False


def first_party_unresolvable_imports(code: str) -> list[tuple[str, int]]:
    """Scan one Python artifact for unresolvable ``gnn.*`` imports.

    Returns ``(module, line)`` pairs for every first-party ``gnn.*`` import
    (``import`` or ``from ... import``) whose full dotted path does not
    resolve via ``importlib.util.find_spec``.
    """
    import ast
    import importlib.util

    tree = ast.parse(code)
    findings: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        modules: list[tuple[str, int]] = []
        if isinstance(node, ast.Import):
            modules = [
                (alias.name, node.lineno)
                for alias in node.names
                if alias.name.split(".")[0] == "gnn"
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "gnn"
        ):
            modules = [(node.module, node.lineno)]
        else:
            continue
        for module, lineno in modules:
            try:
                if importlib.util.find_spec(module) is None:
                    findings.append((module, lineno))
            except (ImportError, AttributeError, ValueError):
                findings.append((module, lineno))
    return sorted(findings, key=lambda item: (item[1], item[0]))


def _literal_shape(node: "object") -> "tuple[int, ...] | None":
    """Return the rectangular shape of an AST list literal, or None.

    Ragged literals, non-numeric leaves, and non-list nodes yield None -
    shapes are never guessed.
    """
    import ast

    if not isinstance(node, ast.List):
        return None
    if not node.elts:
        return (0,)
    first = True
    shape: list[int] = []
    for element in node.elts:
        if isinstance(element, ast.List):
            sub = _literal_shape(element)
            if sub is None:
                return None
            if first:
                shape = [len(node.elts), *sub]
                first = False
            else:
                expected = tuple(shape[1:])
                if sub != expected:
                    return None
        elif (
            isinstance(element, (ast.Constant,))
            and isinstance(element.value, (int, float))
            and not isinstance(element.value, bool)
        ):
            if first:
                shape = [len(node.elts)]
                first = False
            elif len(shape) != 1:
                return None
        elif (
            isinstance(element, ast.UnaryOp)
            and isinstance(element.op, ast.USub)
            and isinstance(element.operand, ast.Constant)
            and isinstance(element.operand.value, (int, float))
        ):
            if first:
                shape = [len(node.elts)]
                first = False
            elif len(shape) != 1:
                return None
        else:
            return None
    return tuple(shape)


def _shape_from_array_call(call: "object") -> "tuple[int, ...] | None":
    """Extract the literal shape from ``jnp.array([...])`` / ``torch.tensor([...])``."""
    import ast

    if not isinstance(call, ast.Call) or not call.args:
        return None
    return _literal_shape(call.args[0])


def matrix_shapes(code: str) -> "dict[str, tuple[int, ...]] | None":
    """Extract canonical A/B/C/D matrix shapes from one emitted Python artifact.

    Understands the four maintained emission conventions:

    - pymdp delegated runners: ``A_data = [...]`` plain list literals;
    - jax: ``'A_matrix': jnp.array([...])`` dict-payload calls;
    - pytorch: ``A = torch.tensor([...])`` plus ``B_slices.append(...)``;
    - numpyro: ``A = jnp.array([...])`` plus ``B_slices.append(...)``.

    Returns None when the module uses a star import (conservative skip - the
    same rule as :func:`undefined_names`). Only letters with a clean
    rectangular literal shape are returned; ragged or call-embedded matrices
    are skipped, never guessed.
    """
    import ast

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None

    names = {
        "A_data": "A",
        "B_data": "B",
        "C_data": "C",
        "D_data": "D",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
    }
    dict_keys = {
        "A_matrix": "A",
        "B_matrix": "B",
        "C_vector": "C",
        "D_vector": "D",
    }
    shapes: dict[str, tuple[int, ...]] = {}
    b_slices: list[tuple[int, ...]] = []

    def _record(letter: str, shape: tuple[int, ...] | None) -> None:
        if shape is not None:
            shapes.setdefault(letter, shape)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                letter = names.get(target.id)
                if letter is None:
                    continue
                if isinstance(value, ast.List):
                    _record(letter, _literal_shape(value))
                elif isinstance(value, ast.Call):
                    _record(letter, _shape_from_array_call(value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in dict_keys
                    and isinstance(value, ast.Call)
                ):
                    _record(dict_keys[key.value], _shape_from_array_call(value))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "B_slices"
            and node.args
        ):
            shape = _shape_from_array_call(node.args[0])
            if shape is not None:
                b_slices.append(shape)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "B" for target in node.targets
        ):
            # ``B = torch.stack(B_slices, dim=2)`` / ``jnp.stack(..., axis=2)``:
            # the final B is each equal slice shape plus the slice-count axis.
            if (
                isinstance(node.value, ast.Call)
                and b_slices
                and all(s == b_slices[0] for s in b_slices)
                and any(
                    isinstance(arg, ast.Name) and arg.id == "B_slices"
                    for arg in node.value.args
                )
            ):
                _record("B", (*b_slices[0], len(b_slices)))
    if b_slices and all(s == b_slices[0] for s in b_slices):
        _record("B", (*b_slices[0], len(b_slices)))
    return shapes


def julia_dimension_constants(code: str) -> dict[str, int]:
    """Extract ``const NUM_STATES/OBSERVATIONS/ACTIONS = <int>`` from a .jl artifact.

    The Julia backends allocate their matrices from these constants and fill
    values at runtime, so dimension parity against the Python backends' literal
    shapes is checked through them. Returns only the constants present.
    """
    import re

    constants: dict[str, int] = {}
    for match in re.finditer(
        r"const\s+NUM_(STATES|OBSERVATIONS|ACTIONS)\s*=\s*(\d+)", code
    ):
        constants[f"NUM_{match.group(1)}"] = int(match.group(2))
    return constants


def julia_render_factorization(code: str) -> "str | None":
    """Extract ``const MODEL_KIND = "..."`` from a .jl artifact, if declared.

    ``"hierarchical"`` marks a native per-level factorized render whose
    dimension constants describe the FAST level only - joint-vs-per-level
    comparisons with the Python backends' expanded joint matrices do not
    apply, and the parity gate must skip it.
    """
    import re

    match = re.search(r'const\s+MODEL_KIND\s*=\s*"([^"]+)"', code)
    return match.group(1) if match else None


# --- Target-language syntax acceptance ---------------------------------------
#
# ``Meta.parseall(source)`` alone is NOT a syntax check: it does not raise and
# returns an expression tree that may contain ``Expr(:error, ...)`` and
# ``Expr(:incomplete, ...)`` nodes. A probe that only calls it - as
# ``Meta.parseall(read(f, String)); println("parsed")`` does - exits 0 for a
# file with unbalanced delimiters, a stray ``end``, or a truncated binary
# expression. The probe below parses incrementally and walks every returned
# expression, which catches those classes (verified against those fixtures).

JULIA_SYNTAX_PROBE = r"""
function gnn_walk_errors!(expr, path, findings)
    if expr isa Expr
        if expr.head === :error || expr.head === :incomplete
            detail = replace(sprint(showerror, expr.args[1]), "\n" => " ")
            push!(findings, string(expr.head, " @ ", path, ": ", detail))
        end
        for (index, arg) in enumerate(expr.args)
            gnn_walk_errors!(arg, string(path, "/", index), findings)
        end
    end
    return findings
end

function gnn_main()
    source = read(stdin, String)
    findings = String[]
    position = firstindex(source)
    count = 0
    while position <= lastindex(source)
        parsed = try
            Meta.parse(source, position)
        catch err
            println("PARSE_FAIL raised: ",
                    replace(sprint(showerror, err), "\n" => " "))
            return 1
        end
        expression, nextpos = parsed
        expression === nothing && break
        count += 1
        gnn_walk_errors!(expression, string("expr", count), findings)
        nextpos > position || break
        position = nextpos
    end
    for finding in findings
        println("PARSE_FAIL ", finding)
    end
    return isempty(findings) ? 0 : 1
end

exit(gnn_main())
"""

#: Wall-clock bound for one front-end invocation.
TOOLCHAIN_TIMEOUT_SECONDS = 300.0


def _stdin_findings(
    command: list[str], code: str, timeout: float
) -> tuple[list[str], bool]:
    """Run *command* with *code* on stdin; return ``(findings, available)``.

    ``available`` is ``False`` only when the executable is missing. A timeout
    or a crash is reported as a finding: an unchecked artifact must not read
    as a valid one.
    """
    import subprocess

    try:
        # Fixed argv from shutil.which(), never a shell; the source travels on
        # stdin so no path or quoting is interpolated into a command line.
        result = subprocess.run(  # nosec B603
            command,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return [], False
    except subprocess.TimeoutExpired:
        return [f"{command[0]} timed out after {timeout:g}s"], True
    if result.returncode == 0:
        return [], True
    findings = [
        line[len("PARSE_FAIL ") :]
        for line in result.stdout.splitlines()
        if line.startswith("PARSE_FAIL ")
    ]
    if not findings:
        detail = (result.stderr or result.stdout).strip().splitlines()
        findings = (
            [detail[-1]] if detail else [f"{command[0]} exited {result.returncode}"]
        )
    return findings, True


def julia_syntax_findings(
    code: str, *, timeout: float = TOOLCHAIN_TIMEOUT_SECONDS
) -> tuple[list[str], bool]:
    """Check Julia source with the real parser; return ``(findings, available)``.

    Findings are empty exactly when every top-level expression parses into a
    tree with no ``:error`` / ``:incomplete`` node. ``available`` is ``False``
    when no ``julia`` executable is on ``PATH`` - callers must treat that as
    "not checked", never as "valid".
    """
    import shutil

    executable = shutil.which("julia")
    if executable is None:
        return [], False
    return _stdin_findings(
        [executable, "--startup-file=no", "--color=no", "-e", JULIA_SYNTAX_PROBE],
        code,
        timeout,
    )


def stan_syntax_findings(
    code: str, *, timeout: float = TOOLCHAIN_TIMEOUT_SECONDS
) -> tuple[list[str], bool]:
    """Check a Stan program with ``stanc``; return ``(findings, available)``.

    ``stanc`` is the compiler front-end CmdStan itself installs, so a clean
    result is the language-level acceptance Step 12's CmdStan compile would
    also demand. ``available`` is ``False`` when ``stanc`` is not on ``PATH``.
    """
    import os
    import shutil

    executable = shutil.which("stanc")
    if executable is None:
        return [], False
    return _stdin_findings([executable, f"--o={os.devnull}", "-"], code, timeout)
