# tests/scripts

Contract tests for the repository's own gate scripts, loaded as modules rather
than imported as packages: the audit line-bands ratchet and the stdlib-shadow
guard (COMP-010, `docs/standards/composability.md`).

```bash
uv run --frozen --no-sync python -m pytest tests/scripts/ -q
```
