# Security, issues and alerts

Use this guide for input/auth/execution boundaries and alert disposition.
[SECURITY.md](../SECURITY.md) owns reporting/support; [security docs](../docs/security/README.md)
owns detailed controls.

## Boundaries

Validate model text, serialized data, scripts, paths, protocol parameters and
provider responses at their documented boundaries. Preserve bounded parsing,
safe XML/literal handling, code admission and explicit sandbox policy.
Schema/AST checks are not OS sandboxes.

Path checks assume trusted filesystem writers and do not prove atomic confinement
against hostile concurrent mutation. Output leases are advisory. Auth, bind,
rate limiting and path admission remain separate. API/MCP have distinct token/
header/allowlist contracts. Do not broaden safe surfaces or enable unsafe/insecure
execution to clear a failure.

## Secrets and dependencies

Use authorized credentials without printing/committing them. Redact sensitive
values, credential-bearing URLs and client-visible paths. Keep credentials out of
worker payloads/receipts and preserve bounded protocol/logging behavior.

Verify advisory identity, dependency path, released fix and locked Python/extra
splits. Update metadata/constraints/lock coherently. Keep dependency-review,
frozen-export audits, Bandit and CodeQL. Success does not prove absence of every
vulnerability. Actions use verified full SHAs; never guess pins or change permissions
silently.

## Disposition

Read the complete issue/alert and reproduce its trigger on the relevant revision.
Fix applicable defects through the public path with meaningful verification.
Close irrelevant/resolved items with specific supporting source/evidence.

False-positive dispositions retain the exact narrow data/control-flow reason;
do not expand them into general security claims. Review issues and available
alert categories separately. Unreadable categories/access are reported, not counted
as zero. Retain failures and bounded limitations.
