# Contributor guidance

Read [AGENTS.md](AGENTS.md) for repository contracts and module ownership, then
the closest guide beside the files you change. Task-specific rules are indexed
in [.agent_rules/README.md](.agent_rules/README.md).

Run `./run.sh gates` (`just gates` is the same arm) for the canonical gate:
ruff format/check, mypy, the parallel unit suite under the coverage floor, the
documentation, terminology and capability-contract audits, and the v3
orchestration acceptance gate.

- [Setup and runtime profiles](README.md#runtime-defaults-and-required-profiles)
- [Current verification commands](TO-DO.md#verification-and-execution-rules)
- [Workflow environments and selections](.github/workflows/README.md)
- [Contribution guide](CONTRIBUTING.md)
- [Handoff checklist](docs/development/contributor_handoff.md)

Source, tests and current run evidence establish behavior. Published releases
and their receipts retain their original source identities.
