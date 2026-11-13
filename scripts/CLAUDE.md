# scripts/ contributor guidance

Read [AGENTS.md](AGENTS.md) for the script inventory — audit and compliance
gates, pipeline orchestration, manuscript figure generation, performance work
and maintenance — and the repository [AGENTS.md](../AGENTS.md) for module
contracts and the architecture rules those scripts serve. Task-specific rules
are indexed in [.agent_rules/README.md](../.agent_rules/README.md).

- Scripts are thin orchestrators: parse arguments, delegate to a `gnn.*`
  implementation, return an exit code. No domain logic lives here.
- Gate scripts run through the `justfile` recipes and
  [.github/workflows/local-gates.yml](../.github/workflows/local-gates.yml);
  `just quality` chains the local gates.
- Verification and execution obligations live in [TO-DO.md](../TO-DO.md).
