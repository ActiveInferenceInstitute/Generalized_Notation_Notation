# Cross-repository workflow contracts

One page per workflow group: the repositories it binds, the concern each group
shares, the canonical implementation of that concern, and the step every member
takes to use it instead of a private copy.

| Page | Group |
| --- | --- |
| [gnn-fep.md](gnn-fep.md) | GNN–FEP: the twenty-two repositories that apply Generalized Notation Notation, bound to one pin reader (`src/gnn/importers.py`) |

Each contract has a machine-readable twin in the `workflow_groups` block of
[../downstream/importers.json](../downstream/importers.json), and
`scripts/check_importer_registry.py` refuses a contract whose member table and
JSON block disagree. That check runs in the `docs` arm of `./run.sh gates`.

## Adding a group

Record the members, their gate commands, documentation hubs, manuscript
producers and pins in the contract page between the `gnn-fep-members` markers,
mirror the same rows in the JSON block, and name one canonical implementation
per shared concern. A group whose members still carry copies records the copies
and the migration; it does not claim they are already gone.

## Related

- [GNN documentation hub](../README.md)
- [Downstream registry](../downstream/README.md)
- [Development guide](../development/README.md)
