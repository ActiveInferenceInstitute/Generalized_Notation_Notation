# InstituteOS integration — Generalized_Notation_Notation

## Identity

| Field | Value |
| --- | --- |
| Repository | `ActiveInferenceInstitute/Generalized_Notation_Notation` |
| Affiliation | `institute` |
| Category | `research-software` |
| Status | `active` |
| Maturity | `released` |
| Steward | Active Inference Institute |
| Default branch | `main` |
| Current release | — |
| License | `CC-BY-NC-SA-4.0` |
| Citation | Active Inference Institute — Generalized_Notation_Notation. |

## Federation

| Setting | Value |
| --- | --- |
| Dashboard mode | `repos` |
| Analytics | `true` |
| Dashboard | `true` |
| Sync policy | `pull` |
| Registries | — |
| Tags | gnn, active-inference, modeling, simulation, python |

## Ecosystem relations

| Repository | Relation | Note |
| --- | --- | --- |
| `ActiveInferenceInstitute/InstituteOS` | `referenced-by` | Federated via the InstituteOS .aii manifest. |
| `ActiveInferenceInstitute/Active_Inference_Ontology` | `references` | Uses Active Inference Ontology terms. |
| `ActiveInferenceInstitute/COGANT` | `feeds` | Downstream GNN consumer for codebase-to-model translation. |
| `ActiveInferenceInstitute/active_inference_battleship` | `feeds` | Downstream GNN consumer for the browser model contract. |
| `ActiveInferenceInstitute/active_inference_language_model` | `feeds` | Downstream GNN consumer for the discrete language stack. |
| `ActiveInferenceInstitute/Active_Cardference` | `feeds` | Downstream GNN consumer for card-game agents. |
| `ActiveInferenceInstitute/ActiveInferenceCategoryTheory` | `feeds` | GNN materials for the category-theory curriculum. |
| `ActiveInferenceInstitute/pymdp` | `depends-on` | Consumes the pymdp active-inference library. |

## Key artifacts

| Path | Kind | Description |
| --- | --- | --- |
| `README.md` | `doc` | Repository overview. |
| `src` | `code` | Source tree. |
| `pyproject.toml` | `code` | Python package config. |
| `AGENTS.md` | `doc` | Agent guidance. |
| `CITATION.cff` | `citation` | Machine-readable citation metadata. |
| `tests` | `test` | Test suite. |
| `docs` | `doc` | Documentation tree. |
| `manuscript` | `publication` | Manuscript sources and artifacts. |
| `CHANGELOG.md` | `doc` | Changelog. |

## Portable tasks

| Task | Command | Working directory | Description |
| --- | --- | --- | --- |
| `test` | `./run.sh test` | `.` | Run the test suite. |

## Validation

```bash
python -m aii_platform.aii_sidecar.validate <repo> --completeness
python -m aii_platform.aii_sidecar docs <repo> --overwrite
python -m aii_platform.aii_sidecar.run <repo> --list
```

The sidecar is valid when `validate` prints `OK` and reports no missing artifact,
manifest, link, or declared-doc path. Completeness is a curation target, not a
runtime requirement.

## Data boundary

This sidecar carries metadata only. It does not grant execution permission, does not
replace the repository's own documentation, and must not be used to move private
content into a public repository. Task execution is limited to trusted checkouts.
