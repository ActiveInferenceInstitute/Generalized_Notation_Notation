"""Accepted FEP bridge source renders through both public structured inputs.

Fixture text copied from fep_lean/specs/gnn-bridge-p1-finite-spike/gnn-input/
FepLeanSymmetricBool.md; provenance is retained in its Signature section.
"""

import ast
from pathlib import Path
from typing import Any

import pytest

from gnn.parsers.common import GNNInternalRepresentation
from gnn.parsers.markdown_parser import MarkdownGNNParser
from gnn.render.processor import render_gnn_spec
from gnn.render.pymdp.pymdp_renderer import parse_gnn_markdown

FEP_SOURCE = """# GNN Bridge: FepLean Symmetric Boolean Generative Model
# GNN Version: 1.0
# Deterministic projection of the compiled Lean instance
#   FEP.ActiveInference.symmetricBoolModel trueBiasedPolicyPrior
#   (fep_lean, lean/FepSketches/active_inference.lean:743-749).
# Every numeric literal below is the exact Lean source value;
# provenance file:line comments name each definition site.

## GNNSection
FepLeanSymmetricBool

## GNNVersionAndFlags
GNN v1

## ModelName
FepLean Symmetric Boolean Generative Model

## ModelAnnotation
Bridge P1 spike: the fep_lean active_inference.lean
GenerativeModel instance `symmetricBoolModel trueBiasedPolicyPrior`
(two policies, two hidden states, two observations, one step)
projected deterministically to GNN v1 syntax.
Extraction record (file:line in the fep_lean checkout at the
commit recorded under Signature):
- D initialState = fairBoolLaw (1/2, 1/2)
  [def active_inference.lean:719-722; use :745]
- B transition = fairBoolKernel, policy-indexed, all entries 1/2
  [def active_inference.lean:725-728; use :746]
- A likelihood = fairBoolKernel, all entries 1/2
  [def active_inference.lean:725-728; use :747]
- C preferences = fairBoolLaw (1/2, 1/2)
  [def active_inference.lean:719-722; use :748]
- E policyPrior = trueBiasedPolicyPrior: E(false)=1/4, E(true)=3/4
  [def active_inference.lean:731-734; parameter :743,:749]
- Timescale: one transition application [active_inference.lean:30-32]
- The Lean GenerativeModel carries no Action type, so no `u`
  variable or action edges are emitted.

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[2,2,type=float]      # likelihood : FiniteKernel State Outcome
# Transition matrix: B[next_state, previous_state, policy]
B[2,2,2,type=float]    # transition : Policy -> FiniteKernel State State
# Preference vector: C[observation_outcomes] (probability law)
C[2,type=float]        # preferences : FiniteLaw Outcome
# Prior vector: D[states]
D[2,type=float]        # initialState : FiniteLaw State
# Policy prior vector: E[policies]
E[2,type=float]        # policyPrior : FiniteLaw Policy
# Hidden state
s[2,1,type=float]      # initialState distribution
s_prime[2,1,type=float]  # predictedState (one-step)
# Observation
o[2,1,type=float]      # predictedOutcome distribution
# Policy
π[2,type=float]        # policy prior / posterior over policies
F[π,type=float]        # variationalFreeEnergy readout
G[π,type=float]        # expectedFreeEnergy readout
# Time
t[1,type=int]          # discrete time step (one-step model)

## Connections
# Plain edges only: the pipeline markdown parser (step 3) does not
# strip v1.1 ':annotation' suffixes and then warns on the targets;
# recorded as a GNN-side parser finding in the slice report.
D>s
s-B
B>s_prime
s_prime-A
A-o
E>π
π-B
C>G
G>π

## InitialParameterization
# Rounding policy: exact Lean rationals -> exact terminating
# decimal strings; no value is rounded.
# A: active_inference.lean:725-728 (mass _ _ := 1 / 2), used :747.
# Rows are observations, columns are hidden states (Bool: false, true).
A={
  (0.5, 0.5),
  (0.5, 0.5)
}

# B: active_inference.lean:725-728, policy-indexed at :746.
# B[next_state, previous_state, policy]; per-policy slices are
# column-stochastic 2x2 matrices (rows next, columns previous).
B={
  ( (0.5, 0.5), (0.5, 0.5) ),
  ( (0.5, 0.5), (0.5, 0.5) )
}

# C: active_inference.lean:719-722, used :748. Probability law
# over outcomes (Bool: false, true); emitted untransformed.
C={(0.5, 0.5)}

# D: active_inference.lean:719-722, used :745.
D={(0.5, 0.5)}

# E: active_inference.lean:731-734 (mass policy := if policy then
# 3 / 4 else 1 / 4), applied at :743,:749.
E={(0.25, 0.75)}

## Equations
# Lean-defined quantities over the projected instance
# (file:line in lean/FepSketches/active_inference.lean):
# predictedState  :30-32   q_π(s') = Σ_s B[s',s,π] D[s]
# predictedOutcome:40-42   p_π(o)  = Σ_s' A[o,s'] q_π(s')
# risk            :128-131 KL(p_π || C)
# ambiguity       :133-135 Σ_s' q_π(s') H(A[·,s'])
# expectedFreeEnergy :148-150 G(π) = risk(π) + ambiguity(π)
# variationalFreeEnergy :209-214 F = KL(q||posterior) + surprisal
# policyPosterior :475-485 Q(π) ∝ E(π) exp(-γ G(π))

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=1

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=Preferences
D=PriorOverHiddenStates
E=Habit
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
t=Time

## ModelParameters
num_hidden_states: 2  # State = Bool
num_obs: 2            # Outcome = Bool
num_actions: 2        # B policy dimension = Policy = Bool
num_timesteps: 1      # one-step model

## Footer
FepLean bridge P1 spike: symmetric Boolean generative model
projected from fep_lean active_inference.lean
symmetricBoolModel trueBiasedPolicyPrior. One-step horizon,
no planning depth, no precision modulation.

## Signature
source_repository: fep_lean
source_commit: b9d315cb448a7e3c8e54a6e4b26c78aef929d73a
pipeline_repository: GeneralizedNotationNotation
pipeline_commit: 64d49355acf197a0570b06ab334d97570774be64
lean_module: lean/FepSketches/active_inference.lean
lean_structure: FEP.ActiveInference.GenerativeModel
lean_instance: FEP.ActiveInference.symmetricBoolModel trueBiasedPolicyPrior
projection_tool: fep_lean specs/gnn-bridge-p1-finite-spike/projection.py (bridge P1/P2 deterministic emitter)
target_syntax: GNN v1 (docs/gnn/gnn_syntax.md v1.1 surface)
rounding_policy: exact Lean rationals emitted as shortest exact terminating decimal strings; non-terminating expansions are a no-go, never rounded
"""


@pytest.mark.parametrize("input_kind", ["mapping", "structured"])
def test_accepted_fep_source_preserves_two_state_policy_dimensions(
    tmp_path: Path, input_kind: str
) -> None:
    source = tmp_path / "FepLeanSymmetricBool.md"
    source.write_text(FEP_SOURCE)
    source_bytes = source.read_bytes()
    spec: GNNInternalRepresentation | dict[str, Any] | None
    if input_kind == "mapping":
        spec = parse_gnn_markdown(FEP_SOURCE, source)
        assert spec is not None
    else:
        result = MarkdownGNNParser().parse_string(FEP_SOURCE)
        assert result.success
        spec = result.model
    assert spec is not None
    success, message, artifacts = render_gnn_spec(spec, "pymdp", tmp_path / "render")
    assert success, message
    assert artifacts
    program = Path(artifacts[0])
    program_bytes = program.read_bytes()
    tree = ast.parse(program_bytes)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    literals = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in main.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"A_data", "B_data", "C_data", "D_data", "E_data", "gnn_spec"}
    }
    embedded = literals["gnn_spec"]
    assert embedded["model_name"] == "FepLean Symmetric Boolean Generative Model"
    parameters = embedded["model_parameters"]
    assert {
        name: parameters[name]
        for name in ("num_hidden_states", "num_obs", "num_actions", "num_timesteps")
    } == {
        "num_hidden_states": 2,
        "num_obs": 2,
        "num_actions": 2,
        "num_timesteps": 1,
    }
    dimensions = {
        variable["name"]: variable["dimensions"] for variable in embedded["variables"]
    }
    assert {name: dimensions[name] for name in ("A", "B", "C", "D", "E")} == {
        "A": [2, 2],
        "B": [2, 2, 2],
        "C": [2],
        "D": [2],
        "E": [2],
    }
    # These terminating decimals are the exact authored Lean rationals.
    expected = {
        "A": [[0.5, 0.5], [0.5, 0.5]],
        "B": [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        "C": [0.5, 0.5],
        "D": [0.5, 0.5],
        "E": [0.25, 0.75],
    }
    assert {name: literals[name + "_data"] for name in expected} == expected
    assert {
        name: embedded["initialparameterization"][name] for name in expected
    } == expected
    if input_kind == "mapping":
        assert (
            embedded["raw_sections"]["Signature"]
            == FEP_SOURCE.split("## Signature\n", 1)[1].strip()
        )
    assert source.read_bytes() == source_bytes
    assert program.read_bytes() == program_bytes
