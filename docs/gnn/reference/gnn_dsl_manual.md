# GNN DSL Manual

**Version**: v2.0.0 documentation bundle\
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: GNN DSL authoring and tool integration. See [framework implementations](../implementations/README.md) for current backend coverage.

This document provides a comprehensive guide to the GNN Domain Specific Language, which is a Markdown-based format for representing computational models, particularly those used in fields like Active Inference.

## Pipeline Processing

GNN DSL files are processed and validated through the pipeline:

- **`src/gnn/3_gnn.py`** → GNN file parsing and multi-format processing
  - See: **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)** for parser implementation details
- **`src/gnn/5_type_checker.py`** → DSL syntax and type validation
  - See: **[src/gnn/type_checker/AGENTS.md](../../../src/gnn/type_checker/AGENTS.md)** for validation details

For complete pipeline documentation, see **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**.

**Quick Start:**

```bash
# Parse and validate GNN DSL files
python src/gnn/3_gnn.py --target-dir input/gnn_files --verbose
python src/gnn/5_type_checker.py --target-dir input/gnn_files --strict
```

| Symbol | Meaning                                                                          | ExampleUse    | MeaningOfExample                                   |
|--------|----------------------------------------------------------------------------------|---------------|----------------------------------------------------|
| ^      | A caret means superscript.                                                       | X^Y           | X with a superscript Y                              |
| ,      | A comma is used to separate items in a list.                                     | X,Y           | List with X and Y as elements                      |
| ##     | A double hashtag signals a new section in the Markdown file.                     | ## Section123 | Has "Section123" as a section name                  |
| #      | A hashtag signals the title header in the Markdown file.                         | # Title123    | Has "Title123" as model title                       |
| #      | Inside variable/connection blocks, lines starting with `#` are comments.         | # Comment123  | Inline comment (skipped by parser)                 |
| {}     | Curly brackets are specification of exact values for a variable.                 | X{1}          | X equals 1 exactly                                 |
| -      | Hyphen is an undirected causal edge between two variables.                       | X-Y           | Undirected relation between X and Y                |
| ()     | Parentheses are used to group expressions.                                       | X^(Y_2)       | X with a superscript that is Y with a subscript 2  |
| []     | Rectangular brackets define the dimensionality, or state space, of a variable.   | X[2,3]        | X is a matrix with dimensions (2,3)                |
| =      | The equals sign declares equality or assignment.   | X=5          | Sets the variable X to value of 5     |
| >      | The greater than symbol represents a directed causal edge between two variables. | X>Y          | Causal influence from X to Y          |
| _      | Underscore means subscript.                        | X_2          | X with a subscript 2                  |
| +      | Plus sign for addition or other operations.        | X+Y          | Sum of X and Y                        |
| *      | Asterisk for multiplication or other operations.   | X*Y          | Product of X and Y                      |
| /      | Forward slash for division or other operations.    | X/Y          | X divided by Y                        |
| \\|     | Vertical bar for conditional probability or alternatives. | P(X\\|Y)     | Probability of X given Y              |

## 3. Detailed Section Descriptions

This section provides detailed syntax, rules, and examples for each GNN section, based on analysis of the parser and type checker code.

### 3.1. `StateSpaceBlock`

The `StateSpaceBlock` section is crucial for defining all variables (also referred to as factors or nodes) within the model and their respective characteristics, such as dimensionality and data type.

**Syntax:**

Each line within this section typically defines one variable. The common format is:

`VariableName[Dimension1,Dimension2,...,DimensionN,type=DataType] # Optional comment`

- `VariableName`: The name of the variable. Can include underscores (`_`) and carets (`^`) for subscripts and superscripts (e.g., `s_t`, `X^observed`).
- `[Dimensions,type=DataType]`: Enclosed in square brackets.
  - `DimensionX`: Numerical values representing the size of each dimension of the variable (e.g., `2`, `3`). For scalars or vectors where one dimension might be 1, it's often still included (e.g., `s_f0[2,1]`).
  - `type=DataType`: An optional parameter specifying the data type of the variable (e.g., `type=float`, `type=int`, `type=string`). If omitted, the type might be inferred or defaulted by consuming applications. The parser specifically looks for `type=` followed by the type string.
- `# Optional comment`: Any text following a `#` symbol on the line is treated as a comment and can be used for annotation.

**Parser Behavior:**

- The parser extracts the `VariableName`, each `Dimension`, the `DataType`, and the `comment`.
- It handles variable names with `^` and `_`.
- It can parse dimensions like `len(π)` or `[2]` as string representations if they don't fit the simple integer model.
- The extracted information is typically stored in a structured format, often a dictionary per variable, under a top-level key like `'Variables'` in the parsed output.

**Example:**

```markdown
## StateSpaceBlock
# A_matrices are defined per modality: A_m[observation_outcomes, state_factor0_states, state_factor1_states]
A_m0[3,2,3,type=float]   # Likelihood for modality 0 ("state_observation")
B_f1[3,3,3,type=float]   # Transitions for factor 1 ("decision_state"), 3 actions
D_f0[2,type=float]       # Prior for factor 0
s_f0[2,1,type=float]     # Hidden state for factor 0 ("reward_level")
π_f1[3,type=float]       # Policy (distribution over actions) for controllable factor 1
t[1,type=int]            # Time step
```

### 3.2. `Connections`

The `Connections` section defines the relationships or edges between the variables declared in the `StateSpaceBlock`. These connections form the graphical structure of the model.

**Syntax:**

Each line typically defines one connection (edge). The format is:

`SourceVariable [->] TargetVariable [=Constraint] [#Comment]`

- `SourceVariable`: The name of the variable where the edge originates.
- `[->]`: The type of edge:
  - `>`: Represents a directed edge (causal influence from `SourceVariable` to `TargetVariable`).
  - `-`: Represents an undirected edge (a mutual relationship or correlation).
- `TargetVariable`: The name of the variable where the edge terminates.
- `=Constraint` (Optional): An optional constraint or label associated with the edge. The parser captures the text following the `=` sign.
- `#Comment` (Optional): Any text following a `#` symbol is treated as a comment for that connection.

**Parser Behavior:**

- The parser extracts the `SourceVariable`, `TargetVariable`, the edge type (directed or undirected), any `Constraint`, and the `Comment`.
- Variable names can include `^`, `_`, and `+` followed by digits (e.g., `X_t+1`).
- The parsed edges are usually stored as a list of dictionaries under a top-level key like `'Edges'`.

**Example:**

```markdown
## Connections
(D_f0,D_f1)-(s_f0,s_f1)
(s_f0,s_f1)-(A_m0,A_m1,A_m2)
(A_m0,A_m1,A_m2)-(o_m0,o_m1,o_m2)
(s_f0,s_f1,u_f1)-(B_f0,B_f1) # u_f1 primarily affects B_f1; B_f0 is uncontrolled
G>π_f1
π_f1-u_f1
G=ExpectedFreeEnergy
t=Time
```

*(Note: The example shows grouped variables like `(D_f0,D_f1)`. The current parser logic described focuses on single source/target per line. Grouped variables might be handled by pre-processing or a more complex parsing step not detailed in the `_process_connections` regex (`src/gnn/visualization/parse/gnn_file_parser.py:241`). The documentation here reflects the identified regex capability.)*

### 3.3. `InitialParameterization`

This section provides the initial numerical or symbolic values for the variables and parameters defined in the `StateSpaceBlock`.

**Syntax:**

The content of this section is generally free-form text that describes the parameter settings. While the main parser (`src/gnn/parsers/system.py`) captures this section's content as a single block of text, specific conventions are often followed in practice, as seen in example GNN files. These conventions might involve:

- Comments (`#`) explaining the parameterization logic.
- Assignments using `=` to set variable values.
- Multi-dimensional arrays or matrices represented using nested curly braces `{}` or parentheses `()`, with comma-separated values.

**Parser Behavior:**

- The primary GNN parser (`src/gnn/parsers/system.py`) treats the entire content under `## InitialParameterization` as a text block.
- Downstream tools or specialized parsers might further process this text block to extract specific values based on conventions used within the GNN file.

**Example:**

```markdown
## InitialParameterization
# A_m0: num_obs[0]=3, num_states[0]=2, num_states[1]=3. Format: A[obs_idx][state_f0_idx][state_f1_idx]
A_m0={
  ( (0.33333,0.33333,0.8), (0.33333,0.33333,0.2) ),  # obs=0; (vals for s_f1 over s_f0=0), (vals for s_f1 over s_f0=1)
  ( (0.33333,0.33333,0.0), (0.33333,0.33333,0.0) ),  # obs=1
  ( (0.33333,0.33333,0.2), (0.33333,0.33333,0.8) )   # obs=2
}

# D_f0: factor 0 (2 states). Uniform prior.
D_f0={(0.5,0.5)}
```

### 3.4. `Equations`

This section is used to describe any mathematical equations associated with the model. These equations can define relationships between variables, update rules, or other computational aspects of the model.

**Syntax:**

The content is typically written in LaTeX format for clear mathematical representation.

**Parser Behavior:**

- Similar to `InitialParameterization`, the primary GNN parser (`src/gnn/parsers/system.py`) captures the content of the `## Equations` section as a single block of text.
- Rendering or interpretation of these LaTeX equations is handled by other tools (e.g., for display in documentation or potentially for symbolic processing).

**Example:**

```markdown
## Equations
# Standard PyMDP agent equations for state inference (infer_states),
# policy inference (infer_policies), and action sampling (sample_action).
qs = \text{infer_states}(o)
q_\pi, \text{efe} = \text{infer_policies}()
\text{action} = \text{sample_action}()
```

### 3.5. `Time`

The `Time` section describes how the model handles the concept of time, which is crucial for dynamic systems.

**Syntax:**

This section uses controlled terms to specify the model's temporal nature. The content is a free-form text block, but specific keywords are interpreted by downstream tools or by convention.

**Controlled Terms (from `src/gnn/documentation/file_structure.md`):**

- `Static`: Indicates the model is static and does not evolve over time.
- `Dynamic`: Indicates the model is dynamic.
- `DiscreteTime=X_t`: Specifies `X_t` as the temporal variable in a discrete-time model.
- `ContinuousTime=X_t`: Specifies `X_t` as the temporal variable in a continuous-time model.
- `ModelTimeHorizon=X`: Specifies `X` as the time horizon for finite-time modeling (e.g., `ModelTimeHorizon=100`, `ModelTimeHorizon=Unbounded`).

**Parser Behavior:**

- The primary GNN parser captures the content of the `## Time` section as a text block.
- Interpretation of these terms relies on conventions and downstream processing logic.

**Example:**

```markdown
## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded # Agent definition is generally unbounded, specific simulation runs have a horizon.
```

### 3.6. `ActInfOntologyAnnotation`

This section links the variables defined in the GNN model to terms from a relevant ontology, such as the Active Inference Ontology. This aids in standardization, comparison, and understanding of models.

**Syntax:**

The section begins with `## ActInfOntologyAnnotation` on its own line. Each subsequent line within this section defines a mapping:

`ModelVariableName = OntologyTerm [# Optional comment]`

- `ModelVariableName`: The name of a variable defined in the `StateSpaceBlock`.
- `=` : An equals sign separates the model variable from the ontology term.
- `OntologyTerm`: The corresponding term from the ontology (e.g., `HiddenState`, `LikelihoodMatrix`).
- `# Optional comment`: Text after a `#` on the line is an ignored comment.

**Parser Behavior (`src/gnn/ontology/processor.py:79` - `parse_gnn_ontology_section`):**

- The parser specifically looks for the `## ActInfOntologyAnnotation` header.
- It reads each line, splitting it at the `=` to get the model variable and the ontology term.
- Empty lines or lines starting with `#` (comments) within the section are ignored.
- Malformed lines (e.g., missing `=`) are skipped.
- Comments at the end of a mapping line are stripped from the ontology term value.
- The result is typically a dictionary mapping model variable names to their ontology terms.

**Example:**

```markdown
## ActInfOntologyAnnotation
A_m0=LikelihoodMatrixModality0
B_f0=TransitionMatrixFactor0
C_m0=LogPreferenceVectorModality0
D_f0=PriorOverHiddenStatesFactor0
s_f0=HiddenStateFactor0         # Hidden state for factor 0
o_m0=ObservationModality0
G=ExpectedFreeEnergy
```

### 3.7 Other Standard Sections

The GNN specification includes several other standard sections. The main parser (`src/gnn/parsers/system.py`) generally treats these sections by capturing their content as a block of text under the section name. Their specific meaning and usage are defined by convention and the `src/gnn/documentation/file_structure.md`.

- **`ImageFromPaper`**: Contains a link or embedded image of the model's graphical representation from a publication. Content is free-form.
- **`GNNVersionAndFlags`**: Specifies the GNN version (e.g., `GNN v1`) and any flags affecting interpretation. Content is a text string.
- **`ModelName`**: A descriptive name for the model. Often the first line of the file if starting with `# ModelName: ...` or `## ModelName`. Content is a text string.
- **`ModelAnnotation`**: A free-text description, caption, or abstract for the model. Content is a multi-line text block.
- **`Footer`**: A concluding marker for the file, often mirroring the model name or version. Content is a text string.
- **`Signature`**: For cryptographic signatures or provenance information. Content is a text block, potentially structured according to a signature scheme.

**Parser Behavior (General):**
For these sections, the parser:

1. Identifies the section by its `## SectionName` heading.
2. Captures all subsequent lines as the content of that section until another `## SectionName` heading or the end of the file is encountered.

The interpretation and use of the content within these sections are typically handled by tools that consume the parsed GNN structure or by users referring to the GNN documentation.
