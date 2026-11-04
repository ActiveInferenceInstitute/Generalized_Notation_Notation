# GNN Quick Start Tutorial

**Software version**: [package metadata](../../../pyproject.toml)
**Last reviewed**: 2026-10-21
**Status**: Maintained

**Goal**: Author, validate and run a small categorical GNN model. Run all commands
from the repository root; setup and native compilation times depend on the host.

## 🎯 What You'll Build

A navigation model with fixed likelihood/transition matrices and preferences
over positions in a 2×2 grid. Belief updates and action selection do not train
those matrices or guarantee an optimal trajectory.

```
[Start] [    ]
[    ] [Goal]
```

## ⚡ Prerequisites

- Basic programming knowledge (any language)
- A supported Python/uv environment from the [setup guide](../../SETUP.md)
- The PyMDP runtime and a writable output directory

> 💡 **No Active Inference background needed!** This tutorial explains concepts as we go.

## 📥 Step 1: Setup

### Install the GNN toolkit

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation

# Provision the uv version specified in the setup guide first.
uv sync --frozen --extra dev --python 3.14

# Test the installation
uv run --frozen --no-sync python src/gnn/main.py --help
```

### Create your workspace

```bash
# Create a folder for your first model
mkdir -p input/gnn_files/my_first_gnn_model
# Stay in the repository root for the commands below.
```

## 🧱 Step 2: Understanding the Basics

### What is Active Inference?

**Simple version**: A mathematical framework where agents:

1. Have **beliefs** about the world (hidden states)
2. Make **observations** about what they can see
3. Take **actions** to achieve their **preferences**

### What does a GNN model specify?

1. **States**: What the agent needs to track (e.g., position)
2. **Observations**: What the agent can see (e.g., visual input)
3. **Actions**: What the agent can do (e.g., move)
4. **Preferences**: What the agent wants (e.g., reach goal)

### Our Grid World Model

- **States**: 4 positions (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
- **Observations**: Current position (can see where it is)
- **Actions**: 4 movements (Up, Down, Left, Right)
- **Preference**: Be at the goal (Bottom-Right)

## 📝 Step 3: Write Your First GNN Model

Create a file called `grid_agent.md`:

```gnn
## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Simple Grid Navigation Agent v1.0

## ModelAnnotation
A 2x2 grid navigation model with fixed parameters and goal preferences.
The agent can observe its current position and can move in 4 directions.
Goal is to reach the bottom-right corner.

## StateSpaceBlock
# Hidden State Factor: Agent's position
s[4,1,type=int]        # Position (0:TopLeft, 1:TopRight, 2:BottomLeft, 3:BottomRight/Goal)
s_prime[4,1,type=int]   # Next position — every variable used in Connections must be declared here

# Observation: What the agent sees (its current position)
o[4,1,type=int]   # Observed position (same as true position)

# Control: Agent's actions
π[4,type=float]  # Policy over movement actions
u[1,type=int]     # Chosen action (0:Up, 1:Down, 2:Left, 3:Right)

# Model matrices
A[4,4,type=float] # Likelihood: P(observation | position)
B[4,4,4,type=float] # Transition: P(next_position | current_position, action)
C[4,type=float]   # Preferences over observations
D[4,type=float]   # Prior beliefs about starting position

# Expected Free Energy and time
G[1,type=float]      # Expected Free Energy for action selection
t[1,type=int]        # Time step

## Connections
# Each connection is a single "source>target" (directed) or "source-target"
# (undirected) pair with no spaces — see the Connections grammar in
# docs/gnn/reference/gnn_syntax.md for the full rule.

# Prior influences initial state
D>s

# Position determines what agent observes
s>A
A>o

# Position and action determine next position
s>B
u>B
B>s_prime

# Preferences and expected outcomes influence action selection
C>G
A>G
B>G
s>G
G>π
π>u

## InitialParameterization

# Keep matrix comments on separate lines and validate dimensions before execution.

# A: the agent observes its position perfectly (identity matrix)

A={
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 1.0)
}

# B: movement transitions, indexed [next_position, current_position, action].
# Actions are 0:Up, 1:Down, 2:Left, 3:Right. Each block below fixes
# next_position; within a block, rows are current_position and columns are
# action. Moving into a wall leaves the agent where it is.

B={
  ( (1.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0) ),

  ( (0.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0) ),

  ( (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0, 0.0),
    (0.0, 0.0, 1.0, 0.0) ),

  ( (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0) )
}

# C: preferences over observations — strongly prefer the goal (BottomRight)

C={(-1.0, -1.0, -1.0, 2.0)}

# D: start at TopLeft with certainty

D={(1.0, 0.0, 0.0, 0.0)}

## ModelParameters

b_tensor_order: next_state_previous_state_action
num_hidden_states: 4
num_obs: 4
num_actions: 4
num_timesteps: 5

## Equations
# Standard Active Inference equations for policy selection:
# G(π) = E_q[ln q(o,s|π) - ln P(o,s|π) - ln C(o)]
# P(π) = softmax(-G(π))

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=5

## ActInfOntologyAnnotation
s=HiddenStatePosition
o=ObservationPosition
π=PolicyMovement
u=ActionMovement
A=LikelihoodMatrixPosition
B=TransitionMatrixMovement
C=PreferenceVector
D=PriorBelief
G=ExpectedFreeEnergy
t=TimeStep

## Footer
Simple Grid Navigation Agent v1.0

## Signature
Creator: GNN Tutorial
Date: 2024
Status: Tutorial Example
```

Save this as `input/gnn_files/my_first_gnn_model/grid_agent.md`.

> Any registered GNN spec extension (`.md`, `.gnn`, and the other non-binary
> extensions in the parser registry) works: Step 5's discovery
> (`src/gnn/type_checker/checking/core.py`) walks every registered non-binary
> extension, not just `*.md`.

## ✅ Step 4: Validate Your Model

Check if your model is correct:

```bash
# Validate the authored file, then run the directory type checker (Step 5).
uv run --frozen --no-sync gnn validate input/gnn_files/my_first_gnn_model/grid_agent.md --strict
uv run --frozen --no-sync python src/gnn/5_type_checker.py --target-dir input/gnn_files/my_first_gnn_model/ --verbose

# Inspect the actual type-checker report; diagnostics depend on the input.
```

**If you see errors**: Check the [Common Errors Guide](../../troubleshooting/error_taxonomy.md) or compare with the template above.

For more information on the type checker, see **[src/gnn/type_checker/AGENTS.md](../../../src/gnn/type_checker/AGENTS.md)**.

## 🚀 Step 5: Generate Runnable Code

Convert your GNN model to executable Python code:

```bash
# Generate PyMDP code (Steps 3, 11, 12)
uv run --frozen --no-sync python src/gnn/main.py --only-steps "3,5,11,12" --target-dir input/gnn_files/my_first_gnn_model/ --output-dir output/my_first_model/ --frameworks pymdp --verbose

# Every step writes under the --output-dir you passed, so with
# --output-dir output/my_first_model/ this run creates:
# - output/my_first_model/03_gnn_output/       (parsed model)
# - output/my_first_model/11_render_output/   (executable code per framework)
# - output/my_first_model/12_execute_output/  (simulation results)
#
# Visualization and export are steps 8 and 7 — add them to --only-steps
# to also get output/my_first_model/08_visualization_output/ and
# output/my_first_model/07_export_output/.
```

Inspect the run summary and per-model execution results. Exit 0 is success, exit
2 is completed work with warnings, and exit 1 is failure. A platform without an
active execution sandbox records that boundary as a warning; successful inference
does not establish process containment or optimal behavior.

For more details on code generation, see:

- **[src/gnn/render/AGENTS.md](../../../src/gnn/render/AGENTS.md)**: Code rendering module documentation  
- **[src/gnn/execute/AGENTS.md](../../../src/gnn/execute/AGENTS.md)**: Execution module documentation

### Test the generated code

Rendered scripts are not written flat into `11_render_output/`. Each model gets its own
directory, and each framework a subdirectory beneath it, with the script named after the
GNN `ModelName`:

```text
output/my_first_model/11_render_output/
└── grid_agent/
    ├── processing_summary.json
    └── pymdp/
        ├── README.md
        └── Simple_Grid_Navigation_Agent_v1.0_pymdp.py
```

So locate the script rather than assuming a flat filename:

```bash
# Find the rendered PyMDP script
find output/my_first_model/11_render_output -name "*_pymdp.py"

# Run it
uv run --frozen --no-sync python output/my_first_model/11_render_output/grid_agent/pymdp/*_pymdp.py
```

The script prints per-timestep observations, beliefs, and selected actions, then writes
its results as JSON next to itself. Step 12 does the same thing for every rendered
backend at once, which is usually what you want:

```bash
uv run --frozen --no-sync python src/gnn/12_execute.py --target-dir input/gnn_files/my_first_gnn_model/ \
  --output-dir output/my_first_model/ --frameworks pymdp --verbose
```

## 🎉 Congratulations

You've just:

1. ✅ Written your first GNN model
2. ✅ Validated it with the type checker
3. ✅ Generated executable code
4. ✅ Run a working Active Inference agent

## 🔄 What Just Happened?

Your agent:

1. **Started** with belief it's at TopLeft
2. **Observed** its true position  
3. **Planned** actions to reach the goal (BottomRight)
4. **Selected** actions based on expected free energy minimization
5. **Updated beliefs and selected actions** under fixed authored model parameters

## 🎯 Next Steps

### Immediate Experiments (5 minutes each)

1. **Change the goal**: Modify `C` to prefer TopRight instead
2. **Add uncertainty**: Make observations noisy by modifying `A`
3. **Bigger world**: Extend to a 3x3 grid (requires updating all matrices)

### Deeper Learning

1. **Understand the math**: Read [Active Inference basics](../about_gnn.md)
2. **Try examples**: Explore [more complex models](gnn_examples_doc.md)
3. **Different domains**: Navigation → Perception → Decision making
4. **Research extensions**: Check [admission limits](../../execution/FRAMEWORK_AVAILABILITY.md) before trying multi-agent or hierarchical sketches

5. **Pipeline architecture**: See **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)** for complete module documentation
6. **Pipeline safety**: Read **[src/gnn/README.md](../../../src/gnn/README.md)** for architecture patterns

### Beyond the discrete kind

Everything in this tutorial is a **discrete categorical** model (`A`/`B`/`C`/`D`
keys, column-stochastic matrices, softmax perception). GNN also treats
**continuous linear-Gaussian** models (`F`/`H`/`Q`/`R` + `prior_mean`/`prior_cov`,
optionally `goal_mean`/`control_gain` for closed-loop control) as first-class.
Runnable exemplars live in [`input/gnn_files/continuous/`](../../../input/gnn_files/continuous/) —
start with `continuous/stochastic_dynamics.md` (passive filtering) and
`continuous/continuous_navigation.md` (belief-driven goal seeking). They render
and execute on JAX, NumPyro, PyTorch, Stan, RxInfer.jl and ngc-learn;
categorical backends (PyMDP, ActiveInference.jl, DisCoPy, bnlearn) report them
`unsupported`, so pick `--frameworks` accordingly.

### Build Your Own Model

1. **Start with the template**: Use [`templates/basic_gnn_template.md`](../../templates/basic_gnn_template.md)
2. **Model your domain**: What states, observations, actions make sense?
3. **Get help**: Check [FAQ](../README.md) and [community discussions](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions)
4. **Process with pipeline**: Use `src/gnn/main.py` to run complete workflow

## 🧠 Key Concepts You've Learned

| Concept | What It Does | In Our Example |
|---------|--------------|----------------|
| **Hidden States** (`s`) | What the agent tracks internally | 4 grid positions |
| **Observations** (`o`) | What the agent can perceive | Current position |
| **Actions** (`u`) | What the agent can do | 4 movement directions |
| **Likelihood** (`A`) | How states relate to observations | Perfect position sensing |
| **Transitions** (`B`) | How actions change states | Movement rules |
| **Preferences** (`C`) | What the agent wants | Reach bottom-right |
| **Expected Free Energy** (`G`) | How the agent chooses actions | Score outcomes using the backend's declared EFE contract |

## 🛠️ Common Issues and Quick Fixes

### "Parser Error: Invalid syntax"

- Check section headers: Use `## StateSpaceBlock` not `## StateSpace`
- Check variable names: Use `s` not `s f0` (underscores, not spaces)

### "Dimension mismatch"

- Ensure matrix sizes match variable definitions
- `A[4,4]` means 4 observations × 4 states

### "Probabilities don't sum to 1"

- Each column in B must sum to 1.0
- Each column in A must sum to 1.0

### "My agent doesn't reach the goal"

- Check `C`: Higher values should be at preferred states
- Check `B`: Ensure movement logic is correct
- Check the backend's admitted planning horizon; the model time horizon is not a universal policy-search setting

### "unsupported framework" for a continuous model

If your spec declares `F`/`H`/`Q`/`R` instead of `A`/`B`, it is a
linear-Gaussian (continuous) model. PyMDP, ActiveInference.jl, DisCoPy, and
bnlearn do not execute that kind — rerun with `--frameworks jax`, `numpyro`,
`pytorch`, `stan`, or `rxinfer`. See [input/gnn_files/INDEX.md](../../../input/gnn_files/INDEX.md).

## 📚 Resources

- **Pipeline Documentation**:
  - [src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md): Complete module registry
  - [src/gnn/README.md](../../../src/gnn/README.md): Pipeline architecture and safety
- **Documentation**: [Full GNN guide](../about_gnn.md) and [GNN Overview](../gnn_overview.md)
- **Examples**: [Model gallery](gnn_examples_doc.md)
- **Help**:
  - [FAQ](../README.md)
  - [Error guide](../../troubleshooting/error_taxonomy.md)
  - [Support](../../SUPPORT.md)
- **Community**: [GitHub Discussions](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions)


Return to the [documentation hub](../../README.md) or [next release scopes](../../../TO-DO.md).
