# GNN Frequently Asked Questions (FAQ)

Common questions and answers about Generalized Notation Notation (GNN).

## 🤔 General Questions

### What is GNN?

**GNN (Generalized Notation Notation)** is a standardized text-based language for expressing Active Inference generative models. It provides a human-readable, machine-parsable format that can be converted to executable code for various simulation frameworks.

### Why use GNN instead of directly coding models?

GNN offers several advantages:

- **Standardization**: Consistent representation across different frameworks
- **Portability**: Same model can be rendered to PyMDP, RxInfer, etc.
- **Documentation**: Self-documenting with clear structure and annotations
- **Validation**: Built-in type checking and constraint validation
- **Collaboration**: Easy to share and understand models between researchers

### What's the "Triple Play" approach?

GNN supports three complementary modalities:

1. **Text-based**: Human-readable specifications and documentation
2. **Graphical**: Visual factor graphs and dependency diagrams
3. **Executable**: Runnable code in target simulation frameworks

### Do I need to know Active Inference to use GNN?

Basic understanding helps, but GNN can be learned incrementally:

- **Beginners**: Start with [simple examples](../gnn/tutorials/gnn_examples_doc.md) and [basic concepts](../gnn/about_gnn.md)
- **Intermediate**: Learn [Active Inference fundamentals](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)
- **Advanced**: Explore [research applications](../gnn/gnn_paper.md) and [complex models](../../input/gnn_files/)

## 🛠️ Getting Started

### How do I install GNN tools?

The GNN toolkit is included in this repository:

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation

# Install dependencies using UV (recommended)
uv sync

# Or install dev extras
uv sync --extra dev

# Run the main pipeline on examples
uv run python src/gnn/main.py --target-dir input/gnn_files/ --output-dir output/
```

### What's the easiest way to create my first model?

1. **Use the template**: Copy [`templates/basic_gnn_template.md`](../templates/basic_gnn_template.md)
2. **Start simple**: Begin with a 2-state, 2-observation static model
3. **Follow examples**: Reference [`gnn_examples_doc.md`](../gnn/tutorials/gnn_examples_doc.md)
4. **Validate early**: Run the type checker frequently

### Which simulation framework should I choose?

| Framework | Best For | Language | Difficulty |
|-----------|----------|----------|------------|
| **PyMDP** | Learning Active Inference, POMDP research | Python | Beginner-friendly |
| **RxInfer** | Bayesian inference, performance | Julia | Intermediate |
| **DisCoPy** | Category theory, quantum computing | Python | Advanced |

## 📝 Modeling Questions

### How do I decide what states and observations to include?

**Start with the minimum viable model:**

1. **Identify the key decision/inference problem**
2. **What can the agent observe?** → Observation modalities
3. **What does the agent need to track?** → Hidden state factors
4. **What can the agent control?** → Action/control factors

**Example**: Navigation agent

- **States**: `position[4]` (North, South, East, West)
- **Observations**: `sensor_reading[3]` (Clear, Obstacle, Goal)
- **Actions**: `movement[4]` (Forward, Back, Left, Right)

### When should I use static vs. dynamic models?

- **Static models**: Single-shot inference, classification, simple perception

  ```gnn
  ## Time
  Static
  ```

- **Dynamic models**: Sequential decision making, temporal reasoning, learning

  ```gnn
  ## Time
  Dynamic
  DiscreteTime=t
  ModelTimeHorizon=10
  ```

### How do I handle multiple agents?

See [Multiagent Systems Guide](../gnn/advanced/gnn_multiagent.md). Key approaches:

1. **Separate models**: Individual GNN files per agent
2. **Shared states**: Common environmental factors
3. **Communication**: Message passing between agents
4. **Hierarchical**: Multi-level agent architectures

### What if my matrices are very large?

For large state spaces:

1. **Use sparse representations**: Only specify non-zero probabilities
2. **Leverage structure**: Exploit symmetries and patterns
3. **Consider approximations**: Use lower-dimensional embeddings or neural networks
4. **See performance guide**: [`troubleshooting/performance.md`](performance.md)

## 🔧 Technical Questions

### Why is my GNN file not parsing?

**Most common issues:**

1. **Section headers**: Use exact names like `## StateSpaceBlock`
2. **Variable naming**: Use underscores, not spaces: `s_f0` not `s f0`
3. **Bracket types**: Use `[]` for dimensions, `{}` for values, `>` or `-` for connections
4. **Missing variables**: All variables in connections must be defined in StateSpaceBlock

See [Common Errors Guide](common_errors.md) for detailed troubleshooting.

### How do I know if my parameterization is correct?

**Validation checklist:**

1. **Probability constraints**: All rows/columns sum to 1
2. **Dimension compatibility**: Matrix sizes match variable definitions
3. **Type consistency**: Variables have compatible types
4. **Run the type checker**: `uv run python src/gnn/5_type_checker.py --target-dir <model-dir>`

### Can I use custom mathematical functions?

**Standard functions are supported:**

- `softmax`, `sigmoid`, `exp`, `log`, `ln`
- Basic arithmetic: `+`, `-`, `*`, `/`
- Matrix operations: `^T` (transpose), `^dagger` (pseudo-inverse)

**For custom functions:**

1. Define in the `Equations` section using LaTeX
2. Implement in your target rendering framework
3. Consider contributing back to the GNN specification

### How do I handle missing or partial observations?

**Strategies:**

1. **Explicit missing state**: Add "unobserved" outcome to observation modality
2. **Probabilistic observations**: Use likelihood matrices with uncertainty
3. **Hierarchical models**: Separate observation and availability processes

```gnn
# Example: Partial observations
o_m0[3,1,type=int]  # 0:Observed_A, 1:Observed_B, 2:Unobserved

A_m0={
  ((0.9, 0.1),   # P(obs_A | state_A, state_B)
   (0.1, 0.9),   # P(obs_B | state_A, state_B)  
   (0.3, 0.3))   # P(unobserved | state_A, state_B)
}
```

## 🚀 Advanced Usage

### How do I integrate with my existing codebase?

**Integration approaches:**

1. **Export to your language**: Use GNN renderers to generate code
2. **Direct parsing**: Parse GNN files in your application
3. **API integration**: Use GNN as configuration format
4. **Hybrid approach**: GNN for model structure, code for implementation details

### Can I use GNN for non-Active Inference models?

**Yes, with adaptations:**

- **Bayesian Networks**: Use StateSpaceBlock for variables, Connections for dependencies
- **Neural Networks**: Represent layers as state factors, weights as parameters
- **Markov Models**: Focus on transition structures in StateSpaceBlock
- **Custom models**: Adapt GNN sections to your needs

### How do I contribute new features?

**Contribution process:**

1. **Discuss**: Open GitHub issue or discussion
2. **Design**: Follow GNN design principles
3. **Implement**: Add to appropriate src/ modules
4. **Document**: Update relevant documentation
5. **Test**: Ensure examples work and pass validation
6. **Submit**: Create pull request with clear description

### Can I use GNN for real-time applications?

**Performance considerations:**

- **Model complexity**: Simpler models run faster
- **Target framework**: Julia (RxInfer) typically faster than Python
- **Compilation**: Pre-compile models when possible
- **Caching**: Cache inference results for repeated scenarios

See [Performance Guide](performance.md) for optimization strategies.

## 🔍 Troubleshooting

### My model runs but gives weird results

**Debugging steps:**

1. **Check parameterization**: Are probabilities normalized?
2. **Verify connections**: Do causal relationships make sense?
3. **Test edge cases**: What happens with extreme inputs?
4. **Compare with simpler model**: Does a reduced version work?
5. **Visualize**: Use GNN visualization tools to inspect structure

### How do I debug complex models?

**Systematic approach:**

1. **Start minimal**: Begin with 2x2 matrices
2. **Add complexity incrementally**: One variable at a time
3. **Use logging**: Enable verbose output in renderers
4. **Unit test components**: Test matrices and connections separately
5. **Compare frameworks**: Cross-validate between PyMDP and RxInfer

### Where can I get help?

**Support channels:**

1. **Documentation**: Start with [troubleshooting guides](common_errors.md)
2. **Examples**: Check similar models in `input/gnn_files/`
3. **GitHub Issues**: Report bugs and ask questions
4. **GitHub Discussions**: Community Q&A and brainstorming
5. **Active Inference Institute**: Connect with the broader community

## 📚 Learning Resources

### Recommended learning path

1. **GNN Basics**: [Overview](../gnn/gnn_overview.md) → [Syntax](../gnn/reference/gnn_syntax.md) → [Examples](../gnn/tutorials/gnn_examples_doc.md)
2. **Active Inference**: Smith et al. tutorial → Parr et al. textbook
3. **Implementation**: Choose framework and work through examples
4. **Advanced Topics**: Multi-agent, learning, optimization

### Key papers and resources

- **Smith, R. et al. (2022)**: Step-by-step Active Inference tutorial
- **Parr, T. et al. (2022)**: Active Inference textbook
- **Friston, K. (2010)**: Free Energy Principle foundations
- **Active Inference Institute**: Community and resources

### Practice projects

1. **Simple agent**: 2D grid navigation
2. **Perceptual inference**: Object recognition
3. **Decision making**: Multi-armed bandit
4. **Learning**: Parameter estimation in changing environment
5. **Multi-agent**: Coordination and communication

---

## 🔧 Installation and Setup

### I'm getting import errors when running GNN tools

**Common solutions:**

1. **Activate virtual environment**: Ensure you're in the correct Python environment

   ```bash
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

2. **Install missing dependencies**:

   ```bash
   uv sync  # Install runtime dependencies from pyproject.toml / uv.lock
   ```

3. **Python version compatibility**: GNN requires Python 3.11 through 3.13

   ```bash
   python --version  # Check your version
   ```

### How do I set up GNN for development?

**Development setup:**

```bash
# Clone and set up development environment
git clone https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git
cd Generalized_Notation_Notation

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies using UV (recommended)
uv sync --extra dev

# Run tests to verify setup
uv run --extra dev python -m pytest tests/
```

### Can I use GNN with Docker?

**Yes! Docker setup:**

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN uv sync --frozen
CMD ["uv", "run", "python", "src/gnn/main.py", "--help"]
```

**Docker Compose for development:**

```yaml
version: '3.8'
services:
  gnn:
    build: .
    volumes:
      - .:/app
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app/src
```

---

## 🚀 Framework-Specific Questions

### PyMDP-specific issues

#### My PyMDP model runs slowly

**Optimization strategies:**

1. **Use JAX backend**: `pymdp.utils.set_backend('jax')`
2. **Reduce precision**: Use `float32` instead of `float64`
3. **Vectorize operations**: Process multiple time steps in batches
4. **Pre-compile models**: Enable compilation flags

#### PyMDP gives different results than expected

**Debugging checklist:**

1. **Check matrix stochasticity**: Use `pymdp.utils.norm_dist()` to normalize
2. **Verify dimensions**: Ensure A and B matrices match state space definitions
3. **Policy depth**: Adjust `policy_len` parameter for planning horizon
4. **Precision parameters**: Check `alpha` (action precision) and `gamma` (state precision)

### RxInfer-specific issues

#### How do I install RxInfer for GNN?

**Julia setup:**

```julia
# Install Julia packages
using Pkg
Pkg.add("RxInfer")
Pkg.add("GraphPPL")
Pkg.add("JSON")  # For GNN model import
```

**Python-Julia bridge:**

```bash
uv pip install julia
python -c "import julia; julia.install()"
```

#### RxInfer models don't converge

**Common fixes:**

1. **Message passing schedule**: Adjust iteration limits and convergence criteria
2. **Prior specification**: Ensure proper prior distributions
3. **Precision parameters**: Use appropriate precision for your problem scale
4. **Initialization**: Try different starting values

### JAX-specific issues

#### JAX models fail with "Abstract tracer" errors

**Solutions:**

1. **Avoid Python control flow**: Use `jax.lax.cond` instead of `if/else`
2. **Use JAX arrays**: Convert numpy arrays with `jnp.array()`
3. **JIT compilation boundaries**: Structure code to avoid side effects in JIT
4. **Debugging mode**: Use `jax.config.update('jax_debug_nans', True)`

---

## 🔬 Advanced Modeling

### How do I model continuous state spaces?

**Approaches:**

1. **Native continuous linear-Gaussian models** (recommended when your
   dynamics are linear-Gaussian): declare F/H/Q/R matrices with
   `prior_mean`/`prior_cov` state factors. GNN renders these to JAX,
   NumPyro, PyTorch, Stan (LGSSM), RxInfer.jl, and cpomdp programs and executes them
   in Step 12. See `input/gnn_files/continuous/` for runnable exemplars such
   as `continuous_navigation.md`.

2. **Discretization**: Divide continuous space into bins

   ```gnn
   # Position discretized into 10 bins
   s_f0[10,1,type=int]  # Discrete position bins
   ```

3. **Gaussian approximations**: Use mean and variance parameters

   ```gnn
   # Continuous position with Gaussian beliefs
   s_f0_mean[1,type=float]    # Position mean
   s_f0_var[1,type=float]     # Position variance
   ```

4. **Particle filters**: Multiple discrete samples of continuous space

### How do I handle time-varying parameters?

**Dynamic parameterization:**

```gnn
## StateSpaceBlock
# Time-varying transition matrix
B_f0[4,4,4,type=float]     # Base transition matrix
B_modulation[4,4,type=float]  # Time-varying modulation
t_index[1,type=int]        # Current time index

## Connections
# Time-dependent transitions
B_modulation>B_f0
t_index>B_f0
```

### How do I implement learning in GNN models?

**Learning patterns:**

1. **Parameter learning**: Update matrix entries based on experience
2. **Structure learning**: Modify connections between variables
3. **Meta-learning**: Learn learning rules themselves
4. **Online adaptation**: Real-time parameter updates

**Example learning implementation:**

```gnn
## StateSpaceBlock
# Learnable parameters
A_m0[4,4,type=float]           # Current likelihood matrix
A_m0_learning_rate[1,type=float]  # Learning rate
prediction_error[4,type=float]    # Error signal for learning

## Connections
# Learning update rule
prediction_error>A_m0_learning_rate
A_m0_learning_rate>A_m0
```

### How do I model hierarchical goals?

**Goal hierarchy patterns:**

```gnn
## StateSpaceBlock
# Goal hierarchy levels
goal_level_0[3,type=float]     # Immediate goals (actions)
goal_level_1[4,type=float]     # Tactical goals (sequences)
goal_level_2[2,type=float]     # Strategic goals (objectives)

# Goal inheritance
goal_weight_01[3,4,type=float] # How tactical goals influence immediate
goal_weight_12[4,2,type=float] # How strategic goals influence tactical

## Connections
# Top-down goal propagation
goal_level_2>goal_weight_12
goal_weight_12>goal_level_1
goal_level_1>goal_weight_01
goal_weight_01>goal_level_0
```

---

## 💻 Integration and Workflows

### How do I integrate GNN with Jupyter notebooks?

**Jupyter integration:**

```python
# Install Jupyter extensions
uv pip install jupyter ipywidgets

# Run from the repository root
from gnn import parse_gnn_file, validate_gnn_source

# Load and process GNN model
model = parse_gnn_file('my_model.md')
validation_result = validate_gnn_source('my_model.md')

# Interactive widgets for parameter tuning
from ipywidgets import interact, FloatSlider
@interact(learning_rate=FloatSlider(min=0.01, max=0.5, step=0.01))
def tune_model(learning_rate):
    # Update model parameters and re-run
    pass
```

### How do I use GNN with version control?

**Git workflows:**

```bash
# Track GNN model changes
git add models/*.gnn
git commit -m "Add navigation agent model v1.2"

# Model diffing
git diff models/agent.gnn

# Branching for model experiments
git checkout -b experiment/hierarchical-planning
# Modify model
git add models/agent.gnn
git commit -m "Experiment: Add hierarchical planning layers"
```

**Model versioning best practices:**

1. **Semantic versioning**: Use version numbers in model names
2. **Change logs**: Document model modifications
3. **Migration behavior**: Document changes for older versions
4. **Branching strategy**: Separate branches for different model variants

### How do I automate GNN workflows?

**CI/CD pipeline example:**

```yaml
# .github/workflows/gnn-validation.yml
name: GNN Model Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        uv sync --frozen
    
    - name: Validate GNN models
      run: |
        uv run python src/gnn/5_type_checker.py --target-dir models/
    
    - name: Generate documentation
      run: |
        uv run python src/gnn/20_website.py --target-dir models/ --output-dir output --verbose
```

### How do I deploy GNN models to production?

**Deployment strategies:**

1. **REST API**: Wrap GNN-generated models in web services
2. **Containerization**: Use Docker for consistent environments
3. **Model serving**: Use frameworks like TensorFlow Serving or MLflow
4. **Edge deployment**: Compile models for embedded systems

**Example API deployment:**

```python
from flask import Flask, request, jsonify
from gnn_runtime import load_model, run_inference

app = Flask(__name__)
model = load_model("production_agent.gnn")


@app.route("/infer", methods=["POST"])
def infer():
    observations = request.json["observations"]
    action = run_inference(model, observations)
    return jsonify({"action": action})
```

---

## 🐛 Common Gotchas and Edge Cases

### My model validation passes but execution fails

**Common causes:**

1. **Numerical instability**: Very small or large probability values
2. **Memory issues**: Matrices too large for available RAM
3. **Framework limitations**: Target framework doesn't support all GNN features
4. **Runtime dependencies**: Missing libraries in execution environment

**Solutions:**

```gnn
# Add numerical stability constraints
## Constraints
probability_floor=1e-8     # Minimum probability value
probability_ceiling=1-1e-8 # Maximum probability value
normalize_matrices=true    # Auto-normalize probability matrices
```

### Matrix dimensions seem correct but I get size mismatch errors

**Hidden dimension issues:**

1. **Broadcasting**: Some frameworks handle singleton dimensions differently
2. **Time dimensions**: Dynamic models have implicit time dimensions
3. **Batch dimensions**: Multi-agent models may have hidden batch dimensions
4. **Index ordering**: Row-major vs column-major matrix ordering

**Debug matrix dimensions:**

```python
# In Python/PyMDP
print(f"A matrix shape: {A.shape}")
print(f"Expected shape: ({n_observations}, {n_states})")
print(f"B matrix shape: {B.shape}")
print(f"Expected shape: ({n_states}, {n_states}, {n_actions})")
```

### My agent gets stuck in loops or converges too quickly

**Behavioral debugging:**

1. **Exploration vs exploitation**: Adjust action precision parameters
2. **Planning horizon**: Increase/decrease `ModelTimeHorizon`
3. **Prior beliefs**: Check if priors are too strong or weak
4. **Preference specification**: Verify preference values and scaling

**Parameter tuning guidelines:**

```gnn
# Exploration parameters
alpha=16.0        # Higher = more deterministic actions
gamma=16.0        # Higher = more confident state estimates

# Planning parameters
ModelTimeHorizon=5    # Longer = more foresight, more computation
policy_depth=3        # How many steps ahead to plan

# Learning parameters
learning_rate=0.05    # Higher = faster adaptation, less stability
```

### State space explosion in multi-factor models

**Complexity management:**

1. **Mean-field approximations**: Approximate joint distributions as products
2. **Structured inference**: Exploit conditional independence
3. **Hierarchical decomposition**: Break complex states into levels
4. **Factorized representations**: Use separate factors for independent aspects

**Example factorization:**

```gnn
## StateSpaceBlock
# Instead of joint state s_f0[100,1,type=int] (10×10 grid)
# Use factorized representation:
s_f0_x[10,1,type=int]  # X coordinate
s_f0_y[10,1,type=int]  # Y coordinate
A_m0_x[10,10,type=float]  # X likelihood
A_m0_y[10,10,type=float]  # Y likelihood
o_m0_x[10,1,type=int]  # X observation
o_m0_y[10,1,type=int]  # Y observation

# Conditional independence assumption
## Connections
s_f0_x>A_m0_x  # X observations
A_m0_x>o_m0_x
s_f0_y>A_m0_y  # Y observations
A_m0_y>o_m0_y
```

---

## 🔄 Version Compatibility and Migration

### I have GNN models from an older version. Do they still work?

**Version compatibility:**

- **GNN v1.0 → v1.1**: Migration preserved the documented public syntax used by maintained examples
- **Pre-v1.0 → v1.x**: May require syntax updates
- **Framework versions**: PyMDP/RxInfer updates may affect rendering

**Migration checklist:**

1. **Syntax updates**: Check for retired section names
2. **Variable naming**: Ensure compliance with current naming conventions
3. **Matrix specifications**: Verify dimension specifications
4. **Validation**: Run updated type checker on old models

### How do I update my models to use new features?

**Feature adoption process:**

1. **Read release notes**: Understand new capabilities
2. **Test on copies**: Don't modify original models directly
3. **Incremental updates**: Add features gradually
4. **Validation at each step**: Ensure models still work

**Example migration (adding ontology annotations):**

```gnn
# Original model (pre-ontology)
s_f0[4,1,type=int]  # Hidden state

# Updated model (with ontology)
s_f0[4,1,type=int]  # Hidden state

## ActInfOntologyAnnotation
s_f0=SpatialPosition  # Map to ontology term
```

### What if a new GNN version breaks my model?

**Troubleshooting steps:**

1. **Check migration guide**: Look for breaking changes documentation
2. **Use validation tools**: Run diagnostic scripts
3. **Compare syntax**: Use diff tools to see what changed
4. **Gradual migration**: Update one section at a time
5. **Seek help**: Post issues with specific error messages

---

## 🤝 Community and Contributing

### How do I report a bug in GNN?

**Bug reporting template:**

1. **Minimal example**: Simplest GNN model that reproduces the issue
2. **Environment details**: Python version, OS, framework versions
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error messages**: Full stack traces and logs

**GitHub issue template:**

```markdown
**Bug Description**
Brief description of the issue

**GNN Model (minimal example)**
```gnn
## GNNVersionAndFlags
GNN v1

## ModelName
Bug Reproduction Example v1.0
[minimal model that reproduces issue]
```

**Environment**

- Python version: 3.9.7
- GNN version: v1.2
- Target framework: PyMDP 0.2.8
- OS: Ubuntu 20.04

**Steps to Reproduce**

1. Create model file
2. Run validation
3. See error

**Expected vs Actual Behavior**
Expected: Model should validate successfully
Actual: Validation fails with error X

```

### How do I contribute new features?
**Contribution workflow:**
1. **Discussion first**: Open GitHub discussion for large features
2. **Fork and branch**: Create feature branch from main
3. **Implementation**: Follow coding standards and add tests
4. **Documentation**: Update relevant docs and examples
5. **Pull request**: Submit with clear description and tests

**Feature development checklist:**
- [ ] **Tests added**: Unit tests for new functionality
- [ ] **Documentation updated**: User guides and API docs
- [ ] **Examples provided**: Working examples demonstrating feature
- [ ] **Migration behavior**: Documents model updates required by the change
- [ ] **Performance tested**: No significant performance regression

### How do I request new functionality?
**Feature request process:**
1. **Search existing**: Check if already requested
2. **Use case description**: Explain why feature is needed
3. **Proposed solution**: Suggest implementation approach
4. **Alternatives considered**: Other ways to achieve goal
5. **Community discussion**: Engage with other users

**Good feature request example:**
```markdown
**Feature Request: Nonlinear State Dynamics**

**Use Case**
I need to model robot navigation with continuous position coordinates where
the dynamics are nonlinear. GNN supports continuous linear-Gaussian models
(F/H/Q/R with `prior_mean`/`prior_cov`), but those dynamics are linear, so
this model is out of scope today.

**Proposed Solution** 
Allow `##Equations` blocks to attach a nonlinear transition function to
continuous state factors:
```gnn
s_f0[2,1,type=float]  # [x, y] position state
f_f0: x_{t+1} = A x_t + g(x_t) + w_t  # nonlinear residual g(x)
```

**Alternatives Considered**

- Discretization (loses precision)
- External preprocessing (breaks GNN workflow)

**Impact**
Would enable robotics, control theory, and nonlinear state-estimation use
cases beyond the existing linear-Gaussian lane.

```

---

## 📊 Performance and Optimization

### How do I profile GNN model performance?
**Performance analysis tools:**
```python
# Timing analysis
import time
import cProfile

def profile_gnn_pipeline():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your GNN workflow
    model = parse_gnn_file('large_model.md')
    result = validate_gnn_source('large_model.md')
    
    pr.disable()
    pr.print_stats(sort='time')

# Memory profiling
import tracemalloc
tracemalloc.start()

# Your GNN code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

### My GNN processing pipeline is too slow

**Optimization strategies:**

1. **Parallel processing**: Process multiple models simultaneously
2. **Caching**: Cache parsed models and validation results
3. **Incremental validation**: Only re-validate changed sections
4. **Lazy loading**: Load model components on demand

**Pipeline optimization example:**

```python
from multiprocessing import Pool
from functools import lru_cache


@lru_cache(maxsize=128)
def cached_parse_gnn(file_path, file_mtime):
    """Cache parsing results based on file modification time."""
    return parse_gnn_file(file_path)


def parallel_process_models(model_files, num_workers=4):
    """Process multiple GNN models in parallel."""
    with Pool(num_workers) as pool:
        results = pool.map(process_single_model, model_files)
    return results
```

### How do I optimize large-scale simulations?

**Simulation optimization:**

1. **Batch processing**: Process multiple episodes together
2. **Approximation methods**: Use faster approximate inference
3. **Model compression**: Reduce model complexity where possible
4. **Hardware acceleration**: Use GPU/TPU for computation

**Large-scale simulation patterns:**

```python
# Vectorized batch processing
import jax.numpy as jnp
from jax import vmap

# Process multiple agents simultaneously
batch_size = 1000
observations = jnp.array([obs for obs in observation_batch])
actions = vmap(agent.infer_action)(observations)  # Vectorized inference


# Memory-efficient iteration
def chunked_simulation(agents, total_steps, chunk_size=100):
    """Process simulation in chunks to manage memory."""
    for start_step in range(0, total_steps, chunk_size):
        end_step = min(start_step + chunk_size, total_steps)
        chunk_results = run_simulation_chunk(agents, start_step, end_step)
        yield chunk_results  # Generator pattern for memory efficiency
```

---

## 💡 Still have questions?

### Quick Help Resources

- **Search this FAQ**: Use Ctrl+F to find specific topics
- **Documentation index**: [Main documentation](../README.md)
- **Examples gallery**: Model examples in `input/gnn_files/`
- **Syntax reference**: [GNN syntax guide](../gnn/reference/gnn_syntax.md)

### Community Support

- **GitHub Discussions**: [Community Q&A](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/discussions)
- **GitHub Issues**: [Bug reports and feature requests](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/issues)
- **Active Inference Institute**: [Broader community](https://www.activeinference.org/)

### Getting Deeper Help

1. **Read the error message carefully**: Most issues have helpful error descriptions
2. **Check similar problems**: Search existing GitHub issues
3. **Create minimal examples**: Isolate the problem to simplest case
4. **Provide context**: Include environment details and full error traces
5. **Be specific**: "It doesn't work" is less helpful than "Validation fails at line 42 with error X"

---

**FAQ Version**: Compatible with GNN v1.x  
**Covers**: Installation, Modeling, Frameworks, Advanced Topics, Troubleshooting, Contributing
