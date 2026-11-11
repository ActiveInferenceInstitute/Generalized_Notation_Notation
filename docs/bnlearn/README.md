# bnlearn: Causal Discovery and Bayesian Networks

[![Python](https://img.shields.io/pypi/pyversions/bnlearn)](https://img.shields.io/pypi/pyversions/bnlearn)
[![PyPI Version](https://img.shields.io/pypi/v/bnlearn)](https://pypi.org/project/bnlearn/)
![GitHub Repo stars](https://img.shields.io/github/stars/erdogant/bnlearn)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/bnlearn/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/bnlearn.svg)](https://github.com/erdogant/bnlearn/network)
[![Open Issues](https://img.shields.io/github/issues/erdogant/bnlearn.svg)](https://github.com/erdogant/bnlearn/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/bnlearn/month)](https://pepy.tech/project/bnlearn/)
[![DOI](https://zenodo.org/badge/231263493.svg)](https://zenodo.org/badge/latestdoi/231263493)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/bnlearn/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://erdogant.github.io/bnlearn/notebook.html)

<div>
<a href="https://erdogant.github.io/bnlearn/"><img src="https://github.com/erdogant/bnlearn/raw/master/docs/figs/logo.png" width="175" align="left" style="margin-right: 15px;" /></a>
<code>bnlearn</code> is a comprehensive Python package for causal discovery by learning the graphical structure of Bayesian networks, parameter learning, inference, and sampling methods. Because probabilistic graphical models can be difficult to use, <code>bnlearn</code> bundles the most-wanted pipelines into a unified, functional, and highly configurable interface.
</div>

<br/>

Navigate to the [API documentation](https://erdogant.github.io/bnlearn/) for more detailed information.

---

## Key Features

| Feature | Description | Medium Blog |
|--------|-------------|---|
| [**Causal Discovery & Starters Guide**](https://erdogant.github.io/bnlearn/Structure%20learning.html) | Learn the basics of causal modelling. | [Read Here](https://medium.com/data-science-collective/the-starters-guide-to-causal-structure-learning-with-bayesian-methods-in-python-e3b90f49c99c) |
| [**Structure Learning**](https://erdogant.github.io/bnlearn/Structure%20learning.html) | Learn model structures from data or via expert knowledge. | [Read Here](https://medium.com/data-science-collective/the-complete-starter-guide-for-causal-discovery-using-bayesian-modeling-8853eb860d02) |
| [**Causal Predictions**](https://erdogant.github.io/bnlearn/Structure%20learning.html) | Learn to make causal predictions. | [Read Here](https://medium.com/data-science-collective/why-prediction-isnt-enough-using-bayesian-models-to-change-the-outcome-5c9cf9f65a75) |
| [**Parameter Learning**](https://erdogant.github.io/bnlearn/Parameter%20learning.html) | Estimate CPDs (Conditional Probability Distributions) from observed data. | [Read Here](https://medium.com/data-science-collective/human-machine-collaboration-with-bayesian-modeling-learn-to-combine-knowledge-with-data-1ee9bcd67745) |
| [**Causal Inference**](https://pgmpy.org/) | Compute interventional and counterfactual distributions via do-calculus. | [Read Here](https://medium.com/data-science-collective/chat-with-your-dataset-using-bayesian-inferences-1afdbfd4bada) |
| [**Generate Synthetic Data**](https://erdogant.github.io/bnlearn/Sampling.html) | Generate synthetic samples using probabilistic models. | [Read Here](https://medium.com/data-science-collective/synthetic-data-the-essentials-of-data-generation-using-bayesian-sampling-6d072e97e09d) |
| [**Discretize Data**](https://erdogant.github.io/bnlearn/Discretizing.html) | Categorize and discretize continuous datasets. | — |
| [**Causal Library Comparison**](https://erdogant.github.io/bnlearn/Discretizing.html) | Benchmarks and comparisons with other causal libraries. | [Read Here](https://medium.com/data-science-collective/six-causal-libraries-compared-which-bayesian-approach-finds-hidden-causes-in-your-data-9fa66fd02825) |

---

## Installation

**Install from PyPI (Recommended):**
```bash
pip install bnlearn
```

**Install from GitHub Source:**
```bash
pip install git+https://github.com/erdogant/bnlearn
```

---

## Available Pipelines & Functions

`bnlearn` offers a deeply modular and streamlined toolset covering the complete lifecycle of Bayesian graphical networks:

### Key Pipelines
- **Structure Learning**: `bn.structure_learning.fit()`
- **Parameter Learning**: `bn.parameter_learning.fit()`
- **Inference**: `bn.inference.fit()`
- **Predictions**: `bn.predict()`
- **Sampling/Synthetic Data**: `bn.sampling()`
- **Edge Strength Independence Tests**: `bn.independence_test()`

### Graph & Metric Manipulations
- **Data Transformation/Imputation**: `bn.discretize()`, `bn.knn_imputer()`, `bn.df2onehot()`
- **Matrix Operations**: `bn.adjmat2vec()`, `bn.vec2adjmat()`, `bn.dag2adjmat()`
- **Topological & Structure Scores**: `bn.topological_sort()`, `bn.structure_scores()`
- **Custom DAG Generation**: `bn.make_DAG()`, `bn.generate_cpt()`, `bn.build_cpts_from_structure()`

### Persistance & I/O
- Save/Load models seamlessly: `bn.save()`, `bn.load()`, `bn.import_DAG()`

### Plotting
- Integrated graphing: `bn.plot()`, `bn.plot_graphviz()`, `bn.compare_networks()`

---

## Unified Operational Examples

### 1. Unified Structure Learning & Plotting

Demonstrating the streamlined process of importing data, learning a structure, computing edge strength metrics, and plotting the resultant DAG.

```python
import bnlearn as bn

# 1. Load the example dataframe (e.g. sprinkler dataset)
df = bn.import_example()

# 2. Learn structure
model = bn.structure_learning.fit(df)

# 3. Compute edge strength with the chi-square test statistic
model = bn.independence_test(model, df)

# 4. Plot Directed Acyclic Graph (DAG)
G = bn.plot(model)
```

#### Highly Configurable Structure Learning Methods
```python
# Structure Learning algorithms with diverse scoring methodologies
model_hc_bic = bn.structure_learning.fit(df, methodtype="hc", scoretype="bic")
model_hc_k2 = bn.structure_learning.fit(df, methodtype="hc", scoretype="k2")
model_hc_bdeu = bn.structure_learning.fit(df, methodtype="hc", scoretype="bdeu")
model_ex_bic = bn.structure_learning.fit(df, methodtype="ex", scoretype="bic")
model_cl = bn.structure_learning.fit(df, methodtype="cl", root_node="Wet_Grass")
model_tan = bn.structure_learning.fit(
    df, methodtype="tan", root_node="Wet_Grass", class_node="Rain"
)
```

### 2. Functional Parameter Learning

Demonstrating the assignment and fitting of parameters over an established topological structure.

```python
import bnlearn as bn

df = bn.import_example()

# Instantiate an 'empty' DAG architecture
model = bn.import_DAG("sprinkler", CPD=False)

# Confidently learn parameters using the data framework
model_update = bn.parameter_learning.fit(model, df)

# Visually verify
G = bn.plot(model_update)
```

### 3. Logged Inference Testing

Conducting robust queries applying specific evidence parameters.

```python
import bnlearn as bn

# Import established model
model = bn.import_DAG("sprinkler")

# Query 1: Evaluate Rain probability given evidence
query_1 = bn.inference.fit(
    model, variables=["Rain"], evidence={"Cloudy": 1, "Sprinkler": 0, "Wet_Grass": 1}
)
print("Query 1 Output:", query_1)
print("Dataframe Record:\n", query_1.df)

# Query 2: Evaluate Rain probability with reduced causal evidence
query_2 = bn.inference.fit(model, variables=["Rain"], evidence={"Cloudy": 1})
print("\nQuery 2 Output:", query_2)
print("Dataframe Record:\n", query_2.df)
```

---

## Resources & Integrations

- **Example Notebooks:** [Google Colab Library](https://erdogant.github.io/bnlearn/notebook.html)
- **Documentation Hub:** [Official Site](https://erdogant.github.io/bnlearn)
- **Bug Reporting & Features:** [GitHub Tracker](https://github.com/erdogant/bnlearn/issues)

### GNN Pipeline Integration

In GNN, the `bnlearn` generator compiles complete stationary finite A/B/D models into native exact fixed-action filters: rendered scripts run at pipeline Step 12 via [`src/gnn/execute/bnlearn/`](../../src/gnn/execute/bnlearn/) with output under `BNLEARN_OUTPUT_DIR`. Scripts skip when the `bnlearn` extra is absent (`gnn.utils.runtime_safety.framework_availability`). Continuous linear-Gaussian models are unsupported (`supports_continuous=False` in `src/gnn/render/framework_registry.py`).

The source compiler retains authored and canonical CPTs, source timesteps,
normalization provenance and model/script identities. Independent native tests
check complete posterior sequences. C/E policy optimization, causal discovery
and component composition remain outside this filtering contract. Metadata-only
compatibility calls still emit a labelled synthetic demonstration. See the
[renderer contract](../../src/gnn/render/bnlearn/README.md).


## Documents in this folder

- [Comprehensive bnlearn Guide](bnlearn.md)
- [bnlearn-GNN Integration Architecture](bnlearn-gnn.md)
