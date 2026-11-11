# ActiveInference.jl Consolidated Examples

This directory contains a streamlined, fully functional ActiveInference.jl examples system that has been consolidated from multiple separate scripts into a single comprehensive runner.

## Files

- `activeinference_runner.jl` - **Main consolidated script** - runs all examples and generates outputs
- `visualization_utils.jl` - **Visualization utilities** - creates analysis and plots from simulation data
- `README.md` - This documentation file

## What the Consolidated Runner Does

The single `activeinference_runner.jl` script provides the complete ActiveInference.jl workflow:

### 1. **Environment Setup**
- Automatically installs required Julia packages (ActiveInference, Distributions, etc.)
- Sets up comprehensive logging with timestamped output directories
- Validates package loading and environment

### 2. **Basic POMDP Simulation**
- Creates simple 2-state, 2-observation POMDP model using correct API
- Initializes Active Inference agent with proper settings
- Runs 20-step simulation with state inference and action selection
- Records observations, actions, and beliefs over time

### 3. **Parameter Learning Example**
- Demonstrates agent learning observation model parameters
- Sets up "true" target parameters and imprecise initial beliefs
- Runs 30 learning episodes with parameter updates
- Tracks learning progress and parameter convergence

### 4. **Multi-Step Planning Example**  
- Creates 3-state navigation task with 3-step lookahead planning
- Sets up transition dynamics and preference structure
- Demonstrates forward planning and policy selection
- Records planning behavior and reward accumulation

### 5. **Comprehensive Analysis & Visualization**
- Generates detailed analysis summaries of all simulation data
- Creates text-based statistical analyses
- Attempts graphical plotting if visualization packages available
- Organizes all outputs into structured directories

## Key Improvements

### ✅ **Consolidated & Simplified**
- Single script replaces 5+ separate demo scripts
- Eliminates code duplication and redundancy
- Uses working ActiveInference.jl API calls validated by testing

### ✅ **Fully Functional Implementation**
- All examples are runnable and use explicit data
- Based on successful `demo_success.jl` API usage
- Real Active Inference computations and data generation

### ✅ **Comprehensive Data Output**
- Organized output directory structure with timestamps
- CSV data files with extensive metadata headers
- Detailed logging of all operations and results
- Analysis summaries and recommendations

### ✅ **Scientific Rigor**
- Reproducible random seeds for consistent results
- Proper error handling and graceful failure modes
- Comprehensive metadata saved with all data files
- Performance tracking and resource monitoring

## How to Run

### Prerequisites

- Julia 1.6+ installed
- Internet connection (for package installation)

### Quick Start

1. Navigate to this directory:
   ```bash
   cd docs/activeinference_jl/actinf_jl_src
   ```

2. Run the consolidated script:
   ```bash
   julia activeinference_runner.jl
   ```

3. Optionally run visualization utilities on the results:
   ```bash
   julia visualization_utils.jl activeinference_outputs_YYYY-MM-DD_HH-MM-SS
   ```

### What Happens

1. **Automatic Setup**: Script installs packages and sets up environment
2. **Timestamped Output**: Creates `activeinference_outputs_YYYY-MM-DD_HH-MM-SS/` directory
3. **Sequential Execution**: Runs all examples in sequence with progress reporting
4. **Complete Results**: Generates data files, logs, analysis, and visualizations

## Output Structure

```
activeinference_outputs_[timestamp]/
├── logs/
│   └── activeinference_run.log          # Detailed execution log
├── models/
│   └── basic_model_structure.csv        # Model architecture information
├── parameters/
│   ├── learning_progress.csv            # Parameter learning over time
│   └── learned_vs_true.csv             # Final parameter comparison
├── simulation_results/
│   ├── basic_simulation.csv             # Basic POMDP simulation data
│   └── planning_summary.csv            # Multi-step planning results
└── analysis/
    ├── comprehensive_analysis.txt        # Overall analysis summary
    ├── basic_simulation_analysis.txt     # Basic simulation statistics
    ├── learning_analysis.txt            # Learning progress analysis
    ├── planning_analysis.txt            # Planning performance analysis
    ├── parameter_comparison.txt          # Parameter learning comparison
    └── plots/                           # Graphical visualizations (if available)
        ├── actions_over_time.png
        ├── beliefs_over_time.png
        ├── learning_curve.png
        └── planning_rewards.png
```

## New: Full Trace and Visualization Data Outputs

- **All per-step/episode traces** from every simulation, learning, and planning run are saved as CSVs in `/data_traces/`.
- **All data underlying each plot** (e.g., beliefs, actions, rewards) is saved as CSVs in `/analysis/plots/`.
- **All analysis and visualization scripts** use only these real, saved data files. If any required data is missing or empty, the scripts halt with a clear error.
- **File locations and sizes** are logged for every data file used, ensuring full traceability and reproducibility.

### Example Output Structure

```
activeinference_outputs_[timestamp]/
├── data_traces/
│   ├── basic_simulation_trace.csv      # Per-step trace for basic simulation
│   ├── learning_trace.csv              # Per-episode trace for learning
│   └── planning_trace.csv              # Per-trial trace for planning
├── analysis/
│   ├── basic_simulation_analysis.txt   # Text-based analysis (from trace)
│   ├── learning_analysis.txt           # Text-based analysis (from trace)
│   ├── planning_analysis.txt           # Text-based analysis (from trace)
│   └── plots/
│       ├── beliefs_over_time.csv       # Data for beliefs plot
│       ├── actions_over_time.csv       # Data for actions plot
│       ├── observations_over_time.csv  # Data for observations plot
│       ├── learning_curve.csv          # Data for learning curve plot
│       ├── learning_comparison.csv     # Data for learning comparison plot
│       ├── planning_rewards.csv        # Data for planning rewards plot
│       ├── planning_actions.csv        # Data for planning actions plot
│       └── [corresponding .png files]
...
```

## Scientific Reproducibility and Traceability

- **Examples use explicit data and report unavailable inputs clearly.**
- **All data for analysis and visualization is real, saved, and checked for completeness.**
- **If any required data is missing or empty, the pipeline halts with a clear error.**
- **File locations and sizes are logged for every data file used.**
- **This ensures full scientific reproducibility and auditability of all results.**

## Removed/Consolidated Scripts

The following scripts have been **removed** as their functionality is now integrated into the main runner:

- ~~`demo_success.jl`~~ → Core functionality integrated into `activeinference_runner.jl`
- ~~`run_activeinference_examples.jl`~~ → Advanced examples consolidated into main runner
- ~~`simple_working_examples.jl`~~ → Simple examples included in main runner
- ~~`test_environment.jl`~~ → Environment testing included in setup phase

This consolidation eliminates redundancy while preserving all working functionality.

## Visualization Options

### Text-Based Analysis (Always Available)
- Statistical summaries of simulation data
- Learning progress analysis
- Planning performance metrics  
- Parameter comparison tables

### Graphical Visualizations (Optional)
- Time series plots of actions and beliefs
- Learning curves showing parameter convergence
- Planning reward distributions
- Comparative parameter visualizations

Graphical plots require optional packages (Plots.jl) which are automatically installed if possible.

## Integration with GNN Pipeline

This consolidated system is designed to integrate seamlessly with the broader GeneralizedNotationNotation (GNN) pipeline:

- **Consistent Output Structure**: Follows GNN standards for organized data directories
- **Comprehensive Logging**: Uses structured logging compatible with GNN pipeline tools
- **Modular Design**: Can be called as component within larger GNN workflows
- **Data Compatibility**: Generates outputs suitable for further GNN processing steps

## Troubleshooting

### Package Installation Issues
```julia
using Pkg
Pkg.update()
Pkg.add("ActiveInference")  
```

### Julia Version Issues
Ensure Julia 1.6+:
```bash
julia --version
```

### Permission Issues
Ensure write permissions in current directory.

### API Issues
The consolidated script uses the verified working API calls from successful testing. If you encounter ActiveInference.jl API issues, check for package version updates.

## Scientific Validation

✅ **API Verification**: All ActiveInference.jl function calls tested and validated  
✅ **Mathematical Correctness**: Proper Active Inference model construction and inference  
✅ **Reproducibility**: Deterministic behavior with fixed random seeds  
✅ **Data Integrity**: Comprehensive metadata and provenance tracking  
✅ **Performance Monitoring**: Resource usage tracking and optimization  

## License

This code follows the same license as the parent GNN project and ActiveInference.jl (MIT License). 


## Documents in this folder

- [ActiveInference.jl POMDP Gridworld Simulation](POMDP_README.md)
