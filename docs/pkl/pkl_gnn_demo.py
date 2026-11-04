#!/usr/bin/env python3
"""
pkl_gnn_demo.py - Comprehensive demonstration of Apple's Pkl integration with GNN

This script demonstrates how Apple's Pkl configuration language could enhance
the Generalized Notation Notation (GNN) project for Active Inference models.

Key Features Demonstrated:
1. Type-safe model definitions with validation
2. Template inheritance for Active Inference patterns
3. Multi-format output generation (JSON, YAML, XML)
4. Mathematical constraint validation
5. Configuration-as-code paradigm
6. Late binding and dynamic configuration

Based on research from:
- https://pkl-lang.org/
- https://github.com/apple/pkl
- Active Inference literature and GNN specifications
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class PklGNNDemo:
    """Demonstrates Pkl integration with GNN for Active Inference models."""

    def __init__(self, use_examples_dir=False):
        if use_examples_dir:
            self.temp_dir = Path("docs/pkl/examples")
            self.temp_dir.mkdir(exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="pkl_gnn_demo_"))
        self.pkl_executable = self._find_pkl_executable()
        print(f"🔧 Demo workspace: {self.temp_dir}")

    def _find_pkl_executable(self) -> Optional[str]:
        """Find Pkl executable in system PATH."""
        for pkl_name in ["pkl", "jpkl", "./pkl"]:
            if subprocess.run(["which", pkl_name], capture_output=True).returncode == 0:
                return pkl_name
        print("⚠️  Pkl executable not found. Install from: https://pkl-lang.org/")
        return None

    def create_base_active_inference_template(self) -> Path:
        """Create base Active Inference model template in Pkl."""
        template_content = """
/// Base Active Inference Model Template
/// Provides foundational structure for all Active Inference models
module BaseActiveInferenceModel

import "pkl:math"

/// Model metadata and identification
modelName: String(!isEmpty)
modelVersion: String = "1.0.0"
modelAnnotation: String = ""

/// GNN version and processing flags
gnnVersion: String = "2.0"
processingFlags: Mapping<String, Any> = new {}

/// Active Inference Ontology mapping
actInfOntology: Mapping<String, String> = new {
  ["hasStateSpace"] = "true"
  ["hasObservationModel"] = "true"
  ["hasPriors"] = "true"
}

/// State space definition with GNN naming conventions
class StateVariable {
  /// Variable name following GNN conventions (s_f0, o_m0, u_c0)
  name: String(!isEmpty && matches(Regex("(s_f|o_m|u_c)\\d+")))
  
  /// Dimensionality constraints
  dimensions: List<Int>(length.isBetween(1, 4) && every { it > 0 })
  
  /// Variable type
  variableType: "categorical" | "continuous" | "binary"
  
  /// Optional description
  description: String = ""
  
  /// Validation function
  function validate() {
    dimensions.every { it <= 1000 } // Practical size limit
  }
}

/// Hidden states (internal model states)
hiddenStates: Mapping<String, StateVariable> = new {}

/// Observations (observable outcomes) 
observations: Mapping<String, StateVariable> = new {}

/// Actions/Control variables
actions: Mapping<String, StateVariable> = new {}

/// Likelihood matrix A: P(o|s)
class LikelihoodMatrix {
  name: String = "A"
  dimensions: List<Int>(length == 2)
  
  /// Matrix values with stochasticity constraints
  values: List<List<Float>>(
    length == dimensions[0] &&
    every { row -> 
      row.length == dimensions[1] && 
      row.every { it >= 0.0 && it <= 1.0 } &&
      math.abs(row.sum() - 1.0) < 1e-10 // Normalization check
    }
  )
  
  modality: String = "m0"
  factor: String = "f0"
}

/// Transition matrix B: P(s'|s,u)
class TransitionMatrix {
  name: String = "B"
  dimensions: List<Int>(length == 3) // [states, states, actions]
  
  /// 3D tensor with stochasticity constraints
  values: List<List<List<Float>>>(
    length == dimensions[0] &&
    every { slice ->
      slice.length == dimensions[1] &&
      slice.every { row ->
        row.length == dimensions[2] &&
        row.every { it >= 0.0 && it <= 1.0 } &&
        math.abs(row.sum() - 1.0) < 1e-10
      }
    }
  )
  
  factor: String = "f0"
}

/// Preference vector C: log preferences
class PreferenceVector {
  name: String = "C"
  dimensions: List<Int>(length == 1)
  values: List<Float>(length == dimensions[0])
  modality: String = "m0"
}

/// Prior vector D: initial state priors
class PriorVector {
  name: String = "D"
  dimensions: List<Int>(length == 1)
  
  /// Probability distribution constraints
  values: List<Float>(
    length == dimensions[0] &&
    every { it >= 0.0 && it <= 1.0 } &&
    math.abs(sum() - 1.0) < 1e-10
  )
  
  factor: String = "f0"
}

/// Time configuration
class TimeConfiguration {
  modelType: "Dynamic" | "Static" = "Dynamic"
  timeDiscretization: "DiscreteTime" | "ContinuousTime" = "DiscreteTime"
  timeHorizon: Int(isBetween(1, 1000)) = 10
}

/// Required Active Inference matrices
A: LikelihoodMatrix?
B: TransitionMatrix?
C: PreferenceVector?
D: PriorVector?

timeSettings: TimeConfiguration = new {}

/// Model validation
function validateActiveInferenceStructure() {
  hiddenStates.keys.every { it.startsWith("s_f") } &&
  observations.keys.every { it.startsWith("o_m") } &&
  actions.keys.every { it.startsWith("u_c") } &&
  (A != null || throw("Likelihood matrix A is required")) &&
  (B != null || throw("Transition matrix B is required"))
}

/// Export configuration
output {
  renderer = new YamlRenderer {}
}
"""

        template_path = self.temp_dir / "BaseActiveInferenceModel.pkl"
        template_path.write_text(template_content.strip())
        return template_path

    def create_visual_foraging_model(self) -> Path:
        """Create a specific Visual Foraging model extending the base template."""
        model_content = '''
/// Visual Foraging Active Inference Model
/// Demonstrates template inheritance and model specialization
amends "BaseActiveInferenceModel.pkl"

modelName = "VisualForagingAgent"
modelAnnotation = """
An Active Inference model of visual foraging behavior.
The agent navigates a 2x2 grid environment searching for rewards.
Features location and context state factors with visual observations.
"""

/// Hidden state factors
hiddenStates = new {
  ["s_f0"] = new StateVariable {
    name = "s_f0"
    dimensions = [4]
    variableType = "categorical"
    description = "Spatial location factor (2x2 grid)"
  }
  ["s_f1"] = new StateVariable {
    name = "s_f1" 
    dimensions = [2]
    variableType = "categorical"
    description = "Context factor (reward/no-reward)"
  }
}

/// Observation modalities
observations = new {
  ["o_m0"] = new StateVariable {
    name = "o_m0"
    dimensions = [4]
    variableType = "categorical" 
    description = "Visual observations of environment"
  }
}

/// Action factors
actions = new {
  ["u_c0"] = new StateVariable {
    name = "u_c0"
    dimensions = [4]
    variableType = "categorical"
    description = "Movement actions (up, down, left, right)"
  }
}

/// Likelihood matrix A[m0]: P(o_m0|s_f0)
A = new LikelihoodMatrix {
  name = "A_m0"
  dimensions = [4, 4]
  values = [
    [0.9, 0.05, 0.05, 0.0],  // Location 0: mostly observe state 0
    [0.05, 0.9, 0.05, 0.0],  // Location 1: mostly observe state 1
    [0.05, 0.05, 0.9, 0.0],  // Location 2: mostly observe state 2
    [0.0, 0.0, 0.0, 1.0]     // Location 3: always observe state 3 (reward)
  ]
  modality = "m0"
  factor = "f0"
}

/// Transition matrix B[f0]: P(s_f0'|s_f0,u_c0)
B = new TransitionMatrix {
  name = "B_f0"
  dimensions = [4, 4, 4]
  values = [
    // From state 0
    [
      [1.0, 0.0, 0.0, 0.0],  // Stay (action 0)
      [0.0, 1.0, 0.0, 0.0],  // Right (action 1)
      [0.0, 0.0, 1.0, 0.0],  // Down (action 2)
      [0.0, 0.0, 0.0, 1.0]   // Diagonal (action 3)
    ],
    // From state 1
    [
      [1.0, 0.0, 0.0, 0.0],  // Left (action 0)
      [0.0, 1.0, 0.0, 0.0],  // Stay (action 1)
      [0.0, 0.0, 0.0, 1.0],  // Down (action 2)
      [0.0, 0.0, 1.0, 0.0]   // Diagonal (action 3)
    ],
    // From state 2
    [
      [1.0, 0.0, 0.0, 0.0],  // Up (action 0)
      [0.0, 0.0, 0.0, 1.0],  // Right (action 1)
      [0.0, 0.0, 1.0, 0.0],  // Stay (action 2)
      [0.0, 1.0, 0.0, 0.0]   // Diagonal (action 3)
    ],
    // From state 3 (absorbing reward state)
    [
      [0.0, 0.0, 1.0, 0.0],  // Up (action 0)
      [0.0, 1.0, 0.0, 0.0],  // Left (action 1)
      [1.0, 0.0, 0.0, 0.0],  // Diagonal (action 2)
      [0.0, 0.0, 0.0, 1.0]   // Stay (action 3)
    ]
  ]
  factor = "f0"
}

/// Preferences C[m0]: log preferences over observations
C = new PreferenceVector {
  name = "C_m0"
  dimensions = [4]
  values = [0.0, 0.0, 0.0, 2.0]  // Strong preference for reward observation
  modality = "m0"
}

/// Initial state priors D[f0]
D = new PriorVector {
  name = "D_f0"
  dimensions = [4]
  values = [0.25, 0.25, 0.25, 0.25]  // Uniform prior over locations
  factor = "f0"
}

/// Time configuration for episodic foraging
timeSettings = new TimeConfiguration {
  modelType = "Dynamic"
  timeDiscretization = "DiscreteTime"
  timeHorizon = 15
}

/// Enhanced ontology mapping
actInfOntology = (super.actInfOntology) {
  ["hasActions"] = "true"
  ["hasPreferences"] = "true"
  ["behaviorType"] = "foraging"
  ["environmentType"] = "spatial_grid"
}

/// Simulation parameters
simulationConfig = new {
  trials = 100
  learningRate = 0.1
  precision = 16.0
  policyDepth = 3
}
'''

        model_path = self.temp_dir / "VisualForagingModel.pkl"
        model_path.write_text(model_content.strip())
        return model_path

    def create_pipeline_configuration(self) -> Path:
        """Create GNN pipeline configuration demonstrating advanced Pkl features."""
        pipeline_content = """
/// GNN Pipeline Configuration
/// Demonstrates dynamic configuration and late binding
module GNNPipelineConfig

import "pkl:platform"

/// Pipeline steps configuration
class StepConfig {
  enabled: Boolean = true
  timeout: Duration = 30.s
  retries: Int(isBetween(0, 5)) = 3
  priority: "low" | "medium" | "high" = "medium"
}

/// Export target configuration
class ExportTarget {
  format: "json" | "yaml" | "xml" | "graphml" | "csv"
  pretty: Boolean = true
  validate: Boolean = true
  outputPath: String?
}

/// Rendering engine configuration
class RenderingEngine {
  target: "pymdp" | "rxinfer" | "jax" | "custom"
  optimizationLevel: Int(isBetween(0, 3)) = 2
  includeComments: Boolean = true
  typeHints: Boolean = true
}

/// LLM integration configuration
class LLMConfig {
  provider: "openai" | "anthropic" | "local"
  model: String
  temperature: Float(isBetween(0.0, 2.0)) = 0.1
  maxTokens: Int(isBetween(100, 10000)) = 4000
  enhancementTasks: List<String>
}

/// Main pipeline configuration
inputFormats: List<String> = ["pkl", "markdown", "json"]

/// Pipeline steps with late binding
steps: Mapping<String, StepConfig> = new {
  ["gnn_parse"] = new StepConfig {
    enabled = true
    timeout = 30.s
  }
  ["validation"] = new StepConfig {
    enabled = true
    timeout = 60.s
    priority = "high"
  }
  ["export"] = new StepConfig {
    enabled = true
    timeout = 45.s
  }
  ["render"] = new StepConfig {
    enabled = exportTargets.any { it.format != "json" }  // Late binding
    timeout = (renderingEngines.length * 30).s  // Dynamic timeout
  }
  ["visualization"] = new StepConfig {
    enabled = platform.current.os.name != "windows"  // Platform-specific
    timeout = 120.s
  }
}

/// Export targets with validation
exportTargets: List<ExportTarget> = new {
  new ExportTarget {
    format = "json"
    pretty = true
    outputPath = "output/model.json"
  }
  new ExportTarget {
    format = "yaml"
    pretty = true
    outputPath = "output/model.yaml"
  }
  new ExportTarget {
    format = "xml"
    validate = true
    outputPath = "output/model.xml"
  }
  new ExportTarget {
    format = "graphml"
    outputPath = "output/model.graphml"
  }
}

/// Rendering engines configuration
renderingEngines: List<RenderingEngine> = new {
  new RenderingEngine {
    target = "pymdp"
    optimizationLevel = 2
    includeComments = true
  }
  new RenderingEngine {
    target = "rxinfer"
    optimizationLevel = 3
    typeHints = true
  }
  new RenderingEngine {
    target = "jax"
    optimizationLevel = 3
    includeComments = false
  }
}

/// LLM enhancement configuration
llmConfig: LLMConfig? = new {
  provider = "openai"
  model = "gpt-4"
  temperature = 0.1
  maxTokens = 4000
  enhancementTasks = List(
    "model_analysis",
    "parameter_optimization", 
    "documentation_generation"
  )
}

/// Resource constraints with dynamic calculation
resourceConstraints: Mapping<String, Any> = new {
  ["maxMemoryMB"] = exportTargets.length * 256  // Scale with exports
  ["maxExecutionTimeMin"] = steps.values.map { it.timeout.value }.sum() / 60
  ["parallelWorkers"] = platform.current.availableProcessors
  ["tempDiskSpaceMB"] = renderingEngines.length * 512
}

/// Validation rules
validationRules: Mapping<String, Any> = new {
  ["strictTypeChecking"] = true
  ["allowExperimentalFeatures"] = false
  ["requireDocumentation"] = llmConfig != null
  ["enforceNamingConventions"] = true
  ["validateMathematicalConstraints"] = true
}

/// Security configuration
securityConfig: Mapping<String, Any> = new {
  ["sandboxExecution"] = true
  ["allowNetworkAccess"] = false
  ["allowFileWrite"] = true
  ["maxFileSize"] = "10MB"
  ["allowedFileTypes"] = List("pkl", "json", "yaml", "md")
}

/// Performance optimization settings
performanceConfig: Mapping<String, Any> = new {
  ["enableCaching"] = true
  ["cacheDirectory"] = ".pkl-cache"
  ["parallelProcessing"] = true
  ["lazyEvaluation"] = true
  ["memoryOptimization"] = true
}

/// Output configuration
output {
  renderer = new YamlRenderer {
    converters {
      [Duration] = (it) -> "\\(it.value)\\(it.unit)"
    }
  }
}
"""

        pipeline_path = self.temp_dir / "GNNPipelineConfig.pkl"
        pipeline_path.write_text(pipeline_content.strip())
        return pipeline_path

    def create_multi_format_export_config(self) -> Path:
        """Create configuration for multi-format exports."""
        export_content = """
/// Multi-Format Export Configuration
/// Demonstrates Pkl's powerful output rendering capabilities
module MultiFormatExportConfig

/// Base export configuration
abstract class BaseExportConfig {
  includeMetadata: Boolean = true
  timestamp: String = "2024-01-15T10:30:00Z"
  generatedBy: String = "GNN-Pkl Pipeline v2.0"
}

/// JSON export configuration
class JsonExportConfig extends BaseExportConfig {
  prettyPrint: Boolean = true
  indentation: Int(isBetween(2, 8)) = 2
  sortKeys: Boolean = true
  escapeNonAscii: Boolean = false
}

/// YAML export configuration  
class YamlExportConfig extends BaseExportConfig {
  flowStyle: Boolean = false
  defaultStyle: String = "plain"
  indentation: Int(isBetween(2, 8)) = 2
  blockSequenceIndent: Int = 2
}

/// XML export configuration
class XmlExportConfig extends BaseExportConfig {
  prettyPrint: Boolean = true
  encoding: String = "UTF-8"
  standalone: Boolean = true
  includeXmlDeclaration: Boolean = true
}

/// GraphML export configuration
class GraphmlExportConfig extends BaseExportConfig {
  includeNodeAttributes: Boolean = true
  includeEdgeWeights: Boolean = true
  layoutHints: Boolean = true
  compressionLevel: Int(isBetween(0, 9)) = 6
}

/// Export targets
exports: Mapping<String, BaseExportConfig> = new {
  ["json"] = new JsonExportConfig {
    prettyPrint = true
    indentation = 2
  }
  ["yaml"] = new YamlExportConfig {
    flowStyle = false
    indentation = 2
  }
  ["xml"] = new XmlExportConfig {
    prettyPrint = true
    encoding = "UTF-8"
  }
  ["graphml"] = new GraphmlExportConfig {
    includeNodeAttributes = true
    layoutHints = true
  }
}

/// Format-specific renderers
output {
  renderers {
    ["json"] = new JsonRenderer {
      indent = exports["json"].indentation
      omitNullProperties = true
    }
    ["yaml"] = new YamlRenderer {
      indent = exports["yaml"].indentation
      outputStyle = "block"
    }
    ["xml"] = new XmlRenderer {
      indent = exports["xml"].indentation
      encoding = exports["xml"].encoding
    }
  }
}
"""

        export_path = self.temp_dir / "MultiFormatExportConfig.pkl"
        export_path.write_text(export_content.strip())
        return export_path

    def evaluate_pkl_file(
        self, pkl_file: Path, output_format: str = "yaml"
    ) -> Optional[str]:
        """Evaluate a Pkl file and return the output."""
        if not self.pkl_executable:
            return None

        try:
            result = subprocess.run(
                [self.pkl_executable, "eval", "-f", output_format, str(pkl_file)],
                capture_output=True,
                text=True,
                cwd=self.temp_dir,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                print(f"❌ Error evaluating {pkl_file.name}: {result.stderr}")
                return None

        except subprocess.SubprocessError as e:
            print(f"❌ Subprocess error: {e}")
            return None

    def run_demonstration(self):
        """Run the complete Pkl-GNN demonstration."""
        print("🚀 Starting Pkl-GNN Integration Demonstration")
        print("=" * 60)

        # 1. Create base template
        print("\n📋 1. Creating Base Active Inference Template")
        base_template = self.create_base_active_inference_template()
        print(f"✅ Created: {base_template.name}")

        # 2. Create specific model
        print("\n🧠 2. Creating Visual Foraging Model")
        foraging_model = self.create_visual_foraging_model()
        print(f"✅ Created: {foraging_model.name}")

        # 3. Create pipeline configuration
        print("\n⚙️  3. Creating Pipeline Configuration")
        pipeline_config = self.create_pipeline_configuration()
        print(f"✅ Created: {pipeline_config.name}")

        # 4. Create export configuration
        print("\n📤 4. Creating Multi-Format Export Configuration")
        export_config = self.create_multi_format_export_config()
        print(f"✅ Created: {export_config.name}")

        # 5. Demonstrate evaluations if Pkl is available
        if self.pkl_executable:
            print(f"\n🔍 5. Evaluating Pkl Files (using {self.pkl_executable})")

            # Evaluate foraging model
            print("\n📊 Visual Foraging Model Output (YAML):")
            yaml_output = self.evaluate_pkl_file(foraging_model, "yaml")
            if yaml_output:
                print("```yaml")
                print(
                    yaml_output[:1000] + "..."
                    if len(yaml_output) > 1000
                    else yaml_output
                )
                print("```")

            # Evaluate foraging model as JSON
            print("\n📊 Visual Foraging Model Output (JSON):")
            json_output = self.evaluate_pkl_file(foraging_model, "json")
            if json_output:
                print("```json")
                print(
                    json_output[:1000] + "..."
                    if len(json_output) > 1000
                    else json_output
                )
                print("```")

            # Evaluate pipeline configuration
            print("\n📊 Pipeline Configuration Output:")
            pipeline_output = self.evaluate_pkl_file(pipeline_config, "yaml")
            if pipeline_output:
                print("```yaml")
                print(
                    pipeline_output[:800] + "..."
                    if len(pipeline_output) > 800
                    else pipeline_output
                )
                print("```")
        else:
            print("\n⚠️  Pkl executable not found - showing file structure only")

        # 6. Show benefits summary
        self.show_benefits_summary()

        # 7. Show file structure
        self.show_file_structure()

        print(f"\n🧹 Demo files available at: {self.temp_dir}")
        print("📚 To install Pkl: https://pkl-lang.org/main/current/pkl-cli/index.html")

    def show_benefits_summary(self):
        """Display summary of Pkl benefits for GNN."""
        print("\n🌟 Key Benefits of Pkl for GNN:")
        print("=" * 50)

        benefits = [
            "✅ Type Safety: Catch Active Inference model errors at configuration time",
            "🔄 Template Inheritance: Reusable patterns for common AI models",
            "📊 Multi-Format Output: Single source generating JSON, YAML, XML, GraphML",
            "🛡️  Validation: Mathematical constraints for stochasticity and dimensions",
            "📖 Documentation: Embedded docs and IDE support",
            "⚡ Performance: Compiled configurations with caching",
            "🔧 Late Binding: Dynamic configuration based on runtime conditions",
            "🏗️  Modularity: Importable, composable configuration modules",
            "🔒 Security: Sandboxed execution environment",
            "🎯 Scientific Reproducibility: Immutable, deterministic configurations",
        ]

        for benefit in benefits:
            print(f"  {benefit}")

    def show_file_structure(self):
        """Display the generated file structure."""
        print("\n📁 Generated File Structure:")
        print("=" * 40)

        for pkl_file in sorted(self.temp_dir.glob("*.pkl")):
            file_size = pkl_file.stat().st_size
            print(f"  📄 {pkl_file.name} ({file_size:,} bytes)")

        print(f"\n📂 Total files: {len(list(self.temp_dir.glob('*.pkl')))}")


def main():
    """Run the Pkl-GNN demonstration."""
    try:
        # Check if running from project root and use examples directory
        use_examples = Path("docs/pkl/examples").exists() or Path("docs/pkl").exists()
        demo = PklGNNDemo(use_examples_dir=use_examples)
        demo.run_demonstration()

        print("\n" + "=" * 60)
        print("🎉 Pkl-GNN Demonstration Complete!")
        print("\nNext Steps:")
        print("1. Install Pkl from https://pkl-lang.org/")
        print("2. Explore the generated .pkl files")
        print("3. Try evaluating them with: pkl eval -f yaml <file.pkl>")
        print("4. Consider integrating Pkl into the GNN pipeline")
        print("\n📖 See docs/pkl/pkl_gnn.md for detailed analysis")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
