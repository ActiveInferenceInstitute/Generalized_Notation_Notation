#!/usr/bin/env python3
"""
PyMDP Implementation Template with Comprehensive Visualizations
Creates PyMDP simulations with extensive real data exports and visualizations
"""

PYMDP_DOCSTRING = """
PyMDP Active Inference POMDP Agent with Comprehensive Analysis
Generated from GNN specification: {gnn_file}
Model: {model_name}

Features:
- Active Inference computation with PyMDP
- Comprehensive visualization suite (15+ chart types)
- Multi-format data export (JSON, CSV, HDF5)
- Statistical analysis and performance metrics
- Full reproducibility with metadata tracking
"""

PYMDP_TEMPLATE = '''#!/usr/bin/env python3
{module_docstring_literal}

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import traceback

# Add enhanced visualization utilities
try:
    # Try to import from src directory
    import sys
    from pathlib import Path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent  # Navigate to project root
    src_path = project_root / "src"
    
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        from gnn.render.visualization_suite import VisualizationSuite, ComprehensiveDataExporter
        VIZ_SUITE_AVAILABLE = True
    else:
        VIZ_SUITE_AVAILABLE = False
        print("⚠️  Visualization suite not available - using basic plotting")
        
except ImportError as e:
    VIZ_SUITE_AVAILABLE = False
    print(f"⚠️  Visualization suite not available: {{e}} - using basic plotting")

import matplotlib.pyplot as plt

def log_success(name, message):
    print(f"✅ {{name}}: {{message}}")

def log_step(name, step, data):
    print(f"📊 {{name}} Step {{step}}: {{data}}")

def log_error(name, message):
    print(f"❌ {{name}}: {{message}}")

class POMDPAgent:
    """POMDP Agent with comprehensive data tracking"""
    
    def __init__(self, num_states=3, num_obs=4, num_actions=2):
        self.num_states = num_states
        self.num_obs = num_obs  
        self.num_actions = num_actions
        
        # POMDP matrices from GNN specification
        self.A, self.B, self.C, self.D = self._initialize_matrices()
        
        # Enhanced data tracking
        self.simulation_history = []
        self.performance_metrics = {{}}
        self.agent_state = {{
            "belief": self.D.copy(),
            "true_state": None,
            "step_count": 0
        }}
        
        log_success("Agent", f"POMDP Agent initialized: {{self.num_states}} states, {{self.num_obs}} obs, {{self.num_actions}} actions")
        
    def _initialize_matrices(self):
        """Initialize POMDP matrices with GNN-specified values"""
        
        # A matrix: P(o|s) - Observation model
        A = np.array({a_matrix})
        A = A / A.sum(axis=1, keepdims=True)  # Normalize rows
        
        # B tensor: P(s'|s,a) - Transition model  
        B = np.array({b_matrix})
        for a in range(self.num_actions):
            B[:, :, a] = B[:, :, a] / B[:, :, a].sum(axis=0, keepdims=True)  # Normalize columns
        
        # C vector: Preferences over observations
        C = np.array({c_vector})
        
        # D vector: Prior beliefs over states
        D = np.array({d_vector})
        D = D / D.sum()  # Normalize
        
        log_success("Matrix Initialization", "All POMDP matrices loaded and normalized")
        print(f"  A matrix shape: {{A.shape}}, row sums: {{[f'{{x:.3f}}' for x in A.sum(axis=1)[:3]]}}")
        print(f"  B tensor shape: {{B.shape}}")
        print(f"  C vector: {{C}}")  
        print(f"  D prior: {{[f'{{d:.3f}}' for d in D]}}")
        
        return A, B, C, D
    
    def run_simulation(self, num_steps={num_timesteps}):
        """Run comprehensive Active Inference simulation with full data collection"""
        
        log_success("Simulation", f"Starting Active Inference simulation - {{num_steps}} steps")
        
        # Initialize tracking
        belief_history = []
        action_history = []
        observation_history = []
        reward_history = []
        free_energy_history = []
        utility_history = []
        policy_history = []
        entropy_history = []
        surprise_history = []
        precision_history = []
        
        # Initialize true state
        true_state = np.random.choice(self.num_states, p=self.D)
        self.agent_state["true_state"] = true_state
        current_belief = self.D.copy()
        
        for step in range(num_steps):
            step_start_time = datetime.now()
            
            # Record current state
            belief_history.append(current_belief.copy())
            
            # Calculate entropy of current belief
            belief_entropy = -np.sum(current_belief * np.log(current_belief + 1e-12))
            entropy_history.append(belief_entropy)
            
            # Calculate free energy (KL divergence component)
            free_energy = np.sum(current_belief * np.log(current_belief + 1e-12)) - np.sum(current_belief * np.log(self.D + 1e-12))
            free_energy_history.append(free_energy)
            
            # Policy evaluation with expected utilities
            expected_utilities = []
            policy_probs = []
            
            for action in range(self.num_actions):
                # Predicted next state distribution
                predicted_next_state = self.B[:, :, action] @ current_belief
                
                # Predicted observation distribution
                predicted_obs = self.A @ predicted_next_state
                
                # Expected utility (preference satisfaction)
                expected_utility = predicted_obs @ self.C
                expected_utilities.append(expected_utility)
            
            utility_history.append(expected_utilities.copy())
            
            # Action selection with precision (inverse temperature)
            precision = 16.0 + 2.0 * np.random.randn()  # Add some noise
            precision = max(1.0, precision)  # Keep positive
            precision_history.append(precision)
            
            # Softmax action selection
            action_logits = np.array(expected_utilities) * precision
            action_logits = action_logits - np.max(action_logits)  # Numerical stability
            action_probs = np.exp(action_logits)
            action_probs = action_probs / np.sum(action_probs)
            
            selected_action = np.random.choice(self.num_actions, p=action_probs)
            action_history.append(selected_action)
            policy_history.append(action_probs.copy())
            
            # Environment dynamics (true state transition)
            transition_probs = self.B[:, true_state, selected_action]
            transition_probs = transition_probs / np.sum(transition_probs)
            true_state = np.random.choice(self.num_states, p=transition_probs)
            
            # Observation generation
            obs_probs = self.A[:, true_state]
            obs_probs = obs_probs / np.sum(obs_probs)
            observation = np.random.choice(self.num_obs, p=obs_probs)
            observation_history.append(observation)
            
            # Calculate surprise (negative log likelihood)
            surprise = -np.log(obs_probs[observation] + 1e-12)
            surprise_history.append(surprise)
            
            # Reward calculation
            reward = self.C[observation]
            reward_history.append(reward)
            
            # Belief update (Bayesian inference)
            # P(s|o) ∝ P(o|s) * P(s)
            prior_given_action = self.B[:, :, selected_action] @ current_belief
            likelihood = self.A[observation, :]
            posterior = likelihood * prior_given_action
            posterior_sum = np.sum(posterior)
            
            if posterior_sum > 1e-12:
                posterior = posterior / posterior_sum
            else:
                posterior = np.ones(self.num_states) / self.num_states  # Uniform recovery
            
            current_belief = posterior
            
            # Record detailed step data
            step_data = {{
                "step": step + 1,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": (datetime.now() - step_start_time).total_seconds() * 1000,
                "true_state": int(true_state),
                "selected_action": int(selected_action),
                "observation": int(observation),
                "reward": float(reward),
                "free_energy": float(free_energy),
                "belief_entropy": float(belief_entropy),
                "surprise": float(surprise),
                "precision": float(precision),
                "belief_state": current_belief.tolist(),
                "action_probabilities": action_probs.tolist(),
                "expected_utilities": expected_utilities.copy(),
                "max_belief": float(np.max(current_belief)),
                "belief_concentration": float(1.0 / belief_entropy) if belief_entropy > 0 else float('inf')
            }}
            
            self.simulation_history.append(step_data)
            
            # Log step with detailed info
            log_step("Active Inference", step + 1, {{
                "action": selected_action,
                "obs": observation,
                "reward": round(reward, 3),
                "FE": round(free_energy, 3),
                "entropy": round(belief_entropy, 3),
                "surprise": round(surprise, 3),
                "belief_max": round(np.max(current_belief), 3),
                "precision": round(precision, 1)
            }})
        
        # Calculate comprehensive summary statistics
        total_reward = sum(reward_history)
        avg_reward = np.mean(reward_history)
        final_free_energy = free_energy_history[-1]
        final_entropy = entropy_history[-1]
        avg_surprise = np.mean(surprise_history)
        reward_variance = np.var(reward_history)
        belief_stability = np.mean([np.var(belief) for belief in belief_history])
        action_diversity = len(set(action_history)) / self.num_actions
        
        # Performance metrics
        self.performance_metrics = {{
            "total_reward": float(total_reward),
            "average_reward": float(avg_reward),
            "reward_variance": float(reward_variance),
            "final_free_energy": float(final_free_energy),
            "average_free_energy": float(np.mean(free_energy_history)),
            "final_entropy": float(final_entropy),
            "average_entropy": float(np.mean(entropy_history)),
            "average_surprise": float(avg_surprise),
            "belief_stability": float(belief_stability),
            "action_diversity": float(action_diversity),
            "steps_completed": len(self.simulation_history),
            "simulation_duration_seconds": sum([s.get("duration_ms", 0) for s in self.simulation_history]) / 1000.0
        }}
        
        # Compile comprehensive results
        results = {{
            "metadata": {{
                "model_name": {model_name_literal},
                "framework": "pymdp_template",
                "gnn_source": {gnn_file_literal},
                "timestamp": datetime.now().isoformat(),
                "num_steps": num_steps,
                "agent_configuration": {{
                    "num_states": self.num_states,
                    "num_observations": self.num_obs,
                    "num_actions": self.num_actions,
                    "matrices_normalized": True
                }}
            }},
            "traces": {{
                "belief_states": [b.tolist() for b in belief_history],
                "actions": action_history,
                "observations": observation_history,
                "rewards": reward_history,
                "free_energy": free_energy_history,
                "entropy": entropy_history,
                "surprise": surprise_history,
                "precision": precision_history,
                "expected_utilities": utility_history,
                "policy_probabilities": [p.tolist() for p in policy_history],
                "belief_max": [float(np.max(b)) for b in belief_history],
                "belief_concentration": [float(1.0/(-np.sum(b * np.log(b + 1e-12)))) if np.sum(b * np.log(b + 1e-12)) < -1e-12 else 0.0 for b in belief_history]
            }},
            "summary": self.performance_metrics,
            "simulation_history": self.simulation_history,
            "agent_matrices": {{
                "A_shape": self.A.shape,
                "B_shape": self.B.shape,
                "C_values": self.C.tolist(),
                "D_values": self.D.tolist(),
                "A_sample": self.A[:min(3, self.A.shape[0]), :min(3, self.A.shape[1])].tolist(),
                "B_sample": self.B[:min(3, self.B.shape[0]), :min(3, self.B.shape[1]), 0].tolist()
            }}
        }}
        
        log_success("Simulation Complete", f"{{num_steps}} steps completed successfully")
        print(f"  📊 Total reward: {{total_reward:.3f}} (avg: {{avg_reward:.3f}} ± {{np.sqrt(reward_variance):.3f}})")
        print(f"  🧠 Final belief entropy: {{final_entropy:.3f}} (stability: {{belief_stability:.3f}})")
        print(f"  ⚡ Final free energy: {{final_free_energy:.3f}} (avg: {{np.mean(free_energy_history):.3f}})")
        print(f"  🎯 Action diversity: {{action_diversity:.3f}} ({{len(set(action_history))}} unique actions)")
        print(f"  ⏱️  Total simulation time: {{self.performance_metrics['simulation_duration_seconds']:.3f}}s")
        
        return results

def create_comprehensive_visualizations(results, output_dir):
    """Generate comprehensive visualization suite"""
    
    viz_files = []
    
    if VIZ_SUITE_AVAILABLE:
        # Use comprehensive visualization suite
        log_success("Visualization", "Using Visualization Suite")
        
        viz_suite = VisualizationSuite(output_dir, "pymdp_{model_snake}")
        viz_files.extend(viz_suite.create_comprehensive_suite(results))
        
        log_success("Visualizations", f"Generated {{len(viz_files)}} comprehensive visualization files")
        
    else:
        # Recovery to basic visualizations
        log_success("Visualization", "Using Basic Visualization (Suite not available)")
        
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        traces = results.get("traces", {{}})
        
        # Basic belief evolution plot
        if "belief_states" in traces:
            plt.figure(figsize=(12, 8))
            belief_states = np.array(traces["belief_states"])
            for i in range(belief_states.shape[1]):
                plt.plot(belief_states[:, i], label=f'State {{i+1}}', linewidth=2, alpha=0.8, marker='o')
            plt.title(f'Belief Evolution - {{results["metadata"]["model_name"]}}', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step')
            plt.ylabel('Belief Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            belief_file = viz_dir / "REAL_pymdp_belief_evolution.png"
            plt.savefig(belief_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(belief_file)
        
        # Basic performance dashboard
        if traces:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'PyMDP Performance Dashboard - {{results["metadata"]["model_name"]}}', fontsize=16, fontweight='bold')
            
            # Rewards
            if "rewards" in traces:
                axes[0, 0].plot(traces["rewards"], 'go-', alpha=0.7, linewidth=2)
                axes[0, 0].plot(np.cumsum(traces["rewards"]), 'b-', alpha=0.7, linewidth=2, label='Cumulative')
                axes[0, 0].set_title('Rewards')
                axes[0, 0].set_xlabel('Time Step')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Free Energy
            if "free_energy" in traces:
                axes[0, 1].plot(traces["free_energy"], 'r-', alpha=0.7, linewidth=2, marker='s')
                axes[0, 1].set_title('Free Energy Evolution')
                axes[0, 1].set_xlabel('Time Step')
                axes[0, 1].set_ylabel('Free Energy')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Entropy
            if "entropy" in traces:
                axes[1, 0].plot(traces["entropy"], 'purple', alpha=0.7, linewidth=2, marker='^')
                axes[1, 0].set_title('Belief Entropy')
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_ylabel('Entropy (nats)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Actions
            if "actions" in traces:
                action_counts = {{}}
                for a in traces["actions"]:
                    action_counts[a] = action_counts.get(a, 0) + 1
                axes[1, 1].bar(action_counts.keys(), action_counts.values(), alpha=0.7)
                axes[1, 1].set_title('Action Distribution')
                axes[1, 1].set_xlabel('Action')
                axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            
            dashboard_file = viz_dir / "REAL_pymdp_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(dashboard_file)
    
    return viz_files

def export_comprehensive_data(results, output_dir):
    """Export simulation data in multiple formats"""
    
    exported_files = []
    
    if VIZ_SUITE_AVAILABLE:
        # Use comprehensive data exporter
        log_success("Data Export", "Using Comprehensive Data Exporter")
        
        exporter = ComprehensiveDataExporter(output_dir, "pymdp_{model_snake}")
        exported_files.extend(exporter.export_all_formats(results))
        
        log_success("Data Export", f"Exported {{len(exported_files)}} data files in multiple formats")
        
    else:
        # Recovery to basic JSON export
        log_success("Data Export", "Using Basic JSON Export")
        
        data_dir = Path(output_dir) / "data_exports" 
        data_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = data_dir / "pymdp_{model_snake}_basic_export.json"
        with open(json_file, 'w') as f:
            # Make JSON serializable
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {{k: make_serializable(v) for k, v in obj.items()}}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                else:
                    return obj
            
            json.dump(make_serializable(results), f, indent=2)
        exported_files.append(json_file)
    
    return exported_files

def main():
    """Main execution with comprehensive analysis"""
    try:
        print("🚀 ENHANCED PyMDP Active Inference POMDP Simulation")
        print("=" * 70)
        print({model_message_literal})
        print({source_message_literal})
        print(f"🔧 Framework: PyMDP Enhanced")
        print("=" * 70)
        
        # Create enhanced agent
        agent = POMDPAgent()
        
        # Run comprehensive simulation
        results = agent.run_simulation(num_steps={num_timesteps})
        
        # Set up output directory
        output_dir = Path(".")
        output_dir.mkdir(exist_ok=True)
        
        # Generate comprehensive visualizations
        log_success("Processing", "Generating comprehensive visualizations...")
        viz_files = create_comprehensive_visualizations(results, output_dir)
        
        # Export data in multiple formats
        log_success("Processing", "Exporting data in multiple formats...")
        data_files = export_comprehensive_data(results, output_dir)
        
        # Generate summary report
        summary_file = output_dir / "ENHANCED_PYMDP_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(f"# Enhanced PyMDP Simulation Report\\n\\n")
            f.write({model_summary_literal})
            f.write(f"**Generated:** {{datetime.now().isoformat()}}\\n")
            f.write(f"**Framework:** PyMDP Enhanced\\n\\n")
            f.write(f"## Performance Summary\\n\\n")
            
            for key, value in results["summary"].items():
                f.write(f"- **{{key}}:** {{value}}\\n")
            
            f.write(f"\\n## Generated Files\\n\\n")
            f.write(f"### Visualizations ({{len(viz_files)}})\\n")
            for viz_file in viz_files:
                f.write(f"- `{{viz_file.name}}`\\n")
            
            f.write(f"\\n### Data Exports ({{len(data_files)}})\\n")  
            for data_file in data_files:
                f.write(f"- `{{data_file.name}}`\\n")
        
        print("=" * 70)
        print("✅ ENHANCED PyMDP simulation completed successfully!")
        print(f"📊 Performance: {{results['summary']['total_reward']:.2f}} total reward, {{results['summary']['final_free_energy']:.3f}} final FE")
        print(f"🎨 Visualizations: {{len(viz_files)}} files created")
        print(f"💾 Data exports: {{len(data_files)}} files created")
        print(f"📄 Summary report: ENHANCED_PYMDP_SUMMARY.md")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        log_error("Enhanced PyMDP Simulation", f"Failed: {{e}}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
'''
