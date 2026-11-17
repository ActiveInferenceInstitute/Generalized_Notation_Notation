"""
Visualization functions for post-simulation analysis.

Provides plot_belief_evolution, animate_belief_evolution, visualize_all_framework_outputs,
generate_belief_heatmaps, generate_action_analysis, generate_free_energy_plots,
generate_observation_analysis, generate_unified_framework_dashboard,
and generate_cross_framework_comparison.

Extracted from post_simulation.py for maintainability.

Mechanical split facade: implementations live in ``viz_*`` sibling
modules; ``__all__`` below is the explicit re-export surface.
"""

from .viz_animations import (
    _gridworld_animation_items,
    animate_belief_evolution,
    animate_cross_framework_gridworld_trajectories,
    animate_gridworld_trajectory,
    generate_gridworld_animation_suite,
    plot_belief_evolution,
)
from .viz_dashboard import (
    generate_confidence_comparison,
    generate_cross_framework_comparison,
    generate_efe_convergence_comparison,
    generate_framework_radar,
    generate_unified_framework_dashboard,
)
from .viz_manifest import (
    _relative_or_absolute,
    write_gridworld_analysis_manifest,
)
from .viz_plots import (
    generate_action_analysis,
    generate_belief_heatmaps,
    generate_free_energy_plots,
    generate_observation_analysis,
    generate_vfe_vs_efe_plot,
    visualize_all_framework_outputs,
)
from .viz_schema import (
    CURRENT_VISUALIZATION_SCHEMAS,
    VISUALIZATION_FRAMEWORK_DIRS,
    _belief_map_states,
    _current_schema_visualization_data,
    _framework_from_path_or_payload,
    _grid_side_for_states,
    _gridworld_state_sequence,
    _is_gridworld_payload,
    _model_name_from_path,
    _normalize_framework_name,
    _series_from_payload,
    _state_count_from_payload,
)

# Explicit re-export surface (no_implicit_reexport).
__all__ = [
    "_gridworld_animation_items",
    "animate_belief_evolution",
    "animate_cross_framework_gridworld_trajectories",
    "animate_gridworld_trajectory",
    "generate_gridworld_animation_suite",
    "plot_belief_evolution",
    "generate_confidence_comparison",
    "generate_cross_framework_comparison",
    "generate_efe_convergence_comparison",
    "generate_framework_radar",
    "generate_unified_framework_dashboard",
    "_relative_or_absolute",
    "write_gridworld_analysis_manifest",
    "generate_action_analysis",
    "generate_belief_heatmaps",
    "generate_free_energy_plots",
    "generate_observation_analysis",
    "generate_vfe_vs_efe_plot",
    "visualize_all_framework_outputs",
    "CURRENT_VISUALIZATION_SCHEMAS",
    "VISUALIZATION_FRAMEWORK_DIRS",
    "_belief_map_states",
    "_current_schema_visualization_data",
    "_framework_from_path_or_payload",
    "_grid_side_for_states",
    "_gridworld_state_sequence",
    "_is_gridworld_payload",
    "_model_name_from_path",
    "_normalize_framework_name",
    "_series_from_payload",
    "_state_count_from_payload",
]
