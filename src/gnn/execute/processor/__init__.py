#!/usr/bin/env python3
"""
Execute Processor module for GNN Processing Pipeline.

This module provides execute processing capabilities for rendered implementations.
"""

import copy
import json
import logging
import os
import platform
import subprocess  # nosec B404
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from gnn.execute.data_extractors import (
    collect_execution_outputs,
)
from gnn.execute.data_extractors import (
    extract_simulation_data as _extract_simulation_data,
)
from gnn.execute.data_extractors import (
    extract_simulation_data_from_files as _extract_simulation_data_from_files,
)
from gnn.execute.detection import (
    _build_script_execution_context,
    _detect_accelerator_type,
    _resolve_render_output_dir,
    determine_script_framework,
    find_executable_scripts,
    parse_frameworks_parameter,
)
from gnn.execute.julia_env import (
    GKSWSTYPE_HEADLESS,
    GKSWSTYPE_VAR,
    _build_script_execution_command,
    _julia_project_for_framework,
    check_julia_dependencies,
)
from gnn.execute.metadata import (
    _STATUS_SEVERITY,
    _atomic_execution_json,
    _execution_detail_key,
    _execution_input_identity,
    _execution_script_identity,
    _load_render_summary_contract,
    _load_rxinfer_execution_metadata_from_script,
    _load_rxinfer_execution_metadata_sidecar,
    _merge_prior_execution_summary,
    _sha256_file,
    _slim_execution_detail,
    _summarize_collected_outputs,
    generate_execution_report,
)
from gnn.execute.security_gate import check_script_allowed
from gnn.execute.subprocess_envelope import (
    INTERNAL_ERROR,
    NEVER_STARTED,
    UNKNOWN_STATE,
    run_subprocess_envelope,
)
from gnn.execute.types import (
    _EXECUTABLE_SUFFIXES,
    ExecutionFrameworkName,
    ExecutionOutcome,
    ScriptExecutionContext,
)
from gnn.utils.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

logger = logging.getLogger(__name__)


from gnn.utils.runtime_safety.framework_availability import (
    FRAMEWORK_IMPORT_CHECK as _FRAMEWORK_IMPORT_CHECK,  # noqa: E402
)


from gnn.utils.runtime_safety.framework_availability import (
    is_framework_available as _is_framework_available_by_name,
)


from gnn.execute.processor.envelope import (
    _base_execution_envelope,
    _is_python_framework_dependency_available,
    _make_distributed_dispatch_failure_result,
    _make_local_worker_pool_failure_result,
    _make_skipped_result,
    _model_framework_from_path,
)


from gnn.execute.processor.pipeline import (
    execute_simulation_from_gnn,
    process_execute,
)


from gnn.execute.processor.single import (
    _GNN_ALLOW_MISSING_DEPS,
    _aggregate_benchmark_samples,
    _backend_version_from_runner_metadata,
    _build_execution_environment,
    _framework_for_data_helpers,
    _gnn_allow_missing_deps,
    _new_execution_result,
    _sandbox_command_prefix,
    _sandbox_mode,
    execute_single_script,
)


from gnn.execute.processor.summary import (
    _classify_execute_outcome,
    _init_execution_summary,
    _update_framework_status,
    _write_execution_summaries,
)


from gnn.execute.processor.workers import (
    _coerce_dispatch_retries,
    _coerce_execution_workers,
    _execute_script_worker,
    _run_scripts_with_local_workers,
)

