#!/usr/bin/env python3
"""Single-spec target rendering for GNN specifications.

``render_gnn_spec`` dispatches one specification to the framework-specific
renderer (structural, continuous, and model-kind gating included), and
``_render_continuous_target`` routes validated continuous models.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from gnn.render.processor.parsing import (
    _internal_representation_to_mapping,
    _normalize_initial_vectors,
    _rehydrate_file_backed_parse_summary,
    _safe_output_stem,
)

if TYPE_CHECKING:
    from gnn.types import GNNInternalRepresentation


def _render_continuous_target(
    spec: Dict[str, Any],
    target: str,
    output_dir: Path,
    stem: str,
    options: Optional[Dict[str, Any]],
) -> Tuple[bool, str, List[str]]:
    """Route validated continuous models without imposing discrete matrices."""
    from importlib import import_module

    from gnn.render.pomdp_contract import ModelKind, detect_model_kinds

    kinds = detect_model_kinds(spec)
    if kinds == frozenset({ModelKind.MULTI_AGENT, ModelKind.CONTINUOUS}):
        from gnn.render.multi_agent_continuous import (
            generate_multi_agent_continuous_script,
        )

        code = generate_multi_agent_continuous_script(spec, target)
        suffix = ".jl" if target == "rxinfer" else ".py"
        output_file = output_dir / f"{stem}_{target}{suffix}"
        output_file.write_text(code, encoding="utf-8")
        return (
            True,
            f"{target} native independent continuous agents",
            [str(output_file)],
        )
    if kinds == frozenset({ModelKind.FACTORED, ModelKind.CONTINUOUS}):
        # Per-factor LGSSM path: JAX renders the factored family through the
        # per-factor generator; every other target is refused rather than
        # silently flattened to one flat LGSSM.
        if target != "jax":
            return (
                False,
                f"unsupported-factored-continuous: {target} renders the flat "
                "linear-Gaussian family only; per-factor F_f/H_f/Q_f/R_f "
                "compositions are refused rather than silently rendered flat",
                [],
            )
        from gnn.render.continuous_common import extract_factored_continuous_spec
        from gnn.render.continuous_script import generate_factored_continuous_script

        factored = extract_factored_continuous_spec(spec)
        code = generate_factored_continuous_script(factored, "jax")
        output_file = output_dir / f"{stem}_jax.py"
        with open(output_file, "w") as handle:
            handle.write(code)
        return (
            True,
            "JAX factored-continuous LGSSM (per-factor)",
            [str(output_file)],
        )

    targets = {
        "cpomdp": ("cpomdp.cpomdp_renderer", "render_gnn_to_cpomdp", "_cpomdp.py"),
        "jax": ("jax.jax_renderer", "render_gnn_to_jax", "_jax.py"),
        "numpyro": ("numpyro.numpyro_renderer", "render_gnn_to_numpyro", "_numpyro.py"),
        "pytorch": ("pytorch.pytorch_renderer", "render_gnn_to_pytorch", "_pytorch.py"),
        "rxinfer": ("rxinfer.rxinfer_renderer", "render_gnn_to_rxinfer", "_rxinfer.jl"),
        "stan": ("stan.stan_renderer", "render_gnn_to_stan", "_stan.py"),
        "ngclearn": (
            "ngclearn.ngclearn_renderer",
            "render_gnn_to_ngclearn",
            "_ngclearn.py",
        ),
    }
    if target not in targets:
        return False, f"Continuous models are unsupported for target: {target}", []
    module_name, function_name, suffix = targets[target]
    renderer = getattr(import_module(f"gnn.render.{module_name}"), function_name)
    output_file = output_dir / f"{stem}{suffix}"
    success, message, artifacts = renderer(spec, output_file, options)
    if not success:
        return False, str(message), []
    if target == "rxinfer":
        # RxInfer's third return field contains warnings, unlike other backends.
        artifacts = [str(output_file)]
        metadata = output_file.with_suffix(".metadata.json")
        if metadata.is_file():
            artifacts.append(str(metadata))
    return True, str(message), list(artifacts)


def render_gnn_spec(
    gnn_spec: "Union[GNNInternalRepresentation, Dict[str, Any]]",
    target: str,
    output_directory: Union[str, Path],
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a target language with POMDP awareness.

    Args:
        gnn_spec: GNN specification dictionary
        target: Target language/environment
        output_directory: Output directory for generated code
        options: Additional options

    Returns:
        Tuple of (success, message, output_files: List[str])
    """
    try:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generator-based renderers: target → (module, generator_name, file_suffix)
        _GENERATOR_TARGETS: dict[str, Any] = {
            "discopy": (".generators", "generate_discopy_code", "_discopy.py"),
            "bnlearn": (".bnlearn", "generate_bnlearn_code", "_bnlearn.py"),
        }

        target_lower = target.lower()
        if isinstance(gnn_spec, dict):
            gnn_spec_mapping = _rehydrate_file_backed_parse_summary(
                gnn_spec,
                native_agents=target_lower == "rxinfer",
                preserve_discrete_structure=target_lower == "thrml",
            )
            model_name = str(
                gnn_spec_mapping.get("model_name")
                or gnn_spec_mapping.get("name")
                or "model"
            )
        else:
            gnn_spec_mapping = _internal_representation_to_mapping(gnn_spec)
            model_name = str(gnn_spec_mapping["model_name"])
        files: list[Any] = []
        requested_stem = (options or {}).get("output_filename", model_name)
        output_stem = _safe_output_stem(requested_stem)

        from gnn.render.pomdp_contract import (
            ModelKind,
            detect_model_kinds,
            unsupported_composition_reason,
            unsupported_nonstationary_reason,
        )

        # A composed spec declares more than one render family (e.g. a
        # linear-Gaussian F/H/Q/R block alongside nr_agents > 1). Rendering
        # the single-winner kind would silently drop the other family, so the
        # composed set is refused with an explicit unsupported-composition
        # receipt — the same unsupported accounting structural wrappers get —
        # for every target. The factored-continuous 2-set
        # ({FACTORED, CONTINUOUS}) is the one exception: JAX renders it via
        # the per-factor factored path; every other target emits an
        # unsupported-factored-continuous refusal instead.
        kinds = detect_model_kinds(gnn_spec_mapping)
        factored_continuous = kinds == frozenset(
            {ModelKind.FACTORED, ModelKind.CONTINUOUS}
        )
        agent_continuous = kinds == frozenset(
            {ModelKind.MULTI_AGENT, ModelKind.CONTINUOUS}
        )
        if (
            ModelKind.CONTINUOUS in kinds
            and len(kinds) > 1
            and not factored_continuous
            and not agent_continuous
        ):
            return (False, unsupported_composition_reason(kinds), [])
        # A nonstationary spec declares time-indexed (B_t) or regime-switched
        # (B_regime + schedule) transitions. Only the pymdp backend executes
        # the switching semantics; every other target renders one static
        # transition tensor, so it is refused with an explicit receipt
        # instead of silently rendering static dynamics.
        if ModelKind.NONSTATIONARY in kinds and target_lower != "pymdp":
            return (False, unsupported_nonstationary_reason(kinds), [])

        if target_lower == "thrml":
            from gnn.render.thrml import render_gnn_to_thrml

            output_file = output_dir / f"{output_stem}_thrml.py"
            success, message, artifacts = render_gnn_to_thrml(
                gnn_spec_mapping, output_file, options
            )
            return (success, message, artifacts if success else [])

        if ModelKind.CONTINUOUS in kinds:
            return _render_continuous_target(
                gnn_spec_mapping, target_lower, output_dir, output_stem, options
            )

        if target_lower in {
            "pymdp",
            "rxinfer",
            "activeinference_jl",
            "pytorch",
            "numpyro",
        }:
            from gnn.render.pomdp_contract import (
                ModelKind,
                build_canonical_pomdp_spec,
                detect_model_kind,
            )

            # Structural wrapper specs (no discrete A/B/C/D[/E] and no
            # continuous F/H/Q/R parameterization) are render-only /
            # informational: never canonicalise them into a discrete POMDP
            # render that would fail on missing matrices. Graph-backed targets
            # (bnlearn, stan, discopy) render structure legitimately and are
            # not gated here.
            if detect_model_kind(gnn_spec_mapping) is ModelKind.STRUCTURAL:
                return (
                    False,
                    "structural-spec: no renderable form — the spec declares "
                    "boundary structure only (no discrete A/B/C/D[/E] and no "
                    "continuous F/H/Q/R parameterization); informational wrapper",
                    [],
                )

            from gnn.render.multi_agent_common import has_native_multi_agent_structure

            if target_lower == "rxinfer" and has_native_multi_agent_structure(
                gnn_spec_mapping
            ):
                canonical_spec = gnn_spec_mapping
            elif ModelKind.NONSTATIONARY in kinds:
                # Raw mapping passthrough: the nonstationary executor
                # consumes B_t/B_regime plus the schedule directly, and
                # build_canonical_pomdp_spec would drop the ^[ABCDE]_ keys
                # and demand a static B that does not exist.
                canonical_spec = gnn_spec_mapping
            else:
                canonical_spec = build_canonical_pomdp_spec(
                    _normalize_initial_vectors(gnn_spec_mapping)
                )
            if target_lower == "pymdp":
                from gnn.render.pymdp.pymdp_renderer import render_gnn_to_pymdp

                output_file = output_dir / f"{output_stem}_pymdp.py"
                success, msg, _warnings = render_gnn_to_pymdp(
                    canonical_spec, output_file, options
                )
                return (True, msg, [str(output_file)]) if success else (False, msg, [])
            if target_lower == "rxinfer":
                from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

                output_file = output_dir / f"{output_stem}_rxinfer.jl"
                success, msg, _warnings = render_gnn_to_rxinfer(
                    canonical_spec, output_file, options
                )
                artifacts = [str(output_file)]
                metadata_file = output_file.with_suffix(".metadata.json")
                if metadata_file.exists():
                    artifacts.append(str(metadata_file))
                return (True, msg, artifacts) if success else (False, msg, [])

            if target_lower == "activeinference_jl":
                from gnn.render.activeinference_jl.activeinference_renderer import (
                    render_gnn_to_activeinference_jl,
                )

                output_file = output_dir / f"{output_stem}_activeinference.jl"
                success, msg, artifacts = render_gnn_to_activeinference_jl(
                    canonical_spec, output_file, options
                )
                return (True, msg, artifacts) if success else (False, msg, [])

            if target_lower == "pytorch":
                from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch

                output_file = output_dir / f"{output_stem}_pytorch.py"
                success, msg, artifacts = render_gnn_to_pytorch(
                    canonical_spec, output_file, options
                )
                return (True, msg, artifacts) if success else (False, msg, [])

            from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

            output_file = output_dir / f"{output_stem}_numpyro.py"
            success, msg, artifacts = render_gnn_to_numpyro(
                canonical_spec, output_file, options
            )
            return (True, msg, artifacts) if success else (False, msg, [])

        if target_lower in _GENERATOR_TARGETS:
            gen_module_name, gen_name, suffix = _GENERATOR_TARGETS[target_lower]
            from importlib import import_module

            generate_fn = getattr(
                import_module(gen_module_name, "gnn.render"), gen_name
            )
            code = generate_fn(gnn_spec)
            output_file = output_dir / f"{output_stem}{suffix}"
            if code:
                output_file.write_text(code)
                files.append(str(output_file))

        elif target_lower == "stan":
            from gnn.render.stan import render_stan

            code = render_stan(
                list(gnn_spec_mapping.get("variables", [])),
                list(gnn_spec_mapping.get("connections", [])),
                model_name=model_name,
            )
            output_file = output_dir / f"{output_stem}_stan.stan"
            output_file.write_text(code)
            files.append(str(output_file))

        elif target_lower == "rxinfer_toml":
            # The TOML emitter is no longer wired into the processor.
            # Use target="rxinfer" for the canonical @model + infer() renderer.
            return (
                False,
                (
                    "rxinfer_toml target is no longer supported. "
                    'Use target="rxinfer" for the canonical RxInfer.jl renderer '
                    "with genuine @model + infer() code."
                ),
                [],
            )

        elif target_lower == "discopy_combined":
            try:
                from gnn.render.discopy import render_gnn_to_discopy

                output_file = output_dir / f"{output_stem}_discopy.py"
                success, msg, _warnings = render_gnn_to_discopy(
                    gnn_spec_mapping, output_file
                )
                return (True, msg, [str(output_file)]) if success else (False, msg, [])
            except ImportError:
                from gnn.render.health import get_remediation

                remediation = get_remediation("discopy")
                return (
                    False,
                    "DisCoPy renderer not available"
                    + (f". {remediation}" if remediation else ""),
                    [],
                )

        elif target_lower in ("jax", "jax_pomdp"):
            try:
                from gnn.render.jax.jax_renderer import (
                    render_gnn_to_jax,
                    render_gnn_to_jax_pomdp,
                )
                from gnn.render.pomdp_contract import (
                    ModelKind,
                    build_canonical_pomdp_spec,
                    detect_model_kind,
                )

                if detect_model_kind(gnn_spec_mapping) is ModelKind.STRUCTURAL:
                    return (
                        False,
                        "structural-spec: no renderable form — the spec declares "
                        "boundary structure only (no discrete A/B/C/D[/E] and no "
                        "continuous F/H/Q/R parameterization); informational wrapper",
                        [],
                    )

                output_file = output_dir / f"{output_stem}_jax.py"
                canonical_spec = build_canonical_pomdp_spec(
                    _normalize_initial_vectors(gnn_spec_mapping)
                )
                render_fn = (
                    render_gnn_to_jax_pomdp
                    if target_lower == "jax_pomdp"
                    else render_gnn_to_jax
                )
                success, msg, art = render_fn(canonical_spec, output_file, options)
                return (True, msg, art) if success else (False, msg, [])
            except ImportError:
                from gnn.render.health import get_remediation

                remediation = get_remediation("jax")
                return (
                    False,
                    "JAX renderer not available"
                    + (f". {remediation}" if remediation else ""),
                    [],
                )

        else:
            from gnn.render.framework_registry import FRAMEWORK_REGISTRY

            known = sorted(set(FRAMEWORK_REGISTRY) | {"jax_pomdp", "discopy_combined"})
            return (
                False,
                f"Unsupported target: {target}. Known targets: {', '.join(known)}",
                [],
            )

        if files:
            return True, f"Successfully generated {target} code", files
        else:
            return False, f"Failed to generate {target} code", []

    except Exception as e:
        return False, f"Error rendering {target}: {type(e).__name__}: {e}", []
