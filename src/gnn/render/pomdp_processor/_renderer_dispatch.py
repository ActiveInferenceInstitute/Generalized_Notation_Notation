#!/usr/bin/env python3
"""Renderer dispatch mixin for the POMDP render processor."""

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, cast

from gnn.render.admission import validate_model_render_options, validate_render_options

from ._routes import RENDERER_ROUTES, RendererRoute, _safe_output_stem
from ._support import _POMDPProcessorSupportMixin

if TYPE_CHECKING:
    pass


class _RendererDispatchMixin(_POMDPProcessorSupportMixin):
    def _call_framework_renderer(
        self, framework: str, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Call the appropriate framework renderer.

        Args:
            framework: Target framework name
            gnn_spec: GNN specification
            output_dir: Output directory for this framework
            **kwargs: Additional renderer options

        Returns:
            Renderer result dictionary
        """
        if framework == "bnlearn":
            return self._call_bnlearn_renderer(gnn_spec, output_dir, **kwargs)
        route = RENDERER_ROUTES.get(framework)
        if route is None:
            return {
                "success": False,
                "message": f"No renderer implemented for {framework}",
                "artifacts": [],
            }
        return self._invoke_renderer(route, framework, gnn_spec, output_dir, **kwargs)

    def _invoke_renderer(
        self,
        route: RendererRoute,
        framework: str,
        gnn_spec: Dict[str, Any],
        output_dir: Path,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Shared renderer invocation skeleton driven by ``route``."""
        try:
            module = importlib.import_module(route.module, "gnn.render")
            render_fn = getattr(module, route.function)

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{_safe_output_stem(model_name)}{route.suffix}"

            warnings: list[Any] = []
            if route.validate:
                # Validate state spaces are present before rendering
                validation_result = self._validate_state_spaces_in_spec(
                    gnn_spec, framework
                )
                if not validation_result["valid"]:
                    warnings = validation_result.get("warnings", [])
                    if validation_result.get("critical", False):
                        return {
                            "success": False,
                            "message": f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                            "artifacts": [],
                            "warnings": warnings,
                        }

            configured_options = kwargs.get("backend_options", {})
            if not isinstance(configured_options, dict):
                raise ValueError("backend_options must be a mapping")
            selected_options = configured_options.get(framework, {})
            if not isinstance(selected_options, dict):
                raise ValueError(f"backend_options.{framework} must be a mapping")
            selected_options = validate_render_options(framework, selected_options)
            validate_model_render_options(framework, selected_options, gnn_spec)
            resolved_kwargs = (
                dict(kwargs) if framework == "thrml" else {**kwargs, **selected_options}
            )

            if route.options_mode == "timesteps":
                # Build options dict with timesteps if available
                options: dict[Any, Any] = {}
                model_params = gnn_spec.get("model_parameters", {})
                if "num_timesteps" in model_params:
                    options["num_timesteps"] = model_params["num_timesteps"]
                renderer_options: Any = {**options, **resolved_kwargs} or None
            elif route.options_mode == "kwargs_or_none":
                renderer_options = resolved_kwargs or None
            else:
                renderer_options = resolved_kwargs

            result = render_fn(gnn_spec, output_file, renderer_options)

            success, message, payload = cast("tuple[bool, str, Any]", result)
            warnings = list(payload) if route.result_mode == "warnings" else []
            artifacts: list[str] = (
                list(payload) if route.result_mode == "artifacts" else []
            )

            # Post-render validation: verify state spaces are in generated script
            if route.validate and success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(
                    output_file, gnn_spec
                )
                if not post_validation["valid"]:
                    warnings.extend(post_validation.get("warnings", []))

            if route.result_mode == "warnings":
                if route.artifacts_mode == "primary_plus_metadata":
                    artifacts = [str(output_file)] if success else []
                    metadata_file = output_file.with_suffix(".metadata.json")
                    if success and metadata_file.exists():
                        artifacts.append(str(metadata_file))
                else:  # primary
                    artifacts = [str(output_file)] if success else []

            return {
                "success": success,
                "unsupported": not success
                and str(message).startswith("unsupported-thrml:"),
                "status": "success"
                if success
                else "unsupported"
                if str(message).startswith("unsupported-thrml:")
                else "failed",
                "message": message,
                "artifacts": artifacts,
                "warnings": warnings,
            }

        except ImportError:
            return {
                "success": False,
                "message": f"{route.label} renderer not available",
                "artifacts": [],
            }

    def _call_pymdp_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call PyMDP renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["pymdp"], "pymdp", gnn_spec, output_dir, **kwargs
        )

    def _call_rxinfer_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call RxInfer renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["rxinfer"], "rxinfer", gnn_spec, output_dir, **kwargs
        )

    def _call_activeinference_jl_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call ActiveInference.jl renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["activeinference_jl"],
            "activeinference_jl",
            gnn_spec,
            output_dir,
            **kwargs,
        )

    def _call_jax_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call JAX renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["jax"], "jax", gnn_spec, output_dir, **kwargs
        )

    def _call_discopy_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call DisCoPy renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["discopy"], "discopy", gnn_spec, output_dir, **kwargs
        )

    def _call_bnlearn_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call bnlearn renderer."""
        try:
            from gnn.render.bnlearn import generate_bnlearn_code

            model_name = gnn_spec.get("name", "pomdp_model")
            output_file = output_dir / f"{_safe_output_stem(model_name)}_bnlearn.py"

            code = generate_bnlearn_code(gnn_spec, output_file)
            success = bool(code)

            return {
                "success": success,
                "message": "bnlearn code generated"
                if success
                else "Failed to generate bnlearn code",
                "artifacts": [str(output_file)] if success else [],
                "warnings": [],
            }
        except ImportError as e:
            return {
                "success": False,
                "message": f"bnlearn generator not available: {e}",
                "artifacts": [],
            }

    def _call_pytorch_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call PyTorch renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["pytorch"], "pytorch", gnn_spec, output_dir, **kwargs
        )

    def _call_numpyro_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call NumPyro renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["numpyro"], "numpyro", gnn_spec, output_dir, **kwargs
        )

    def _call_stan_renderer(
        self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs: Any
    ) -> Dict[str, Any]:
        """Call Stan renderer."""
        return self._invoke_renderer(
            RENDERER_ROUTES["stan"], "stan", gnn_spec, output_dir, **kwargs
        )
