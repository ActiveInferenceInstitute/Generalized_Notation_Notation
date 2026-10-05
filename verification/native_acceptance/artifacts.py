"""Validate fresh reported native traces and retain supplementary public-API HTML.

This worker is invoked under a supervised shared-deadline process envelope. It
never edits pipeline/native results, supplies probabilities, or re-executes the
model. Supplemental HTML has its own artifact custody, outside finalized output.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite_array(value, name, dimensions):
    import numpy as np
    def numeric(item):
        if isinstance(item, list):
            return bool(item) and all(numeric(child) for child in item)
        return isinstance(item, (float, int)) and not isinstance(item, bool) and math.isfinite(item)
    require(numeric(value), name + " requires finite numeric, non-Boolean values")
    array = np.asarray(value, dtype=float)
    require(array.ndim == dimensions and all(array.shape), name + " has wrong dimensions")
    return array


def strict_views(data, timesteps, backend):
    import numpy as np
    from gnn.analysis.result_adapter import result_views
    require(isinstance(data, dict), "Native payload must be a mapping")
    schemas = {"numpyro": {"numpyro_simulation_v1"}, "rxinfer": {"rxinfer_simulation_v1", "rxinfer_stigmergic_swarm_v1"}}
    require(data.get("schema_version") in schemas[backend], "Unexpected native schema")
    # These eight reported native schemas are categorical. Contradictory
    # Gaussian declarations cannot turn simplex-shaped means into acceptance.
    require(not any(key in data for key in ("posterior_cov", "posterior_cov_by_agent", "posterior_cov_by_factor")), "Gaussian covariance conflicts with categorical reported-case scope")
    for metadata in (data, data.get("runtime_metadata", {}), data.get("model_parameters", {})):
        require(isinstance(metadata, dict), "Malformed native semantic metadata")
        declaration = metadata.get("model_kind")
        if declaration is not None:
            require(isinstance(declaration, str) and declaration in {"flat", "discrete", "multi_agent", "factored", "hierarchical", "learning"}, "Unsupported/continuous family conflicts with categorical reported-case scope")
    declared = data.get("num_timesteps")
    require(type(declared) is int and declared == timesteps, "Full authored timesteps required")
    validation = data.get("validation")
    require(isinstance(validation, dict) and validation.get("all_valid") is True, "Native validation is not successful")
    # Validate original JSON types before the shared adapter's float conversion.
    # Boolean/string probabilities must not acquire numeric meaning in that seam.
    for raw_key in ('beliefs', 'beliefs_by_agent', 'beliefs_by_factor'):
        raw = data.get(raw_key)
        if isinstance(raw, dict):
            require(bool(raw) and all(isinstance(name, str) and name for name in raw), 'Invalid raw belief identities')
            for name, values in raw.items():
                finite_array(values, raw_key + '/' + name, 2)
        elif raw is not None:
            finite_array(raw, raw_key, 2)
    for raw_key in ('actions', 'observations', 'actions_by_agent', 'observations_by_agent', 'actions_by_factor', 'observations_by_factor'):
        raw = data.get(raw_key)
        if isinstance(raw, dict):
            for name, values in raw.items():
                finite_array(values, raw_key + '/' + str(name), 1)
        elif raw is not None:
            finite_array(raw, raw_key, 1)
    views = result_views(data)
    require(bool(views), "No native belief views")
    shapes = {}
    for name, view in views.items():
        beliefs = finite_array(view.get("beliefs"), name + ":beliefs", 2)
        require(beliefs.shape[0] == timesteps, "Posterior length differs from authored T")
        require(np.all((beliefs >= 0) & (beliefs <= 1)), "Posterior probabilities out of range")
        require(np.allclose(beliefs.sum(axis=1), 1, rtol=1e-6, atol=1e-8), "Posterior mass invalid; no normalization permitted")
        for key in ("observations", "actions"):
            values = finite_array(view.get(key), name + ":" + key, 1)
            require(len(values) == timesteps and np.equal(values, np.floor(values)).all(), key + " must be aligned categorical indices")
            require((values >= 0).all(), key + " must be nonnegative")
        if backend == "rxinfer":
            vfe = view.get("vfe_per_iteration", view.get("variational_free_energy"))
            if vfe is not None and len(vfe):
                finite_array(vfe, name + ":iteration free energy", 1)
        shapes[name] = list(beliefs.shape)
    if backend == "rxinfer":
        runtime = data.get("runtime_metadata")
        require(isinstance(runtime, dict), "Missing native Julia dependency metadata")
        require(str(runtime.get("julia_version", "")).removeprefix("v") == "1.12.7", "Unexpected native Julia version")
        require(str(runtime.get("rxinfer_version", "")).removeprefix("v") == "5.5.0", "Unexpected native RxInfer version")
        parameters = data.get("model_parameters")
        require(isinstance(parameters, dict) and type(parameters.get("inference_iterations")) is int and parameters["inference_iterations"] == 20, "Full renderer-default20 inference iterations required")
        agents = data.get("agents")
        if isinstance(agents, list) and agents:
            require(set(agents) == set(views), "Agent identities not preserved")
    return views, shapes


def emitted_binding(data, script_path, views, backend):
    """Bind native labels and categorical indices to actual emitted semantics."""
    import numpy as np
    script = script_path.read_text()
    names = list(views)
    bounds = {}
    if backend == "numpyro":
        tree = ast.parse(script)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_simulation"]
        require(len(functions) == 1, "Unexpected NumPyro entrypoint")
        assignments = {node.targets[0].id: node.value for node in functions[0].body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        output = assignments.get("results")
        require(isinstance(output, ast.Dict), "Missing emitted results dictionary")
        fields = {ast.literal_eval(key): value for key, value in zip(output.keys, output.values)}
        label = ast.literal_eval(fields["model_name"])
        states, observations, actions = (ast.literal_eval(assignments[key]) for key in ("num_states", "num_obs", "num_actions"))
        require(len(views) == 1 and data.get("num_states") == states and data.get("num_observations") == observations and data.get("num_actions") == actions, "NumPyro emitted/native cardinalities differ")
        bounds[names[0]] = (observations, actions, states)
    else:
        def const(name):
            matches = re.findall(r"^const\s+" + name + r"\s*=\s*(.+)$", script, flags=re.MULTILINE)
            require(len(matches) == 1, "Missing/duplicate emitted constant:" + name)
            return json.loads(matches[0])
        label = const("MODEL_NAME")
        require(const("SCHEMA_VERSION") == data["schema_version"], "Emitted/native schema differs")
        if "beliefs_by_agent" in data:
            agents = const("AGENTS")
            require(agents == data.get("agents") and set(agents) == set(views) and const("NUM_AGENTS") == len(agents), "Emitted/native agent identities differ")
            arrays = {key: const(key) for key in ("AGENT_AS", "AGENT_BS", "AGENT_DS")}
            require(all(len(values) == len(agents) for values in arrays.values()), "Emitted agent tensor group count differs")
            for index, name in enumerate(agents):
                A = finite_array(arrays["AGENT_AS"][index], "emitted agent likelihood", 2)
                B = finite_array(arrays["AGENT_BS"][index], "emitted agent transitions", 3)
                D = finite_array(arrays["AGENT_DS"][index], "emitted agent prior", 1)
                require(A.shape[1] == len(D) and B.shape[:2] == (len(D), len(D)), "Emitted agent tensor dimensions differ")
                require((A >= 0).all() and (B >= 0).all() and (D >= 0).all(), "Invalid emitted agent probabilities")
                require(np.allclose(A.sum(axis=0), 1, rtol=1e-12, atol=1e-12) and np.allclose(B.sum(axis=0), 1, rtol=1e-12, atol=1e-12) and np.isclose(D.sum(), 1, rtol=1e-12, atol=1e-12), "Invalid emitted agent probability mass")
                bounds[name] = (A.shape[0], B.shape[2], len(D))
        else:
            require(len(views) == 1, "Unexpected flat named view set")
            bounds[names[0]] = (const("NUM_OBSERVATIONS"), const("NUM_ACTIONS"), const("NUM_STATES"))
    require(isinstance(label, str) and label == data.get("model_name"), "Emitted/native model display labels differ")
    for name, (observations, actions, states) in bounds.items():
        require(all(type(value) is int and value > 0 for value in (observations, actions, states)), "Invalid emitted categorical cardinality")
        view = views[name]
        require(np.asarray(view["beliefs"]).shape[1] == states, "Source/posterior state width differs")
        for key, upper in (("observations", observations), ("actions", actions)):
            values = finite_array(view[key], name + ":" + key, 1)
            require(np.equal(values, np.floor(values)).all() and ((values >= 0) & (values < upper)).all(), "Native categorical index outside emitted bounds:" + key)
    return {"native_display_name": label, "generated_native_labels_equal": True, "categorical_bounds_by_view": {name: {"observations": limits[0], "actions": limits[1], "states": limits[2]} for name, limits in bounds.items()}, "identity_claim": "The generated/native display label is checked separately from the current selected model_id/source SHA/script binding. Generic NumPyro labels are not authored-name preservation."}


def numpyro_literals(script, timesteps, data):
    import numpy as np
    tree = ast.parse(script.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_simulation"]
    require(len(functions) == 1, "Unexpected generated NumPyro entry function")
    nodes = {node.targets[0].id: node.value for node in functions[0].body
             if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in {"T", "num_states", "num_obs", "num_actions", "D", "A"}}
    require(ast.literal_eval(nodes["T"]) == timesteps, "Generated script silently changed T")
    prior_node = nodes["D"]
    require(isinstance(prior_node, ast.Call) and len(prior_node.args) == 1
            and isinstance(prior_node.func, ast.Attribute) and isinstance(prior_node.func.value, ast.Name)
            and prior_node.func.value.id == "jnp" and prior_node.func.attr == "array", "Prior must be an emitted literal, not a repair")
    prior = finite_array(ast.literal_eval(prior_node.args[0]), "emitted D", 1)
    require(np.all((prior >= 0) & (prior <= 1)) and np.isclose(prior.sum(), 1, rtol=1e-12, atol=1e-12), "Emitted prior is not a float64 simplex")
    require(len(prior) == ast.literal_eval(nodes["num_states"]) == data["num_states"], "Prior/native state dimensions differ")
    # Explicit distribution checks are supplementary; they do not alter the
    # original script or its pipeline execution dtype/configuration.
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpyro.distributions as dist
    witnesses = []
    for dtype in (jnp.float32, jnp.float64):
        emitted = jnp.asarray(prior, dtype=dtype)
        distribution = dist.Categorical(probs=emitted, validate_args=True)
        sample = distribution.sample(jax.random.PRNGKey(42))
        require(0 <= int(sample) < len(prior), "Invalid native Categorical draw")
        witnesses.append({"dtype": str(emitted.dtype), "mass": float(emitted.sum()), "distribution_constructor_succeeded": True, "sample": int(sample)})
    return {"emitted_prior_length": len(prior), "emitted_prior_mass_float64": float(prior.sum()), "emitted_prior_minimum": float(prior.min()), "dtype_distribution_witnesses": witnesses, "normalization_applied": False, "claim": "Emitted literal simplex/distribution construction and full native pipeline execution; no universal inference equivalence claim."}


def precision_boundaries():
    """Independently decode emitted values and witness unchanged malformed refusal."""
    import numpy as np
    from gnn.render.spec_matrices import format_array_literal
    from gnn.render.pomdp_contract import normalise_vector, normalise_matrix_columns
    shapes = [(), (3,), (2, 3), (2, 2, 3), (2, 2, 2, 3)]
    round_trips = []
    for shape in shapes:
        values = np.resize([1 / 3, 1e-15, 1 - 1e-15], max(1, math.prod(shape))).reshape(shape)
        literal = format_array_literal(values, prefix="array")
        node = ast.parse(literal, mode="eval").body
        require(isinstance(node, ast.Call) and len(node.args) == 1, "Unexpected literal expression")
        decoded = np.asarray(ast.literal_eval(node.args[0]), dtype=np.float64)
        require(decoded.shape == values.shape and np.array_equal(decoded.view(np.uint64), values.view(np.uint64)), "Tiny/rank numeric values changed during emission")
        round_trips.append({"shape": list(shape), "bitwise_float64_equal": True, "minimum": float(values.min())})
    refusals = []
    for name, values in (("negative", [-0.1, 1.1]), ("nan", [float("nan"), 1]), ("infinite", [float("inf"), 1]), ("zero_mass", [0, 0]), ("nonunit_mass", [0.4, 0.4])):
        original = np.asarray(values, dtype=float).copy()
        try:
            normalise_vector(values, name="D")
        except ValueError:
            require(np.array_equal(np.asarray(values, dtype=float), original, equal_nan=True), "Malformed rejection changed source values")
            refusals.append(name)
        else:
            raise AssertionError("Malformed probability silently accepted:" + name)
    try:
        normalise_matrix_columns([[0.9, 0.1], [0.2, 0.8]], name="A")
    except ValueError:
        refusals.append("nonunit_likelihood_columns")
    else:
        raise AssertionError("Malformed likelihood silently accepted")
    return {"round_trips": round_trips, "malformed_refusals": refusals, "no_source_repair": True, "execution_plane": "Fresh installed public formatter/validator calls with independently decoded literal values; separate from native authored-case execution."}


class HTMLInventory(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scripts = []
        self.iframes = []
        self.body = None
        self.has_html = False
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.has_html |= tag.lower() == "html"
        if tag.lower() == "script" and "src" not in values:
            self.body = []
        if tag.lower() == "iframe":
            self.iframes.append(values)
    def handle_data(self, value):
        if self.body is not None:
            self.body.append(value)
    def handle_endtag(self, tag):
        if tag.lower() == "script" and self.body is not None:
            self.scripts.append("".join(self.body))
            self.body = None


def html_witness(path, views, timesteps):
    inventory = HTMLInventory()
    inventory.feed(path.read_text())
    require(inventory.has_html and inventory.body is None, "Malformed/unclosed native HTML")
    panels = {}
    if inventory.iframes:
        require(len(inventory.iframes) == len(views), "HTML dropped named native views")
        for iframe in inventory.iframes:
            require(iframe.get("title") in views and "srcdoc" in iframe, "HTML changed view identities")
            require(iframe["title"] not in panels, "Duplicate HTML view")
            parser = HTMLInventory()
            parser.feed(iframe["srcdoc"])
            require(parser.has_html and parser.body is None, "Malformed view HTML")
            panels[iframe["title"]] = parser.scripts
    else:
        require(len(views) == 1, "Missing per-view HTML panels")
        panels[next(iter(views))] = inventory.scripts
    for name, scripts in panels.items():
        candidates = []
        for body in scripts:
            match = re.search(r"\bconst\s+DATA\s*=\s*", body)
            if match:
                value, _ = json.JSONDecoder().raw_decode(body[match.end():])
                candidates.append(value)
        require(len(candidates) == 1, "Missing/duplicate native HTML trace payload")
        value = candidates[0]
        require(value["n_steps"] == timesteps and value["beliefs"] == views[name]["beliefs"], "HTML belief trace differs from native posterior")
        for key in ("actions", "observations"):
            require(value[key] == views[name][key], "HTML inputs/actions differ from native trace")
    return {"named_panels": sorted(panels), "full_trace_values_match_native": True, "browser_javascript_execution_claimed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpyro", "rxinfer"), required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-relative-path", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    args.supplemental.mkdir(parents=True, exist_ok=False)
    receipt = {"accepted": False, "backend": args.backend, "native_result_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest()}
    try:
        require(not args.native.is_symlink() and not args.script.is_symlink(), "Native/script symlink refused")
        require(args.source.is_file() and not args.source.is_symlink() and hashlib.sha256(args.source.read_bytes()).hexdigest() == args.source_sha256, "Selected case source SHA differs")
        relative = args.source_relative_path
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
                and relative == Path(relative).as_posix()
                and args.source.absolute() == args.input_root.absolute() / relative,
                "Selected source relative path/input root differs")
        independent_id = re.sub(r'[^A-Za-z0-9_-]', '_', Path(relative).stem) + '-' + hashlib.sha256(relative.encode()).hexdigest()[:12]
        require(args.model_id == independent_id, "Selected case model ID differs from path-derived identity")
        receipt.update(selected_model_id=args.model_id, selected_source_sha256=args.source_sha256, selected_source_relative_path=args.source_relative_path, acceptance_family="Categorical-only eight reported cases; no Gaussian native acceptance in this worker.")
        data = json.loads(args.native.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON:" + value)))
        views, shapes = strict_views(data, args.timesteps, args.backend)
        receipt["emitted_native_binding"] = emitted_binding(data, args.script, views, args.backend)
        receipt.update(view_shapes=shapes, declared_timesteps=args.timesteps, validation=data["validation"], runtime_metadata=data.get("runtime_metadata"), native_model_parameters=data.get("model_parameters"), supplemental_scope="Public API HTML generated from exact native bytes after finalized pipeline; separate artifact inventory, no pipeline manifest mutation.")
        from PIL import Image
        pngs = sorted(args.analysis.rglob("*.png"))
        require(bool(pngs), "No analysis PNGs")
        decoded = {}
        for path in pngs:
            require(not path.is_symlink(), "Analysis PNG symlink refused")
            with Image.open(path) as picture:
                require(picture.format == "PNG" and min(picture.size) > 0, "Not a readable PNG")
                picture.load()
                decoded[str(path.relative_to(args.analysis))] = {"dimensions": list(picture.size), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        receipt["readable_pngs"] = decoded
        if args.backend == "numpyro":
            receipt["probability_precision"] = numpyro_literals(args.script, args.timesteps, data)
            receipt["precision_boundaries"] = precision_boundaries()
        else:
            script = args.script.read_text()
            for name, expected in (("TIME_STEPS", args.timesteps), ("INFERENCE_ITERATIONS", 20)):
                matches = re.findall(r"^const\s+" + name + r"\s*=\s*(\d+)\s*$", script, flags=re.MULTILINE)
                require(matches == [str(expected)], "Generated Julia T/iterations changed")
            require("using RxInfer" in script and "infer(" in script, "Missing native RxInfer execution")
            schema_matches = re.findall(r'^const\s+SCHEMA_VERSION\s*=\s*("[^"\n]+")\s*$', script, flags=re.MULTILINE)
            require(len(schema_matches) == 1 and json.loads(schema_matches[0]) == data["schema_version"], "Native/generated Julia schema differs")
            agent_matches = re.findall(r"^const\s+AGENTS\s*=\s*(.+)$", script, flags=re.MULTILINE)
            if agent_matches:
                require(len(agent_matches) == 1, "Duplicate declared agent array")
                names = json.loads(agent_matches[0])
                require(names == data.get("agents") and set(names) == set(views), "Generated/native agent identities differ")
                source_priors = re.findall(r"^const\s+AGENT_DS\s*=\s*(.+)$", script, flags=re.MULTILINE)
                require(len(source_priors) == 1, "Missing emitted per-agent priors")
                priors = json.loads(source_priors[0])
                require(len(priors) == len(names), "Agent source cardinalities differ")
                import numpy as np
                for name, values in zip(names, priors):
                    prior = finite_array(values, "emitted agent prior", 1)
                    require(len(prior) == shapes[name][1] and (prior >= 0).all() and np.isclose(prior.sum(), 1, rtol=1e-12, atol=1e-12), "Agent source/posterior dimension or mass differs")
            else:
                widths = re.findall(r"^const\s+NUM_STATES\s*=\s*(\d+)\s*$", script, flags=re.MULTILINE)
                require(len(widths) == 1 and len(views) == 1 and next(iter(shapes.values()))[1] == int(widths[0]), "Flat source/posterior dimensions differ")
            runtime_digest = (data.get("runtime_metadata") or {}).get("script_sha256")
            if runtime_digest is not None:
                require(runtime_digest == receipt["script_sha256"], "Native runtime script hash differs")
            receipt["runtime_script_hash_present"] = runtime_digest is not None
            gifs = sorted(path for path in args.analysis.rglob("*_rxinfer_animation.gif"))
            require(len(gifs) == 1, "Missing or duplicate per-model RxInfer pipeline GIF")
            gif = gifs[0]
            require(not gif.is_symlink(), "Analysis GIF symlink refused")
            with Image.open(gif) as picture:
                require(picture.format == "GIF" and picture.n_frames == args.timesteps, "GIF does not preserve all native timestep frames")
                for index in range(picture.n_frames):
                    picture.seek(index)
                    picture.load()
                receipt["pipeline_gif"] = {"sha256": hashlib.sha256(gif.read_bytes()).hexdigest(), "decoded_frames": picture.n_frames, "dimensions": list(picture.size)}
            from gnn.analysis.rxinfer import generate_animated_html
            html = args.supplemental / "native_animated.html"
            require(generate_animated_html(data, html, model_name=data["model_name"]) == str(html), "Public HTML API did not produce artifact")
            receipt["supplemental_html"] = {"sha256": hashlib.sha256(html.read_bytes()).hexdigest(), "bytes": html.stat().st_size, **html_witness(html, views, args.timesteps)}
        require(hashlib.sha256(args.native.read_bytes()).hexdigest() == receipt["native_result_sha256"], "Validator changed native result bytes")
        receipt["accepted"] = True
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    (args.supplemental / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"accepted": receipt["accepted"], "error": receipt.get("error")}, sort_keys=True))
    return 0 if receipt["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
