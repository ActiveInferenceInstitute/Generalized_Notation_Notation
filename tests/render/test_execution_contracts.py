"""Independent numerical and real native execution witnesses for semantic contracts."""

from copy import deepcopy
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.execution_contracts import contract_payload
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.processor import render_gnn_spec

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    "hierarchical/hierarchical_pomdp",
    "hierarchical/temporal_hierarchy",
    "discrete/tmaze_epistemic",
]


def spec(index):
    space = extract_pomdp_from_file(
        ROOT / "input/gnn_files" / (MODELS[index] + ".md"), on_error="raise"
    )
    return pomdp_to_gnn_spec(space)


def native_payload(index):
    payload = contract_payload(spec(index))
    matrices = {
        k: jnp.asarray(v, dtype=jnp.float64) for k, v in payload["matrices"].items()
    }
    return payload, matrices


@pytest.fixture(autouse=True)
def precision():
    with (
        jax.enable_x64(True)
        if hasattr(jax, "enable_x64")
        else jax.experimental.enable_x64()
    ):
        yield


@pytest.mark.parametrize("index", range(3))
def test_dispatch_preserves_structure_and_refuses_other_backends(index, tmp_path):
    space = extract_pomdp_from_file(
        ROOT / "input/gnn_files" / (MODELS[index] + ".md"), on_error="raise"
    )
    s = pomdp_to_gnn_spec(space)
    assert s["canonical_pomdp_schema"] == "raw_discrete_components_v1"
    assert "A" not in s["initialparameterization"]
    s["name"] = s["model_name"] = "unrelated_renamed_spec"
    assert render_gnn_spec(s, "jax", tmp_path / "jax")[0]
    processor = POMDPRenderProcessor(tmp_path)
    for backend in processor.framework_configs:
        if backend == "jax":
            continue
        receipt = processor._process_single_framework(space, backend)
        assert receipt["status"] == "unsupported", (backend, receipt)
        ok, reason, files = render_gnn_spec(s, backend, tmp_path / backend)
        assert (
            not ok
            and not files
            and reason.startswith("unsupported-execution-contract:")
        )


def test_missing_contract_cannot_flatten_mixed_actions():
    space = extract_pomdp_from_file(
        ROOT / "input/gnn_files" / (MODELS[1] + ".md"), on_error="raise"
    )
    space.model_parameters.pop("execution_contract")
    with pytest.raises(ValueError, match="unsupported-action-composition"):
        pomdp_to_gnn_spec(space)


@pytest.mark.parametrize("index", range(3))
def test_unknown_contract_and_invalid_probabilities_fail_closed(index, tmp_path):
    s = spec(index)
    unknown = deepcopy(s)
    unknown["model_parameters"]["execution_contract"] = "future_contract_v99"
    assert (
        "unsupported-execution-contract" in render_gnn_spec(unknown, "jax", tmp_path)[1]
    )
    missing = deepcopy(s)
    missing["model_parameters"].pop("b_tensor_order")
    with pytest.raises(ValueError, match="canonical b_tensor_order"):
        contract_payload(missing)
    key = "A_loc" if index == 2 else ("A_level1" if index == 0 else "A_level0")
    s["structured_pomdp"]["matrices"][key][0][0] += 0.05
    with pytest.raises(ValueError, match="probability mass"):
        contract_payload(s)


def test_block_likelihood_against_latent_trajectory_enumeration():
    from gnn.execute.jax.semantic_runner import block_filter

    payload, m = native_payload(0)
    a, b, reset, d = (
        np.asarray(m[k]) for k in ("A_level1", "B_level1", "A_level2", "D_level2")
    )
    actions = [0, 1, 2, 0]
    # Sum over all 4**5 latent paths independently of forward recursion.
    for observations in ([0, 1, 2, 3, 0], [3, 3, 3, 3, 3], [0, 0, 1, 1, 2]):
        evidence = np.zeros(2)
        for states in product(range(4), repeat=5):
            emission = np.prod([a[o, s] for o, s in zip(observations, states)])
            transition = np.prod(
                [b[states[t + 1], states[t], u] for t, u in enumerate(actions)]
            )
            evidence += reset[states[0]] * emission * transition
        actual = block_filter(
            m["A_level1"],
            m["B_level1"],
            m["A_level2"],
            m["D_level2"],
            observations,
            actions,
        )
        np.testing.assert_allclose(actual["likelihood"], evidence, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            actual["context_posterior"],
            evidence * d / (evidence @ d),
            rtol=0,
            atol=1e-13,
        )
    with pytest.raises(ValueError, match="five observations"):
        block_filter(
            m["A_level1"], m["B_level1"], m["A_level2"], m["D_level2"], [0] * 5, [0] * 5
        )


def test_reset_and_higher_preference_diagnostic_only():
    from gnn.execute.jax.semantic_runner import run_contract

    payload, m = native_payload(0)
    r = run_contract(payload)
    assert (
        r["num_observations"],
        r["num_controlled_transitions"],
        r["num_resets"],
    ) == (20, 16, 3)
    for previous, current in zip(r["blocks"], r["blocks"][1:]):
        expected = np.asarray(m["B_level2"])[:, :, 0] @ previous["context_posterior"]
        np.testing.assert_allclose(current["context_prior"], expected, atol=1e-13)
        np.testing.assert_allclose(
            current["reset_lower_prior"],
            np.asarray(m["A_level2"]) @ expected,
            atol=1e-13,
        )
    altered = deepcopy(payload)
    altered["matrices"]["C_level2"] = [100, -50, 30, -20]
    changed = run_contract(altered)
    for left, right in zip(r["blocks"], changed["blocks"]):
        assert left["observations"] == right["observations"]
        assert left["actions"] == right["actions"]
        assert left["context_posterior"] == right["context_posterior"]
        assert (
            left["higher_preference_diagnostic"]
            != right["higher_preference_diagnostic"]
        )


def numpy_action(a, b, c, q):
    # Independent entropy identity I(S;O)=H(O)-E H(O|S).
    def entropy(v):
        positive = v[v > 0]
        return -np.sum(positive * np.log(positive))

    scores = []
    for action in range(b.shape[2]):
        predicted = b[:, :, action] @ q
        scores.append(
            c @ (a @ predicted)
            + entropy(a @ predicted)
            - sum(predicted[s] * entropy(a[:, s]) for s in range(len(q)))
        )
    scores = np.asarray(scores)
    bound = 8 * np.finfo(scores.dtype).eps * max(1, np.max(abs(scores)))
    return int(np.flatnonzero(scores >= max(scores) - bound)[0])


def test_timed_native_against_independent_numpy_replay():
    from gnn.execute.jax.semantic_runner import run_contract

    payload, _ = native_payload(1)
    payload["parameters"]["num_fast_transitions"] = 200
    payload["parameters"]["seed"] = 0
    result = run_contract(payload)
    m = {k: np.asarray(v) for k, v in payload["matrices"].items()}
    a = [m[f"A_level{k}"] for k in range(3)]
    b = [m[f"B_level{k}"] for k in range(3)]
    base = [m[f"C_level{k}"] for k in range(3)]
    q = [m["D_level0"], m["N21"] @ m["D_level2"], m["D_level2"]]
    fast, medium = [], []
    old_actions = None
    messages_differ_from_tenfold = []
    for row in result["trace"]:
        t = row["transition"]
        if t:
            q[0] = b[0][:, :, old_actions[0]] @ q[0]
        q[0] = q[0] * a[0][row["observation"]]
        q[0] /= q[0].sum()
        if t:
            fast.append(q[0].copy())
        if t and t % 10 == 0:
            h = np.mean(fast, axis=0)
            fast = []
            prediction = b[1][:, :, old_actions[1]] @ q[1]
            # Product of fractional likelihoods independently from exp/log kernel.
            likelihood = np.prod(a[1] ** h[:, None], axis=0)
            q[1] = prediction * likelihood
            q[1] /= q[1].sum()
            wrong = prediction * likelihood**10
            wrong /= wrong.sum()
            messages_differ_from_tenfold.append(np.max(abs(wrong - q[1])))
            medium.append(q[1].copy())
        if t and t % 100 == 0:
            h = np.mean(medium, axis=0)
            medium = []
            q[2] = (b[2][:, :, old_actions[2]] @ q[2]) * np.prod(
                a[2] ** h[:, None], axis=0
            )
            q[2] /= q[2].sum()
            assert row["updates"] == [
                "fast_observation",
                "medium_soft_message",
                "slow_soft_message",
            ]
        preferences = [base[0] + m["M10"] @ q[1], base[1] + m["M21"] @ q[2], base[2]]
        for k in range(3):
            np.testing.assert_allclose(row["beliefs"][k], q[k], rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                row["preferences"][k], preferences[k], rtol=0, atol=1e-12
            )
            if t == 0 or t % (1, 10, 100)[k] == 0:
                assert row["next_actions"][k] == numpy_action(
                    a[k], b[k], preferences[k], q[k]
                )
            else:
                assert row["next_actions"][k] == old_actions[k]
        old_actions = row["next_actions"]
    assert result["slow_updates"] == 2 and result["medium_updates"] == 20
    assert max(messages_differ_from_tenfold) > 0.1
    assert all(row["next_actions"][2] < 2 for row in result["trace"])


def test_episodic_analytic_policy_scores_and_terminal_reward_once():
    from gnn.execute.jax.episodic_contingent import evaluate_policies
    from gnn.execute.jax.semantic_runner import run_contract

    payload, m = native_payload(2)
    policies = evaluate_policies(m, payload["parameters"], 0, m["D_ctx"])
    best = max(policies, key=lambda r: r["score"])
    assert best["first_action"] == 2
    assert best["expected_raw_utility"] == pytest.approx(1.7)
    assert best["trajectory_information"] == pytest.approx(np.log(2))
    assert {x["observation"]: x["action"] for x in best["second_actions"]} == {
        0: 1,
        2: 0,
    }
    for r in policies:
        if r["first_action"] in (0, 1):
            assert r["expected_raw_utility"] == pytest.approx(1)
            assert r["trajectory_information"] == pytest.approx(np.log(2))
            assert r["second_actions"] == []
    fixed = next(
        r
        for r in policies
        if r["first_action"] == 2 and all(x["action"] == 0 for x in r["second_actions"])
    )
    assert fixed["expected_raw_utility"] == pytest.approx(0.5)
    assert fixed["trajectory_information"] == pytest.approx(np.log(2))
    for seed in (0, 1, 2):
        payload["parameters"]["seed"] = seed
        result = run_contract(payload)
        assert result["terminated"] and len(result["trace"]) == 2
        assert result["trace"][0]["location"] == 3
        assert result["trace"][1]["location"] in (1, 2)
    broken = spec(2)
    broken["structured_pomdp"]["matrices"]["B_loc"][1][1][0] = 0
    broken["structured_pomdp"]["matrices"]["B_loc"][0][1][0] = 1
    with pytest.raises(ValueError, match="absorbing"):
        contract_payload(broken)


@pytest.mark.parametrize("index", range(3))
def test_real_render_execute(index, tmp_path, monkeypatch):
    import json

    from gnn.execute.jax import execute_jax_script

    monkeypatch.setenv("GNN_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    success, reason, files = render_gnn_spec(
        spec(index), "jax", tmp_path, {"num_fast_transitions": 200}
    )
    assert success, reason
    assert execute_jax_script(
        Path(files[0]), device="cpu", output_dir=tmp_path / "native", timeout=120
    )
    result = json.loads((tmp_path / "native" / "simulation_results.json").read_text())
    assert result["schema"] == "jax_execution_contract_v1"
    assert result["precision"] == "float64"
    assert result["devices"]
    if index == 0:
        assert result["num_resets"] == 3
    elif index == 1:
        assert result["slow_updates"] == 2
    else:
        assert result["selected_policy"]["expected_raw_utility"] == pytest.approx(1.7)


def test_direct_renderers_cannot_bypass_contract_admission(tmp_path):
    from importlib import import_module

    from gnn.render.bnlearn.bnlearn_renderer import generate_bnlearn_code
    from gnn.render.pomdp_contract import build_canonical_pomdp_spec
    from gnn.render.pomdp_processor._routes import RENDERER_ROUTES

    s = spec(0)
    for backend, route in RENDERER_ROUTES.items():
        if backend == "jax":
            continue
        renderer = getattr(import_module(route.module, "gnn.render"), route.function)
        success, message, files = renderer(s, tmp_path / (backend + route.suffix))
        assert not success and not files
        assert message.startswith("unsupported-execution-contract:")
    with pytest.raises(ValueError, match="unsupported-execution-contract"):
        generate_bnlearn_code(s, tmp_path / "bnlearn.py")
    with pytest.raises(ValueError, match="cannot be canonicalized"):
        build_canonical_pomdp_spec(s)


@pytest.mark.parametrize("index", range(3))
def test_typed_parser_entrypoint_preserves_contract(index, tmp_path):
    from gnn.parsers.markdown_parser import MarkdownGNNParser

    model = (
        MarkdownGNNParser()
        .parse_file(str(ROOT / "input/gnn_files" / (MODELS[index] + ".md")))
        .model
    )
    success, reason, _ = render_gnn_spec(model, "jax", tmp_path)
    assert success, reason
    success, reason, files = render_gnn_spec(model, "rxinfer", tmp_path)
    assert (
        not success
        and not files
        and reason.startswith("unsupported-execution-contract:")
    )


def test_all_256_episodic_policies_against_independent_numpy_tree():
    from gnn.execute.jax.episodic_contingent import evaluate_policies

    payload, native = native_payload(2)
    m = {key: np.asarray(value) for key, value in payload["matrices"].items()}
    rows = evaluate_policies(native, payload["parameters"], 0, native["D_ctx"])
    a, b, prior, utility = m["A_rew"], m["B_loc"], m["D_ctx"], m["C_rew"]
    terminal = {1, 2}
    # A full policy assigns a second action to all three possible signals,
    # even unreachable ones. The implementation emits only reachable branches.
    for first in range(4):
        for second in product(range(4), repeat=3):
            joint = {}
            rewards = {}
            for context in range(2):
                for loc1, obs1 in product(range(4), range(3)):
                    prob1 = prior[context] * b[loc1, 0, first] * a[obs1, loc1, context]
                    if prob1 == 0:
                        continue
                    if loc1 in terminal:
                        key = (loc1, obs1)
                        joint.setdefault(key, np.zeros(2))[context] += prob1
                        rewards[key] = utility[obs1]
                    else:
                        for loc2, obs2 in product(range(4), range(3)):
                            prob2 = (
                                prob1
                                * b[loc2, loc1, second[obs1]]
                                * a[obs2, loc2, context]
                            )
                            if prob2 == 0:
                                continue
                            key = (loc1, obs1, loc2, obs2)
                            joint.setdefault(key, np.zeros(2))[context] += prob2
                            rewards[key] = utility[obs1] + utility[obs2]
            expected_utility = sum(
                prob.sum() * rewards[key] for key, prob in joint.items()
            )
            # Information as prior entropy minus expected posterior entropy.
            expected_entropy = 0.0
            for prob in joint.values():
                posterior = prob / prob.sum()
                positive = posterior[posterior > 0]
                expected_entropy -= prob.sum() * sum(positive * np.log(positive))
            information = np.log(2) - expected_entropy
            row = next(
                r
                for r in rows
                if r["first_action"] == first
                and all(
                    x["action"] == second[x["observation"]] for x in r["second_actions"]
                )
            )
            assert row["expected_raw_utility"] == pytest.approx(
                expected_utility, abs=1e-13
            )
            assert row["trajectory_information"] == pytest.approx(
                information, abs=1e-13
            )
    # Changing utilities to log-softmax introduces a length-dependent penalty.
    log_partition = np.log(np.exp(utility).sum())
    assert (1.7 - 2 * log_partition) < (1 - log_partition)


def test_semantic_metadata_and_precision_negative_controls():
    from gnn.execute.jax.semantic_runner import run_contract

    s = spec(1)
    for dtype in (np.float32, np.float64):
        candidate = deepcopy(s)
        candidate["structured_pomdp"]["matrices"] = {
            k: np.asarray(v, dtype=dtype).tolist()
            for k, v in candidate["structured_pomdp"]["matrices"].items()
        }
        p = contract_payload(candidate)
        np.testing.assert_array_equal(
            np.asarray(p["matrices"]["A_level0"], dtype=dtype)[:, 3],
            np.full(3, 1 / 3, dtype=dtype),
        )
    wrong = deepcopy(s)
    wrong["model_parameters"]["timescale_ratio_2_1"] = 1
    with pytest.raises(ValueError, match="10/100"):
        contract_payload(wrong)
    wrong = deepcopy(s)
    wrong["initialparameterization"]["N21"][0][0] = 0.6
    with pytest.raises(ValueError, match="probability mass"):
        contract_payload(wrong)
    wrong = deepcopy(s)
    wrong["structured_pomdp"]["matrices"]["B_level1"] = (
        np.asarray(wrong["structured_pomdp"]["matrices"]["B_level1"])
        .transpose(2, 0, 1)
        .tolist()
    )
    with pytest.raises(ValueError, match="probability mass"):
        contract_payload(wrong)
    payload = contract_payload(s)
    payload["matrices"]["A_level0"][0][0] = 0.5
    with pytest.raises(ValueError, match="probability mass"):
        run_contract(payload)
