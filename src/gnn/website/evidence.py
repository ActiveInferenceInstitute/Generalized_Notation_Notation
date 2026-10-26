"""Read-only, offline presentation of recorded numerical and formal evidence.

This module does not validate, replay or certify receipts. It keeps reported
quantity outcomes, scientific assumptions and separate evidence planes visible.
"""

from __future__ import annotations

import json
import math
from html import escape
from typing import Any

_SCHEMAS = {
    "gnn-semantic-witness-v3",
    "gnn-semantic-witness-v2",
    "gnn-independent-numerical-witness-v1",
    "fep-semantic-witness-audit-v1",
    "fep-gnn-semantic-native-v1",
    "fep-gaussian-product-witness-audit-v1",
    "fep-gnn-finite-product-native-v1",
}
_NATIVE_SCOPE = (
    "Finite-weight normalization and scalar arbitrary-H prior measurement algebra only; "
    "no GNN parser/runner, matrix Gaussian conditioning or universal translation proof."
)
_PRODUCT_NATIVE_SCOPE = (
    "Finite post-evidence product-weight normalization/nonnegativity with explicit "
    "masses and a correlated counterexample; no Gaussian measure/kernel, "
    "matrix-conditioning, parser/runner or general composition proof."
)
_PRODUCT_AUDIT_BOUNDARY = (
    "Saved FEP audit, shown as unverified reported evidence. Source selection matches "
    "the declared digest only; this audit supplies no retained source bytes and does "
    "not establish source-byte verification, authenticated execution or formal equivalence. "
    "No measured native joint covariance or Gaussian conditioning proof is inferred."
)
_NATIVE_PROFILES = {
    "fep-gnn-semantic-native-v1": (
        _NATIVE_SCOPE,
        ("zero-normalizer", "omitted-observation-noise"),
        "Retained native Lean compiler controls",
    ),
    "fep-gnn-finite-product-native-v1": (
        _PRODUCT_NATIVE_SCOPE,
        ("zero-product-normalizer", "omitted-product-assumption"),
        "Retained finite product algebra compiler controls",
    ),
}
_STATUSES = {"passed", "failed", "refused", "unavailable", "comparable"}
PROOF_BOUNDARY = (
    "Saved receipts are unverified in this viewer. Status and errors below are "
    "reported values. Schema validation, source/hash checks, numerical replay, "
    "native execution, Lean compilation and theorem proof require their own "
    "evidence. Numerical agreement alone establishes no universal equivalence."
)

_CSS = """
#semantic-evidence { margin-bottom: 40px; }
.evidence-boundary { margin: 12px 0 20px; line-height: 1.6; color: var(--text-2); }
.evidence-controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: end; margin: 20px 0 12px; }
.evidence-controls label { display: block; margin-bottom: 6px; }
.evidence-controls input, .evidence-controls select {
  background: var(--bg-card); color: var(--text-1); border: 1px solid var(--text-3);
  border-radius: 8px; padding: 10px; font: inherit; max-width: 100%;
}
.evidence-count { margin-bottom: 16px; color: var(--text-2); }
.evidence-card[hidden] { display: none; }
.evidence-card .evidence-outcome { display: inline-block; margin: 4px 0 16px; }
.evidence-card details { margin-top: 16px; }
.evidence-card th { white-space: nowrap; vertical-align: top; }
.evidence-card td { overflow-wrap: anywhere; }
.evidence-card td summary { white-space: nowrap; gap: 8px; padding: 8px 10px; }
.evidence-card caption { text-align: left; padding: 12px 18px; color: var(--text-2); }
.evidence-card dl { display: grid; grid-template-columns: minmax(100px, 1fr) minmax(0, 3fr); gap: 8px 16px; margin-bottom: 20px; }
.evidence-card dt { color: var(--text-2); }
.evidence-card dd { margin: 0; overflow-wrap: anywhere; }
.evidence-card h4 { margin: 20px 0 12px; }
.evidence-trace-explorer svg { display: block; width: 100%; height: auto; background: #101522; border-radius: 8px; }
.evidence-trace-status, .evidence-trace-note { margin: 12px 0; color: var(--text-2); line-height: 1.6; }
@media (max-width: 600px) { .evidence-card dl { grid-template-columns: 1fr; } }
"""

_FILTER_JS = """(function () {
  'use strict';
  var root = document.getElementById('semantic-evidence');
  if (!root) { return; }
  var query = root.querySelector('#evidence-query');
  var status = root.querySelector('#evidence-status');
  var count = root.querySelector('#evidence-count');
  var cards = root.querySelectorAll('.evidence-card');
  if (!query || !status || !count) { return; }
  function update() {
    var needle = query.value.trim().toLowerCase();
    var visible = 0;
    Array.prototype.forEach.call(cards, function (card) {
      var matchesText = !needle || card.textContent.toLowerCase().indexOf(needle) !== -1;
      var matchesStatus = !status.value || card.getAttribute('data-evidence-status') === status.value;
      card.hidden = !(matchesText && matchesStatus);
      if (!card.hidden) { visible += 1; }
    });
    count.textContent = visible + ' of ' + cards.length + ' recorded evidence entries shown.';
  }
  query.addEventListener('input', update);
  status.addEventListener('change', update);
  update();
})();"""

_TRACE_JS = """(function () {
  'use strict';
  document.querySelectorAll('[data-trace-explorer]').forEach(function (root) {
    var data;
    try { data = JSON.parse(root.querySelector('[data-trace-data]').textContent); }
    catch (error) { return; }
    var quantity = root.querySelector('[data-trace-quantity]');
    var state = root.querySelector('[data-trace-state]');
    var mode = root.querySelector('[data-trace-mode]');
    var svg = root.querySelector('svg');
    var ns = svg.namespaceURI;
    var status = root.querySelector('[data-trace-status]');
    function node(tag, attributes, text) {
      var el = document.createElementNS(ns, tag);
      Object.keys(attributes).forEach(function (key) { el.setAttribute(key, attributes[key]); });
      if (text !== undefined) { el.textContent = text; }
      return el;
    }
    function draw() {
      var series = data.series[quantity.value];
      var axis = Number(state.value);
      if (!series || !Number.isInteger(axis) || axis < 0 || axis >= data.states.length) { return; }
      var residual = mode.value === 'residual';
      var traces = residual ? [series.residual] : [series.reference, series.candidate];
      var values = [];
      traces.forEach(function (trace) { trace.forEach(function (row) { values.push(row[axis]); }); });
      if (!values.length || !values.every(Number.isFinite)) { return; }
      var scale = Math.max.apply(null, values.map(Math.abs)) || 1;
      var normalized = values.map(function (value) { return value / scale; });
      var low = Math.min.apply(null, normalized);
      var high = Math.max.apply(null, normalized);
      var span = high - low;
      if (!span) { low -= 0.5; high += 0.5; span = 1; }
      function x(index) { return data.prefixes.length === 1 ? 330 : 65 + index * 535 / (data.prefixes.length - 1); }
      function y(value) { return 200 - (value / scale - low) * 160 / span; }
      var previous = svg.querySelector('[data-trace-drawing]');
      if (previous) { previous.remove(); }
      var empty = svg.querySelector('[data-trace-empty]');
      if (empty) { empty.remove(); }
      var group = node('g', {'data-trace-drawing': ''});
      group.appendChild(node('path', {d: 'M65 35 V200 H600', stroke: '#aeb6c2', fill: 'none'}));
      [low, high].forEach(function (value) {
        var tick = value * scale;
        var label = Number.isFinite(tick) ? tick.toPrecision(4) : value.toPrecision(3) + ' × ' + scale.toExponential(2);
        group.appendChild(node('text', {x: 58, y: 204 - (value - low) * 160 / span, fill: '#aeb6c2', 'text-anchor': 'end', 'font-size': 11}, label));
      });
      if (residual && low <= 0 && high >= 0) {
        group.appendChild(node('path', {d: 'M65 ' + y(0) + ' H600', stroke: '#aeb6c2', 'stroke-dasharray': '2 4', fill: 'none'}));
      }
      data.prefixes.forEach(function (prefix, index) {
        group.appendChild(node('text', {x: x(index), y: 220, fill: '#aeb6c2', 'text-anchor': 'middle', 'font-size': 11}, String(prefix)));
      });
      traces.forEach(function (trace, traceIndex) {
        var color = residual || traceIndex === 0 ? '#00f0ff' : '#a881fb';
        var points = trace.map(function (row, index) { return x(index) + ',' + y(row[axis]); }).join(' ');
        group.appendChild(node('polyline', {points: points, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-dasharray': traceIndex === 1 ? '7 4' : 'none'}));
        trace.forEach(function (row, index) { group.appendChild(node('circle', {cx: x(index), cy: y(row[axis]), r: 3, fill: color})); });
      });
      svg.appendChild(group);
      var title = data.states[axis] + ': ' + series.label + (residual ? ' residual (candidate minus reference)' : ' reference and candidate');
      svg.querySelector('title').textContent = title;
      svg.querySelector('desc').textContent = 'Unverified retained values by conditioning prefix. Reported quantity status: ' + series.status + '. Complete values are in the following table.';
      status.textContent = title + '. Reported quantity status: ' + series.status + '. ' + (series.status === 'failed' ? 'Diagnostic visualization of a reported mismatch.' : 'Display only; no verification claim.');
    }
    [quantity, state, mode].forEach(function (control) { control.addEventListener('change', draw); });
    draw();
  });
})();"""

_TRACE_IDENTITIES = (
    "dtype",
    "initial_time",
    "axes",
    "conditioning_horizon",
    "num_timesteps",
    "num_states",
    "num_observations",
    "random_seed",
    "inference_contract",
    "inference_mode",
    "control_mode",
)


def _mapping(value: Any) -> dict[str, Any]:
    """Treat malformed optional mappings as absent without claiming validity."""
    return value if isinstance(value, dict) else {}


def _json(value: Any) -> str:
    """Format a recorded field for display, preserving unusable-data diagnostics."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (ValueError, TypeError, RecursionError) as error:
        return f"Receipt could not be serialized for display: {type(error).__name__}"


def _text(value: Any) -> str:
    """Escape a recorded scalar or structured value without interpreting markup."""
    if value is None or value == "":
        return "unavailable"
    text = _json(value) if isinstance(value, (dict, list, tuple)) else str(value)
    # JSON can retain lone UTF-16 surrogates; show their escapes in UTF-8 pages.
    return escape(text.encode("utf-8", errors="backslashreplace").decode("utf-8"))


def _status(value: Any) -> str:
    """Retain known reported outcomes; unknown values do not become passes."""
    return value if isinstance(value, str) and value in _STATUSES else "unavailable"


def is_evidence_artifact(item: Any) -> bool:
    """Recognize declared receipts and the existing Step 16 comparison dataset."""
    if not isinstance(item, dict):
        return False
    if isinstance(item.get("schema"), str) and item["schema"] in _SCHEMAS:
        return True
    comparisons = _mapping(item.get("comparisons"))
    return isinstance(comparisons.get("metric_agreement"), dict) or isinstance(
        comparisons.get("scientific_exclusions"), dict
    )


def _details(label: str, value: Any) -> str:
    """A native disclosure keeps complete recorded fields available offline."""
    rendered = _json(value).encode("utf-8", errors="backslashreplace").decode("utf-8")
    return (
        f"<details><summary>{escape(label)}</summary>"
        f'<div class="details-body"><pre>{escape(rendered, quote=False)}</pre></div></details>'
    )


def _content_metadata(value: Any, fields: tuple[str, ...]) -> Any:
    """Summarize duplicated text bodies while retaining their full raw disclosure."""
    if not isinstance(value, dict):
        return value
    metadata = dict(value)
    for field in fields:
        content = value.get(field)
        if isinstance(content, str):
            try:
                size: int | str = len(content.encode("utf-8"))
            except UnicodeEncodeError:
                size = "unavailable: invalid UTF-8 text"
            metadata[field] = {
                "utf8_bytes": size,
                "content_location": "Complete saved receipt (unverified) below",
            }
    return metadata


def _artifact_provenance(witness: dict[str, Any]) -> dict[str, Any]:
    """Keep recorded digests/runtime metadata without copying large bodies twice."""
    provenance: dict[str, Any] = {
        key: witness[key]
        for key in ("formal_evidence", "execution_evidence")
        if key in witness
    }
    if "artifacts" in witness:
        artifacts = witness["artifacts"]
        provenance["artifacts"] = (
            {
                key: _content_metadata(value, ("utf8",))
                for key, value in artifacts.items()
            }
            if isinstance(artifacts, dict)
            else artifacts
        )
    if "native" in witness:
        provenance["native"] = _content_metadata(
            witness["native"], ("script_utf8", "result_utf8")
        )
    return provenance


def _quantity_table(comparisons: Any, title: str) -> str:
    """Display recorded errors and tolerances, never infer numerical agreement."""
    rows = ""
    for name, raw in _mapping(comparisons).items():
        record = _mapping(raw)
        status = _status(record.get("status"))
        reason = record.get("reason", record.get("reasons"))
        if not record:
            reason = "Unusable comparison record; no outcome established."
        diagnostics = {
            key: record[key]
            for key in (
                "expected",
                "actual",
                "expected_shape",
                "actual_shape",
                "shape",
                "numeric_status",
            )
            if key in record
        }
        detail = (
            _details("Recorded comparison detail", diagnostics) if diagnostics else ""
        )
        rows += (
            f'<tr><th scope="row">{_text(name)}</th>'
            f"<td>Reported {status}</td>"
            f"<td>{_text(record.get('max_abs_error'))}</td>"
            f"<td>{_text(record.get('max_rel_error'))}</td>"
            f"<td>{_text(record.get('atol'))}</td>"
            f"<td>{_text(record.get('rtol'))}</td>"
            f"<td>{_text(reason)}{detail}</td></tr>"
        )
    if not rows:
        rows = '<tr><td colspan="7">No recorded quantity comparisons; unavailable.</td></tr>'
    return (
        '<div class="table-wrap"><table>'
        f"<caption>{escape(title)}</caption>"
        '<thead><tr><th scope="col">Quantity</th><th scope="col">Reported outcome</th>'
        '<th scope="col">Maximum absolute error</th><th scope="col">Maximum relative error</th>'
        '<th scope="col">Absolute tolerance</th><th scope="col">Relative tolerance</th>'
        f'<th scope="col">Reason</th></tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _trace_array(value: Any, shape: tuple[int, ...]) -> Any:
    """Admit only finite nonboolean values at one small declared display shape."""
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            return None
        values = [_trace_array(item, shape[1:]) for item in value]
        return None if any(item is None for item in values) else values
    try:
        return (
            float(value)
            if type(value) in (int, float) and math.isfinite(value)
            else None
        )
    except OverflowError:
        return None


def _same_json(left: Any, right: Any) -> bool:
    """Require serializable typed metadata equality without accepting display errors."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (ValueError, TypeError, RecursionError, OverflowError):
        return False


def _bounded_labels(value: Any, maximum: int) -> bool:
    """Admit small ordered label sets before formatting optional source data."""
    return (
        isinstance(value, list)
        and 1 <= len(value) <= maximum
        and all(
            isinstance(label, str)
            and 1 <= len(label) <= 256
            and not any(0xD800 <= ord(character) <= 0xDFFF for character in label)
            for label in value
        )
        and len(set(value)) == len(value)
    )


def _parameters_align(
    declared: Any, native: Any, states: int, observations: int
) -> bool:
    """Check bounded recorded parameter alignment, without inference or replay."""
    shapes = {
        "F": (states, states),
        "H": (observations, states),
        "Q": (states, states),
        "R": (observations, observations),
        "prior_mean": (states,),
        "prior_cov": (states, states),
    }
    if not isinstance(declared, dict) or not isinstance(native, dict):
        return False
    if set(declared) != set(shapes) or set(native) != set(shapes):
        return False
    for name, shape in shapes.items():
        expected, actual = (
            _trace_array(declared[name], shape),
            _trace_array(native[name], shape),
        )
        if expected is None or actual is None or expected != actual:
            return False
    return True


def _trace_payload(
    witness: dict[str, Any], reported_status: str
) -> dict[str, Any] | None:
    """Build a bounded view only when recorded identities permit aligned display.

    These structural guards protect the explorer; they do not verify a receipt,
    authenticate native labels or recompute scientific reference algorithms.
    """
    if (
        witness.get("schema") != "gnn-semantic-witness-v2"
        or reported_status not in {"passed", "failed"}
        or _status(witness.get("status")) not in {"passed", "failed"}
    ):
        return None
    return _aligned_trace_payload(
        _mapping(witness.get("source_contract")),
        _mapping(witness.get("reference")),
        _mapping(witness.get("candidate")),
        _mapping(witness.get("comparisons")),
        _mapping(witness.get("assumptions")),
    )


def _aligned_trace_payload(
    contract: dict[str, Any],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    comparisons: dict[str, Any],
    assumptions: dict[str, Any],
    native_quantities: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Apply display-only alignment guards to one retained component."""
    parameters = _mapping(contract.get("model_parameters"))
    states, observations = contract.get("state_ids"), contract.get("observation_ids")
    if not isinstance(states, list) or not isinstance(observations, list):
        return None
    if any(not _bounded_labels(labels, 4) for labels in (states, observations)):
        return None
    count = parameters.get("num_timesteps")
    if (
        type(count) is not int
        or not 1 <= count <= 8
        or not isinstance(contract.get("model"), str)
        or contract["model"] not in {"finite", "gaussian"}
        or contract.get("question") != "filtering"
        or type(parameters.get("random_seed")) is not int
        or parameters["random_seed"] < 0
    ):
        return None
    if any(
        _mapping(comparisons.get("identity:" + key)).get("status") != "passed"
        for key in _TRACE_IDENTITIES
    ):
        return None
    if any(
        _mapping(row).get("status") != "passed"
        for key, row in comparisons.items()
        if isinstance(key, str) and key.startswith(("identity:", "parameter:"))
    ):
        return None
    prefixes = list(range(1, count + 1))
    expected = {
        "dtype": "float64",
        "initial_time": "first_observation_uses_prior",
        "axes": {"F": ["next_state", "previous_state"], "H": ["observation", "state"]}
        if contract["model"] == "gaussian"
        else {
            "A": ["observation", "state"],
            "B": ["next_state", "previous_state", "action"],
        },
        "conditioning_horizon": prefixes,
        "num_timesteps": count,
        "num_states": len(states),
        "num_observations": len(observations),
        "random_seed": parameters.get("random_seed"),
        "inference_contract": "bnlearn_fixed_action_filter_v1"
        if contract["model"] == "finite"
        else "gnn_lgssm_filter_v1",
        "inference_mode": "exact_categorical_filtering"
        if contract["model"] == "finite"
        else "gaussian_filtering",
        "control_mode": "fixed_transition_actions"
        if contract["model"] == "finite"
        else "passive",
    }
    if (
        contract.get("dtype") != expected["dtype"]
        or contract.get("initial_time") != expected["initial_time"]
        or not _same_json(contract.get("axes"), expected["axes"])
    ):
        return None
    if any(
        not _same_json(candidate.get(key), value) for key, value in expected.items()
    ) or not _same_json(reference.get("conditioning_horizon"), prefixes):
        return None
    if any(
        not _same_json(assumptions.get(key), value)
        for key, value in {
            "question": "filtering",
            "initial_time": expected["initial_time"],
            "state_ids": states,
            "observation_ids": observations,
            "axes": expected["axes"],
            "conditioning_horizon": prefixes,
        }.items()
    ):
        return None
    ref_quantities, native_quantities = (
        _mapping(reference.get("quantities")),
        _mapping(candidate.get("quantities"))
        if native_quantities is None
        else native_quantities,
    )
    series = {}
    matrices = None
    for quantity, label in (
        (
            "beliefs",
            "Categorical posterior probability"
            if contract["model"] == "finite"
            else "Gaussian posterior mean",
        ),
        ("posterior_cov", "Posterior diagonal variance"),
    ):
        status = _status(_mapping(comparisons.get(quantity)).get("status"))
        if status not in {"passed", "failed"} or (
            quantity == "posterior_cov" and contract["model"] != "gaussian"
        ):
            continue
        shape = (
            (count, len(states))
            if quantity == "beliefs"
            else (count, len(states), len(states))
        )
        left, right = (
            _trace_array(ref_quantities.get(quantity), shape),
            _trace_array(native_quantities.get(quantity), shape),
        )
        if left is None or right is None:
            continue
        if quantity == "posterior_cov":
            full_matrices = {"reference": left, "candidate": right}
            left = [
                [matrix[index][index] for index in range(len(states))]
                for matrix in left
            ]
            right = [
                [matrix[index][index] for index in range(len(states))]
                for matrix in right
            ]
            if any(
                value <= 0 for trace in (left, right) for row in trace for value in row
            ):
                continue
            matrices = full_matrices
        elif contract["model"] == "finite" and any(
            any(value < 0 or value > 1 for value in row)
            or not math.isclose(sum(row), 1, abs_tol=1e-12, rel_tol=0)
            for trace in (left, right)
            for row in trace
        ):
            continue
        residual = [
            [
                actual - wanted
                for wanted, actual in zip(ref_row, native_row, strict=True)
            ]
            for ref_row, native_row in zip(left, right, strict=True)
        ]
        if any(not math.isfinite(value) for row in residual for value in row):
            continue
        series["beliefs" if quantity == "beliefs" else "posterior_variance"] = {
            "label": label,
            "status": status,
            "reference": left,
            "candidate": right,
            "residual": residual,
        }
    return (
        {
            "states": states,
            "prefixes": prefixes,
            "series": series,
            "covariance_matrices": matrices,
        }
        if series
        else None
    )


def _trace_explorer(witness: dict[str, Any], index: int, reported_status: str) -> str:
    """Offer state/quantity/residual controls plus a complete native table fallback."""
    if witness.get("schema") == "gnn-semantic-witness-v3":
        return _composition_explorers(witness, index, reported_status)
    payload = _trace_payload(witness, reported_status)
    return _trace_markup(payload, str(index)) if payload is not None else ""


def _trace_markup(payload: dict[str, Any], index: str) -> str:
    """Render aligned retained arrays, never a scientific replay."""
    options = "".join(
        f'<option value="{quantity}">{escape(series["label"])}</option>'
        for quantity, series in payload["series"].items()
    )
    states = "".join(
        f'<option value="{axis}">{escape(label)}</option>'
        for axis, label in enumerate(payload["states"])
    )
    rows = ""
    for quantity in payload["series"].values():
        for step, prefix in enumerate(payload["prefixes"]):
            for axis, state in enumerate(payload["states"]):
                rows += f"<tr><td>{escape(quantity['label'])}</td><td>{prefix}</td><td>{escape(state)}</td><td>{quantity['reference'][step][axis]:.17g}</td><td>{quantity['candidate'][step][axis]:.17g}</td><td>{quantity['residual'][step][axis]:.17g}</td></tr>"
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    return f"""
<details><summary>Explore retained reference and candidate traces (unverified)</summary>
<div class="details-body evidence-trace-explorer" data-trace-explorer>
  <p class="evidence-trace-note">Source state labels name array axes; native label provenance is not authenticated. Variance uses only covariance diagonal entries and is distinct from the posterior mean. Residuals are candidate minus reference for display. No confidence bands or verification claims are inferred.</p>
  <div class="evidence-controls">
    <div><label for="trace-quantity-{index}">Recorded quantity</label><select id="trace-quantity-{index}" data-trace-quantity>{options}</select></div>
    <div><label for="trace-state-{index}">State axis (source label)</label><select id="trace-state-{index}" data-trace-state>{states}</select></div>
    <div><label for="trace-mode-{index}">Display mode</label><select id="trace-mode-{index}" data-trace-mode><option value="values">Reference and candidate values</option><option value="residual">Residual (candidate minus reference)</option></select></div>
  </div>
  <p class="evidence-trace-status" data-trace-status role="status" aria-live="polite">Retained values only; failed quantity comparisons are diagnostic.</p>
  <svg viewBox="0 0 640 250" role="img" aria-labelledby="trace-title-{index} trace-desc-{index}">
    <title id="trace-title-{index}">Unverified retained numerical traces</title>
    <desc id="trace-desc-{index}">Reference and candidate series with a complete value table below.</desc>
    <text data-trace-empty x="20" y="125" fill="#aeb6c2">Interactive plot requires JavaScript; full values follow.</text>
  </svg>
  <p class="evidence-trace-note">Horizontal axis: conditioning prefix (number of observations included). Values: reference is solid cyan; candidate is dashed purple. Residual mode shows one cyan difference trace.</p>
  <details><summary>Full retained trace values and display residuals</summary><div class="table-wrap"><table>
    <caption>Unverified recorded arrays and display-only residuals</caption>
    <thead><tr><th scope="col">Quantity</th><th scope="col">Conditioning prefix</th><th scope="col">Source state label</th><th scope="col">Reference</th><th scope="col">Candidate</th><th scope="col">Candidate − reference</th></tr></thead><tbody>{rows}</tbody>
  </table></div></details>
  {_covariance_table(payload)}
  <script type="application/json" data-trace-data>{encoded}</script>
</div></details>"""


def _covariance_table(payload: dict[str, Any]) -> str:
    """Keep off-diagonal covariance values available alongside diagonal plots."""
    matrices = payload.get("covariance_matrices")
    if not isinstance(matrices, dict):
        return ""
    rows = []
    for step, prefix in enumerate(payload["prefixes"]):
        for row, row_label in enumerate(payload["states"]):
            for column, column_label in enumerate(payload["states"]):
                reference = matrices["reference"][step][row][column]
                candidate = matrices["candidate"][step][row][column]
                residual = candidate - reference
                # Extreme finite operands can overflow a display-only subtraction.
                difference = (
                    f"{residual:.17g}"
                    if math.isfinite(residual)
                    else "unavailable (display overflow)"
                )
                rows.append(
                    f"<tr><td>{prefix}</td><td>{escape(row_label)}</td><td>{escape(column_label)}</td><td>{reference:.17g}</td><td>{candidate:.17g}</td><td>{difference}</td></tr>"
                )
    return (
        "<details><summary>Full retained covariance matrices, including off-diagonal entries</summary>"
        '<div class="table-wrap"><table><caption>Recorded full covariance; display only</caption>'
        '<thead><tr><th scope="col">Conditioning prefix</th><th scope="col">Row state</th>'
        '<th scope="col">Column state</th><th scope="col">Reference</th><th scope="col">Candidate</th>'
        '<th scope="col">Candidate − reference</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></details>"
    )


def _composition_explorers(
    witness: dict[str, Any], index: int, reported_status: str
) -> str:
    """Present aligned retained agents separately, without claiming native joint covariance."""
    if reported_status not in {"passed", "failed"} or _status(
        witness.get("status")
    ) not in {"passed", "failed"}:
        return ""
    contract = _mapping(witness.get("source_contract"))
    assumptions = _mapping(witness.get("assumptions"))
    reference = _mapping(witness.get("reference"))
    candidate = _mapping(witness.get("candidate"))
    comparisons = _mapping(witness.get("comparisons"))
    agents = ["agent1", "agent2"]
    axes = {"F": ["next_state", "previous_state"], "H": ["observation", "state"]}
    digest = contract.get("source_sha256")
    options = _mapping(contract.get("model_parameters"))
    if (
        contract.get("schema") != "gnn-independent-gaussian-composition-v1"
        or contract.get("model") != "gaussian_product"
        or contract.get("agents") != agents
        or candidate.get("agents") != agents
        or contract.get("question") != "filtering"
        or contract.get("dtype") != "float64"
        or contract.get("initial_time") != "first_observation_uses_prior"
        or not _same_json(contract.get("axes"), axes)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or set(options)
        != {
            "nr_agents",
            "agent_coupling",
            "num_timesteps",
            "random_seed_agent1",
            "random_seed_agent2",
        }
        or type(options.get("nr_agents")) is not int
        or options["nr_agents"] != 2
        or options.get("agent_coupling") != "independent"
        or type(options.get("num_timesteps")) is not int
        or not 1 <= options["num_timesteps"] <= 8
        or assumptions.get("question") != "filtering"
        or assumptions.get("agent_coupling") != "independent"
        or assumptions.get("control_mode") != "passive"
    ):
        return ""
    required = (
        "schema_version",
        "model_kind",
        "framework",
        "agents",
        "agent_coupling",
        "num_timesteps",
        "validation.all_valid",
        "agent_results",
        "beliefs_projection_agents",
        "posterior_cov_projection_agents",
        "controls_projection_agents",
        "observations_continuous_projection_agents",
    )
    if any(
        _mapping(comparisons.get("identity:" + key)).get("status") != "passed"
        for key in required
    ):
        return ""
    if any(
        _mapping(row).get("status") != "passed"
        for key, row in comparisons.items()
        if isinstance(key, str) and key.startswith("identity:")
    ):
        return ""
    if any(
        not _same_json(assumptions.get(key), contract.get(key))
        for key in ("state_ids", "observation_ids", "axes", "initial_time")
    ):
        return ""
    components = _mapping(contract.get("components"))
    references = _mapping(reference.get("components"))
    candidates = _mapping(candidate.get("components"))
    if any(
        set(mapping) != set(agents) for mapping in (components, references, candidates)
    ):
        return ""
    for key in ("state_ids", "observation_ids"):
        ordered: list[list[Any]] = []
        for agent in agents:
            labels = _mapping(components[agent]).get(key)
            if not isinstance(labels, list) or not _bounded_labels(labels, 3):
                return ""
            ordered.append(labels)
        if sum(len(labels) for labels in ordered) > 4:
            return ""
        qualified = [
            f"{agent}:{label}"
            for agent, labels in zip(agents, ordered, strict=True)
            for label in labels
        ]
        if qualified != contract.get(key):
            return ""
    declared_parameters = {
        f"{key}_{agent}": value
        for agent in agents
        for key, value in _mapping(
            _mapping(components[agent]).get("parameters")
        ).items()
    }
    if not _same_json(declared_parameters, contract.get("parameters")):
        return ""
    result = []
    for agent in agents:
        payload = _composition_component_trace(
            agent,
            contract,
            reference,
            assumptions,
            comparisons,
            _mapping(components[agent]),
            _mapping(references[agent]),
            _mapping(candidates[agent]),
        )
        if payload is None:
            return ""
        result.append(
            f"<h4>Retained {agent} component</h4>"
            + _trace_markup(payload, f"{index}-{agent}")
        )
    return (
        '<p class="evidence-trace-note">Independent agent components are shown separately. '
        "Zero cross-agent covariance belongs to the declared product reference model; "
        "it is not a measured native joint covariance or a proof of independence.</p>"
        + "".join(result)
    )


def _composition_component_trace(
    agent: str,
    contract: dict[str, Any],
    reference: dict[str, Any],
    assumptions: dict[str, Any],
    comparisons: dict[str, Any],
    component: dict[str, Any],
    recorded: dict[str, Any],
    native: dict[str, Any],
) -> dict[str, Any] | None:
    """Check one recorded component's bounded source/native metadata alignment."""
    options = _mapping(contract.get("model_parameters"))
    parameters = _mapping(component.get("model_parameters"))
    count = parameters.get("num_timesteps")
    seed = parameters.get("random_seed")
    if (
        type(count) is not int
        or count != options["num_timesteps"]
        or type(seed) is not int
        or not 0 <= seed <= 2**32 - 1
        or type(options.get("random_seed_" + agent)) is not int
        or seed != options["random_seed_" + agent]
        or component.get("schema") != contract.get("schema")
        or component.get("model") != "gaussian"
        or any(
            not _same_json(component.get(key), contract.get(key))
            for key in ("dtype", "initial_time", "axes", "question", "source_sha256")
        )
        or any(
            not _same_json(recorded.get(key), component.get(key))
            for key in ("state_ids", "observation_ids")
        )
        or native.get("schema_version") != "gnn_lgssm_filter_v1"
        or native.get("model_kind") != "continuous"
        or native.get("framework") != "jax"
        or _mapping(native.get("validation")).get("all_valid") is not True
        or ("success" in native and native["success"] is not True)
        or ("status" in native and native["status"] not in ("success", "completed"))
        or any(
            key in native and not _same_json(native[key], component[key])
            for key in ("state_ids", "observation_ids", "source_sha256")
        )
    ):
        return None
    prefixes = list(range(1, count + 1))
    if any(
        not _same_json(item.get("conditioning_horizon"), prefixes)
        for item in (reference, assumptions)
    ):
        return None
    states, observations = (
        len(component["state_ids"]),
        len(component["observation_ids"]),
    )
    if not _parameters_align(
        component.get("parameters"), native.get("parameters"), states, observations
    ):
        return None
    if _trace_array(native.get("controls"), (count, states)) != [
        [0.0] * states for _ in prefixes
    ]:
        return None
    observed = _trace_array(recorded.get("observations"), (count, observations))
    if observed is None or observed != _trace_array(
        native.get("observations_continuous"), (count, observations)
    ):
        return None
    local = {
        key[len(agent) + 1 :]: value
        for key, value in comparisons.items()
        if isinstance(key, str) and key.startswith(agent + ":")
    }
    required = (
        "identity:framework",
        "identity:schema_version",
        "identity:model_kind",
        "identity:validation.all_valid",
        "identity:parameter_keys",
        "parameter:F",
        "parameter:H",
        "parameter:Q",
        "parameter:R",
        "parameter:prior_mean",
        "parameter:prior_cov",
        "controls",
        "observations",
        "projection:beliefs",
        "projection:posterior_cov",
        "projection:controls",
        "projection:observations_continuous",
        "validity:beliefs",
        "validity:posterior_cov",
    )
    if any(_mapping(local.get(key)).get("status") != "passed" for key in required):
        return None
    if any(
        _mapping(row).get("status") != "passed"
        for key, row in local.items()
        if key.startswith(("identity:", "parameter:", "projection:", "validity:"))
        or key in {"controls", "observations"}
    ):
        return None
    view_assumptions = {
        key: component.get(key)
        for key in ("question", "initial_time", "state_ids", "observation_ids", "axes")
    }
    view_assumptions["conditioning_horizon"] = prefixes
    payload = _aligned_trace_payload(
        component,
        {**recorded, "conditioning_horizon": prefixes},
        native,
        local,
        view_assumptions,
        {key: native.get(key) for key in ("beliefs", "posterior_cov")},
    )
    if payload is None or set(payload["series"]) != {"beliefs", "posterior_variance"}:
        return None
    return payload


def _pair_quantities(agreement: dict[str, Any]) -> dict[str, Any]:
    """Present existing comparison booleans as their explicitly reported outcomes."""
    if agreement.get("status") != "comparable":
        return {
            "comparison admission": {
                "status": "refused",
                "reason": agreement.get("reasons") or "No comparable pair recorded.",
            }
        }
    result: dict[str, Any] = {}
    tolerance = _mapping(agreement.get("tolerance"))
    quantities = dict(_mapping(agreement.get("views")))
    for name in ("free_energy", "expected_free_energy", "variational_free_energy"):
        if name in agreement:
            quantities[name] = agreement[name]
    for name, raw in quantities.items():
        recorded = _mapping(raw)
        for prefix, label in (("", "posterior"), ("covariance_", "covariance")):
            key = prefix + "within_tolerance"
            if prefix and key not in recorded:
                continue
            flag = recorded.get(key)
            quantity_name = str(name) if name in agreement else f"{name} / {label}"
            result[quantity_name] = {
                "status": "passed"
                if flag is True
                else "failed"
                if flag is False
                else "unavailable",
                "max_abs_error": recorded.get(prefix + "max_absolute_difference"),
                "atol": tolerance.get("atol"),
                "rtol": tolerance.get("rtol"),
                "reason": recorded.get("convention"),
            }
        for quantity, reason in _mapping(recorded.get("unavailable_metrics")).items():
            result[f"{name} / {quantity}"] = {"status": "unavailable", "reason": reason}
    for quantity, reason in _mapping(agreement.get("unavailable_metrics")).items():
        result[str(quantity)] = {"status": "unavailable", "reason": reason}
    return result


def _entries(analysis: Any) -> list[dict[str, Any]]:
    """Build display records without changing or numerically evaluating receipts."""
    entries: list[dict[str, Any]] = []
    if not isinstance(analysis, list):
        return entries
    for item in analysis:
        if not is_evidence_artifact(item):
            continue
        schema = item.get("schema")
        if isinstance(schema, str) and schema in _NATIVE_PROFILES:
            entries.append(
                {
                    "kind": "native_formal",
                    "status": _status(item.get("status")),
                    "raw": item,
                }
            )
            continue
        if schema == "fep-gaussian-product-witness-audit-v1":
            entries.append(
                {
                    "kind": "product_audit",
                    "status": _status(item.get("status")),
                    "raw": item,
                }
            )
            continue
        if isinstance(schema, str) and schema in _SCHEMAS:
            witness = (
                _mapping(item.get("gnn_witness"))
                if schema == "fep-semantic-witness-audit-v1"
                else item
            )
            contract = _mapping(witness.get("source_contract"))
            entries.append(
                {
                    "title": contract.get("model_id")
                    or contract.get("model")
                    or schema,
                    "status": _status(item.get("status")),
                    "schema": schema,
                    "contract": contract,
                    "comparisons": witness.get("comparisons"),
                    "witness": witness,
                    "raw": item,
                }
            )
            continue
        comparisons = _mapping(item.get("comparisons"))
        for label, raw in _mapping(comparisons.get("metric_agreement")).items():
            agreement = _mapping(raw)
            entries.append(
                {
                    "title": label,
                    "status": "comparable"
                    if agreement.get("status") == "comparable"
                    else "refused",
                    "schema": "Step 16 cross-framework comparison",
                    "contract": {
                        key: agreement.get(key)
                        for key in (
                            "model_id",
                            "source_sha256",
                            "model_kind",
                            "inference_mode",
                            "dtypes",
                        )
                    },
                    "comparisons": _pair_quantities(agreement),
                    "witness": agreement,
                    "raw": agreement,
                }
            )
        for label, reasons in _mapping(
            comparisons.get("scientific_exclusions")
        ).items():
            entries.append(
                {
                    "title": label,
                    "status": "refused",
                    "schema": "Step 16 scientific exclusion",
                    "contract": {},
                    "comparisons": {
                        "comparison admission": {"status": "refused", "reason": reasons}
                    },
                    "witness": {},
                    "raw": {"label": label, "reasons": reasons},
                }
            )
    return entries


def _evidence_planes(raw: dict[str, Any], witness: dict[str, Any]) -> str:
    """Keep reference, proof, compilation, execution and custody claims separate."""
    formal = _mapping(raw.get("formal_evidence", witness.get("formal_evidence")))
    execution = _mapping(
        raw.get("execution_evidence", witness.get("execution_evidence"))
    )
    reference = witness.get("reference", witness.get("expected"))
    replay = _mapping(raw.get("independent_replay"))
    planes = {
        "Reference record": "supplied, unverified"
        if isinstance(reference, dict)
        else "unavailable",
        "Reference algorithm": _mapping(reference).get(
            "algorithm", witness.get("reference_algorithm", "unavailable")
        ),
        "Independent replay": replay.get("status", "unavailable"),
        "Formal references / proof": formal.get("status", "unavailable"),
        "Lean compilation": _mapping(raw.get("compilation_evidence")).get(
            "status", "unavailable"
        ),
        "Execution": execution.get("status", "unavailable"),
        "Execution source verified": execution.get(
            "execution_source_verified", "unavailable"
        ),
        "Reported framework": execution.get("reported_framework", "unavailable"),
        "Reported runtime versions": execution.get("reported_versions", "unavailable"),
        "Pair custody": _mapping(raw.get("pair_evidence")).get("status", "unavailable"),
        "Universal equivalence claim": formal.get(
            "universal_equivalence", "unavailable"
        ),
    }
    rows = "".join(
        f"<dt>{escape(name)}</dt><dd>{_text(value)}</dd>"
        for name, value in planes.items()
    )
    return f"<h4>Separate recorded evidence</h4><dl>{rows}</dl>"


def _product_audit_card(entry: dict[str, Any], index: int) -> str:
    """Present a saved FEP replay audit without inventing retained source bytes."""
    receipt = entry["raw"]
    replay = _mapping(receipt.get("independent_replay"))
    formal = _mapping(receipt.get("formal_evidence"))
    execution = _mapping(receipt.get("execution_evidence"))
    identity = {
        "Schema": receipt.get("schema"),
        "Declared GNN source SHA-256": receipt.get("source_sha256"),
        "Saved input witness SHA-256": _mapping(receipt.get("input")).get("sha256"),
        "Reported input bookends equal": receipt.get("input_bookends_equal"),
        "Recorded replay algorithm": replay.get("algorithm"),
        "Recorded replay components": list(_mapping(replay.get("components"))),
        "Formal references / proof": formal.get("status", "unavailable"),
        "Formal alignment": formal.get("alignment", "unavailable"),
        "Lean compilation": "not supplied by this audit schema",
        "Execution": execution.get("status", "unavailable"),
        "Execution source verified": execution.get(
            "execution_source_verified", "unavailable"
        ),
        "Pair custody": _mapping(receipt.get("pair_evidence")).get(
            "status", "unavailable"
        ),
        "Universal equivalence claim": formal.get(
            "universal_equivalence", "unavailable"
        ),
    }
    metadata = "".join(
        f"<dt>{escape(label)}</dt><dd>{_text(value)}</dd>"
        for label, value in identity.items()
    )
    return (
        f'<article class="card evidence-card" data-evidence-status="{entry["status"]}" aria-labelledby="evidence-title-{index}">'
        f'<h3 id="evidence-title-{index}">FEP recorded independent Gaussian product replay</h3>'
        f'<span class="pill badge-pending evidence-outcome">Reported audit status: {entry["status"]}</span>'
        f'<p class="evidence-boundary">{_PRODUCT_AUDIT_BOUNDARY}</p><dl>{metadata}</dl>'
        + _quantity_table(
            replay.get("comparisons"),
            "FEP recorded independent Gaussian product replay comparisons",
        )
        + "<p>Complete recorded component arrays remain in the saved audit disclosure below. No numerical replay or trace alignment is performed here.</p>"
        + _details("Recorded finite algebra references and unmatched scope", formal)
        + _details(
            "Recorded audit provenance and diagnostics",
            {
                key: receipt[key]
                for key in (
                    "input",
                    "input_bookends_equal",
                    "execution_evidence",
                    "pair_evidence",
                    "cross_agent_covariance",
                    "errors",
                    "reasons",
                )
                if key in receipt
            },
        )
        + _details("Complete saved FEP product audit (unverified)", receipt)
        + "</article>"
    )


def _native_formal_card(entry: dict[str, Any], index: int) -> str:
    """Present retained compiler controls as global formal, not model evidence."""
    receipt = entry["raw"]
    scope, negative_controls, title = _NATIVE_PROFILES[receipt["schema"]]
    rows, outputs = "", ""
    for name, raw in _mapping(receipt.get("results")).items():
        result = _mapping(raw)
        compiles = result.get("compiles")
        compiler = (
            "compiled"
            if compiles is True
            else "rejected"
            if compiles is False
            else "unavailable"
        )
        if name in negative_controls:
            role = "Negative assertion (expected rejection)"
            if (
                compiles is False
                and result.get("accepted") is True
                and result.get("has_sorry") is False
            ):
                outcome = "Reported negative control: false claim rejected"
            elif compiles is True:
                outcome = "Reported negative control: false claim compiled unexpectedly"
            else:
                outcome = "Negative control acceptance unavailable or not recorded"
        elif name == "positive":
            role = "Positive theorem statements"
            outcome = "Reported positive compilation: " + compiler
        else:
            role = "Recorded compiler control"
            outcome = "Reported compiler outcome: " + compiler
        rows += (
            f'<tr><th scope="row">{_text(name)}</th><td>{role}</td><td>{outcome}</td>'
            f"<td>{_text(result.get('accepted'))}</td><td>{_text(result.get('has_sorry'))}</td>"
            f"<td>{_text(result.get('source_sha256'))}</td></tr>"
        )
        outputs += _details(
            f"Retained compiler and axiom output: {name}",
            {
                key: result.get(key, "unavailable")
                for key in (
                    "stdout",
                    "stderr",
                    "errors",
                    "warnings",
                    "compiles",
                    "has_sorry",
                    "accepted",
                    "source_sha256",
                )
            },
        )
    if not rows:
        rows = '<tr><td colspan="6">No retained compiler controls available.</td></tr>'
    provenance = {
        "Schema": receipt.get("schema"),
        "Formal source path (not a GNN model identity)": receipt.get("source_path"),
        "Retained formal source SHA-256": receipt.get("source_sha256"),
        "Reported toolchain": receipt.get("toolchain"),
        "Execution source verified": receipt.get("execution_source_verified"),
        "Universal equivalence claim": receipt.get("universal_equivalence"),
    }
    metadata = "".join(
        f"<dt>{escape(label)}</dt><dd>{_text(value)}</dd>"
        for label, value in provenance.items()
    )
    return (
        f'<article class="card evidence-card" data-evidence-status="{entry["status"]}" aria-labelledby="evidence-title-{index}">'
        f'<h3 id="evidence-title-{index}">{title}</h3>'
        f'<span class="pill badge-pending evidence-outcome">Reported status: {entry["status"]}</span>'
        "<p>Global formal artifact; no source-model numerical binding is claimed.</p>"
        f'<p class="evidence-boundary">{scope}</p><dl>{metadata}</dl>'
        '<div class="table-wrap"><table><caption>Recorded theorem compilation and expected rejection controls</caption>'
        '<thead><tr><th scope="col">Control</th><th scope="col">Role</th><th scope="col">Reported compiler outcome</th><th scope="col">Control accepted</th><th scope="col">Has sorry</th><th scope="col">Control source SHA-256</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>{outputs}"
        + _details("Complete saved native compiler receipt (unverified)", receipt)
        + "</article>"
    )


def render_evidence_section(analysis: Any) -> str:
    """Render accessible offline evidence cards inside the existing analysis page."""
    entries = _entries(analysis)
    cards = ""
    formal_cards = ""
    for index, entry in enumerate(entries):
        if entry.get("kind") == "native_formal":
            formal_cards += _native_formal_card(entry, index)
            continue
        if entry.get("kind") == "product_audit":
            cards += _product_audit_card(entry, index)
            continue
        contract, witness, raw = entry["contract"], entry["witness"], entry["raw"]
        identity = {
            "Schema": entry["schema"],
            "Model": contract.get("model_id", contract.get("model")),
            "Source SHA-256": contract.get("source_sha256"),
            "Inference question": contract.get(
                "question", contract.get("inference_mode")
            ),
            "Precision": contract.get("dtype", contract.get("dtypes")),
        }
        identity_html = "".join(
            f"<dt>{escape(key)}</dt><dd>{_text(value)}</dd>"
            for key, value in identity.items()
        )
        assumptions = {
            "source_contract": contract,
            "assumptions": witness.get("assumptions", "unavailable"),
        }
        provenance = _artifact_provenance(witness)
        provenance.update(
            {
                key: raw[key]
                for key in (
                    "live_sources",
                    "pair_evidence",
                    "formal_evidence",
                    "execution_evidence",
                )
                if key in raw
            }
        )
        replay = _mapping(raw.get("independent_replay"))
        replay_html = (
            _quantity_table(
                replay.get("comparisons"), "FEP recorded replay comparisons"
            )
            if replay
            else ""
        )
        reference_html = (
            _details(
                "Formal references and unmatched assumptions", raw["formal_references"]
            )
            if "formal_references" in raw
            else ""
        )
        diagnostic_html = _details(
            "Recorded diagnostics",
            {
                key: raw[key]
                for key in ("errors", "reasons", "boundary", "claim")
                if key in raw
            },
        )
        cards += (
            f'<article class="card evidence-card" data-evidence-status="{entry["status"]}" aria-labelledby="evidence-title-{index}">'
            f'<h3 id="evidence-title-{index}">{_text(entry["title"])}</h3>'
            f'<span class="pill badge-pending evidence-outcome">Reported status: {entry["status"]}</span>'
            f"<dl>{identity_html}</dl>"
            + _quantity_table(entry["comparisons"], "Recorded numerical comparisons")
            + replay_html
            + _trace_explorer(witness, index, entry["status"])
            + _evidence_planes(raw, witness)
            + _details("Scientific assumptions and source contract", assumptions)
            + _details(
                "Reference and candidate records (unverified)",
                {
                    "reference": witness.get(
                        "reference", witness.get("expected", "unavailable")
                    ),
                    "candidate": witness.get(
                        "candidate", witness.get("actual", "unavailable")
                    ),
                },
            )
            + reference_html
            + _details("Recorded artifact and runtime provenance", provenance)
            + diagnostic_html
            + _details("Complete saved receipt (unverified)", raw)
            + "</article>"
        )
    if cards:
        cards = (
            '<section aria-label="Source-bound numerical evidence"><h3>Source-bound numerical evidence</h3>'
            + cards
            + "</section>"
        )
    if formal_cards:
        cards += (
            '<section aria-label="Retained native Lean compilation"><h3>Retained native Lean compilation</h3>'
            + formal_cards
            + "</section>"
        )
    if not cards:
        cards = '<div class="card"><p>No recognized numerical evidence receipts collected for this view.</p></div>'
    controls = (
        """
<div class="evidence-controls">
  <div><label for="evidence-query">Filter evidence by model, source digest, quantity or reason</label><input type="search" id="evidence-query" aria-controls="evidence-records" aria-describedby="evidence-count"></div>
  <div><label for="evidence-status">Reported receipt status</label><select id="evidence-status" aria-controls="evidence-records"><option value="">All recorded outcomes</option><option value="passed">Passed</option><option value="failed">Failed</option><option value="refused">Refused</option><option value="unavailable">Unavailable or unrecognized</option><option value="comparable">Admitted for comparison</option></select></div>
</div>"""
        if entries
        else ""
    )
    return (
        f'<style>{_CSS}</style><section id="semantic-evidence" aria-labelledby="evidence-heading">'
        '<h2 id="evidence-heading">Recorded scientific evidence</h2>'
        f'<p class="evidence-boundary">{PROOF_BOUNDARY}</p>'
        f'{controls}<p id="evidence-count" class="evidence-count" role="status" aria-live="polite" aria-atomic="true">{len(entries)} of {len(entries)} recorded evidence entries shown.</p>'
        f'<div id="evidence-records">{cards}</div></section><script>{_FILTER_JS}</script><script>{_TRACE_JS}</script>'
    )
