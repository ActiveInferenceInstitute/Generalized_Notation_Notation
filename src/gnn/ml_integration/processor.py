#!/usr/bin/env python3
"""
ML Integration Processor module for GNN Processing Pipeline.

Extracts real GNN features (num_states, num_observations, connectivity_ratio, etc.)
and trains ML models to classify model families and predict complexity.

Features extracted from GNN structure:
- num_states: hidden state dimensions from StateSpaceBlock
- num_observations: observation dimensions
- num_actions: action space size (from B tensor)
- num_variables: total variable count
- connectivity_ratio: connections / (variables * (variables-1))
- max_dimension: largest single variable dimension
- total_parameters: total element count across all variables
- model_family: detected family label (pomdp, hmm, hierarchical, continuous, ...)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Canonical numeric feature order used to build training and inference vectors.
#: The ``.pkl`` artifacts written by :func:`process_ml_integration` are only
#: meaningful against this exact column order.
NUMERIC_FEATURE_NAMES: tuple[str, ...] = (
    "num_states",
    "num_observations",
    "num_actions",
    "num_variables",
    "connectivity_ratio",
    "max_dimension",
    "total_parameters",
    "planning_horizon",
    "directed_connections",
    "undirected_connections",
    "has_precision",
    "has_learning",
    "has_ontology",
    "has_parameterization",
)

# Default value per numeric feature when a mapping omits the key
# (mirrors the historical training-matrix construction).
_DEFAULT_FEATURE_VALUE: dict[str, float] = dict.fromkeys(NUMERIC_FEATURE_NAMES, 0.0)
_DEFAULT_FEATURE_VALUE["planning_horizon"] = 1.0

#: Parameter-count thresholds delimiting the small/medium/large complexity
#: buckets (``total_parameters < 100`` -> small, ``< 1000`` -> medium, else large).
COMPLEXITY_THRESHOLDS: tuple[int, int] = (100, 1000)

#: Complexity bucket labels, ordered from smallest to largest; the index of a
#: label in this tuple is the integer class used for complexity classification.
COMPLEXITY_LABELS: tuple[str, ...] = ("small", "medium", "large")

#: Numeric keys summarized by :func:`summarize_features` and the fallback
#: feature-analysis output (``feature_statistics`` in the results JSON).
SUMMARY_STATISTIC_KEYS: tuple[str, ...] = (
    "num_states",
    "num_observations",
    "num_actions",
    "total_parameters",
    "connectivity_ratio",
)


def extract_gnn_features(file_path: Path) -> dict[str, Any]:
    """
    Extract structural features from a GNN file for ML training.

    Returns a feature dict with:
    - Numeric features for ML models (see :data:`NUMERIC_FEATURE_NAMES`)
    - Categorical features for labeling (``model_family``, ``time_type``)
    - Raw metadata for reporting (``file_name``)
    """
    try:
        content = file_path.read_text()
    except Exception as e:
        logger.error(f"Could not read {file_path}: {e}")
        return {}

    features: dict[str, Any] = {
        "file_name": file_path.name,
        "model_family": _detect_model_family(content),
        "num_variables": 0,
        "num_states": 0,
        "num_observations": 0,
        "num_actions": 0,
        "max_dimension": 0,
        "total_parameters": 0,
        "connectivity_ratio": 0.0,
        "directed_connections": 0,
        "undirected_connections": 0,
        "has_precision": False,
        "has_learning": False,
        "has_ontology": bool("## ActInfOntologyAnnotation" in content),
        "has_parameterization": bool("## InitialParameterization" in content),
        "planning_horizon": _extract_planning_horizon(content),
        "time_type": _extract_time_type(content),
    }

    # Extract variable dimensions from StateSpaceBlock
    dims = _extract_dimensions(content)
    features["num_variables"] = len(dims)

    if dims:
        # Compute dimension-based features
        all_elements: list[int] = []
        for _, var_dims in dims.items():
            elements = 1
            for d in var_dims:
                elements *= d
            all_elements.append(elements)
            features["max_dimension"] = max(features["max_dimension"], max(var_dims))

        features["total_parameters"] = sum(all_elements)

        # Extract specific Active Inference dimensions
        if "A" in dims:
            a_dims = dims["A"]
            if len(a_dims) >= 2:
                features["num_observations"] = a_dims[0]
                features["num_states"] = a_dims[1]

        if "B" in dims:
            b_dims = dims["B"]
            if len(b_dims) >= 3:
                features["num_actions"] = b_dims[2]

        if "s" in dims and features["num_states"] == 0:
            features["num_states"] = dims["s"][0]

        if "o" in dims and features["num_observations"] == 0:
            features["num_observations"] = dims["o"][0]

    # Connectivity analysis
    connections = _count_connections(content)
    features["directed_connections"] = connections["directed"]
    features["undirected_connections"] = connections["undirected"]
    total_conn = connections["directed"] + connections["undirected"]

    n = features["num_variables"]
    if n > 1:
        features["connectivity_ratio"] = total_conn / (n * (n - 1))

    # Qualitative features
    features["has_precision"] = bool(
        re.search(r"\b(precision|omega|γ|gamma|α|alpha)\b", content, re.IGNORECASE)
        and re.search(r"^(ω|γ|β|Π)\s*\[", content, re.MULTILINE)
    )
    features["has_learning"] = bool(
        re.search(
            r"\b(learning|concentration|dirichlet|update.*param)\b",
            content,
            re.IGNORECASE,
        )
    )

    return features


def _detect_model_family(content: str) -> str:
    """Detect model family from GNN content."""
    section_match = re.search(r"## GNNSection\s*\n\s*(\S+)", content)
    if section_match:
        section = section_match.group(1).lower()
        if "hierarchical" in section:
            return "hierarchical"
        if "continuous" in section:
            return "continuous"
        if "multiagent" in section or "multi_agent" in section:
            return "multi_agent"
        if "factor" in section:
            return "factor_graph"
        if "hmm" in section:
            return "hmm"
        if "pomdp" in section:
            return "pomdp"

    has_B_with_actions = bool(re.search(r"^B\s*\[\d+,\d+,\d+", content, re.MULTILINE))
    has_pi = bool(re.search(r"^π\s*\[|^pi\s*\[", content, re.MULTILINE))

    if has_B_with_actions and has_pi:
        return "pomdp"
    if re.search(r"^B\s*\[", content, re.MULTILINE) and not has_pi:
        return "hmm"
    return "unknown"


def _extract_dimensions(content: str) -> dict[str, list[int]]:
    """Extract variable dimensions from StateSpaceBlock."""
    dims: dict[str, list[int]] = {}
    in_state_space = False
    # ``[^\W\d]`` is a Unicode letter/underscore, so canonical GNN names
    # such as π are retained alongside ASCII identifiers.
    pattern = r"^([^\W\d]\w*\'?)\s*\[([^\]]+)\]"

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## StateSpaceBlock"):
            in_state_space = True
            continue
        elif stripped.startswith("##") and in_state_space:
            in_state_space = False
            continue

        if in_state_space and not stripped.startswith("#"):
            match = re.match(pattern, stripped)
            if match:
                var_name = match.group(1)
                dim_str = match.group(2)
                var_dims: list[int] = []
                for part in dim_str.split(","):
                    part = part.strip()
                    if part.startswith("type="):
                        continue
                    try:
                        var_dims.append(int(part))
                    except ValueError:
                        logger.debug("Skipping non-integer dimension part: %s", part)
                if var_dims:
                    dims[var_name] = var_dims

    return dims


def _count_connections(content: str) -> dict[str, int]:
    """Count directed and undirected connections."""
    in_connections = False
    directed = 0
    undirected = 0

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Connections"):
            in_connections = True
            continue
        elif stripped.startswith("##") and in_connections:
            in_connections = False
            continue

        if in_connections and not stripped.startswith("#") and stripped:
            directed += len(re.findall(r">", stripped))
            undirected += len(re.findall(r"(?<![>])-(?![>])", stripped))

    return {"directed": directed, "undirected": undirected}


def _extract_planning_horizon(content: str) -> int:
    """Extract planning horizon from model parameters."""
    match = re.search(r"planning_horizon\s*:\s*(\d+)", content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"ModelTimeHorizon\s*=\s*(\d+)", content)
    if match:
        return int(match.group(1))
    return 1  # Default: 1-step planning


def _extract_time_type(content: str) -> str:
    """Extract time type (Discrete/Continuous)."""
    if re.search(r"^Continuous\s*$", content, re.MULTILINE):
        return "continuous"
    if re.search(r"^Discrete\s*$", content, re.MULTILINE):
        return "discrete"
    return "unknown"


def feature_vector(features: Mapping[str, Any]) -> list[float]:
    """Project a feature mapping onto the canonical training vector order.

    The returned values follow :data:`NUMERIC_FEATURE_NAMES` exactly — the
    column order used to fit the ``gnn_*.pkl`` classifier artifacts. Missing
    keys fall back to the training-time defaults (``0`` everywhere except
    ``planning_horizon`` which defaults to ``1``); booleans coerce to ``0/1``.

    Use this (not a hand-rolled list) whenever scoring a saved artifact, e.g.
    via :func:`ml_integration.predict_with_model`.
    """
    return [
        float(features.get(name, _DEFAULT_FEATURE_VALUE[name]))
        for name in NUMERIC_FEATURE_NAMES
    ]


def complexity_label(total_parameters: float) -> str:
    """Classify a parameter count into the small/medium/large complexity buckets.

    Thresholds are :data:`COMPLEXITY_THRESHOLDS`; labels are
    :data:`COMPLEXITY_LABELS`. This mirrors the fallback task used when every
    sample in a training corpus shares one model family.
    """
    small_max, medium_max = COMPLEXITY_THRESHOLDS
    if total_parameters < small_max:
        return "small"
    if total_parameters < medium_max:
        return "medium"
    return "large"


def summarize_features(
    features: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-key min/max/mean summary statistics for a feature collection.

    Only :data:`SUMMARY_STATISTIC_KEYS` entries that are numeric in at least
    one mapping are summarized; keys with no numeric values are omitted.
    Pure function — no filesystem or global state.
    """
    stats: dict[str, dict[str, Any]] = {}
    for key in SUMMARY_STATISTIC_KEYS:
        values: list[Any] = []
        for f in features:
            value = f.get(key)
            if isinstance(value, (int, float)):
                values.append(value)
        if values:
            stats[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return stats


def process_ml_integration(
    target_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Process ML integration for GNN models.

    Extracts real GNN features and trains models to classify model families.
    Uses RandomForestClassifier alongside Decision Tree for comparison.
    Adds cross-validation and feature importance reporting.

    When training is not possible (scikit-learn missing or fewer than two
    usable samples), per-file structural-analysis entries and a feature
    summary are recorded instead and the step still succeeds.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.info(f"Starting ML integration: target={target_dir}")

        ml_results: dict[str, Any] = {
            "status": "completed",
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "models_trained": [],
            "feature_importance": {},
            "cross_validation": {},
            "framework_status": {},
            "extracted_features": [],
        }

        has_sklearn = _check_training_dependencies(ml_results)
        gnn_files = _discover_gnn_files(target_dir, recursive)
        if not gnn_files:
            logger.warning(f"No GNN files found in {target_dir}")
            ml_results["status"] = "no_files"

        all_features = _collect_features(gnn_files, ml_results, verbose)

        if has_sklearn and len(all_features) >= 2:
            _train_models(all_features, ml_results, output_dir, verbose)
        else:
            if not has_sklearn:
                logger.info("sklearn not available -- skipping ML training")
                note = "sklearn not available"
            else:
                logger.info(
                    f"Only {len(all_features)} files -- need >=2 for ML training"
                )
                note = f"Need >=2 GNN files for ML classification (have {len(all_features)})"
            ml_results["models_trained"].extend(
                _structural_analysis_entries(all_features, note)
            )
            _save_feature_analysis(all_features, ml_results, output_dir)

        # Save results
        results_file = output_dir / "ml_integration_results.json"
        with open(results_file, "w") as f:
            json.dump(ml_results, f, indent=2, default=str)

        if verbose:
            logger.info(f"ML integration completed -- results at {results_file}")

        return True

    except Exception as e:
        logger.error(f"ML integration failed: {e}")
        return False


def _check_training_dependencies(ml_results: dict[str, Any]) -> bool:
    """Probe numpy/scikit-learn availability and record it in ``ml_results``."""
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401

        ml_results["framework_status"]["sklearn"] = "available"
        return True
    except ImportError:
        ml_results["framework_status"]["sklearn"] = "missing"
        return False


def _discover_gnn_files(target_dir: Path, recursive: bool) -> list[Path]:
    """Return the sorted GNN markdown files under ``target_dir``."""
    from gnn.processing.discovery import is_model_source_path

    discovery = target_dir.rglob("*.md") if recursive else target_dir.glob("*.md")
    return sorted(path for path in discovery if is_model_source_path(path))


def _collect_features(
    gnn_files: Sequence[Path], ml_results: dict[str, Any], verbose: bool
) -> list[dict[str, Any]]:
    """Extract features from every discovered file, recording results."""
    all_features: list[dict[str, Any]] = []
    for gnn_file in gnn_files:
        try:
            feats = extract_gnn_features(gnn_file)
            if feats:
                all_features.append(feats)
                ml_results["extracted_features"].append(feats)
                if verbose:
                    logger.info(
                        f"  {gnn_file.name}: family={feats['model_family']}, "
                        f"states={feats['num_states']}, params={feats['total_parameters']}"
                    )
        except Exception as e:
            logger.error(f"Feature extraction failed for {gnn_file}: {e}")
    return all_features


def _structural_analysis_entries(
    all_features: Sequence[Mapping[str, Any]], note: str
) -> list[dict[str, Any]]:
    """Build per-file structural-analysis fallback entries."""
    return [
        {
            "source": feats["file_name"],
            "type": "structural_analysis",
            "framework": "internal_stats",
            "validation_status": "not_applicable",
            "note": note,
            "model_family": feats.get("model_family"),
            "num_states": feats.get("num_states", 0),
            "total_parameters": feats.get("total_parameters", 0),
        }
        for feats in all_features
    ]


def _cross_validation_folds(labels: Sequence[int]) -> int:
    """Return a valid stratified fold count, or zero when CV is unsupported."""
    class_counts = Counter(labels)
    if len(class_counts) < 2 or min(class_counts.values()) < 2:
        return 0
    return min(5, len(labels), min(class_counts.values()))


def _train_models(
    all_features: list[dict[str, Any]],
    ml_results: dict[str, Any],
    output_dir: Path,
    verbose: bool,
) -> None:
    """Train Decision Tree and Random Forest on extracted GNN features."""
    import pickle  # nosec B403

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier

    # Build feature matrix using real GNN features in canonical order
    X = np.array([feature_vector(f) for f in all_features], dtype=float)

    # Labels: model family
    y_labels: list[str] = [f.get("model_family", "unknown") for f in all_features]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Need at least 2 samples and 2 classes for meaningful training
    if len(np.unique(y)) < 2:
        # All same family -- use complexity classification instead
        y = np.array(
            [
                COMPLEXITY_LABELS.index(complexity_label(f.get("total_parameters", 0)))
                for f in all_features
            ]
        )
        label_names: list[str] = list(COMPLEXITY_LABELS)
        task = "complexity_classification"
    else:
        label_names = list(le.classes_)
        task = "family_classification"

    if len(np.unique(y)) < 2:
        ml_results["classification_task"] = task
        ml_results["classification_status"] = "insufficient_label_variation"
        ml_results["label_names"] = label_names
        ml_results["training_note"] = (
            "No classifier was trained because every sample has the same label"
        )
        _save_feature_analysis(all_features, ml_results, output_dir)
        return

    ml_results["classification_task"] = task
    ml_results["classification_status"] = "trained"
    ml_results["label_names"] = label_names

    # Cross-validation requires at least two members of every represented class.
    n_folds = _cross_validation_folds([int(label) for label in y.tolist()])

    models: list[tuple[str, Any]] = [
        ("decision_tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42),
        ),
    ]
    for model_name, clf in models:
        try:
            if n_folds >= 2:
                cv_scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
                validation_status = "cross_validated"
            else:
                cv_mean = None
                cv_std = None
                validation_status = "insufficient_class_support"

            # Fit on all data
            clf.fit(X, y)
            train_accuracy = float(clf.score(X, y))

            # Save model
            model_path = output_dir / f"gnn_{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)

            model_info: dict[str, Any] = {
                "type": model_name,
                "framework": "sklearn",
                "task": task,
                "train_accuracy": train_accuracy,
                "validation_status": validation_status,
                "cv_mean_accuracy": cv_mean,
                "cv_std": cv_std,
                "n_folds": n_folds,
                "n_samples": len(X),
                "n_features": len(NUMERIC_FEATURE_NAMES),
                "feature_names": list(NUMERIC_FEATURE_NAMES),
                "artifact_path": str(model_path),
            }

            # Feature importance (both models support this)
            importances = clf.feature_importances_.tolist()
            model_info["feature_importance"] = dict(
                zip(NUMERIC_FEATURE_NAMES, importances)
            )

            # Top 5 features
            sorted_feats = sorted(
                zip(NUMERIC_FEATURE_NAMES, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            model_info["top_features"] = [(f, round(i, 4)) for f, i in sorted_feats[:5]]

            ml_results["models_trained"].append(model_info)

            if verbose:
                if cv_mean is not None and cv_std is not None:
                    logger.info(
                        f"  {model_name}: CV accuracy={cv_mean:.3f}+/-{cv_std:.3f}, "
                        f"top feature={sorted_feats[0][0]}"
                    )
                else:
                    logger.info(
                        f"  {model_name}: train accuracy={train_accuracy:.3f}; "
                        "cross-validation unavailable due to class support"
                    )

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    # Store feature importance from best model
    if ml_results["models_trained"]:
        trained_models: list[dict[str, Any]] = [
            model for model in ml_results["models_trained"] if isinstance(model, dict)
        ]
        validated_models = [
            model
            for model in trained_models
            if isinstance(model.get("cv_mean_accuracy"), (int, float))
        ]
        selected = (
            max(validated_models, key=lambda model: model["cv_mean_accuracy"])
            if validated_models
            else trained_models[0]
        )
        ml_results["feature_importance"] = selected.get("feature_importance", {})


def _save_feature_analysis(
    all_features: Sequence[Mapping[str, Any]],
    ml_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save feature analysis when insufficient data for full ML training."""
    if not all_features:
        return

    ml_results["feature_statistics"] = summarize_features(all_features)
    ml_results["model_families"] = [f.get("model_family") for f in all_features]
