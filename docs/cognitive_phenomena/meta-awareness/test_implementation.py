#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Sandved-Smith et al. (2021) implementation

This script verifies that our implementation produces results consistent
with the computational phenomenology paper.
"""

import pathlib as _p

# Ensure local imports resolve when running from repo root
import sys

import numpy as np

_here = _p.Path(__file__).parent
sys.path.insert(0, str(_here))
from sandved_smith_2021 import (
    SandvedSmithModel,
    run_figure_7_simulation,
    run_figure_10_simulation,
    run_figure_11_simulation,
)
from utils import softmax, softmax_dim2
from visualizations import display_results_summary


def test_utility_functions():
    """Test core utility functions."""
    print("Testing utility functions...")

    # Test softmax
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0), "Softmax should sum to 1"
    assert np.all(probs >= 0), "Softmax should be non-negative"
    print("✓ Softmax function works correctly")

    # Test softmax_dim2
    logits_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    probs_2d = softmax_dim2(logits_2d)
    assert np.allclose(np.sum(probs_2d, axis=0), 1.0), (
        "Softmax_dim2 columns should sum to 1"
    )
    print("✓ Softmax_dim2 function works correctly")

    print("All utility tests passed! ✓\n")


def test_model_consistency():
    """Test model consistency across runs."""
    print("Testing model consistency...")

    # Run same model twice with same seed
    model1 = SandvedSmithModel(T=20, three_level=False, random_seed=123)
    results1 = model1.run_simulation()
    model2 = SandvedSmithModel(T=20, three_level=False, random_seed=123)
    results2 = model2.run_simulation()

    # Check that results are identical
    assert np.allclose(results1["X1_bar"], results2["X1_bar"], equal_nan=True), (
        "Results should be reproducible"
    )
    assert np.allclose(results1["X2_bar"], results2["X2_bar"], equal_nan=True), (
        "Results should be reproducible"
    )
    print("✓ Model produces consistent results with same seed")

    print("Consistency tests passed! ✓\n")


def test_three_level_vs_two_level():
    """Test differences between two-level and three-level models."""
    print("Testing three-level vs two-level models...")

    # Run both models
    model2 = SandvedSmithModel(T=50, three_level=False, random_seed=42)
    model3 = SandvedSmithModel(T=50, three_level=True, random_seed=42)

    results2 = model2.run_simulation()
    results3 = model3.run_simulation()

    # Three-level should have additional variables
    assert "X3_bar" in results3, "Three-level model should have meta-awareness states"
    assert "gamma_A2" in results3, "Three-level model should have attentional precision"
    assert "X3_bar" not in results2, (
        "Two-level model should not have meta-awareness states"
    )

    # Check shapes
    assert results3["X3_bar"].shape == (2, 50), (
        "Meta-awareness states should have correct shape"
    )
    assert results3["gamma_A2"].shape == (50,), (
        "Attentional precision should have correct shape"
    )

    print("✓ Three-level model has expected additional variables")
    print("Model comparison tests passed! ✓\n")


def test_mind_wandering_dynamics():
    """Test mind-wandering dynamics are reasonable."""
    print("Testing mind-wandering dynamics...")

    # Run longer simulation to observe dynamics
    model = SandvedSmithModel(T=100, three_level=False, random_seed=42)
    results = model.run_simulation()

    x2 = results["x2"]  # True attentional states

    # Should have both focused and distracted periods
    focused_time = np.sum(x2 == 0)
    distracted_time = np.sum(x2 == 1)

    assert focused_time > 0, "Should have some focused time"
    assert distracted_time > 0, "Should have some distracted time"

    # Should have state transitions
    transitions = np.sum(np.diff(x2) != 0)
    assert transitions > 0, "Should have attentional state transitions"

    print(
        f"✓ Mind-wandering dynamics: {focused_time}% focused, {distracted_time}% distracted"
    )
    print(f"✓ Number of attentional transitions: {transitions}")
    print("Mind-wandering tests passed! ✓\n")


def test_precision_dynamics():
    """Test precision dynamics are reasonable."""
    print("Testing precision dynamics...")

    model = SandvedSmithModel(T=50, three_level=True, random_seed=42)
    results = model.run_simulation()

    gamma_A1 = results["gamma_A1"]
    gamma_A2 = results["gamma_A2"]

    # Precision should be positive
    assert np.all(gamma_A1 > 0), "Perceptual precision should be positive"
    assert np.all(gamma_A2 > 0), "Attentional precision should be positive"

    # Precision should vary over time (not constant)
    assert np.var(gamma_A1) > 0, "Perceptual precision should vary"
    assert np.var(gamma_A2) > 0, "Attentional precision should vary"

    print(
        f"✓ Perceptual precision range: [{np.min(gamma_A1):.3f}, {np.max(gamma_A1):.3f}]"
    )
    print(
        f"✓ Attentional precision range: [{np.min(gamma_A2):.3f}, {np.max(gamma_A2):.3f}]"
    )
    print("Precision dynamics tests passed! ✓\n")


def test_figure_modes():
    """Test different figure modes produce expected patterns."""
    print("Testing figure modes...")

    # Figure 7: Fixed attentional schedule
    model = SandvedSmithModel(T=100, three_level=False, random_seed=42)
    results = model.run_simulation(figure_mode="fig7")

    x2 = results["x2"]
    # Should be focused first half, distracted second half
    first_half = x2[:50]
    second_half = x2[50:]

    assert np.all(first_half == 0), "First half should be focused (fig7 mode)"
    assert np.all(second_half == 1), "Second half should be distracted (fig7 mode)"
    print("✓ Figure 7 mode produces expected attentional schedule")

    # Figure 11: Meta-awareness schedule
    model = SandvedSmithModel(T=100, three_level=True, random_seed=42)
    results = model.run_simulation(figure_mode="fig11")

    x3 = results["x3"]
    # Should be high meta-awareness first half, low second half
    first_half = x3[:50]
    second_half = x3[50:]

    assert np.all(first_half == 0), (
        "First half should be high meta-awareness (fig11 mode)"
    )
    assert np.all(second_half == 1), (
        "Second half should be low meta-awareness (fig11 mode)"
    )
    print("✓ Figure 11 mode produces expected meta-awareness schedule")

    print("Figure mode tests passed! ✓\n")


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("=" * 60)
    print("SANDVED-SMITH 2021 IMPLEMENTATION TEST SUITE")
    print("=" * 60)

    test_utility_functions()
    test_model_consistency()
    test_three_level_vs_two_level()
    test_mind_wandering_dynamics()
    test_precision_dynamics()
    test_figure_modes()

    print("=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("Implementation verified successfully!")
    print("=" * 60)


def demo_simulation_results():
    """Demonstrate simulation results."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION OF SIMULATION RESULTS")
    print("=" * 60)

    # Run example simulations
    print("Running demonstration simulations...")

    results_fig7 = run_figure_7_simulation()
    results_fig10 = run_figure_10_simulation()
    results_fig11 = run_figure_11_simulation()

    # Display summaries
    display_results_summary(results_fig7)
    display_results_summary(results_fig10)
    display_results_summary(results_fig11)


if __name__ == "__main__":
    # Run tests
    run_comprehensive_test()

    # Run demonstration
    demo_simulation_results()
