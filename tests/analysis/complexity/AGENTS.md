# tests/analysis/complexity — agent rules

## Overview

Complexity-estimator regressions for the analysis step.

## Purpose

Keep estimates derived from the model's declared state space rather than from
hard-coded expectations.

## Directory Identity

Test package under `tests/analysis/`.

## Rules

- Assert relationships (growth with state/observation counts) as well as exact
  values where the formula is frozen.
- Pin the estimator's output, not a whole report payload.
