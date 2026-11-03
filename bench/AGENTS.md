# bench/ — agent rules

## Overview

Standalone benchmark harnesses, outside the pipeline and the test suite.

## Purpose

Keep deterministic, offline metrics that back consolidation decisions.

## Directory Identity

Harness directory; one file per metric.

## Rules

- Stay deterministic and offline: no network access, no wall-clock or
  host-dependent thresholds.
- A harness that becomes a gate moves under `scripts/` with its contract test in
  `tests/scripts/`.
