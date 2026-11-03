# tests/testing — agent rules

## Overview

Round-trip and infrastructure regressions; the harness the rest of `tests/`
runs on is owned at this level.

## Purpose

Pin round-trip scope, report shapes and the native-artifact consumers.

## Directory Identity

Test package over `src/gnn/testing/` and the runner's contract surfaces.

## Rules

- Round-trip scope is declared in `src/gnn/testing/round_trip_config.py`; never
  add a format to the test list to raise a pass rate.
- Native-artifact cases read committed 4.1.0 artifacts. Do not regenerate them
  here; their custody belongs to the manuscript/render chain.
- A round-trip failure is a parser or serializer defect until proven otherwise;
  do not convert it into an expected difference without a source reason.
