# tests/api/fixtures — agent rules

## Overview

Negative and warning-path inputs for the API tests.

## Purpose

Hold the exact bytes the failure-path assertions were written against.

## Directory Identity

Fixture directory; no test code.

## Rules

- These files are inputs, not examples; the exemplar corpus lives in
  `input/gnn_files/`.
- Editing a fixture invalidates the assertion built on it: update both together,
  and never soften an expected error to match edited bytes.
